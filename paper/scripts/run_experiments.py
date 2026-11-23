"""Benchmark suite behind the IEEE SPL letter (writes ``results/benchmark_results.json``).

Runs:
  1. Numerical verification of the implementation (Table S1).
  2. Synthetic ER / BA / ring sweeps over SNR and network size with 24 trials per cell. These
     are superseded by ``rerun_synthetic_mc.py`` (200 / 100 trials), which must be run after
     this script because it replaces the ``synthetic`` block of the results file.
  3. Batched joint vs. anchor-only CRLB over graph density and size (CUDA if available; Table S4).
  4. Semi-synthetic networks from the 15 PhysioNet BIDMC patients, the raw five-channel BIDMC
     check and the WESAD check (Tables II, S5-S7). Data locations are read from the
     environment variables GRAPHSYNC_BIDMC_DIR and GRAPHSYNC_WESAD_NPZ.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
if PKG_SRC not in sys.path:
    sys.path.insert(0, PKG_SRC)

import graphsync as gs  # noqa: E402

OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
# PhysioNet BIDMC CSVs (https://physionet.org/content/bidmc/1.0.0/); set GRAPHSYNC_BIDMC_DIR.
BIDMC_DIR = os.environ.get("GRAPHSYNC_BIDMC_DIR", os.path.join(HERE, "..", "data", "bidmc_csv"))
WESAD_NPZ = os.path.join(PKG_SRC, "graphsync", "data", "bidmc_5node_slice.npz")
# WESAD subject S10 slice (ECG 250 Hz, BVP/ACC 64 Hz); set GRAPHSYNC_WESAD_NPZ.
WESAD_SRC = os.environ.get("GRAPHSYNC_WESAD_NPZ", os.path.join(HERE, "..", "data", "wesad_s10_slice.npz"))


def run_parity_and_noiseless_audit() -> Dict[str, float]:
    """Run exact Parseval, gradient, Hessian, FIM, and noiseless recovery checks."""
    s = gs.simulate_asynchronous_network(m_nodes=8, n_samples=512, snr_db=np.inf, seed=7)
    p = gs.verify_parseval_and_derivatives(s.observed, s.true_delays + 0.17, s.true_adjacency, fs=s.fs)
    c = gs.verify_crlb_fim_parity(s.true_laplacian, s.omegas, s.spectral_power, noise_var=0.05, gamma=s.gamma)

    # Exact noiseless deterministic broadband recovery (Theorem 1 & 2)
    rng = np.random.default_rng(101)
    n_samples = 512
    spec = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec[5:160] = rng.standard_normal(155) + 1j * rng.standard_normal(155)
    base = np.fft.irfft(spec, n=n_samples)
    clean = np.tile(base, (8, 1))
    tau_true = np.array([0.0, 2.34, -3.17, 1.82, -0.95, 4.11, -2.48, 0.63])
    obs = gs.apply_offsets(clean, tau_true, fs=1.0)
    res0 = gs.fit_graphsync(obs, fs=1.0, max_lag=5.0, fmin=5 / n_samples, fmax=160 / n_samples)

    out = {}
    out.update(p)
    out.update(c)
    out["noiseless_init_max_err"] = float(np.max(np.abs(res0.init_result.delays - tau_true)))
    out["noiseless_final_max_err"] = float(np.max(np.abs(res0.delays - tau_true)))
    out["noiseless_kkt"] = float(res0.kkt_residual)
    return out


def run_gpu_connectivity_crlb_sweep(device: torch.device) -> List[Dict[str, float]]:
    """Run the batched joint vs. anchor-only CRLB sweep over graph density and size (CUDA if available)."""
    t0 = time.perf_counter()
    results = []
    omegas_np = np.linspace(0.05, 1.1, 200)
    p_np = 1.0 + 0.5 * np.cos(np.linspace(0.0, np.pi, 200))
    om_t = torch.tensor(omegas_np, dtype=torch.float64, device=device)
    pk_t = torch.tensor(p_np, dtype=torch.float64, device=device)
    gamma = 0.1

    for m_nodes in [6, 10, 15, 20]:
        for edge_prob in [0.22, 0.35, 0.55, 0.80]:
            lam2_list, joint_rmse_list, anchor_rmse_list = [], [], []
            for seed in range(25):
                _, lap_np = gs.generate_connected_graph(
                    m_nodes, graph_type="er", edge_prob=edge_prob, seed=8000 + 100 * m_nodes + seed
                )
                lam2 = gs.algebraic_connectivity(lap_np)
                lap_t = torch.tensor(lap_np, dtype=torch.float64, device=device)
                eye_m = torch.eye(m_nodes, dtype=torch.float64, device=device)
                cov_base = torch.linalg.inv(lap_t + gamma * eye_m)  # (M, M)
                # Calibrate noise_var to SNR = 5 dB
                sig_var = float(torch.mean(pk_t) * torch.mean(torch.diag(cov_base)))
                noise_var = sig_var / (10.0 ** 0.5)

                # Batched C0 across all K=200 frequencies on GPU: shape (K, M, M)
                c0 = pk_t[:, None, None] * cov_base[None, :, :] + noise_var * eye_m[None, :, :]
                c0_inv = torch.linalg.inv(c0)
                w_sq = (2.0 * (om_t ** 2))[:, None, None]
                fim = torch.sum(w_sq * (c0_inv * c0 - eye_m[None, :, :]), dim=0)
                crlb_mat = torch.linalg.inv(fim[1:, 1:])
                joint_rmse = float(torch.sqrt(torch.mean(torch.diag(crlb_mat))).item())

                anchor_rmse = gs.compute_pairwise_anchor_crlb(
                    lap_np, omegas_np, p_np, noise_var=noise_var, gamma=gamma
                )
                lam2_list.append(lam2)
                joint_rmse_list.append(joint_rmse)
                anchor_rmse_list.append(anchor_rmse)

            results.append(
                {
                    "m_nodes": m_nodes,
                    "edge_prob": edge_prob,
                    "mean_lam2": float(np.mean(lam2_list)),
                    "joint_crlb_rmse": float(np.mean(joint_rmse_list)),
                    "anchor_crlb_rmse": float(np.mean(anchor_rmse_list)),
                    "variance_reduction_ratio": float(
                        (np.mean(anchor_rmse_list) / np.mean(joint_rmse_list)) ** 2
                    ),
                }
            )
    elapsed = time.perf_counter() - t0
    print(f"[GPU CRLB Sweep on {device}] Completed {len(results)} configs (400 graphs x 200 freqs) in {elapsed:.2f}s")
    return results


def run_synthetic_snr_and_size_benchmarks() -> Dict[str, object]:
    """Run synthetic SNR sweeps across ER/BA/Ring and node-count scaling M in {6,10,15,20}."""
    snr_grid = [-5, 0, 5, 10, 15, 20, 25]
    n_trials = 24  # 24 Monte Carlo trials per cell -> tight SE, clean runtime
    snr_rows = []

    for gtype in ["er", "ring", "ba"]:
        for snr in snr_grid:
            mse_gs, mse_z0, mse_cy, mse_xc, mse_ph = [], [], [], [], []
            crlb_mses, anchor_crlb_mses = [], []
            f1_gs, f1_un = [], []
            rt_gs = []

            for tr in range(n_trials):
                s = gs.simulate_asynchronous_network(
                    m_nodes=10,
                    n_samples=2048,
                    fs=1.0,
                    snr_db=float(snr),
                    max_lag=5.0,
                    graph_type=gtype,
                    edge_prob=0.30,
                    gamma=0.10,
                    k_min=8,
                    k_max=360,
                    seed=10000 + 500 * (["er", "ring", "ba"].index(gtype)) + 30 * (snr + 10) + tr,
                )
                fmin = float(s.omegas[0] / (2.0 * np.pi))
                fmax = float(s.omegas[-1] / (2.0 * np.pi))

                # Use fixed alpha=1.0, beta=0.10 (slightly higher beta at -5dB avoided? NO: keep FIXED beta=0.10 across ALL SNRs for 100% fairness!)
                t_start = time.perf_counter()
                r = gs.fit_graphsync(
                    s.observed,
                    fs=1.0,
                    max_lag=6.0,
                    alpha=1.0,
                    beta=0.10,
                    fmin=fmin,
                    fmax=fmax,
                    coherence_power=3.0,
                )
                rt_gs.append((time.perf_counter() - t_start) * 1000.0)

                r0 = gs.fit_graphsync(
                    s.observed,
                    fs=1.0,
                    max_lag=6.0,
                    alpha=1.0,
                    beta=0.10,
                    fmin=fmin,
                    fmax=fmax,
                    init="zeros",
                )
                tp = gs.gcc_phat_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
                tx = gs.xcorr_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
                tc = gs.unweighted_cycle_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
                wu, _, _ = gs.unsynced_kalofolias_graph(
                    s.observed, alpha=1.0, beta=0.10, fs=1.0, fmin=fmin, fmax=fmax
                )

                mse_gs.append(float(np.mean((r.delays[1:] - s.true_delays[1:]) ** 2)))
                mse_z0.append(float(np.mean((r0.delays[1:] - s.true_delays[1:]) ** 2)))
                mse_cy.append(float(np.mean((tc[1:] - s.true_delays[1:]) ** 2)))
                mse_xc.append(float(np.mean((tx[1:] - s.true_delays[1:]) ** 2)))
                mse_ph.append(float(np.mean((tp[1:] - s.true_delays[1:]) ** 2)))
                f1_gs.append(gs.edge_f1_score(r.adjacency, s.true_adjacency)[0])
                f1_un.append(gs.edge_f1_score(wu, s.true_adjacency)[0])
                crlb_mses.append(
                    gs.compute_joint_crlb(
                        s.true_laplacian, s.omegas, s.spectral_power, s.noise_var, s.gamma
                    ).mean_mse_bound
                )
                anchor_crlb_mses.append(
                    gs.compute_pairwise_anchor_crlb(
                        s.true_laplacian, s.omegas, s.spectral_power, s.noise_var, s.gamma
                    )
                    ** 2
                )

            row = {
                "graph_type": gtype,
                "snr_db": snr,
                "crlb_rmse": float(np.sqrt(np.mean(crlb_mses))),
                "anchor_crlb_rmse": float(np.sqrt(np.mean(anchor_crlb_mses))),
                "gs_rmse": float(np.sqrt(np.mean(mse_gs))),
                "gs_zero_init_rmse": float(np.sqrt(np.mean(mse_z0))),
                "cycle_rmse": float(np.sqrt(np.mean(mse_cy))),
                "xcorr_rmse": float(np.sqrt(np.mean(mse_xc))),
                "phat_rmse": float(np.sqrt(np.mean(mse_ph))),
                "crlb_ratio": float(np.sqrt(np.mean(mse_gs)) / np.sqrt(np.mean(crlb_mses))),
                "f1_gs": float(np.mean(f1_gs)),
                "f1_gs_std": float(np.std(f1_gs)),
                "f1_unsync": float(np.mean(f1_un)),
                "f1_unsync_std": float(np.std(f1_un)),
                "runtime_ms": float(np.mean(rt_gs)),
            }
            snr_rows.append(row)
            print(
                f"[{gtype.upper()} SNR={snr:+3d}dB] CRLB={row['crlb_rmse']:.4f} | "
                f"GS={row['gs_rmse']:.4f} ({row['crlb_ratio']:.2f}x) | "
                f"ZeroInit={row['gs_zero_init_rmse']:.2f} | Cyc={row['cycle_rmse']:.4f} | "
                f"XC={row['xcorr_rmse']:.4f} | PHAT={row['phat_rmse']:.4f} | "
                f"F1={row['f1_gs']:.3f} vs UN={row['f1_unsync']:.3f}"
            )

    # Node scaling sweep M in {6, 10, 15, 20} at SNR = 5 dB on ER graphs (12 trials each)
    size_rows = []
    for m_nodes in [6, 10, 15, 20]:
        mse_gs, mse_ph, mse_xc, mse_cy, crlb_mses = [], [], [], [], []
        f1_gs, f1_un, rt_ms = [], [], []
        p_edge = min(0.45, max(0.22, 3.0 / m_nodes))
        for tr in range(12):
            s = gs.simulate_asynchronous_network(
                m_nodes=m_nodes,
                n_samples=2048,
                fs=1.0,
                snr_db=5.0,
                max_lag=5.0,
                graph_type="er",
                edge_prob=p_edge,
                gamma=0.10,
                k_min=8,
                k_max=360,
                seed=20000 + 100 * m_nodes + tr,
            )
            fmin = float(s.omegas[0] / (2.0 * np.pi))
            fmax = float(s.omegas[-1] / (2.0 * np.pi))
            t0 = time.perf_counter()
            r = gs.fit_graphsync(
                s.observed, fs=1.0, max_lag=6.0, alpha=1.0, beta=0.10, fmin=fmin, fmax=fmax, coherence_power=3.0
            )
            rt_ms.append((time.perf_counter() - t0) * 1000.0)
            tp = gs.gcc_phat_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
            tx = gs.xcorr_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
            tc = gs.unweighted_cycle_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
            wu, _, _ = gs.unsynced_kalofolias_graph(
                s.observed, alpha=1.0, beta=0.10, fs=1.0, fmin=fmin, fmax=fmax
            )
            mse_gs.append(float(np.mean((r.delays[1:] - s.true_delays[1:]) ** 2)))
            mse_ph.append(float(np.mean((tp[1:] - s.true_delays[1:]) ** 2)))
            mse_xc.append(float(np.mean((tx[1:] - s.true_delays[1:]) ** 2)))
            mse_cy.append(float(np.mean((tc[1:] - s.true_delays[1:]) ** 2)))
            f1_gs.append(gs.edge_f1_score(r.adjacency, s.true_adjacency)[0])
            f1_un.append(gs.edge_f1_score(wu, s.true_adjacency)[0])
            crlb_mses.append(
                gs.compute_joint_crlb(
                    s.true_laplacian, s.omegas, s.spectral_power, s.noise_var, s.gamma
                ).mean_mse_bound
            )
        srow = {
            "m_nodes": m_nodes,
            "edge_prob": p_edge,
            "crlb_rmse": float(np.sqrt(np.mean(crlb_mses))),
            "gs_rmse": float(np.sqrt(np.mean(mse_gs))),
            "cycle_rmse": float(np.sqrt(np.mean(mse_cy))),
            "xcorr_rmse": float(np.sqrt(np.mean(mse_xc))),
            "phat_rmse": float(np.sqrt(np.mean(mse_ph))),
            "f1_gs": float(np.mean(f1_gs)),
            "f1_unsync": float(np.mean(f1_un)),
            "runtime_ms": float(np.mean(rt_ms)),
        }
        size_rows.append(srow)
        print(
            f"[M={m_nodes:2d} SNR=+5dB] CRLB={srow['crlb_rmse']:.4f} | GS={srow['gs_rmse']:.4f} | "
            f"Cyc={srow['cycle_rmse']:.4f} | XC={srow['xcorr_rmse']:.4f} | PHAT={srow['phat_rmse']:.4f} | "
            f"F1={srow['f1_gs']:.3f} vs UN={srow['f1_unsync']:.3f} | t={srow['runtime_ms']:.1f}ms"
        )

    return {"snr_sweep": snr_rows, "size_sweep": size_rows}


def run_real_bidmc_and_wesad_benchmarks() -> Dict[str, object]:
    """Run all 15 real PhysioNet BIDMC ICU patients and WESAD wearable validation."""
    files = sorted(glob.glob(os.path.join(BIDMC_DIR, "*Signals.csv")))
    assert len(files) == 15, f"Expected 15 BIDMC files, found {len(files)}"
    fs = 125.0

    # Part A: 15-Subject Real BIDMC 10-Node Sparse Physiological Sensor Network across SNR in {0, 5, 10} dB
    bidmc_snr_summary = []
    bidmc_per_subject_5db = []

    for snr_db in [0.0, 5.0, 10.0]:
        sigma = 10.0 ** (-snr_db / 20.0)
        rng = np.random.default_rng(2026 + int(snr_db))
        subj_rows = []
        for idx, fpath in enumerate(files):
            sid = os.path.basename(fpath).replace("_Signals.csv", "")
            df = pd.read_csv(fpath)
            df.columns = [c.strip() for c in df.columns]
            ecg_extra = [c for c in ["V", "AVR", "I", "MCL"] if c in df.columns][:2]
            cols = ["II"] + ecg_extra + ["PLETH", "RESP"]
            x_raw = df[cols].values.T

            w_true, l_true = gs.generate_connected_graph(
                10, graph_type="er", edge_prob=0.30, seed=3000 + idx
            )
            seg_len = int(40 * fs)
            exc = np.vstack(
                [
                    (lambda s: (s - s.mean()) / (s.std() + 1e-12))(
                        x_raw[
                            k % 5,
                            (k * int(42 * fs)) % (x_raw.shape[1] - seg_len - 1) : (k * int(42 * fs))
                            % (x_raw.shape[1] - seg_len - 1)
                            + seg_len,
                        ]
                    )
                    for k in range(10)
                ]
            )
            ev, u_mat = np.linalg.eigh(l_true + 0.12 * np.eye(10))
            clean_net = ((u_mat * (1.0 / np.sqrt(ev))) @ u_mat.T) @ exc
            clean_net = (clean_net - clean_net.mean(1, keepdims=True)) / (
                clean_net.std(1, keepdims=True) + 1e-12
            )

            tau_true = rng.uniform(-0.15, 0.15, size=10)
            tau_true[0] = 0.0
            obs = gs.apply_offsets(clean_net, tau_true, fs=fs) + rng.normal(
                0.0, sigma, size=clean_net.shape
            )

            r = gs.fit_graphsync(
                obs,
                fs=fs,
                max_lag=0.25,
                alpha=1.0,
                beta=0.06,
                fmin=0.2,
                fmax=20.0,
                nperseg=1024,
                whitening="trace",
                coherence_power=3.0,
            )
            tp = gs.gcc_phat_anchor_sync(obs, fs=fs, max_lag=0.25, fmin=0.2, fmax=20.0, nperseg=1024)
            tx = gs.xcorr_anchor_sync(obs, fs=fs, max_lag=0.25, fmin=0.2, fmax=20.0, nperseg=1024)
            tc = gs.unweighted_cycle_sync(obs, fs=fs, max_lag=0.25, fmin=0.2, fmax=20.0, nperseg=1024)
            wu, _, _ = gs.unsynced_kalofolias_graph(
                obs, alpha=1.0, beta=0.06, fs=fs, fmin=0.2, fmax=20.0, nperseg=1024, whitening="trace"
            )

            row = {
                "subject_id": sid,
                "snr_db": snr_db,
                "gs_mae_ms": float(np.mean(np.abs(r.delays[1:] - tau_true[1:])) * 1000.0),
                "phat_mae_ms": float(np.mean(np.abs(tp[1:] - tau_true[1:])) * 1000.0),
                "xcorr_mae_ms": float(np.mean(np.abs(tx[1:] - tau_true[1:])) * 1000.0),
                "cycle_mae_ms": float(np.mean(np.abs(tc[1:] - tau_true[1:])) * 1000.0),
                "f1_gs": gs.edge_f1_score(r.adjacency, w_true)[0],
                "f1_unsync": gs.edge_f1_score(wu, w_true)[0],
            }
            subj_rows.append(row)

        if snr_db == 5.0:
            bidmc_per_subject_5db = subj_rows

        gs_arr = np.array([r["gs_mae_ms"] for r in subj_rows])
        ph_arr = np.array([r["phat_mae_ms"] for r in subj_rows])
        xc_arr = np.array([r["xcorr_mae_ms"] for r in subj_rows])
        cy_arr = np.array([r["cycle_mae_ms"] for r in subj_rows])
        f1g_arr = np.array([r["f1_gs"] for r in subj_rows])
        f1u_arr = np.array([r["f1_unsync"] for r in subj_rows])

        p_phat = float(stats.wilcoxon(gs_arr, ph_arr, alternative="less").pvalue)
        p_xc = float(stats.wilcoxon(gs_arr, xc_arr, alternative="less").pvalue)
        p_cyc = float(stats.wilcoxon(gs_arr, cy_arr, alternative="less").pvalue)
        p_f1 = float(stats.wilcoxon(f1g_arr, f1u_arr, alternative="greater").pvalue)

        summary = {
            "snr_db": snr_db,
            "gs_mae_mean": float(np.mean(gs_arr)),
            "gs_mae_median": float(np.median(gs_arr)),
            "phat_mae_mean": float(np.mean(ph_arr)),
            "phat_mae_median": float(np.median(ph_arr)),
            "xcorr_mae_mean": float(np.mean(xc_arr)),
            "xcorr_mae_median": float(np.median(xc_arr)),
            "cycle_mae_mean": float(np.mean(cy_arr)),
            "cycle_mae_median": float(np.median(cy_arr)),
            "f1_gs_mean": float(np.mean(f1g_arr)),
            "f1_unsync_mean": float(np.mean(f1u_arr)),
            "wilcoxon_p_vs_phat": p_phat,
            "wilcoxon_p_vs_xcorr": p_xc,
            "wilcoxon_p_vs_cycle": p_cyc,
            "wilcoxon_p_f1": p_f1,
        }
        bidmc_snr_summary.append(summary)
        print(
            f"[BIDMC-15subj SNR={snr_db:4.1f}dB] MAE(ms): GS={summary['gs_mae_mean']:.2f} "
            f"(med {summary['gs_mae_median']:.2f}) | PHAT={summary['phat_mae_mean']:.2f} (p={p_phat:.2e}) | "
            f"XC={summary['xcorr_mae_mean']:.2f} (p={p_xc:.2e}) | Cyc={summary['cycle_mae_mean']:.2f} | "
            f"F1={summary['f1_gs_mean']:.3f} vs UN={summary['f1_unsync_mean']:.3f} (p={p_f1:.2e})"
        )

    # Part B: Direct Raw 5-Channel BIDMC Clinical Monitor Topology Recovery (15 ICU patients, reference graph learned from the original synchronized recording)
    rng_raw = np.random.default_rng(777)
    raw5_rows = []
    avg_w_ref = np.zeros((5, 5), dtype=np.float64)
    avg_w_gs = np.zeros((5, 5), dtype=np.float64)
    avg_w_un = np.zeros((5, 5), dtype=np.float64)

    for idx, fpath in enumerate(files):
        sid = os.path.basename(fpath).replace("_Signals.csv", "")
        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
        ecg_extra = [c for c in ["V", "AVR", "I", "MCL"] if c in df.columns][:2]
        cols = ["II"] + ecg_extra + ["PLETH", "RESP"]
        sb = df[cols].values[int(120 * fs) : int(210 * fs)].T.copy()
        sb = (sb - sb.mean(1, keepdims=True)) / (sb.std(1, keepdims=True) + 1e-12)
        # Align clinical lead polarity relative to reference lead II on clean synchronized data
        for ch in range(1, 5):
            if np.corrcoef(sb[0], sb[ch])[0, 1] < 0:
                sb[ch] *= -1.0

        w_ref, _, _ = gs.unsynced_kalofolias_graph(
            sb, alpha=1.0, beta=0.08, fs=fs, fmin=0.15, fmax=15.0, nperseg=1024
        )
        tau5 = rng_raw.uniform(-0.10, 0.10, size=5)
        tau5[0] = 0.0
        sb_async = gs.apply_offsets(sb, tau5, fs=fs) + rng_raw.normal(0.0, 0.35, size=sb.shape)

        r5 = gs.fit_graphsync(
            sb_async, fs=fs, max_lag=0.20, alpha=1.0, beta=0.08, fmin=0.15, fmax=15.0, nperseg=1024
        )
        wu5, _, _ = gs.unsynced_kalofolias_graph(
            sb_async, alpha=1.0, beta=0.08, fs=fs, fmin=0.15, fmax=15.0, nperseg=1024
        )
        # Also evaluate direct 3-lead ECG sync (II, ECG2, ECG3) where intrinsic physical lag == 0
        ecg3_mae = float(np.mean(np.abs(r5.delays[1:3] - tau5[1:3])) * 1000.0)
        f1_g = gs.edge_f1_score(r5.adjacency, w_ref)[0]
        f1_u = gs.edge_f1_score(wu5, w_ref)[0]
        raw5_rows.append(
            {"subject_id": sid, "ecg3_mae_ms": ecg3_mae, "f1_gs": f1_g, "f1_unsync": f1_u}
        )
        avg_w_ref += w_ref / len(files)
        avg_w_gs += r5.adjacency / len(files)
        avg_w_un += wu5 / len(files)

    raw5_f1_g = np.array([r["f1_gs"] for r in raw5_rows])
    raw5_f1_u = np.array([r["f1_unsync"] for r in raw5_rows])
    raw5_p = float(stats.wilcoxon(raw5_f1_g, raw5_f1_u, alternative="greater").pvalue)
    print(
        f"[RAW 5ch BIDMC Clinical Graph, 15subj] F1_GS={np.mean(raw5_f1_g):.3f} vs "
        f"F1_UN={np.mean(raw5_f1_u):.3f} (Wilcoxon p={raw5_p:.2e}) | "
        f"3-lead ECG median MAE={np.median([r['ecg3_mae_ms'] for r in raw5_rows]):.2f} ms"
    )

    # Part C: Real WESAD 4-sensor wearable network check (ECG 250Hz->64Hz, BVP 64Hz, Chest ACC 64Hz, Wrist ACC 64Hz)
    wesad_summary = {}
    if os.path.exists(WESAD_SRC):
        wd = np.load(WESAD_SRC)
        fs_w = float(wd["fs_bvp_hz"])  # 64.0 Hz
        ecg_64 = signal.resample(wd["ecg_250hz"].astype(np.float64), len(wd["bvp_64hz"]))
        bvp_64 = wd["bvp_64hz"].astype(np.float64)
        cacc_64 = wd["chest_acc_64hz"].astype(np.float64)
        wacc_64 = wd["wrist_acc_64hz"].astype(np.float64)
        w_raw = np.vstack([ecg_64, bvp_64, cacc_64, wacc_64])
        # Construct 8-node wearable sensor graph driven by non-overlapping 60s WESAD physiological epochs
        seg_w = int(60 * fs_w)  # 3840 samples
        rng_w = np.random.default_rng(909)
        w_errs_gs, w_errs_ph, w_errs_xc, w_errs_cy, w_f1_g, w_f1_u = [], [], [], [], [], []
        for tr in range(10):
            wt, lt = gs.generate_connected_graph(8, graph_type="ring", seed=910 + tr)
            exc_w = np.vstack(
                [
                    (lambda s: (s - s.mean()) / (s.std() + 1e-12))(
                        w_raw[k % 4, (k * int(40 * fs_w)) % (w_raw.shape[1] - seg_w - 1) : (k * int(40 * fs_w)) % (w_raw.shape[1] - seg_w - 1) + seg_w]
                    )
                    for k in range(8)
                ]
            )
            ev, u_mat = np.linalg.eigh(lt + 0.12 * np.eye(8))
            clean_w = ((u_mat * (1.0 / np.sqrt(ev))) @ u_mat.T) @ exc_w
            clean_w = (clean_w - clean_w.mean(1, keepdims=True)) / (clean_w.std(1, keepdims=True) + 1e-12)
            tau_w = rng_w.uniform(-0.20, 0.20, size=8)
            tau_w[0] = 0.0
            obs_w = gs.apply_offsets(clean_w, tau_w, fs=fs_w) + rng_w.normal(0.0, 0.56, size=clean_w.shape)
            rw = gs.fit_graphsync(
                obs_w, fs=fs_w, max_lag=0.35, alpha=1.0, beta=0.06, fmin=0.2, fmax=15.0, nperseg=512, whitening="trace", coherence_power=3.0
            )
            tpw = gs.gcc_phat_anchor_sync(obs_w, fs=fs_w, max_lag=0.35, fmin=0.2, fmax=15.0, nperseg=512)
            txw = gs.xcorr_anchor_sync(obs_w, fs=fs_w, max_lag=0.35, fmin=0.2, fmax=15.0, nperseg=512)
            tcw = gs.unweighted_cycle_sync(obs_w, fs=fs_w, max_lag=0.35, fmin=0.2, fmax=15.0, nperseg=512)
            wuw, _, _ = gs.unsynced_kalofolias_graph(
                obs_w, alpha=1.0, beta=0.06, fs=fs_w, fmin=0.2, fmax=15.0, nperseg=512, whitening="trace"
            )
            w_errs_gs.append(float(np.mean(np.abs(rw.delays[1:] - tau_w[1:])) * 1000.0))
            w_errs_ph.append(float(np.mean(np.abs(tpw[1:] - tau_w[1:])) * 1000.0))
            w_errs_xc.append(float(np.mean(np.abs(txw[1:] - tau_w[1:])) * 1000.0))
            w_errs_cy.append(float(np.mean(np.abs(tcw[1:] - tau_w[1:])) * 1000.0))
            w_f1_g.append(gs.edge_f1_score(rw.adjacency, wt)[0])
            w_f1_u.append(gs.edge_f1_score(wuw, wt)[0])
        wesad_summary = {
            "fs_hz": fs_w,
            "gs_mae_ms": float(np.mean(w_errs_gs)),
            "phat_mae_ms": float(np.mean(w_errs_ph)),
            "xcorr_mae_ms": float(np.mean(w_errs_xc)),
            "cycle_mae_ms": float(np.mean(w_errs_cy)),
            "f1_gs": float(np.mean(w_f1_g)),
            "f1_unsync": float(np.mean(w_f1_u)),
        }
        print(
            f"[WESAD 8-node Wearable, 64Hz, 5dB] MAE(ms): GS={wesad_summary['gs_mae_ms']:.2f} | "
            f"PHAT={wesad_summary['phat_mae_ms']:.2f} | XC={wesad_summary['xcorr_mae_ms']:.2f} | "
            f"Cyc={wesad_summary['cycle_mae_ms']:.2f} | F1={wesad_summary['f1_gs']:.3f} vs UN={wesad_summary['f1_unsync']:.3f}"
        )

    return {
        "bidmc_10node_summary": bidmc_snr_summary,
        "bidmc_10node_per_subject_5db": bidmc_per_subject_5db,
        "bidmc_raw5_rows": raw5_rows,
        "bidmc_raw5_summary": {
            "f1_gs_mean": float(np.mean(raw5_f1_g)),
            "f1_unsync_mean": float(np.mean(raw5_f1_u)),
            "wilcoxon_p_f1": raw5_p,
            "ecg3_mae_median_ms": float(np.median([r["ecg3_mae_ms"] for r in raw5_rows])),
            "avg_w_ref": avg_w_ref.tolist(),
            "avg_w_gs": avg_w_gs.tolist(),
            "avg_w_unsync": avg_w_un.tolist(),
        },
        "wesad_summary": wesad_summary,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== graphsync Benchmark Suite | Compute Device: {device} ({dev_name}) ===")

    t_all = time.perf_counter()
    parity = run_parity_and_noiseless_audit()
    print("[Parity & Noiseless]", json.dumps({k: f"{v:.2e}" for k, v in parity.items()}))

    gpu_crlb = run_gpu_connectivity_crlb_sweep(device)
    synth = run_synthetic_snr_and_size_benchmarks()
    real = run_real_bidmc_and_wesad_benchmarks()

    bundle = {
        "device": str(device),
        "device_name": dev_name,
        "elapsed_s": float(time.perf_counter() - t_all),
        "parity": parity,
        "gpu_crlb_sweep": gpu_crlb,
        "synthetic": synth,
        "real_physiological": real,
    }
    out_json = os.path.join(OUT_DIR, "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    print(f"\nSaved complete verified benchmark bundle -> {out_json} ({bundle['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
