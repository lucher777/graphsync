"""High-trial Monte Carlo rerun of the synthetic benchmarks (Table I, Fig. 2a-b, Table S3).

The original sweep used 24 trials per cell, which left the timing RMSE / CRLB ratio
with a 95% interval of roughly +/-15%. This script reruns every cell with
``N_TRIALS`` independent trials (non-overlapping seeds), reports a trial-level
bootstrap 95% interval for the RMSE / CRLB ratio, and writes
``results/synthetic_mc.json``. It then replaces the ``synthetic`` block of
``results/benchmark_results.json`` so that ``make_figures_spl.py`` and ``make_spl_tables.py``
use the new values. Run it after ``run_experiments.py``.

Usage: python rerun_synthetic_mc.py [n_trials_snr] [n_trials_size] [n_workers]
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, PKG_SRC)

import graphsync as gs  # noqa: E402

GRAPHS = ["er", "ring", "ba"]
SNRS = [-5, 0, 5, 10, 15, 20, 25]
SIZES = [6, 10, 15, 20]


def _one_trial(s, fmin, fmax, zero_init=True):
    r = gs.fit_graphsync(s.observed, fs=1.0, max_lag=6.0, alpha=1.0, beta=0.10,
                         fmin=fmin, fmax=fmax, coherence_power=3.0)
    out = {"gs": r.delays, "f1_gs": gs.edge_f1_score(r.adjacency, s.true_adjacency)[0]}
    if zero_init:
        r0 = gs.fit_graphsync(s.observed, fs=1.0, max_lag=6.0, alpha=1.0, beta=0.10,
                              fmin=fmin, fmax=fmax, init="zeros")
        out["z0"] = r0.delays
    out["ph"] = gs.gcc_phat_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
    out["xc"] = gs.xcorr_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
    out["cy"] = gs.unweighted_cycle_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
    wu, _, _ = gs.unsynced_kalofolias_graph(s.observed, alpha=1.0, beta=0.10, fs=1.0,
                                            fmin=fmin, fmax=fmax)
    out["f1_un"] = gs.edge_f1_score(wu, s.true_adjacency)[0]
    return out


def _summarise(mse, crlb, rng):
    """RMSE, ratio to sqrt(mean CRLB MSE), and trial-bootstrap 95% CI of the ratio."""
    mse = np.asarray(mse)
    crlb = np.asarray(crlb)
    n = mse.size
    idx = rng.integers(0, n, size=(2000, n))
    boot = np.sqrt(mse[idx].mean(1) / crlb[idx].mean(1))
    return (float(np.sqrt(mse.mean())), float(np.sqrt(mse.mean() / crlb.mean())),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def run_snr_cell(args):
    gtype, snr, n_trials = args
    g_idx = GRAPHS.index(gtype)
    acc = {k: [] for k in ["gs", "z0", "ph", "xc", "cy", "crlb", "anchor", "f1_gs", "f1_un"]}
    for tr in range(n_trials):
        seed = 1_000_000 * (g_idx + 1) + 10_000 * (snr + 10) + tr
        s = gs.simulate_asynchronous_network(m_nodes=10, n_samples=2048, fs=1.0, snr_db=float(snr),
                                             max_lag=5.0, graph_type=gtype, edge_prob=0.30,
                                             gamma=0.10, k_min=8, k_max=360, seed=seed)
        fmin = float(s.omegas[0] / (2 * np.pi))
        fmax = float(s.omegas[-1] / (2 * np.pi))
        o = _one_trial(s, fmin, fmax)
        for k in ["gs", "z0", "ph", "xc", "cy"]:
            acc[k].append(float(np.mean((o[k][1:] - s.true_delays[1:]) ** 2)))
        acc["f1_gs"].append(o["f1_gs"])
        acc["f1_un"].append(o["f1_un"])
        acc["crlb"].append(gs.compute_joint_crlb(s.true_laplacian, s.omegas, s.spectral_power,
                                                 s.noise_var, s.gamma).mean_mse_bound)
        acc["anchor"].append(gs.compute_pairwise_anchor_crlb(s.true_laplacian, s.omegas,
                                                             s.spectral_power, s.noise_var,
                                                             s.gamma) ** 2)
    rng = np.random.default_rng(g_idx * 100 + snr + 50)
    gs_rmse, ratio, lo, hi = _summarise(acc["gs"], acc["crlb"], rng)
    f1g, f1u = np.array(acc["f1_gs"]), np.array(acc["f1_un"])
    return {
        "graph_type": gtype, "snr_db": snr, "n_trials": n_trials,
        "crlb_rmse": float(np.sqrt(np.mean(acc["crlb"]))),
        "anchor_crlb_rmse": float(np.sqrt(np.mean(acc["anchor"]))),
        "gs_rmse": gs_rmse, "crlb_ratio": ratio, "crlb_ratio_ci95": [lo, hi],
        "gs_zero_init_rmse": float(np.sqrt(np.mean(acc["z0"]))),
        "cycle_rmse": float(np.sqrt(np.mean(acc["cy"]))),
        "xcorr_rmse": float(np.sqrt(np.mean(acc["xc"]))),
        "phat_rmse": float(np.sqrt(np.mean(acc["ph"]))),
        "f1_gs": float(f1g.mean()), "f1_gs_std": float(f1g.std()),
        "f1_gs_se": float(f1g.std(ddof=1) / np.sqrt(f1g.size)),
        "f1_unsync": float(f1u.mean()), "f1_unsync_std": float(f1u.std()),
        "f1_unsync_se": float(f1u.std(ddof=1) / np.sqrt(f1u.size)),
    }


def run_size_cell(args):
    m_nodes, n_trials = args
    p_edge = min(0.45, max(0.22, 3.0 / m_nodes))
    acc = {k: [] for k in ["gs", "ph", "xc", "cy", "crlb", "f1_gs", "f1_un", "rt"]}
    for tr in range(n_trials):
        s = gs.simulate_asynchronous_network(m_nodes=m_nodes, n_samples=2048, fs=1.0, snr_db=5.0,
                                             max_lag=5.0, graph_type="er", edge_prob=p_edge,
                                             gamma=0.10, k_min=8, k_max=360,
                                             seed=50_000_000 + 100_000 * m_nodes + tr)
        fmin = float(s.omegas[0] / (2 * np.pi))
        fmax = float(s.omegas[-1] / (2 * np.pi))
        t0 = time.perf_counter()
        o = _one_trial(s, fmin, fmax, zero_init=False)
        acc["rt"].append((time.perf_counter() - t0) * 1000.0)
        for k in ["gs", "ph", "xc", "cy"]:
            acc[k].append(float(np.mean((o[k][1:] - s.true_delays[1:]) ** 2)))
        acc["f1_gs"].append(o["f1_gs"])
        acc["f1_un"].append(o["f1_un"])
        acc["crlb"].append(gs.compute_joint_crlb(s.true_laplacian, s.omegas, s.spectral_power,
                                                 s.noise_var, s.gamma).mean_mse_bound)
    rng = np.random.default_rng(m_nodes)
    gs_rmse, ratio, lo, hi = _summarise(acc["gs"], acc["crlb"], rng)
    return {
        "m_nodes": m_nodes, "edge_prob": p_edge, "n_trials": n_trials,
        "crlb_rmse": float(np.sqrt(np.mean(acc["crlb"]))),
        "gs_rmse": gs_rmse, "crlb_ratio": ratio, "crlb_ratio_ci95": [lo, hi],
        "cycle_rmse": float(np.sqrt(np.mean(acc["cy"]))),
        "xcorr_rmse": float(np.sqrt(np.mean(acc["xc"]))),
        "phat_rmse": float(np.sqrt(np.mean(acc["ph"]))),
        "f1_gs": float(np.mean(acc["f1_gs"])), "f1_unsync": float(np.mean(acc["f1_un"])),
        "runtime_ms_incl_baselines": float(np.mean(acc["rt"])),
    }


def main():
    n_snr = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    print(f"Synthetic MC rerun: {n_snr} trials/SNR cell, {n_size} trials/size cell, "
          f"{n_workers} CPU workers (single-threaded BLAS)", flush=True)
    t0 = time.perf_counter()
    snr_jobs = [(g, s, n_snr) for g in GRAPHS for s in SNRS]
    size_jobs = [(m, n_size) for m in SIZES]
    with Pool(n_workers) as pool:
        size_rows = pool.map_async(run_size_cell, size_jobs)
        snr_rows = pool.map(run_snr_cell, snr_jobs, chunksize=1)
        size_rows = size_rows.get()
    for r in snr_rows:
        print(f"[{r['graph_type'].upper():4s} {r['snr_db']:+3d} dB] CRLB={r['crlb_rmse']:.4f} "
              f"GS={r['gs_rmse']:.4f} ratio={r['crlb_ratio']:.3f} "
              f"[{r['crlb_ratio_ci95'][0]:.3f},{r['crlb_ratio_ci95'][1]:.3f}] "
              f"Z0={r['gs_zero_init_rmse']:.3f} Cyc={r['cycle_rmse']:.4f} XC={r['xcorr_rmse']:.4f} "
              f"PHAT={r['phat_rmse']:.4f} F1={r['f1_gs']:.3f}/{r['f1_unsync']:.3f}")
    for r in size_rows:
        print(f"[M={r['m_nodes']:2d}] CRLB={r['crlb_rmse']:.4f} GS={r['gs_rmse']:.4f} "
              f"ratio={r['crlb_ratio']:.3f} PHAT={r['phat_rmse']:.4f} F1={r['f1_gs']:.3f}/{r['f1_unsync']:.3f}")
    synth = {"snr_sweep": snr_rows, "size_sweep": size_rows,
             "n_trials_snr": n_snr, "n_trials_size": n_size,
             "elapsed_s": float(time.perf_counter() - t0)}
    with open(os.path.join(OUT_DIR, "synthetic_mc.json"), "w", encoding="utf-8") as fh:
        json.dump(synth, fh, indent=2)
    bpath = os.path.join(OUT_DIR, "benchmark_results.json")
    with open(bpath, encoding="utf-8") as fh:
        bundle = json.load(fh)
    bundle["synthetic"] = synth
    with open(bpath, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    print(f"Done in {synth['elapsed_s']:.0f}s -> synthetic_mc.json; benchmark_results.json updated")


if __name__ == "__main__":
    main()
