"""Tests for bundled real BIDMC physiological network, graph generators, and CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

import graphsync as gs
from graphsync import cli


def test_load_bundled_bidmc_slice() -> None:
    """Bundled real BIDMC 5-channel physiological dataset loads cleanly with expected shape."""
    data = gs.load_bundled_bidmc_slice()
    assert data["signals"].ndim == 3
    assert data["signals"].shape[1] == 5
    assert data["fs"] == 125.0
    assert list(data["channels"]) == ["II", "V", "AVR", "PLETH", "RESP"]
    assert np.all(np.isfinite(data["signals"]))


def test_bidmc_real_physiological_sync_and_topology() -> None:
    """JSGB-Sync recovers injected offsets and graph topology on real BIDMC 5-channel recordings."""
    data = gs.load_bundled_bidmc_slice()
    fs = data["fs"]
    sigs = data["signals"][0].copy()  # Subject 1: 5 channels x 3750 samples (30s)
    sigs = (sigs - sigs.mean(axis=1, keepdims=True)) / (sigs.std(axis=1, keepdims=True) + 1e-12)

    # 1. Direct polarity-aware synchronization of the 3 hardware-synchronized ECG leads (II, V, AVR)
    ecg3 = sigs[:3]
    tau_ecg = np.array([0.0, 0.064, -0.096])  # +64 ms, -96 ms
    obs_ecg = gs.apply_offsets(ecg3, tau_ecg, fs=fs)
    res_ecg = gs.fit_graphsync(
        obs_ecg,
        fs=fs,
        max_lag=0.20,
        alpha=1.0,
        beta=0.1,
        fmin=0.5,
        fmax=25.0,
        nperseg=512,
        allow_polarity_flip=True,
    )
    mae_ms = float(np.mean(np.abs(res_ecg.delays[1:] - tau_ecg[1:])) * 1000.0)
    assert mae_ms < 5.0  # Sub-5ms on real clinical ECG leads

    # 2. 5-node physiological sensor network driven by real BIDMC channels
    w_true, l_true = gs.generate_connected_graph(5, graph_type="ring", seed=60)
    rng = np.random.default_rng(60)
    # Use 5 temporally distinct real segments from bundled subjects so excitations are independent
    exc = np.stack([data["signals"][k, k % 5] for k in range(5)], axis=0)
    exc = (exc - exc.mean(axis=1, keepdims=True)) / (exc.std(axis=1, keepdims=True) + 1e-12)
    ev, u_mat = np.linalg.eigh(l_true + 0.12 * np.eye(5))
    net_sig = ((u_mat * (1.0 / np.sqrt(ev))) @ u_mat.T) @ exc
    net_sig = (net_sig - net_sig.mean(axis=1, keepdims=True)) / (net_sig.std(axis=1, keepdims=True) + 1e-12)

    tau5 = np.array([0.0, 0.088, -0.112, 0.056, -0.072])
    obs5 = gs.apply_offsets(net_sig, tau5, fs=fs) + rng.normal(0.0, 0.35, size=net_sig.shape)
    res5 = gs.fit_graphsync(
        obs5, fs=fs, max_lag=0.25, alpha=1.0, beta=0.06, fmin=0.2, fmax=20.0, nperseg=512, whitening="trace"
    )
    w_un5, _, _ = gs.unsynced_kalofolias_graph(
        obs5, alpha=1.0, beta=0.06, fs=fs, fmin=0.2, fmax=20.0, nperseg=512, whitening="trace"
    )
    mae5_ms = float(np.mean(np.abs(res5.delays[1:] - tau5[1:])) * 1000.0)
    f1_sync = gs.edge_f1_score(res5.adjacency, w_true)[0]
    f1_un = gs.edge_f1_score(w_un5, w_true)[0]
    assert mae5_ms < 5.0
    assert f1_sync >= f1_un


@pytest.mark.parametrize("graph_type", ["er", "ba", "ring", "sbm"])
def test_generate_connected_graph_families(graph_type: str) -> None:
    """All supported graph families generate valid connected symmetric graphs."""
    w_mat, lap = gs.generate_connected_graph(10, graph_type=graph_type, seed=61)
    assert w_mat.shape == (10, 10)
    assert np.allclose(w_mat, w_mat.T)
    assert np.all(w_mat >= 0.0)
    assert np.allclose(np.diag(w_mat), 0.0)
    assert gs.algebraic_connectivity(lap) > 1e-4


def test_generate_connected_graph_invalid_args() -> None:
    """Invalid node count or graph family raises ValueError."""
    with pytest.raises(ValueError):
        gs.generate_connected_graph(2)
    with pytest.raises(ValueError):
        gs.generate_connected_graph(6, graph_type="invalid")


def test_cli_demo_and_sync(tmp_path, capsys) -> None:
    """CLI demo and sync subcommands execute cleanly and emit valid JSON."""
    rc = cli.main(["demo", "--nodes", "6", "--samples", "512", "--snr", "12", "--seed", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["nodes"] == 6
    assert payload["timing_rmse"] >= 0.0
    assert payload["crlb_rmse"] > 0.0

    sample = gs.simulate_asynchronous_network(m_nodes=5, n_samples=256, snr_db=12.0, seed=8)
    in_npy = tmp_path / "signals.npy"
    out_json = tmp_path / "result.json"
    np.save(in_npy, sample.observed)

    rc2 = cli.main(["sync", str(in_npy), "--fs", "1.0", "--max-lag", "2.0", "--output-json", str(out_json)])
    assert rc2 == 0
    res_data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(res_data["delays"]) == 5
    assert res_data["delays"][0] == 0.0
    assert res_data["kkt_residual"] < 1e-8
