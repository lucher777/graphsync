"""Tests for the Joint Smooth-Graph Blind Synchronization (JSGB-Sync) solver."""

from __future__ import annotations

import numpy as np
import pytest

import graphsync as gs


def test_fit_graphsync_exact_noiseless_population_recovery() -> None:
    """In the noiseless smooth-graph regime, JSGB-Sync recovers exact delays (< 1e-6)."""
    rng = np.random.default_rng(30)
    m_nodes, n_samples = 8, 512
    w_true, l_true = gs.generate_connected_graph(m_nodes, graph_type="ring", seed=30)
    # Strongly smooth broadband deterministic graph signal (dominant smooth mode + small perturbations)
    spec_base = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec_base[6:150] = rng.standard_normal(144) + 1j * rng.standard_normal(144)
    base = np.fft.irfft(spec_base, n=n_samples)
    clean = np.tile(base, (m_nodes, 1))
    tau_true = np.array([0.0, 2.41, -3.12, 1.55, -2.08, 3.67, -1.24, 0.91])
    obs = gs.apply_offsets(clean, tau_true, fs=1.0)

    res = gs.fit_graphsync(
        obs,
        fs=1.0,
        max_lag=4.5,
        alpha=1.0,
        beta=0.1,
        fmin=6 / n_samples,
        fmax=150 / n_samples,
    )
    assert res.converged
    assert res.delays[0] == 0.0
    assert np.max(np.abs(res.delays - tau_true)) < 1e-6
    assert res.kkt_residual < 1e-9
    assert res.algebraic_connectivity > 0.0


def test_fit_graphsync_monotone_objective_descent() -> None:
    """Alternating JSGB-Sync strictly decreases (non-increases) J(tau, W) every iteration."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=8, n_samples=1024, snr_db=8.0, max_lag=4.0, graph_type="er", k_min=6, k_max=200, seed=31
    )
    fmin = float(sample.omegas[0] / (2.0 * np.pi))
    fmax = float(sample.omegas[-1] / (2.0 * np.pi))
    res = gs.fit_graphsync(
        sample.observed, fs=1.0, max_lag=5.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax
    )
    diffs = np.diff(res.objective_history)
    assert np.all(diffs <= 1e-9)
    assert res.objective_history[-1] <= res.objective_history[0] + 1e-12


def test_spectral_init_escapes_local_minima_unlike_zero_init() -> None:
    """Theorem 2 spectral initialization escapes multi-cycle local minima where zero-init fails."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=10,
        n_samples=2048,
        snr_db=12.0,
        max_lag=6.0,
        graph_type="er",
        edge_prob=0.30,
        gamma=0.1,
        k_min=80,
        k_max=420,
        seed=32,
    )
    fmin = float(sample.omegas[0] / (2.0 * np.pi))
    fmax = float(sample.omegas[-1] / (2.0 * np.pi))

    res_spec = gs.fit_graphsync(
        sample.observed, fs=1.0, max_lag=7.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax, init="spectral"
    )
    res_zero = gs.fit_graphsync(
        sample.observed, fs=1.0, max_lag=7.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax, init="zeros"
    )
    rmse_spec = float(np.sqrt(np.mean((res_spec.delays[1:] - sample.true_delays[1:]) ** 2)))
    rmse_zero = float(np.sqrt(np.mean((res_zero.delays[1:] - sample.true_delays[1:]) ** 2)))
    assert rmse_spec < 0.25
    assert rmse_zero > 1.0
    assert rmse_spec < 0.2 * rmse_zero


def test_joint_sync_recovers_topology_while_unsynced_collapses() -> None:
    """Joint synchronization achieves high topology F1 whereas unsynchronized Kalofolias fails."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=10,
        n_samples=2048,
        snr_db=15.0,
        max_lag=5.0,
        graph_type="er",
        edge_prob=0.30,
        gamma=0.1,
        k_min=8,
        k_max=360,
        seed=33,
    )
    fmin = float(sample.omegas[0] / (2.0 * np.pi))
    fmax = float(sample.omegas[-1] / (2.0 * np.pi))

    res = gs.fit_graphsync(
        sample.observed, fs=1.0, max_lag=6.0, alpha=1.0, beta=0.05, fmin=fmin, fmax=fmax
    )
    w_unsync, _, _ = gs.unsynced_kalofolias_graph(
        sample.observed, alpha=1.0, beta=0.05, fs=1.0, fmin=fmin, fmax=fmax
    )
    f1_sync = gs.edge_f1_score(res.adjacency, sample.true_adjacency)[0]
    f1_unsync = gs.edge_f1_score(w_unsync, sample.true_adjacency)[0]
    assert f1_sync > 0.75
    assert f1_sync > f1_unsync + 0.25


def test_fit_graphsync_custom_init_and_invalid_mode() -> None:
    """Custom initialization works and unknown init mode raises ValueError."""
    sample = gs.simulate_asynchronous_network(m_nodes=6, n_samples=512, snr_db=10.0, seed=34)
    init_tau = sample.true_delays + 0.1
    res = gs.fit_graphsync(sample.observed, max_lag=3.0, init="custom", init_delays=init_tau)
    assert res.delays[0] == 0.0
    assert res.init_result is None

    with pytest.raises(ValueError):
        gs.fit_graphsync(sample.observed, init="unknown_mode")
    with pytest.raises(ValueError):
        gs.fit_graphsync(sample.observed, init="custom", init_delays=None)
