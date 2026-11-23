"""Tests for Theorem 3 analytical Fisher Information Matrix and Joint CRLB."""

from __future__ import annotations

import numpy as np
import pytest

import graphsync as gs


def test_crlb_information_orthogonality_and_nullspace() -> None:
    """Theorem 3: cross-block I_{tau,w} == 0 identically and I_{tau,tau} 1 == 0."""
    _, lap = gs.generate_connected_graph(7, graph_type="er", edge_prob=0.4, seed=40)
    omegas = np.linspace(0.2, 2.2, 15)
    p_k = np.exp(-0.25 * omegas)

    crlb = gs.compute_joint_crlb(
        lap, omegas, p_k, noise_var=0.05, gamma=0.2, n_blocks=3, verify_cross_block=True
    )
    assert crlb.cross_block_norm < 1e-12
    assert np.max(np.abs(crlb.fim_full @ np.ones(7))) < 1e-12
    assert crlb.per_node_std[0] == 0.0
    assert np.all(crlb.per_node_std[1:] > 0.0)
    assert crlb.algebraic_connectivity > 0.0


def test_crlb_monotone_in_snr_and_blocks() -> None:
    """CRLB strictly decreases as noise variance decreases or block count B increases."""
    _, lap = gs.generate_connected_graph(8, graph_type="ring", seed=41)
    omegas = np.linspace(0.1, 2.0, 20)
    p_k = np.ones_like(omegas)

    c_low_snr = gs.compute_joint_crlb(lap, omegas, p_k, noise_var=1.0, gamma=0.15, n_blocks=1)
    c_high_snr = gs.compute_joint_crlb(lap, omegas, p_k, noise_var=0.05, gamma=0.15, n_blocks=1)
    c_4blocks = gs.compute_joint_crlb(lap, omegas, p_k, noise_var=0.05, gamma=0.15, n_blocks=4)

    assert c_high_snr.rmse_bound < c_low_snr.rmse_bound
    assert np.isclose(c_4blocks.rmse_bound, 0.5 * c_high_snr.rmse_bound, rtol=1e-12)


def test_crlb_beats_pairwise_anchor_bound() -> None:
    """Joint graph-aware CRLB strictly improves over pairwise anchor-only synchronization."""
    _, lap = gs.generate_connected_graph(10, graph_type="ring", seed=42)
    omegas = np.linspace(0.1, 2.0, 25)
    p_k = np.ones_like(omegas)

    joint_rmse = gs.compute_joint_crlb(lap, omegas, p_k, noise_var=0.2, gamma=0.1).rmse_bound
    anchor_rmse = gs.compute_pairwise_anchor_crlb(lap, omegas, p_k, noise_var=0.2, gamma=0.1)
    assert joint_rmse < anchor_rmse


def test_crlb_empirical_validity_and_high_snr_tightness() -> None:
    """Empirical MSE satisfies MSE >= CRLB (within MC tolerance) and closely tracks CRLB."""
    w_fix, l_fix = gs.generate_connected_graph(8, graph_type="ring", seed=43)
    n_trials = 30
    mses = []
    crlb_mse = None

    for t in range(n_trials):
        s = gs.simulate_asynchronous_network(
            m_nodes=8,
            n_samples=2048,
            snr_db=15.0,
            max_lag=4.0,
            gamma=0.1,
            k_min=8,
            k_max=340,
            seed=4300 + t,
            adjacency=w_fix,
        )
        if crlb_mse is None:
            crlb_mse = gs.compute_joint_crlb(
                s.true_laplacian, s.omegas, s.spectral_power, s.noise_var, s.gamma
            ).mean_mse_bound
        fmin = float(s.omegas[0] / (2.0 * np.pi))
        fmax = float(s.omegas[-1] / (2.0 * np.pi))
        res = gs.fit_graphsync(s.observed, fs=1.0, max_lag=5.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax)
        mses.append(float(np.mean((res.delays[1:] - s.true_delays[1:]) ** 2)))

    emp_rmse = float(np.sqrt(np.mean(mses)))
    crlb_rmse = float(np.sqrt(crlb_mse))
    # Empirical RMSE tracks CRLB tightly (within factor 0.85..1.4 accounting for finite 30-trial MC)
    assert 0.85 * crlb_rmse <= emp_rmse <= 1.40 * crlb_rmse


def test_crlb_input_validation() -> None:
    """Invalid CRLB inputs raise ValueError."""
    lap = gs.adjacency_to_laplacian(np.array([[0, 1], [1, 0]], dtype=float))
    with pytest.raises(ValueError):
        gs.compute_joint_crlb(lap, np.array([1.0]), np.array([1.0]), noise_var=0.0)
    with pytest.raises(ValueError):
        gs.compute_joint_crlb(lap, np.array([1.0]), np.array([1.0]), noise_var=0.1, gamma=-0.1)
    with pytest.raises(ValueError):
        gs.compute_joint_crlb(lap, np.array([1.0]), np.array([1.0, 2.0]), noise_var=0.1)
    with pytest.raises(ValueError):
        gs.compute_joint_crlb(lap, np.array([1.0]), np.array([1.0]), noise_var=0.1, n_blocks=0)
