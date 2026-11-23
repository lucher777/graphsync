"""Tests for exact primal-dual Kalofolias smooth graph learning."""

from __future__ import annotations

import numpy as np
import pytest

import graphsync as gs


def test_learn_smooth_graph_kkt_optimality_and_laplacian_properties() -> None:
    """Primal-dual Kalofolias solver satisfies exact KKT conditions (< 1e-10)."""
    rng = np.random.default_rng(20)
    m_nodes = 10
    pts = rng.normal(size=(m_nodes, 3))
    diff = pts[:, None, :] - pts[None, :, :]
    z_mat = np.sum(diff ** 2, axis=-1)

    w_mat, lap, kkt = gs.learn_smooth_graph(z_mat, alpha=1.0, beta=0.2, tol=1e-10)
    assert kkt < 1e-10
    assert np.allclose(w_mat, w_mat.T, atol=1e-14)
    assert np.all(w_mat >= 0.0)
    assert np.allclose(np.diag(w_mat), 0.0)
    # Valid combinatorial Laplacian
    assert np.allclose(lap @ np.ones(m_nodes), 0.0, atol=1e-12)
    eigvals = np.linalg.eigvalsh(lap)
    assert eigvals[0] > -1e-12
    assert eigvals[1] > 0.0


def test_learn_smooth_graph_sparsity_monotonicity_in_beta() -> None:
    """Smaller beta yields sparser graphs (fewer active edges) for fixed Z and alpha."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=10, n_samples=2048, snr_db=20.0, graph_type="er", edge_prob=0.3, seed=21
    )
    cs = gs.compute_cross_spectrum(
        sample.observed,
        fs=sample.fs,
        fmin=float(sample.omegas[0] / (2.0 * np.pi)),
        fmax=float(sample.omegas[-1] / (2.0 * np.pi)),
    )
    z_mat = gs.pairwise_distance_matrix(cs, sample.true_delays)
    z_mat /= float(np.mean(cs.energies))

    w_sparse, _, _ = gs.learn_smooth_graph(z_mat, alpha=1.0, beta=0.04)
    w_dense, _, _ = gs.learn_smooth_graph(z_mat, alpha=1.0, beta=0.80)
    nnz_sparse = int(np.sum(w_sparse > 1e-5))
    nnz_dense = int(np.sum(w_dense > 1e-5))
    assert nnz_sparse < nnz_dense


def test_learn_smooth_graph_warm_start_consistency() -> None:
    """Warm-starting reaches the exact same unique convex optimum."""
    rng = np.random.default_rng(22)
    m_nodes = 8
    z_raw = rng.uniform(0.1, 2.0, size=(m_nodes, m_nodes))
    z_mat = 0.5 * (z_raw + z_raw.T)
    np.fill_diagonal(z_mat, 0.0)

    w1, _, kkt1 = gs.learn_smooth_graph(z_mat, alpha=1.0, beta=0.15, tol=1e-11)
    w2, _, kkt2 = gs.learn_smooth_graph(z_mat, alpha=1.0, beta=0.15, tol=1e-11, warm_start=w1 + 0.05)
    assert kkt1 < 1e-10 and kkt2 < 1e-10
    assert np.allclose(w1, w2, atol=1e-8)


def test_edge_f1_score_extremes() -> None:
    """edge_f1_score returns 1.0 for identical graphs and 0.0 for disjoint supports."""
    w_true = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    f1_id, p_id, r_id = gs.edge_f1_score(w_true, w_true)
    assert np.isclose(f1_id, 1.0) and np.isclose(p_id, 1.0) and np.isclose(r_id, 1.0)

    w_disjoint = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    f1_zero, p_zero, r_zero = gs.edge_f1_score(w_disjoint, w_true)
    assert f1_zero == 0.0 and p_zero == 0.0 and r_zero == 0.0


def test_learn_smooth_graph_validation() -> None:
    """Non-square distance matrix or non-positive alpha/beta raises ValueError."""
    with pytest.raises(ValueError):
        gs.learn_smooth_graph(np.ones((3, 4)))
    with pytest.raises(ValueError):
        gs.learn_smooth_graph(np.ones((4, 4)), alpha=0.0)
    with pytest.raises(ValueError):
        gs.learn_smooth_graph(np.ones((4, 4)), beta=-1.0)
