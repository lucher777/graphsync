"""Additional edge-case, scaling, and theorem-property unit tests."""

from __future__ import annotations

import numpy as np
import pytest

import graphsync as gs


def test_joint_objective_matches_time_domain_plus_regularizers() -> None:
    """joint_objective agrees with explicit time-domain Dirichlet + log-barrier + Frobenius."""
    s = gs.simulate_asynchronous_network(m_nodes=6, n_samples=512, snr_db=np.inf, seed=70)
    cs = gs.compute_cross_spectrum(s.observed, fs=1.0, nperseg=None, demean=True)
    tau = s.true_delays + 0.15
    alpha, beta = 0.8, 0.25
    j_freq = gs.joint_objective(cs, tau, s.true_adjacency, alpha=alpha, beta=beta)

    e_time = gs.time_domain_dirichlet_energy(s.observed, tau, s.true_laplacian, fs=1.0, demean=True)
    deg = np.sum(s.true_adjacency, axis=1)
    j_explicit = e_time - alpha * float(np.sum(np.log(deg))) + 0.5 * beta * float(np.sum(s.true_adjacency ** 2))
    assert abs(j_freq - j_explicit) < 1e-12


def test_pairwise_distance_matrix_symmetry_and_zero_diag() -> None:
    """pairwise_distance_matrix is symmetric, nonnegative, with exact zero diagonal."""
    s = gs.simulate_asynchronous_network(m_nodes=8, n_samples=256, snr_db=5.0, seed=71)
    cs = gs.compute_cross_spectrum(s.observed, fs=1.0)
    z = gs.pairwise_distance_matrix(cs, np.linspace(-0.5, 0.5, 8))
    assert np.allclose(z, z.T, atol=1e-14)
    assert np.allclose(np.diag(z), 0.0, atol=1e-15)
    assert np.all(z >= 0.0)


def test_two_node_minimal_network() -> None:
    """Minimal M=2 sensor network works cleanly end-to-end."""
    rng = np.random.default_rng(72)
    n_samples = 512
    spec = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec[5:120] = rng.standard_normal(115) + 1j * rng.standard_normal(115)
    base = np.fft.irfft(spec, n=n_samples)
    clean = np.vstack([base, base])
    tau = np.array([0.0, 1.73])
    obs = gs.apply_offsets(clean, tau, fs=1.0)

    res = gs.fit_graphsync(obs, fs=1.0, max_lag=3.0, fmin=5 / n_samples, fmax=120 / n_samples)
    assert abs(res.delays[1] - 1.73) < 1e-5
    assert res.adjacency[0, 1] > 0.0


def test_disconnected_graph_crlb_graceful_pinv() -> None:
    """Disconnected Laplacian triggers pseudoinverse fallback cleanly in CRLB."""
    w = np.zeros((4, 4), dtype=float)
    w[0, 1] = w[1, 0] = 1.0
    w[2, 3] = w[3, 2] = 1.0
    lap = gs.adjacency_to_laplacian(w)
    res = gs.compute_joint_crlb(lap, np.array([0.5, 1.0]), np.array([1.0, 1.0]), noise_var=0.1)
    assert res.algebraic_connectivity == 0.0
    assert np.all(np.isfinite(res.crlb_matrix))


def test_non_unit_sampling_rate_invariance() -> None:
    """Rescaling fs and delays proportionally yields identical relative timing accuracy."""
    s = gs.simulate_asynchronous_network(
        m_nodes=6, n_samples=512, fs=100.0, snr_db=15.0, max_lag=0.03, k_min=6, k_max=120, seed=73
    )
    fmin = float(s.omegas[0] / (2.0 * np.pi))
    fmax = float(s.omegas[-1] / (2.0 * np.pi))
    res = gs.fit_graphsync(s.observed, fs=100.0, max_lag=0.05, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax)
    rmse_samples = float(np.sqrt(np.mean(((res.delays[1:] - s.true_delays[1:]) * 100.0) ** 2)))
    assert rmse_samples < 0.3


@pytest.mark.parametrize("m_nodes", [6, 12, 16])
def test_scaling_across_network_sizes(m_nodes: int) -> None:
    """JSGB-Sync converges reliably across M in {6, 12, 16} sensor nodes."""
    s = gs.simulate_asynchronous_network(
        m_nodes=m_nodes,
        n_samples=1024,
        snr_db=10.0,
        max_lag=3.5,
        graph_type="ba",
        k_min=6,
        k_max=200,
        seed=740 + m_nodes,
    )
    fmin = float(s.omegas[0] / (2.0 * np.pi))
    fmax = float(s.omegas[-1] / (2.0 * np.pi))
    res = gs.fit_graphsync(s.observed, fs=1.0, max_lag=4.5, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax)
    rmse = float(np.sqrt(np.mean((res.delays[1:] - s.true_delays[1:]) ** 2)))
    assert rmse < 0.35
    assert res.kkt_residual < 1e-8
    assert res.algebraic_connectivity > 0.0


def test_algebraic_connectivity_and_adjacency_validation() -> None:
    """adjacency_to_laplacian rejects non-square matrices and computes valid Laplacian."""
    with pytest.raises(ValueError):
        gs.adjacency_to_laplacian(np.ones((3, 5)))
    w = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]], dtype=float)
    lap = gs.adjacency_to_laplacian(w)
    assert np.isclose(gs.algebraic_connectivity(lap), 2.0, atol=1e-12)


def test_graph_learning_objective_infinite_on_zero_degree() -> None:
    """Log-barrier objective returns +inf when any node has zero degree."""
    w = np.zeros((3, 3), dtype=float)
    z = np.ones((3, 3), dtype=float)
    np.fill_diagonal(z, 0.0)
    assert np.isinf(gs.graph_learning_objective(w, z, alpha=1.0, beta=1.0))


def test_spectral_init_invalid_max_lag_and_threshold_fallback() -> None:
    """spectral_angular_init rejects non-positive max_lag and falls back on high threshold."""
    s = gs.simulate_asynchronous_network(m_nodes=6, n_samples=256, snr_db=5.0, seed=75)
    cs = gs.compute_cross_spectrum(s.observed, fs=1.0)
    with pytest.raises(ValueError):
        gs.spectral_angular_init(cs, max_lag=0.0)
    # Threshold 0.9999 would disconnect the graph; safeguard keeps it connected and finite
    init_res = gs.spectral_angular_init(cs, max_lag=2.0, threshold=0.9999)
    assert np.all(np.isfinite(init_res.delays))
    assert init_res.delays[0] == 0.0


def test_unweighted_cycle_sync_phat_option() -> None:
    """unweighted_cycle_sync supports both XCorr and PHAT pairwise modes."""
    s = gs.simulate_asynchronous_network(m_nodes=6, n_samples=512, snr_db=10.0, seed=76)
    fmin = float(s.omegas[0] / (2.0 * np.pi))
    fmax = float(s.omegas[-1] / (2.0 * np.pi))
    t1 = gs.unweighted_cycle_sync(s.observed, fs=1.0, max_lag=2.5, use_phat=False, fmin=fmin, fmax=fmax)
    t2 = gs.unweighted_cycle_sync(s.observed, fs=1.0, max_lag=2.5, use_phat=True, fmin=fmin, fmax=fmax)
    assert t1.shape == (6,) and t2.shape == (6,)
    assert t1[0] == 0.0 and t2[0] == 0.0
