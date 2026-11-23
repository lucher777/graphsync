"""Strict numerical parity (< 1e-10) and baseline comparison tests."""

from __future__ import annotations

import numpy as np

import graphsync as gs


def test_parseval_gradient_and_hessian_strict_parity() -> None:
    """Verify Parseval energy (< 1e-12), gradient (< 1e-8), and Hessian (< 1e-9) parity."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=7, n_samples=512, snr_db=np.inf, max_lag=2.0, seed=50
    )
    tau_eval = sample.true_delays + np.linspace(0.0, 0.25, 7)
    parity = gs.verify_parseval_and_derivatives(
        sample.observed, tau_eval, sample.true_adjacency, fs=sample.fs
    )
    assert parity["parseval_abs_err"] < 1e-12
    assert parity["parseval_rel_err"] < 1e-12
    assert parity["grad_freq_fd_rel_err"] < 1e-8
    assert parity["grad_time_fd_rel_err"] < 1e-8
    assert parity["hess_fd_rel_err"] < 1e-9


def test_crlb_fim_strict_numerical_parity() -> None:
    """Verify analytical FIM matches exact numerical KL Hessian to < 1e-10 relative error."""
    _, lap = gs.generate_connected_graph(6, graph_type="ba", seed=51)
    omegas = np.linspace(0.25, 2.5, 14)
    p_k = 1.0 + 0.4 * np.cos(omegas)
    parity = gs.verify_crlb_fim_parity(lap, omegas, p_k, noise_var=0.08, gamma=0.2, n_blocks=5)
    assert parity["fim_rel_err"] < 1e-10
    assert parity["fim_nullspace_err"] < 1e-12
    assert parity["cross_block_norm"] < 1e-12


def test_gauss_newton_hessian_is_psd_laplacian() -> None:
    """Gauss-Newton Hessian H_GN is symmetric PSD with exact nullspace span(1)."""
    sample = gs.simulate_asynchronous_network(m_nodes=8, n_samples=512, snr_db=10.0, seed=52)
    cs = gs.compute_cross_spectrum(sample.observed, fs=1.0)
    _, _, hess_gn = gs.timing_gradient_and_hessian(cs, sample.true_delays, sample.true_adjacency)
    assert np.allclose(hess_gn, hess_gn.T, atol=1e-13)
    assert np.allclose(hess_gn @ np.ones(8), 0.0, atol=1e-12)
    eigs = np.linalg.eigvalsh(hess_gn)
    assert eigs[0] > -1e-12
    assert eigs[1] > 0.0


def test_baselines_recover_noiseless_identical_signals() -> None:
    """All timing baselines recover exact delays on noiseless identical broadband signals."""
    rng = np.random.default_rng(53)
    m_nodes, n_samples = 5, 512
    spec = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec[5:140] = rng.standard_normal(135) + 1j * rng.standard_normal(135)
    base = np.fft.irfft(spec, n=n_samples)
    clean = np.tile(base, (m_nodes, 1))
    tau_true = np.array([0.0, 1.45, -2.2, 0.78, -1.33])
    obs = gs.apply_offsets(clean, tau_true, fs=1.0)
    fmin, fmax = 5 / n_samples, 140 / n_samples

    t_phat = gs.gcc_phat_anchor_sync(obs, fs=1.0, max_lag=3.0, fmin=fmin, fmax=fmax)
    t_xc = gs.xcorr_anchor_sync(obs, fs=1.0, max_lag=3.0, fmin=fmin, fmax=fmax)
    t_cyc = gs.unweighted_cycle_sync(obs, fs=1.0, max_lag=3.0, fmin=fmin, fmax=fmax)

    assert np.max(np.abs(t_phat - tau_true)) < 1e-5
    assert np.max(np.abs(t_xc - tau_true)) < 1e-5
    assert np.max(np.abs(t_cyc - tau_true)) < 1e-5


def test_graphsync_outperforms_baselines_at_low_snr() -> None:
    """At low SNR (-2 dB) on sparse ER graphs, JSGB-Sync beats pairwise anchor baselines."""
    errs_gs, errs_phat, errs_xc = [], [], []
    for t in range(5):
        s = gs.simulate_asynchronous_network(
            m_nodes=10,
            n_samples=2048,
            snr_db=-2.0,
            max_lag=5.0,
            graph_type="er",
            edge_prob=0.28,
            gamma=0.1,
            k_min=8,
            k_max=360,
            seed=540 + t,
        )
        fmin = float(s.omegas[0] / (2.0 * np.pi))
        fmax = float(s.omegas[-1] / (2.0 * np.pi))
        r = gs.fit_graphsync(s.observed, fs=1.0, max_lag=6.0, alpha=1.0, beta=0.12, fmin=fmin, fmax=fmax)
        tp = gs.gcc_phat_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
        tx = gs.xcorr_anchor_sync(s.observed, fs=1.0, max_lag=6.0, fmin=fmin, fmax=fmax)
        errs_gs.append(np.mean((r.delays[1:] - s.true_delays[1:]) ** 2))
        errs_phat.append(np.mean((tp[1:] - s.true_delays[1:]) ** 2))
        errs_xc.append(np.mean((tx[1:] - s.true_delays[1:]) ** 2))

    assert np.sqrt(np.mean(errs_gs)) < np.sqrt(np.mean(errs_phat))
    assert np.sqrt(np.mean(errs_gs)) < np.sqrt(np.mean(errs_xc))
