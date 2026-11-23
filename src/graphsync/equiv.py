"""Numerical parity checks against explicit time-domain forms and finite differences."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .crlb import compute_joint_crlb
from .solver import timing_gradient_and_hessian
from .spectral import align_signals, compute_cross_spectrum, pairwise_distance_matrix
from .topology import adjacency_to_laplacian


def time_domain_dirichlet_energy(
    signals: np.ndarray,
    delays: np.ndarray,
    laplacian: np.ndarray,
    fs: float = 1.0,
    demean: bool = True,
) -> float:
    """Compute exact time-domain graph Dirichlet energy ``(1/N) sum_n y_tau[n]^T L y_tau[n]``.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        delays: Clock offset vector ``tau`` of shape ``(M,)``.
        laplacian: Combinatorial graph Laplacian ``L`` of shape ``(M, M)``.
        fs: Sampling rate in Hz.
        demean: Whether to remove per-channel means before computing energy.

    Returns:
        Scalar time-domain Dirichlet energy.
    """
    x = np.asarray(signals, dtype=np.float64)
    if demean:
        x = x - np.mean(x, axis=1, keepdims=True)
    y_aligned = align_signals(x, delays, fs=fs)
    # (1/N) * tr(Y^T L Y) = (1/N) * sum(Y * (L @ Y))
    return float(np.sum(y_aligned * (laplacian @ y_aligned)) / float(y_aligned.shape[1]))


def verify_parseval_and_derivatives(
    signals: np.ndarray,
    delays: np.ndarray,
    adjacency: np.ndarray,
    fs: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Verify exact numerical parity between frequency-domain and time-domain formulations.

    Checks four identities to strict precision (``< 1e-10`` for Parseval energy
    and ``< 1e-8`` / ``< 1e-10`` relative error for finite-difference derivatives):

    1. Frequency-domain Parseval energy ``0.5 * tr(W Z(tau))`` vs. explicit time-domain
       quadratic form ``(1/N) sum_n y_tau[n]^T L y_tau[n]``.
    2. Analytic timing gradient ``g(tau)`` vs. central finite-difference gradient of
       the frequency-domain Dirichlet energy.
    3. Analytic timing gradient ``g(tau)`` vs. central finite-difference gradient of
       the explicit time-domain Dirichlet energy.
    4. Exact timing Hessian ``H(tau)`` vs. central finite-difference Jacobian of ``g(tau)``.

    Args:
        signals: Bandlimited multi-sensor array of shape ``(M, N)`` (zero Nyquist energy
            for exact time-domain shift parity when ``N`` is even).
        delays: Evaluation delay vector ``tau`` of shape ``(M,)``.
        adjacency: Symmetric nonnegative adjacency matrix ``W`` of shape ``(M, M)``.
        fs: Sampling rate in Hz.
        eps: Central finite-difference step size.

    Returns:
        Dictionary of maximum absolute and relative discrepancies.
    """
    x = np.asarray(signals, dtype=np.float64)
    tau = np.asarray(delays, dtype=np.float64)
    w_mat = 0.5 * (np.asarray(adjacency, dtype=np.float64) + np.asarray(adjacency, dtype=np.float64).T)
    np.fill_diagonal(w_mat, 0.0)
    lap = adjacency_to_laplacian(w_mat)

    cs = compute_cross_spectrum(x, fs=fs, nperseg=None, demean=True)
    z_mat = pairwise_distance_matrix(cs, tau)
    e_freq = 0.5 * float(np.sum(w_mat * z_mat))
    e_time = time_domain_dirichlet_energy(x, tau, lap, fs=fs, demean=True)

    grad_analytic, hess_analytic, _ = timing_gradient_and_hessian(cs, tau, w_mat)

    m_nodes = tau.shape[0]
    grad_fd_freq = np.zeros(m_nodes, dtype=np.float64)
    grad_fd_time = np.zeros(m_nodes, dtype=np.float64)
    hess_fd = np.zeros((m_nodes, m_nodes), dtype=np.float64)

    for m in range(m_nodes):
        em = np.zeros(m_nodes, dtype=np.float64)
        em[m] = eps

        ef_p = 0.5 * float(np.sum(w_mat * pairwise_distance_matrix(cs, tau + em)))
        ef_m = 0.5 * float(np.sum(w_mat * pairwise_distance_matrix(cs, tau - em)))
        grad_fd_freq[m] = (ef_p - ef_m) / (2.0 * eps)

        et_p = time_domain_dirichlet_energy(x, tau + em, lap, fs=fs, demean=True)
        et_m = time_domain_dirichlet_energy(x, tau - em, lap, fs=fs, demean=True)
        grad_fd_time[m] = (et_p - et_m) / (2.0 * eps)

        gp, _, _ = timing_gradient_and_hessian(cs, tau + em, w_mat)
        gm, _, _ = timing_gradient_and_hessian(cs, tau - em, w_mat)
        hess_fd[:, m] = (gp - gm) / (2.0 * eps)

    scale_e = max(abs(e_time), 1.0)
    scale_g = max(float(np.max(np.abs(grad_analytic))), 1.0)
    scale_h = max(float(np.max(np.abs(hess_analytic))), 1.0)

    return {
        "parseval_abs_err": abs(e_freq - e_time),
        "parseval_rel_err": abs(e_freq - e_time) / scale_e,
        "grad_freq_fd_abs_err": float(np.max(np.abs(grad_analytic - grad_fd_freq))),
        "grad_freq_fd_rel_err": float(np.max(np.abs(grad_analytic - grad_fd_freq))) / scale_g,
        "grad_time_fd_abs_err": float(np.max(np.abs(grad_analytic - grad_fd_time))),
        "grad_time_fd_rel_err": float(np.max(np.abs(grad_analytic - grad_fd_time))) / scale_g,
        "hess_fd_abs_err": float(np.max(np.abs(hess_analytic - hess_fd))),
        "hess_fd_rel_err": float(np.max(np.abs(hess_analytic - hess_fd))) / scale_h,
    }


def verify_crlb_fim_parity(
    laplacian: np.ndarray,
    omegas: np.ndarray,
    spectral_power: np.ndarray,
    noise_var: float,
    gamma: float = 0.2,
    n_blocks: int = 1,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Verify analytical Theorem 3 FIM against exact numerical KL-divergence Hessian.

    Args:
        laplacian: Combinatorial graph Laplacian ``L``, shape ``(M, M)``.
        omegas: Positive one-sided angular frequencies, shape ``(K,)``.
        spectral_power: Per-frequency power envelope, shape ``(K,)``.
        noise_var: Noise variance ``sigma^2``.
        gamma: Precision ridge parameter.
        n_blocks: Number of temporal blocks.
        eps: Central finite-difference step size.

    Returns:
        Dictionary with FIM max relative error, row-sum nullspace error, and
        explicit cross-block Frobenius norm ``||I_{tau,w}||_F``.
    """
    crlb_res = compute_joint_crlb(
        laplacian,
        omegas,
        spectral_power,
        noise_var=noise_var,
        gamma=gamma,
        n_blocks=n_blocks,
        verify_cross_block=True,
    )
    m_nodes = laplacian.shape[0]
    cov_base = np.linalg.inv(0.5 * (laplacian + laplacian.T) + float(gamma) * np.eye(m_nodes))
    eye_m = np.eye(m_nodes)

    def grad_kl(tau_vec: np.ndarray) -> np.ndarray:
        g = np.zeros(m_nodes, dtype=np.float64)
        for om, pk in zip(omegas, spectral_power):
            c_true = float(pk) * cov_base + float(noise_var) * eye_m
            d_mat = np.diag(np.exp(1j * float(om) * tau_vec))
            c_cur = d_mat.conj().T @ (float(pk) * cov_base) @ d_mat + float(noise_var) * eye_m
            c_inv = np.linalg.inv(c_cur)
            m_mat = c_inv - c_inv @ c_true @ c_inv
            g += float(n_blocks) * (-2.0 * float(om) * np.imag(np.diag(m_mat @ c_cur)))
        return g

    tau0 = np.zeros(m_nodes, dtype=np.float64)
    h_fd = np.zeros((m_nodes, m_nodes), dtype=np.float64)
    for m in range(m_nodes):
        em = np.zeros(m_nodes, dtype=np.float64)
        em[m] = eps
        h_fd[:, m] = (grad_kl(tau0 + em) - grad_kl(tau0 - em)) / (2.0 * eps)

    scale = max(float(np.max(np.abs(crlb_res.fim_full))), 1.0)
    return {
        "fim_abs_err": float(np.max(np.abs(crlb_res.fim_full - h_fd))),
        "fim_rel_err": float(np.max(np.abs(crlb_res.fim_full - h_fd))) / scale,
        "fim_nullspace_err": float(np.max(np.abs(np.sum(crlb_res.fim_full, axis=1)))),
        "cross_block_norm": float(crlb_res.cross_block_norm),
    }
