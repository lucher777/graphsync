"""Analytical Fisher Information Matrix and Joint Cramér-Rao Lower Bound (Theorem 3)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg

from .topology import algebraic_connectivity
from .types import CRLBResult


def compute_joint_crlb(
    laplacian: np.ndarray,
    omegas: np.ndarray,
    spectral_power: np.ndarray,
    noise_var: float,
    gamma: float = 0.2,
    n_blocks: int = 1,
    verify_cross_block: bool = False,
) -> CRLBResult:
    """Compute the exact analytical Joint CRLB for blind multi-sensor synchronization.

    Under the one-sided circularly symmetric complex Gaussian smooth-graph model
        ``Y(omega_k) ~ CN(0, D_k^H [p_k (L + gamma I)^{-1}] D_k + sigma^2 I)``,
    Theorem 3 establishes:

    1. **Exact Information Orthogonality**: The Fisher Information cross-block
       ``I_{tau, w}`` between unknown clock offsets ``tau`` and unknown edge
       weights ``w`` vanishes identically (``I_{tau, w} == 0``), so unknown
       topology incurs zero asymptotic timing CRLB penalty.
    2. **Closed-Form Timing FIM**:
       ``I_{tau,tau} = 2 * B * sum_k omega_k^2 * (C_{k,0}^{-1} * C_{k,0} - I_M)``,
       where ``C_{k,0} = p_k (L + gamma I)^{-1} + sigma^2 I`` and ``*`` denotes
       the Hadamard product.

    Args:
        laplacian: True combinatorial graph Laplacian ``L`` of shape ``(M, M)``.
        omegas: Strictly positive one-sided angular frequencies ``omega_k > 0``,
            shape ``(K,)``.
        spectral_power: Signal power spectrum ``p(omega_k) >= 0`` at each frequency
            (matching the one-sided complex DFT variance convention), shape ``(K,)``.
        noise_var: Additive sensor noise variance ``sigma^2 > 0`` per DFT bin.
        gamma: Precision ridge ``gamma > 0`` ensuring ``L + gamma I`` is invertible.
        n_blocks: Number of independent temporal blocks ``B >= 1``.
        verify_cross_block: If ``True``, explicitly computes the Slepian-Bangs
            timing-edge cross-block Frobenius norm to verify orthogonality.

    Returns:
        :class:`CRLBResult` containing FIM matrices, per-node std bounds, and RMSE bound.
    """
    lap = np.asarray(laplacian, dtype=np.float64)
    om = np.asarray(omegas, dtype=np.float64)
    pk = np.asarray(spectral_power, dtype=np.float64)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise ValueError(f"Expected square Laplacian, got shape {lap.shape}.")
    if om.ndim != 1 or pk.shape != om.shape:
        raise ValueError(f"omegas and spectral_power must be 1-D arrays of equal length.")
    if noise_var <= 0.0:
        raise ValueError(f"noise_var must be strictly positive, got {noise_var}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")

    m_nodes = lap.shape[0]
    lap_sym = 0.5 * (lap + lap.T)
    prec_base = lap_sym + float(gamma) * np.eye(m_nodes)
    cov_base = linalg.inv(prec_base)

    fim_full = np.zeros((m_nodes, m_nodes), dtype=np.float64)
    eye_m = np.eye(m_nodes)

    for w_k, p_k in zip(om, pk):
        if w_k <= 0.0 or p_k <= 0.0:
            continue
        c0 = float(p_k) * cov_base + float(noise_var) * eye_m
        c0_inv = linalg.inv(c0)
        fim_full += 2.0 * float(n_blocks) * (float(w_k) ** 2) * (c0_inv * c0 - eye_m)

    fim_full = 0.5 * (fim_full + fim_full.T)
    fim_reduced = fim_full[1:, 1:].copy()

    # Use pseudoinverse fallback only if the graph is disconnected
    lam2 = algebraic_connectivity(lap_sym)
    if lam2 > 1e-12:
        crlb_matrix = linalg.inv(fim_reduced)
    else:
        crlb_matrix = linalg.pinv(fim_reduced)
    crlb_matrix = 0.5 * (crlb_matrix + crlb_matrix.T)

    per_node_var = np.zeros(m_nodes, dtype=np.float64)
    per_node_var[1:] = np.maximum(np.diag(crlb_matrix), 0.0)
    per_node_std = np.sqrt(per_node_var)
    mean_mse = float(np.mean(per_node_var[1:]))
    rmse_bound = float(np.sqrt(mean_mse))

    cross_norm = 0.0
    if verify_cross_block:
        # Explicitly evaluate Slepian-Bangs tr(C^{-1} dC/dtau_m C^{-1} dC/dw_e) for all edges
        iu, ju = np.triu_indices(m_nodes, k=1)
        cross_mat = np.zeros((m_nodes - 1, iu.shape[0]), dtype=np.complex128)
        for w_k, p_k in zip(om, pk):
            if w_k <= 0.0 or p_k <= 0.0:
                continue
            c0 = float(p_k) * cov_base + float(noise_var) * eye_m
            c0_inv = linalg.inv(c0)
            for e_idx, (i, j) in enumerate(zip(iu, ju)):
                e_ij = np.zeros((m_nodes, m_nodes), dtype=np.float64)
                e_ij[i, i] = 1.0
                e_ij[j, j] = 1.0
                e_ij[i, j] = -1.0
                e_ij[j, i] = -1.0
                # dC0/dw_e = -p_k * cov_base @ E_ij @ cov_base
                dc_dw = -float(p_k) * (cov_base @ e_ij @ cov_base)
                a_mat = c0_inv @ dc_dw @ c0_inv
                for m in range(1, m_nodes):
                    e_m = np.zeros((m_nodes, m_nodes), dtype=np.float64)
                    e_m[m, m] = 1.0
                    dc_dt = 1j * float(w_k) * (c0 @ e_m - e_m @ c0)
                    cross_mat[m - 1, e_idx] += float(n_blocks) * np.trace(dc_dt @ a_mat)
        cross_norm = float(np.linalg.norm(cross_mat))

    return CRLBResult(
        fim_full=fim_full,
        fim_reduced=fim_reduced,
        crlb_matrix=crlb_matrix,
        per_node_std=per_node_std,
        mean_mse_bound=mean_mse,
        rmse_bound=rmse_bound,
        cross_block_norm=cross_norm,
        algebraic_connectivity=lam2,
    )


def compute_pairwise_anchor_crlb(
    laplacian: np.ndarray,
    omegas: np.ndarray,
    spectral_power: np.ndarray,
    noise_var: float,
    gamma: float = 0.2,
    n_blocks: int = 1,
) -> float:
    """Compute the RMSE CRLB achievable by pairwise anchor-only synchronization.

    Restricts observation to the ``(M - 1)`` independent 2-node pairs ``(0, m)``
    with marginal ``2 x 2`` covariance matrices, illustrating the information gain
    obtained by exploiting multi-hop graph connectivity in Theorem 3.

    Args:
        laplacian: True combinatorial graph Laplacian ``L`` of shape ``(M, M)``.
        omegas: Positive one-sided angular frequencies, shape ``(K,)``.
        spectral_power: Per-frequency power spectrum, shape ``(K,)``.
        noise_var: Additive noise variance per DFT bin.
        gamma: Precision ridge parameter.
        n_blocks: Number of independent temporal blocks.

    Returns:
        Pairwise anchor-only RMSE lower bound across non-anchor nodes.
    """
    lap = 0.5 * (np.asarray(laplacian, dtype=np.float64) + np.asarray(laplacian, dtype=np.float64).T)
    m_nodes = lap.shape[0]
    cov_base = linalg.inv(lap + float(gamma) * np.eye(m_nodes))
    eye2 = np.eye(2)

    vars_anchor = np.zeros(m_nodes - 1, dtype=np.float64)
    for m in range(1, m_nodes):
        fim_m = 0.0
        sub_cov = cov_base[np.ix_([0, m], [0, m])]
        for w_k, p_k in zip(omegas, spectral_power):
            if w_k <= 0.0 or p_k <= 0.0:
                continue
            c2 = float(p_k) * sub_cov + float(noise_var) * eye2
            c2_inv = linalg.inv(c2)
            # Scalar FIM for pair (0, m) is entry (1, 1) of 2*B*w^2*(C2^{-1} * C2 - I)
            fim_m += 2.0 * float(n_blocks) * (float(w_k) ** 2) * (c2_inv[1, 1] * c2[1, 1] - 1.0)
        vars_anchor[m - 1] = 1.0 / max(fim_m, 1e-18)

    return float(np.sqrt(np.mean(vars_anchor)))
