"""Core data structures for joint graph-topology and time-offset synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CrossSpectrum:
    """One-sided frequency-domain cross-spectral density representation.

    Attributes:
        freqs: Frequency grid in Hz (or cycles/sample when ``fs=1.0``), shape ``(K,)``.
        omegas: Angular frequency grid ``2 * pi * freqs``, shape ``(K,)``.
        weights: Parseval quadrature weights (``1.0`` for DC/Nyquist, ``2.0`` for
            interior one-sided bins), shape ``(K,)``.
        csd: Hermitian cross-spectral density matrices, shape ``(K, M, M)``.
        energies: Per-node signal energies ``sum_k weights[k] * Re(csd[k, m, m])``,
            shape ``(M,)``.
        fs: Sampling frequency in Hz.
        n_samples: Signal length in samples per segment.
        n_blocks: Effective number of averaged spectral blocks.
    """

    freqs: np.ndarray
    omegas: np.ndarray
    weights: np.ndarray
    csd: np.ndarray
    energies: np.ndarray
    fs: float
    n_samples: int
    n_blocks: int = 1


@dataclass(frozen=True)
class SpectralInitResult:
    """Output of the closed-form broadband angular spectral initializer.

    Attributes:
        delays: Initialized clock offsets (anchor ``delays[0] == 0.0``), shape ``(M,)``.
        polarities: Estimated node polarities in ``{-1.0, +1.0}``, shape ``(M,)``.
        pairwise_lags: Pairwise estimated lags ``Delta_ij``, shape ``(M, M)``.
        coherence: Pairwise peak correlation/coherence matrix, shape ``(M, M)``.
        weights: Soft-thresholded coherence weights used for spectral sync, shape ``(M, M)``.
        spectral_gap: Difference ``lambda_1 - lambda_2`` of the normalized Hermitian
            synchronization operator.
        omega_ref: Non-wrapping reference angular frequency used for phase embedding.
    """

    delays: np.ndarray
    polarities: np.ndarray
    pairwise_lags: np.ndarray
    coherence: np.ndarray
    weights: np.ndarray
    spectral_gap: float
    omega_ref: float


@dataclass(frozen=True)
class GraphSyncResult:
    """Result returned by the joint blind synchronization solver ``fit_graphsync``.

    Attributes:
        delays: Estimated clock offsets with ``delays[0] == 0.0``, shape ``(M,)``.
        adjacency: Estimated symmetric nonnegative adjacency matrix ``W``, shape ``(M, M)``.
        laplacian: Estimated combinatorial graph Laplacian ``L = diag(W 1) - W``,
            shape ``(M, M)``.
        polarities: Estimated node polarities in ``{-1.0, +1.0}``, shape ``(M,)``.
        objective_history: Monotone history of the joint objective ``J(tau, W)``.
        init_result: Closed-form spectral initialization diagnostics (if used).
        n_iter: Number of alternating iterations completed.
        converged: Whether the solver met the stopping tolerance.
        kkt_residual: Final primal-dual KKT residual of the graph learning subproblem.
        grad_norm: Final infinity norm of the reduced timing gradient.
        algebraic_connectivity: Fiedler eigenvalue ``lambda_2(L)`` of the estimated graph.
        meta: Auxiliary diagnostics dictionary.
    """

    delays: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    polarities: np.ndarray
    objective_history: List[float]
    init_result: Optional[SpectralInitResult]
    n_iter: int
    converged: bool
    kkt_residual: float
    grad_norm: float
    algebraic_connectivity: float
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CRLBResult:
    """Analytical Fisher Information Matrix and Joint Cramér-Rao Lower Bound.

    Attributes:
        fim_full: Full ``(M, M)`` timing Fisher Information Matrix ``I_{tau,tau}``.
        fim_reduced: Gauge-fixed ``(M-1, M-1)`` timing FIM with node 0 anchored.
        crlb_matrix: Gauge-fixed ``(M-1, M-1)`` inverse FIM covariance lower bound.
        per_node_std: Lower bound on timing standard deviation per node (``0.0`` for
            anchor node 0), shape ``(M,)``.
        mean_mse_bound: Average per-node (non-anchor) MSE lower bound
            ``tr(crlb_matrix) / (M - 1)``.
        rmse_bound: Root-mean-square error lower bound ``sqrt(mean_mse_bound)``.
        cross_block_norm: Verified Frobenius norm of the exact timing-topology
            cross-information block ``I_{tau,w}`` (theoretically ``0.0``).
        algebraic_connectivity: Algebraic connectivity ``lambda_2(L)`` of the true graph.
    """

    fim_full: np.ndarray
    fim_reduced: np.ndarray
    crlb_matrix: np.ndarray
    per_node_std: np.ndarray
    mean_mse_bound: float
    rmse_bound: float
    cross_block_norm: float
    algebraic_connectivity: float


@dataclass(frozen=True)
class SyntheticNetworkSample:
    """Synthetic asynchronous multi-sensor graph signal benchmark instance.

    Attributes:
        observed: Asynchronous noisy sensor signals ``y_m(t)``, shape ``(M, N)``.
        clean: Synchronized ground-truth smooth graph signals ``x_m(t)``, shape ``(M, N)``.
        true_delays: Ground-truth clock offsets with ``true_delays[0] == 0.0``,
            shape ``(M,)``.
        true_adjacency: Ground-truth symmetric weighted adjacency ``W*``, shape ``(M, M)``.
        true_laplacian: Ground-truth graph Laplacian ``L*``, shape ``(M, M)``.
        fs: Sampling frequency.
        snr_db: Nominal signal-to-noise ratio in dB.
        noise_var: Per-sample additive noise variance ``sigma^2``.
        gamma: Graph-Laplacian precision ridge ``gamma`` in ``(L* + gamma I)^{-1}``.
        omegas: Active angular frequencies used in signal synthesis, shape ``(K,)``.
        spectral_power: Per-frequency power envelope ``p(omega_k)``, shape ``(K,)``.
        graph_type: Graph family identifier (``"er"``, ``"ba"``, ``"ring"``, ``"sbm"``).
    """

    observed: np.ndarray
    clean: np.ndarray
    true_delays: np.ndarray
    true_adjacency: np.ndarray
    true_laplacian: np.ndarray
    fs: float
    snr_db: float
    noise_var: float
    gamma: float
    omegas: np.ndarray
    spectral_power: np.ndarray
    graph_type: str
