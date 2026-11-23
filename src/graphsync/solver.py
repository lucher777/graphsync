"""Joint Smooth-Graph Blind Synchronization (JSGB-Sync) solver."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import linalg

from .spectral import (
    compute_cross_spectrum,
    evaluate_cross_correlation_derivatives,
    pairwise_distance_matrix,
    spectral_angular_init,
)
from .topology import (
    adjacency_to_laplacian,
    algebraic_connectivity,
    graph_learning_objective,
    learn_smooth_graph,
)
from .types import CrossSpectrum, GraphSyncResult, SpectralInitResult


def joint_objective(
    cs: CrossSpectrum,
    delays: np.ndarray,
    adjacency: np.ndarray,
    alpha: float,
    beta: float,
    polarities: Optional[np.ndarray] = None,
) -> float:
    """Evaluate the Joint Smooth-Graph Blind Synchronization objective ``J(tau, W)``.

    Args:
        cs: Precomputed :class:`CrossSpectrum`.
        delays: Clock offset vector ``tau`` of shape ``(M,)``.
        adjacency: Symmetric nonnegative adjacency matrix ``W`` of shape ``(M, M)``.
        alpha: Logarithmic degree-barrier weight.
        beta: Frobenius regularization weight.
        polarities: Optional sign vector in ``{-1, +1}^M``.

    Returns:
        Scalar objective value.
    """
    z_mat = pairwise_distance_matrix(cs, delays, polarities=polarities)
    return graph_learning_objective(adjacency, z_mat, alpha=alpha, beta=beta)


def timing_gradient_and_hessian(
    cs: CrossSpectrum,
    delays: np.ndarray,
    adjacency: np.ndarray,
    polarities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact timing gradient, exact Hessian, and PSD Gauss-Newton Laplacian Hessian.

    For fixed ``W``, the timing-dependent part of ``J(tau, W)`` is
    ``0.5 * sum_{i,j} W_{ij} (E_i + E_j - 2 R_{ij}(tau_i - tau_j))``.
    Its exact gradient is ``g_i = -2 sum_j W_{ij} R'_{ij}(tau_i - tau_j)``,
    and its exact Hessian has off-diagonals ``H_{ij} = 2 W_{ij} R''_{ij}(tau_i - tau_j)``.
    The Gauss-Newton surrogate replaces ``-R''_{ij}`` with ``max(-R''_{ij}, eps)``,
    yielding a guaranteed positive semidefinite weighted graph Laplacian ``2 * L_{W_tilde}``.

    Args:
        cs: Precomputed :class:`CrossSpectrum`.
        delays: Clock offset vector ``tau`` of shape ``(M,)``.
        adjacency: Symmetric nonnegative adjacency matrix ``W`` of shape ``(M, M)``.
        polarities: Optional sign vector in ``{-1, +1}^M``.

    Returns:
        Tuple ``(grad, hess_exact, hess_gn)`` of shapes ``(M,)``, ``(M, M)``, ``(M, M)``.
    """
    _, r1, r2 = evaluate_cross_correlation_derivatives(cs, delays, polarities=polarities)
    w_sym = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(w_sym, 0.0)

    grad = -2.0 * np.sum(w_sym * r1, axis=1)

    # Exact Hessian: off-diagonal 2 * W_ij * R2_ij, diagonal -2 * sum_{j!=i} W_ij * R2_ij
    off_exact = 2.0 * w_sym * r2
    np.fill_diagonal(off_exact, 0.0)
    hess_exact = off_exact - np.diag(np.sum(off_exact, axis=1))

    # PSD Gauss-Newton weighted graph Laplacian: 2 * L_{W_tilde}, W_tilde_ij = W_ij * max(-R2_ij, 1e-6)
    curv = np.maximum(-r2, 1e-6)
    w_tilde = w_sym * curv
    np.fill_diagonal(w_tilde, 0.0)
    hess_gn = 2.0 * adjacency_to_laplacian(w_tilde)
    return grad, hess_exact, hess_gn


def _refine_delays_gauss_newton(
    cs: CrossSpectrum,
    delays: np.ndarray,
    adjacency: np.ndarray,
    max_lag: float,
    polarities: np.ndarray,
    max_steps: int = 12,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Damped Gauss-Newton / Levenberg-Marquardt update for ``tau`` with Armijo backtracking."""
    tau = delays.copy()
    tau[0] = 0.0
    m_nodes = tau.shape[0]
    damping = 1e-6

    def dirichlet_energy(t_vec: np.ndarray) -> float:
        z_m = pairwise_distance_matrix(cs, t_vec, polarities=polarities)
        return 0.5 * float(np.sum(adjacency * z_m))

    cur_val = dirichlet_energy(tau)
    grad_norm = float("inf")
    # Trust-region cap prevents single-step jumps across high-frequency correlation lobes
    f_rms = float(
        np.sqrt(
            np.sum(cs.weights * (cs.freqs ** 2)) / max(float(np.sum(cs.weights)), 1e-30)
        )
    )
    trust_cap = min(0.5 * max_lag, 0.25 / max(f_rms, 1e-6))

    for _ in range(max_steps):
        grad, _, hess_gn = timing_gradient_and_hessian(cs, tau, adjacency, polarities=polarities)
        g_red = grad[1:]
        grad_norm = float(np.max(np.abs(g_red))) if g_red.size else 0.0
        if grad_norm <= tol:
            break

        h_red = hess_gn[1:, 1:]
        h_scale = float(np.trace(h_red)) / max(1, m_nodes - 1) + 1e-12
        eye_red = np.eye(m_nodes - 1)

        accepted = False
        for _lm in range(8):
            try:
                step_red = linalg.solve(
                    h_red + (damping * h_scale) * eye_red,
                    -g_red,
                    assume_a="pos",
                )
            except linalg.LinAlgError:
                damping *= 10.0
                continue

            step_inf = float(np.max(np.abs(step_red))) if step_red.size else 0.0
            if step_inf > trust_cap and step_inf > 0.0:
                step_red = step_red * (trust_cap / step_inf)

            # Armijo backtracking line search with box projection
            eta = 1.0
            for _ls in range(12):
                cand = tau.copy()
                cand[1:] = np.clip(tau[1:] + eta * step_red, -max_lag, max_lag)
                cand_val = dirichlet_energy(cand)
                if cand_val < cur_val - 1e-12 * abs(cur_val) - 1e-15:
                    tau = cand
                    cur_val = cand_val
                    damping = max(1e-9, damping * 0.5)
                    accepted = True
                    break
                eta *= 0.5

            if accepted:
                break
            damping = min(1e6, damping * 10.0)

        if not accepted:
            break

    grad_final, _, _ = timing_gradient_and_hessian(cs, tau, adjacency, polarities=polarities)
    grad_norm = float(np.max(np.abs(grad_final[1:]))) if m_nodes > 1 else 0.0
    return tau, grad_norm


def fit_graphsync(
    signals: np.ndarray,
    fs: float = 1.0,
    max_lag: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.12,
    max_iter: int = 30,
    tol: float = 1e-7,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    whitening: str = "none",
    scale_energy: bool = True,
    init: str = "spectral",
    init_delays: Optional[np.ndarray] = None,
    coherence_power: float = 2.0,
    coherence_threshold: float = 0.0,
    allow_polarity_flip: bool = False,
    use_phat_init: bool = False,
    normalize_signals: bool = False,
    graph_tol: float = 1e-10,
) -> GraphSyncResult:
    """Jointly estimate sensor clock offsets ``tau`` and unknown smooth graph Laplacian ``L``.

    Solves the Joint Smooth-Graph Blind Synchronization (``JSGB-Sync``) problem:
        ``min_{tau, tau_1=0, W in W_M} 0.5 tr(W Z(tau)) - alpha 1^T log(W 1) + 0.5 beta ||W||_F^2``
    by combining:
    1. Closed-form broadband weighted Hermitian angular spectral initialization (Theorem 2).
    2. Exact primal-dual smooth graph learning (Kalofolias 2016) alternating with
       damped Gauss-Newton / Levenberg-Marquardt offset refinement using the exact
       frequency-domain gradient and PSD graph-Laplacian Hessian.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        fs: Sampling frequency in Hz (default ``1.0`` for sample units).
        max_lag: Maximum absolute clock offset per sensor (in seconds or samples).
        alpha: Positive logarithmic degree-barrier parameter.
        beta: Positive Frobenius regularization parameter (controls edge sparsity).
        max_iter: Maximum alternating minimization outer iterations.
        tol: Outer convergence tolerance on delays and adjacency.
        fmin: Optional lower frequency bound in Hz.
        fmax: Optional upper frequency bound in Hz.
        nperseg: Optional Welch segment length for long/noisy streams.
        whitening: Spectral whitening mode in ``{"none", "trace", "coherency"}``.
        scale_energy: If ``True``, divides the CSD by the constant mean node energy
            ``(1/M) sum_m E_m`` (independent of ``tau``) so ``alpha, beta`` are
            dimensionless and scale-invariant while preserving exact monotonic descent.
        init: Initialization mode: ``"spectral"`` (default, Theorem 2), ``"zeros"``,
            or ``"custom"`` (requires ``init_delays``).
        init_delays: Custom initial delay vector of shape ``(M,)`` when ``init="custom"``.
        coherence_power: Exponent for spectral angular synchronization weights.
        coherence_threshold: Soft threshold on pairwise coherence for spectral init.
        allow_polarity_flip: If ``True``, jointly estimates ``{-1, +1}`` channel polarities.
        use_phat_init: If ``True``, uses coherence-regularized PHAT in spectral init.
        normalize_signals: If ``True``, z-scores each channel before spectral estimation.
        graph_tol: KKT convergence tolerance for the inner graph learning subproblem.

    Returns:
        :class:`GraphSyncResult` containing estimated offsets, adjacency, Laplacian,
        and convergence diagnostics.
    """
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D signals array (M, N), got shape {arr.shape}.")
    if normalize_signals:
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        stds = np.std(arr, axis=1, keepdims=True)
        arr = arr / np.where(stds > 1e-12, stds, 1.0)

    cs = compute_cross_spectrum(
        arr, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg, whitening=whitening
    )
    if scale_energy:
        mean_e = float(np.mean(cs.energies))
        if mean_e > 1e-15:
            cs = CrossSpectrum(
                freqs=cs.freqs,
                omegas=cs.omegas,
                weights=cs.weights,
                csd=cs.csd / mean_e,
                energies=cs.energies / mean_e,
                fs=cs.fs,
                n_samples=cs.n_samples,
                n_blocks=cs.n_blocks,
            )
    m_nodes = cs.csd.shape[1]

    init_res: Optional[SpectralInitResult] = None
    if init == "spectral":
        init_res = spectral_angular_init(
            cs,
            max_lag=max_lag,
            coherence_power=coherence_power,
            threshold=coherence_threshold,
            allow_polarity_flip=allow_polarity_flip,
            use_phat=use_phat_init,
        )
        tau = init_res.delays.copy()
        polarities = init_res.polarities.copy()
    elif init == "zeros":
        tau = np.zeros(m_nodes, dtype=np.float64)
        polarities = np.ones(m_nodes, dtype=np.float64)
    elif init == "custom":
        if init_delays is None or np.asarray(init_delays).shape != (m_nodes,):
            raise ValueError("init='custom' requires init_delays of shape (M,).")
        tau = np.asarray(init_delays, dtype=np.float64).copy()
        tau = tau - tau[0]
        tau = np.clip(tau, -max_lag, max_lag)
        polarities = np.ones(m_nodes, dtype=np.float64)
    else:
        raise ValueError(f"Unknown init mode '{init}'. Expected 'spectral', 'zeros', or 'custom'.")

    # Initial graph learning at tau^(0)
    z_mat = pairwise_distance_matrix(cs, tau, polarities=polarities)
    w_mat, lap, kkt = learn_smooth_graph(z_mat, alpha=alpha, beta=beta, tol=graph_tol)
    obj_hist: List[float] = [graph_learning_objective(w_mat, z_mat, alpha=alpha, beta=beta)]

    converged = False
    grad_norm = float("inf")
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        tau_prev = tau.copy()
        w_prev = w_mat.copy()

        # Step 1: Damped Gauss-Newton offset update given W
        tau, grad_norm = _refine_delays_gauss_newton(
            cs,
            tau,
            w_mat,
            max_lag=max_lag,
            polarities=polarities,
            max_steps=12,
            tol=min(tol * 0.1, 1e-9),
        )

        # Step 2: Exact primal-dual smooth graph learning given tau
        z_mat = pairwise_distance_matrix(cs, tau, polarities=polarities)
        w_mat, lap, kkt = learn_smooth_graph(
            z_mat,
            alpha=alpha,
            beta=beta,
            tol=graph_tol,
            warm_start=w_mat,
        )

        obj_val = graph_learning_objective(w_mat, z_mat, alpha=alpha, beta=beta)
        obj_hist.append(obj_val)

        d_tau = float(np.max(np.abs(tau - tau_prev)))
        d_w = float(np.linalg.norm(w_mat - w_prev) / (np.linalg.norm(w_prev) + 1e-12))
        if d_tau <= tol and d_w <= tol:
            converged = True
            break

    lam2 = algebraic_connectivity(lap)
    return GraphSyncResult(
        delays=tau,
        adjacency=w_mat,
        laplacian=lap,
        polarities=polarities,
        objective_history=obj_hist,
        init_result=init_res,
        n_iter=n_iter,
        converged=converged,
        kkt_residual=float(kkt),
        grad_norm=float(grad_norm),
        algebraic_connectivity=lam2,
        meta={
            "alpha": float(alpha),
            "beta": float(beta),
            "max_lag": float(max_lag),
            "fs": float(fs),
            "n_freqs": float(cs.freqs.shape[0]),
            "spectral_gap": float(init_res.spectral_gap) if init_res is not None else 0.0,
        },
    )
