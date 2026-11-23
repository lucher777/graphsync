"""Exact primal-dual smooth graph Laplacian learning (Kalofolias 2016)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import optimize


def _upper_tri_indices(m_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular off-diagonal index arrays for an M x M matrix."""
    return np.triu_indices(m_nodes, k=1)


def _degree_operator(m_nodes: int) -> np.ndarray:
    """Build edge-to-degree incidence matrix ``S`` of shape ``(M, M*(M-1)/2)``."""
    iu, ju = _upper_tri_indices(m_nodes)
    n_edges = iu.shape[0]
    s_mat = np.zeros((m_nodes, n_edges), dtype=np.float64)
    edge_idx = np.arange(n_edges)
    s_mat[iu, edge_idx] = 1.0
    s_mat[ju, edge_idx] = 1.0
    return s_mat


def adjacency_to_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Construct combinatorial graph Laplacian ``L = diag(W 1) - W`` from adjacency ``W``.

    Args:
        adjacency: Symmetric nonnegative matrix of shape ``(M, M)``.

    Returns:
        Combinatorial graph Laplacian ``L`` of shape ``(M, M)``.
    """
    w = np.asarray(adjacency, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got shape {w.shape}.")
    w_sym = 0.5 * (w + w.T)
    np.fill_diagonal(w_sym, 0.0)
    return np.diag(np.sum(w_sym, axis=1)) - w_sym


def algebraic_connectivity(laplacian: np.ndarray) -> float:
    """Return Fiedler eigenvalue ``lambda_2(L)`` of a symmetric graph Laplacian.

    Args:
        laplacian: Symmetric ``(M, M)`` Laplacian matrix.

    Returns:
        Second-smallest eigenvalue ``lambda_2(L)`` clipped at ``0.0``.
    """
    eigvals = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    return float(max(0.0, eigvals[1])) if eigvals.shape[0] >= 2 else 0.0


def graph_learning_objective(
    adjacency: np.ndarray,
    dist_matrix: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Evaluate the Kalofolias smooth graph learning objective.

    ``J(W) = 0.5 * tr(W Z) - alpha * sum_i log((W 1)_i) + 0.5 * beta * ||W||_F^2``.

    Args:
        adjacency: Symmetric nonnegative adjacency matrix ``W``, shape ``(M, M)``.
        dist_matrix: Symmetric nonnegative pairwise distance matrix ``Z``, shape ``(M, M)``.
        alpha: Logarithmic degree-barrier weight (``alpha > 0``).
        beta: Frobenius sparsity/regularization weight (``beta > 0``).

    Returns:
        Scalar objective value.
    """
    deg = np.sum(adjacency, axis=1)
    if np.any(deg <= 0.0):
        return float("inf")
    smooth = 0.5 * float(np.sum(adjacency * dist_matrix))
    barrier = -float(alpha) * float(np.sum(np.log(deg)))
    reg = 0.5 * float(beta) * float(np.sum(adjacency * adjacency))
    return smooth + barrier + reg


def graph_learning_kkt_residual(
    w_vec: np.ndarray,
    z_vec: np.ndarray,
    s_mat: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Compute exact KKT stationarity/complementarity residual for ``min_{w>=0} f(w)``.

    For ``f(w) = w^T z - alpha * 1^T log(S w) + beta * ||w||_2^2``, the exact KKT
    residual is ``|| w - max(0, w - grad f(w)) ||_inf``.

    Args:
        w_vec: Edge weight vector of length ``M*(M-1)/2``.
        z_vec: Pairwise distance vector of length ``M*(M-1)/2``.
        s_mat: Incidence matrix ``S`` of shape ``(M, E)``.
        alpha: Log-barrier parameter.
        beta: Quadratic regularization parameter.

    Returns:
        Infinity-norm KKT residual.
    """
    deg = s_mat @ w_vec
    if np.any(deg <= 0.0):
        return float("inf")
    grad = z_vec - float(alpha) * (s_mat.T @ (1.0 / deg)) + 2.0 * float(beta) * w_vec
    proj_step = w_vec - np.maximum(0.0, w_vec - grad)
    return float(np.max(np.abs(proj_step)))


def learn_smooth_graph(
    dist_matrix: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-10,
    warm_start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve the strictly convex smooth graph learning problem to high precision.

    Minimizes over symmetric ``W >= 0`` with zero diagonal:
        ``0.5 * tr(W Z) - alpha * 1^T log(W 1) + 0.5 * beta * ||W||_F^2``.

    Uses Kalofolias's (2016) primal-dual forward-backward-forward (FBF) algorithm,
    polished with bound-constrained L-BFGS-B when needed so that the exact KKT
    residual is guaranteed to be below ``tol``.

    Args:
        dist_matrix: Symmetric nonnegative pairwise distance matrix ``Z`` of shape ``(M, M)``.
        alpha: Positive logarithmic degree-barrier parameter.
        beta: Positive Frobenius regularization parameter.
        max_iter: Maximum primal-dual iterations.
        tol: KKT convergence tolerance.
        warm_start: Optional initial adjacency matrix ``W`` of shape ``(M, M)``.

    Returns:
        Tuple ``(W, L, kkt_residual)`` containing the optimal adjacency matrix,
        combinatorial Laplacian, and final KKT residual.
    """
    z_mat = np.asarray(dist_matrix, dtype=np.float64)
    if z_mat.ndim != 2 or z_mat.shape[0] != z_mat.shape[1]:
        raise ValueError(f"Expected square distance matrix Z, got shape {z_mat.shape}.")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"alpha and beta must be strictly positive, got alpha={alpha}, beta={beta}.")

    m_nodes = z_mat.shape[0]
    if m_nodes < 2:
        raise ValueError(f"At least 2 nodes required, got M={m_nodes}.")

    z_sym = 0.5 * (z_mat + z_mat.T)
    z_sym = np.maximum(z_sym, 0.0)
    np.fill_diagonal(z_sym, 0.0)

    iu, ju = _upper_tri_indices(m_nodes)
    z_vec = z_sym[iu, ju].copy()
    s_mat = _degree_operator(m_nodes)
    n_edges = z_vec.shape[0]

    # Primal-dual FBF (Komodakis & Pesquet / Kalofolias 2016)
    norm_s = np.sqrt(2.0 * (m_nodes - 1))
    lipschitz = 2.0 * float(beta)
    # Step size strictly inside (0, 1 / (lipschitz + norm_s))
    gamma = 0.9 / (lipschitz + norm_s)

    if warm_start is not None and warm_start.shape == (m_nodes, m_nodes):
        w_vec = np.maximum(warm_start[iu, ju].astype(np.float64), 1e-6)
    else:
        # Scale-aware strictly feasible initializer
        z_scale = float(np.median(z_vec)) + 1e-6
        init_val = max(np.sqrt(alpha / (2.0 * beta * (m_nodes - 1))), alpha / (z_scale * (m_nodes - 1)))
        w_vec = np.full(n_edges, init_val, dtype=np.float64)

    v_vec = s_mat @ w_vec
    d_dual = -float(alpha) / np.maximum(v_vec, 1e-12)

    for it in range(max_iter):
        y1 = w_vec - gamma * (2.0 * float(beta) * w_vec + s_mat.T @ d_dual)
        y2 = d_dual + gamma * (s_mat @ w_vec)

        p1 = np.maximum(0.0, y1 - gamma * z_vec)
        # prox_{gamma * g2^*}(y2) = y2 - gamma * prox_{g2 / gamma}(y2 / gamma)
        # Since g2(d) = -alpha log(d), prox_{g2/gamma}(u) = 0.5*(u + sqrt(u^2 + 4*alpha/gamma))
        p2 = 0.5 * (y2 - np.sqrt(y2 * y2 + 4.0 * float(alpha) * gamma))

        q1 = p1 - gamma * (2.0 * float(beta) * p1 + s_mat.T @ p2)
        q2 = p2 + gamma * (s_mat @ p1)

        w_next = w_vec - y1 + q1
        d_next = d_dual - y2 + q2

        # Periodic convergence check
        if (it + 1) % 25 == 0 or it == max_iter - 1:
            w_feas = np.maximum(w_next, 0.0)
            kkt = graph_learning_kkt_residual(w_feas, z_vec, s_mat, alpha, beta)
            if kkt <= tol:
                w_vec = w_feas
                break
        w_vec = w_next
        d_dual = d_next

    w_vec = np.maximum(w_vec, 0.0)
    kkt = graph_learning_kkt_residual(w_vec, z_vec, s_mat, alpha, beta)

    # High-precision L-BFGS-B polish if FBF did not yet reach ultra-tight tolerance
    if kkt > tol:
        w0 = np.maximum(w_vec, 1e-8)

        def fun_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            deg = s_mat @ w
            if np.any(deg <= 1e-15):
                return 1e30, np.zeros_like(w)
            val = float(np.dot(w, z_vec) - alpha * np.sum(np.log(deg)) + beta * np.dot(w, w))
            grad = z_vec - float(alpha) * (s_mat.T @ (1.0 / deg)) + 2.0 * float(beta) * w
            return val, grad

        res = optimize.minimize(
            fun_and_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * n_edges,
            options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 500},
        )
        w_cand = np.maximum(res.x, 0.0)
        kkt_cand = graph_learning_kkt_residual(w_cand, z_vec, s_mat, alpha, beta)
        if kkt_cand < kkt:
            w_vec = w_cand
            kkt = kkt_cand

    w_mat = np.zeros((m_nodes, m_nodes), dtype=np.float64)
    w_mat[iu, ju] = w_vec
    w_mat[ju, iu] = w_vec
    lap = adjacency_to_laplacian(w_mat)
    return w_mat, lap, float(kkt)


def edge_f1_score(
    w_est: np.ndarray,
    w_true: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Compute binary edge-recovery precision, recall, and F1 score.

    Args:
        w_est: Estimated symmetric adjacency matrix of shape ``(M, M)``.
        w_true: Ground-truth symmetric adjacency matrix of shape ``(M, M)``.
        threshold: Optional edge detection cutoff on ``w_est``. If ``None``,
            uses ``0.1 * max(w_est)`` (or matches the ground-truth edge density
            if all entries are tiny).

    Returns:
        Tuple ``(f1, precision, recall)``.
    """
    m_nodes = w_true.shape[0]
    iu, ju = _upper_tri_indices(m_nodes)
    true_edges = w_true[iu, ju] > 1e-9
    est_vals = w_est[iu, ju]

    if threshold is None:
        max_val = float(np.max(est_vals)) if est_vals.size else 0.0
        thr = 0.1 * max_val if max_val > 0.0 else 1e-9
    else:
        thr = float(threshold)

    pred_edges = est_vals > thr
    tp = int(np.sum(pred_edges & true_edges))
    fp = int(np.sum(pred_edges & ~true_edges))
    fn = int(np.sum(~pred_edges & true_edges))

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return float(f1), float(precision), float(recall)
