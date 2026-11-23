"""Classical synchronization and unsynchronized graph learning baselines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import optimize

from .spectral import _pairwise_lag_curve, compute_cross_spectrum, pairwise_distance_matrix
from .topology import learn_smooth_graph


def _estimate_pair_lag(
    signals: np.ndarray,
    i: int,
    j: int,
    fs: float,
    max_lag: float,
    use_phat: bool,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    allow_polarity_flip: bool = False,
    cs=None,
) -> float:
    """Estimate sub-sample lag Delta_ij = tau_i - tau_j for a single sensor pair."""
    if cs is None:
        cs = compute_cross_spectrum(signals, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg)
    search_bound = float(max_lag)
    n_half = max(32, int(np.ceil(search_bound * fs * 8)))
    lag_grid = np.linspace(-search_bound, search_bound, 2 * n_half + 1)
    step = float(lag_grid[1] - lag_grid[0])

    if use_phat:
        # Classical Knapp & Carter (1976) GCC-PHAT: S_ij / (|S_ij| + eps)
        s_ij = cs.csd[:, i, j]
        mag = np.abs(s_ij)
        phat_spec = s_ij / (mag + 1e-12 * np.max(mag) + 1e-30)
        weighted = cs.weights * phat_spec
        curve = np.real(weighted @ np.exp(1j * np.outer(cs.omegas, lag_grid)))
    else:
        curve = _pairwise_lag_curve(cs, i, j, lag_grid, use_phat=False)

    if allow_polarity_flip:
        idx = int(np.argmax(np.abs(curve)))
        sgn = 1.0 if curve[idx] >= 0.0 else -1.0
    else:
        idx = int(np.argmax(curve))
        sgn = 1.0

    left = max(-search_bound, lag_grid[idx] - step)
    right = min(search_bound, lag_grid[idx] + step)

    if use_phat:
        def neg_fn(delta: float) -> float:
            val = np.real(np.sum(weighted * np.exp(1j * cs.omegas * delta)))
            return -float(sgn * val)
    else:
        def neg_fn(delta: float) -> float:
            val = _pairwise_lag_curve(cs, i, j, np.array([delta]), use_phat=False)[0]
            return -float(sgn * val)

    res = optimize.minimize_scalar(neg_fn, bounds=(left, right), method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def gcc_phat_anchor_sync(
    signals: np.ndarray,
    fs: float = 1.0,
    max_lag: float = 2.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    allow_polarity_flip: bool = False,
) -> np.ndarray:
    """Pairwise GCC-PHAT (Knapp & Carter 1976) anchored to sensor 0.

    Estimates ``tau_m`` independently for each sensor ``m = 1, ..., M - 1`` by
    maximizing the PHAT-whitened generalized cross-correlation with anchor node 0.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        fs: Sampling frequency in Hz.
        max_lag: Maximum absolute clock offset.
        fmin: Optional lower frequency cutoff.
        fmax: Optional upper frequency cutoff.
        nperseg: Optional Welch segment length.
        allow_polarity_flip: Whether to take peak magnitude ``|GCC-PHAT|``.

    Returns:
        Estimated offset vector ``tau`` of shape ``(M,)`` with ``tau[0] == 0.0``.
    """
    cs = compute_cross_spectrum(signals, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg)
    m_nodes = cs.csd.shape[1]
    delays = np.zeros(m_nodes, dtype=np.float64)
    for m in range(1, m_nodes):
        # Delta_{m, 0} = tau_m - tau_0 = tau_m
        delays[m] = _estimate_pair_lag(
            signals,
            m,
            0,
            fs=fs,
            max_lag=max_lag,
            use_phat=True,
            allow_polarity_flip=allow_polarity_flip,
            cs=cs,
        )
    return delays


def xcorr_anchor_sync(
    signals: np.ndarray,
    fs: float = 1.0,
    max_lag: float = 2.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    allow_polarity_flip: bool = False,
) -> np.ndarray:
    """Pairwise maximum cross-correlation anchored to sensor 0.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        fs: Sampling frequency in Hz.
        max_lag: Maximum absolute clock offset.
        fmin: Optional lower frequency cutoff.
        fmax: Optional upper frequency cutoff.
        nperseg: Optional Welch segment length.
        allow_polarity_flip: Whether to take peak magnitude ``|XCorr|``.

    Returns:
        Estimated offset vector ``tau`` of shape ``(M,)`` with ``tau[0] == 0.0``.
    """
    cs = compute_cross_spectrum(signals, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg)
    m_nodes = cs.csd.shape[1]
    delays = np.zeros(m_nodes, dtype=np.float64)
    for m in range(1, m_nodes):
        delays[m] = _estimate_pair_lag(
            signals,
            m,
            0,
            fs=fs,
            max_lag=max_lag,
            use_phat=False,
            allow_polarity_flip=allow_polarity_flip,
            cs=cs,
        )
    return delays


def unweighted_cycle_sync(
    signals: np.ndarray,
    fs: float = 1.0,
    max_lag: float = 2.0,
    use_phat: bool = False,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    allow_polarity_flip: bool = False,
) -> np.ndarray:
    """Unweighted least-squares cycle synchronization over all sensor pairs.

    Computes all ``M*(M-1)/2`` pairwise lags ``Delta_ij`` and projects onto the
    cycle-consistent subspace via the unweighted complete-graph average
    ``tau_i = (1/M) sum_j Delta_ij`` (anchored to ``tau_0 = 0``), without
    inferring or weighting by the underlying graph topology.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        fs: Sampling frequency in Hz.
        max_lag: Maximum absolute clock offset per node.
        use_phat: Whether to use GCC-PHAT (``True``) or XCorr (``False``) pairwise lags.
        fmin: Optional lower frequency cutoff.
        fmax: Optional upper frequency cutoff.
        nperseg: Optional Welch segment length.
        allow_polarity_flip: Whether to take peak magnitude.

    Returns:
        Cycle-consistent delay vector ``tau`` of shape ``(M,)`` with ``tau[0] == 0.0``.
    """
    cs = compute_cross_spectrum(signals, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg)
    m_nodes = cs.csd.shape[1]
    delta_mat = np.zeros((m_nodes, m_nodes), dtype=np.float64)
    for i in range(m_nodes):
        for j in range(i + 1, m_nodes):
            lag = _estimate_pair_lag(
                signals,
                i,
                j,
                fs=fs,
                max_lag=2.0 * max_lag,
                use_phat=use_phat,
                allow_polarity_flip=allow_polarity_flip,
                cs=cs,
            )
            delta_mat[i, j] = lag
            delta_mat[j, i] = -lag

    # Exact unweighted least-squares solution to min_tau sum_{i<j} (tau_i - tau_j - Delta_ij)^2
    tau = np.mean(delta_mat, axis=1)
    tau = tau - tau[0]
    return np.clip(tau, -max_lag, max_lag)


def unsynced_kalofolias_graph(
    signals: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.08,
    fs: float = 1.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    whitening: str = "none",
    scale_energy: bool = True,
    normalize_signals: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run exact Kalofolias (2016) smooth graph learning without time synchronization.

    Serves as the topology-inference ablation baseline illustrating performance
    degradation when sensor clock offsets are ignored.

    Args:
        signals: Asynchronous multi-sensor array of shape ``(M, N)``.
        alpha: Logarithmic degree-barrier parameter.
        beta: Frobenius regularization parameter.
        fs: Sampling frequency in Hz.
        fmin: Optional lower frequency cutoff.
        fmax: Optional upper frequency cutoff.
        nperseg: Optional Welch segment length.
        whitening: Spectral whitening mode matching ``fit_graphsync``.
        scale_energy: Whether to apply constant mean-energy scaling matching ``fit_graphsync``.
        normalize_signals: If ``True``, z-scores each channel first.

    Returns:
        Tuple ``(W, L, kkt_residual)``.
    """
    arr = np.asarray(signals, dtype=np.float64)
    if normalize_signals:
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        stds = np.std(arr, axis=1, keepdims=True)
        arr = arr / np.where(stds > 1e-12, stds, 1.0)
    cs = compute_cross_spectrum(
        arr, fs=fs, fmin=fmin, fmax=fmax, nperseg=nperseg, whitening=whitening
    )
    zeros = np.zeros(cs.csd.shape[1], dtype=np.float64)
    z_mat = pairwise_distance_matrix(cs, zeros)
    if scale_energy:
        mean_e = float(np.mean(cs.energies))
        if mean_e > 1e-15:
            z_mat = z_mat / mean_e
    return learn_smooth_graph(z_mat, alpha=alpha, beta=beta)
