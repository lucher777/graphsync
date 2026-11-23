"""Spectral operations, exact fractional delays, and angular spectral initialization."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg, optimize, signal

from .types import CrossSpectrum, SpectralInitResult


def validate_signals(signals: np.ndarray) -> np.ndarray:
    """Validate and coerce multi-sensor signals to a contiguous 2-D float64 array.

    Args:
        signals: Array of shape ``(M, N)`` where ``M >= 2`` is the number of
            sensors and ``N >= 8`` is the number of time samples.

    Returns:
        Contiguous ``float64`` array of shape ``(M, N)``.

    Raises:
        ValueError: If dimensionality, shape, or finiteness constraints fail.
    """
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D signals array (M, N), got shape {arr.shape}.")
    m_nodes, n_samples = arr.shape
    if m_nodes < 2:
        raise ValueError(f"At least 2 sensor nodes are required, got M={m_nodes}.")
    if n_samples < 8:
        raise ValueError(f"At least 8 time samples are required, got N={n_samples}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signals contain NaN or infinite values.")
    return np.ascontiguousarray(arr)


def fractional_shift(
    signals: np.ndarray,
    delays: np.ndarray,
    fs: float = 1.0,
    sign: float = -1.0,
) -> np.ndarray:
    """Apply exact circular Fourier-domain fractional time shifts to multi-sensor signals.

    With ``sign = -1.0`` (forward observation model), sensor ``m`` is mapped to
    ``x_m(t - tau_m)``, multiplying its spectrum by ``exp(-j * omega * tau_m)``.
    With ``sign = +1.0`` (inverse alignment), sensor ``m`` is mapped to
    ``y_m(t + tau_m)``, multiplying its spectrum by ``exp(+j * omega * tau_m)``.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        delays: Per-sensor delays of shape ``(M,)`` in seconds (or samples when ``fs=1.0``).
        fs: Sampling frequency in Hz.
        sign: Phase sign (``-1.0`` to delay, ``+1.0`` to advance/align).

    Returns:
        Shifted multi-sensor array of shape ``(M, N)``.
    """
    x = validate_signals(signals)
    tau = np.asarray(delays, dtype=np.float64)
    if tau.shape != (x.shape[0],):
        raise ValueError(f"Expected delays of shape ({x.shape[0]},), got {tau.shape}.")
    if fs <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive, got {fs}.")

    n_samples = x.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    omegas = 2.0 * np.pi * freqs
    phase = np.exp(1j * sign * np.outer(tau, omegas))
    if n_samples % 2 == 0:
        # Preserve exact real-valued Hermitian symmetry at the Nyquist bin.
        phase[:, -1] = 1.0
    spec = np.fft.rfft(x, axis=1)
    shifted = np.fft.irfft(spec * phase, n=n_samples, axis=1)
    return np.ascontiguousarray(shifted)


def apply_offsets(signals: np.ndarray, delays: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Inject asynchronous sensor clock offsets ``y_m(t) = x_m(t - tau_m)``.

    Args:
        signals: Synchronized signal array of shape ``(M, N)``.
        delays: Clock offsets ``tau`` of shape ``(M,)``.
        fs: Sampling frequency.

    Returns:
        Asynchronous signal array of shape ``(M, N)``.
    """
    return fractional_shift(signals, delays, fs=fs, sign=-1.0)


def align_signals(signals: np.ndarray, delays: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Compensate estimated clock offsets ``y_{tau,m}(t) = y_m(t + tau_m)``.

    Args:
        signals: Asynchronous signal array of shape ``(M, N)``.
        delays: Estimated clock offsets ``tau`` of shape ``(M,)``.
        fs: Sampling frequency.

    Returns:
        Time-aligned signal array of shape ``(M, N)``.
    """
    return fractional_shift(signals, delays, fs=fs, sign=+1.0)


def compute_cross_spectrum(
    signals: np.ndarray,
    fs: float = 1.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    overlap: float = 0.5,
    demean: bool = True,
    whitening: str = "none",
    whitening_reg: float = 1e-6,
) -> CrossSpectrum:
    """Compute one-sided Hermitian cross-spectral density matrices.

    When ``nperseg is None`` and ``whitening == "none"``, computes the exact
    full-length DFT cross-spectrum such that Parseval's identity holds to machine
    precision (``< 1e-13``) against the time-domain Dirichlet energy. When
    ``nperseg`` is specified, computes a Welch block-averaged Hann-windowed
    cross-spectrum suitable for long, noisy multi-sensor streams.

    Optional Whittle spectral whitening preserves exact Hermitian positive
    semidefiniteness while equalizing strongly colored or heterogeneous multi-modal
    spectra (such as physiological ECG/PPG/RESP networks):

    - ``"none"``: Raw Parseval-exact CSD ``S_0(omega_k)``.
    - ``"trace"``: Divides each bin by its mean PSD ``(1/M) tr(S_0(omega_k))``
      (exact Whittle MLE ``1 / p(omega_k)`` weighting).
    - ``"coherency"``: Normalizes each bin to the Hermitian coherency matrix
      ``D_S(omega_k)^{-1/2} S_0(omega_k) D_S(omega_k)^{-1/2}``.

    Args:
        signals: Multi-sensor array of shape ``(M, N)``.
        fs: Sampling rate in Hz.
        fmin: Optional lower frequency cutoff (inclusive, in Hz).
        fmax: Optional upper frequency cutoff (inclusive, in Hz).
        nperseg: Optional Welch segment length in samples. ``None`` uses exact DFT.
        overlap: Fractional segment overlap in ``[0, 1)`` for Welch estimation.
        demean: Whether to subtract per-channel mean prior to spectral estimation.
        whitening: Spectral whitening mode in ``{"none", "trace", "coherency"}``.
        whitening_reg: Relative diagonal regularization floor for whitening.

    Returns:
        :class:`CrossSpectrum` containing frequencies, Parseval weights, CSD tensor,
        and per-node energies.
    """
    x = validate_signals(signals)
    if fs <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive, got {fs}.")
    if whitening not in ("none", "trace", "coherency"):
        raise ValueError(f"Unknown whitening mode '{whitening}'.")
    if demean:
        x = x - np.mean(x, axis=1, keepdims=True)
    m_nodes, n_samples = x.shape

    if nperseg is None:
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        spec = np.fft.rfft(x, axis=1) / float(n_samples)  # (M, K)
        # S[k, i, j] = X_i[k] * conj(X_j[k]) / N^2 -> Parseval weight w_k gives 1/N sum_n x_i x_j
        csd = np.einsum("ik,jk->kij", spec, np.conj(spec))
        weights = np.full(freqs.shape, 2.0, dtype=np.float64)
        weights[0] = 1.0
        if n_samples % 2 == 0:
            weights[-1] = 1.0
        n_blocks = 1
        seg_len = n_samples
    else:
        seg_len = int(nperseg)
        if seg_len < 8 or seg_len > n_samples:
            raise ValueError(f"nperseg must satisfy 8 <= nperseg <= N ({n_samples}), got {seg_len}.")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
        step = max(1, int(round(seg_len * (1.0 - overlap))))
        win = signal.windows.hann(seg_len, sym=False)
        win_norm = float(np.sum(win ** 2))
        starts = list(range(0, n_samples - seg_len + 1, step))
        n_blocks = len(starts)
        freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
        k_bins = freqs.shape[0]
        csd = np.zeros((k_bins, m_nodes, m_nodes), dtype=np.complex128)
        for st in starts:
            seg = x[:, st : st + seg_len]
            if demean:
                seg = seg - np.mean(seg, axis=1, keepdims=True)
            seg_w = seg * win[None, :]
            spec = np.fft.rfft(seg_w, axis=1)
            csd += np.einsum("ik,jk->kij", spec, np.conj(spec))
        csd /= float(n_blocks * seg_len * win_norm)
        weights = np.full(freqs.shape, 2.0, dtype=np.float64)
        weights[0] = 1.0
        if seg_len % 2 == 0:
            weights[-1] = 1.0

    mask = np.ones(freqs.shape, dtype=bool)
    if fmin is not None:
        mask &= freqs >= float(fmin)
    if fmax is not None:
        mask &= freqs <= float(fmax)
    if not np.any(mask):
        raise ValueError(f"Frequency band [{fmin}, {fmax}] selected zero FFT bins.")

    freqs_sel = np.ascontiguousarray(freqs[mask])
    weights_sel = np.ascontiguousarray(weights[mask])
    csd_sel = np.ascontiguousarray(csd[mask])
    # Enforce exact Hermitian symmetry numerically
    csd_sel = 0.5 * (csd_sel + np.conj(np.swapaxes(csd_sel, 1, 2)))

    if whitening == "trace":
        traces = np.real(np.einsum("kii->k", csd_sel)) / float(m_nodes)
        floor = float(whitening_reg) * (float(np.median(traces)) + 1e-30) + 1e-30
        csd_sel = csd_sel / (traces[:, None, None] + floor)
        # Scale so average energy per node is 1.0
        total_w = float(np.sum(weights_sel))
        if total_w > 0.0:
            csd_sel = csd_sel / total_w
    elif whitening == "coherency":
        diag_psd = np.maximum(np.real(np.diagonal(csd_sel, axis1=1, axis2=2)), 0.0)  # (K, M)
        med_psd = np.median(diag_psd, axis=0, keepdims=True) + 1e-30
        inv_sqrt = 1.0 / np.sqrt(diag_psd + float(whitening_reg) * med_psd + 1e-30)
        csd_sel = csd_sel * (inv_sqrt[:, :, None] * inv_sqrt[:, None, :])
        csd_sel = 0.5 * (csd_sel + np.conj(np.swapaxes(csd_sel, 1, 2)))
        total_w = float(np.sum(weights_sel))
        if total_w > 0.0:
            csd_sel = csd_sel / total_w

    omegas_sel = 2.0 * np.pi * freqs_sel
    energies = np.real(np.einsum("k,kii->i", weights_sel, csd_sel))
    energies = np.maximum(energies, 0.0)

    return CrossSpectrum(
        freqs=freqs_sel,
        omegas=omegas_sel,
        weights=weights_sel,
        csd=csd_sel,
        energies=energies,
        fs=float(fs),
        n_samples=seg_len,
        n_blocks=n_blocks,
    )


def evaluate_cross_correlation_derivatives(
    cs: CrossSpectrum,
    delays: np.ndarray,
    polarities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact cross-correlation matrix ``R`` and its 1st and 2nd lag derivatives.

    For every pair ``(i, j)`` at lag ``Delta_ij = tau_i - tau_j``:
        ``R[i, j]   = sum_k w_k Re(S[k, i, j] * exp(j * omega_k * Delta_ij))``
        ``R1[i, j]  = dR/dDelta = sum_k w_k Re(j * omega_k * S[k, i, j] * exp(j * omega_k * Delta_ij))``
        ``R2[i, j]  = d^2R/dDelta^2 = -sum_k w_k * omega_k^2 * Re(S[k, i, j] * exp(j * omega_k * Delta_ij))``

    Args:
        cs: Precomputed :class:`CrossSpectrum`.
        delays: Current delay vector ``tau`` of shape ``(M,)``.
        polarities: Optional sign vector in ``{-1, +1}^M``.

    Returns:
        Tuple ``(R, R1, R2)`` of real ``(M, M)`` matrices.
    """
    tau = np.asarray(delays, dtype=np.float64)
    m_nodes = cs.csd.shape[1]
    if tau.shape != (m_nodes,):
        raise ValueError(f"Expected delays of shape ({m_nodes},), got {tau.shape}.")

    # Phase matrix D_k[m] = exp(j * omega_k * tau_m), shape (K, M)
    d_mat = np.exp(1j * np.outer(cs.omegas, tau))
    # Shifted CSD: S_tau[k, i, j] = D_k[i] * S[k, i, j] * conj(D_k[j])
    s_tau = cs.csd * (d_mat[:, :, None] * np.conj(d_mat[:, None, :]))

    if polarities is not None:
        pol = np.asarray(polarities, dtype=np.float64)
        s_tau = s_tau * np.outer(pol, pol)[None, :, :]

    w = cs.weights
    om = cs.omegas
    r0 = np.einsum("k,kij->ij", w, np.real(s_tau))
    # Re(j * z) = -Im(z)
    r1 = np.einsum("k,kij->ij", -w * om, np.imag(s_tau))
    r2 = np.einsum("k,kij->ij", -w * (om ** 2), np.real(s_tau))
    return r0, r1, r2


def pairwise_distance_matrix(
    cs: CrossSpectrum,
    delays: np.ndarray,
    polarities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute exact frequency-domain pairwise squared-distance matrix ``Z(tau)``.

    By Parseval's theorem, ``Z[i, j] = E_i + E_j - 2 * s_i * s_j * R_{ij}(tau_i - tau_j)``
    equals ``(1/N) || s_i y_{tau,i} - s_j y_{tau,j} ||_2^2``.

    Args:
        cs: Precomputed :class:`CrossSpectrum`.
        delays: Delay vector ``tau`` of shape ``(M,)``.
        polarities: Optional node polarity signs in ``{-1, +1}^M``.

    Returns:
        Symmetric nonnegative matrix ``Z`` of shape ``(M, M)`` with zero diagonal.
    """
    r0, _, _ = evaluate_cross_correlation_derivatives(cs, delays, polarities=polarities)
    z_mat = cs.energies[:, None] + cs.energies[None, :] - 2.0 * r0
    z_mat = 0.5 * (z_mat + z_mat.T)
    np.fill_diagonal(z_mat, 0.0)
    return np.maximum(z_mat, 0.0)


def _pairwise_lag_curve(
    cs: CrossSpectrum,
    i: int,
    j: int,
    lag_grid: np.ndarray,
    use_phat: bool = False,
) -> np.ndarray:
    """Evaluate normalized broadband cross-correlation curve on a 1-D lag grid."""
    s_ij = cs.csd[:, i, j]
    if use_phat:
        mag = np.abs(s_ij)
        # Coherence-regularized PHAT weighting avoids dividing by pure noise bins
        denom_ij = np.sqrt(np.maximum(np.real(cs.csd[:, i, i]) * np.real(cs.csd[:, j, j]), 1e-30))
        coh = np.clip(mag / denom_ij, 0.0, 1.0)
        # Weight by coherence^2 / (|S_ij| + eps) so high-coherence bins dominate cleanly
        spec = s_ij * (coh ** 2) / (mag + 1e-12 * np.max(mag) + 1e-30)
        norm = float(np.sum(cs.weights * (coh ** 2))) + 1e-30
    else:
        spec = s_ij
        norm = float(np.sqrt(max(cs.energies[i] * cs.energies[j], 1e-30)))

    # Vectorized evaluation: Re(sum_k w_k * spec_k * exp(j * omega_k * lag))
    # Shape: (K, L)
    weighted = cs.weights * spec
    exp_mat = np.exp(1j * np.outer(cs.omegas, lag_grid))
    curve = np.real(weighted @ exp_mat) / norm
    return curve


def spectral_angular_init(
    cs: CrossSpectrum,
    max_lag: float,
    coherence_power: float = 2.0,
    threshold: float = 0.0,
    allow_polarity_flip: bool = False,
    use_phat: bool = False,
    grid_oversample: int = 8,
) -> SpectralInitResult:
    """Closed-form broadband weighted Hermitian angular spectral initializer (Theorem 2).

    Estimates cycle-consistent global clock offsets ``tau^(0)`` without requiring
    prior knowledge of the graph topology ``L``. For each node pair ``(i, j)``,
    estimates the continuous sub-sample lag ``Delta_ij`` and broadband coherence
    ``c_ij``, builds the normalized Hermitian phase-synchronization operator
    ``D_w^{-1/2} H D_w^{-1/2}`` at the non-wrapping reference frequency
    ``omega_ref = pi / (2.2 * max_lag)``, and extracts the principal eigenvector
    in closed form.

    Args:
        cs: Precomputed :class:`CrossSpectrum`.
        max_lag: Maximum plausible absolute clock offset per node (in seconds or samples
            matching ``cs.fs``). Pairwise lags are searched over ``[-2 * max_lag, 2 * max_lag]``.
        coherence_power: Exponent ``p >= 1`` emphasizing strongly coherent edges.
        threshold: Optional coherence soft-threshold ``eta in [0, 1)``.
        allow_polarity_flip: If ``True``, searches ``|r_ij(Delta)|`` and recovers
            cycle-consistent node polarities via ``Z_2`` spectral synchronization.
        use_phat: If ``True``, uses coherence-regularized PHAT weighting for pairwise
            lag estimation (useful for heterogeneous multi-modal sensor spectra).
        grid_oversample: Grid density multiplier relative to ``cs.fs``.

    Returns:
        :class:`SpectralInitResult` containing closed-form delays, polarities,
        coherence weights, and normalized spectral gap.
    """
    if max_lag <= 0.0:
        raise ValueError(f"max_lag must be positive, got {max_lag}.")

    m_nodes = cs.csd.shape[1]
    search_bound = 2.0 * float(max_lag)
    # Construct fine lag search grid
    n_half = max(32, int(np.ceil(search_bound * cs.fs * grid_oversample)))
    lag_grid = np.linspace(-search_bound, search_bound, 2 * n_half + 1)
    step = float(lag_grid[1] - lag_grid[0])

    pairwise_lags = np.zeros((m_nodes, m_nodes), dtype=np.float64)
    coherence = np.eye(m_nodes, dtype=np.float64)
    sign_mat = np.ones((m_nodes, m_nodes), dtype=np.float64)

    for i in range(m_nodes):
        for j in range(i + 1, m_nodes):
            curve = _pairwise_lag_curve(cs, i, j, lag_grid, use_phat=use_phat)
            if allow_polarity_flip:
                idx = int(np.argmax(np.abs(curve)))
                sgn = 1.0 if curve[idx] >= 0.0 else -1.0
            else:
                idx = int(np.argmax(curve))
                sgn = 1.0

            # Continuous sub-sample Brent refinement around coarse grid peak
            left = max(-search_bound, lag_grid[idx] - step)
            right = min(search_bound, lag_grid[idx] + step)

            def neg_obj(delta: float, _i: int = i, _j: int = j, _sgn: float = sgn) -> float:
                val = _pairwise_lag_curve(cs, _i, _j, np.array([delta]), use_phat=use_phat)[0]
                return -float(_sgn * val)

            res = optimize.minimize_scalar(
                neg_obj,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-11},
            )
            best_lag = float(res.x)
            best_coh = float(max(0.0, -res.fun))

            pairwise_lags[i, j] = best_lag
            pairwise_lags[j, i] = -best_lag
            coherence[i, j] = best_coh
            coherence[j, i] = best_coh
            sign_mat[i, j] = sgn
            sign_mat[j, i] = sgn

    # Construct nonnegative cycle-consistent weights with connectivity safeguard
    eta = float(threshold)
    raw_coh = coherence.copy()
    np.fill_diagonal(raw_coh, 0.0)
    weights = np.maximum(0.0, raw_coh - eta) ** float(coherence_power)
    deg = np.sum(weights, axis=1)
    if np.any(deg <= 1e-12):
        # Adaptively fall back to unthresholded weights so every node is connected
        weights = np.maximum(raw_coh, 1e-6) ** float(coherence_power)
        np.fill_diagonal(weights, 0.0)
        deg = np.sum(weights, axis=1)

    # Non-wrapping reference angular frequency: |omega_ref * 2 * max_lag| < pi
    omega_ref = np.pi / (2.2 * float(max_lag))

    # Build Hermitian phase-synchronization matrix H[i, j] = w_ij * exp(-j * omega_ref * Delta_ij)
    h_mat = weights * np.exp(-1j * omega_ref * pairwise_lags)
    h_mat = 0.5 * (h_mat + np.conj(h_mat.T))

    # Normalized Hermitian operator D^{-1/2} H D^{-1/2}
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(deg, 1e-15))
    h_norm = (inv_sqrt_d[:, None] * h_mat) * inv_sqrt_d[None, :]
    h_norm = 0.5 * (h_norm + np.conj(h_norm.T))

    eigvals, eigvecs = linalg.eigh(h_norm)
    top_vec = eigvecs[:, -1]
    u_vec = inv_sqrt_d * top_vec
    spectral_gap = float(eigvals[-1] - eigvals[-2]) if m_nodes >= 2 else 0.0

    # Normalize to unit circle and unwrap relative to anchor node 0
    z_hat = u_vec / np.maximum(np.abs(u_vec), 1e-15)
    z_rel = z_hat * np.conj(z_hat[0])
    delays = -np.angle(z_rel) / omega_ref
    delays[0] = 0.0
    delays = np.clip(delays, -float(max_lag), float(max_lag))

    # Optional Z_2 spectral synchronization for polarity flips
    if allow_polarity_flip:
        m_signed = (inv_sqrt_d[:, None] * (weights * sign_mat)) * inv_sqrt_d[None, :]
        m_signed = 0.5 * (m_signed + m_signed.T)
        _, s_vecs = linalg.eigh(m_signed)
        s_top = inv_sqrt_d * s_vecs[:, -1]
        polarities = np.where(s_top >= 0.0, 1.0, -1.0)
        if polarities[0] < 0.0:
            polarities = -polarities
    else:
        polarities = np.ones(m_nodes, dtype=np.float64)

    return SpectralInitResult(
        delays=delays,
        polarities=polarities,
        pairwise_lags=pairwise_lags,
        coherence=coherence,
        weights=weights,
        spectral_gap=spectral_gap,
        omega_ref=float(omega_ref),
    )
