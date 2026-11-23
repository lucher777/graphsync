"""Tests for spectral operations, fractional shifts, and Theorem 2 angular initialization."""

from __future__ import annotations

import numpy as np
import pytest

import graphsync as gs


def test_fractional_shift_invertibility_bandlimited() -> None:
    """Applying and aligning fractional offsets recovers bandlimited signals to < 1e-13."""
    rng = np.random.default_rng(1)
    m_nodes, n_samples = 6, 512
    spec = np.zeros((m_nodes, n_samples // 2 + 1), dtype=np.complex128)
    spec[:, 5:120] = rng.standard_normal((m_nodes, 115)) + 1j * rng.standard_normal((m_nodes, 115))
    x = np.fft.irfft(spec, n=n_samples, axis=1)
    taus = np.array([0.0, 1.37, -2.61, 0.84, -1.19, 3.05])

    y = gs.apply_offsets(x, taus, fs=1.0)
    x_rec = gs.align_signals(y, taus, fs=1.0)
    assert np.max(np.abs(x_rec - x)) < 1e-13


def test_fractional_shift_integer_matches_roll() -> None:
    """Integer delays match exact circular sample shifts."""
    rng = np.random.default_rng(2)
    n_samples = 128
    spec = np.zeros((3, n_samples // 2 + 1), dtype=np.complex128)
    spec[:, 1:40] = rng.standard_normal((3, 39)) + 1j * rng.standard_normal((3, 39))
    x = np.fft.irfft(spec, n=n_samples, axis=1)
    y = gs.apply_offsets(x, np.array([0.0, 3.0, -5.0]), fs=1.0)
    assert np.allclose(y[1], np.roll(x[1], 3), atol=1e-12)
    assert np.allclose(y[2], np.roll(x[2], -5), atol=1e-12)


def test_cross_spectrum_hermitian_psd_and_whitening() -> None:
    """Cross-spectral density matrices are Hermitian PSD across all whitening modes."""
    sample = gs.simulate_asynchronous_network(m_nodes=6, n_samples=512, snr_db=10.0, seed=3)
    for mode in ("none", "trace", "coherency"):
        cs = gs.compute_cross_spectrum(sample.observed, fs=1.0, nperseg=128, whitening=mode)
        assert cs.csd.shape[1:] == (6, 6)
        assert np.allclose(cs.csd, np.conj(np.swapaxes(cs.csd, 1, 2)), atol=1e-13)
        min_eig = min(float(np.min(np.linalg.eigvalsh(mat))) for mat in cs.csd)
        assert min_eig >= -1e-12
        assert np.all(cs.energies > 0.0)


def test_spectral_angular_init_exact_noiseless_recovery() -> None:
    """Theorem 2 closed-form angular spectral init recovers deterministic offsets to < 1e-6."""
    rng = np.random.default_rng(10)
    m_nodes, n_samples = 8, 512
    # Identical broadband latent signal observed with unknown offsets (noiseless exact regime)
    spec = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec[5:160] = rng.standard_normal(155) + 1j * rng.standard_normal(155)
    base = np.fft.irfft(spec, n=n_samples)
    clean = np.tile(base, (m_nodes, 1))
    tau_true = np.array([0.0, 2.34, -3.17, 1.82, -0.95, 4.11, -2.48, 0.63])
    obs = gs.apply_offsets(clean, tau_true, fs=1.0)

    cs = gs.compute_cross_spectrum(obs, fs=1.0, fmin=5 / n_samples, fmax=160 / n_samples)
    init_res = gs.spectral_angular_init(cs, max_lag=5.0)
    assert np.max(np.abs(init_res.delays - tau_true)) < 1e-6
    assert init_res.spectral_gap > 0.5


def test_spectral_angular_init_recovers_polarity_flips() -> None:
    """Optional Z_2 spectral sync recovers inverted channel polarities alongside delays."""
    rng = np.random.default_rng(11)
    m_nodes, n_samples = 6, 512
    spec = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
    spec[5:120] = rng.standard_normal(115) + 1j * rng.standard_normal(115)
    base = np.fft.irfft(spec, n=n_samples)
    true_pol = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    clean = true_pol[:, None] * base[None, :]
    tau_true = np.array([0.0, 1.25, -2.1, 0.75, -1.5, 2.4])
    obs = gs.apply_offsets(clean, tau_true, fs=1.0)

    cs = gs.compute_cross_spectrum(obs, fs=1.0, fmin=5 / n_samples, fmax=120 / n_samples)
    init_res = gs.spectral_angular_init(cs, max_lag=3.5, allow_polarity_flip=True)
    assert np.array_equal(init_res.polarities, true_pol)
    assert np.max(np.abs(init_res.delays - tau_true)) < 1e-6


def test_input_validation_errors() -> None:
    """Invalid signal shapes, NaNs, or bad parameters raise ValueError."""
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(np.ones((10,)))
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(np.ones((1, 64)))
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(np.ones((4, 4)))
    bad = np.ones((4, 64))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(bad)
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(np.ones((4, 64)), fs=-1.0)
    with pytest.raises(ValueError):
        gs.compute_cross_spectrum(np.ones((4, 64)), whitening="invalid")
