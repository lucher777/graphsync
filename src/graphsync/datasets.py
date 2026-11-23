"""Synthetic smooth graph network generators and bundled real physiological slice."""

from __future__ import annotations

from importlib import resources
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import linalg

from .spectral import apply_offsets
from .topology import adjacency_to_laplacian, algebraic_connectivity
from .types import SyntheticNetworkSample


def generate_connected_graph(
    m_nodes: int,
    graph_type: str = "er",
    edge_prob: float = 0.35,
    weight_range: Tuple[float, float] = (0.5, 1.5),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a connected undirected weighted graph and its Laplacian.

    Supports Erdős-Rényi (``"er"``), Barabási-Albert preferential attachment
    (``"ba"``), small-world ring with chords (``"ring"``), and stochastic block
    model (``"sbm"``) topologies.

    Args:
        m_nodes: Number of sensor nodes ``M >= 3``.
        graph_type: Topology family in ``{"er", "ba", "ring", "sbm"}``.
        edge_prob: Connection density parameter for ``"er"`` / ``"sbm"``.
        weight_range: Uniform interval ``(low, high)`` for positive edge weights.
        seed: Optional random seed.

    Returns:
        Tuple ``(W, L)`` of symmetric adjacency and combinatorial Laplacian matrices.
    """
    if m_nodes < 3:
        raise ValueError(f"m_nodes must be >= 3, got {m_nodes}.")
    rng = np.random.default_rng(seed)
    low, high = float(weight_range[0]), float(weight_range[1])

    for attempt in range(200):
        w_mat = np.zeros((m_nodes, m_nodes), dtype=np.float64)
        if graph_type == "er":
            for i in range(m_nodes):
                for j in range(i + 1, m_nodes):
                    if rng.random() < edge_prob:
                        val = float(rng.uniform(low, high))
                        w_mat[i, j] = w_mat[j, i] = val
        elif graph_type == "ba":
            # Preferential attachment starting from a 3-node clique
            m_attach = min(2, m_nodes - 1)
            for i in range(m_attach + 1):
                for j in range(i + 1, m_attach + 1):
                    val = float(rng.uniform(low, high))
                    w_mat[i, j] = w_mat[j, i] = val
            for new_node in range(m_attach + 1, m_nodes):
                deg = np.sum(w_mat[:new_node, :new_node] > 0.0, axis=1).astype(np.float64)
                probs = deg / np.sum(deg)
                targets = rng.choice(new_node, size=m_attach, replace=False, p=probs)
                for tgt in targets:
                    val = float(rng.uniform(low, high))
                    w_mat[new_node, int(tgt)] = w_mat[int(tgt), new_node] = val
        elif graph_type == "ring":
            for i in range(m_nodes):
                j = (i + 1) % m_nodes
                val = float(rng.uniform(low, high))
                w_mat[i, j] = w_mat[j, i] = val
            # Add a small number of deterministic-density random chords
            n_chords = max(1, m_nodes // 4)
            for _ in range(n_chords):
                i, j = rng.choice(m_nodes, size=2, replace=False)
                if w_mat[i, j] == 0.0:
                    val = float(rng.uniform(low, high))
                    w_mat[i, j] = w_mat[j, i] = val
        elif graph_type == "sbm":
            half = m_nodes // 2
            for i in range(m_nodes):
                for j in range(i + 1, m_nodes):
                    same_block = (i < half and j < half) or (i >= half and j >= half)
                    p_ij = min(0.85, edge_prob * 2.0) if same_block else max(0.08, edge_prob * 0.25)
                    if rng.random() < p_ij:
                        val = float(rng.uniform(low, high))
                        w_mat[i, j] = w_mat[j, i] = val
        else:
            raise ValueError(f"Unknown graph_type '{graph_type}'.")

        lap = adjacency_to_laplacian(w_mat)
        if algebraic_connectivity(lap) > 1e-4:
            return w_mat, lap

        # Ensure connectivity if random draw was disconnected on final retries
        if attempt >= 100:
            for i in range(m_nodes - 1):
                if w_mat[i, i + 1] == 0.0:
                    val = float(rng.uniform(low, high))
                    w_mat[i, i + 1] = w_mat[i + 1, i] = val
            lap = adjacency_to_laplacian(w_mat)
            if algebraic_connectivity(lap) > 1e-4:
                return w_mat, lap

    raise RuntimeError("Failed to generate a connected graph.")


def simulate_asynchronous_network(
    m_nodes: int = 8,
    n_samples: int = 512,
    fs: float = 1.0,
    snr_db: float = 10.0,
    max_lag: float = 1.5,
    graph_type: str = "er",
    edge_prob: float = 0.35,
    gamma: float = 0.15,
    k_min: int = 4,
    k_max: Optional[int] = None,
    seed: Optional[int] = None,
    adjacency: Optional[np.ndarray] = None,
    true_delays: Optional[np.ndarray] = None,
) -> SyntheticNetworkSample:
    """Synthesize a bandlimited smooth Gaussian graph signal with unknown clock offsets.

    Draws exact circularly symmetric complex Gaussian DFT coefficients
    ``X(omega_k) ~ CN(0, p(omega_k) * (L* + gamma I)^{-1})`` over positive active
    frequencies ``k in [k_min, k_max]``, transforms to real time-domain signals
    via inverse rFFT, injects continuous sub-sample delays ``tau*``, and adds
    white Gaussian sensor noise calibrated to ``snr_db`` in the active band.

    The spectral parameters ``(omegas, spectral_power, noise_var, gamma)`` returned
    in :class:`SyntheticNetworkSample` match :func:`compute_joint_crlb` identically
    (unitary DFT variance convention), enabling exact empirical-vs-CRLB validation.

    Args:
        m_nodes: Number of sensors ``M``.
        n_samples: Number of time samples ``N`` (even integer recommended).
        fs: Sampling frequency in Hz.
        snr_db: Signal-to-noise ratio in dB (``np.inf`` for noiseless regime).
        max_lag: Maximum absolute clock offset magnitude.
        graph_type: Graph family if ``adjacency`` is not provided.
        edge_prob: Graph edge probability for random graph families.
        gamma: Graph-Laplacian precision ridge ``gamma > 0``.
        k_min: Lowest active positive FFT bin index (>= 1).
        k_max: Highest active positive FFT bin index (defaults to ``N // 8``).
        seed: Random seed.
        adjacency: Optional user-supplied ground-truth adjacency ``W*``.
        true_delays: Optional user-supplied ground-truth offsets ``tau*``.

    Returns:
        :class:`SyntheticNetworkSample` with observed/clean signals and exact metadata.
    """
    rng = np.random.default_rng(seed)
    if adjacency is None:
        w_true, l_true = generate_connected_graph(
            m_nodes, graph_type=graph_type, edge_prob=edge_prob, seed=int(rng.integers(0, 2**31 - 1))
        )
    else:
        w_true = 0.5 * (np.asarray(adjacency, dtype=np.float64) + np.asarray(adjacency, dtype=np.float64).T)
        np.fill_diagonal(w_true, 0.0)
        l_true = adjacency_to_laplacian(w_true)
        m_nodes = w_true.shape[0]

    if true_delays is None:
        tau_true = rng.uniform(-max_lag, max_lag, size=m_nodes)
        tau_true[0] = 0.0
    else:
        tau_true = np.asarray(true_delays, dtype=np.float64).copy()
        tau_true = tau_true - tau_true[0]

    n_freq_total = n_samples // 2 + 1
    k_lo = max(1, int(k_min))
    k_hi = int(k_max) if k_max is not None else max(k_lo + 8, min(n_freq_total - 2, n_samples // 8))
    if k_hi >= n_freq_total - 1:
        k_hi = n_freq_total - 2

    freqs_all = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    active_bins = np.arange(k_lo, k_hi + 1)
    omegas_active = 2.0 * np.pi * freqs_all[active_bins]

    # Smooth broadband power envelope p(omega_k)
    norm_f = (active_bins - k_lo) / max(1.0, float(k_hi - k_lo))
    p_active = 1.0 + 0.5 * np.cos(np.pi * norm_f)

    # Exact spatial covariance (L* + gamma I)^{-1}
    cov_spatial = linalg.inv(l_true + float(gamma) * np.eye(m_nodes))
    chol_spatial = linalg.cholesky(cov_spatial, lower=True)

    # Unitary rFFT synthesis: E[|X_unitary(k)|^2] = p_k * cov_spatial
    # Recall np.fft.irfft(spec, n=N) scales by 1/N; unitary DFT is spec_rfft / sqrt(N).
    # For a real signal, if U_k ~ CN(0, C_k) at interior k>0, then x[n] = (2/sqrt(N)) Re(sum U_k e^{j w_k n})
    # wait: in numpy irfft, x[n] = (1/N) * (X_0 + 2 Re(sum_{k=1}^{N/2-1} X_k e^{j 2pi k n / N}) + X_{N/2}(-1)^n).
    # Setting X_k = sqrt(N / 2) * U_k makes 1/sqrt(N) * rfft(x)[k] = (1/sqrt(2)) * U_k? Let's be exact!
    # Check Parseval for a single real Gaussian white noise w[n] ~ N(0, sigma_t^2):
    # W_k = rfft(w)[k] has E[|W_k|^2] = N * sigma_t^2 for 1 <= k <= N/2 - 1 (real and imag each var N*sigma_t^2 / 2).
    # Therefore U_k = rfft(w)[k] / sqrt(N) ~ CN(0, sigma_t^2 * I) EXACTLY!
    spec_clean = np.zeros((m_nodes, n_freq_total), dtype=np.complex128)
    for idx_b, k_bin in enumerate(active_bins):
        # Standard CN(0, I) draw
        z_std = (rng.standard_normal(m_nodes) + 1j * rng.standard_normal(m_nodes)) / np.sqrt(2.0)
        u_k = np.sqrt(p_active[idx_b]) * (chol_spatial @ z_std)
        # rfft coefficient X_k = sqrt(N) * U_k
        spec_clean[:, k_bin] = np.sqrt(float(n_samples)) * u_k

    clean = np.fft.irfft(spec_clean, n=n_samples, axis=1)
    delayed = apply_offsets(clean, tau_true, fs=fs)

    # Active-band signal variance per node per unitary DFT bin:
    # mean_k (p_k) * mean(diag(cov_spatial))
    mean_sig_bin_var = float(np.mean(p_active) * np.mean(np.diag(cov_spatial)))
    if np.isinf(snr_db):
        noise_var = 0.0
        observed = delayed.copy()
    else:
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_var = mean_sig_bin_var / snr_lin
        # White Gaussian noise with time-domain variance = noise_var has
        # unitary DFT bin variance E[|rfft(eps)[k]/sqrt(N)|^2] == noise_var EXACTLY!
        noise = rng.normal(0.0, np.sqrt(noise_var), size=delayed.shape)
        observed = delayed + noise

    return SyntheticNetworkSample(
        observed=np.ascontiguousarray(observed),
        clean=np.ascontiguousarray(clean),
        true_delays=tau_true,
        true_adjacency=w_true,
        true_laplacian=l_true,
        fs=float(fs),
        snr_db=float(snr_db),
        noise_var=float(noise_var),
        gamma=float(gamma),
        omegas=omegas_active,
        spectral_power=p_active,
        graph_type=graph_type,
    )


def load_bundled_bidmc_slice() -> Dict[str, np.ndarray]:
    """Load the compact bundled real PhysioNet BIDMC 5-channel physiological slice.

    Contains hardware-synchronized multi-channel clinical monitor recordings
    (3 ECG leads + photoplethysmogram ``PLETH`` + respiratory impedance ``RESP``)
    sampled at 125 Hz from multiple ICU patients in the public BIDMC PPG and
    Respiration Dataset (PhysioNet).

    Returns:
        Dictionary with keys ``"signals"`` (shape ``(S, 5, N)``), ``"fs"`` (float),
        ``"channels"`` (array of channel names), and ``"subject_ids"`` (array of IDs).
    """
    data_ref = resources.files("graphsync").joinpath("data").joinpath("bidmc_5node_slice.npz")
    with resources.as_file(data_ref) as path:
        with np.load(path, allow_pickle=False) as npz:
            return {
                "signals": npz["signals"].astype(np.float64),
                "fs": float(npz["fs"]),
                "channels": npz["channels"],
                "subject_ids": npz["subject_ids"],
            }
