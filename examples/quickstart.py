"""Quickstart example demonstrating joint blind synchronization and CRLB verification."""

from __future__ import annotations

import numpy as np

import graphsync as gs


def main() -> None:
    """Run a synthetic 10-node synchronization example and compare with CRLB."""
    sample = gs.simulate_asynchronous_network(
        m_nodes=10,
        n_samples=2048,
        fs=1.0,
        snr_db=10.0,
        max_lag=5.0,
        graph_type="er",
        edge_prob=0.30,
        gamma=0.1,
        k_min=8,
        k_max=360,
        seed=42,
    )
    fmin = float(sample.omegas[0] / (2.0 * np.pi))
    fmax = float(sample.omegas[-1] / (2.0 * np.pi))

    res = gs.fit_graphsync(
        sample.observed,
        fs=sample.fs,
        max_lag=6.0,
        alpha=1.0,
        beta=0.06,
        fmin=fmin,
        fmax=fmax,
    )
    t_phat = gs.gcc_phat_anchor_sync(sample.observed, fs=sample.fs, max_lag=6.0, fmin=fmin, fmax=fmax)
    w_unsync, _, _ = gs.unsynced_kalofolias_graph(
        sample.observed, alpha=1.0, beta=0.06, fs=sample.fs, fmin=fmin, fmax=fmax
    )
    crlb = gs.compute_joint_crlb(
        sample.true_laplacian,
        sample.omegas,
        sample.spectral_power,
        noise_var=sample.noise_var,
        gamma=sample.gamma,
        verify_cross_block=True,
    )

    rmse_gs = float(np.sqrt(np.mean((res.delays[1:] - sample.true_delays[1:]) ** 2)))
    rmse_phat = float(np.sqrt(np.mean((t_phat[1:] - sample.true_delays[1:]) ** 2)))
    f1_gs = gs.edge_f1_score(res.adjacency, sample.true_adjacency)[0]
    f1_un = gs.edge_f1_score(w_unsync, sample.true_adjacency)[0]

    print(f"JSGB-Sync RMSE:      {rmse_gs:.4f} samples (CRLB: {crlb.rmse_bound:.4f})")
    print(f"GCC-PHAT RMSE:       {rmse_phat:.4f} samples")
    print(f"JSGB-Sync Edge F1:   {f1_gs:.3f} (Unsynced Kalofolias: {f1_un:.3f})")
    print(f"FIM ||I_{{tau,w}}||: {crlb.cross_block_norm:.2e}")


if __name__ == "__main__":
    main()
