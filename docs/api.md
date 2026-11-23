# API Reference

## Core Solvers

- `graphsync.fit_graphsync(...)` -> `GraphSyncResult`
- `graphsync.spectral_angular_init(...)` -> `SpectralInitResult`
- `graphsync.learn_smooth_graph(...)` -> `(W, L, kkt_residual)`
- `graphsync.joint_objective(...)` -> `float`
- `graphsync.timing_gradient_and_hessian(...)` -> `(grad, hess_exact, hess_gn)`

## Spectral Utilities

- `graphsync.compute_cross_spectrum(...)` -> `CrossSpectrum`
- `graphsync.fractional_shift(...)` -> `np.ndarray`
- `graphsync.apply_offsets(...)` -> `np.ndarray`
- `graphsync.align_signals(...)` -> `np.ndarray`
- `graphsync.pairwise_distance_matrix(...)` -> `np.ndarray`

## Cramér-Rao Lower Bound & Parity Verification

- `graphsync.compute_joint_crlb(...)` -> `CRLBResult`
- `graphsync.compute_pairwise_anchor_crlb(...)` -> `float`
- `graphsync.verify_parseval_and_derivatives(...)` -> `Dict[str, float]`
- `graphsync.verify_crlb_fim_parity(...)` -> `Dict[str, float]`

## Baselines & Datasets

- `graphsync.gcc_phat_anchor_sync(...)`
- `graphsync.xcorr_anchor_sync(...)`
- `graphsync.unweighted_cycle_sync(...)`
- `graphsync.unsynced_kalofolias_graph(...)`
- `graphsync.generate_connected_graph(...)`
- `graphsync.simulate_asynchronous_network(...)`
- `graphsync.load_bundled_bidmc_slice(...)`
