# graphsync Documentation

`graphsync` provides a mathematically principled, self-contained Python implementation of **Joint Smooth-Graph Blind Synchronization (`JSGB-Sync`)** for asynchronous multi-sensor networks.

## Key Capabilities

- **Joint Timing & Topology Inference (`fit_graphsync`)**: Simultaneously recovers continuous sub-sample clock offsets and the underlying smooth graph Laplacian.
- **Closed-Form Spectral Initialization (`spectral_angular_init`)**: Global cycle-consistent angular synchronization without requiring known topology.
- **Exact Primal-Dual Graph Learning (`learn_smooth_graph`)**: High-precision Kalofolias log-barrier graph learning with verifiable KKT residuals (`< 1e-10`).
- **Analytical Joint CRLB (`compute_joint_crlb`)**: Closed-form Fisher Information Matrix and timing-topology orthogonality verification.
- **Standard Baselines (`baselines.py`)**: Pairwise GCC-PHAT (Knapp & Carter 1976), pairwise cross-correlation, unweighted cycle synchronization, and unsynchronized Kalofolias graph learning.
