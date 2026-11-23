"""graphsync: Joint blind time-offset and unknown graph-topology synchronization."""

from .baselines import (
    gcc_phat_anchor_sync,
    unsynced_kalofolias_graph,
    unweighted_cycle_sync,
    xcorr_anchor_sync,
)
from .crlb import compute_joint_crlb, compute_pairwise_anchor_crlb
from .datasets import (
    generate_connected_graph,
    load_bundled_bidmc_slice,
    simulate_asynchronous_network,
)
from .equiv import (
    time_domain_dirichlet_energy,
    verify_crlb_fim_parity,
    verify_parseval_and_derivatives,
)
from .solver import fit_graphsync, joint_objective, timing_gradient_and_hessian
from .spectral import (
    align_signals,
    apply_offsets,
    compute_cross_spectrum,
    evaluate_cross_correlation_derivatives,
    fractional_shift,
    pairwise_distance_matrix,
    spectral_angular_init,
)
from .topology import (
    adjacency_to_laplacian,
    algebraic_connectivity,
    edge_f1_score,
    graph_learning_kkt_residual,
    graph_learning_objective,
    learn_smooth_graph,
)
from .types import (
    CRLBResult,
    CrossSpectrum,
    GraphSyncResult,
    SpectralInitResult,
    SyntheticNetworkSample,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CRLBResult",
    "CrossSpectrum",
    "GraphSyncResult",
    "SpectralInitResult",
    "SyntheticNetworkSample",
    "adjacency_to_laplacian",
    "algebraic_connectivity",
    "align_signals",
    "apply_offsets",
    "compute_cross_spectrum",
    "compute_joint_crlb",
    "compute_pairwise_anchor_crlb",
    "edge_f1_score",
    "evaluate_cross_correlation_derivatives",
    "fit_graphsync",
    "fractional_shift",
    "gcc_phat_anchor_sync",
    "generate_connected_graph",
    "graph_learning_kkt_residual",
    "graph_learning_objective",
    "joint_objective",
    "learn_smooth_graph",
    "load_bundled_bidmc_slice",
    "pairwise_distance_matrix",
    "simulate_asynchronous_network",
    "spectral_angular_init",
    "time_domain_dirichlet_energy",
    "timing_gradient_and_hessian",
    "unsynced_kalofolias_graph",
    "unweighted_cycle_sync",
    "verify_crlb_fim_parity",
    "verify_parseval_and_derivatives",
    "xcorr_anchor_sync",
]
