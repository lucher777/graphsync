"""Command-line interface for graphsync."""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

import numpy as np

from .crlb import compute_joint_crlb
from .datasets import simulate_asynchronous_network
from .solver import fit_graphsync
from .topology import edge_f1_score


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ``graphsync`` CLI."""
    parser = argparse.ArgumentParser(
        prog="graphsync",
        description="Joint blind time-offset and smooth graph-topology synchronization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run a synthetic joint synchronization demo.")
    p_demo.add_argument("--nodes", type=int, default=8, help="Number of sensor nodes.")
    p_demo.add_argument("--samples", type=int, default=512, help="Time samples per sensor.")
    p_demo.add_argument("--snr", type=float, default=12.0, help="SNR in dB.")
    p_demo.add_argument("--graph", type=str, default="er", choices=["er", "ba", "ring", "sbm"])
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed.")

    p_sync = sub.add_parser("sync", help="Synchronize a 2-D .npy multi-sensor array.")
    p_sync.add_argument("input_npy", type=str, help="Path to input .npy file of shape (M, N).")
    p_sync.add_argument("--fs", type=float, default=1.0, help="Sampling rate in Hz.")
    p_sync.add_argument("--max-lag", type=float, default=2.0, help="Max absolute offset.")
    p_sync.add_argument("--alpha", type=float, default=1.0, help="Log-barrier weight.")
    p_sync.add_argument("--beta", type=float, default=1.0, help="Frobenius regularization.")
    p_sync.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ``graphsync`` command-line interface.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Exit status code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        sample = simulate_asynchronous_network(
            m_nodes=args.nodes,
            n_samples=args.samples,
            snr_db=args.snr,
            graph_type=args.graph,
            seed=args.seed,
        )
        res = fit_graphsync(
            sample.observed,
            fs=sample.fs,
            max_lag=2.0,
            fmin=float(sample.omegas[0] / (2.0 * np.pi)),
            fmax=float(sample.omegas[-1] / (2.0 * np.pi)),
        )
        err = res.delays[1:] - sample.true_delays[1:]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        f1, prec, rec = edge_f1_score(res.adjacency, sample.true_adjacency)
        crlb = compute_joint_crlb(
            sample.true_laplacian,
            sample.omegas,
            sample.spectral_power,
            noise_var=max(sample.noise_var, 1e-12),
            gamma=sample.gamma,
        )
        payload = {
            "nodes": args.nodes,
            "snr_db": args.snr,
            "timing_rmse": rmse,
            "crlb_rmse": crlb.rmse_bound,
            "edge_f1": f1,
            "edge_precision": prec,
            "edge_recall": rec,
            "converged": res.converged,
            "n_iter": res.n_iter,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "sync":
        signals = np.load(args.input_npy)
        res = fit_graphsync(
            signals,
            fs=args.fs,
            max_lag=args.max_lag,
            alpha=args.alpha,
            beta=args.beta,
        )
        payload = {
            "delays": res.delays.tolist(),
            "adjacency": res.adjacency.tolist(),
            "algebraic_connectivity": res.algebraic_connectivity,
            "converged": res.converged,
            "n_iter": res.n_iter,
            "kkt_residual": res.kkt_residual,
        }
        text = json.dumps(payload, indent=2)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            print(text)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
