# graphsync

**Joint blind clock synchronization and graph learning from asynchronous smooth signals**

`graphsync` implements JSGB-Sync. Given multichannel recordings $y_m(t) = x_m(t - \tau_m) + \varepsilon_m(t)$ in which both the continuous clock offsets $\boldsymbol{\tau}$ ($\tau_1 = 0$) and the graph Laplacian $\mathbf{L}$ on which $\mathbf{x}$ is smooth are unknown, it estimates $(\hat{\boldsymbol{\tau}}, \hat{\mathbf{L}})$ jointly. It also provides the joint Cramér–Rao bound of the Gaussian smooth-graph model and the baselines used in the accompanying paper.

## Method

1. **Cross-spectral objective.** The graph Dirichlet energy is written through the phase-shifted cross-spectrum $\mathbf{S}_{\boldsymbol{\tau}}(\omega) = \mathbf{D}(\omega\boldsymbol{\tau})\mathbf{S}_0(\omega)\mathbf{D}(\omega\boldsymbol{\tau})^\mathsf{H}$, so offsets act as phase rotations and have closed-form sub-sample gradients.
2. **Spectral initializer.** A coherence-weighted, normalized Hermitian synchronization matrix at a non-wrapping reference frequency gives closed-form initial offsets.
3. **Alternating refinement.** Kalofolias log-barrier graph learning (primal–dual, KKT residual below $10^{-10}$) alternates with a Levenberg–Marquardt timing step whose Gauss–Newton matrix is a graph Laplacian.
4. **Bounds.** Closed-form Fisher information of the joint model (the timing–graph block is zero) and the anchor-only bound for comparison.

## Installation

The package is not on PyPI. Install it from GitHub:

```bash
pip install "git+https://github.com/lucher777/graphsync.git@v0.1.0"
```

For development:

```bash
git clone https://github.com/lucher777/graphsync.git
cd graphsync
pip install -e ".[dev]"
pytest
```

Requires Python 3.9+, NumPy and SciPy. The paper scripts also need Matplotlib, and `run_experiments.py` needs PyTorch (CPU or CUDA) for the batched CRLB sweep.

## Quick start

```python
import numpy as np
import graphsync as gs

sample = gs.simulate_asynchronous_network(
    m_nodes=10, n_samples=2048, snr_db=10.0, max_lag=5.0, graph_type="er", seed=42
)
fmin = float(sample.omegas[0] / (2.0 * np.pi))
fmax = float(sample.omegas[-1] / (2.0 * np.pi))

res = gs.fit_graphsync(
    sample.observed, fs=sample.fs, max_lag=6.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax
)
crlb = gs.compute_joint_crlb(
    sample.true_laplacian, sample.omegas, sample.spectral_power, sample.noise_var, sample.gamma
)
rmse = float(np.sqrt(np.mean((res.delays[1:] - sample.true_delays[1:]) ** 2)))
f1, prec, rec = gs.edge_f1_score(res.adjacency, sample.true_adjacency)
print(f"Timing RMSE: {rmse:.4f} (CRLB: {crlb.rmse_bound:.4f}) | Edge F1: {f1:.3f}")
```

## Command-line interface

```bash
graphsync demo --nodes 10 --samples 2048 --snr 12 --graph er
graphsync sync recordings.npy --fs 125.0 --max-lag 0.25 --output-json result.json
```

## Reproducing the paper

All numbers, tables and figures of the paper come from the scripts in [`paper/scripts`](paper/scripts); the stored outputs are in `paper/results` and `paper/figures`.

```bash
cd paper/scripts
export GRAPHSYNC_BIDMC_DIR=/path/to/bidmc_csv          # PhysioNet BIDMC, CSV export
export GRAPHSYNC_WESAD_NPZ=/path/to/wesad_s10_slice.npz # WESAD subject slice
python run_experiments.py        # verification, CRLB sweep, BIDMC and WESAD experiments
python rerun_synthetic_mc.py     # 200/100-trial synthetic sweeps (about 7 min on 6 cores)
python make_spl_tables.py ../tables
python make_figures_spl.py
```

Run `rerun_synthetic_mc.py` **after** `run_experiments.py`: the latter writes a 24-trial synthetic sweep, and the former replaces that block of `results/benchmark_results.json` with the 200-trial results reported in the paper. The physiological datasets are public: [PhysioNet BIDMC](https://physionet.org/content/bidmc/) and [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).

## Citation

If you use `graphsync`, please cite the software (see [`CITATION.cff`](CITATION.cff)).

## License

MIT License. See [LICENSE](LICENSE).
