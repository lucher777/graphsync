# Tutorial

## 1. Synthetic Asynchronous Graph Synchronization

```python
import numpy as np
import graphsync as gs

sample = gs.simulate_asynchronous_network(
    m_nodes=10, n_samples=2048, snr_db=10.0, max_lag=5.0, graph_type="er", seed=123
)
fmin = float(sample.omegas[0] / (2.0 * np.pi))
fmax = float(sample.omegas[-1] / (2.0 * np.pi))

res = gs.fit_graphsync(
    sample.observed, fs=sample.fs, max_lag=6.0, alpha=1.0, beta=0.06, fmin=fmin, fmax=fmax
)
print("Estimated delays:", np.round(res.delays, 3))
print("True delays:     ", np.round(sample.true_delays, 3))
print("Edge F1:         ", gs.edge_f1_score(res.adjacency, sample.true_adjacency)[0])
```

## 2. Comparing Against the Analytical Joint CRLB

```python
crlb = gs.compute_joint_crlb(
    sample.true_laplacian,
    sample.omegas,
    sample.spectral_power,
    noise_var=sample.noise_var,
    gamma=sample.gamma,
    verify_cross_block=True,
)
print("Analytical CRLB RMSE:", crlb.rmse_bound)
print("Cross-block ||I_{tau,w}||_F:", crlb.cross_block_norm)
```

## 3. Working with the Bundled Real Physiological Slice

```python
data = gs.load_bundled_bidmc_slice()
signals = data["signals"][0]  # 5 channels x 3750 samples at 125 Hz
fs = data["fs"]
res = gs.fit_graphsync(
    signals, fs=fs, max_lag=0.25, alpha=1.0, beta=0.08, fmin=0.2, fmax=15.0, nperseg=512
)
print("Channels:", data["channels"])
print("Learned adjacency:\n", np.round(res.adjacency, 3))
```
