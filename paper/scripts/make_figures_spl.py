"""Figures for the IEEE SPL letter (two double-column figures) and its supplement.

Fig. 1  a: profile objective min_W J between the zero-init and JSGB-Sync solutions (one network)
        b: timing RMSE vs SNR on 10-node ER graphs with the joint CRLB (Theorem 3)
        c: edge F1 vs SNR for ER / ring / BA graphs (mean +/- 1.96 SE)
Fig. 2  a: semi-synthetic BIDMC network, mean timing MAE per SNR (15 patients)
        b: paired per-patient edge F1 at 5 dB
        c: raw 5-channel BIDMC, reference graph (upper triangle) vs JSGB-Sync (lower)
Fig. S1 adjacency matrices for the Fig. 1a network
Fig. S2 CRLB variance gain (anchor / joint) vs algebraic connectivity (GPU sweep)

Okabe-Ito palette, lowercase bold panel labels, vector PDF + 300 dpi PNG.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))
import graphsync as gs  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results", "benchmark_results.json"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))

OI = {"blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#111111",
      "gray": "#7F7F7F"}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })


def style_ax(ax, grid="both"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color="#E6E6E6", linewidth=0.6)


def label(ax, s, x=-0.2, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


def illustrative_network():
    s = gs.simulate_asynchronous_network(m_nodes=10, n_samples=2048, fs=1.0, snr_db=10.0, max_lag=5.5,
                                         graph_type="er", edge_prob=0.30, gamma=0.10, k_min=60,
                                         k_max=400, seed=42)
    fmin, fmax = float(s.omegas[0] / (2 * np.pi)), float(s.omegas[-1] / (2 * np.pi))
    kw = dict(fs=1.0, max_lag=6.5, alpha=1.0, beta=0.07, fmin=fmin, fmax=fmax)
    spec = gs.fit_graphsync(s.observed, init="spectral", **kw)
    zero = gs.fit_graphsync(s.observed, init="zeros", **kw)
    wu, _, _ = gs.unsynced_kalofolias_graph(s.observed, alpha=1.0, beta=0.07, fs=1.0, fmin=fmin, fmax=fmax)
    return s, fmin, fmax, spec, zero, wu


def make_fig1(bundle, net):
    s, fmin, fmax, spec, zero, _ = net
    rmse = lambda d: float(np.sqrt(np.mean((d[1:] - s.true_delays[1:]) ** 2)))  # noqa: E731
    # Profile objective P(tau) = min_W J(tau, W), on the same standardized, energy-scaled
    # cross-spectrum that fit_graphsync uses, along the segment from the zero-init solution
    # (t = 0) to the JSGB-Sync solution (t = 1).
    x = s.observed - s.observed.mean(1, keepdims=True)
    x = x / x.std(1, keepdims=True)
    cs = gs.compute_cross_spectrum(x, fs=1.0, fmin=fmin, fmax=fmax)
    me = float(np.mean(cs.energies))
    cs = dataclasses.replace(cs, csd=cs.csd / me, energies=cs.energies / me)

    def profile(tau, w0=None):
        w, _, _ = gs.learn_smooth_graph(gs.pairwise_distance_matrix(cs, tau), alpha=1.0, beta=0.07,
                                        warm_start=w0)
        return gs.joint_objective(cs, tau, w, alpha=1.0, beta=0.07), w

    path = np.linspace(-0.2, 1.2, 141)
    en = np.empty_like(path)
    w_prev = None
    for k, t in enumerate(path):
        en[k], w_prev = profile((1 - t) * zero.delays + t * spec.delays, w_prev)
    e_init, _ = profile(spec.init_result.delays)

    rows = bundle["synthetic"]["snr_sweep"]
    by = {g: sorted([r for r in rows if r["graph_type"] == g], key=lambda r: r["snr_db"]) for g in ("er", "ring", "ba")}
    snr = np.array([r["snr_db"] for r in by["er"]])

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1, 1.05, 1.05]})
    fig.subplots_adjust(wspace=0.36)

    ax = axs[0]
    style_ax(ax)
    ax.plot(path, en, color="#333333", lw=1.1, label="Along segment")
    ax.axhline(e_init, color=OI["blue"], lw=0.9, ls=":", label="Spectral init (5)")
    ev = lambda t: en[int(np.argmin(np.abs(path - t)))]  # noqa: E731
    ax.scatter([0.0], [ev(0.0)], marker="X", s=34, color=OI["vermilion"], zorder=5,
               label=f"Zero-init solution (RMSE {rmse(zero.delays):.2f})")
    ax.scatter([1.0], [ev(1.0)], s=26, color=OI["blue"], zorder=6,
               label=f"JSGB-Sync solution (RMSE {rmse(spec.delays):.2f})")
    lo, hi = en.min(), en.max()
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.95 * (hi - lo))
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Position on segment")
    ax.set_ylabel("Profile objective")
    ax.legend(loc="upper center", frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.2)
    label(ax, "a", x=-0.27)

    ax = axs[1]
    style_ax(ax)
    er = by["er"]
    series = [("gs_zero_init_rmse", "Zero init", OI["purple"], "^", ":"),
              ("phat_rmse", "GCC-PHAT", OI["orange"], "s", "-"),
              ("xcorr_rmse", "XCorr", OI["sky"], "v", "-"),
              ("cycle_rmse", "Unw. cycle", OI["green"], "d", "-"),
              ("gs_rmse", "JSGB-Sync", OI["blue"], "o", "-")]
    for key, name, col, mk, ls in series:
        ax.semilogy(snr, [r[key] for r in er], ls=ls, color=col, marker=mk, ms=3.2,
                    lw=1.5 if key == "gs_rmse" else 0.9, label=name, zorder=4 if key == "gs_rmse" else 3)
    ax.semilogy(snr, [r["crlb_rmse"] for r in er], ls="--", color=OI["black"], lw=1.1, label="Joint CRLB", zorder=5)
    ax.set_xticks(snr)
    ax.set_ylim(0.045, 12)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Timing RMSE (samples)")
    ax.legend(loc="upper right", ncol=2, frameon=False, columnspacing=0.6, handlelength=1.6,
              handletextpad=0.3, labelspacing=0.15, borderaxespad=0.1)
    label(ax, "b", x=-0.25)

    ax = axs[2]
    style_ax(ax)
    for g, name, col, mk in [("ring", "Ring", OI["green"], "s"), ("er", "ER", OI["blue"], "o"),
                             ("ba", "BA", OI["vermilion"], "^")]:
        y = np.array([r["f1_gs"] for r in by[g]])
        se = np.array([r.get("f1_gs_se", 0.0) for r in by[g]])
        ax.plot(snr, y, color=col, marker=mk, ms=3.2, lw=1.3, label=name)
        ax.fill_between(snr, y - 1.96 * se, y + 1.96 * se, color=col, alpha=0.18, lw=0)
        yu = np.array([r["f1_unsync"] for r in by[g]])
        ax.plot(snr, yu, color=col, marker=mk, ms=2.6, lw=0.8, ls="--", mfc="white")
    ax.plot([], [], color="#555555", lw=1.3, label="JSGB-Sync")
    ax.plot([], [], color="#555555", lw=0.8, ls="--", label="Unsynced")
    ax.set_xticks(snr)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Edge F1")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.47), ncol=2, frameon=False, columnspacing=0.6,
              handlelength=1.6, handletextpad=0.3, labelspacing=0.15)
    label(ax, "c", x=-0.25)
    save(fig, "Fig1_spl_synthetic")


def make_fig2(bundle):
    real = bundle["real_physiological"]
    summ = real["bidmc_10node_summary"]
    subj = real["bidmc_10node_per_subject_5db"]
    raw5 = real["bidmc_raw5_summary"]

    fig = plt.figure(figsize=(7.16, 2.1))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.95], wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    style_ax(ax, grid="y")
    x = np.arange(len(summ))
    w = 0.2
    for k, (key, name, col) in enumerate([("gs_mae_mean", "JSGB-Sync", OI["blue"]),
                                          ("phat_mae_mean", "GCC-PHAT", OI["orange"]),
                                          ("xcorr_mae_mean", "XCorr", OI["sky"]),
                                          ("cycle_mae_mean", "Unw. cycle", OI["green"])]):
        ax.bar(x + (k - 1.5) * w, [r[key] for r in summ], width=w * 0.9, color=col, edgecolor="none",
               label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r['snr_db'])} dB" for r in summ])
    ax.set_ylabel("Mean timing MAE (ms)")
    ax.set_xlabel("SNR (semi-synthetic, 15 patients)")
    ax.set_ylim(0, max(r["cycle_mae_mean"] for r in summ) * 1.05)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.15, handletextpad=0.3)
    label(ax, "a", x=-0.24)

    ax = fig.add_subplot(grid[0, 1])
    style_ax(ax, grid="y")
    fu = np.array([r["f1_unsync"] for r in subj])
    fg = np.array([r["f1_gs"] for r in subj])
    for u, g in zip(fu, fg):
        ax.plot([0, 1], [u, g], color="#C8C8C8", lw=0.7, zorder=2)
    ax.scatter(np.zeros_like(fu), fu, s=12, color=OI["vermilion"], zorder=3, edgecolors="none")
    ax.scatter(np.ones_like(fg), fg, s=12, color=OI["blue"], zorder=3, edgecolors="none")
    for xx, v, col, ha, dx in [(0, fu.mean(), OI["vermilion"], "right", -0.2), (1, fg.mean(), OI["blue"], "left", 0.2)]:
        ax.plot([xx - 0.1, xx + 0.1], [v, v], color="#222222", lw=1.6, zorder=4)
        ax.text(xx + dx, v, f"{v:.2f}", ha=ha, va="center", fontsize=6.8, color="#222222")
    p = [r["wilcoxon_p_f1"] for r in summ if r["snr_db"] == 5.0][0]
    e = int(np.floor(np.log10(p)))
    n_up = int(np.sum(fg > fu))
    ax.text(0.5, 1.02, rf"{n_up}/{len(fg)} improved, $p={p / 10 ** e:.1f}\times10^{{{e}}}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unsynced", "JSGB-Sync"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Edge F1 (5 dB)")
    label(ax, "b", x=-0.3)

    ax = fig.add_subplot(grid[0, 2])
    wr = np.array(raw5["avg_w_ref"])
    wg = np.array(raw5["avg_w_gs"])
    wr, wg = wr / wr.max(), wg / wg.max()
    mat = np.triu(wr, 1) + np.tril(wg, -1)
    mask = np.eye(5, dtype=bool)
    im = ax.imshow(np.ma.array(mat, mask=mask), cmap="Blues", vmin=0, vmax=1)
    names = ["II", "ECG2", "ECG3", "PPG", "RESP"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="#888888", lw=0.6)
    for i in range(5):
        for j in range(5):
            if i != j and mat[i, j] >= 0.1:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if mat[i, j] > 0.55 else "#111111")
    ax.set_title("Upper: ref.; lower: JSGB-Sync", fontsize=6.4, pad=3)
    for sp in ax.spines.values():
        sp.set_color("#BBBBBB")
        sp.set_linewidth(0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    fig.canvas.draw()
    y_ref = fig.axes[0].get_position().y1
    fig.text(ax.get_position().x0 - 0.075, y_ref + 0.035, "c", fontsize=9.5, fontweight="bold", va="bottom")
    save(fig, "Fig2_spl_bidmc")


def make_figS1(net):
    s, _, _, spec, _, wu = net
    f1g = gs.edge_f1_score(spec.adjacency, s.true_adjacency)[0]
    f1u = gs.edge_f1_score(wu, s.true_adjacency)[0]
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.2))
    fig.subplots_adjust(wspace=0.15)
    for ax, (m, lab, t) in zip(axs, [(s.true_adjacency, "a", r"Ground truth $\mathbf{W}^*$"),
                                     (spec.adjacency, "b", f"JSGB-Sync (F1 = {f1g:.2f})"),
                                     (wu, "c", f"Unsynced (F1 = {f1u:.2f})")]):
        im = ax.imshow(m / m.max(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 4, 9])
        ax.set_xticklabels(["1", "5", "10"])
        ax.set_yticks([0, 4, 9])
        ax.set_yticklabels(["1", "5", "10"] if lab == "a" else [])
        ax.set_xlabel("Node")
        ax.set_title(t, fontsize=7.2, pad=3)
        label(ax, lab, x=-0.12, y=1.08)
    axs[0].set_ylabel("Node")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label("Normalized weight", fontsize=7)
    save(fig, "FigS1_adjacency")


def make_figS2(bundle):
    rows = bundle["gpu_crlb_sweep"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    style_ax(ax)
    for m, col, mk in [(6, OI["sky"], "s"), (10, OI["blue"], "o"), (15, OI["green"], "^"), (20, OI["vermilion"], "D")]:
        sub = sorted([r for r in rows if r["m_nodes"] == m], key=lambda r: r["mean_lam2"])
        ax.plot([r["mean_lam2"] for r in sub], [r["variance_reduction_ratio"] for r in sub], color=col,
                marker=mk, ms=3.5, lw=1.2, label=f"$M={m}$")
    ax.axhline(1.0, color="#666666", ls="--", lw=0.8)
    ax.set_xlabel(r"Mean algebraic connectivity $\lambda_2(\mathbf{L})$")
    ax.set_ylabel("CRLB variance ratio (anchor / joint)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "FigS2_connectivity_gain")


def main():
    setup_style()
    with open(RES, encoding="utf-8") as fh:
        bundle = json.load(fh)
    net = illustrative_network()
    make_fig1(bundle, net)
    make_fig2(bundle)
    make_figS1(net)
    make_figS2(bundle)


if __name__ == "__main__":
    main()
