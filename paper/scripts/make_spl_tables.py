"""Write LaTeX tables for the IEEE SPL letter and its supplement from stored results.

Usage: python make_spl_tables.py <out_dir>
Writes tab_main_synth.tex, tab_main_bidmc.tex and supp_tables.tex.
Sources: results/benchmark_results.json (synthetic block = 200-trial rerun from
rerun_synthetic_mc.py; parity, GPU sweep and physiological blocks from run_experiments.py).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results", "benchmark_results.json"))


def sci(x, d=2):
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.{d}f}\\times10^{{{e}}}$"


def pfmt(p):
    return "$<10^{-4}$" if p < 1e-4 else sci(p, 1)


def snr_s(s):
    return f"${s:+d}$" if s else "$0$"


def bold_min(vals, fmt):
    m = min(vals)
    return [("\\textbf{" + fmt(v) + "}") if v == m else fmt(v) for v in vals]


def synth_rows(rows, with_f1=True):
    out = []
    for r in rows:
        f = lambda v: f"{v:.3f}"  # noqa: E731
        errs = bold_min([r["gs_rmse"], r["gs_zero_init_rmse"], r["cycle_rmse"], r["xcorr_rmse"], r["phat_rmse"]], f)
        lo, hi = r["crlb_ratio_ci95"]
        cells = [snr_s(r["snr_db"]), f"{r['crlb_rmse']:.3f}", errs[0],
                 f"{r['crlb_ratio']:.2f} [{lo:.2f}, {hi:.2f}]", errs[1], errs[2], errs[3], errs[4]]
        if with_f1:
            cells += [f"{r['f1_gs']:.2f}", f"{r['f1_unsync']:.2f}"]
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out)


SYNTH_HEAD = (r"SNR & CRLB & JSGB & Ratio [95\% CI] & Zero init & Unw.\ cycle & XCorr & GCC-PHAT"
              r" & F1 JSGB & F1 unsync. \\")


def main_synth(b):
    rows = sorted([r for r in b["synthetic"]["snr_sweep"] if r["graph_type"] == "er"], key=lambda r: r["snr_db"])
    n = rows[0]["n_trials"]
    return rf"""\begin{{table*}}[!t]
\centering
\caption{{Synthetic 10-node Erd\H{{o}}s--R\'enyi networks ({n} trials per SNR). Timing RMSE in samples
(lowest in bold), ratio of JSGB-Sync RMSE to the joint CRLB with trial-bootstrap 95\% interval, and edge F1.}}
\label{{tab:synth}}
\footnotesize
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{rrrcrrrrrr}}
\toprule
{SYNTH_HEAD}
\midrule
{synth_rows(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main_bidmc(b):
    real = b["real_physiological"]
    lines = []
    for s in real["bidmc_10node_summary"]:
        vals = [s["gs_mae_median"], s["phat_mae_median"], s["xcorr_mae_median"], s["cycle_mae_median"]]
        cells = bold_min(vals, lambda v: f"{v:.2f}")
        lines.append(f"Semi-syn., {int(s['snr_db'])} dB & " + " & ".join(cells)
                     + f" & {s['f1_gs_mean']:.2f} & {s['f1_unsync_mean']:.2f} \\\\")
    w = real["wesad_summary"]
    cells = bold_min([w["gs_mae_ms"], w["phat_mae_ms"], w["xcorr_mae_ms"], w["cycle_mae_ms"]], lambda v: f"{v:.2f}")
    lines.append("WESAD$^\\dagger$ & " + " & ".join(cells) + f" & {w['f1_gs']:.2f} & {w['f1_unsync']:.2f} \\\\")
    r5 = real["bidmc_raw5_summary"]
    lines.append(f"Raw 5-ch.$^\\ddagger$ & {r5['ecg3_mae_median_ms']:.2f} & -- & -- & -- & "
                 f"{r5['f1_gs_mean']:.2f} & {r5['f1_unsync_mean']:.2f} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{PhysioNet BIDMC (15 patients, 125\,Hz): median timing MAE (ms; lowest in bold) and mean edge F1.}}
\label{{tab:bidmc}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
 & \multicolumn{{4}}{{c}}{{Median MAE (ms)}} & \multicolumn{{2}}{{c}}{{Edge F1}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-7}}
Regime & JSGB & PHAT & XCorr & Cycle & JSGB & Unsync. \\
\midrule
{body}
\bottomrule
\end{{tabular}}\\[2pt]
\parbox{{\columnwidth}}{{\scriptsize $^\dagger$One WESAD subject, 64\,Hz, 10 networks, mean MAE.
$^\ddagger$Median MAE over the two ECG channels; F1 against the graph learned from the original synchronized
recording.}}
\end{{table}}
"""


def supp(b):
    p = b["parity"]
    par = [
        ("Parseval identity (2), relative error", p["parseval_rel_err"]),
        ("Analytic vs.\\ finite-difference gradient (frequency domain)", p["grad_freq_fd_rel_err"]),
        ("Analytic vs.\\ finite-difference gradient (time domain)", p["grad_time_fd_rel_err"]),
        ("Analytic vs.\\ finite-difference Hessian", p["hess_fd_rel_err"]),
        ("Closed-form FIM (Thm.~3) vs.\\ numerical KL Hessian, relative", p["fim_rel_err"]),
        ("$\\|\\mathbf{I}_{\\tau\\tau}\\mathbf{1}\\|$ (shift null space)", p["fim_nullspace_err"]),
        ("$\\|\\mathbf{I}_{\\tau\\mathbf{w}}\\|_F$ (orthogonality, Thm.~3)", p["cross_block_norm"]),
        ("Noiseless spectral initializer, max offset error (samples)", p["noiseless_init_max_err"]),
        ("Noiseless JSGB-Sync, max offset error (samples)", p["noiseless_final_max_err"]),
        ("KKT residual of the $\\mathbf{W}$-step", p["noiseless_kkt"]),
    ]
    t1 = "\n".join(f"{a} & {sci(v)} \\\\" for a, v in par)

    snr = b["synthetic"]["snr_sweep"]
    blocks = []
    for g, name in [("ring", "Ring with chords"), ("ba", "Barab\\'asi--Albert")]:
        rows = sorted([r for r in snr if r["graph_type"] == g], key=lambda r: r["snr_db"])
        blocks.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{name}}}}} \\\\\n" + synth_rows(rows))
    t2 = "\n\\midrule\n".join(blocks)
    n_snr = snr[0]["n_trials"]

    size = b["synthetic"]["size_sweep"]
    t3 = []
    for r in size:
        errs = bold_min([r["gs_rmse"], r["cycle_rmse"], r["xcorr_rmse"], r["phat_rmse"]], lambda v: f"{v:.3f}")
        lo, hi = r["crlb_ratio_ci95"]
        t3.append(f"{r['m_nodes']} & {r['edge_prob']:.2f} & {r['crlb_rmse']:.3f} & {errs[0]} & "
                  f"{r['crlb_ratio']:.2f} [{lo:.2f}, {hi:.2f}] & " + " & ".join(errs[1:])
                  + f" & {r['f1_gs']:.2f} & {r['f1_unsync']:.2f} \\\\")
    t3 = "\n".join(t3)
    n_size = size[0]["n_trials"]

    gpu = b["gpu_crlb_sweep"]
    t4 = "\n".join(f"{r['m_nodes']} & {r['edge_prob']:.2f} & {r['mean_lam2']:.2f} & {r['joint_crlb_rmse']:.3f} & "
                   f"{r['anchor_crlb_rmse']:.3f} & {r['variance_reduction_ratio']:.2f} \\\\" for r in gpu)

    real = b["real_physiological"]
    per = real["bidmc_10node_per_subject_5db"]
    t5 = []
    for i, r in enumerate(per, 1):
        errs = bold_min([r["gs_mae_ms"], r["phat_mae_ms"], r["xcorr_mae_ms"], r["cycle_mae_ms"]], lambda v: f"{v:.2f}")
        t5.append(f"{i:02d} & " + " & ".join(errs) + f" & {r['f1_gs']:.2f} & {r['f1_unsync']:.2f} \\\\")
    s5 = [s for s in real["bidmc_10node_summary"] if s["snr_db"] == 5.0][0]
    t5 = "\n".join(t5)
    stats = []
    for s in real["bidmc_10node_summary"]:
        stats.append(f"{int(s['snr_db'])} dB & {s['gs_mae_mean']:.2f} & {s['phat_mae_mean']:.2f} & "
                     f"{s['xcorr_mae_mean']:.2f} & {s['cycle_mae_mean']:.2f} & {pfmt(s['wilcoxon_p_vs_phat'])} & "
                     f"{pfmt(s['wilcoxon_p_vs_xcorr'])} & {pfmt(s['wilcoxon_p_vs_cycle'])} & {pfmt(s['wilcoxon_p_f1'])} \\\\")
    stats = "\n".join(stats)

    raw = real["bidmc_raw5_rows"]
    r5 = real["bidmc_raw5_summary"]
    t7 = "\n".join(f"{i:02d} & {r['ecg3_mae_ms']:.1f} & {r['f1_gs']:.2f} & {r['f1_unsync']:.2f} \\\\"
                   for i, r in enumerate(raw, 1))
    n_up = sum(r["f1_gs"] > r["f1_unsync"] for r in raw)
    n_tie = sum(r["f1_gs"] == r["f1_unsync"] for r in raw)
    n_fail = sum(r["ecg3_mae_ms"] > 40 for r in raw)

    return rf"""\begin{{table}}[!htbp]
\centering
\caption{{Numerical verification of the implementation (float64).}}
\label{{tab:S1}}
\footnotesize
\begin{{tabular}}{{lr}}
\toprule
Check & Error \\
\midrule
{t1}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table*}}[!htbp]
\centering
\caption{{Synthetic 10-node ring and Barab\'asi--Albert networks ({n_snr} trials per SNR); columns as in Table~I
of the letter.}}
\label{{tab:S2}}
\footnotesize
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{rrrcrrrrrr}}
\toprule
{SYNTH_HEAD}
\midrule
{t2}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table*}}[!htbp]
\centering
\caption{{Network size, ER graphs, SNR $=5$\,dB ({n_size} trials per size). Timing RMSE in samples.}}
\label{{tab:S3}}
\footnotesize
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{rrrrcrrrrr}}
\toprule
$M$ & $p$ & CRLB & JSGB & Ratio [95\% CI] & Unw.\ cycle & XCorr & GCC-PHAT & F1 JSGB & F1 unsync. \\
\midrule
{t3}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table}}[!htbp]
\centering
\caption{{Joint vs.\ anchor-only CRLB (25 ER graphs per cell, 5\,dB, 200 frequencies; computed on GPU).
The anchor bound uses only the $M-1$ pairs $(1,m)$.}}
\label{{tab:S4}}
\footnotesize
\begin{{tabular}}{{rrrrrr}}
\toprule
$M$ & $p$ & $\bar\lambda_2$ & Joint & Anchor & Var.\ ratio \\
\midrule
{t4}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[!htbp]
\centering
\caption{{Semi-synthetic BIDMC networks at 5\,dB, per patient: timing MAE (ms; lowest in bold) and edge F1.}}
\label{{tab:S5}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
Patient & JSGB & PHAT & XCorr & Cycle & F1 JSGB & F1 uns. \\
\midrule
{t5}
\midrule
Mean & {s5['gs_mae_mean']:.2f} & {s5['phat_mae_mean']:.2f} & {s5['xcorr_mae_mean']:.2f} & {s5['cycle_mae_mean']:.2f} & {s5['f1_gs_mean']:.2f} & {s5['f1_unsync_mean']:.2f} \\
Median & {s5['gs_mae_median']:.2f} & {s5['phat_mae_median']:.2f} & {s5['xcorr_mae_median']:.2f} & {s5['cycle_mae_median']:.2f} & & \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table*}}[!htbp]
\centering
\caption{{Semi-synthetic BIDMC networks: mean MAE (ms) and one-sided paired Wilcoxon signed-rank $p$-values
(15 patients; JSGB-Sync error lower, or F1 higher). Values are not corrected for multiple comparisons.}}
\label{{tab:S6}}
\footnotesize
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
SNR & JSGB & PHAT & XCorr & Cycle & $p$ vs.\ PHAT & $p$ vs.\ XCorr & $p$ vs.\ cycle & $p$ (F1) \\
\midrule
{stats}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table}}[!htbp]
\centering
\caption{{Raw five-channel BIDMC recordings with injected offsets ($\pm100$\,ms) and noise: MAE over the two
ECG channels (ms) and F1 against the reference graph. F1 is higher for {n_up}/15 patients ({n_tie} tie);
offset recovery fails (MAE $>40$\,ms) for {n_fail}/15. Paired Wilcoxon $p$ (F1) $=$ {pfmt(r5['wilcoxon_p_f1'])}.}}
\label{{tab:S7}}
\footnotesize
\begin{{tabular}}{{rrrr}}
\toprule
Patient & ECG MAE & F1 JSGB & F1 uns. \\
\midrule
{t7}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "tables")
    os.makedirs(out, exist_ok=True)
    with open(RES, encoding="utf-8") as fh:
        b = json.load(fh)
    for name, txt in [("tab_main_synth.tex", main_synth(b)), ("tab_main_bidmc.tex", main_bidmc(b)),
                      ("supp_tables.tex", supp(b))]:
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("wrote", name)


if __name__ == "__main__":
    main()
