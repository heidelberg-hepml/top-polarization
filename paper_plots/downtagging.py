"""Paper plots for the down tagger (d = hard vs. soft light jet): ROC (eps vs. 1/eps), calibration,
and on the polarized evaluation the cos(theta_d) histogram and tilde beta_d vs. selected fraction.

usage: python -m paper_plots.downtagging runs/downtagging/down_slim_s* [--outdir DIR]
"""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from paper_plots.common import (
    COLORS,
    FIGSIZE,
    FONTSIZE,
    LINE_WIDTH,
    N_BOOTSTRAP,
    REPO_DIR,
    auc_errors,
    corner_text,
    hist_band,
    legend,
    load_all,
    mean_seed_std,
    parse_runs,
    pm_text,
    save,
    step,
)

LABEL_1 = r"$d = q_{\text{hard}}$"
TPR_LABEL, FPR_LABEL = r"\epsilon_d^{q_\text{hard}}", r"\epsilon_u^{q_\text{hard}}"

# beta vs. efficiency of a GNN down tagger, for comparison
GNN_EFFICIENCY = os.path.join(REPO_DIR, "data", "beta_opt_GNN.csv")

ROC_GRID = np.linspace(0, 1, 401)
FRACTIONS = np.linspace(0.2, 1.0, 81)


def plot_roc(outdir, preds_dirs, rocs, scores, model):
    """Mean over seeds of 1/fpr at fixed tpr, band: seed std and binomial error of the first seed
    in quadrature; below one background event eps_B is not measured. AUC with seed and bootstrap
    errors."""
    labels_true = scores[0]["labels_true"]
    n_0 = len(labels_true) - int(labels_true.sum())
    auc, seed, stat = auc_errors(preds_dirs)

    fprs = [np.interp(ROC_GRID, r["tpr"], r["fpr"]) for r in rocs]
    fpr, fpr_seed = mean_seed_std(fprs)
    fpr_err = np.hypot(fpr_seed, np.sqrt(fprs[0] * (1 - fprs[0]) / n_0))

    rnd = np.linspace(1e-3, 1, 100)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_yscale("log")
    (rnd_line,) = ax.plot(rnd, 1 / rnd, "k--", lw=LINE_WIDTH)
    mask = fpr >= 1 / n_0
    lo, hi = fpr - fpr_err, fpr + fpr_err
    ax.fill_between(
        ROC_GRID[mask],
        1 / hi[mask],
        1 / np.maximum(lo, 1 / n_0)[mask],
        color="red",
        alpha=0.3,
        lw=0,
    )
    (line,) = ax.plot(ROC_GRID[mask], 1 / fpr[mask], color="red", lw=LINE_WIDTH)
    ax.set_xlabel(f"${TPR_LABEL}$", fontsize=FONTSIZE)
    ax.set_ylabel(f"$1 / {FPR_LABEL}$", fontsize=FONTSIZE)
    ax.set_xlim(0, 1)
    corner_text(ax, 0.95, 0.95, f"AUC $= {pm_text(auc, seed, stat)}$", ha="right")
    legend(ax, [line, rnd_line], [model, "Random"], loc="upper right", bbox_to_anchor=(1, 0.87))
    save(fig, ax, outdir, "roc")


def calibration(labels_true, labels_predict, idx=None, bins=20):
    # same binning as plots.plot_calibration; idx resamples the events
    if idx is not None:
        labels_true, labels_predict = labels_true[idx], labels_predict[idx]
    edges = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(labels_predict, edges) - 1, 0, bins - 1)
    count = np.bincount(i, minlength=bins)
    n = np.maximum(count, 1)
    score_mean = np.bincount(i, weights=labels_predict, minlength=bins) / n
    frac = np.bincount(i, weights=labels_true, minlength=bins) / n
    ece = np.sum(count * np.abs(frac - score_mean)) / count.sum()
    return count, score_mean, frac, ece


def plot_calibration(outdir, scores, model):
    calibs = [calibration(s["labels_true"], s["labels_predict"]) for s in scores]
    count = np.stack([c[0] for c in calibs])
    mask = (count > 0).all(axis=0)
    score_mean, _ = mean_seed_std([c[1] for c in calibs])
    frac, frac_seed = mean_seed_std([c[2] for c in calibs])
    ece, ece_seed = mean_seed_std([c[3] for c in calibs])
    frac_stat = np.sqrt(calibs[0][2] * (1 - calibs[0][2]) / np.maximum(count[0], 1))
    rng = np.random.default_rng(0)
    n = len(scores[0]["labels_true"])
    ece_stat = np.std(
        [
            calibration(
                scores[0]["labels_true"], scores[0]["labels_predict"], idx=rng.integers(0, n, n)
            )[3]
            for _ in range(N_BOOTSTRAP)
        ]
    )
    print(
        f"  ECE over {len(scores)} runs: {ece:.4f} +- {ece_seed:.4f} (seed) +- {ece_stat:.4f} (stat)"
    )
    x, y, dy = score_mean[mask], frac[mask], np.hypot(frac_seed, frac_stat)[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    (diag,) = ax.plot([0, 1], [0, 1], "k--", lw=LINE_WIDTH)
    ax.fill_between(x, y - dy, y + dy, color=COLORS[2], alpha=0.3, lw=0)
    (line,) = ax.plot(x, y, color=COLORS[2], lw=LINE_WIDTH)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p(d = q_{\text{hard}})$", fontsize=FONTSIZE)
    ax.set_ylabel(f"Fraction of {LABEL_1}", fontsize=FONTSIZE)
    legend(ax, [line, diag], [model, "Perfect calibration"], loc="lower right")
    corner_text(ax, 0.05, 0.95, f"ECE $= {pm_text(ece, ece_seed, ece_stat)}$")
    save(fig, ax, outdir, "calibration")


def plot_polarization(outdir, polarizations):
    """cos(tilde theta_d) of the network's down probe to the spin axis, dashed: of the true down jet
    (the same events for every seed, so first seed only)."""
    bins = np.linspace(-1, 1, 21)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    top_l = hist_band(ax, [p["score_l"] for p in polarizations], bins, COLORS[1])
    top_r = hist_band(ax, [p["score_r"] for p in polarizations], bins, COLORS[2])
    top = max(top_l.max(), top_r.max())
    for key, color in [("cos_true_l", COLORS[1]), ("cos_true_r", COLORS[2])]:
        counts, _ = np.histogram(polarizations[0][key], bins=bins)
        hist = counts / len(polarizations[0][key])
        step(ax, bins, hist, color, ls="--", alpha=0.5)
        top = max(top, hist.max())
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.6 * top)
    # wide tick labels such as 0.025 push the y label out of the figure
    ax.set_yticks([0, 0.05, 0.1])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"$\cos \theta$", fontsize=FONTSIZE)
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    pols = [mlines.Line2D([], [], color=c) for c in COLORS[1:3]]
    legend(ax, pols, [r"$P_{t_{\text{had}}} = L$", r"$P_{t_{\text{had}}} = R$"], loc="upper left")
    ax.add_artist(ax.get_legend())
    angles = [mlines.Line2D([], [], color="gray", ls=ls) for ls in ("-", "--")]
    legend(ax, angles, [r"$\cos \tilde\theta_d$", r"$\cos \theta_d$"], loc="upper right")
    save(fig, ax, outdir, "polarization")


def beta_vs_fraction(polarization, score):
    """beta_d of the L and R samples when keeping the given fraction of most confident events.

    Same cut family as likelihood.cut_efficiency (|p - 1/2| above a threshold), but the x axis is
    the fraction of events actually kept. Errors are the standard error of the per-event beta.
    """
    labels_true, conf = score["labels_true"], np.abs(score["labels_predict"] - 0.5)
    cuts = np.quantile(conf, 1 - FRACTIONS)
    out = {}
    for lbl, key in [(0, "beta_l"), (1, "beta_r")]:
        conf_lbl, beta = conf[labels_true == lbl], polarization[key]
        assert len(conf_lbl) == len(beta)
        selected = [beta[conf_lbl >= cut] for cut in cuts]
        out[key] = np.array([b.mean() for b in selected])
        out[f"{key}_err"] = np.array([b.std() / np.sqrt(len(b)) for b in selected])
    return out


def plot_polarization_efficiency(outdir, polarizations, scores, model):
    """tilde beta_d vs. the fraction of most confident events, with the GNN down tagger and the
    W helicity angle probe at full efficiency."""
    curves = [beta_vs_fraction(p, s) for p, s in zip(polarizations, scores, strict=True)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, color, label in [("beta_l", COLORS[1], "L"), ("beta_r", COLORS[2], "R")]:
        mean, seed = mean_seed_std([c[key] for c in curves])
        err = np.hypot(seed, curves[0][f"{key}_err"])
        print(
            f"  beta_d ({label}) at fraction 1.0 / 0.5 / 0.2: "
            + " / ".join(
                f"{mean[i]:.4f} +- {seed[i]:.4f} (seed) +- {curves[0][f'{key}_err'][i]:.4f} (stat)"
                for i in (-1, 30, 0)
            )
        )
        ax.fill_between(FRACTIONS, mean - err, mean + err, color=color, alpha=0.3, lw=0)
        ax.plot(FRACTIONS, mean, color=color, lw=LINE_WIDTH)
    model_labels = [model]
    # the GNN curve is not distributed with the code
    if os.path.exists(GNN_EFFICIENCY):
        gnn = pd.read_csv(GNN_EFFICIENCY)
        for key, color in [("beta_opt_tL", COLORS[1]), ("beta_opt_tR", COLORS[2])]:
            ax.plot(gnn["efficiency"], gnn[key], color=color, ls="--", lw=LINE_WIDTH)
        model_labels.append("GNN")
    else:
        print(f"  {GNN_EFFICIENCY} not found, skipping the GNN curve")
    ax.plot([0.93, 1.0], [0.64, 0.64], color="#D95F02", lw=LINE_WIDTH)
    ax.text(
        0.92,
        0.64,
        r"$c_{W\text{hel}}$",
        ha="right",
        va="center",
        color="#D95F02",
        fontsize=FONTSIZE,
    )
    ax.set_xlim(0.2, 1.0)
    # headroom above beta = 1 for the legends
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Efficiency", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\tilde\beta_d$", fontsize=FONTSIZE)
    pols = [mlines.Line2D([], [], color=c) for c in COLORS[1:3]]
    models = [mlines.Line2D([], [], color="gray", ls=ls) for ls in ("-", "--")]
    legend(
        ax,
        pols + models[: len(model_labels)],
        [r"$P_{t_{\text{had}}} = L$", r"$P_{t_{\text{had}}} = R$", *model_labels],
        loc="upper right",
    )
    save(fig, ax, outdir, "polarization_efficiency")


if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("downtagging", {"exp_type": "downtagging"})
    scores = load_all(preds_dirs, "score.npz")
    plot_roc(outdir, preds_dirs, load_all(preds_dirs, "roc.npz"), scores, model)
    plot_calibration(outdir, scores, model)

    pol_dirs = [d + "_polarized" for d in preds_dirs]
    polarizations = load_all(pol_dirs, "polarization.npz")
    # the polarized evaluation writes no score.npz, labels.txt holds (labels_true, labels_predict)
    pol_scores = [np.loadtxt(os.path.join(d, "labels.txt")) for d in pol_dirs]
    pol_scores = [{"labels_true": s[:, 0], "labels_predict": s[:, 1]} for s in pol_scores]
    plot_polarization(outdir, polarizations)
    plot_polarization_efficiency(outdir, polarizations, pol_scores, model)
