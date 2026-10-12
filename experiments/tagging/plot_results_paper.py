import argparse
import glob
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.tagging.asymmetry import asymmetry

plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
        "font.family": "serif",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPATH = os.path.join(REPO_DIR, "runs")
OUTPATH = os.path.join(REPO_DIR, "runs", "paper_plots")

# reference beta vs. efficiency extracted from https://arxiv.org/abs/2407.01663
GNN_EFFICIENCY = os.path.join(REPO_DIR, "data", "beta_opt_GNN.csv")

COLORS = ["black", "blue", "red", "orange"]

# network name in the axis and legend labels, set with --model
MODEL = "L-GATr-slim"

# Uncertainties: "seed" is the std over the training seeds (all evaluated on the same test set),
# "stat" the test-set statistical error (bootstrap, Poisson or binomial, from the first seed).
# Bands and legend values show both added in quadrature.
N_BOOTSTRAP = 50

SCORE_LABELS = {
    "downtagging": (r"$d = q_{\text{soft}}$", r"$d = q_{\text{hard}}$"),
    "vbf_pseudoscalar": (r"$\text{Neg}$", r"$\text{Pos}$"),
    "vbf_cp": (r"$\text{SM}$", r"$\text{Pos}$"),
    "tt_smeft": (r"$\mathrm{Im}(d_{tG}) = 0$", r"$\mathrm{Im}(d_{tG}) = 2.6$"),
    "tt_smeft_posneg": (r"$\mathrm{Im}(d_{tG}) = -2.6$", r"$\mathrm{Im}(d_{tG}) = 2.6$"),
}

EXP_CHOICES = list(SCORE_LABELS.keys())

SCORE_XLABELS = {
    "downtagging": r"$p(d = q_{\text{hard}})$",
    "vbf_pseudoscalar": r"$\mathcal{O}_{\text{L-GATr-slim}}$",
    "vbf_cp": "Score",
    "tt_smeft": "Score",
    "tt_smeft_posneg": r"$\mathcal{O}_{\text{L-GATr-slim}}$",
}

SCORE_XRANGES = {
    "downtagging": (0.0, 1.0),
    "vbf_pseudoscalar": (-3.5, 3.5),
    "vbf_cp": (0.0, 1.0),
    "tt_smeft": (0.35, 0.85),
    "tt_smeft_posneg": (-0.5, 0.5),
}


def logit(score):
    # network output before the sigmoid; the scores are float32, exact for |logit| < ~15
    score = np.clip(score.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(score / (1 - score))


# pseudoscalar tasks: show the network output (the C-odd observable), not the sigmoid
SCORE_TRANSFORMS = {
    "vbf_pseudoscalar": logit,
    "tt_smeft_posneg": logit,
}

ROC_AXIS_LABELS = {
    "downtagging": (r"\epsilon_d^{q_\text{hard}}", r"\epsilon_u^{q_\text{hard}}"),
}

SCORE_PROCESS_LABELS = {
    "tt_smeft_posneg": (r"$t\bar{t} \to \ell\ell$", r"$t\bar{t} \to q\ell$"),
}

ASYMMETRY_EXPS = {"tt_smeft", "tt_smeft_posneg"}


def latex_escape(text):
    return text.replace("_", r"\_")


def find_run_dir(run, run_idx):
    """Resolve `run` (a path relative to INPATH) to a directory containing preds_{run_idx}."""
    if os.path.isdir(os.path.join(INPATH, run, f"preds_{run_idx}")):
        return os.path.join(INPATH, run, f"preds_{run_idx}")
    candidates = sorted(glob.glob(os.path.join(INPATH, run)))
    candidates += sorted(glob.glob(os.path.join(INPATH, run + "*")))

    seen = []
    for c in candidates:
        preds_dir = os.path.join(c, f"preds_{run_idx}")
        if os.path.isdir(preds_dir) and preds_dir not in seen:
            seen.append(preds_dir)

    if len(seen) == 0:
        raise FileNotFoundError(
            f"Could not find a 'preds_{run_idx}' directory for run '{run}' under {INPATH}"
        )
    if len(seen) > 1:
        raise ValueError(
            f"Run '{run}' is ambiguous, matches multiple directories:\n" + "\n".join(seen)
        )
    return seen[0]


def load_npz(preds_dir, name):
    path = os.path.join(preds_dir, name)
    if not os.path.exists(path):
        return None
    print(path)
    return dict(np.load(path, allow_pickle=True))


def load_all(preds_dirs, name):
    """Load `name` from every seed; None if any seed lacks it."""
    out = [load_npz(d, name) for d in preds_dirs]
    return None if any(o is None for o in out) else out


def mean_seed_std(values):
    values = np.stack(values)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), std


def pm_text(mean, seed, stat, fmt=".4f"):
    return rf"{mean:{fmt}} \pm {np.hypot(seed, stat):{fmt}}"


def guess_model_name(preds_dir):
    leaf = os.path.basename(os.path.dirname(preds_dir))
    return leaf.split("_")[0]


# ------------------------------------------------------------------
# Training / validation loss
# ------------------------------------------------------------------


def _val_iters(n_train, n_val):
    if n_val == 0:
        return np.array([])
    # first validation runs before training (step 0), then every n_train / (n_val - 1) steps
    return np.arange(n_val) * n_train / max(n_val - 1, 1)


def plot_loss(outdir, training, title=None):
    train_loss = training["train_loss"]
    val_loss = training["val_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # running mean, the per-iteration loss is too noisy to read
    window = max(1, len(train_loss) // 200)
    smooth = np.convolve(train_loss, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(train_loss) + 1), smooth, color=COLORS[1], label="Train loss")
    if len(val_loss) > 0:
        its = _val_iters(len(train_loss), len(val_loss))
        ax.plot(its, val_loss, color=COLORS[3], label="Val loss", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.margins(x=0)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=12)
    )
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.legend(frameon=False, loc="upper right")
    if title is not None:
        ax.text(
            0.95,
            0.95,
            s=title,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
    file = os.path.join(outdir, "loss.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def _plot_output_single(
    outdir,
    name,
    its,
    output_0,
    output_1,
    ylabel,
    label_0,
    label_1,
    title=None,
    std_0=None,
    std_1=None,
):
    # centered running mean like the loss curve; the window shrinks at the edges so every
    # validation point is kept (only 11 to 101 points per run); the step-0 point (untrained
    # network) is left out so its offset does not leak into the trained curve
    window = max(3, len(its) // 10) | 1

    def smooth(x):
        kernel = np.ones(window)
        y = np.convolve(x[1:], kernel, "same") / np.convolve(np.ones(len(x) - 1), kernel, "same")
        return np.concatenate([x[:1], y])

    output_0, output_1 = smooth(output_0), smooth(output_1)
    std_0 = None if std_0 is None else smooth(std_0)
    std_1 = None if std_1 is None else smooth(std_1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.axhline(0, color="gray", ls="--", lw=1)
    if std_0 is not None:
        ax.fill_between(its, output_0 - std_0, output_0 + std_0, color=COLORS[1], alpha=0.2, lw=0)
    if std_1 is not None:
        ax.fill_between(its, output_1 - std_1, output_1 + std_1, color=COLORS[2], alpha=0.2, lw=0)
    ax.plot(its, output_0, color=COLORS[1], label=label_0, lw=2)
    ax.plot(its, output_1, color=COLORS[2], label=label_1, lw=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    ax.set_xlim(left=0)
    proxy = [
        mlines.Line2D([], [], ls="-", color=COLORS[2]),
        mlines.Line2D([], [], ls="-", color=COLORS[1]),
    ]
    labels = [label_1, label_0]
    ax.legend(proxy, labels, frameon=False, loc="best")
    if title is not None:
        ax.text(
            0.95,
            0.95,
            s=title,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_training_outputs(outdir, training, label_0, label_1, title=None):
    n_train = len(training["train_loss"])
    its = _val_iters(n_train, len(training.get("val_loss", [])))

    if "scalar_output_0" in training:
        _plot_output_single(
            outdir,
            "output_scalar",
            its,
            training["scalar_output_0"],
            training["scalar_output_1"],
            ylabel="Scalar output (val)",
            label_0=label_0,
            label_1=label_1,
            title=title,
            std_0=training.get("scalar_output_0_std"),
            std_1=training.get("scalar_output_1_std"),
        )
    if "pseudoscalar_output_0" in training:
        _plot_output_single(
            outdir,
            "output_pseudo",
            its,
            training["pseudoscalar_output_0"],
            training["pseudoscalar_output_1"],
            ylabel="Pseudoscalar output (val)",
            label_0=label_0,
            label_1=label_1,
            title=title,
            std_0=training.get("pseudoscalar_output_0_std"),
            std_1=training.get("pseudoscalar_output_1_std"),
        )


# ------------------------------------------------------------------
# ROC curve(s)
# ------------------------------------------------------------------


def bootstrap_auc(labels_true, labels_predict, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels_true)
    aucs = [
        roc_auc_score(labels_true[i], labels_predict[i])
        for i in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))
    ]
    return np.std(aucs)


def _plot_roc_single(
    outdir, name, fpr, tpr, auc_label, title=None, tpr_label=r"\epsilon_S", fpr_label=r"\epsilon_B"
):
    rnd = np.linspace(1e-3, 1, 100)

    # standard ROC: eps_S vs eps_B
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rnd, rnd, "k--", lw=2)
    ax.plot(fpr, tpr, color="red", label=auc_label, lw=2)
    ax.set_xlabel(f"${fpr_label}$")
    ax.set_ylabel(f"${tpr_label}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    if title is not None:
        ax.text(0.05, 0.95, s=title, ha="left", va="top", transform=ax.transAxes)
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")

    # physics convention: eps_S vs 1/eps_B (log)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_yscale("log")
    ax.plot(rnd, 1 / rnd, "k--", lw=2)
    mask = fpr > 0
    ax.plot(tpr[mask], 1 / fpr[mask], color="red", label=auc_label, lw=2)
    ax.set_xlabel(f"${tpr_label}$")
    ax.set_ylabel(f"$1 / {fpr_label}$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=13)
    if title is not None:
        ax.text(0.05, 0.05, s=title, ha="left", va="bottom", transform=ax.transAxes)
    file = os.path.join(outdir, f"{name}_physics.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_roc(outdir, rocs, scores, title=None, exp=None):
    # curve of the first seed (the seed spread is invisible), AUC with seed and bootstrap errors
    # strip any surrounding $ so the labels can be embedded in larger math expressions
    tpr_label, fpr_label = (
        lbl.strip("$") for lbl in ROC_AXIS_LABELS.get(exp, (r"\epsilon_S", r"\epsilon_B"))
    )
    channels = [("", "roc", None, "labels_predict")]
    if "fpr_scalar" in rocs[0]:
        channels.append(("_scalar", "roc_scalar", "Scalar channel", "labels_predict_scalar"))
    if "fpr_pseudo" in rocs[0]:
        channels.append(("_pseudo", "roc_pseudo", "Pseudoscalar channel", "labels_predict_pseudo"))

    for suffix, name, label, key in channels:
        aucs = [float(roc[f"auc{suffix}"]) for roc in rocs]
        stat = bootstrap_auc(scores[0]["labels_true"], scores[0][key])
        mean, seed = mean_seed_std(aucs)
        print(
            f"  AUC ({name}) over {len(aucs)} runs: {mean:.4f} +- {seed:.4f} (seed) +- {stat:.4f} (stat)"
        )
        auc_label = f"AUC $= {pm_text(mean, seed, stat)}$"
        roc = rocs[0]
        _plot_roc_single(
            outdir,
            name,
            roc[f"fpr{suffix}"],
            roc[f"tpr{suffix}"],
            auc_label,
            title=label,
            tpr_label=tpr_label,
            fpr_label=fpr_label,
        )


# ------------------------------------------------------------------
# Score distributions
# ------------------------------------------------------------------


def hist_band(ax, samples, bins, color, label=None):
    """Mean over seeds of the histograms normalized to the number of events, band: seed std
    and Poisson error in quadrature."""
    hists, stats = [], []
    for x in samples:
        counts, _ = np.histogram(x, bins=bins)
        hists.append(counts / len(x))
        stats.append(np.sqrt(counts) / len(x))
    mean, seed = mean_seed_std(hists)
    err = np.hypot(seed, stats[0])
    ax.stairs(mean, bins, color=color, label=label, lw=2)
    ax.stairs(mean + err, bins, baseline=mean - err, fill=True, color=color, alpha=0.3, lw=0)
    return mean + err


def _plot_score_single(
    outdir,
    name,
    scores_0,
    scores_1,
    label_0,
    label_1,
    downtag=False,
    title=None,
    logy=False,
    xrange=(0.4, 0.6),
    xlabel="Score",
    exp=None,
):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if logy:
        ax.set_yscale("log")
    bins = np.linspace(*xrange, 31)
    top_0 = hist_band(ax, scores_0, bins, COLORS[1])
    top_1 = hist_band(ax, scores_1, bins, COLORS[2])
    ax.set_xlim(xrange)
    # room for the legend and labels
    top = max(top_0.max(), top_1.max())
    if logy:
        ax.set_ylim(top=30 * top)
    else:
        ax.set_ylim(0, 1.35 * top)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized")
    proxy = [
        mlines.Line2D([], [], ls="-", color=COLORS[1]),
        mlines.Line2D([], [], ls="-", color=COLORS[2]),
    ]
    labels = [label_0, label_1]
    ax.legend(proxy, labels, frameon=False, loc="upper left", fontsize=13)
    if title is not None:
        ax.text(0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes)
    if exp in SCORE_PROCESS_LABELS:
        process_label = SCORE_PROCESS_LABELS[exp][1 if downtag else 0]
        # move below the title if both are shown
        ypos = 0.95 if title is None else 0.85
        ax.text(0.95, ypos, s=process_label, ha="right", va="top", transform=ax.transAxes)
    file = os.path.join(
        outdir, f"{name}{'_logy' if logy else ''}{'_downtag' if downtag else ''}.pdf"
    )
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_scores(outdir, scores, label_0, label_1, title=None, exp=None, downtag=False):
    xlabel = SCORE_XLABELS.get(exp, "Score")
    xrange = SCORE_XRANGES.get(exp, (0.4, 0.6))
    transform = SCORE_TRANSFORMS.get(exp, lambda score: score)

    channels = [("labels_predict", "score", "")]
    if "labels_predict_scalar" in scores[0]:
        channels.append(("labels_predict_scalar", "score_scalar", " (scalar)"))
    if "labels_predict_pseudo" in scores[0]:
        channels.append(("labels_predict_pseudo", "score_pseudo", " (pseudoscalar)"))

    for key, name, suffix in channels:
        chan_title = None if title is None else f"{title}{suffix}"
        scores_0 = [transform(s[key][s["labels_true"] == 0]) for s in scores]
        scores_1 = [transform(s[key][s["labels_true"] == 1]) for s in scores]
        for logy in (False, True):
            _plot_score_single(
                outdir,
                name,
                scores_0,
                scores_1,
                label_0,
                label_1,
                downtag=downtag,
                title=chan_title,
                logy=logy,
                xrange=xrange,
                xlabel=xlabel,
                exp=exp,
            )

        if exp in ASYMMETRY_EXPS:
            for label, samples in [(label_0, scores_0), (label_1, scores_1)]:
                center = 0.5 if exp not in SCORE_TRANSFORMS else 0.0
                asyms = [asymmetry(x - center)[0] for x in samples]
                stat = asymmetry(samples[0] - center)[1]
                mean, seed = mean_seed_std(asyms)
                print(
                    f"  asymmetry ({name}{suffix}{', downtag' if downtag else ''}) {label}: "
                    f"{mean:.4f} +- {seed:.4f} (seed) +- {stat:.4f} (stat)"
                )


# ------------------------------------------------------------------
# Calibration (reliability diagram)
# ------------------------------------------------------------------


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


def plot_calibration(outdir, scores, xlabel, label_1, title=None):
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

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.fill_between(x, y - dy, y + dy, color=COLORS[2], alpha=0.3, lw=0)
    ax.plot(x, y, color=COLORS[2], lw=2, marker="o", ms=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Fraction of {label_1}")
    ax.text(
        0.05,
        0.95,
        s=f"ECE $= {pm_text(ece, ece_seed, ece_stat)}$",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=13,
    )
    if title is not None:
        ax.text(0.95, 0.05, s=title, ha="right", va="bottom", transform=ax.transAxes)
    file = os.path.join(outdir, "calibration.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


# ------------------------------------------------------------------
# Experiment-specific extras: polarization / efficiency plots
# ------------------------------------------------------------------

FRACTIONS = np.linspace(0.2, 1.0, 81)


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


def plot_polarization_extra(outdir, preds_dirs, label_0=None, label_1=None, title=None, exp=None):
    polarizations = load_all(preds_dirs, "polarization.npz")
    if polarizations is None:
        print("  polarization.npz missing, skipping downtagging extras")
        return
    # the polarized evaluation writes no score.npz, labels.txt holds (labels_true, labels_predict)
    scores = [np.loadtxt(os.path.join(d, "labels.txt")) for d in preds_dirs]
    scores = [{"labels_true": s[:, 0], "labels_predict": s[:, 1]} for s in scores]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bins = np.linspace(-1, 1, 21)
    top_l = hist_band(ax, [p["score_l"] for p in polarizations], bins, COLORS[1])
    top_r = hist_band(ax, [p["score_r"] for p in polarizations], bins, COLORS[2])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.15 * max(top_l.max(), top_r.max()))
    ax.set_xlabel(r"$\cos \theta_{d}$")
    ax.set_ylabel("Normalized")
    proxy = [
        mlines.Line2D([], [], ls="-", color=COLORS[1]),
        mlines.Line2D([], [], ls="-", color=COLORS[2]),
    ]
    labels = [r"$P_{t_{\text{had}}} = L$", r"$P_{t_{\text{had}}} = R$"]
    ax.legend(proxy, labels, frameon=False, loc="lower center")
    if title is not None:
        ax.text(0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes)
    file = os.path.join(outdir, "polarization.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")

    if "beta_l" not in polarizations[0]:
        print("  polarization.npz has no per-event beta (old evaluation), skipping efficiency plot")
        return
    curves = [beta_vs_fraction(p, s) for p, s in zip(polarizations, scores, strict=True)]

    # optimal polarization sensitivity vs. selection efficiency
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
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
        ax.plot(
            FRACTIONS,
            mean,
            color=color,
            label=rf"$P_{{t_{{\text{{had}}}}}} = {label}$ ({MODEL})",
            lw=2,
        )
    if os.path.exists(GNN_EFFICIENCY):
        gnn = pd.read_csv(GNN_EFFICIENCY)
        ax.plot(
            gnn["efficiency"],
            gnn["beta_opt_tL"],
            color=COLORS[1],
            ls="--",
            label=r"$P_{t_{\text{had}}} = L$ (GNN)",
            lw=2,
        )
        ax.plot(
            gnn["efficiency"],
            gnn["beta_opt_tR"],
            color=COLORS[2],
            ls="--",
            label=r"$P_{t_{\text{had}}} = R$ (GNN)",
            lw=2,
        )
    else:
        print(f"  {GNN_EFFICIENCY} not found, skipping GNN curves")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Fraction of most confident events")
    ax.set_ylabel(r"$\beta_{d}$")
    ax.legend(frameon=False, loc="upper right", fontsize=14)
    if title is not None:
        ax.text(0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes)
    file = os.path.join(outdir, "polarization_efficiency.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_scores_downtag(outdir, preds_dirs, label_0, label_1, title=None, exp=None):
    scores = load_all(preds_dirs, "score.npz")
    if scores is None:
        print("  score.npz missing in a _downtag directory, skipping downtag scores")
        return
    plot_scores(outdir, scores, label_0, label_1, title=title, exp=exp, downtag=True)


EXTRA_PLOTTERS = {
    "downtagging": ("_polarized", plot_polarization_extra),
    "tt_smeft": ("_downtag", plot_scores_downtag),
    "tt_smeft_posneg": ("_downtag", plot_scores_downtag),
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        choices=EXP_CHOICES,
        help="Experiment type, used to label plots and select experiment-specific extras.",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path of the run relative to "
        f"{INPATH}. "
        "May also be a glob pattern, as long as it resolves to a single run.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Glob relative to "
        f"{INPATH} matching the runs of all seeds (e.g. paper_3/down_slim_s*), "
        "for the seed uncertainties. --run is the first seed, used for the loss and stat errors.",
    )
    parser.add_argument("--run-idx", type=int, default=0, help="Index N of the preds_N directory.")
    parser.add_argument("--outdir", default=OUTPATH, help="Output directory for the plots.")
    parser.add_argument("--title", default=None, help="Override the plot title.")
    parser.add_argument("--model", default=MODEL, help="Network name in the plot labels.")
    args = parser.parse_args()

    for exp, xlabel in SCORE_XLABELS.items():
        SCORE_XLABELS[exp] = xlabel.replace(MODEL, args.model)
    MODEL = args.model

    preds_dir = find_run_dir(args.run, args.run_idx)
    print(f"Reading predictions from {preds_dir}")
    preds_dirs = [preds_dir]
    if args.seeds is not None:
        others = sorted(glob.glob(os.path.join(INPATH, args.seeds, f"preds_{args.run_idx}")))
        preds_dirs += [d for d in others if os.path.realpath(d) != os.path.realpath(preds_dir)]
    print(f"Seeds: {[os.path.basename(os.path.dirname(d)) for d in preds_dirs]}")

    title = args.title if args.title is not None else None

    run_label = os.path.basename(os.path.dirname(preds_dir))
    outdir = os.path.join(args.outdir, args.exp, run_label)
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving plots to {outdir}")

    label_0, label_1 = SCORE_LABELS[args.exp]

    training = load_npz(preds_dir, "training.npz")
    if training is not None:
        plot_loss(outdir, training, title=title)
        plot_training_outputs(outdir, training, label_0=label_0, label_1=label_1, title=title)
    else:
        print("  training.npz not found, skipping loss plot")

    scores = load_all(preds_dirs, "score.npz")
    rocs = load_all(preds_dirs, "roc.npz")
    if rocs is not None and scores is not None:
        plot_roc(outdir, rocs, scores, title=title, exp=args.exp)
    else:
        print("  roc.npz or score.npz not found, skipping ROC plot")

    if scores is not None:
        plot_scores(outdir, scores, label_0, label_1, title=title, exp=args.exp)
        if args.exp == "downtagging":
            plot_calibration(outdir, scores, SCORE_XLABELS[args.exp], label_1, title=title)
    else:
        print("  score.npz not found, skipping score plot")

    if args.exp in EXTRA_PLOTTERS:
        extra_path, extra_fn = EXTRA_PLOTTERS[args.exp]
        extra_dirs = [d + extra_path for d in preds_dirs]
        missing = [d for d in extra_dirs if not os.path.isdir(d)]
        if missing:
            print(f"  missing {missing}, extras use the other seeds")
            extra_dirs = [d for d in extra_dirs if os.path.isdir(d)]
        if extra_dirs:
            extra_fn(
                outdir, extra_dirs, label_0=label_0, label_1=label_1, title=title, exp=args.exp
            )


if __name__ == "__main__":
    main()
