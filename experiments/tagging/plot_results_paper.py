import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'font.family': 'serif',
    'ytick.right': True,
    'xtick.top': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'text.latex.preamble': r'\usepackage{amsmath}'
    })

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPATH = os.path.join(REPO_DIR, "runs")
OUTPATH = os.path.join(REPO_DIR, "runs", "paper_plots")

# reference beta vs. efficiency extracted from https://arxiv.org/abs/2407.01663
GNN_EFFICIENCY = os.path.join(REPO_DIR, "data", "beta_opt_GNN.csv")

COLORS = ["black", "blue", "red", "orange"]

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
    "vbf_pseudoscalar": "Score",
    "vbf_cp": "Score",
    "tt_smeft": "Score",
    "tt_smeft_posneg": r"$\mathcal{O}_{\text{L-GATr-slim}}$",
}

SCORE_XRANGES = {
    "downtagging": (0., 1.),
    "vbf_pseudoscalar": (-0.3, 0.3),
    "vbf_cp": (0.4, 0.6),
    "tt_smeft": (0.4, 0.6),
    "tt_smeft_posneg": (-0.3, 0.3),
}

# Remove if pseudoscalar observable before sigmoid is sufficient
SCORE_TRANSFORMS = {
    "tt_smeft_posneg": lambda score: 2 * score - 1,
}

ROC_AXIS_LABELS = {
    "downtagging": (r"\epsilon_d^{j_\text{hard}}", r"\epsilon_u^{j_\text{hard}}"),
}

SCORE_PROCESS_LABELS = {
    "tt_smeft_posneg": (r"$t\bar{t} \to \ell\ell$", r"$t\bar{t} \to q\ell$"),
}

ASYMMETRY_EXPS = {"tt_smeft", "tt_smeft_posneg"}


def latex_escape(text):
    return text.replace("_", r"\_")


def find_run_dir(run, run_idx):
    """Resolve `run` (a path relative to INPATH) to a directory containing preds_{run_idx}."""
    candidates = [os.path.join(INPATH, run)]
    candidates += sorted(glob.glob(os.path.join(INPATH, run)))
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
            f"Run '{run}' is ambiguous, matches multiple directories:\n"
            + "\n".join(seen)
        )
    return seen[0]


def load_npz(preds_dir, name):
    path = os.path.join(preds_dir, name)
    if not os.path.exists(path):
        return None
    print(path)
    return dict(np.load(path, allow_pickle=True))


def guess_model_name(preds_dir):
    leaf = os.path.basename(os.path.dirname(preds_dir))
    return leaf.split("_")[0]


# ------------------------------------------------------------------
# Training / validation loss
# ------------------------------------------------------------------

def _val_iters(n_train, n_val):
    if n_val == 0:
        return np.array([])
    frac = n_train / n_val
    return np.arange(1, n_val + 1) * frac


def plot_loss(outdir, training, title=None):
    train_loss = training["train_loss"]
    val_loss = training["val_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(np.arange(1, len(train_loss) + 1), train_loss, color=COLORS[1], label="Train loss")
    if len(val_loss) > 0:
        its = _val_iters(len(train_loss), len(val_loss))
        ax.plot(its, val_loss, color=COLORS[3], label="Val loss", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.margins(x=0)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.legend(frameon=False, loc="upper right")
    if title is not None:
        ax.text(
            0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes,
        )
    file = os.path.join(outdir, "loss.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def _plot_output_single(outdir, name, its, output_0, output_1, ylabel, label_0, label_1, title=None, std_0=None, std_1=None):
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
    proxy = ([mlines.Line2D([],[], ls="-", color=COLORS[2]),
                    mlines.Line2D([],[], ls="-", color=COLORS[1]),
                    ])
    labels = [label_1, label_0]
    ax.legend(proxy, labels, frameon=False, loc='best')
    if title is not None:
        ax.text(
            0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes,
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
            outdir, "output_scalar", its,
            training["scalar_output_0"], training["scalar_output_1"],
            ylabel="Scalar output (val)", label_0=label_0, label_1=label_1, title=title,
            std_0=training.get("scalar_output_0_std"), std_1=training.get("scalar_output_1_std"),
        )
    if "pseudoscalar_output_0" in training:
        _plot_output_single(
            outdir, "output_pseudo", its,
            training["pseudoscalar_output_0"], training["pseudoscalar_output_1"],
            ylabel="Pseudoscalar output (val)", label_0=label_0, label_1=label_1, title=title,
            std_0=training.get("pseudoscalar_output_0_std"), std_1=training.get("pseudoscalar_output_1_std"),
        )


# ------------------------------------------------------------------
# ROC curve(s)
# ------------------------------------------------------------------

def _plot_roc_single(outdir, name, fpr, tpr, auc_val, title=None, tpr_label=r"\epsilon_S", fpr_label=r"\epsilon_B"):
    rnd = np.linspace(1e-3, 1, 100)

    # standard ROC: eps_S vs eps_B
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rnd, rnd, "k--", lw=2)
    ax.plot(fpr, tpr, color="red", label=f"AUC = {auc_val:.4f}", lw=2)
    ax.set_xlabel(f"${fpr_label}$")
    ax.set_ylabel(f"${tpr_label}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    if title is not None:
        ax.set_title(title)
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")

    # physics convention: eps_S vs 1/eps_B (log)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_yscale("log")
    ax.plot(rnd, 1 / rnd, "k--", lw=2)
    mask = fpr > 0
    ax.plot(tpr[mask], 1 / fpr[mask], color="red", label=f"AUC = {auc_val:.4f}", lw=2)
    ax.set_xlabel(f"${tpr_label}$")
    ax.set_ylabel(f"$1 / {fpr_label}$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    if title is not None:
        ax.set_title(title)
    file = os.path.join(outdir, f"{name}_physics.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_roc(outdir, roc, title=None, exp=None):
    # strip any surrounding $ so the labels can be embedded in larger math expressions
    tpr_label, fpr_label = (lbl.strip("$") for lbl in ROC_AXIS_LABELS.get(exp, (r"\epsilon_S", r"\epsilon_B")))
    channels = [("", "roc", "Combined")]
    if "fpr_scalar" in roc:
        channels.append(("_scalar", "roc_scalar", "Scalar"))
    if "fpr_pseudo" in roc:
        channels.append(("_pseudo", "roc_pseudo", "Pseudoscalar"))

    for suffix, name, label in channels:
        fpr, tpr, auc_val = roc[f"fpr{suffix}"], roc[f"tpr{suffix}"], float(roc[f"auc{suffix}"])
        chan_title = f"{label} channel"
        _plot_roc_single(outdir, name, fpr, tpr, auc_val, title=chan_title, tpr_label=tpr_label, fpr_label=fpr_label)


# ------------------------------------------------------------------
# Score distributions
# ------------------------------------------------------------------

def asym(counts):
    assert len(counts) % 2 == 0, "Number of bins must be even to compute asymmetry"
    n_pos = np.sum(counts[len(counts) // 2:])
    n_neg = np.sum(counts[:len(counts) // 2])
    return (n_pos - n_neg) / (n_pos + n_neg)


def err_asym(counts):
    assert len(counts) % 2 == 0, "Number of bins must be even to compute asymmetry"
    n_pos = np.sum(counts[len(counts) // 2:])
    n_neg = np.sum(counts[:len(counts) // 2])
    return 1 / (n_pos + n_neg)**2 * np.sqrt(2 * n_pos**2 * n_neg + 2 * n_pos * n_neg**2)


def score_asymmetry(scores, center=0.5):
    counts = np.array([np.sum(scores < center), np.sum(scores >= center)])
    return asym(counts), err_asym(counts)


def _plot_score_single(outdir, name, labels_true, labels_predict, label_0, label_1, downtag=False, title=None, logy=False, xrange=(0.4, 0.6), xlabel="Score", exp=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if logy:
        ax.set_yscale("log")
    if exp in SCORE_TRANSFORMS:
        labels_predict = SCORE_TRANSFORMS[exp](labels_predict)
    scores_0 = labels_predict[labels_true == 0]
    scores_1 = labels_predict[labels_true == 1]
    _, bin_edges, _ = ax.hist(
        scores_0, range=xrange, bins=20,
        histtype="step", weights=np.ones_like(scores_0) / len(scores_0),
        color=COLORS[1], label=label_0, lw=2,
    )
    ax.hist(
        scores_1, bins=bin_edges,
        histtype="step", weights=np.ones_like(scores_1) / len(scores_1),
        color=COLORS[2], label=label_1, lw=2,
    )
    ax.set_xlim(xrange)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized")
    proxy = ([mlines.Line2D([],[], ls="-", color=COLORS[1]),
                mlines.Line2D([],[], ls="-", color=COLORS[2]),
                ])
    labels = [label_0, label_1]
    ax.legend(proxy, labels, frameon=False, loc='upper center')
    if title is not None:
        ax.text(0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes)
    if exp in SCORE_PROCESS_LABELS:
        process_label = SCORE_PROCESS_LABELS[exp][1 if downtag else 0]
        # move below the title if both are shown
        ypos = 0.95 if title is None else 0.85
        ax.text(0.95, ypos, s=process_label, ha="right", va="top", transform=ax.transAxes)
    file = os.path.join(outdir, f"{name}{'_logy' if logy else ''}{'_downtag' if downtag else ''}.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


def plot_scores(outdir, score, label_0, label_1, title=None, exp=None):
    labels_true = score["labels_true"]
    xlabel = SCORE_XLABELS.get(exp, "Score")
    xrange = SCORE_XRANGES.get(exp, (0.4, 0.6))

    channels = [("labels_predict", "score", "")]
    if "labels_predict_scalar" in score:
        channels.append(("labels_predict_scalar", "score_scalar", " (scalar)"))
    if "labels_predict_pseudo" in score:
        channels.append(("labels_predict_pseudo", "score_pseudo", " (pseudoscalar)"))

    for key, name, suffix in channels:
        chan_title = None if title is None else f"{title}{suffix}"
        _plot_score_single(outdir, name, labels_true, score[key], label_0, label_1, title=chan_title, logy=False, xrange=xrange, xlabel=xlabel, exp=exp)
        _plot_score_single(outdir, name, labels_true, score[key], label_0, label_1, title=chan_title, logy=True, xrange=xrange, xlabel=xlabel, exp=exp)

        if exp in ASYMMETRY_EXPS:
            labels_predict = score[key]
            scores_0 = labels_predict[labels_true == 0]
            scores_1 = labels_predict[labels_true == 1]
            asym_0, err_0 = score_asymmetry(scores_0)
            asym_1, err_1 = score_asymmetry(scores_1)
            print(f"  asymmetry ({name}{suffix}) {label_0}: {asym_0:.4f} ± {err_0:.4f}")
            print(f"  asymmetry ({name}{suffix}) {label_1}: {asym_1:.4f} ± {err_1:.4f}")


def plot_scores_downtag(outdir, score, label_0, label_1, title=None, exp=None):
    labels_true = score["labels_true"]
    xlabel = SCORE_XLABELS.get(exp, "Score")
    xrange = SCORE_XRANGES.get(exp, (0.4, 0.6))

    channels = [("labels_predict", "score", "")]
    if "labels_predict_scalar" in score:
        channels.append(("labels_predict_scalar", "score_scalar", " (scalar)"))
    if "labels_predict_pseudo" in score:
        channels.append(("labels_predict_pseudo", "score_pseudo", " (pseudoscalar)"))

    for key, name, suffix in channels:
        chan_title = None if title is None else f"{title}{suffix}"
        _plot_score_single(outdir, name, labels_true, score[key], label_0, label_1, downtag=True, title=chan_title, logy=False, xrange=xrange, xlabel=xlabel, exp=exp)
        _plot_score_single(outdir, name, labels_true, score[key], label_0, label_1, downtag=True, title=chan_title, logy=True, xrange=xrange, xlabel=xlabel, exp=exp)

        if exp in ASYMMETRY_EXPS:
            labels_predict = score[key]
            scores_0 = labels_predict[labels_true == 0]
            scores_1 = labels_predict[labels_true == 1]
            asym_0, err_0 = score_asymmetry(scores_0)
            asym_1, err_1 = score_asymmetry(scores_1)
            print(f"  asymmetry ({name}{suffix}) {label_0}: {asym_0:.4f} ± {err_0:.4f}")
            print(f"  asymmetry ({name}{suffix}) {label_1}: {asym_1:.4f} ± {err_1:.4f}")


# ------------------------------------------------------------------
# Experiment-specific extras: polarization / efficiency plots
# ------------------------------------------------------------------


def plot_polarization_extra(outdir, polarization, label_0=None, label_1=None, title=None, exp=None):
    score_l, score_r = polarization["score_l"], polarization["score_r"]
    efficiencies, pol_l, pol_r = (
        polarization["efficiencies"], polarization["pol_l"], polarization["pol_r"],
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xrange = (-1, 1)
    ax.hist(
        score_l, range=xrange, bins=20, histtype="step", color=COLORS[1],
        weights=np.ones_like(score_l) / len(score_l), lw=2,
    )
    ax.hist(
        score_r, range=xrange, bins=20, histtype="step", color=COLORS[2],
        weights=np.ones_like(score_r) / len(score_r), lw=2,
    )
    ax.set_xlim(xrange)
    ax.set_xlabel(r"$\cos \theta_{d}$")
    ax.set_ylabel("Normalized")
    proxy = ([mlines.Line2D([],[], ls="-", color=COLORS[1]),
            mlines.Line2D([],[], ls="-", color=COLORS[2]),
            ])
    labels = [r'$P_{t_{\text{had}}} = L$', r'$P_{t_{\text{had}}} = R$']
    ax.legend(proxy, labels, frameon=False, loc='lower center')
    if title is not None:
        ax.text(0.95, 0.95, s=title, ha="right", va="top", transform=ax.transAxes)
    file = os.path.join(outdir, "polarization.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")

    # efficiency plot: optimal polarization sensitivity vs. selection efficiency
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(efficiencies, pol_l, color=COLORS[1], label=r'$P_{t_{\text{had}}} = L$ (L-GATr-slim)', lw=2)
    ax.plot(efficiencies, pol_r, color=COLORS[2], label=r'$P_{t_{\text{had}}} = R$ (L-GATr-slim)', lw=2)
    if os.path.exists(GNN_EFFICIENCY):
        gnn = pd.read_csv(GNN_EFFICIENCY)
        ax.plot(gnn["efficiency"], gnn["beta_opt_tL"], color=COLORS[1], ls="--", label=r'$P_{t_{\text{had}}} = L$ (GNN)', lw=2)
        ax.plot(gnn["efficiency"], gnn["beta_opt_tR"], color=COLORS[2], ls="--", label=r'$P_{t_{\text{had}}} = R$ (GNN)', lw=2)
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


EXTRA_PLOTTERS = {
    "downtagging": ("_polarized", "polarization.npz", plot_polarization_extra),
    "tt_smeft": ("_downtag", "score.npz", plot_scores_downtag),
    "tt_smeft_posneg": ("_downtag", "score.npz", plot_scores_downtag),
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--exp", required=True, choices=EXP_CHOICES,
        help="Experiment type, used to label plots and select experiment-specific extras.",
    )
    parser.add_argument(
        "--run", required=True,
        help="Path of the run relative to "f"{INPATH}. "
             "May also be a glob pattern, as long as it resolves to a single run.",
    )
    parser.add_argument("--run-idx", type=int, default=0, help="Index N of the preds_N directory.")
    parser.add_argument("--outdir", default=OUTPATH, help="Output directory for the plots.")
    parser.add_argument("--title", default=None, help="Override the plot title.")
    args = parser.parse_args()

    preds_dir = find_run_dir(args.run, args.run_idx)
    print(f"Reading predictions from {preds_dir}")

    modelname = guess_model_name(preds_dir)
    title = args.title if args.title is not None else None

    run_label = os.path.basename(os.path.dirname(preds_dir))
    outdir = os.path.join(args.outdir, args.exp, run_label)
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving plots to {outdir}")

    label_0, label_1 = SCORE_LABELS[args.exp]

    print(preds_dir)
    training = load_npz(preds_dir, "training.npz")
    if training is not None:
        plot_loss(outdir, training, title=title)
        plot_training_outputs(outdir, training, label_0=label_0, label_1=label_1, title=title)
    else:
        print("  training.npz not found, skipping loss plot")

    roc = load_npz(preds_dir, "roc.npz")
    if roc is not None:
        plot_roc(outdir, roc, title=title, exp=args.exp)
    else:
        print("  roc.npz not found, skipping ROC plot")

    score = load_npz(preds_dir, "score.npz")
    if score is not None:
        plot_scores(outdir, score, label_0, label_1, title=title, exp=args.exp)
    else:
        print("  score.npz not found, skipping score plot")

    if args.exp in EXTRA_PLOTTERS:
        extra_path, extra_file, extra_fn = EXTRA_PLOTTERS[args.exp]
        extra = load_npz(preds_dir+extra_path, extra_file)
        if extra is not None:
            extra_fn(outdir, extra, label_0=label_0, label_1=label_1, title=title, exp=args.exp)
        else:
            print(f"  {extra_file} not found, skipping {args.exp} extras")


if __name__ == "__main__":
    main()
