import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from omegaconf import OmegaConf

plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Charter",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.pad": 8,
        "ytick.major.pad": 6,
    }
)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
    r"\DeclareSymbolFont{usualmathcal}{OMS}{cmsy}{m}{n}"
    r"\DeclareSymbolFontAlphabet{\mathcal}{usualmathcal}"
)

FONTSIZE = 20
FONTSIZE_LEGEND = 20
FONTSIZE_TICK = 18
LINE_WIDTH = 1.8
FIGSIZE = (5.5, 4.5)
MARGINS = {"left": 0.99, "bottom": 0.84, "right": 0.25, "top": 0.14}
LEFT, BOTTOM = MARGINS["left"] / FIGSIZE[0], MARGINS["bottom"] / FIGSIZE[1]
RIGHT, TOP = 1 - MARGINS["right"] / FIGSIZE[0], 1 - MARGINS["top"] / FIGSIZE[1]

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPATH = os.path.join(REPO_DIR, "paper_plots")

COLORS = ["black", "blue", "red", "orange"]

MODEL_NAMES = {
    "LGATrSlimPseudoWrapper": "L-GATr",
    "LGATrSlimWrapper": "L-GATr",
    "LGATrWrapper": "L-GATr",
    "TransformerWrapper": "Transformer",
}

# Uncertainties: "seed" is the std over the training seeds (all evaluated on the same test set),
# "stat" the test-set statistical error (bootstrap, Poisson or binomial, from the first seed).
# Bands and legend values show both added in quadrature.
N_BOOTSTRAP = 50


def logit(score):
    # network output before the sigmoid
    score = np.clip(score.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(score / (1 - score))


def load_npz(preds_dir, name):
    path = os.path.join(preds_dir, name)
    print(path)
    return dict(np.load(path, allow_pickle=True))


def load_all(preds_dirs, name):
    return [load_npz(d, name) for d in preds_dirs]


def mean_seed_std(values):
    values = np.stack(values)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), std


def pm_text(mean, seed, stat, fmt=".4f"):
    return rf"{mean:{fmt}} \pm {np.hypot(seed, stat):{fmt}}"


def save(fig, ax, outdir, name):
    # fixed margins instead of bbox_inches, so that figures in a row have identical axes boxes
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    fig.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf")
    plt.close(fig)
    print(f"  wrote {file}")


def legend(ax, *args, **kwargs):
    kwargs = {"frameon": False, "fontsize": FONTSIZE_LEGEND, "handlelength": 1.0} | kwargs
    ax.legend(*args, **kwargs)


def corner_text(ax, x, y, s, ha="left", **kwargs):
    ax.text(x, y, s, fontsize=FONTSIZE, ha=ha, va="top", transform=ax.transAxes, **kwargs)


def check_configs(run_dirs, match):
    """All runs share one config up to the seed, and the config is the task's."""
    skip = {"seed", "run_name", "run_dir", "hydra"}
    cfgs = [OmegaConf.load(os.path.join(d, "config.yaml")) for d in run_dirs]
    for key, value in match.items():
        found = OmegaConf.select(cfgs[0], key)
        assert found == value, f"{run_dirs[0]}: {key} = {found}, this script needs {value}"
    ref = {k: v for k, v in OmegaConf.to_container(cfgs[0]).items() if k not in skip}
    for d, cfg in zip(run_dirs[1:], cfgs[1:], strict=True):
        other = {k: v for k, v in OmegaConf.to_container(cfg).items() if k not in skip}
        diff = sorted(k for k in ref.keys() | other.keys() if ref.get(k) != other.get(k))
        assert not diff, f"{d} differs from {run_dirs[0]} in {diff}"
    return cfgs[0]


def parse_runs(name, match):
    """Command line of every task script: run dirs (one per seed) -> (preds dirs, model, outdir).

    The first run gives the statistical errors and the spread over all runs the seed errors; the
    training curves and asymmetries use `best_run`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories (runs/FINAL/<run>), one per seed.")
    parser.add_argument("--run-idx", type=int, default=0, help="Index N of the preds_N directory.")
    parser.add_argument(
        "--outdir", default=None, help="Output directory, default paper_plots/<task>."
    )
    parser.add_argument("--model", default=None, help="Network name, default from the config.")
    args = parser.parse_args()

    run_dirs = [os.path.normpath(r) for r in args.runs]
    cfg = check_configs(run_dirs, match)
    model = args.model or MODEL_NAMES[cfg.model._target_.split(".")[-1]]
    print(f"Seeds: {[os.path.basename(d) for d in run_dirs]}, model {model}")
    outdir = args.outdir or os.path.join(OUTPATH, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving plots to {outdir}")
    return [os.path.join(d, f"preds_{args.run_idx}") for d in run_dirs], model, outdir


def best_run(preds_dirs):
    """Preds dir of the run with the smallest validation loss; with es_load_best_model its
    evaluated checkpoint is the one at that loss."""
    losses = [load_npz(d, "training.npz")["val_loss"].min() for d in preds_dirs]
    best = preds_dirs[int(np.argmin(losses))]
    print(f"Best run (val loss {min(losses):.6f}): {best}")
    return best


def hist_band(ax, samples, bins, color):
    """Mean over seeds of the histograms normalized to the number of events, band: seed std
    and Poisson error in quadrature."""
    hists, stats = [], []
    for x in samples:
        counts, _ = np.histogram(x, bins=bins)
        hists.append(counts / len(x))
        stats.append(np.sqrt(counts) / len(x))
    mean, seed = mean_seed_std(hists)
    err = np.hypot(seed, stats[0])
    step(ax, bins, mean, color)
    for y in (mean - err, mean + err):
        ax.step(bins, dup_last(y), color=color, alpha=0.5, lw=0.5, where="post")
    ax.fill_between(
        bins, dup_last(mean - err), dup_last(mean + err), facecolor=color, alpha=0.3, step="post"
    )
    return mean + err


def dup_last(a):
    return np.append(a, a[-1])


def step(ax, bins, y, color, **kwargs):
    """Histogram line: step on the bin edges, closed at both ends."""
    ax.step(bins, dup_last(y), color=color, lw=LINE_WIDTH, where="post", **kwargs)
    ax.vlines([bins[0], bins[-1]], 0, [y[0], y[-1]], colors=color, lw=1.0, **kwargs)


def plot_output(
    outdir,
    training,
    channel,
    label_0,
    label_1,
    ylim=None,
    yticks=None,
    legend_fontsize=FONTSIZE_LEGEND,
    order=(0, 1),
):
    """Validation mean of the scalar (s) or pseudoscalar (p) output per class vs. iterations,
    band: its spread over the validation events. One run only. `order` gives the classes in
    plotting and legend order (blue, then red)."""
    key = {"s": "scalar", "p": "pseudoscalar"}[channel]
    labels = {0: label_0, 1: label_1}
    # first validation runs before training (step 0), then every n_train / (n_val - 1) steps
    n_train, n_val = len(training["train_loss"]), len(training["val_loss"])
    its = np.arange(n_val) * n_train / max(n_val - 1, 1)

    # centered running mean, the window shrinks at the edges so every validation point is kept;
    # the step-0 point (untrained network) is left out so its offset does not leak into the curve
    window = max(3, n_val // 10) | 1

    def smooth(x):
        kernel = np.ones(window)
        y = np.convolve(x[1:], kernel, "same") / np.convolve(np.ones(len(x) - 1), kernel, "same")
        return np.concatenate([x[:1], y])

    curves = [
        (
            smooth(training[f"{key}_output_{c}"]),
            smooth(training[f"{key}_output_{c}_std"]),
            color,
        )
        for c, color in zip(order, COLORS[1:3], strict=True)
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0, color="gray", ls="--", lw=1)
    for out, std, color in curves:
        ax.fill_between(its, out - std, out + std, color=color, alpha=0.2, lw=0)
    for out, _, color in curves:
        ax.plot(its, out, color=color, lw=LINE_WIDTH)
    ax.set_xlabel(r"Iterations $[10^3]$", fontsize=FONTSIZE)
    ax.set_ylabel(f"${channel}(x)$", fontsize=FONTSIZE)
    ax.margins(x=0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    proxy = [mlines.Line2D([], [], ls="-", color=c) for _, _, c in curves]
    legend(
        ax,
        proxy,
        [labels[c] for c in order],
        loc="upper center",
        ncol=2,
        columnspacing=0.4,
        handletextpad=0.3,
        borderaxespad=0.2,
        fontsize=legend_fontsize,
    )
    save(fig, ax, outdir, {"s": "output_scalar", "p": "output_pseudo"}[channel])
