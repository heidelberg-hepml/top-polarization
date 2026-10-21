import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

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

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPATH = os.path.join(REPO_DIR, "paper_plots")

COLORS = ["black", "blue", "red", "orange"]

MODEL_NAMES = {
    "LGATrSlimPseudoWrapper": "L-GATr-slim",
    "LGATrSlimWrapper": "L-GATr-slim",
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


def save(fig, outdir, name):
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {file}")


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

    The first run gives the training curves and the statistical errors, the spread over all runs
    the seed errors.
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
    ax.stairs(mean, bins, color=color, lw=2)
    ax.stairs(mean + err, bins, baseline=mean - err, fill=True, color=color, alpha=0.3, lw=0)
    return mean + err


def plot_output(outdir, training, channel, label_0, label_1, ylim=None):
    """Validation mean of the scalar (s) or pseudoscalar (p) output per class vs. iterations,
    band: its spread over the validation events. First seed only."""
    key = {"s": "scalar", "p": "pseudoscalar"}[channel]
    output_0, output_1 = training[f"{key}_output_0"], training[f"{key}_output_1"]
    std_0, std_1 = training[f"{key}_output_0_std"], training[f"{key}_output_1_std"]
    # first validation runs before training (step 0), then every n_train / (n_val - 1) steps
    n_train, n_val = len(training["train_loss"]), len(training["val_loss"])
    its = np.arange(n_val) * n_train / max(n_val - 1, 1)

    # centered running mean; the window shrinks at the edges so every validation point is kept
    # (only 11 to 101 points per run); the step-0 point (untrained network) is left out so its
    # offset does not leak into the trained curve
    window = max(3, n_val // 10) | 1

    def smooth(x):
        kernel = np.ones(window)
        y = np.convolve(x[1:], kernel, "same") / np.convolve(np.ones(len(x) - 1), kernel, "same")
        return np.concatenate([x[:1], y])

    curves = [
        (smooth(output_0), smooth(std_0), COLORS[1]),
        (smooth(output_1), smooth(std_1), COLORS[2]),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.axhline(0, color="gray", ls="--", lw=1)
    for out, std, color in curves:
        ax.fill_between(its, out - std, out + std, color=color, alpha=0.2, lw=0)
    for out, _, color in curves:
        ax.plot(its, out, color=color, lw=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"${channel}(x)$")
    ax.margins(x=0)
    ax.set_xlim(left=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    proxy = [
        mlines.Line2D([], [], ls="-", color=COLORS[2]),
        mlines.Line2D([], [], ls="-", color=COLORS[1]),
    ]
    ax.legend(proxy, [label_1, label_0], frameon=False, loc="best")
    save(fig, outdir, {"s": "output_scalar", "p": "output_pseudo"}[channel])
