"""Paper plots for tt_smeft, Im(d_tG) = -2.6 vs. 2.6: s(x) and p(x) vs. iterations, and the
network output O on the test set with the truth (score) and the predicted down tag (score_downtag).

usage: python paper_plots/tt_smeft_posneg.py runs/FINAL/ttsmeft_neg_s* [--outdir DIR]
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from experiments.tagging.asymmetry import asymmetry
from paper_plots.common import (
    COLORS,
    hist_band,
    load_all,
    load_npz,
    logit,
    mean_seed_std,
    parse_runs,
    plot_output,
    save,
)

LABELS = (r"$\mathrm{Im}(d_{tG}) = -2.6$", r"$\mathrm{Im}(d_{tG}) = 2.6$")
MATCH = {"exp_type": "tt_smeft", "data.lbl_0": "neg", "data.bsm": "im", "data.c_value": 2.6}


def plot_score(outdir, scores, model, name, process_label):
    # network output before the sigmoid, the C-odd observable; its asymmetry around 0 is printed
    samples = [
        [logit(s["labels_predict"][s["labels_true"] == lbl]) for s in scores] for lbl in (0, 1)
    ]
    bins = np.linspace(-0.5, 0.5, 31)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    top = max(
        hist_band(ax, x, bins, color).max() for x, color in zip(samples, COLORS[1:3], strict=True)
    )
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.35 * top)
    ax.set_xlabel(rf"$\mathcal{{O}}_{{\text{{{model}}}}}$")
    ax.set_ylabel("Normalized")
    proxy = [mlines.Line2D([], [], ls="-", color=c) for c in COLORS[1:3]]
    ax.legend(proxy, LABELS, frameon=False, loc="upper left", fontsize=13)
    ax.text(0.95, 0.95, s=process_label, ha="right", va="top", transform=ax.transAxes)
    save(fig, outdir, name)

    for label, x in zip(LABELS, samples, strict=True):
        mean, seed = mean_seed_std([asymmetry(xi)[0] for xi in x])
        stat = asymmetry(x[0])[1]
        print(f"  asymmetry ({name}) {label}: {mean:.4f} +- {seed:.4f} (seed) +- {stat:.4f} (stat)")


if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("tt_smeft_posneg", MATCH)
    training = load_npz(preds_dirs[0], "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.25, 0.25))
    plot_output(outdir, training, "p", *LABELS, ylim=(-0.25, 0.25))
    plot_score(
        outdir, load_all(preds_dirs, "score.npz"), model, "score", r"$t\bar{t} \to \ell\ell$"
    )
    downtag = load_all([d + "_downtag" for d in preds_dirs], "score.npz")
    plot_score(outdir, downtag, model, "score_downtag", r"$t\bar{t} \to q\ell$")
