"""Paper plots for tt_smeft, Im(c_tG) = -2.6 vs. 2.6: s(x) and p(x) vs. iterations, and the
pseudoscalar output p on the test set with the truth (score) and the predicted down tag
(score_downtag), and the sign asymmetry of p pooled over Im(c_tG) = +-c (on the test events at
2.6 and on all events at 0.1). Curves, scores and asymmetries: run with the smallest validation
loss.
O1: paper_plots/tt_smeft_o1.py.

usage: python -m paper_plots.tt_smeft_posneg runs/tt_smeft/neg_slim_s* [--outdir DIR]
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from experiments.tagging.asymmetry import pooled_asymmetry
from paper_plots.common import (
    COLORS,
    FIGSIZE,
    FONTSIZE,
    FONTSIZE_TICK,
    best_run,
    corner_text,
    hist_band,
    legend,
    load_npz,
    logit,
    parse_runs,
    plot_output,
    save,
)

LABELS = (r"$\mathrm{Im}(c_{tG}) = -2.6$", r"$\mathrm{Im}(c_{tG}) = +2.6$")
MATCH = {"exp_type": "tt_smeft", "data.lbl_0": "neg", "data.bsm": "im", "data.c_value": 2.6}


def plot_score(outdir, scores, name, process_label, channel):
    # pseudoscalar output p(x) before the sigmoid, the CP-odd observable O_L-GATr
    samples = [
        [logit(s["labels_predict_pseudo"][s["labels_true"] == lbl]) for s in scores]
        for lbl in (0, 1)
    ]
    bins = np.linspace(-0.5, 0.5, 31)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    top = max(
        hist_band(ax, x, bins, color).max() for x, color in zip(samples, COLORS[1:3], strict=True)
    )
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.6 * top)
    ax.set_xlabel(rf"$O_\text{{L-GATr}}^{{{channel}}}$", fontsize=FONTSIZE)
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    proxy = [mlines.Line2D([], [], ls="-", color=c) for c in COLORS[1:3]]
    legend(ax, proxy, LABELS, loc="upper right")
    corner_text(ax, 0.05, 0.95, process_label)
    save(fig, ax, outdir, name)


def print_asymmetry(asym, name):
    """Sign asymmetry of p, pooled over the +c and -c samples, with its binomial error."""
    a, stat = pooled_asymmetry(asym["omega"], asym["labels"])
    print(
        f"  asymmetry ({name}) p |c| = {asym['c1']:g}: {a:+.4f} +- {stat:.4f}"
        f" (N = {len(asym['labels'])})"
    )


if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("tt_smeft_posneg", MATCH)
    best = best_run(preds_dirs)
    training = load_npz(best, "training.npz")
    # the two labels do not fit in one row at the default legend size
    for channel in ("s", "p"):
        plot_output(
            outdir, training, channel, *LABELS, ylim=(-0.27, 0.27), legend_fontsize=FONTSIZE_TICK
        )
    for suffix, name, process, channel in [
        ("", "score", r"$t\bar{t} \to \ell\ell$", r"\ell\ell"),
        ("_mixdown", "score_downtag", r"$t\bar{t} \to q\ell$", r"q\ell"),
    ]:
        plot_score(outdir, [load_npz(best + suffix, "score.npz")], name, process, channel)
        print_asymmetry(load_npz(best + suffix, "asymmetry.npz"), name)
        # same network on all Im(c_tG) = +-0.1 events (never trained on), numbers only
        print_asymmetry(load_npz(best + suffix + "_c0p1_all", "asymmetry.npz"), name + "_c0p1")
