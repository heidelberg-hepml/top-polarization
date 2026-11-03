"""O1 on all events of one tt_smeft dataset, with the true down quark (emulated dileptonic) and
the down-tagger probability (semileptonic, labels.txt of downtagging runs with
data.ttsmeft.split=all, one per down-tagger seed, of which the run with the smallest validation
loss is used). Prints its sign asymmetry per coupling and, for lbl_0=neg, pooled over +c and -c
(N+ counts O1 > 0 at +c and O1 < 0 at -c), and writes o1_<lbl_0>_c<c>.npz to outdir.

--plot draws O1 for Im(c_tG) = 0, -2.6, 2.6 from o1_neg_c2p6.npz and o1_sm_c2p6.npz in outdir.
Each distribution is normalized to the expected events XS * LUMI * eff of its channel over all
its events, including those outside BINS; band: Poisson error of the simulated counts. Lower
panel: ratio to the SM.

usage: python -m paper_plots.tt_smeft_o1 --lbl_0 neg --c_value 2.6 \
           --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_neg_all/labels.txt
       python -m paper_plots.tt_smeft_o1 --plot
"""

import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.tagging.asymmetry import asymmetry, o1, pooled_asymmetry
from experiments.tagging.dataset import TTSmeftDataset
from experiments.tagging.experiment import TTSmeftExperiment
from paper_plots.common import (
    BOTTOM,
    COLORS,
    FIGSIZE,
    FONTSIZE,
    FONTSIZE_TICK,
    LEFT,
    LINE_WIDTH,
    OUTPATH,
    RIGHT,
    TOP,
    best_run,
    corner_text,
    dup_last,
    legend,
    step,
)

XS = 826_400  # fb
LUMI = 35.9  # fb^-1 as used in https://arxiv.org/pdf/2205.07434
# decay + acceptance efficiencies; their ratio (12) is the event ratio of the channels in the paper
LEP_DECAY_EFF = 0.0154  # Derived from https://arxiv.org/pdf/2509.15066
HAD_DECAY_EFF = 0.1877  # Derived from https://arxiv.org/pdf/2006.13076
BINS = np.linspace(-3, 3, 31)
# key: (file name, process, channel, expected events)
CHANNELS = {
    "o1_truth": (
        "tt_smeft_2p6_O1_delphes",
        r"$t\bar{t} \to \ell\ell$",
        r"\ell\ell",
        XS * LUMI * LEP_DECAY_EFF,
    ),
    "o1_tagger": (
        "tt_smeft_2p6_O1_downtag_delphes",
        r"$t\bar{t} \to q\ell$",
        r"q\ell",
        XS * LUMI * HAD_DECAY_EFF,
    ),
}


def compute_o1(data_dir, lbl_0, c_value, label_file):
    files = TTSmeftExperiment._select_filenames(data_dir, "loose", lbl_0, "im", c_value)
    data = TTSmeftDataset()
    data.load_data(data_dir, files)

    # momenta (n, 4, 6) as in `stack_jets` and the truth down-is-harder flag
    start, end = data.ptr[:-1], data.ptr[1:]
    jets = torch.stack(
        [data.x[i] for i in (start, start + 1, start + 2, start + 3, end - 2, end - 1)], dim=-1
    ).double()
    harder = data.down_harder

    out = {
        "labels": data.labels.numpy(),
        "o1_truth": o1(jets, harder.double()).numpy(),
        "c0": 0.0 if lbl_0 == "sm" else -c_value,
        "c1": c_value,
    }
    if label_file is not None:
        # (down subjet is the harder one, predicted probability of that)
        labels = np.loadtxt(label_file)
        assert np.array_equal(labels[:, 0], harder.numpy()), f"{label_file} misaligned with dataset"
        out["o1_tagger"] = o1(jets, torch.from_numpy(labels[:, 1])).numpy()
    return out


def print_asymmetries(out, pooled):
    for key in [k for k in CHANNELS if k in out]:
        x = out[key]
        cases = {f"c = {out[f'c{lbl}']:g}": asymmetry(x[out["labels"] == lbl]) for lbl in (0, 1)}
        if pooled:
            cases[f"pooled |c| = {out['c1']:g}"] = pooled_asymmetry(x, out["labels"])
        for name, (a, stat) in cases.items():
            print(f"  asymmetry {key} {name}: {a:+.4f} +- {stat:.4f} (binomial)")


def plot(outdir, data, key):
    name, process, channel, n_events = CHANNELS[key]
    # same figure size and margins as the other figures, the ratio panel takes a quarter
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    counts = {c: np.histogram(data[c], BINS)[0] for c in data}
    handles, labels = [], []
    for c, color in zip((0.0, -2.6, 2.6), COLORS[:3], strict=True):
        # all events scaled to the expected events, Poisson error of the simulated counts
        scale = n_events / len(data[c])
        mean, err = counts[c] * scale, np.sqrt(counts[c]) * scale
        step(ax, BINS, mean, color)
        ax.fill_between(
            BINS,
            dup_last(mean - err),
            dup_last(mean + err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )
        # ratio to the SM, both normalized to their number of events; Poisson errors of both
        ratio = (counts[c] / len(data[c])) / (counts[0.0] / len(data[0.0]))
        rerr = ratio * np.sqrt(1 / counts[c] + 1 / counts[0.0])
        rax.step(BINS, dup_last(ratio), color=color, lw=LINE_WIDTH, where="post")
        if c:
            rax.fill_between(
                BINS,
                dup_last(ratio - rerr),
                dup_last(ratio + rerr),
                facecolor=color,
                alpha=0.3,
                step="post",
            )
        handles.append(mlines.Line2D([], [], color=color, lw=LINE_WIDTH))
        labels.append(rf"$\mathrm{{Im}}(c_{{tG}}) = {c:+g}$" if c else r"$\mathrm{Im}(c_{tG}) = 0$")
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_yscale("log")
    # headroom above the peak for the legend
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * (top / bottom) ** 0.45)
    ax.set_ylabel("Events", fontsize=FONTSIZE)
    corner_text(ax, 0.05, 0.95, process)
    legend(ax, handles, labels, loc="upper right", fontsize=FONTSIZE - 4, borderaxespad=0.2)
    rax.set_ylim(0.85, 1.4)
    rax.set_yticks([1.0, 1.2])
    rax.set_ylabel(r"$\frac{\mathrm{BSM}}{\mathrm{SM}}$", fontsize=FONTSIZE)
    rax.set_xlabel(rf"$O_1^{{{channel}}}/m_t^4$", fontsize=FONTSIZE)
    for a in (ax, rax):
        a.tick_params(axis="both", labelsize=FONTSIZE_TICK)
        a.yaxis.set_label_coords(-0.15, 0.5)
    fig.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
    file = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(file, format="pdf")
    plt.close(fig)
    print(f"  wrote {file}")


def npz_name(lbl_0, c_value):
    return f"o1_{lbl_0}_c{c_value:g}".replace(".", "p") + ".npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tt_smeft/")
    parser.add_argument("--lbl_0", default="neg", choices=["neg", "sm"])
    parser.add_argument("--c_value", type=float, default=2.6)
    parser.add_argument(
        "--labels", nargs="*", default=[], help="labels.txt, one per down-tagger seed."
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--outdir", default=os.path.join(OUTPATH, "tt_smeft"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.plot:
        neg, sm = (
            dict(np.load(os.path.join(args.outdir, npz_name(t, 2.6)))) for t in ("neg", "sm")
        )
        for key in CHANNELS:
            # {c: O1 of that sample}
            data = {
                float(o[f"c{lbl}"]): o[key][o["labels"] == lbl]
                for o, lbl in [(neg, 0), (neg, 1), (sm, 0)]
            }
            plot(args.outdir, data, key)
    else:
        label_file = None
        if args.labels:
            # labels.txt sits in <run>/preds_0_ttsmeft_*/, the validation loss in <run>/preds_0/
            preds = {
                os.path.join(os.path.dirname(os.path.dirname(f)), "preds_0"): f for f in args.labels
            }
            label_file = preds[best_run(list(preds))]
            print(f"  down tagger labels: {label_file}")
        out = compute_o1(args.data_dir, args.lbl_0, args.c_value, label_file)
        print_asymmetries(out, pooled=args.lbl_0 == "neg")
        path = os.path.join(args.outdir, npz_name(args.lbl_0, args.c_value))
        np.savez(path, **out)
        print(f"  wrote {path}")
