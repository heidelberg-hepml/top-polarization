"""O1 on all events of one tt_smeft dataset, with the true down quark (emulated dileptonic) and
the down-tagger probability (semileptonic, labels.txt of downtagging runs with
data.ttsmeft.split=all, one per down-tagger seed). Prints its sign asymmetry per coupling and, for
lbl_0=neg, pooled over +c and -c (N+ counts O1 > 0 at +c and O1 < 0 at -c), and writes
o1_<lbl_0>_c<c>.npz to outdir.

--plot draws O1 for Im(d_tG) = 0, -2.6, 2.6 from o1_neg_c2p6.npz and o1_sm_c2p6.npz in outdir.
Each sample is normalized to the 172 085 selected dileptonic tt events of CMS at 35.9 fb^-1
(arXiv:2205.07434, Table 1, "tt signal" summed over ee, emu, mumu). Band: Poisson error of the
expected events and, for the tagger, the spread over the down-tagger seeds, in quadrature.

usage: python paper_plots/tt_smeft_o1.py --lbl_0 neg --c_value 2.6 \
           --labels runs/FINAL/down_slim_s*/preds_0_ttsmeft_neg_all/labels.txt
       python paper_plots/tt_smeft_o1.py --plot
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
from experiments.tagging.top_events import rest_frame_energy
from paper_plots.common import (
    COLORS,
    FIGSIZE,
    FONTSIZE,
    LINE_WIDTH,
    OUTPATH,
    corner_text,
    dup_last,
    legend,
    mean_seed_std,
    save,
    step,
)

N_CMS = 22216 + 104051 + 45818
BINS = np.linspace(-0.03, 0.03, 25)
CHANNELS = {
    "o1_truth": ("tt_smeft_2p6_O1_delphes", r"$t\bar{t} \to \ell\ell$"),
    "o1_tagger": ("tt_smeft_2p6_O1_downtag_delphes", r"$t\bar{t} \to q\ell$"),
}


def compute_o1(data_dir, lbl_0, c_value, label_files):
    files = TTSmeftExperiment._select_filenames(data_dir, "loose", lbl_0, "im", c_value)
    data = TTSmeftDataset(tag_down=True)
    data.load_data(data_dir, files)

    # momenta (n, 4, 6) as in `stack_jets`, lepton charge and the truth down-is-harder flag
    start, end = data.ptr[:-1], data.ptr[1:]
    _, first, second = data.nonb_subjets(torch.arange(len(data)))
    down_first = data.down_harder[:, None]
    down, other = torch.where(down_first, first, second), torch.where(down_first, second, first)
    harder = rest_frame_energy(down, data.x[start]) >= rest_frame_energy(other, data.x[start])
    hard, soft = (torch.where(harder[:, None], a, b) for a, b in [(down, other), (other, down)])
    jets = torch.stack(
        [data.x[start], data.x[start + 1], hard, soft, data.x[end - 2], data.x[end - 1]], dim=-1
    )
    charge = data.scalars[end - 1, data.layout.charge_col].double()

    out = {
        "labels": data.labels.numpy(),
        "o1_truth": o1(jets, charge, harder.double()).numpy(),
        "c0": 0.0 if lbl_0 == "sm" else -c_value,
        "c1": c_value,
    }
    tagger = []
    for path in label_files:
        labels = np.loadtxt(path)  # (down subjet is the harder one, predicted probability of that)
        assert np.array_equal(labels[:, 0], harder.numpy()), f"{path} misaligned with dataset"
        tagger.append(o1(jets, charge, torch.from_numpy(labels[:, 1])).numpy())
    if tagger:
        out["o1_tagger"] = np.stack(tagger)  # (seeds, events)
    return out


def print_asymmetries(out, pooled):
    for key in [k for k in CHANNELS if k in out]:
        seeds = np.atleast_2d(out[key])  # truth: one row, tagger: one row per seed
        cases = {
            f"c = {out[f'c{lbl}']:g}": lambda x, lbl=lbl: asymmetry(x[out["labels"] == lbl])
            for lbl in (0, 1)
        }
        if pooled:
            cases[f"pooled |c| = {out['c1']:g}"] = lambda x: pooled_asymmetry(x, out["labels"])
        for name, f in cases.items():
            asym = [f(x) for x in seeds]
            mean, seed = mean_seed_std([a for a, _ in asym])
            stat = asym[0][1]
            print(
                f"  asymmetry {key} {name}: {mean:+.4f} +- {np.hypot(seed, stat):.4f}"
                f" ({seed:.4f} seed, {stat:.4f} binomial)"
            )


def plot(outdir, data, key):
    name, process = CHANNELS[key]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles, labels = [], []
    for c, color in zip((0.0, -2.6, 2.6), COLORS[:3], strict=True):
        # in units of 10^3 events
        hists = [np.histogram(x, BINS)[0] / len(x) * N_CMS / 1e3 for x in data[c]]
        mean, seed = mean_seed_std(hists)
        err = np.hypot(np.sqrt(mean / 1e3), seed)
        step(ax, BINS * 100, mean, color)
        ax.fill_between(
            BINS * 100,
            dup_last(mean - err),
            dup_last(mean + err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )
        handles.append(mlines.Line2D([], [], color=color, lw=LINE_WIDTH))
        labels.append(rf"$\mathrm{{Im}}(d_{{tG}}) = {c:g}$")
    ax.set_xlim(BINS[0] * 100, BINS[-1] * 100)
    ax.set_ylim(0, 1.45 * ax.get_ylim()[1])
    ax.set_xlabel(r"$\mathcal{O}_1$ [a.u.]", fontsize=FONTSIZE)
    ax.set_ylabel(r"Events $[10^3]$", fontsize=FONTSIZE)
    corner_text(ax, 0.05, 0.95, process)
    legend(ax, handles, labels, loc="upper right", fontsize=FONTSIZE - 4, borderaxespad=0.2)
    save(fig, ax, outdir, name)


def npz_name(lbl_0, c_value):
    return f"o1_{lbl_0}_c{c_value:g}".replace(".", "p") + ".npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tt_smeft/")
    parser.add_argument("--lbl_0", default="neg", choices=["neg", "sm"])
    parser.add_argument("--c_value", type=float, default=2.6)
    parser.add_argument("--labels", nargs="*", default=[], help="labels.txt, one per seed.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--outdir", default=os.path.join(OUTPATH, "tt_smeft"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.plot:
        neg, sm = (
            dict(np.load(os.path.join(args.outdir, npz_name(t, 2.6)))) for t in ("neg", "sm")
        )
        for key in CHANNELS:
            # {c: [O1 of that sample, one array per seed]}
            data = {
                float(o[f"c{lbl}"]): list(np.atleast_2d(o[key])[:, o["labels"] == lbl])
                for o, lbl in [(neg, 0), (neg, 1), (sm, 0)]
            }
            plot(args.outdir, data, key)
    else:
        out = compute_o1(args.data_dir, args.lbl_0, args.c_value, args.labels)
        print_asymmetries(out, pooled=args.lbl_0 == "neg")
        path = os.path.join(args.outdir, npz_name(args.lbl_0, args.c_value))
        np.savez(path, **out)
        print(f"  wrote {path}")
