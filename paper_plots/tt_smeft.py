"""Paper plots for tt_smeft, SM vs. Im(d_tG) = 2.6: s(x) and p(x) vs. iterations.

usage: python paper_plots/tt_smeft.py runs/FINAL/ttsmeft_sm_s* [--outdir DIR]
"""

from paper_plots.common import load_npz, parse_runs, plot_output

LABELS = (r"$\mathrm{Im}(d_{tG}) = 0$", r"$\mathrm{Im}(d_{tG}) = 2.6$")
MATCH = {"exp_type": "tt_smeft", "data.lbl_0": "sm", "data.bsm": "im", "data.c_value": 2.6}

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("tt_smeft", MATCH)
    training = load_npz(preds_dirs[0], "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.4, 0.4))
    plot_output(outdir, training, "p", *LABELS, ylim=(-0.1, 0.1))
