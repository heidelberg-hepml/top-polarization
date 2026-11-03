"""Paper plots for tt_smeft, SM vs. Im(c_tG) = 2.6: s(x) and p(x) vs. iterations.

usage: python -m paper_plots.tt_smeft runs/tt_smeft/sm_slim_s* [--outdir DIR]
"""

from paper_plots.common import best_run, load_npz, parse_runs, plot_output

LABELS = (r"$\mathrm{Im}(c_{tG}) = 0$", r"$\mathrm{Im}(c_{tG}) = +2.6$")
MATCH = {"exp_type": "tt_smeft", "data.lbl_0": "sm", "data.bsm": "im", "data.c_value": 2.6}

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("tt_smeft", MATCH)
    training = load_npz(best_run(preds_dirs), "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.45, 0.45))
    plot_output(outdir, training, "p", *LABELS, ylim=(-0.15, 0.15), yticks=[-0.1, 0, 0.1])
