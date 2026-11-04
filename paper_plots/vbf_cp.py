"""Paper plots for the VBF toy, SM vs. Pos: s(x) and p(x) vs. iterations, and the AUC.

usage: python -m paper_plots.vbf_cp runs/vbf/cp_slim_s* [--outdir DIR]
"""

from paper_plots.common import auc_errors, best_run, load_npz, parse_runs, plot_output

LABELS = (r"$c_{H\tilde{W}} = 0$", r"$c_{H\tilde{W}} = +1$")

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("vbf_cp", {"data.exp_type": "cp"})
    training = load_npz(best_run(preds_dirs), "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.8, 0.8), yticks=[-0.5, 0, 0.5])
    plot_output(outdir, training, "p", *LABELS, ylim=(-0.8, 0.8), yticks=[-0.5, 0, 0.5])
    auc_errors(preds_dirs)
