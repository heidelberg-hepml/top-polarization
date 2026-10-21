"""Paper plots for the VBF toy, SM vs. Pos: s(x) and p(x) vs. iterations.

usage: python paper_plots/vbf_cp.py runs/FINAL/vbf_cp_s* [--outdir DIR]
"""

from paper_plots.common import load_npz, parse_runs, plot_output

LABELS = (r"$\text{SM}$", r"$\text{Pos}$")

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("vbf_cp", {"data.exp_type": "cp"})
    training = load_npz(preds_dirs[0], "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.7, 0.7))
    plot_output(outdir, training, "p", *LABELS, ylim=(-0.7, 0.7))
