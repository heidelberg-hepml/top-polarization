"""Paper plots for the VBF toy, Neg vs. Pos (pseudoscalar): s(x) and p(x) vs. iterations.

usage: python paper_plots/vbf_pseudoscalar.py runs/FINAL/vbf_pseudoscalar_s* [--outdir DIR]
"""

from paper_plots.common import load_npz, parse_runs, plot_output

LABELS = (r"$\text{Neg}$", r"$\text{Pos}$")

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("vbf_pseudoscalar", {"data.exp_type": "pseudoscalar"})
    training = load_npz(preds_dirs[0], "training.npz")
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.3, 0.3))
    plot_output(outdir, training, "p", *LABELS)
