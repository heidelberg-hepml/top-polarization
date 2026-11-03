"""Paper plots for the VBF toy, Neg vs. Pos (pseudoscalar): s(x) and p(x) vs. iterations.

usage: python -m paper_plots.vbf_pseudoscalar runs/vbf/pseudoscalar_slim_s* [--outdir DIR]
"""

from paper_plots.common import best_run, load_npz, parse_runs, plot_output

# class 0: c_HWtil = -1, class 1: c_HWtil = +1
LABELS = (r"$c_{H\tilde{W}} = -1$", r"$c_{H\tilde{W}} = +1$")

if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("vbf_pseudoscalar", {"data.exp_type": "pseudoscalar"})
    training = load_npz(best_run(preds_dirs), "training.npz")
    # the sign of a pseudoscalar is a convention: orient p(x) so that the +1 sample is positive
    if training["pseudoscalar_output_1"][-1] < 0:
        for c in (0, 1):
            training[f"pseudoscalar_output_{c}"] = -training[f"pseudoscalar_output_{c}"]
    plot_output(outdir, training, "s", *LABELS, ylim=(-0.12, 0.12))
    plot_output(outdir, training, "p", *LABELS, ylim=(-1.7, 1.7))
