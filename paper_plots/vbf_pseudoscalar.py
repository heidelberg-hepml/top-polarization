"""Paper plots for the VBF toy, Neg vs. Pos (pseudoscalar): s(x) and p(x) vs. iterations.

usage: python paper_plots/vbf_pseudoscalar.py runs/FINAL/vbf_pseudoscalar_s* [--outdir DIR]
"""

import os
import zipfile

from paper_plots.common import load_npz, parse_runs, plot_output

NEG, POS = r"$c_{H\tilde{W}} = -1$", r"$c_{H\tilde{W}} = +1$"


def pos_class(run_dir):
    """Class of cHWtil_pos.npy in the code the run was trained with: 1 now, 0 in older runs
    (including runs/FINAL)."""
    with zipfile.ZipFile(os.path.join(run_dir, "source.zip")) as z:
        code = z.read("tagging/experiment.py").decode()
    task = code.split('exp_type == "pseudoscalar"')[1].split("elif")[0]
    return int(task.index("cHWtil_pos") > task.index("cHWtil_neg"))


if __name__ == "__main__":
    preds_dirs, model, outdir = parse_runs("vbf_pseudoscalar", {"data.exp_type": "pseudoscalar"})
    training = load_npz(preds_dirs[0], "training.npz")
    pos = pos_class(os.path.dirname(preds_dirs[0]))
    labels = (POS, NEG) if pos == 0 else (NEG, POS)
    # the sign of a pseudoscalar is a convention: orient p(x) so that the +1 sample is positive
    if training[f"pseudoscalar_output_{pos}"][-1] < 0:
        for c in (0, 1):
            training[f"pseudoscalar_output_{c}"] = -training[f"pseudoscalar_output_{c}"]
    # negative coefficient first
    order = (1 - pos, pos)
    plot_output(outdir, training, "s", *labels, ylim=(-0.3, 0.3), order=order)
    plot_output(outdir, training, "p", *labels, ylim=(-1.7, 1.7), order=order)
