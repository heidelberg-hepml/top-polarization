"""Sign asymmetries of CP-odd observables (O1, network output) and the Im(c_tG) sensitivity fit.

usage: python -m experiments.tagging.asymmetry <preds dir with o1.npz>... [--yield name=events]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.hep import lorentz_boost
from experiments.tagging.likelihood import probe_frame
from experiments.tagging.pseudoscalars import eps_tensor

FONTSIZE = 14


def o1(jets, charge, p_hard):
    """eps(t_lep, t_had, l, d) / m_t^4 with the down proxy k = p q_hard + (1 - p) q_soft.

    jets: (n, 4, 6) hadtop, b, hard, soft, leptop, lepton in the lab frame. k is a spatial vector
    in the hadronic top rest frame, where the contraction is evaluated (it is Lorentz invariant).
    """
    q_hard, q_soft, _ = probe_frame(jets, charge)
    k = p_hard[:, None] * q_hard + (1 - p_hard[:, None]) * q_soft
    hadtop = jets[:, :, 0]
    leptop, lepton, top = (lorentz_boost(jets[:, :, i], hadtop) for i in (4, 5, 0))
    d = torch.cat([torch.zeros_like(k[:, :1]), k], dim=1)
    return eps_tensor(leptop, top, lepton, d, dtype=jets.dtype)


def asymmetry(x):
    """Signed asymmetry A = (N+ - N-) / N of the sign of x and its error sqrt(4 n+ n- / N^3)."""
    x = np.asarray(x)
    n_pos, n_neg = float((x > 0).sum()), float((x < 0).sum())
    n = n_pos + n_neg
    return (n_pos - n_neg) / n, 2 * np.sqrt(n_pos * n_neg / n**3)


def fit_linear(c, a, sigma):
    """Weighted fit a = slope * c + offset. Return (slope, offset), their errors, chi2 and ndf."""
    c, a, w = np.asarray(c, float), np.asarray(a, float), 1 / np.asarray(sigma, float) ** 2
    design = np.stack([c, np.ones_like(c)], axis=1)
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    params = cov @ (design.T @ (w * a))
    chi2 = float((w * (a - design @ params) ** 2).sum())
    return params, np.sqrt(np.diag(cov)), chi2, len(c) - 2


def sigma_c(slope, n_events):
    """Error on the coupling for A = 0 and `n_events` selected events: 1 / (|slope| sqrt(N))."""
    return 1 / (abs(slope) * np.sqrt(n_events))


def _plot_o1(path, per_c, scale):
    fig, ax = plt.subplots(figsize=(5, 4))
    lo, hi = np.quantile(np.concatenate(list(per_c.values())), [0.005, 0.995])
    bins = np.linspace(lo, hi, 41)
    for c, v in sorted(per_c.items()):
        weights = np.full(len(v), (scale or 1.0) / len(v))
        ax.hist(v, bins=bins, weights=weights, histtype="step", label=rf"$c = {c:g}$")
    ax.set_xlabel(r"$\mathcal{O}_1 / m_t^4$", fontsize=FONTSIZE)
    ax.set_ylabel("Events" if scale else "Normalized", fontsize=FONTSIZE)
    ax.legend(frameon=False, fontsize=FONTSIZE)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="preds directories containing o1.npz")
    parser.add_argument("--yield", dest="yields", nargs="*", default=[], help="name=events")
    parser.add_argument("--outdir", default=None, help="write the O1 histograms here")
    args = parser.parse_args()

    # key -> list of (coupling, events): every directory is an independent estimate of A(c)
    samples = {}
    for path in args.dirs:
        data = np.load(os.path.join(path, "o1.npz"))
        for key in data.files:
            if key.startswith(("o1_", "omega")):
                for label, c in enumerate([float(data["c0"]), float(data["c1"])]):
                    samples.setdefault(key, []).append((c, data[key][data["labels"] == label]))

    yields = {s.split("=")[0]: float(s.split("=")[1]) for s in args.yields}
    for key, entries in samples.items():
        rows = sorted((c, *asymmetry(x), len(x)) for c, x in entries)
        print(f"== {key}")
        for c, a, sig, n in rows:
            print(f"  c = {c:+5.1f}  A = {a:+.5f} +- {sig:.5f}  (N = {n})")
        c, a, sig, _ = (np.array(col) for col in zip(*rows, strict=True))
        if len(set(c)) > 2:
            (slope, offset), (d_slope, d_offset), chi2, ndf = fit_linear(c, a, sig)
            print(
                f"  fit A = a c + b: a = {slope:.3e} +- {d_slope:.1e}, b = {offset:.2e} "
                f"+- {d_offset:.1e}, chi2/ndf = {chi2:.1f}/{ndf}"
            )
            for name, events in yields.items():
                print(f"  sigma(c) at {events:.3g} events ({name}) = {sigma_c(slope, events):.3f}")
        if args.outdir and key.startswith("o1_"):
            os.makedirs(args.outdir, exist_ok=True)
            per_c = {}
            for c, x in entries:
                per_c.setdefault(c, []).append(x)
            per_c = {c: np.concatenate(xs) for c, xs in per_c.items()}
            for name, events in yields.items() or {None: None}.items():
                suffix = f"_{name}" if name else ""
                _plot_o1(os.path.join(args.outdir, f"{key}{suffix}.pdf"), per_c, events)


if __name__ == "__main__":
    main()
