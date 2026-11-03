"""Sign asymmetries of CP-odd observables (O1, network output)."""

import numpy as np
import torch

from experiments.hep import lorentz_boost
from experiments.tagging.pseudoscalars import eps_tensor


def o1(jets, p_hard):
    """eps(t_had, t_lep, d, l) / m_t^4, evaluated in the hadronic top rest frame.

    jets: (n, 4, 6) hadtop, b, hard, soft, leptop, lepton in the lab frame. The down momentum is
    the probe direction k = p k_hard / |k_hard| + (1 - p) k_soft / |k_soft|, scaled to the length
    p |k_hard| + (1 - p) |k_soft|: for p = 0 or 1 it is the down subjet momentum, and the sign of O1
    is that of the probe. The top is at rest, so the energy of d does not enter.
    """
    hadtop, hard, soft, leptop, lepton = (
        lorentz_boost(jets[:, :, i], jets[:, :, 0]) for i in (0, 2, 3, 4, 5)
    )
    k_hard, k_soft = hard[:, 1:4], soft[:, 1:4]
    n_hard, n_soft = k_hard.norm(dim=-1, keepdim=True), k_soft.norm(dim=-1, keepdim=True)
    p = p_hard[:, None]
    k = p * k_hard / n_hard + (1 - p) * k_soft / n_soft
    k = k / k.norm(dim=-1, keepdim=True) * (p * n_hard + (1 - p) * n_soft)
    d = torch.cat([torch.zeros_like(k[:, :1]), k], dim=1)
    return eps_tensor(hadtop, leptop, d, lepton, dtype=jets.dtype)


def asymmetry(x):
    """Signed asymmetry A = (N+ - N-) / N of the sign of x and its error sqrt(4 n+ n- / N^3)."""
    x = np.asarray(x)
    n_pos, n_neg = float((x > 0).sum()), float((x < 0).sum())
    n = n_pos + n_neg
    return (n_pos - n_neg) / n, 2 * np.sqrt(n_pos * n_neg / n**3)


def pooled_asymmetry(x, labels):
    """Asymmetry of the +c (label 1) and -c (label 0) samples together: x of the -c sample, the
    CP image of the +c one, enters with flipped sign."""
    return asymmetry(np.where(np.asarray(labels) == 1, x, -np.asarray(x)))
