"""Sign asymmetries of CP-odd observables (O1, network output)."""

import numpy as np
import torch

from experiments.hep import lorentz_boost
from experiments.tagging.likelihood import probe_frame
from experiments.tagging.pseudoscalars import eps_tensor


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
