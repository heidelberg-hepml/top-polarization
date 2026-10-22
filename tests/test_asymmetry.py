import numpy as np
import torch

from experiments.tagging.asymmetry import asymmetry, o1
from experiments.tagging.likelihood import kinematic_p


def _jets(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, 6, generator=g, dtype=torch.float64) * 50
    p[:, 2, :] += 300  # all along +z, so the boosts are well defined
    e = torch.linalg.norm(p, dim=1) + 20
    return torch.cat([e[:, None], p], dim=1)


def test_asymmetry_error():
    a, sigma = asymmetry(np.r_[np.ones(300), -np.ones(100)])
    assert np.isclose(a, 0.5) and np.isclose(sigma, np.sqrt((1 - a**2) / 400))


def test_kinematic_p_in_unit_interval():
    p = kinematic_p(_jets())
    assert ((p >= 0) & (p <= 1)).all()


def test_o1_is_parity_odd():
    jets, charge = _jets(), torch.ones(64, dtype=torch.float64)
    p = torch.rand(64, dtype=torch.float64)
    flip = jets * torch.tensor([1.0, -1, -1, -1], dtype=torch.float64)[None, :, None]
    assert torch.allclose(o1(jets, charge, p), -o1(flip, charge, p), atol=1e-9)
