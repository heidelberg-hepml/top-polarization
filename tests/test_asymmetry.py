import numpy as np
import torch

from experiments.tagging.asymmetry import asymmetry, fit_linear, o1
from experiments.tagging.likelihood import kinematic_p, lr_slope


def _jets(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, 6, generator=g, dtype=torch.float64) * 50
    p[:, 2, :] += 300  # all along +z, so the boosts are well defined
    e = torch.linalg.norm(p, dim=1) + 20
    return torch.cat([e[:, None], p], dim=1)


def test_asymmetry_error():
    a, sigma = asymmetry(np.r_[np.ones(300), -np.ones(100)])
    assert np.isclose(a, 0.5) and np.isclose(sigma, np.sqrt((1 - a**2) / 400))


def test_fit_linear_recovers_line():
    c = np.array([-2.6, -0.1, 0, 0.1, 2.6])
    (slope, offset), _, chi2, ndf = fit_linear(c, 0.02 * c + 0.001, np.full(5, 1e-3))
    assert np.allclose([slope, offset], [0.02, 0.001]) and chi2 < 1e-12 and ndf == 3


def test_lr_slope_linear_densities():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 400000)
    y = (rng.uniform(size=x.size) < (1 + 0.6 * x) / 2).astype(int)  # R: 1 + s x, L: 1 - s x
    slope, err = lr_slope(x, y)
    assert abs(slope - 0.6) < 5 * err


def test_kinematic_p_in_unit_interval():
    p = kinematic_p(_jets())
    assert ((p >= 0) & (p <= 1)).all()


def test_o1_is_parity_odd():
    jets, charge = _jets(), torch.ones(64, dtype=torch.float64)
    p = torch.rand(64, dtype=torch.float64)
    flip = jets * torch.tensor([1.0, -1, -1, -1], dtype=torch.float64)[None, :, None]
    assert torch.allclose(o1(jets, charge, p), -o1(flip, charge, p), atol=1e-9)
