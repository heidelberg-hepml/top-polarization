import torch

EPS = 1e-10
CUTOFF = 10


def stable_arctanh(x, eps=EPS):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def EPPP_to_PtPhiEtaM(fourmomenta):
    E, px, py, pz = torch.unbind(fourmomenta, dim=-1)

    pt = torch.sqrt((px**2 + py**2).clamp(min=EPS))
    phi = torch.arctan2(avoid_zero(py), avoid_zero(px))
    p_abs = torch.sqrt((pz**2 + pt**2).clamp(min=EPS))
    eta = stable_arctanh(pz / p_abs).clamp(min=-CUTOFF, max=CUTOFF)
    m2 = E**2 - px**2 - py**2 - pz**2
    m = torch.sqrt(m2.clamp(min=EPS))
    return torch.stack((pt, phi, eta, m), dim=-1)


def PtPhiEtaM_to_EPPP(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == 4, "Input tensor must have last dimension of size 4"
    pt, phi, eta, m2 = torch.unbind(x, dim=-1)
    m2 = m2**2
    eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt((m2 + pt**2 * torch.cosh(eta) ** 2).clamp(min=EPS))
    return torch.stack((E, px, py, pz), dim=-1)



def lorentz_boost(particle, frame, eps=EPS):
    # lorentz boost particle to rest frame
    E  = particle[:, 0]
    p  = particle[:, 1:]
    Ef = frame[:, 0]
    pf = frame[:, 1:]

    beta = pf / (Ef.unsqueeze(1) + eps)
    beta2 = (beta * beta).sum(dim=1)
    gamma = 1.0 / torch.sqrt(1.0 - beta2 + eps)
    bp = (beta * p).sum(dim=1)

    Eprime = gamma * (E - bp)
    factor = ((gamma - 1.0) * bp / (beta2 + eps) - gamma * E)
    pprime = p + factor.unsqueeze(1) * beta

    mask = beta2 > eps
    Eprime = torch.where(mask, Eprime, E)
    pprime = torch.where(mask.unsqueeze(1), pprime, p)

    return torch.cat([Eprime.unsqueeze(1), pprime], dim=1)


def get_pt(p):
    # transverse momentum
    return torch.sqrt((p[..., 1] ** 2 + p[..., 2] ** 2).clamp(min=EPS))


def avoid_zero(x, eps=EPS):
    # set small-abs values to eps for numerical stability
    return torch.where(x.abs() < eps, eps, x)


def get_phi(p):
    # azimuthal angle
    return torch.arctan2(avoid_zero(p[..., 2]), avoid_zero(p[..., 1]))


def get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1).clamp(min=EPS))
    return stable_arctanh(p[..., 3] / p_abs)


def get_deltaR(v1, v2):
    # deltaR = sqrt((eta1-eta2)^2 + (phi1 - phi2)^2)
    eta1 = get_eta(v1)
    eta2 = get_eta(v2)

    phi1 = get_phi(v1)
    phi2 = get_phi(v2)

    delta_y = eta1 - eta2
    delta_phi = (phi1 - phi2 + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.sqrt((delta_y**2 + delta_phi**2).clamp(min=EPS))
