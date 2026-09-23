from itertools import permutations

import torch

from experiments.hep import EPS, lorentz_boost


def eps_tensor(*p, dtype, norm=173**4):
    def permutation_sign(perm):
        n = len(perm)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1
        return 1.0 if inversions % 2 == 0 else -1.0

    def levi_cevita_tensor(dim):
        arr = torch.zeros(tuple([dim for _ in range(dim)]), dtype=dtype)
        for x in permutations(tuple(range(dim))):
            arr[x] = permutation_sign(x)
        return arr

    N = len(p)
    if N != 4:
        raise ValueError("Only 4D Levi-Civita tensor is supported.")
    eps = levi_cevita_tensor(N)
    g_munu = torch.diag(torch.tensor([1, -1, -1, -1], dtype=dtype))

    lowered = [torch.einsum("ij, nj -> ni", g_munu, q.reshape(-1, 4)) for q in p]
    return torch.einsum("ijkl, ni, nj, nk, nl -> n", eps, *lowered) / norm


def _normalize(v, eps=EPS):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def spin_correlation_nr(p_top_had, p_top_lep, p_prox_down, p_lep, lep_charge, dtype):
    beam = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=dtype).expand_as(p_top_had)

    ttbar = p_top_had + p_top_lep

    is_lep_pos = (lep_charge > 0).unsqueeze(-1)
    top = torch.where(is_lep_pos, p_top_lep, p_top_had)
    atop = torch.where(is_lep_pos, p_top_had, p_top_lep)
    analyzer_top = torch.where(is_lep_pos, p_lep, p_prox_down)
    analyzer_atop = torch.where(is_lep_pos, p_prox_down, p_lep)

    k_axis = _normalize(lorentz_boost(top, ttbar)[:, 1:4])
    p_beam = _normalize(lorentz_boost(beam, ttbar)[:, 1:4])

    costheta_kb = (p_beam * k_axis).sum(dim=-1)
    sintheta_kb = torch.sqrt((1.0 - costheta_kb**2).clamp(min=0.0))
    sign_kb = torch.sign(costheta_kb)

    n_axis = _normalize(sign_kb.unsqueeze(-1) * torch.cross(p_beam, k_axis, dim=-1))
    r_axis = _normalize(
        sign_kb.unsqueeze(-1)
        * (p_beam - costheta_kb.unsqueeze(-1) * k_axis)
        / sintheta_kb.clamp(min=EPS).unsqueeze(-1)
    )

    d_top = _normalize(lorentz_boost(analyzer_top, top)[:, 1:4])
    d_atop = _normalize(lorentz_boost(analyzer_atop, atop)[:, 1:4])

    cos_top_n = (n_axis * d_top).sum(dim=-1)
    cos_top_r = (r_axis * d_top).sum(dim=-1)
    cos_atop_n = (n_axis * d_atop).sum(dim=-1)
    cos_atop_r = (r_axis * d_atop).sum(dim=-1)

    q_nr = cos_top_n * cos_atop_r - cos_top_r * cos_atop_n
    return -q_nr
