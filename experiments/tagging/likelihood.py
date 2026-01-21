import torch
import numpy as np
from lloca.utils.polar_decomposition import restframe_boost


def get_polarization(cfg, results, momenta):
    pol = {"efficiency": [], "pol_l": [], "pol_r": []}
    labels = torch.stack(
        (results["test"]["labels_true"], 
        results["test"]["labels_predict"]), 
        axis=-1,
        )
    jets = torch.stack(
        (momenta["test"]["q_b"], 
        momenta["test"]["q_hard"], 
        momenta["test"]["q_soft"]), 
        axis=-1,
    )
    efficiencies = [1., 0.5, 0.2]
    for eff in efficiencies:
        pol["efficiency"].append(eff)
        if cfg.data.dataset == "polarized":
            beta_l, beta_r = polarized(labels, efficiency=eff)
        if cfg.data.dataset == "unpolarized":
            beta_l, beta_r = unpolarized(labels, jets, efficiency=eff)
        pol["pol_l"].append(beta_l)
        pol["pol_r"].append(beta_r)
    return pol


def polarized(labels, efficiency=1):
    mask = cut_efficiency(labels, efficiency=efficiency)
    labels_cut = labels[mask]
    neg_pred = 2 * labels_cut[labels_cut[:, 0] == 0, 1] - 1
    pos_pred = 2 * labels_cut[labels_cut[:, 0] == 1, 1] - 1
    beta_l = (neg_pred < 0).float().mean()
    beta_r = (pos_pred >= 0).float().mean()

    return beta_l, beta_r


def unpolarized(labels, jets, efficiency=1):
    mask = cut_efficiency(labels, efficiency=efficiency)
    labels_cut = labels[mask]
    jets_cut = jets[mask]
    beta = pol_opt(labels_cut[:, -1], jets_cut)
    beta_l = torch.mean(beta[labels_cut[:, 0] == 0])
    beta_r = torch.mean(beta[labels_cut[:, 0] == 1])

    return beta_l, beta_r


def cut_efficiency(labels, efficiency=1):
    cut_mask = torch.logical_or(labels[:, -1] >= 1 - (efficiency / 2), labels[:, -1] < (efficiency / 2))
    return cut_mask


def pol_opt(p, q):
    top_mom = torch.sum(q, axis=-1)[:, 1:4]
    top_e = torch.sqrt(173.5**2 + torch.linalg.norm(top_mom, axis=1) ** 2)[:, None]
    top_fourmom = torch.concatenate((top_e, top_mom), axis=1)
    top_rest = restframe_boost(top_fourmom)
    p = p[:, None]
    q_hard, q_soft = (
        torch.einsum("ijk,ik->ij", top_rest, q[:, :, 1]),
        torch.einsum("ijk,ik->ij", top_rest, q[:, :, 2]),
    )
    q_hard, q_soft = q_hard[:, 1:4], q_soft[:, 1:4]
    qhat_hard, qhat_soft = (
        q_hard / torch.linalg.norm(q_hard, axis=1, keepdims=True),
        q_soft / torch.linalg.norm(q_soft, axis=1, keepdims=True),
    )
    q_opt = p * qhat_hard + (1 - p) * qhat_soft

    return torch.linalg.norm(q_opt, axis=1)


def get_likelihood(vals, bins=40, threshold=3):
    """
    Not used right now
    """

    if int(bins) % 2 == 1:
        bins += 1

    def asym(binned):
        Nm, Np = torch.split(binned, 2)
        Nm = torch.flip(Nm)

        sm, sp = torch.sqrt(Nm), torch.sqrt(Np)

        asym_val = (Np - Nm) / (Np + Nm)
        asym_err = 2 * torch.sqrt((Np * Nm) / (Np + Nm) ** 3)

        return asym_val, asym_err

    neg, pos = vals
    neg_binned, pos_binned = (
        np.histogram(neg, bins=int(bins), range=(-1, 1))[0],
        np.histogram(pos, bins=int(bins), range=(-1, 1))[0],
    )
    asym_neg, err_neg = asym(neg_binned)
    asym_pos, err_pos = asym(pos_binned)
