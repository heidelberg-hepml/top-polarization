import numpy as np
import torch

from experiments.hep import lorentz_boost


def get_polarization(results, exp):
    pol = {
        "efficiency": [],
        "pol_l": [],
        "pol_r": [],
        "score_l": [],
        "score_r": [],
        "like_ratio": [],
    }

    labels = torch.stack(
        (results["labels_true"], results["labels_predict"]),
        axis=-1,
    )

    if exp == "downtagging":
        jets = torch.stack(
            (
                results["q_hadtop"],
                results["q_b"],
                results["q_hard"],
                results["q_soft"],
                results["q_leptop"],
                results["q_lep"],
            ),
            axis=-1,
        )
        charge = results["charge"]

    efficiencies = np.linspace(0.2, 1.0, num=1001)
    for eff in efficiencies:
        pol["efficiency"].append(eff)
        if exp == "polarization":
            beta_l, beta_r, score_l, score_r = polarized(labels, efficiency=eff)
        elif exp == "downtagging":
            beta_l, beta_r, score_l, score_r = unpolarized(labels, jets, charge, efficiency=eff)
        else:
            raise ValueError(f"Experiment type {exp} not implemented")
        # like_ratio = get_likelihood(score_l, score_r)
        pol["pol_l"].append(beta_l)
        pol["pol_r"].append(beta_r)
        pol["score_l"].append(score_l)
        pol["score_r"].append(score_r)
        # pol["like_ratio"].append(like_ratio)
    return pol


def polarized(labels, efficiency=1):
    mask = cut_efficiency(labels, efficiency=efficiency)
    labels_cut = labels[mask]
    score_l = 2 * labels_cut[labels_cut[:, 0] == 0, 1] - 1
    score_r = 2 * labels_cut[labels_cut[:, 0] == 1, 1] - 1
    beta_l = (score_l < 0).float().mean()
    beta_r = (score_r >= 0).float().mean()

    return beta_l, beta_r, score_l, score_r


def unpolarized(labels, jets, charge, efficiency=1):
    mask = cut_efficiency(labels, efficiency=efficiency)
    labels_cut = labels[mask]
    jets_cut = jets[mask]
    charge_cut = charge[mask]
    beta_l, costheta_l = pol_opt(
        labels_cut[labels_cut[:, 0] == 0][:, -1],
        jets_cut[labels_cut[:, 0] == 0],
        charge_cut[labels_cut[:, 0] == 0],
    )
    beta_r, costheta_r = pol_opt(
        labels_cut[labels_cut[:, 0] == 1][:, -1],
        jets_cut[labels_cut[:, 0] == 1],
        charge_cut[labels_cut[:, 0] == 1],
    )
    beta_l, beta_r = torch.mean(beta_l), torch.mean(beta_r)

    return beta_l, beta_r, costheta_l, costheta_r


def cut_efficiency(labels, efficiency=1):
    cut_mask = torch.logical_or(
        labels[:, -1] >= 1 - (efficiency / 2), labels[:, -1] < (efficiency / 2)
    )
    return cut_mask


def pol_opt(p, mom, sign):
    (
        hadtop,
        q_hard,
        q_soft,
        leptop,
        _,
    ) = (
        mom[:, :, 0],
        mom[:, :, 2],
        mom[:, :, 3],
        mom[:, :, 4],
        mom[:, :, 5],
    )

    spin_mom = lorentz_boost(leptop, hadtop)
    spin_ax = spin_mom[:, 1:4]

    p = p[:, None]
    q_hard, q_soft = (
        lorentz_boost(q_hard, hadtop),
        lorentz_boost(q_soft, hadtop),
    )
    q_hard, q_soft = q_hard[:, 1:4], q_soft[:, 1:4]
    qhat_hard, qhat_soft = (
        q_hard / torch.linalg.norm(q_hard, axis=1, keepdims=True),
        q_soft / torch.linalg.norm(q_soft, axis=1, keepdims=True),
    )
    pol_vec = p * qhat_hard + (1 - p) * qhat_soft

    costheta = (
        sign
        * torch.einsum("ij,ij->i", spin_ax, pol_vec)
        / (torch.linalg.norm(spin_ax, axis=1) * torch.linalg.norm(pol_vec, axis=1))
    )

    return torch.linalg.norm(pol_vec, axis=1), costheta


def get_asymmetry(lp, lt):

    def asym(n_pos, n_neg):
        return (n_pos - n_neg) / (n_pos + n_neg)
    
    def err_asym(n_pos, n_neg):
        return 1 / (n_pos + n_neg)**2 * np.sqrt(2*n_pos**2*n_neg + 2*n_pos*n_neg**2)
    
    score_0 = lp[lt == 0] * 2 - 1
    score_1 = lp[lt == 1] * 2 - 1
    n_pos_0 = torch.sum(score_0 > 0); n_neg_0 = torch.sum(score_0 < 0)
    n_pos_1 = torch.sum(score_1 > 0); n_neg_1 = torch.sum(score_1 < 0)

    asym_0 = asym(n_pos_0, n_neg_0)
    asym_1 = asym(n_pos_1, n_neg_1)
    err_asym_0 = err_asym(n_pos_0, n_neg_0)
    err_asym_1 = err_asym(n_pos_1, n_neg_1)

    return (asym_0, err_asym_0), (asym_1, err_asym_1)


def get_likelihood(left, right, bins=40, threshold=3):
    "Not used right now. Need expected number of events to obtain valid likelihood"

    if int(bins) % 2 == 1:
        bins += 1

    def asym(binned):
        Nm, Np = torch.chunk(binned, 2)
        Nm = torch.flip(Nm, [0])

        asym_val = (Np - Nm) / (Np + Nm)
        asym_err = 2 * torch.sqrt((Np * Nm) / (Np + Nm) ** 3)

        return asym_val, asym_err

    def stat_q(asym_pol, err_pol, asym_unpol, err_unpol):
        q = torch.where(
            asym_pol >= 3,
            (asym_pol - asym_unpol) ** 2 / (err_pol) ** 2
            + torch.log((err_pol) ** 2 / (err_unpol) ** 2),
            0,
        )

        return torch.sum(q)

    l_binned, r_binned = (
        torch.histogram(left, bins=int(bins), range=(-1, 1))[0],
        torch.histogram(right, bins=int(bins), range=(-1, 1))[0],
    )
    unpol_binned = (l_binned + r_binned) / 2
    asym_l, err_l = asym(l_binned)
    asym_r, err_r = asym(r_binned)
    asym_unpol, err_unpol = asym(unpol_binned)

    stat_q_l = stat_q(asym_l, err_l, asym_unpol, err_unpol)
    stat_q_r = stat_q(asym_r, err_r, asym_unpol, err_unpol)
    return (stat_q_l + stat_q_r) / 2
