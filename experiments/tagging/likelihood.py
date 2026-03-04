import torch
import numpy as np
from experiments.hep import lorentz_boost


def get_polarization(cfg, results):
    pol = {"efficiency": [], "pol_l": [], "pol_r": [], "score_l": [], "score_r": []}
    labels = torch.stack(
        (results["labels_true"], 
        results["labels_predict"]), 
        axis=-1,
        )
    jets = torch.stack(
        (results["q_hadtop"],
        results["q_b"], 
        results["q_hard"], 
        results["q_soft"], 
        results["q_leptop"],
        results["q_lep"]),
        axis=-1,
    )
    charge = results["charge"]
    # print(f"Example lepton 4-momentum: {results['q_lep'][0]}")
    # print(f"Example leptop 4-momentum: {results['q_leptop'][0]}")
    # print(f"Example hadtop 4-momentum: {results['q_hadtop'][0]}")
    # spin_ax = lorentz_boost(results['q_hadtop'][0].unsqueeze(0), results['q_leptop'][0].unsqueeze(0))
    # pol_vec = lorentz_boost(results['q_lep'][0].unsqueeze(0), results['q_leptop'][0].unsqueeze(0))
    # print(f"Example spin axis: {spin_ax}")
    # print(f"Example pol vec: {pol_vec}")
    # costheta = torch.einsum('i,i->', spin_ax[0][1:4], pol_vec[0][1:4]) / (torch.linalg.norm(spin_ax[0][1:4]) * torch.linalg.norm(pol_vec[0][1:4]))
    # print(f"Exampe costheta value: {costheta}")
    efficiencies = [1., 0.5, 0.2]
    for eff in efficiencies:
        pol["efficiency"].append(eff)
        if cfg.data.dataset == "polarized":
            beta_l, beta_r, score_l, score_r = polarized(labels, efficiency=eff)
        if cfg.data.dataset == "unpolarized":
            beta_l, beta_r, score_l, score_r = unpolarized(labels, jets, charge, efficiency=eff)
        pol["pol_l"].append(beta_l)
        pol["pol_r"].append(beta_r)
        pol["score_l"].append(score_l)
        pol["score_r"].append(score_r)
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
    beta_l, costheta_l = pol_opt(labels_cut[labels_cut[:, 0] == 0][:, -1], jets_cut[labels_cut[:, 0] == 0], charge_cut[labels_cut[:, 0] == 0])
    beta_r, costheta_r = pol_opt(labels_cut[labels_cut[:, 0] == 1][:, -1], jets_cut[labels_cut[:, 0] == 1], charge_cut[labels_cut[:, 0] == 1])
    beta_l, beta_r = torch.mean(beta_l), torch.mean(beta_r)

    return beta_l, beta_r, costheta_l, costheta_r


def cut_efficiency(labels, efficiency=1):
    cut_mask = torch.logical_or(labels[:, -1] >= 1 - (efficiency / 2), labels[:, -1] < (efficiency / 2))
    return cut_mask


def pol_opt(p, mom, sign):
    hadtop, q_hard, q_soft, leptop, lep, = (
        mom[:, :, 0],
        mom[:, :, 2],
        mom[:, :, 3],
        mom[:, :, 4],
        mom[:, :, 5],
    )

    spin_mom = lorentz_boost(hadtop, leptop)
    # spin_mom = lorentz_boost(leptop, hadtop)
    spin_ax = spin_mom[:, 1:4]

    # idx = torch.where(p == 1, 2, 3) # If true label is one, choose harder subjet, else softer subjet
    # idx = idx.view(len(idx), 1, 1).expand(-1, 4, 1)
    # q_down = torch.gather(mom, dim=2, index=idx) # Should be true down quark
    # q_down = q_down.squeeze(-1)
    # print(f"Down quark momentum in lab frame: {q_down}")
    # q_down_t = torch.einsum("ijk,ik->ij", top_rest_frame, q_down)
    # print(f"Down quark momentum in top frame: {q_down_t}")
    # pol_vec = q_down_t[:, 1:4]

    lep_mom_t = lorentz_boost(lep, leptop)
    pol_vec = lep_mom_t[:, 1:4]

    # p = p[:, None]
    # q_hard, q_soft = (
    #     lorentz_boost(q_hard, hadtop),
    #     lorentz_boost(q_soft, hadtop),
    # )
    # q_hard, q_soft = q_hard[:, 1:4], q_soft[:, 1:4]
    # qhat_hard, qhat_soft = (
    #     q_hard / torch.linalg.norm(q_hard, axis=1, keepdims=True),
    #     q_soft / torch.linalg.norm(q_soft, axis=1, keepdims=True),
    # )
    # pol_vec = p * qhat_hard + (1 - p) * qhat_soft
    
    costheta = sign * torch.einsum('ij,ij->i', spin_ax, pol_vec) / (torch.linalg.norm(spin_ax, axis=1) * torch.linalg.norm(pol_vec, axis=1))
    # print(f"Top vector: {hadtop[:, 1:4] / torch.linalg.norm(hadtop[:, 1:4], axis=1, keepdims=True)}")
    # print(f"Anti-top vector: {leptop[:, 1:4] / torch.linalg.norm(leptop[:, 1:4], axis=1, keepdims=True)}")
    # print(f"Polarization vector: {pol_vec / torch.linalg.norm(pol_vec, axis=1, keepdims=True)}")
    print(f"Polarization angle mean: {torch.mean(costheta)}")

    return torch.linalg.norm(pol_vec, axis=1), costheta


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
