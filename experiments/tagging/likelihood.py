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
        pol["pol_l"].append(beta_l.mean())
        pol["pol_r"].append(beta_r.mean())
        pol["score_l"].append(score_l)
        pol["score_r"].append(score_r)
        # pol["like_ratio"].append(like_ratio)
    # per-event beta at efficiency 1, aligned with labels_predict, for the statistical error
    pol["beta_l"], pol["beta_r"] = beta_l, beta_r
    return pol


def polarized(labels, efficiency=1):
    mask = cut_efficiency(labels, efficiency=efficiency)
    labels_cut = labels[mask]
    score_l = 2 * labels_cut[labels_cut[:, 0] == 0, 1] - 1
    score_r = 2 * labels_cut[labels_cut[:, 0] == 1, 1] - 1
    beta_l = (score_l < 0).float()
    beta_r = (score_r >= 0).float()

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

    return beta_l, beta_r, costheta_l, costheta_r


def cut_efficiency(labels, efficiency=1):
    """Keep the fraction `efficiency` of all events with the largest |p - 1/2|.

    Ties at the threshold are all kept, so a constant score keeps every event.
    """
    conf = (labels[:, -1].double() - 0.5).abs()
    return conf >= torch.quantile(conf, 1 - efficiency)


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


M_TOP, M_W = 172.5, 80.4
EFFICIENCIES = np.linspace(0.2, 1.0, 33)  # contains 0.2, 0.5 and 1.0
PROBE_KEYS = ["fraction", "beta_abs", "beta_truth", "beta_truth_err", "beta_truth_raw"]
SLOPE_KEYS = ["slope_cos", "slope_cos_err", "slope_u", "slope_u_err"]


def _unit(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def stack_jets(results):
    """(n, 4, 6) momenta in the order hadtop, b, hard, soft, leptop, lepton."""
    return torch.stack(
        [results[k] for k in ["q_hadtop", "q_b", "q_hard", "q_soft", "q_leptop", "q_lep"]], dim=-1
    )


def probe_frame(jets, charge):
    """Unit vectors of the hard and soft light jet and the spin axis in the hadronic top rest frame.

    The spin axis is the leptonic top direction, signed by the lepton charge (as in `pol_opt`).
    """
    hadtop = jets[:, :, 0]
    hard, soft, leptop = (lorentz_boost(jets[:, :, i], hadtop)[:, 1:4] for i in (2, 3, 4))
    return _unit(hard), _unit(soft), charge[:, None] * _unit(leptop)


def kinematic_p(jets):
    """p(d = hard jet) from the W helicity angle."""
    w = jets[:, :, 2] + jets[:, :, 3]
    b, hard = (_unit(lorentz_boost(jets[:, :, i], w)[:, 1:4]) for i in (1, 2))
    c = -(b * hard).sum(-1)  # cos of the hard jet to -b in the W frame
    f0 = M_TOP**2 / (M_TOP**2 + 2 * M_W**2)

    def rho(c):
        return 0.75 * f0 * (1 - c**2) + 0.375 * (1 - f0) * (1 - c) ** 2

    return rho(c) / (rho(c) + rho(-c))


def probe_report(results, polarized, efficiencies=EFFICIENCIES):
    """Spin analyzing power of the down tagger and of baselines on the same events.

    beta_abs = <|k|> is the draft's beta_d, valid only for a calibrated p. beta_truth =
    <d_true . k / |k|> and beta_truth_raw = <d_true . k> use the true down direction.
    Probes: net (labels_predict), kinematic (c_W), soft (d = softer jet), truth.
    """
    jets, charge = stack_jets(results).double(), results["charge"].double()
    q_hard, q_soft, n_hat = probe_frame(jets, charge)
    d_hard = results["down_hard"].double()
    d_true = torch.where(d_hard[:, None] == 1, q_hard, q_soft)
    probes = {
        "net": results["labels_predict"].double(),
        "kinematic": kinematic_p(jets),
        "soft": torch.zeros_like(d_hard),
        "truth": d_hard,
    }

    keys = PROBE_KEYS
    report = {"efficiencies": [float(e) for e in efficiencies]}
    for name, p in probes.items():
        report[name] = {key: [] for key in keys}
        for eff in efficiencies:
            mask = cut_efficiency(p[:, None], efficiency=eff)
            if mask.sum() < 2:  # an untrained network can leave the cut empty
                for key in keys:
                    report[name][key].append(float("nan"))
                continue
            pm = p[mask, None]
            k = pm * q_hard[mask] + (1 - pm) * q_soft[mask]
            norm = torch.linalg.norm(k, dim=-1)
            dot = (d_true[mask] * k).sum(-1)
            row = {
                "fraction": mask.double().mean().item(),
                "beta_abs": norm.mean().item(),
                "beta_truth": (dot / norm).mean().item(),
                "beta_truth_err": (dot / norm).std().item() / mask.sum().double().sqrt().item(),
                "beta_truth_raw": dot.mean().item(),
            }
            for key in keys:
                report[name][key].append(float(row[key]))
    return report


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
