import torch
from lloca.utils.lorentz import lorentz_squarednorm
from lloca.utils.polar_decomposition import restframe_boost


def likelihood_calculator(cfg, results, momenta):

    labels = torch.stack((results["test"]["labels_true"], results["test"]["labels_predict"]), axis=-1)

    if cfg.data.dataset == "polarized":

        neg_pred = 2*labels[labels[:, 0] == 0, -1]-1
        pos_pred = 2*labels[labels[:, 0] == 1, -1]-1
        beta_l = (neg_pred < 0).float().mean()
        beta_r = (pos_pred >= 0).float().mean()

    elif cfg.data.dataset == "unpolarized":

        jets = torch.stack((momenta["test"]["q_b"], momenta["test"]["q_hard"], momenta["test"]["q_soft"]), axis=-1)
        beta = pol_opt(labels[:, 1], jets)
        beta_l = torch.mean(beta[labels[:, 0] == 0])
        beta_r = torch.mean(beta[labels[:, 0] == 1])
    
    return beta_l, beta_r


def pol_opt(p, q):

    top_mom = torch.sum(q, axis=-1)[:, 1:4]
    top_e = torch.sqrt(173.5**2+torch.linalg.norm(top_mom, axis=1)**2)[:, None]
    top_fourmom = torch.concatenate((top_e, top_mom), axis=1)
    top_rest = restframe_boost(top_fourmom)
    p = p[:, None]
    q_hard, q_soft = torch.einsum("ijk,ik->ij", top_rest, q[:, :, 1]), torch.einsum("ijk,ik->ij", top_rest, q[:, :, 2])
    q_hard, q_soft = q_hard[:, 1:4], q_soft[:, 1:4]
    qhat_hard, qhat_soft = q_hard / torch.linalg.norm(q_hard, axis=1, keepdims=True), q_soft / torch.linalg.norm(q_soft, axis=1, keepdims=True)
    q_opt = p * qhat_hard + (1-p) * qhat_soft
    
    return torch.linalg.norm(q_opt, axis=1)


def get_likelihood(vals, bins=40, threshold=3):

    '''
    Not used right now
    '''

    if int(bins) % 2 == 1:
        bins += 1

    def asym(binned):

        Nm, Np = torch.split(binned, 2)
        Nm = torch.flip(Nm)

        sm, sp = torch.sqrt(Nm), torch.sqrt(Np) 

        asym_val = (Np - Nm) / (Np + Nm)
        asym_err = 2 * torch.sqrt( (Np * Nm) / (Np + Nm)**3 )

        return asym_val, asym_err

    neg, pos = vals
    neg_binned, pos_binned = np.histogram(neg, bins=int(bins), range=(-1,1))[0], np.histogram(pos, bins=int(bins), range=(-1,1))[0]
    asym_neg, err_neg = asym(neg_binned)
    asym_pos, err_pos = asym(pos_binned)
