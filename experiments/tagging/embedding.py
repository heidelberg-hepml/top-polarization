import torch
from lloca.utils.lorentz import lorentz_squarednorm
from lloca.utils.polar_decomposition import restframe_boost
from lloca.utils.utils import get_batch_from_ptr
from torch_geometric.utils import scatter

from experiments.hep import get_eta, get_phi, get_pt

# fourmomenta with all components below this are padding
EPS = 1e-5

TAGGING_FEATURES = (
    "log_pt",
    "log_energy",
    "log_pt_rel",
    "log_energy_rel",
    "dphi",
    "deta",
    "dr",
)

# weaver defaults for tagging features standardization (mean, std)
TAGGING_FEATURES_PREPROCESSING = {
    "log_pt": (1.7, 0.7),
    "log_energy": (2.0, 0.7),
    "log_pt_rel": (-4.7, 0.7),
    "log_energy_rel": (-4.7, 0.7),
    "dphi": (0, 1),
    "deta": (0, 1),
    "dr": (0.2, 4),
    "abs_dphi": (0, 1),
    "abs_deta": (0, 1),
    "dphi_deta": (0, 1),
}

# Subsets selectable through cfg.data.tagging_features
TAGGING_FEATURE_SUBSETS = {
    "all": TAGGING_FEATURES,
    "zinvariant": ("log_pt", "log_pt_rel", "dphi", "deta", "dr"),
    "so3invariant": ("log_energy", "log_energy_rel"),
    "beamsym": (
        "log_pt",
        "log_energy",
        "log_pt_rel",
        "log_energy_rel",
        "abs_dphi",
        "abs_deta",
        "dr",
        "dphi_deta",
    ),
    # beamsym without the parity-odd dphi_deta
    "beamsym_even": (
        "log_pt",
        "log_energy",
        "log_pt_rel",
        "log_energy_rel",
        "abs_dphi",
        "abs_deta",
        "dr",
    ),
    None: (),
}

# pseudoscalars
TAGGING_FEATURES_PARITY_ODD = ("deta", "dphi_deta")


def get_tagging_feature_names(tagging_features="all"):
    """Names of the features returned by get_tagging_features(), in order."""
    try:
        return TAGGING_FEATURE_SUBSETS[tagging_features]
    except KeyError:
        raise ValueError(f"Unknown tagging_features: {tagging_features}") from None


def get_num_tagging_features(tagging_features="all"):
    return len(get_tagging_feature_names(tagging_features))


def get_num_parity_odd_tagging_features(tagging_features="all"):
    names = get_tagging_feature_names(tagging_features)
    return sum(name in TAGGING_FEATURES_PARITY_ODD for name in names)


def split_tagging_features_by_parity(features, tagging_features="all"):
    """Split the tagging features into their parity-even and parity-odd columns."""
    names = get_tagging_feature_names(tagging_features)
    even = [i for i, name in enumerate(names) if name not in TAGGING_FEATURES_PARITY_ODD]
    odd = [i for i, name in enumerate(names) if name in TAGGING_FEATURES_PARITY_ODD]
    return features[..., even], features[..., odd]


def embed_tagging_data(fourmomenta, scalars, ptr, cfg_data, n_ps=0, split_parity=False, ref=None):
    """
    Embed tagging data
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    scalars: torch.tensor of shape (n_particles, n_features)
        Optional scalar features, n_features=0 is possible
    ptr: torch.tensor of shape (batchsize+1)
        Indices of the first particle for each jet
        Also includes the first index after the batch ends
    cfg_data: settings for embedding
    n_ps: int
        Number of trailing `scalars` columns holding parity-odd features.
    split_parity: bool
        Move the n_ps trailing `scalars` columns and the parity-odd tagging features into
        embedding["pseudoscalars"], for models with a pseudoscalar stream.
    ref: torch.tensor of shape (n_particles) or None
        Index within its event of the token each particle's tagging features are taken
        relative to. None takes them relative to the sum of all particles of the event.

    Returns
    -------
    embedding: dict
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)
    ptr = ptr.clone()
    if ref is not None:
        # before cropping, which can drop the reference tokens
        ref = fourmomenta[ptr[get_batch_from_ptr(ptr)] + ref.long()]

    # crop jets to max_particles
    if cfg_data.max_particles is not None:
        counts = ptr[1:] - ptr[:-1]
        kept_counts = counts.clamp_max(cfg_data.max_particles)
        if kept_counts.max() > 0:
            kept_idx = torch.repeat_interleave(
                torch.arange(len(kept_counts), device=ptr.device), kept_counts
            )
            segment_start = torch.cat([kept_counts.new_zeros(1), kept_counts.cumsum(dim=0)[:-1]])
            start_offsets = torch.repeat_interleave(segment_start, kept_counts)
            local_idx = torch.arange(kept_counts.sum(), device=ptr.device) - start_offsets
            offsets = ptr[kept_idx] + local_idx
            fourmomenta = fourmomenta.index_select(0, offsets)
            scalars = scalars.index_select(0, offsets)
            if ref is not None:
                ref = ref.index_select(0, offsets)
            ptr = torch.cat([ptr.new_zeros(1), kept_counts.cumsum(0)])

    # beam reference
    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    spurions *= cfg_data.spurion_scale

    n_spurions = spurions.shape[0]
    is_spurion = torch.zeros(
        fourmomenta.shape[0] + n_spurions * batchsize,
        dtype=torch.bool,
        device=fourmomenta.device,
    )
    if n_spurions > 0:
        # prepend spurions to the token list (within each block)
        spurion_idxs = torch.stack(
            [ptr[:-1] + i for i in range(n_spurions)], dim=0
        ) + n_spurions * torch.arange(batchsize, device=ptr.device)
        spurion_idxs = spurion_idxs.permute(1, 0).flatten()
        is_spurion[spurion_idxs] = True
        fourmomenta_buffer = fourmomenta.clone()
        fourmomenta = torch.empty(
            is_spurion.shape[0],
            *fourmomenta.shape[1:],
            dtype=fourmomenta.dtype,
            device=fourmomenta.device,
        )
        fourmomenta[~is_spurion] = fourmomenta_buffer
        fourmomenta[is_spurion] = spurions.repeat(batchsize, 1)

        scalars_buffer = scalars.clone()
        scalars = torch.zeros(
            fourmomenta.shape[0],
            scalars.shape[1],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        scalars[~is_spurion] = scalars_buffer
        if ref is not None:
            ref_buffer = ref
            ref = fourmomenta.clone()
            ref[~is_spurion] = ref_buffer
        ptr[1:] = ptr[1:] + (arange + 1) * n_spurions

    # add mass regulator
    if cfg_data.mass_reg is not None:
        mass_reg = cfg_data.mass_reg
        mask = lorentz_squarednorm(fourmomenta) < mass_reg**2
        fourmomenta[mask, 0] = (fourmomenta[mask, 1:] ** 2).sum(dim=-1).add(mass_reg**2).sqrt()

    batch = get_batch_from_ptr(ptr)

    if cfg_data.boost_jet:
        # boost to the jet rest frame to avoid large boosts
        jet = scatter(
            fourmomenta[~is_spurion], batch[~is_spurion], dim=0, reduce="sum"
        ).index_select(0, batch)
        jet_boost = restframe_boost(jet)
        fourmomenta = torch.einsum("ijk,ik->ij", jet_boost, fourmomenta)
        if ref is not None:
            ref = torch.einsum("ijk,ik->ij", jet_boost, ref)

    if ref is not None:
        jet = ref
    else:
        jet = scatter(
            fourmomenta[~is_spurion], batch[~is_spurion], dim=0, reduce="sum"
        ).index_select(0, batch)
    tagging_features = get_tagging_features(
        fourmomenta,
        jet,
        tagging_features=cfg_data.tagging_features,
    )
    tagging_features[is_spurion] = 0

    tagging_features = tagging_features.to(scalars.dtype)

    # split off the parity-odd inputs
    if split_parity:
        n_even = scalars.shape[-1] - n_ps
        scalars, pseudoscalars = scalars[..., :n_even], scalars[..., n_even:]
        tagging_features, odd_tagging_features = split_tagging_features_by_parity(
            tagging_features, cfg_data.tagging_features
        )
        pseudoscalars = torch.cat([pseudoscalars, odd_tagging_features], dim=-1)
    else:
        pseudoscalars = scalars[..., :0]

    embedding = {
        "fourmomenta": fourmomenta,
        "scalars": scalars,
        "pseudoscalars": pseudoscalars,
        "is_spurion": is_spurion,
        "tagging_features": tagging_features,
        "batch": batch,
        "ptr": ptr,
    }
    return embedding


def dense_to_sparse_jet(fourmomenta_dense, scalars_dense):
    """
    Transform dense jet into sparse jet

    Parameters
    ----------
    fourmomenta_dense: torch.tensor of shape (batchsize, 4, num_particles_max)
    scalars_dense: torch.tensor of shape (batchsize, num_features, num_particles_max)

    Returns
    -------
    fourmomenta_sparse: torch.tensor of shape (num_particles, 4)
        Fourmomenta for concatenated list of particles of all jets
    scalars_sparse: torch.tensor of shape (num_particles, num_features)
        Scalar features for concatenated list of particles of all jets
    ptr: torch.tensor of shape (batchsize+1)
        Start indices of each jet, this way we don't lose information when concatenating everything
        Starts with 0 and ends with the first non-accessible index (=total number of particles)
    """
    fourmomenta_dense = torch.transpose(fourmomenta_dense, 1, 2)  # (batchsize, num_particles, 4)
    scalars_dense = torch.transpose(scalars_dense, 1, 2)  # (batchsize, num_particles, num_features)

    mask = (fourmomenta_dense.abs() > EPS).any(dim=-1)
    num_particles = mask.sum(dim=-1)
    fourmomenta_sparse = fourmomenta_dense[mask]
    scalars_sparse = scalars_dense[mask]

    ptr = torch.zeros(len(num_particles) + 1, device=fourmomenta_dense.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(num_particles, dim=0)
    return fourmomenta_sparse, scalars_sparse, ptr


def get_spurion(
    beam_reference,
    add_time_reference,
    two_beams,
    device,
    dtype,
):
    """
    Construct spurion

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    device
    dtype

    Returns
    -------
    spurion: torch.tensor with shape (n_spurions, 4)
        spurion embedded as fourmomenta object
    """

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, device=device, dtype=dtype).reshape(1, 4)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 3] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)
    elif beam_reference == "all":
        beam = torch.tensor(
            [
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
            ],
            device=device,
            dtype=dtype,
        )

    elif beam_reference is None:
        beam = torch.empty(0, 4, device=device, dtype=dtype)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=device, dtype=dtype).reshape(1, 4)
    else:
        time = torch.empty(0, 4, device=device, dtype=dtype)

    spurion = torch.cat((beam, time), dim=-2)
    return spurion


def get_tagging_features(fourmomenta, jet, tagging_features="all", eps=1e-10):
    """
    Compute features typically used in jet tagging

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    jet: torch.tensor of shape (n_particles, 4)
        Jet momenta in the shape (E, px, py, pz)
    tagging_features: str
        Which subset of TAGGING_FEATURES to include, see TAGGING_FEATURE_SUBSETS.
    eps: float

    Returns
    -------
    features: torch.tensor of shape (n_particles, n_selected)
        Columns in the order given by get_tagging_feature_names(tagging_features)
    """
    names = get_tagging_feature_names(tagging_features)
    if not names:
        return torch.zeros(
            fourmomenta.shape[0], 0, device=fourmomenta.device, dtype=fourmomenta.dtype
        )

    log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
    log_energy = fourmomenta[..., 0].unsqueeze(-1).clamp(min=eps).log()

    log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
    log_energy_rel = (
        fourmomenta[..., 0].clamp(min=eps).log() - jet[..., 0].clamp(min=eps).log()
    ).unsqueeze(-1)
    phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
    dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
    deta = (eta_4 - eta_jet).unsqueeze(-1)
    dr = torch.sqrt((dphi**2 + deta**2).clamp(min=eps))

    values = {
        "log_pt": log_pt,
        "log_energy": log_energy,
        "log_pt_rel": log_pt_rel,
        "log_energy_rel": log_energy_rel,
        "dphi": dphi,
        "deta": deta,
        "dr": dr,
        "abs_dphi": dphi.abs(),
        "abs_deta": deta.abs(),
        "dphi_deta": dphi * deta,
    }

    features = []
    for name in names:
        mean, factor = TAGGING_FEATURES_PREPROCESSING[name]
        features.append((values[name] - mean) * factor)
    return torch.cat(features, dim=-1)
