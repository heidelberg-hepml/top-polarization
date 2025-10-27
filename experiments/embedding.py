import torch
from lgatr.interface import embed_vector, get_spurions

from experiments.utils import get_batch_from_ptr


def embed_data_into_ga(fourmomenta, scalars, ptr, ga_cfg=None):
    """
    Embed data into geometric algebra representation
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
    ga_cfg: settings for embedding

    Returns
    -------
    multivectors: torch.tensor of shape (n_particles (+ n_spurions*batchsize)), n_ga_features, n_channels)
        Embedded multivectors, with the spurions added as tokens or channels as specified
    scalars: torch.tensor of shape (n_particles (+ n_spurions*batchsize), n_features (+ n_spurions))
        Scalars with spurions one-hot encoding added if specified
    batch: torch.tensor of shape (n_particles (+ n_spurions*batchsize))
        Batch indices for each particle
    mask: torch.tensor of shape (n_particles (+ n_spurions*batchsize))
        Mask indicating which entries are spurions (False for spurions, True for original data)
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)
    new_ptr = ptr.clone()

    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    if ga_cfg is not None:
        spurions = get_spurions(
            ga_cfg.beam_spurion,
            ga_cfg.add_time_spurion,
            ga_cfg.beam_mirror,
            fourmomenta.device,
            fourmomenta.dtype,
        )
        n_spurions = spurions.shape[0]
        if getattr(ga_cfg, "spurion_channels", False):
            spurions = spurions.unsqueeze(0).repeat(multivectors.shape[0], 1, 1)
            multivectors = torch.cat([multivectors, spurions], dim=1)
            mask = torch.zeros(
                multivectors.shape[0],
                dtype=torch.bool,
                device=multivectors.device,
            )

        else:
            spurion_idxs = torch.stack(
                [ptr[:-1] + i for i in range(n_spurions)], dim=0
            ) + n_spurions * torch.arange(batchsize, device=ptr.device)

            spurion_idxs = spurion_idxs.permute(1, 0).flatten()
            insert_spurion = torch.zeros(
                multivectors.shape[0] + n_spurions * batchsize,
                dtype=torch.bool,
                device=multivectors.device,
            )
            insert_spurion[spurion_idxs] = True
            multivectors_buffer = multivectors.clone()
            multivectors = torch.empty(
                insert_spurion.shape[0],
                *multivectors.shape[1:],
                dtype=multivectors.dtype,
                device=multivectors.device,
            )
            multivectors[~insert_spurion] = multivectors_buffer
            multivectors[insert_spurion] = spurions.repeat(batchsize, 1).unsqueeze(-2)

            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                multivectors.shape[0],
                scalars_buffer.shape[1],
                dtype=scalars.dtype,
                device=scalars.device,
            )
            scalars[~insert_spurion] = scalars_buffer
            if hasattr(ga_cfg, "spurion_channels"):
                scalars = torch.cat(
                    [scalars, insert_spurion.to(scalars.dtype).unsqueeze(-1)], dim=-1
                )
            new_ptr[1:] = new_ptr[1:] + (arange + 1) * n_spurions

            mask = insert_spurion
    else:
        mask = torch.zeros(
            multivectors.shape[0],
            dtype=torch.bool,
            device=multivectors.device,
        )

    batch = get_batch_from_ptr(new_ptr)

    return multivectors, scalars, batch, mask


def add_jet_to_sequence(batch):
    """
    Add exactly one token per sequence in both x_gen and x_det (the jet four-momentum),
    and append exactly one new scalars_gen/det channel that flags those tokens.

    Parameters
    ----------
    batch: Batch object
        The original batch to modify
    Returns
    -------
    new_batch: Batch object
        The modified batch with jet tokens added for x_gen and x_det
    ~insert_gen_jets: torch.tensor of shape (gen_mult + batchsize)
        Mask indicating which entries in x_gen are original data (True) vs inserted jet tokens (False)
    ~insert_det_jets: torch.tensor of shape (det_mult + batchsize)
        Mask indicating which entries in x_det are original data (True) vs inserted jet tokens (False)
    """
    new_batch = batch.clone()

    batchsize = len(new_batch.ptr) - 1
    arange = torch.arange(batchsize, device=new_batch.x.device)

    jets_idx = new_batch.ptr[:-1] + arange

    insert_jets = torch.zeros(
        new_batch.x.shape[0] + batchsize,
        dtype=torch.bool,
        device=new_batch.x.device,
    )
    insert_jets[jets_idx] = True
    x = torch.empty(
        insert_jets.shape[0],
        *new_batch.x.shape[1:],
        dtype=new_batch.x.dtype,
        device=new_batch.x.device,
    )
    x[~insert_jets] = new_batch.x

    x[insert_jets] = new_batch.jet
    scalars = torch.zeros(
        x.shape[0],
        new_batch.scalars.shape[1],
        dtype=new_batch.scalars.dtype,
        device=new_batch.scalars.device,
    )
    scalars[~insert_jets] = new_batch.scalars
    scalars = torch.cat([scalars, insert_jets.unsqueeze(-1).to(scalars.dtype)], dim=-1)

    new_batch.x = x
    new_batch.scalars = scalars
    new_batch.ptr[1:] = new_batch.ptr[1:] + (arange + 1)
    new_batch.batch = get_batch_from_ptr(new_batch.ptr)

    new_batch.x = x
    new_batch.scalars = scalars
    new_batch.ptr[1:] = new_batch.ptr[1:] + (arange + 1)
    new_batch.batch = get_batch_from_ptr(new_batch.ptr)

    return new_batch, ~insert_jets
