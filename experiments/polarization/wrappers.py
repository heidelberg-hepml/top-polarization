import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from lgatr.interface import extract_scalar
from experiments.utils import xformers_mask
from experiments.embedding import add_jet_to_sequence, embed_data_into_ga
import experiments.coordinates as c


class PolarizationTransformerWrapper(nn.Module):
    def __init__(self, net, wrapper_cfg, range=None):
        super().__init__()
        self.net = net
        self.wrapper_cfg = wrapper_cfg
        self.use_xformers = torch.cuda.is_available()
        self.aggregation = MeanAggregation()

    def init_coordinates(self):
        self.const_coordinates = getattr(c, self.wrapper_cfg.const_coordinates)(
            **self.wrapper_cfg.const_coordinates_options
        )
        self.condition_const_coordinates = getattr(
            c, self.wrapper_cfg.const_coordinates
        )(**self.wrapper_cfg.const_coordinates_options)
        self.jet_coordinates = getattr(c, self.wrapper_cfg.jet_coordinates)(
            **self.wrapper_cfg.jet_coordinates_options
        )
        self.condition_jet_coordinates = getattr(c, self.wrapper_cfg.jet_coordinates)(
            **self.wrapper_cfg.jet_coordinates_options
        )
        if self.wrapper_cfg.transforms_float64:
            self.const_coordinates.to(torch.float64)
            self.condition_const_coordinates.to(torch.float64)
            self.jet_coordinates.to(torch.float64)
            self.condition_jet_coordinates.to(torch.float64)

    def forward(self, batch):
        if self.wrapper_cfg.add_jet:
            new_batch, const_mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch
        input = torch.cat([new_batch.x, new_batch.scalars], dim=-1)

        mask = xformers_mask(new_batch.batch, materialize=not self.use_xformers)
        attn_kwargs = {"attn_bias" if self.use_xformers else "attn_mask": mask}

        outputs = self.net(input.unsqueeze(0), **attn_kwargs)
        outputs = self.aggregation(outputs, new_batch.batch).squeeze(0)

        return outputs


class PolarizationLGATrWrapper(PolarizationTransformerWrapper):
    """
    L-GATr for polarization
    """

    def __init__(self, GA_config, scalar_inputs, **kwargs):
        super().__init__(**kwargs)
        self.ga_cfg = GA_config
        self.scalar_inputs = scalar_inputs

    def forward(self, batch):
        if self.wrapper_cfg.add_jet:
            new_batch, const_mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch.clone()
            const_mask = torch.ones(
                new_batch.x.shape[0],
                dtype=torch.bool,
                device=new_batch.x.device,
            )

        jets = self.jet_coordinates.x_to_fourmomenta(batch.jet)
        ext_jets = torch.repeat_interleave(jets, batch.ptr.diff(), dim=0)

        fourmomenta = torch.zeros_like(new_batch.x)
        fourmomenta[const_mask] = self.const_coordinates.x_to_fourmomenta(
            new_batch.x[const_mask],
            jet=ext_jets,
            ptr=new_batch.ptr,
        )

        if self.wrapper_cfg.add_jet:
            fourmomenta[~const_mask] = jets

        if len(self.scalar_inputs) > 0:
            scalars = torch.cat(
                [new_batch.scalars, new_batch.x[:, self.scalar_inputs]], dim=-1
            )
        else:
            scalars = new_batch.scalars

        mv, s, batch_idx, _ = embed_data_into_ga(
            new_batch.x,
            scalars,
            new_batch.ptr,
            self.ga_cfg,
        )
        multivector = mv.unsqueeze(0)
        scalars = s.unsqueeze(0)

        mask = xformers_mask(batch_idx, materialize=not self.use_xformers)
        attn_kwargs = {"attn_bias" if self.use_xformers else "attn_mask": mask}
        multivector_outputs, _ = self.net(multivector, scalars=scalars, **attn_kwargs)
        outputs = extract_scalar(multivector_outputs)[0, :, :, 0]
        params = self.aggregation(outputs, index=batch_idx)

        return params
