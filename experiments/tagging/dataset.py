from typing import NamedTuple

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.logger import LOGGER
from experiments.tagging.pseudoscalars import eps_tensor, spin_correlation_nr
from experiments.tagging.top_events import (
    DOWN_TAGS,
    N_HEAD,
    N_SUBJETS,
    PHOTON_PID,
    PID_FEATURES,
    TAIL,
    read_events,
    rest_frame_energy,
)

# arrays of a file with one entry per token, all others have one entry per event
TOKEN_ARRAYS = ("x", "pid")


class TaggingDataset(torch.utils.data.Dataset):
    """A list of torch_geometric graphs."""

    def __init__(self, data_list=()):
        super().__init__()
        self.data_list = list(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TokenDataset(torch.utils.data.Dataset):
    """Events stored as flat token tensors, sliced into torch_geometric graphs on access.

    torch_geometric concatenates the graphs of a batch along the token dimension and keeps the
    event boundaries in batch.ptr. Subclasses read one file into flat arrays (`_read`) and turn
    the selected events into `x`, `scalars` and `labels` (`_build`).
    """

    # truncate every label to the size of the smallest one
    BALANCE = False

    def __init__(self, length=-1, seed=12345, split_fracs=(0.8, 0.1, 0.1), shuffle=True):
        super().__init__()
        self.length = -1 if length is None else length
        self.seed = seed
        self.split_fracs = split_fracs
        self.shuffle = shuffle
        self.filenames = None
        self.splits = {}
        # per-event tensors attached to every graph
        self.extras = {}
        # per token, the index within its event of the token its tagging features are relative to
        self.ref = None

    def load_data(self, data_dir, filenames, network_float64=False, momentum_float64=True):
        """Read `filenames`, a dict from each label to its files or a list of unlabelled files."""
        network_dtype = torch.float64 if network_float64 else torch.float32
        momentum_dtype = torch.float64 if momentum_float64 else torch.float32
        self.filenames = filenames

        groups = list(filenames.values()) if isinstance(filenames, dict) else [filenames]
        sources, source_group = [], []
        for group, paths in enumerate(groups):
            for path in [paths] if isinstance(paths, str) else paths:
                sources.append(self._read(path))
                source_group.append(group)
        counts = np.array([len(source["ntokens"]) for source in sources])
        source_ids, event_ids = self._select_events(counts, np.array(source_group))

        events = _gather_events(sources, source_ids, event_ids)
        events["label"] = np.array(source_group)[source_ids]
        offsets = np.concatenate([[0], np.cumsum(events["ntokens"])])
        position = np.arange(offsets[-1]) - offsets[:-1].repeat(events["ntokens"])
        tokens = _gather_tokens(
            sources, source_ids, event_ids, events["ntokens"], self._reorder(events, position)
        )
        tokens["position"], tokens["offsets"] = position, offsets
        del sources

        self._build(events, tokens, network_dtype, momentum_dtype)
        self.ptr = torch.from_numpy(np.concatenate([[0], np.cumsum(self._ntokens)]))
        self._split_data()
        LOGGER.info(
            f"[{self.__class__.__name__}] {len(self)} events, {self.x.shape[0]} tokens, "
            f"label=1 ratio: {self.labels.double().mean().item():.4f}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lo, hi = self.ptr[idx], self.ptr[idx + 1]
        graph = Data(x=self.x[lo:hi], scalars=self.scalars[lo:hi], label=int(self.labels[idx]))
        if self.ref is not None:
            graph.ref = self.ref[lo:hi]
        for name, value in self.extras.items():
            graph[name] = value[idx]
        return graph

    def get_subset(self, mode):
        if mode not in self.splits:
            raise ValueError(f"Unknown dataset split '{mode}'")
        LOGGER.info(
            f"[{self.__class__.__name__}] mode={mode}, requested_length={self.length}, "
            f"selected={len(self.splits[mode])}/{len(self)}"
        )
        return self.splits[mode]

    def _read(self, path):
        raise NotImplementedError

    def _reorder(self, events, position):
        """Return the position in its file of every token, after reordering `events` in place."""
        return position

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        """Set x, scalars, labels and _ntokens (tokens per event) from the gathered arrays."""
        raise NotImplementedError

    def _permutation(self, num_events):
        gen = torch.Generator().manual_seed(self.seed)
        perm = (
            torch.randperm(num_events, generator=gen) if self.shuffle else torch.arange(num_events)
        )
        return perm.numpy()

    def _select_events(self, counts, source_group):
        """Return the (source, event) indices to use, shuffled within and across labels."""
        num_groups = source_group.max() + 1
        sizes = [counts[source_group == group].sum() for group in range(num_groups)]
        if self.BALANCE:
            sizes = [min(sizes)] * num_groups
        if self.length >= 0:
            sizes = [min(size, self.length // num_groups) for size in sizes]
        assert min(sizes) > 0, f"data.length={self.length} leaves a label without events"

        source_ids, event_ids = [], []
        for group, size in enumerate(sizes):
            ids = np.nonzero(source_group == group)[0]
            offsets = np.concatenate([[0], np.cumsum(counts[ids])])
            selected = self._permutation(offsets[-1])[:size]
            source = np.searchsorted(offsets[1:], selected, side="right")
            source_ids.append(ids[source])
            event_ids.append(selected - offsets[source])

        source_ids, event_ids = np.concatenate(source_ids), np.concatenate(event_ids)
        if num_groups == 1:
            return source_ids, event_ids
        perm = self._permutation(len(source_ids))
        return source_ids[perm], event_ids[perm]

    def _split_data(self):
        fracs = np.asarray(self.split_fracs, dtype=np.float64)
        fracs = fracs / fracs.sum()
        num_events = len(self)
        n_train = int(num_events * fracs[0])
        n_val = int(num_events * fracs[1])
        assert min(n_train, n_val, num_events - n_train - n_val) > 0, (
            f"Split fractions {self.split_fracs} leave an empty part for {num_events} events"
        )
        indices = torch.arange(num_events)
        self.splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }


def _gather_events(sources, source_ids, event_ids):
    """Collect the per-event arrays of the selected events."""
    events = {}
    for name, value in sources[0].items():
        if name in TOKEN_ARRAYS:
            continue
        events[name] = np.empty((len(source_ids),) + value.shape[1:], dtype=value.dtype)
        for i, source in enumerate(sources):
            selected = source_ids == i
            events[name][selected] = source[name][event_ids[selected]]
    return events


def _gather_tokens(sources, source_ids, event_ids, ntokens, source_position):
    """Collect the per-token arrays of the selected events, taking every token from
    `source_position` within its event in the file."""
    num_tokens = ntokens.sum()
    tokens = {
        name: np.empty((num_tokens,) + sources[0][name].shape[1:], dtype=sources[0][name].dtype)
        for name in TOKEN_ARRAYS
        if name in sources[0]
    }
    for i, source in enumerate(sources):
        selected = source_ids == i
        source_offsets = np.concatenate([[0], np.cumsum(source["ntokens"])])
        idx = np.nonzero(selected.repeat(ntokens))[0]
        src = source_offsets[event_ids[selected]].repeat(ntokens[selected]) + source_position[idx]
        for name, value in tokens.items():
            value[idx] = source[name][src]
    return tokens


def _indices(mask):
    return torch.from_numpy(np.nonzero(mask)[0])


def _top_category(tokens, events, n_tail, n_cols, dtype):
    """One-hot token roles shared by the top datasets, in the first five of `n_cols` columns:
    [is_hadtop, is_subjet, sub0, sub1, sub2], where subjets mark their own slot and
    constituents the slot of their subjet."""
    position, ntokens = tokens["position"], events["ntokens"]
    is_subjet = (position >= 1) & (position <= N_SUBJETS)
    is_const = (position >= N_HEAD) & (position < ntokens.repeat(ntokens) - n_tail)
    parent = np.repeat(np.tile(np.arange(N_SUBJETS), len(ntokens)), events["nconst"].ravel())

    category = torch.zeros(len(position), n_cols, dtype=dtype)
    category[_indices(position == 0), 0] = 1
    category[_indices(is_subjet), 1] = 1
    category[_indices(is_subjet), torch.from_numpy(1 + position[is_subjet])] = 1
    category[_indices(is_const), torch.from_numpy(2 + parent)] = 1
    return category


def _tail_tokens(tokens, n_tail, i):
    """Index of the i-th tail token of every event."""
    return torch.from_numpy(tokens["offsets"][1:] - n_tail + i)


def _top_ref(tokens, events, n_tail):
    """Reference tokens of the top datasets: the hadronic top (token 0) for the hadronic side,
    the leptonic top (first tail token) for itself and the lepton (second tail token)."""
    ref = torch.zeros(len(tokens["position"]), dtype=torch.int32)
    toplep_position = torch.from_numpy(events["ntokens"] - n_tail).to(torch.int32)
    for i in range(2):
        ref[_tail_tokens(tokens, n_tail, i)] = toplep_position
    return ref


def _pid_features(pid, dtype):
    """PID_FEATURES rows of all tokens, zero for tokens without a PID."""
    rows = PID_FEATURES[np.where(pid < 0, len(PID_FEATURES) - 1, pid)]
    return torch.from_numpy(rows).to(dtype)


class TopTaggingDataset(TokenDataset):
    """Semileptonic ttbar events with the hadronic top resolved into subjets and constituents.

    Scalars: [is_hadtop, is_subjet, sub0, sub1, sub2] (+ [is_toplep, lepton charge] with
    use_ditop) (+ 6 PID features with pid). Without use_ditop the leptonic top and the lepton
    are no tokens but attached to every graph as lep_vec and lep_charge.
    """

    def __init__(self, length=-1, pid=False, use_ditop=False):
        super().__init__(length=length)
        self.include_pid = pid
        self.use_ditop = use_ditop

    def _read(self, path):
        return read_events(path)

    def _labels(self, events):
        return events["label"]

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        n_tail = len(TAIL)
        lep_charge = torch.from_numpy(events["lep_charge"])
        scalars = _top_category(tokens, events, n_tail, 7 if self.use_ditop else 5, network_dtype)
        if self.use_ditop:
            scalars[_tail_tokens(tokens, n_tail, 0), 5] = 1
            scalars[_tail_tokens(tokens, n_tail, 1), 6] = lep_charge.to(network_dtype)
        if self.include_pid:
            scalars = torch.cat([scalars, _pid_features(tokens["pid"], network_dtype)], dim=-1)

        x = torch.from_numpy(tokens["x"]).to(momentum_dtype)
        ref = _top_ref(tokens, events, n_tail)
        self._ntokens = events["ntokens"]
        if not self.use_ditop:
            is_tail = tokens["position"] >= events["ntokens"].repeat(events["ntokens"]) - n_tail
            self.extras = {
                "lep_vec": x[is_tail].reshape(-1, n_tail, 4),
                "lep_charge": lep_charge.to(torch.long),
            }
            x, scalars, ref = x[~is_tail], scalars[~is_tail], ref[~is_tail]
            self._ntokens = events["ntokens"] - n_tail
        # truth: the harder non-b subjet is the down quark (evaluation only)
        self.extras["down_hard"] = torch.from_numpy(np.isin(events["tags"][:, 1], DOWN_TAGS)).long()
        self.x, self.scalars, self.ref = x, scalars, ref
        self.labels = torch.from_numpy(self._labels(events)).to(torch.long)


class DownTaggingDataset(TopTaggingDataset):
    """Label 1 if the harder non-b subjet (in the top rest frame) has one of the parton tags
    `tag` (down quarks)."""

    def __init__(self, tag=DOWN_TAGS, length=-1, pid=False, use_ditop=False):
        super().__init__(length=length, pid=pid, use_ditop=use_ditop)
        self.tag = tag

    def _labels(self, events):
        return np.isin(events["tags"][:, 1], self.tag)


class PolarizationDataset(TopTaggingDataset):
    """Left- (label 0) against right-handed (label 1) hadronic tops.

    Scalars as in TopTaggingDataset, but the C-odd charges (lepton charge with use_ditop, PID
    charge with pid) move into one trailing charge column, and the lepton gets an is_lepton flag.
    charge_last=False keeps the TopTaggingDataset layout, as the down tagger expects.
    """

    def __init__(self, length=-1, pid=False, use_ditop=False, charge_last=True):
        super().__init__(length=length, pid=pid, use_ditop=use_ditop)
        self.charge_last = charge_last

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        super()._build(events, tokens, network_dtype, momentum_dtype)
        if not self.charge_last or not (self.use_ditop or self.include_pid):
            return
        s = self.scalars
        charge = s.new_zeros(len(s))
        if self.include_pid:
            pid_charge_col = 7 if self.use_ditop else 5
            charge = s[:, pid_charge_col].clone()
            s = torch.cat([s[:, :pid_charge_col], s[:, pid_charge_col + 1 :]], dim=-1)
        if self.use_ditop:
            is_lepton = s[:, 6] != 0
            charge[is_lepton] = s[is_lepton, 6]
            s[:, 6] = is_lepton.to(s.dtype)
        self.scalars = torch.cat([s, charge[:, None]], dim=-1)


class PseudoscalarDataset(TokenDataset):
    """CP-even against CP-odd ttH events, with H -> gamma gamma.

    Scalars: [is_hadtop, is_subjet, sub0, sub1, sub2, is_toplep, is_lepton, is_higgs or
    is_photon, photon0, photon1] (+ 5 PID features with pid, for constituents and photons), then
    the C-odd charge column (lepton charge, and the constituent charges with pid).
    """

    BALANCE = True
    TAIL = TAIL + ("higgs", "photon_leading", "photon_subleading")

    def __init__(self, length=-1, pid=False):
        super().__init__(length=length)
        self.include_pid = pid

    def _read(self, path):
        return read_events(path, tail=self.TAIL)

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        n_tail = len(self.TAIL)
        toplep, lepton, higgs, photon0, photon1 = (
            _tail_tokens(tokens, n_tail, i) for i in range(n_tail)
        )
        scalars = _top_category(tokens, events, n_tail, 10, network_dtype)
        scalars[toplep, 5] = 1
        scalars[lepton, 6] = 1
        for col, photon in enumerate([photon0, photon1]):
            scalars[photon, 7] = 1
            scalars[photon, 8 + col] = 1
        scalars[higgs, 7] = 1

        charge = scalars.new_zeros(len(scalars), 1)
        if self.include_pid:
            pid = tokens["pid"].copy()
            pid[torch.cat([photon0, photon1]).numpy()] = PHOTON_PID
            rows = _pid_features(pid, network_dtype)
            scalars = torch.cat([scalars, rows[:, 1:]], dim=-1)
            charge = rows[:, :1]
        charge[lepton, 0] = torch.from_numpy(events["lep_charge"]).to(network_dtype)
        scalars = torch.cat([scalars, charge], dim=-1)

        # the Higgs is its own reference and that of the photons
        self.ref = _top_ref(tokens, events, n_tail)
        higgs_position = torch.from_numpy(events["ntokens"] - n_tail + 2).to(torch.int32)
        for token in [higgs, photon0, photon1]:
            self.ref[token] = higgs_position

        self.x = torch.from_numpy(tokens["x"]).to(momentum_dtype)
        self.scalars = scalars
        self.labels = torch.from_numpy(events["label"]).to(torch.long)
        self._ntokens = events["ntokens"]


class VBFDataset(TokenDataset):
    """VBF parton-level events of three particles each, one file per label."""

    def _read(self, path):
        x = np.load(path)
        return {"x": x.reshape(-1, 4), "ntokens": np.full(len(x), x.shape[1])}

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        n_particles = events["ntokens"][0]
        self.x = torch.from_numpy(tokens["x"]).to(momentum_dtype)
        self.scalars = torch.eye(n_particles, dtype=network_dtype).repeat(len(events["ntokens"]), 1)
        self.labels = torch.from_numpy(events["label"]).to(torch.long)
        self._ntokens = events["ntokens"]


# tt_smeft scalars: [category(7), pid(5), charge, eps, nr], with the CP-odd columns last.
# category = [is_hadtop, is_subjet, sub0, sub1, sub2, is_toplep, is_lepton]
# pid = [is_electron, is_muon, is_photon, is_charged_hadron, is_neutral_hadron]
TTSMEFT_N_CATEGORY = 7
TTSMEFT_N_PID = 5
TTSMEFT_N_GLOBAL_PS = 2  # eps_tensor, spin_correlation_nr


class TTSmeftLayout(NamedTuple):
    """Column layout of the tt_smeft ``scalars`` tensor.

    The last ``n_ps`` columns are the CP-odd ones: the charge (C-odd), then the jet-level
    pseudoscalars (P-odd, with ``include_ps``).
    """

    pid: bool = False
    include_ps: bool = False

    @property
    def n_scalars(self):
        return TTSMEFT_N_CATEGORY + (TTSMEFT_N_PID if self.pid else 0)

    @property
    def n_ps(self):
        # the charge column always holds the lepton charge, and the constituent charges with pid
        return 1 + (TTSMEFT_N_GLOBAL_PS if self.include_ps else 0)

    @property
    def n_cols(self):
        return self.n_scalars + self.n_ps

    @property
    def charge_col(self):
        return self.n_scalars


class TTSmeftDataset(TokenDataset):
    """CP-even vs CP-odd SMEFT ttbar events.

    With tag_down the down subjet takes the first non-b slot, otherwise the non-b subjets keep
    their top rest frame energy order and the pseudoscalars use the harder one in place of the down
    subjet.
    """

    BALANCE = True

    def __init__(self, length=-1, pid=False, include_ps=False, tag_down=True):
        super().__init__(length=length)
        self.include_pid = pid
        self.include_ps = include_ps
        self.tag_down = tag_down
        self.layout = TTSmeftLayout(pid=pid, include_ps=include_ps)

    def _read(self, path):
        return read_events(path)

    def _reorder(self, events, position):
        if not self.tag_down:
            return position
        swap = ~np.isin(events["tags"][:, 1], DOWN_TAGS)
        source = _swapped_positions(position, events["nconst"], events["ntokens"], swap)
        for name in ["nconst", "tags"]:
            events[name][swap] = events[name][swap][:, [0, 2, 1]]
        return source

    def _build(self, events, tokens, network_dtype, momentum_dtype):
        layout, offsets, n_tail = self.layout, tokens["offsets"], len(TAIL)
        toplep, lepton = _tail_tokens(tokens, n_tail, 0), _tail_tokens(tokens, n_tail, 1)
        scalars = _top_category(tokens, events, n_tail, layout.n_cols, network_dtype)
        scalars[toplep, 5] = 1
        scalars[lepton, 6] = 1

        if layout.pid:
            rows = _pid_features(tokens["pid"], network_dtype)
            scalars[:, TTSMEFT_N_CATEGORY : TTSMEFT_N_CATEGORY + TTSMEFT_N_PID] = rows[:, 1:]
            scalars[:, layout.charge_col] = rows[:, 0]
        lep_charge = torch.from_numpy(events["lep_charge"])
        scalars[lepton, layout.charge_col] = lep_charge.to(network_dtype)

        x = torch.from_numpy(tokens["x"]).to(momentum_dtype)
        if layout.include_ps:
            down_slot = (
                np.where(np.isin(events["tags"][:, 1], DOWN_TAGS), 2, 3) if self.tag_down else 2
            )
            args = (x[offsets[:-1]], x[toplep], x[offsets[:-1] + down_slot], x[lepton])
            pseudoscalars = torch.stack(
                [
                    eps_tensor(*args, dtype=momentum_dtype),
                    spin_correlation_nr(*args, lep_charge.to(momentum_dtype), dtype=momentum_dtype),
                ],
                dim=-1,
            ).to(network_dtype)
            # jet-level, so the same value on every token of an event
            scalars[:, -TTSMEFT_N_GLOBAL_PS:] = pseudoscalars.repeat_interleave(
                torch.from_numpy(events["ntokens"]), dim=0
            )

        self.x, self.scalars = x, scalars
        self.ref = _top_ref(tokens, events, n_tail)
        self.labels = torch.from_numpy(events["label"]).to(torch.long)
        self._ntokens = events["ntokens"]


def _swapped_positions(position, nconst, ntokens, swap):
    """Position in the file of every token, when the events in `swap` exchange their two non-b
    subjets (and their constituent blocks) with respect to the file."""
    if not swap.any():
        return position
    swapped = swap.repeat(ntokens)
    n_b, n_first, n_second = (nconst[:, i].repeat(ntokens) for i in range(N_SUBJETS))
    const = position - N_HEAD
    is_const = (position >= N_HEAD) & (const < n_b + n_first + n_second)

    source = np.where(swapped & (position == 2), 3, position)
    source = np.where(swapped & (position == 3), 2, source)
    shifted = np.where(
        const < n_b, const, np.where(const < n_b + n_second, const + n_first, const - n_second)
    )
    return np.where(swapped & is_const, N_HEAD + shifted, source)


def _ttsmeft_role_masks(scalars):
    is_jet = scalars[:, 0] == 1
    is_subjet = scalars[:, 1] == 1
    is_b = scalars[:, 2] == 1
    is_down = scalars[:, 3] == 1
    is_other = scalars[:, 4] == 1
    is_toplep = scalars[:, 5] == 1
    is_lep = scalars[:, 6] == 1
    return is_jet, is_subjet, is_b, is_down, is_other, is_toplep, is_lep


def ttsmeft_graph_to_downtag_input(graph, layout, pid, network_dtype):
    """Rebuild a tt_smeft graph in the format the down-tagger was trained on:
    non-b subjets ordered by energy in the top rest frame, no leptonic side."""
    if pid and not layout.pid:
        raise ValueError(
            "Down-tagger expects PID features, but the tt_smeft graph was built "
            "without them. Load the tt_smeft dataset with pid=True."
        )

    x, scalars = graph.x, graph.scalars
    _, is_subjet, _, is_down, is_other, is_toplep, is_lep = _ttsmeft_role_masks(scalars)

    down_row = torch.nonzero(is_subjet & is_down, as_tuple=True)[0].item()
    other_row = torch.nonzero(is_subjet & is_other, as_tuple=True)[0].item()
    hadtop = x[torch.nonzero(scalars[:, 0] == 1, as_tuple=True)[0]]
    e_down, e_other = rest_frame_energy(x[[down_row, other_row]][None], hadtop)[0]
    down_is_leading = (e_down >= e_other).item()

    new_category = torch.zeros(scalars.shape[0], 5, dtype=network_dtype)
    new_category[:, :3] = scalars[:, :3]
    down_col, other_col = (3, 4) if down_is_leading else (4, 3)
    new_category[:, 3] = scalars[:, down_col]
    new_category[:, 4] = scalars[:, other_col]

    keep = ~(is_toplep | is_lep)
    new_scalars = new_category[keep]
    if pid:
        pid_lo = TTSMEFT_N_CATEGORY
        # the down-tagger's pid encoding leads with the charge
        new_scalars = torch.cat(
            [
                new_scalars,
                scalars[keep, layout.charge_col, None].to(network_dtype),
                scalars[keep, pid_lo : pid_lo + TTSMEFT_N_PID].to(network_dtype),
            ],
            dim=-1,
        )

    downtag_graph = Data(
        x=x[keep],
        ref=graph.ref[keep],
        scalars=new_scalars,
        label=torch.tensor([float(down_is_leading)], dtype=network_dtype),
        lep_vec=torch.cat([x[is_toplep], x[is_lep]], dim=0),
        lep_charge=scalars[is_lep, layout.charge_col].to(network_dtype),
    )
    return downtag_graph, down_is_leading


def relabel_ttsmeft_graph(graph, swap, layout, network_dtype, momentum_dtype):
    """Swap the down/other subjet labels of a tt_smeft graph and refresh its pseudoscalars."""
    # the graphs are views into the dataset tensors, so they are cloned before being stored
    if not swap:
        return graph.clone()

    x, scalars = graph.x, graph.scalars
    is_jet, is_subjet, _, _, is_other, is_toplep, is_lep = _ttsmeft_role_masks(scalars)

    new_scalars = scalars.clone()
    new_scalars[:, 3] = scalars[:, 4]
    new_scalars[:, 4] = scalars[:, 3]

    if layout.include_ps:
        # the jet-level pseudoscalars depend on which subjet is the down quark
        new_down_four = x[is_subjet & is_other]
        args = (x[is_jet], x[is_toplep], new_down_four, x[is_lep])
        new_scalars[:, -TTSMEFT_N_GLOBAL_PS:] = torch.cat(
            [
                eps_tensor(*args, dtype=momentum_dtype),
                spin_correlation_nr(
                    *args, scalars[is_lep, layout.charge_col], dtype=momentum_dtype
                ),
            ]
        ).to(network_dtype)

    return Data(x=x.clone(), ref=graph.ref.clone(), scalars=new_scalars, label=graph.label)
