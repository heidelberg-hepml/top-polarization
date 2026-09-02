import hashlib
import json
import os

import awkward as ak
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from experiments.hep import PtPhiEtaM_to_EPPP
from experiments.logger import LOGGER
from experiments.tagging.pseudoscalars import eps_tensor, spin_correlation_nr

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage]
)  # Needed to cache sequence data

EPS = 1e-5


class TaggingDataset(torch.utils.data.Dataset):
    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    scalars : empty placeholder
        label : torch.tensor of shape (1), dtype torch.int
        Binary label supplied by the concrete dataset implementation
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "tagging"

    def __init__(self, length=-1):
        super().__init__()
        self.data_list = []
        self.length = length

    def load_data(self, filename, mode):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _hash_cache_key(self, cache_key) -> str:
        key_bytes = json.dumps(cache_key, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_bytes).hexdigest()

    def _cache_path(self, cache_hash: str, data_dir: str) -> str:
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.CACHE_PREFIX}_{cache_hash}.pt")

    def _save_to_cache(self, cache_key, data, indices, data_dir):
        cache_hash = self._hash_cache_key(cache_key)
        path = self._cache_path(cache_hash, data_dir)
        try:
            torch.save(
                {"key": cache_key, "data": data, "indices": indices},
                path,
            )
        except Exception as e:
            LOGGER.warning(f"[{self.__class__.__name__}] failed to save cache to {path}: {e}")

    def _load_from_cache(self, cache_key, data_dir):
        cache_hash = self._hash_cache_key(cache_key)
        path = self._cache_path(cache_hash, data_dir)

        if not os.path.exists(path):
            return None

        cached = torch.load(path)
        return cached["data"], cached["indices"]

    def _get_target_length(self, num_events):
        if self.length is None or self.length < 0:
            return num_events
        if self.length == 0:
            raise ValueError("Requested dataset length must be positive")
        return min(int(self.length), num_events)

    @classmethod
    def _normalize_filename_key(cls, filename):
        if isinstance(filename, (list, tuple)):
            return tuple(cls._normalize_filename_key(f) for f in filename)
        elif isinstance(filename, (dict)):
            return tuple(sorted((k, cls._normalize_filename_key(v)) for k, v in filename.items()))
        return filename


class HEPTaggingDataset(TaggingDataset):
    """
    Optimized dataset for HEP classification tasks.
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "hep_tagging"

    _parquet_cache = {}
    _split_cache = {}

    def __init__(
        self,
        length=-1,
        seed=12345,
        split_fracs=(0.8, 0.1, 0.1),
        shuffle=True,
        use_cache=True,
    ):
        super().__init__(length=length)
        self.seed = seed
        self.split_fracs = split_fracs
        self.shuffle = shuffle
        self.filename = None
        self.cache_key = {}
        self.use_cache = use_cache

    @staticmethod
    def pid_encoding(pids: torch.Tensor) -> torch.Tensor:
        """
        Convert PDG PIDs to a 6-dimensional encoded representation.

        Encoding:
        [charge, is_electron, is_muon, is_photon, is_charged_hadron, is_neutral_hadron]
        """

        pid_map = {
            22: [0, 0, 0, 1, 0, 0],  # photon
            11: [-1, 1, 0, 0, 0, 0],  # e-
            -11: [1, 1, 0, 0, 0, 0],  # e+
            13: [-1, 0, 1, 0, 0, 0],  # mu-
            -13: [1, 0, 1, 0, 0, 0],  # mu+
            211: [1, 0, 0, 0, 1, 0],  # pi+
            -211: [-1, 0, 0, 0, 1, 0],  # pi-
            321: [1, 0, 0, 0, 1, 0],  # K+
            -321: [-1, 0, 0, 0, 1, 0],  # K-
            130: [0, 0, 0, 0, 0, 1],  # K0_L
            2212: [1, 0, 0, 0, 1, 0],  # proton
            -2212: [-1, 0, 0, 0, 1, 0],  # anti-proton
            2112: [0, 0, 0, 0, 0, 1],  # neutron
            -2112: [0, 0, 0, 0, 0, 1],  # anti-neutron
        }

        pids_int = pids.to(torch.int64)
        encoded = torch.zeros(*pids.shape, 6, device=pids.device, dtype=pids.dtype)

        for pid, encoding in pid_map.items():
            mask = pids_int == pid
            if mask.any():
                encoded[mask] = torch.tensor(encoding, device=pids.device, dtype=pids.dtype)

        return encoded

    def load_data(self, data_dir, filename, network_float64=False, momentum_float64=True):
        network_dtype = torch.float64 if network_float64 else torch.float32
        momentum_dtype = torch.float64 if momentum_float64 else torch.float32

        self.filename = filename

        # Load data with caching
        parquet, num_events = self._get_parquet()
        target_length = self._get_target_length(num_events)
        self.cache_key = self._cache_key_gen(target_length)

        if self.use_cache:
            cache = self._load_from_cache(self.cache_key, data_dir)
        else:
            cache = None

        if cache is None:
            data_list, indices = self._fill_data(parquet, network_dtype, momentum_dtype)

            if not data_list:
                raise RuntimeError(f"No usable events found for {self.filename}")

            self._save_to_cache(self.cache_key, data_list, indices, data_dir)

            LOGGER.info(f"[{self.__class__.__name__}] saved to cache")

        else:
            data_list, indices = cache

            LOGGER.info(f"[{self.__class__.__name__}] loaded from cache")

        self.data_list = data_list
        self._split_data()

    def _shuffle_data(self, data, evt_types=-1):
        # target_length = num_events if self.length is None else min(self.length, num_events)
        target_length = self._get_target_length(len(data))
        if evt_types > 0 and target_length < len(data):
            target_length = int(target_length / evt_types)

        if target_length == 0:
            raise RuntimeError("No events found in loaded data")

        # Generate permutation
        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(data), generator=gen) if self.shuffle else torch.arange(len(data))
        perm = perm[:target_length]

        if isinstance(data, (list)):
            return [data[x] for x in perm.tolist()], perm

        return data[perm.tolist()], perm

    def _split_data(self):
        split_indices = torch.arange(len(self.data_list), dtype=torch.long)

        if self.cache_key not in self._split_cache:
            # Calculate split sizes
            fracs = torch.as_tensor(self.split_fracs, dtype=torch.double)
            fracs = fracs / fracs.sum()

            n_train = int(len(self.data_list) * fracs[0].item())
            n_val = int(len(self.data_list) * fracs[1].item())
            n_test = len(self.data_list) - n_train - n_val

            if min(n_train, n_val, n_test) <= 0:
                raise ValueError(
                    f"Split fractions result in empty dataset parts. "
                    f"num_events={len(self.data_list)}, fractions={self.split_fracs}"
                )

            self._split_cache[self.cache_key] = {
                "train": split_indices[:n_train],
                "val": split_indices[n_train : n_train + n_val],
                "test": split_indices[n_train + n_val :],
            }

    def get_subset(self, mode):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split '{mode}'")

        if self.cache_key not in self._split_cache:
            raise KeyError(f"{self.cache_key} not found in cache")

        split_indices = self._split_cache[self.cache_key][mode]

        LOGGER.info(
            f"[{self.__class__.__name__}] mode={mode}, requested_length={self.length}, "
            f"selected={len(split_indices)}/{len(self.data_list)}"
        )

        return split_indices

    def _cache_key_gen(self):
        raise NotImplementedError()

    def _get_parquet(self):
        raise NotImplementedError()

    def _fill_data(self):
        raise NotImplementedError()

    def _build_event_graph(self):
        raise NotImplementedError()


class DownTaggingDataset(HEPTaggingDataset):
    """
    Dataset for tagging down quarks from the two non-b-tagged subjets in a binary classification task.
    Parameters:
        tag: PID of the quark to tag. Default 1 (= down quark)
        use_ditop: If set to true, also include the lepton and the leptonic top quark in the event/graph
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "down_quark_tagging"

    def __init__(
        self,
        tag=[-3, -1, 1, 3],
        length=-1,
        pid=False,
        use_ditop=False,
        use_cache=True,
    ):
        super().__init__(length=length, use_cache=use_cache)
        self.tag = tag
        self.include_pid = pid
        self.use_ditop = use_ditop

    def _cache_key_gen(self, target_length):
        cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            bool(self.use_ditop),
            target_length,
        )
        return cache_key

    def _get_parquet(self):
        parquet = ak.concatenate(
            [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in self.filename]
        )
        num_events = len(parquet)
        return parquet, num_events

    def _fill_data(self, parquet, network_dtype, momentum_dtype):
        data_list = []
        indices = []
        label_counts = 0

        parquet, perm = self._shuffle_data(parquet)
        for event in parquet:
            graph = self._build_event_graph(event, network_dtype, momentum_dtype)
            label_counts += graph.label.item()
            data_list.append(graph)
        indices += perm

        LOGGER.info(
            f"[{self.__class__.__name__}] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        return data_list, indices

    def _build_event_graph(self, event, network_dtype, momentum_dtype):
        ncat_ditop = 2 if self.use_ditop else 0

        top_had = event["TopHad"]
        subjets = event["Subjets"]

        jet_kin = torch.stack(
            [
                torch.as_tensor(top_had[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        jet_four = PtPhiEtaM_to_EPPP(jet_kin).unsqueeze(0)

        subjet_kin = torch.stack(
            [
                torch.as_tensor(subjets[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        subjet_tags = torch.as_tensor(subjets["tag"], dtype=torch.long)
        subjet_four = PtPhiEtaM_to_EPPP(subjet_kin)

        top_lep = event["TopLep"]
        lepton = event["Lepton"]

        toplep_kin = torch.stack(
            [
                torch.as_tensor(top_lep[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        toplep_four = PtPhiEtaM_to_EPPP(toplep_kin).unsqueeze(0)

        lep_kin = torch.stack(
            [torch.as_tensor(lepton[k], dtype=momentum_dtype) for k in ["pt", "phi", "eta"]]
            + [torch.as_tensor(0.0, dtype=momentum_dtype)],
            dim=-1,
        )
        lep_charge = torch.as_tensor(lepton["charge"], dtype=torch.long)
        lep_four = PtPhiEtaM_to_EPPP(lep_kin).unsqueeze(0)

        b_idx = torch.argmax((subjet_tags == 5).to(torch.int)).item()
        other_indices = [i for i in range(len(subjet_tags)) if i != b_idx]
        other_indices.sort(key=lambda idx: subjet_kin[idx, 0].item(), reverse=True)
        subjet_order = torch.tensor([b_idx] + other_indices, dtype=torch.long)

        subjet_four_sorted = subjet_four[subjet_order]

        # Label the jet as 1 if the leading non-b subjet matches the target tag
        label = (
            torch.isin(subjet_tags[subjet_order[1]], torch.tensor(self.tag))
            .to(network_dtype)
            .unsqueeze(0)
        )

        constituents = subjets["Constituents"]
        const_four_list, const_pid_list, const_parent_list = [], [], []
        for parent_idx, original_idx in enumerate(subjet_order.tolist()):
            subj_const = constituents[original_idx]
            const_kin = torch.stack(
                [
                    torch.as_tensor(subj_const[k], dtype=momentum_dtype)
                    for k in ["pt", "phi", "eta", "mass"]
                ],
                dim=-1,
            )
            const_pid = torch.as_tensor(subj_const["pid"], dtype=momentum_dtype)

            const_order = torch.argsort(const_kin[:, 0], descending=True)
            const_four_list.append(PtPhiEtaM_to_EPPP(const_kin[const_order]))
            const_pid_list.append(const_pid[const_order])
            const_parent_list.append(
                torch.full((const_order.numel(),), parent_idx, dtype=torch.long)
            )

        const_four_sorted = torch.cat(const_four_list, dim=0)
        const_pid_sorted = torch.cat(const_pid_list, dim=0)
        const_parents_sorted = torch.cat(const_parent_list, dim=0)

        sequence = torch.cat([jet_four, subjet_four_sorted, const_four_sorted], dim=0)

        n_constituents = const_four_sorted.shape[0]

        category = torch.zeros(4 + n_constituents + ncat_ditop, 5 + ncat_ditop, dtype=network_dtype)
        # Hadronic top: [1, 0, 0, 0, 0]
        category[0, 0] = 1

        # 3 Subjets: [0, 1, X, X, X] where one X=1 for subjet index
        category[1:4, 1] = 1
        category[1, 2] = 1
        category[2, 3] = 1
        category[3, 4] = 1

        # Constituents: [0, 0, X, X, X] where one X=1 for parent subjet (0, 1, or 2)
        for j, parent_idx in enumerate(const_parents_sorted):
            category[4 + j, 2 + parent_idx] = 1

        if self.use_ditop:
            # Leptonic top: [0, 0, 0, 0, 0, 1, 0]
            category[4 + len(const_parents_sorted), 5] = 1

            # Lepton: [0, 0, 0, 0, 0, 0, charge]
            category[4 + len(const_parents_sorted) + 1, 6] = lep_charge

        if self.include_pid:
            pid_features = torch.zeros(4 + n_constituents + ncat_ditop, 6, dtype=network_dtype)
            pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
            pid_features[4 : 4 + n_constituents] = pid_encoded.to(network_dtype)
            scalars = torch.cat([category, pid_features], dim=-1)
        else:
            scalars = category

        if self.use_ditop:
            sequence = torch.cat([sequence, toplep_four, lep_four], dim=0)
            return Data(x=sequence, scalars=scalars, label=label, lep_vec=None, lep_charge=None)

        lep_vec = torch.cat([toplep_four, lep_four], dim=0)
        return Data(
            x=sequence, scalars=scalars, label=label, lep_vec=lep_vec, lep_charge=lep_charge
        )


class PolarizationDataset(HEPTaggingDataset):
    """
    Dataset for directly identifying the top quark polarization in a binary classification task.
    Parameters:
        use_ditop: If set to true, also include the lepton and the leptonic top quark in the event/graph
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "polarization_tagging"

    def __init__(
        self,
        length=-1,
        pid=False,
        use_ditop=False,
        use_cache=True,
    ):
        super().__init__(length=length, use_cache=use_cache)
        self.include_pid = pid
        self.use_ditop = use_ditop

    def _cache_key_gen(self, target_length):
        cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            bool(self.use_ditop),
            target_length,
        )
        return cache_key

    def _get_parquet(self):
        parquet = [
            ak.concatenate([self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in files])
            for files in self.filename.values()
        ]
        num_events = len(parquet)
        return parquet, num_events

    def _fill_data(self, parquet, network_dtype, momentum_dtype):
        data_list = []
        indices = []
        label_counts = 0

        offset = 0
        for lbl, p in enumerate(parquet):
            p, perm = self._shuffle_data(p, evt_types=len(parquet))
            if offset > 0:
                perm = [pm + offset for pm in perm]
            for event in p:
                graph = self._build_event_graph(event, lbl, network_dtype, momentum_dtype)
                label_counts += graph.label
                data_list.append(graph)
            indices += perm
            offset += len(data_list)
        data_list, _ = self._shuffle_data(data_list)

        LOGGER.info(
            f"[{self.__class__.__name__}] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        return data_list, indices

    def _build_event_graph(self, event, label, network_dtype, momentum_dtype):
        ncat_ditop = 2 if self.use_ditop else 0

        top_had = event["TopHad"]
        subjets = event["Subjets"]

        jet_kin = torch.stack(
            [
                torch.as_tensor(top_had[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        jet_four = PtPhiEtaM_to_EPPP(jet_kin).unsqueeze(0)

        subjet_kin = torch.stack(
            [
                torch.as_tensor(subjets[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        subjet_tags = torch.as_tensor(subjets["tag"], dtype=torch.long)
        subjet_four = PtPhiEtaM_to_EPPP(subjet_kin)

        top_lep = event["TopLep"]
        lepton = event["Lepton"]

        toplep_kin = torch.stack(
            [
                torch.as_tensor(top_lep[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        toplep_four = PtPhiEtaM_to_EPPP(toplep_kin).unsqueeze(0)

        lep_kin = torch.stack(
            [torch.as_tensor(lepton[k], dtype=momentum_dtype) for k in ["pt", "phi", "eta"]]
            + [torch.as_tensor(0.0, dtype=momentum_dtype)],
            dim=-1,
        )
        lep_charge = torch.as_tensor(lepton["charge"], dtype=torch.long)
        lep_four = PtPhiEtaM_to_EPPP(lep_kin).unsqueeze(0)

        b_idx = torch.argmax((subjet_tags == 5).to(torch.int)).item()
        other_indices = [i for i in range(len(subjet_tags)) if i != b_idx]
        other_indices.sort(key=lambda idx: subjet_kin[idx, 0].item(), reverse=True)
        subjet_order = torch.tensor([b_idx] + other_indices, dtype=torch.long)

        subjet_four_sorted = subjet_four[subjet_order]

        constituents = subjets["Constituents"]
        const_four_list, const_pid_list, const_parent_list = [], [], []
        for parent_idx, original_idx in enumerate(subjet_order.tolist()):
            subj_const = constituents[original_idx]
            const_kin = torch.stack(
                [
                    torch.as_tensor(subj_const[k], dtype=momentum_dtype)
                    for k in ["pt", "phi", "eta", "mass"]
                ],
                dim=-1,
            )
            const_pid = torch.as_tensor(subj_const["pid"], dtype=momentum_dtype)

            const_order = torch.argsort(const_kin[:, 0], descending=True)
            const_four_list.append(PtPhiEtaM_to_EPPP(const_kin[const_order]))
            const_pid_list.append(const_pid[const_order])
            const_parent_list.append(
                torch.full((const_order.numel(),), parent_idx, dtype=torch.long)
            )

        const_four_sorted = torch.cat(const_four_list, dim=0)
        const_pid_sorted = torch.cat(const_pid_list, dim=0)
        const_parents_sorted = torch.cat(const_parent_list, dim=0)

        sequence = torch.cat([jet_four, subjet_four_sorted, const_four_sorted], dim=0)

        n_constituents = const_four_sorted.shape[0]

        category = torch.zeros(4 + n_constituents + ncat_ditop, 5 + ncat_ditop, dtype=network_dtype)
        # Hadronic top: [1, 0, 0, 0, 0]
        category[0, 0] = 1

        # 3 Subjets: [0, 1, X, X, X] where one X=1 for subjet index
        category[1:4, 1] = 1
        category[1, 2] = 1
        category[2, 3] = 1
        category[3, 4] = 1

        # Constituents: [0, 0, X, X, X] where one X=1 for parent subjet (0, 1, or 2)
        for j, parent_idx in enumerate(const_parents_sorted):
            category[4 + j, 2 + parent_idx] = 1

        if self.use_ditop:
            # Leptonic top: [0, 0, 0, 0, 0, 1, 0]
            category[4 + len(const_parents_sorted), 5] = 1

            # Lepton: [0, 0, 0, 0, 0, 0, charge]
            category[4 + len(const_parents_sorted) + 1, 6] = lep_charge

        if self.include_pid:
            pid_features = torch.zeros(4 + n_constituents + ncat_ditop, 6, dtype=network_dtype)
            pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
            pid_features[4 : 4 + n_constituents] = pid_encoded.to(network_dtype)
            scalars = torch.cat([category, pid_features], dim=-1)
        else:
            scalars = category

        if self.use_ditop:
            sequence = torch.cat([sequence, toplep_four, lep_four], dim=0)
            return Data(x=sequence, scalars=scalars, label=label, lep_vec=None, lep_charge=None)

        lep_vec = torch.cat([toplep_four, lep_four], dim=0)
        return Data(
            x=sequence, scalars=scalars, label=label, lep_vec=lep_vec, lep_charge=lep_charge
        )


class TTSmeftDataset(HEPTaggingDataset):
    """
    Dataset for identifying CP-even and CP-odd SMEFT contributions to ttbar production.
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "ttsmeft"

    def __init__(
        self,
        length=-1,
        pid=False,
        include_ps=False,
        use_cache=True,
    ):
        super().__init__(length=length, use_cache=use_cache)
        self.include_pid = pid
        self.include_ps = include_ps

    def _cache_key_gen(self, target_length):
        cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            target_length,
            # both change the scalar channels, so they have to invalidate the cache
            bool(self.include_pid),
            bool(self.include_ps),
        )
        return cache_key

    def _get_parquet(self):
        parquet = {
            key: ak.concatenate(
                [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in value]
            )
            for key, value in self.filename.items()
        }
        num_events = len(parquet[0]) + len(parquet[1])
        return parquet, num_events

    def _fill_data(self, parquet, network_dtype, momentum_dtype, balance=True):
        data_list = []
        indices = []
        label_counts = 0

        if balance:
            label_events = {lbl: list(p) for lbl, p in parquet.items()}
            min_count = min(len(events) for events in label_events.values())
            balanced_parquet = {}
            for lbl, events in label_events.items():
                if len(events) > min_count:
                    perm = np.random.permutation(len(events))[:min_count]
                    balanced_parquet[lbl] = [events[i] for i in perm]
                else:
                    balanced_parquet[lbl] = events
            parquet = balanced_parquet

        offset = 0
        for lbl, p in parquet.items():
            p, perm = self._shuffle_data(p, evt_types=len(parquet))
            if offset > 0:
                perm = [pm + offset for pm in perm]
            for event in p:
                graph = self._build_event_graph(event, lbl, network_dtype, momentum_dtype)
                label_counts += graph.label
                data_list.append(graph)
            indices += perm
            offset += len(data_list)
        data_list, _ = self._shuffle_data(data_list)

        LOGGER.info(
            f"[{self.__class__.__name__}] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        return data_list, indices

    def _build_event_graph(self, event, label, network_dtype, momentum_dtype):
        top_had = event["TopHad"]
        subjets = event["Subjets"]

        jet_kin = torch.stack(
            [
                torch.as_tensor(top_had[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        jet_four = PtPhiEtaM_to_EPPP(jet_kin).unsqueeze(0)

        subjet_kin = torch.stack(
            [
                torch.as_tensor(subjets[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        subjet_tags = torch.as_tensor(subjets["tag"], dtype=torch.long)
        subjet_four = PtPhiEtaM_to_EPPP(subjet_kin)

        top_lep = event["TopLep"]
        lepton = event["Lepton"]

        toplep_kin = torch.stack(
            [
                torch.as_tensor(top_lep[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        toplep_four = PtPhiEtaM_to_EPPP(toplep_kin).unsqueeze(0)

        lep_kin = torch.stack(
            [torch.as_tensor(lepton[k], dtype=momentum_dtype) for k in ["pt", "phi", "eta"]]
            + [torch.as_tensor(0.0, dtype=momentum_dtype)],
            dim=-1,
        )
        lep_charge = torch.as_tensor(lepton["charge"], dtype=torch.long)
        lep_four = PtPhiEtaM_to_EPPP(lep_kin).unsqueeze(0)

        b_idx = torch.argmax((subjet_tags == 5).to(torch.int)).item()
        d_idx = torch.argmax(
            torch.isin(subjet_tags, torch.tensor([-3, -1, 1, 3])).to(torch.int)
        ).item()
        other_indices = [i for i in range(len(subjet_tags)) if i not in (b_idx, d_idx)]
        other_indices.sort(key=lambda idx: subjet_kin[idx, 0].item(), reverse=True)
        subjet_order = torch.tensor([b_idx, d_idx] + other_indices, dtype=torch.long)

        subjet_four_sorted = subjet_four[subjet_order]

        constituents = subjets["Constituents"]
        const_four_list, const_pid_list, const_parent_list = [], [], []
        for parent_idx, original_idx in enumerate(subjet_order.tolist()):
            subj_const = constituents[original_idx]
            const_kin = torch.stack(
                [
                    torch.as_tensor(subj_const[k], dtype=momentum_dtype)
                    for k in ["pt", "phi", "eta", "mass"]
                ],
                dim=-1,
            )
            const_pid = torch.as_tensor(subj_const["pid"], dtype=momentum_dtype)

            const_order = torch.argsort(const_kin[:, 0], descending=True)
            const_four_list.append(PtPhiEtaM_to_EPPP(const_kin[const_order]))
            const_pid_list.append(const_pid[const_order])
            const_parent_list.append(
                torch.full((const_order.numel(),), parent_idx, dtype=torch.long)
            )

        const_four_sorted = torch.cat(const_four_list, dim=0)
        const_pid_sorted = torch.cat(const_pid_list, dim=0)
        const_parents_sorted = torch.cat(const_parent_list, dim=0)

        sequence = torch.cat(
            [
                jet_four,
                subjet_four_sorted,
                const_four_sorted,
                toplep_four,
                lep_four,
            ],
            dim=0,
        )

        n_constituents = const_four_sorted.shape[0]

        category = torch.zeros(6 + n_constituents, 7, dtype=network_dtype)
        # Hadronic top: [1, 0, 0, 0, 0, 0, 0]
        category[0, 0] = 1

        # 3 Subjets: [0, 1, X, X, X, 0, 0] where one X=1 for subjet index
        category[1:4, 1] = 1
        category[1, 2] = 1
        category[2, 3] = 1
        category[3, 4] = 1

        # Constituents: [0, 0, X, X, X, 0, 0] where one X=1 for parent subjet (0, 1, or 2)
        for j, parent_idx in enumerate(const_parents_sorted):
            category[4 + j, 2 + parent_idx] = 1

        # Leptonic top: [0, 0, 0, 0, 0, 1, 0]
        category[4 + len(const_parents_sorted), 5] = 1

        # Lepton: [0, 0, 0, 0, 0, 0, charge]
        category[4 + len(const_parents_sorted) + 1, 6] = lep_charge

        scalars = category
        if self.include_pid:
            pid_features = torch.zeros(6 + n_constituents, 6, dtype=network_dtype)
            pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
            pid_features[4 : 4 + n_constituents] = pid_encoded.to(network_dtype)
            scalars = torch.cat([scalars, pid_features], dim=-1)
        if self.include_ps:
            pseudo_eps = eps_tensor(
                jet_four, toplep_four, subjet_four_sorted[1], lep_four, dtype=momentum_dtype
            )
            pseudo_nr = spin_correlation_nr(
                jet_four,
                toplep_four,
                subjet_four_sorted[1],
                lep_four,
                lep_charge,
                dtype=momentum_dtype,
            )
            pseudoscalars = torch.tensor(
                [pseudo_eps, pseudo_nr], dtype=network_dtype
            )  # Array (n_pseudoscalars,)
            # the pseudoscalars are inserted in the 16-vector of the GA for L-GATr
            # max of 6 pseudoscalars for now (6 class tokens), to avoid new tokens
            n_pseudoscalars = len(pseudoscalars)
            assert n_pseudoscalars <= 6
            pseudoscalars = F.pad(pseudoscalars, (0, 6 + n_constituents - n_pseudoscalars))
            scalars = torch.cat([scalars, pseudoscalars.unsqueeze(1)], dim=-1)

        return Data(x=sequence, scalars=scalars, label=label)


class PseudoscalarDataset(HEPTaggingDataset):
    """
    Dataset for extracting the pseudoscalar component to build a CP-odd observable in ttH.
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "pseudoscalar"

    def __init__(
        self,
        length=-1,
        pid=False,
        use_cache=True,
    ):
        super().__init__(length=length, use_cache=use_cache)
        self.include_pid = pid

    def _cache_key_gen(self, target_length):
        cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            target_length,
        )
        return cache_key

    def _get_parquet(self):
        parquet = {
            key: ak.concatenate(
                [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in value]
            )
            for key, value in self.filename.items()
        }
        num_events = len(parquet[0]) + len(parquet[1])
        return parquet, num_events

    def _fill_data(self, parquet, network_dtype, momentum_dtype, balance=True):
        data_list = []
        indices = []
        label_counts = 0

        if balance:
            label_events = {lbl: list(p) for lbl, p in parquet.items()}
            min_count = min(len(events) for events in label_events.values())
            balanced_parquet = {}
            for lbl, events in label_events.items():
                if len(events) > min_count:
                    perm = np.random.permutation(len(events))[:min_count]
                    balanced_parquet[lbl] = [events[i] for i in perm]
                else:
                    balanced_parquet[lbl] = events
            parquet = balanced_parquet

        offset = 0
        for lbl, p in parquet.items():
            p, perm = self._shuffle_data(p, evt_types=len(parquet))
            if offset > 0:
                perm = [pm + offset for pm in perm]
            for event in p:
                graph = self._build_event_graph(event, lbl, network_dtype, momentum_dtype)
                label_counts += graph.label
                data_list.append(graph)
            indices += perm
            offset += len(data_list)
        data_list, _ = self._shuffle_data(data_list)

        LOGGER.info(
            f"[{self.__class__.__name__}] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        return data_list, indices

    def _build_event_graph(self, event, label, network_dtype, momentum_dtype):
        top_had = event["TopHad"]
        subjets = event["Subjets"]

        jet_kin = torch.stack(
            [
                torch.as_tensor(top_had[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        jet_four = PtPhiEtaM_to_EPPP(jet_kin).unsqueeze(0)

        subjet_kin = torch.stack(
            [
                torch.as_tensor(subjets[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        subjet_tags = torch.as_tensor(subjets["tag"], dtype=torch.long)
        subjet_four = PtPhiEtaM_to_EPPP(subjet_kin)

        top_lep = event["TopLep"]
        lepton = event["Lepton"]

        toplep_kin = torch.stack(
            [
                torch.as_tensor(top_lep[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        toplep_four = PtPhiEtaM_to_EPPP(toplep_kin).unsqueeze(0)

        lep_kin = torch.stack(
            [torch.as_tensor(lepton[k], dtype=momentum_dtype) for k in ["pt", "phi", "eta"]]
            + [torch.as_tensor(0.0, dtype=momentum_dtype)],
            dim=-1,
        )
        lep_charge = torch.as_tensor(lepton["charge"], dtype=torch.long)
        lep_four = PtPhiEtaM_to_EPPP(lep_kin).unsqueeze(0)

        b_idx = torch.argmax((subjet_tags == 5).to(torch.int)).item()
        other_indices = [i for i in range(len(subjet_tags)) if i != b_idx]
        other_indices.sort(key=lambda idx: subjet_kin[idx, 0].item(), reverse=True)
        subjet_order = torch.tensor([b_idx] + other_indices, dtype=torch.long)

        subjet_four_sorted = subjet_four[subjet_order]

        constituents = subjets["Constituents"]
        const_four_list, const_pid_list, const_parent_list = [], [], []
        for parent_idx, original_idx in enumerate(subjet_order.tolist()):
            subj_const = constituents[original_idx]
            const_kin = torch.stack(
                [
                    torch.as_tensor(subj_const[k], dtype=momentum_dtype)
                    for k in ["pt", "phi", "eta", "mass"]
                ],
                dim=-1,
            )
            const_pid = torch.as_tensor(subj_const["pid"], dtype=momentum_dtype)

            const_order = torch.argsort(const_kin[:, 0], descending=True)
            const_four_list.append(PtPhiEtaM_to_EPPP(const_kin[const_order]))
            const_pid_list.append(const_pid[const_order])
            const_parent_list.append(
                torch.full((const_order.numel(),), parent_idx, dtype=torch.long)
            )

        const_four_sorted = torch.cat(const_four_list, dim=0)
        const_pid_sorted = torch.cat(const_pid_list, dim=0)
        const_parents_sorted = torch.cat(const_parent_list, dim=0)

        higgs = event["higgs"]
        photon1 = event["photon_leading"]
        photon2 = event["photon_subleading"]

        higgs_kin = torch.stack(
            [torch.as_tensor(higgs[k], dtype=momentum_dtype) for k in ["pt", "phi", "eta", "mass"]],
            dim=-1,
        )
        higgs_four = PtPhiEtaM_to_EPPP(higgs_kin).unsqueeze(0)

        photon1_kin = torch.stack(
            [
                torch.as_tensor(photon1[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        photon1_four = PtPhiEtaM_to_EPPP(photon1_kin).unsqueeze(0)

        photon2_kin = torch.stack(
            [
                torch.as_tensor(photon2[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        photon2_four = PtPhiEtaM_to_EPPP(photon2_kin).unsqueeze(0)

        photon_pids = torch.as_tensor([22, 22], dtype=momentum_dtype)

        sequence = torch.cat(
            [
                jet_four,
                subjet_four_sorted,
                const_four_sorted,
                toplep_four,
                lep_four,
                higgs_four,
                photon1_four,
                photon2_four,
            ],
            dim=0,
        )

        n_constituents = const_four_sorted.shape[0]

        category = torch.zeros(9 + n_constituents, 10, dtype=network_dtype)
        # Hadronic top: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        category[0, 0] = 1

        # 3 Subjets: [0, 1, X, X, X, 0, 0, 0, 0, 0] where one X=1 for subjet index
        category[1:4, 1] = 1
        category[1, 2] = 1
        category[2, 3] = 1
        category[3, 4] = 1

        # Constituents: [0, 0, X, X, X, 0, 0, 0, 0, 0] where one X=1 for parent subjet (0, 1, or 2)
        for j, parent_idx in enumerate(const_parents_sorted):
            category[4 + j, 2 + parent_idx] = 1

        # Leptonic top: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        category[4 + len(const_parents_sorted), 5] = 1

        # Lepton: [0, 0, 0, 0, 0, 0, charge, 0, 0, 0]
        category[4 + len(const_parents_sorted) + 1, 6] = lep_charge

        # Higgs: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        category[4 + len(const_parents_sorted) + 2, 7] = 1

        # Photons: [0, 0, 0, 0, 0, 0, 0, 1, X, X] where one X=1 for photon index
        category[4 + len(const_parents_sorted) + 3, 7] = 1
        category[4 + len(const_parents_sorted) + 3, 8] = 1
        category[4 + len(const_parents_sorted) + 4, 7] = 1
        category[4 + len(const_parents_sorted) + 4, 9] = 1

        if self.include_pid:
            pid_features = torch.zeros(9 + n_constituents, 6, dtype=network_dtype)
            pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
            pid_encoded_photons = self.pid_encoding(photon_pids.unsqueeze(-1)).squeeze(-2)
            pid_features[4 : 4 + n_constituents] = pid_encoded.to(network_dtype)
            pid_features[-2:] = pid_encoded_photons.to(network_dtype)
            scalars = torch.cat([category, pid_features], dim=-1)
        else:
            scalars = category

        return Data(x=sequence, scalars=scalars, label=label)


class VBFDataset(HEPTaggingDataset):
    """
    Dataset for VBF parton-level events.
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "vbf"

    def __init__(
        self,
        length=-1,
        use_cache=True,
    ):
        super().__init__(length=length, use_cache=use_cache)

    def _cache_key_gen(self, target_length):
        cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            target_length,
        )
        return cache_key

    def _get_parquet(self):
        files = {key: torch.from_numpy(np.load(value)) for key, value in self.filename.items()}
        num_events = sum(len(v) for v in files.values())
        return files, num_events

    def _fill_data(self, parquet, network_dtype, momentum_dtype):
        data_list = []
        indices = []
        label_counts = 0

        offset = 0
        for lbl, p in parquet.items():
            p, perm = self._shuffle_data(p, evt_types=len(parquet))
            if offset > 0:
                perm = [pm + offset for pm in perm]
            for event in p:
                graph = self._build_event_graph(event, lbl, network_dtype, momentum_dtype)
                label_counts += graph.label
                data_list.append(graph)
            indices += perm
            offset += len(data_list)
        data_list, _ = self._shuffle_data(data_list)

        LOGGER.info(
            f"[{self.__class__.__name__}] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        return data_list, indices

    def _build_event_graph(self, event, label, network_dtype, momentum_dtype):
        fourmomenta = event.to(dtype=momentum_dtype)
        category = torch.eye(3, dtype=network_dtype)
        return Data(x=fourmomenta, scalars=category, label=label)
