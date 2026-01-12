import os
import hashlib
import json

import awkward as ak
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]) # Needed to cache sequence data

from experiments.hep import PtPhiEtaM2_to_EPPP
from experiments.logger import LOGGER

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
        torch.save(
            {
                "key": cache_key,
                "data": data,
                "indices": indices
            },
            path,
        )

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


class TestDataset(TaggingDataset):
    """
    Minimal dataset mirroring the notebook
    Each event consists of three four-momenta and a binary label.
    """

    _npy_cache = {}
    _split_cache = {}
    n_token_types = 3

    def __init__(
        self,
        seed=42,
        split_fracs=(0.8, 0.1, 0.1),
        shuffle=True,
        length=-1,
    ):
        super().__init__(length=length)
        self.seed = seed
        self.split_fracs = split_fracs
        self.shuffle = shuffle

    def load_data(
        self,
        filename,
        mode,
        network_float64=False,
        momentum_float64=True,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split '{mode}'")

        network_dtype = torch.float64 if network_float64 else torch.float32
        momentum_dtype = torch.float64 if momentum_float64 else torch.float32

        npy = self._load_npy(filename)

        target_length = self._get_target_length(len(npy))

        split_indices = self._get_split_indices(
            filename,
            len(npy),
            self.split_fracs,
            self.seed,
            self.shuffle,
            target_length,
        )[mode]

        subset = npy[split_indices]
        data_list = []
        for row in subset:
            graph = self._build_event_graph(
                row,
                network_dtype=network_dtype,
                momentum_dtype=momentum_dtype,
            )
            if graph is not None:
                data_list.append(graph)

        if not data_list:
            raise RuntimeError(f"No usable events found in split '{mode}' for {filename}")

        self.data_list = data_list

    @classmethod
    def _load_npy(cls, filename):
        if isinstance(filename, (list, tuple)):
            return np.concatenate([cls._load_npy(f) for f in filename])
        if not isinstance(filename, str):
            raise ValueError("filename must be a string or a list/tuple of strings")

        if filename not in cls._npy_cache:
            cls._npy_cache[filename] = np.load(filename)
        return cls._npy_cache[filename]

    @classmethod
    def _get_split_indices(cls, filename, num_events, fracs, seed, shuffle, length):
        effective_length = num_events if length is None else min(length, num_events)
        key = (
            cls._normalize_filename_key(filename),
            tuple(fracs),
            seed,
            bool(shuffle),
            effective_length,
        )
        if key in cls._split_cache:
            return cls._split_cache[key]

        if effective_length == 0:
            raise RuntimeError(f"No events found in '{filename}'")

        frac_tensor = torch.as_tensor(fracs, dtype=torch.double)
        frac_tensor = frac_tensor / frac_tensor.sum()

        gen = torch.Generator()
        gen.manual_seed(seed)
        if shuffle:
            permutation = torch.randperm(num_events, generator=gen)
        else:
            permutation = torch.arange(num_events, dtype=torch.long)

        if effective_length < num_events:
            permutation = permutation[:effective_length]

        n_selected = permutation.numel()
        n_train = int(n_selected * frac_tensor[0].item())
        n_val = int(n_selected * frac_tensor[1].item())
        n_test = n_selected - n_train - n_val

        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                "Requested split fractions result in empty dataset parts. "
                f"num_events={effective_length}, fractions={fracs}"
            )

        splits = {
            "train": permutation[:n_train].tolist(),
            "val": permutation[n_train : n_train + n_val].tolist(),
            "test": permutation[n_train + n_val :].tolist(),
        }
        cls._split_cache[key] = splits
        return splits

    def _build_event_graph(
        self,
        row,
        network_dtype,
        momentum_dtype,
    ):
        features = torch.as_tensor(row[:-1], dtype=momentum_dtype)
        if features.numel() != self.n_token_types * 4:
            return None

        fourmomenta = features.view(self.n_token_types, 4)
        scalars = torch.eye(self.n_token_types, dtype=network_dtype)
        label = torch.tensor([row[-1]], dtype=network_dtype)

        data = Data(
            x=fourmomenta,
            scalars=scalars,
            label=label,
        )
        return data


class QuarkTaggingDataset(TaggingDataset):
    """
    Optimized dataset for subjet tagging on the polarization parquet file.
    Each event/graph contains the fat jet, its 3 subjets, and all subjet constituents.
    """

    CACHE_VERSION = 1
    CACHE_PREFIX = "quark_tagging"

    _parquet_cache = {}
    _split_cache = {}

    def __init__(
        self,
        tag=1,
        seed=12345,
        split_fracs=(0.8, 0.1, 0.1),
        shuffle=True,
        length=-1,
        pid=False,
        balance_factor=0.0,
        use_cache=True,
    ):
        super().__init__(length=length)
        self.tag = tag
        self.seed = seed
        self.split_fracs = split_fracs
        self.shuffle = shuffle
        self.balance_factor = balance_factor
        self.include_pid = pid
        self.use_cache = use_cache
        self.filename = None
        self.cache_key = {}

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
        if isinstance(self.filename, (list, tuple)):
            parquet = ak.concatenate(
                [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in self.filename]
            )
        elif isinstance(self.filename, (dict)):
            parquet = [
                ak.concatenate(
                [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in files]
                )
                for files in self.filename.values()
            ]
        else:
            parquet = self._parquet_cache.setdefault(self.filename, ak.from_parquet(self.filename))

        num_events = sum(len(p) for p in parquet) if isinstance(parquet, (list)) else len(parquet)
        target_length = self._get_target_length(num_events)

        self.cache_key = (
            self.CACHE_VERSION,
            self._normalize_filename_key(self.filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            target_length,
        )

        if self.use_cache:
            cache = self._load_from_cache(self.cache_key, data_dir)
        else:
            cache = None

        if cache is None:

            data_list = []
            indices = []
            label_counts = 0

            if isinstance(parquet, (list)):
                offset = 0
                for l, p in enumerate(parquet):
                    p, perm = self._shuffle_data(p, evt_types=len(parquet))
                    if offset > 0:
                        perm = [pm+offset for pm in perm]
                    for event in p:
                        graph = self._build_event_graph(event, network_dtype, momentum_dtype, label_ext=l)
                        label_counts += graph.label
                        data_list.append(graph)
                    indices += perm
                    offset += len(data_list)
                data_list, _ = self._shuffle_data(data_list)
            else:
                parquet, perm = self._shuffle_data(parquet)
                for event in parquet:
                    graph = self._build_event_graph(event, network_dtype, momentum_dtype)
                    if (
                        graph.label.item() == 0
                        and torch.rand(1).item() > self.balance_factor
                    ):
                        # Artificially balance dataset
                        continue
                    label_counts += graph.label.item()
                    data_list.append(graph)
                indices += perm

            LOGGER.info(
                f"[QuarkTaggingDataset] label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
            )

            if not data_list:
                raise RuntimeError(f"No usable events found for {self.filename}")

            self._save_to_cache(self.cache_key, data_list, indices, data_dir)

            LOGGER.info(
                f"[QuarkTaggingDataset] saved to cache"
            )

        else:

            data_list, indices = cache

            LOGGER.info(
                f"[QuarkTaggingDataset] loaded from cache"
            )

        self.data_list = data_list
        self._split_data(indices)

    def _shuffle_data(self, data, evt_types=-1):
        
        # target_length = num_events if self.length is None else min(self.length, num_events)
        target_length = self._get_target_length(len(data))
        if evt_types > 0 and target_length < len(data):
            target_length = int(target_length / evt_types)

        if target_length == 0:
            raise RuntimeError(f"No events found in loaded data")

        # Generate permutation
        gen = torch.Generator().manual_seed(self.seed)
        perm = (
            torch.randperm(len(data), generator=gen)
            if self.shuffle
            else torch.arange(len(data))
        )
        perm = perm[:target_length]

        if isinstance(data, (list)):
            return [data[x] for x in perm.tolist()], perm

        return data[perm.tolist()], perm
            
    def _split_data(self, indices):

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
                "train": indices[:n_train],
                "val": indices[n_train : n_train + n_val],
                "test": indices[n_train + n_val :],
            }

    def get_subset(self, mode):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split '{mode}'")

        if self.cache_key not in self._split_cache:
            raise KeyError(f"{self.cache_key} not found in cache")
        
        split_indices = self._split_cache[self.cache_key][mode]

        LOGGER.info(
            f"[QuarkTaggingDataset] mode={mode}, requested_length={self.length}, "
            f"selected={len(split_indices)}/{len(self.data_list)}"
        )

        return split_indices

    def load_data_old(self, filename, mode, network_float64=False, momentum_float64=True):

        # raise DeprecationWarning(f"Function replaced due to incompatibility with L vs R polarization")

        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split '{mode}'")

        network_dtype = torch.float64 if network_float64 else torch.float32
        momentum_dtype = torch.float64 if momentum_float64 else torch.float32

        # Load parquet with caching
        if isinstance(filename, (list, tuple)):
            parquet = ak.concatenate(
                [self._parquet_cache.setdefault(f, ak.from_parquet(f)) for f in filename]
            )
        else:
            parquet = self._parquet_cache.setdefault(filename, ak.from_parquet(filename))

        # Get split indices with caching
        num_events = len(parquet)
        # target_length = num_events if self.length is None else min(self.length, num_events)
        target_length = self._get_target_length(len(parquet))

        cache_key = (
            self._normalize_filename_key(filename),
            tuple(self.split_fracs),
            self.seed,
            bool(self.shuffle),
            target_length,
        )

        if cache_key not in self._split_cache:
            if target_length == 0:
                raise RuntimeError(f"No events found in '{filename}'")

            # Calculate split sizes
            fracs = torch.as_tensor(self.split_fracs, dtype=torch.double)
            fracs = fracs / fracs.sum()

            # Generate permutation
            gen = torch.Generator().manual_seed(self.seed)
            perm = (
                torch.randperm(num_events, generator=gen)
                if self.shuffle
                else torch.arange(num_events)
            )
            perm = perm[:target_length]

            n_train = int(target_length * fracs[0].item())
            n_val = int(target_length * fracs[1].item())
            n_test = target_length - n_train - n_val

            if min(n_train, n_val, n_test) <= 0:
                raise ValueError(
                    f"Split fractions result in empty dataset parts. "
                    f"num_events={target_length}, fractions={self.split_fracs}"
                )

            self._split_cache[cache_key] = {
                "train": perm[:n_train].tolist(),
                "val": perm[n_train : n_train + n_val].tolist(),
                "test": perm[n_train + n_val :].tolist(),
            }

        split_indices = self._split_cache[cache_key][mode]
        subset = parquet[split_indices]

        LOGGER.info(
            f"[QuarkTaggingDataset] mode={mode}, requested_length={self.length}, "
            f"selected={len(subset)}/{num_events}"
        )

        data_list = []
        label_counts = 0
        for event in subset:
            graph = self._build_event_graph(event, network_dtype, momentum_dtype)
            if (
                graph.label.item() == 0
                and torch.rand(1).item() > self.balance_factor
                and mode != "test"
            ):
                # Artificially balance dataset
                continue
            label_counts += graph.label.item()
            data_list.append(graph)

        LOGGER.info(
            f"[QuarkTaggingDataset] mode={mode}, label=1 ratio: {label_counts}/{len(data_list)} = {label_counts / len(data_list)}"
        )

        if not data_list:
            raise RuntimeError(f"No usable events found in split '{mode}' for {filename}")

        self.data_list = data_list

    def _build_event_graph(self, event, network_dtype, momentum_dtype, label_ext=-1):
        fat_jet = event["FatJet"]
        subjets = event["Subjets"]

        jet_kin = torch.stack(
            [
                torch.as_tensor(fat_jet[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        jet_four = PtPhiEtaM2_to_EPPP(jet_kin).unsqueeze(0)

        subjet_kin = torch.stack(
            [
                torch.as_tensor(subjets[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ],
            dim=-1,
        )
        subjet_tags = torch.as_tensor(subjets["tag"], dtype=torch.long)
        subjet_four = PtPhiEtaM2_to_EPPP(subjet_kin)

        b_idx = torch.argmax((subjet_tags == 5).to(torch.int)).item()
        other_indices = [i for i in range(len(subjet_tags)) if i != b_idx]
        other_indices.sort(key=lambda idx: subjet_kin[idx, 0].item(), reverse=True)
        subjet_order = torch.tensor([b_idx] + other_indices, dtype=torch.long)

        subjet_four_sorted = subjet_four[subjet_order]

        if label_ext >= 0:
            # Label by the polarization of the dataset
            label = label_ext
        else:
            if self.tag > 0:
                # Label the jet as 1 if the leading non-b subjet matches the target tag
                label = (subjet_tags[subjet_order[1]] == self.tag).to(network_dtype).unsqueeze(0)
            else:
                raise ValueError(f"Invalid subjet label of {self.tag}, must be positive")

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
            const_four_list.append(PtPhiEtaM2_to_EPPP(const_kin[const_order]))
            const_pid_list.append(const_pid[const_order])
            const_parent_list.append(
                torch.full((const_order.numel(),), parent_idx, dtype=torch.long)
            )

        const_four_sorted = torch.cat(const_four_list, dim=0)
        const_pid_sorted = torch.cat(const_pid_list, dim=0)
        const_parents_sorted = torch.cat(const_parent_list, dim=0)

        sequence = torch.cat([jet_four, subjet_four_sorted, const_four_sorted], dim=0)

        n_constituents = const_four_sorted.shape[0]

        category = torch.zeros(4 + n_constituents, 5, dtype=network_dtype)
        # Jet: [1, 0, 0, 0, 0]
        category[0, 0] = 1

        # 3 Subjets: [0, 1, X, X, X] where one X=1 for subjet index
        category[1:4, 1] = 1
        category[1, 2] = 1
        category[2, 3] = 1
        category[3, 4] = 1

        # Constituents: [0, 0, X, X, X] where one X=1 for parent subjet (0, 1, or 2)
        for j, parent_idx in enumerate(const_parents_sorted):
            category[4 + j, 2 + parent_idx] = 1

        if self.include_pid:
            pid_features = torch.zeros(4 + n_constituents, 6, dtype=network_dtype)
            pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
            pid_features[4:] = pid_encoded.to(network_dtype)
            scalars = torch.cat([category, pid_features], dim=-1)
        else:
            scalars = category

        return Data(x=sequence, scalars=scalars, label=label)
