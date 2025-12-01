import awkward as ak
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

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

    def _get_target_length(self, num_events):
        if self.length is None or self.length < 0:
            return num_events
        if self.length == 0:
            raise ValueError("Requested dataset length must be positive")
        return min(int(self.length), num_events)

    @staticmethod
    def _normalize_filename_key(filename):
        if isinstance(filename, (list, tuple)):
            return tuple(filename)
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


class BTaggingDataset(TaggingDataset):
    """
    Optimized dataset for subjet b-tagging on the polarization parquet file.
    Each event/graph contains the fat jet, its 3 subjets, and all subjet constituents.
    """

    _parquet_cache = {}
    _split_cache = {}

    def __init__(self, seed=12345, split_fracs=(0.8, 0.1, 0.1), shuffle=True, length=-1):
        super().__init__(length=length)
        self.seed = seed
        self.split_fracs = split_fracs
        self.shuffle = shuffle

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

    def load_data(self, filename, mode, network_float64=False, momentum_float64=True):
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
        target_length = num_events if self.length is None else min(self.length, num_events)

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
            f"[BTaggingDataset] mode={mode}, requested_length={self.length}, "
            f"selected={len(subset)}/{num_events} (full)"
        )

        # Build graphs
        data_list = []
        for event in subset:
            fat_jet = event["FatJet"]
            subjets = event["Subjets"]

            # Jet kinematics
            jet_kin = torch.tensor(
                [[fat_jet["pt"], fat_jet["phi"], fat_jet["eta"], fat_jet["mass"]]],
                dtype=momentum_dtype,
            )
            jet_four = PtPhiEtaM2_to_EPPP(
                jet_kin[:, 0], jet_kin[:, 1], jet_kin[:, 2], jet_kin[:, 3]
            )

            # Subjet kinematics (3 subjets)
            subjet_kin = torch.stack(
                [
                    torch.as_tensor(subjets[k], dtype=momentum_dtype)
                    for k in ["pt", "phi", "eta", "mass"]
                ],
                dim=-1,
            )
            subjet_btag = torch.as_tensor(subjets["btag"], dtype=torch.bool)
            subjet_four = PtPhiEtaM2_to_EPPP(
                subjet_kin[:, 0], subjet_kin[:, 1], subjet_kin[:, 2], subjet_kin[:, 3]
            )

            # Sort subjets by pT descending
            subjet_order = torch.argsort(subjet_kin[:, 0], descending=True)
            subjet_four_sorted = subjet_four[subjet_order]

            # Constituent extraction
            constituents = ak.flatten(subjets["Constituents"], axis=1)
            const_pt, const_phi, const_eta, const_mass = [
                torch.tensor(constituents[k], dtype=momentum_dtype)
                for k in ["pt", "phi", "eta", "mass"]
            ]
            const_four = PtPhiEtaM2_to_EPPP(const_pt, const_phi, const_eta, const_mass)
            const_pid = torch.tensor(constituents["pid"], dtype=momentum_dtype)

            # Parent subjet indices
            const_counts = torch.tensor(ak.num(subjets["Constituents"], axis=1), dtype=torch.long)
            const_parents = torch.repeat_interleave(torch.arange(3, dtype=torch.long), const_counts)

            # Map parent indices to sorted order
            inverse_order = torch.argsort(subjet_order)
            const_parents_sorted = inverse_order[const_parents]

            # Sort constituents by pT descending
            const_order = torch.argsort(const_pt, descending=True)
            const_four_sorted = const_four[const_order]
            const_pid_sorted = const_pid[const_order]
            const_parents_sorted = const_parents_sorted[const_order]

            # Build sequence: 1 jet + 3 subjets + N constituents
            sequence = torch.cat([jet_four, subjet_four_sorted, const_four_sorted], dim=0)

            # Build scalars: [category (5), pid_encoding (6)] = 11 features

            n_constituents = const_four_sorted.shape[0]
            category = torch.zeros(4 + n_constituents, 5, dtype=network_dtype)

            # Jet: [1, 0, 0, 0, 0]
            category[0, 0] = 1

            # 3 Subjets: [0, 1, X, X, X] where one X=1 for subjet index
            category[1:4, 1] = 1  # All subjets have 1 in position 1
            category[1, 2] = 1  # Subjet 1
            category[2, 3] = 1  # Subjet 2
            category[3, 4] = 1  # Subjet 3

            # Constituents: [0, 0, X, X, X] where one X=1 for parent subjet (0, 1, or 2)
            for j, parent_idx in enumerate(const_parents_sorted):
                category[4 + j, 2 + parent_idx] = 1

            # PID encoding: only for constituents, zero for jet and subjets
            pid_features = torch.zeros(4 + n_constituents, 6, dtype=network_dtype)
            if n_constituents > 0:
                pid_encoded = self.pid_encoding(const_pid_sorted.unsqueeze(-1)).squeeze(-2)
                pid_features[4:] = pid_encoded.to(network_dtype)

            scalars = torch.cat([category, pid_features], dim=-1)

            # Label: hardest subjet b-tag
            label = subjet_btag[subjet_order[0]].unsqueeze(0)

            # Create graph
            data = Data(x=sequence, scalars=scalars, label=label)

            data_list.append(data)

        if not data_list:
            raise RuntimeError(f"No usable events found in split '{mode}' for {filename}")

        self.data_list = data_list
