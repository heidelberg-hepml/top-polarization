import glob
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from experiments.base_experiment import BaseExperiment
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.asymmetry import asymmetry, o1
from experiments.tagging.dataset import (
    DownTaggingDataset,
    PolarizationDataset,
    PseudoscalarDataset,
    TaggingDataset,
    TTSmeftDataset,
    TTSmeftLayout,
    VBFDataset,
    relabel_ttsmeft_graph,
    ttsmeft_graph_to_downtag_input,
)
from experiments.tagging.embedding import (
    embed_tagging_data,
    get_num_parity_odd_tagging_features,
    get_num_tagging_features,
)
from experiments.tagging.likelihood import get_polarization, probe_report
from experiments.tagging.plots import plot_mixer
from experiments.tagging.top_events import rest_frame_energy

PSEUDOSCALAR_MODELS = ("LGATr", "LGATrSlimPseudo")


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments, focusing on binary classification
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default values
        self.data = None
        self.num_outputs = 1  # binary classification
        self.extra_scalars = 0  # number of extra scalar input features per particle
        self.ps_task = False  # whether the task requires ps channels
        self.n_ps_data = 0  # parity-odd columns the dataset appends to `scalars`
        self.n_ps = 0  # pseudoscalar channels fed to the network, set in init_physics
        self.split_parity = False  # route the parity-odd inputs to the pseudoscalar stream

    def init_physics(self):
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        self.momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else torch.float32

        self.cfg.model.out_channels = self.num_outputs

        n_tagging = get_num_tagging_features(self.cfg.data.tagging_features)
        n_tagging_odd = get_num_parity_odd_tagging_features(self.cfg.data.tagging_features)
        if not self.ps_task:
            assert self.n_ps_data == 0, "No pseudoscalars can be given unless ps_task is set"
            self.n_ps = 0
        elif self.modelname in PSEUDOSCALAR_MODELS:
            # the parity-odd tagging features join the dataset pseudoscalars
            self.split_parity = True
            self.n_ps = self.n_ps_data + n_tagging_odd
            n_tagging -= n_tagging_odd
        else:
            # no pseudoscalar stream, so they travel as scalars and parity is broken
            LOGGER.info(f"{self.modelname} takes no pseudoscalar inputs, passing them as scalars.")
            self.extra_scalars += self.n_ps_data
            self.n_ps, self.n_ps_data = 0, 0

        if self.modelname in ["LGATr", "LGATrSlim", "LGATrSlimPseudo"]:
            # Lorentz-equivariance by internal representations
            in_s_channels = self.extra_scalars + n_tagging
            self.cfg.model.net.in_s_channels = 0 if self.cfg.model.mean_aggregation else 1
            self.cfg.model.net.in_s_channels += in_s_channels
            if self.modelname == "LGATr":
                # 1 mv for the vector, n_ps mv for the pseudoscalars
                self.cfg.model.net.in_mv_channels = max(1, self.n_ps)
            elif self.modelname == "LGATrSlimPseudo":
                self.cfg.model.net.in_p_channels = self.n_ps

            # doesn't affect results and never needed
            self.cfg.data.boost_jet = False
        elif self.modelname == "Transformer":
            # Non-equivariant or canonicalization
            self.cfg.model.in_channels = 7 + self.extra_scalars
            if self.cfg.model.add_fourmomenta_backbone:
                self.cfg.model.in_channels += 4

            if self.modelname == "Transformer":
                self.cfg.model.in_channels += 0 if self.cfg.model.mean_aggregation else 1

            # decide which entries to use for the framesnet
            if "equivectors" in self.cfg.model.framesnet:
                num_tagging_features = get_num_tagging_features(
                    tagging_features=self.cfg.data.tagging_features
                )
                self.cfg.model.framesnet.equivectors.num_scalars = self.extra_scalars
                self.cfg.model.framesnet.equivectors.num_scalars += num_tagging_features
            else:
                # not allowed, because the network is not Lorentz-equivariant
                self.cfg.data.boost_jet = False
        else:
            raise NotImplementedError(f"Model {self.modelname} not implemented")

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path, filenames, dataset_kwargs=None):
        LOGGER.info(f"Creating {Dataset.__name__} from {filenames}")
        t0 = time.time()
        dataset_kwargs = dataset_kwargs or {}
        self.data = Dataset(**dataset_kwargs)
        kwargs = dict(
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )
        self.data.load_data(data_path, filenames, **kwargs)
        self.data_train = Subset(self.data, self.data.get_subset("train"))
        self.data_val = Subset(self.data, self.data.get_subset("val"))
        self.data_test = Subset(self.data, self.data.get_subset("test"))
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

        self.init_standardization()

    def init_standardization(self):
        if hasattr(self.model, "init_standardization"):
            batch = next(iter(self.train_loader))
            fourmomenta, scalars, ptr, *_ = self._extract_batch(batch)
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                ptr,
                self.cfg.data,
                n_ps=self.n_ps_data,
                split_parity=self.split_parity,
                ref=self._batch_ref(batch),
            )
            self.model.init_standardization(embedding["fourmomenta"], embedding["ptr"])

    def _init_optimizer(self, param_groups=None):
        if self.cfg.model.net._target_.rsplit(".", 1)[-1] in [
            "ParticleTransformer",
            "MIParticleTransformer",
        ]:
            # special treatment for ParT, see
            # https://github.com/hqucms/weaver-core/blob/dev/custom_train_eval/weaver/train.py#L464
            decay, no_decay = {}, {}
            for name, param in self.model.net.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or (hasattr(self.model.net, "no_weight_decay") and name in {"cls_token"})
                ):
                    no_decay[name] = param
                else:
                    decay[name] = param
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
            param_groups = [
                {
                    "params": no_decay_1x,
                    "weight_decay": 0.0,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": decay_1x,
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": self.model.framesnet.parameters(),
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                    "lr": self.cfg.training.lr * self.cfg.training.lr_factor_framesnet,
                },
            ]

        super()._init_optimizer(param_groups=param_groups)

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results[set_label] = self._evaluate_single(
                        loader_dict[set_label],
                        f"{set_label}_ema",
                        mode="eval",
                    )

                self._evaluate_single(loader_dict[set_label], set_label, mode="eval")

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

        if self.cfg.evaluation.save_labels:
            self._save_metrics()
        if self.cfg.save:
            self._save_results()

    def _save_results(self):
        """Write the scalar metrics and event counts of the evaluated sets to results_<idx>.json."""
        out = {"n_events": {s: len(getattr(self, f"data_{s}")) for s in ["train", "val", "test"]}}
        for set_label, res in self.results.items():
            out[set_label] = {
                k: float(v) for k, v in res.items() if isinstance(v, (int, float, np.number))
            } | res.get("extra", {})
        path = os.path.join(
            self.cfg.run_dir, f"results_{self.cfg.run_idx}{self._eval_output_suffix()}.json"
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        LOGGER.info(f"Saved results in {path}")

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        trackers = defaultdict(list)
        self.model.eval()
        for batch in loader:
            y_pred, label, tracker, _ = self._get_ypred_and_label(batch, val=(mode == "val"))
            labels_true.append(label)
            labels_predict.append(y_pred)
            for key, value in tracker.items():
                trackers[key].append(value.detach().float().flatten().cpu())
        labels_true, labels_predict = (
            torch.cat(labels_true).cpu().float(),
            torch.cat(labels_predict).cpu().float(),
        )

        # tracked model outputs: mean and std over the full dataset
        for key, values in trackers.items():
            values = torch.cat(values)
            metrics[key] = values.mean().item()
            metrics[f"{key}_std"] = values.std().item() if values.numel() > 1 else 0.0

        # bce loss
        metrics["loss"] = self.loss(labels_predict, labels_true).item()

        if not torch.isfinite(labels_predict).all():
            mask = torch.isfinite(labels_predict)
            LOGGER.warning(
                f"Non-finite values ({(~mask).sum().item()}) in labels_predict detected during evaluation"
            )
            if mode == "eval":
                # only loss is required
                return metrics
            else:
                # set non-finite predictions to zero and continue
                labels_predict[~mask] = 0.0
        labels_predict = torch.nn.functional.sigmoid(labels_predict)

        if self.model.output == "split":
            labels_predict_scalar = labels_predict[:, 0]
            labels_predict_pseudo = labels_predict[:, 1]
            labels_predict = (labels_predict_scalar + labels_predict_pseudo) / 2

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = labels_true, labels_predict
            if self.model.output == "split":
                metrics["labels_predict_scalar"] = labels_predict_scalar
                metrics["labels_predict_pseudo"] = labels_predict_pseudo

        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()
        if self.model.output == "split":
            labels_predict_scalar = labels_predict_scalar.numpy()
            labels_predict_pseudo = labels_predict_pseudo.numpy()

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if mode == "eval" and self.model.output == "split":
            for suffix, lp in [
                ("scalar", labels_predict_scalar),
                ("pseudo", labels_predict_pseudo),
            ]:
                fpr_s, tpr_s, _ = roc_curve(labels_true, lp)
                metrics[f"fpr_{suffix}"] = fpr_s
                metrics[f"tpr_{suffix}"] = tpr_s
                metrics[f"auc_{suffix}"] = roc_auc_score(labels_true, lp)
                LOGGER.info(f"AUC ({suffix}) on {title} dataset: {metrics[f'auc_{suffix}']:.4f}")

        _skip_keys = {
            "labels_true",
            "labels_predict",
            "fpr",
            "tpr",
            "labels_predict_scalar",
            "labels_predict_pseudo",
            "fpr_scalar",
            "tpr_scalar",
            "fpr_pseudo",
            "tpr_pseudo",
        }
        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in _skip_keys:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if mode == "eval":
            framesString = type(self.model.framesnet).__name__
            num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            LOGGER.info(
                f"table {title}: {framesString} ({self.cfg.training.iterations} iterations)"
                f" & {num_parameters} & {metrics['accuracy']:.4f} & {metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f} & {metrics['rej05']:.0f} & {metrics['rej08']:.0f} \\\\"
            )
        return metrics

    def _eval_output_suffix(self):
        if not self.cfg.train and self.cfg.data.dataset_eval:
            return f"_{self.cfg.data.dataset_eval}"
        return ""

    def _save_metrics(self):
        metric_path = os.path.join(
            self.cfg.run_dir, f"preds_{self.cfg.run_idx}{self._eval_output_suffix()}"
        )
        os.makedirs(metric_path, exist_ok=True)
        LOGGER.info(f"Saving labels in {metric_path}")

        file = f"{metric_path}/labels.txt"
        labels = np.stack(
            (self.results["test"]["labels_true"], self.results["test"]["labels_predict"]), axis=-1
        )
        np.savetxt(file, labels)

    def plot(self):
        suffix = self._eval_output_suffix()
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}{suffix}")
        preds_path = os.path.join(self.cfg.run_dir, f"preds_{self.cfg.run_idx}{suffix}")
        os.makedirs(plot_path, exist_ok=True)
        title = f"{type(self.model.net).__name__} - {self.cfg.exp_name}"
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack((self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1)
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = torch.stack(self.grad_norm_train).cpu()
            plot_dict["grad_norm_frames"] = torch.stack(self.grad_norm_frames).cpu()
            plot_dict["grad_norm_net"] = torch.stack(self.grad_norm_net).cpu()
            for metric_dict in [self.train_metrics, self.val_metrics]:
                for key, value in metric_dict.items():
                    if len(value) > 0:
                        plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, preds_path, title, plot_dict)

    def _init_loss(self):
        if self.model.output == "split":

            def custom_loss(y_pred, label):
                scalar_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_pred[:, 0], label
                )
                pseudoscalar_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_pred[:, 1], label
                )
                return scalar_loss + pseudoscalar_loss

            self.loss = custom_loss
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        else:
            metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        self.val_loss.append(metrics["loss"])
        for key in self.val_metrics:
            if key in metrics:
                self.val_metrics[key].append(metrics[key])

        LOGGER.info(
            ("Validation before training: " if step < 0 else f"Validation at iteration {step}: ")
            + ", ".join(
                f"{key}={value:.6f}"
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            )
        )
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker, *_ = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)

        metrics = tracker
        return loss, metrics

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x.to(self.momentum_dtype)
        scalars = batch.scalars.to(self.dtype)
        ptr = batch.ptr
        label = batch.label.to(self.dtype)
        return fourmomenta, scalars, ptr, label

    def _batch_ref(self, batch):
        return batch.get("ref")

    def _get_ypred_and_label(self, batch, val=False):
        fourmomenta, scalars, ptr, label, *_ = self._extract_batch(batch)
        embedding = embed_tagging_data(
            fourmomenta,
            scalars,
            ptr,
            self.cfg.data,
            n_ps=self.n_ps_data,
            split_parity=self.split_parity,
            ref=self._batch_ref(batch),
        )
        embedding["num_graphs"] = label.shape[0]
        embedding["label"] = label
        y_pred, tracker, frames = self.model(embedding, val)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss) and self.model.output != "split":
            y_pred = y_pred[:, 0]
        return y_pred, label, tracker, frames

    def _init_metrics(self):
        metrics = {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }
        if self.modelname in PSEUDOSCALAR_MODELS:
            for output in ["scalar", "pseudoscalar"]:
                for lbl in [0, 1]:
                    metrics[f"{output}_output_{lbl}"] = []
                    metrics[f"{output}_output_{lbl}_std"] = []

        return metrics


class DownTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 5  # Differentiate between fatjet, subjets, and constituents
        if self.cfg.data.pid:
            self.extra_scalars += 6
        if self.cfg.data.use_ditop:
            self.extra_scalars += 2

    def _init_experiment(self, run_name_add=None):
        run_name_add = [
            "dataset=unpolarized",
            f"mass={self.cfg.data.mass_window}",
            f"ditop={self.cfg.data.use_ditop}",
            f"data={'all' if self.cfg.data.length == -1 else self.cfg.data.length}",
            f"lr={self.cfg.training.lr}",
            f"opt={self.cfg.training.optimizer}",
            f"bs={self.cfg.training.batchsize}",
        ]
        super()._init_experiment(run_name_add=run_name_add)

    def _eval_output_suffix(self):
        suffix = super()._eval_output_suffix()
        if suffix == "_ttsmeft":
            # one labels.txt per tt_smeft task, read by tt_smeft via data.downtag_labels
            suffix += f"_{self.cfg.data.ttsmeft.lbl_0}"
            if self.cfg.data.ttsmeft.c_value != 2.6:  # e.g. 0.1 -> _c0p1
                suffix += f"_c{self.cfg.data.ttsmeft.c_value}".replace(".", "p")
        return suffix

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x.to(self.momentum_dtype)
        scalars = batch.scalars.to(self.dtype)
        ptr = batch.ptr
        label = batch.label.to(self.dtype)
        lep_vec = batch.lep_vec.to(self.momentum_dtype) if not self.cfg.data.use_ditop else None
        lep_charge = batch.lep_charge.to(self.dtype) if not self.cfg.data.use_ditop else None
        return fourmomenta, scalars, ptr, label, lep_vec, lep_charge

    def init_data(self, mode="train"):
        if not self.cfg.train and self.cfg.data.dataset_eval == "polarized":
            pol_experiment = PolarizationExperiment
            pol_experiment.init_data(self, mode=mode)

        elif not self.cfg.train and self.cfg.data.dataset_eval == "ttsmeft":
            self._init_ttsmeft_eval_data()

        else:
            if self.data:
                return

            if self.cfg.data.mass_window == "strict":
                top_window = "165_190"
            elif self.cfg.data.mass_window == "loose":
                top_window = "120_220"
            else:
                raise ValueError(f"Unknown mass window definition '{self.cfg.data.mass_window}'")

            filenames = sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"top_pol_0_*_{top_window}_train*.parquet",
                    )
                )
            )

            dataset_kwargs = {
                "length": getattr(self.cfg.data, "length", None),
                "tag": self.cfg.data.tag,
                "pid": self.cfg.data.pid,
                "use_ditop": self.cfg.data.use_ditop,
            }
            self._init_data(DownTaggingDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)

    def _init_ttsmeft_eval_data(self):
        if self.data:
            return

        if self.cfg.data.use_ditop:
            raise NotImplementedError(
                "data.dataset_eval='ttsmeft' currently only supports use_ditop=False."
            )

        ttsmeft_cfg = self.cfg.data.ttsmeft
        assert ttsmeft_cfg.tag_down, "data.ttsmeft.tag_down=false leaves no true down tag"
        data_dir = ttsmeft_cfg.data_dir

        filenames = TTSmeftExperiment._select_filenames(
            data_dir,
            ttsmeft_cfg.mass_window,
            ttsmeft_cfg.lbl_0,
            ttsmeft_cfg.bsm,
            ttsmeft_cfg.c_value,
        )

        dataset_kwargs = TTSmeftExperiment._dataset_kwargs(ttsmeft_cfg)
        ttsmeft_dataset = TTSmeftDataset(**dataset_kwargs)
        ttsmeft_dataset.load_data(
            data_dir,
            filenames,
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )

        network_dtype = torch.float64 if self.cfg.use_float64 else torch.float32
        downtag_graphs = [
            ttsmeft_graph_to_downtag_input(
                ttsmeft_dataset[i], ttsmeft_dataset.layout, self.cfg.data.pid, network_dtype
            )[0]
            for i in ttsmeft_dataset.get_subset("test").tolist()
        ]

        downtag_dataset = TaggingDataset(downtag_graphs)

        self.data = downtag_dataset
        self.data_train = downtag_dataset
        self.data_val = downtag_dataset
        self.data_test = downtag_dataset

        LOGGER.info(
            f"Built {len(downtag_graphs)} down-tagger-compatible events from the fixed tt_smeft "
            f"test split ({data_dir}) for evaluation with data.dataset_eval='ttsmeft'"
        )

    def evaluate(self):
        super().evaluate()

        for set_label in self.cfg.evaluation.eval_set:
            pol = get_polarization(self.results[set_label], exp="downtagging")
            LOGGER.info(
                f"Degree of polarization: "
                f"{pol['pol_l'][-1]:.3f} (L) and {pol['pol_r'][-1]:.3f} (R) at efficiency {pol['efficiency'][-1]}, "
                f"{pol['pol_l'][375]:.3f} (L) and {pol['pol_r'][375]:.3f} (R) at efficiency {pol['efficiency'][375]}, "
                f"{pol['pol_l'][0]:.3f} (L) and {pol['pol_r'][0]:.3f} (R) at efficiency {pol['efficiency'][0]}"
            )
            self.results[set_label]["efficiencies"] = pol["efficiency"]
            self.results[set_label]["pol_l"] = pol["pol_l"]
            self.results[set_label]["pol_r"] = pol["pol_r"]
            self.results[set_label]["score_l"] = pol["score_l"][-1]
            self.results[set_label]["score_r"] = pol["score_r"][-1]
            self.results[set_label]["beta_l"] = pol["beta_l"]
            self.results[set_label]["beta_r"] = pol["beta_r"]

            polarized = self.cfg.data.dataset_eval == "polarized"
            probe = probe_report(self.results[set_label], polarized)
            self.results[set_label]["probe"] = probe
            at = [
                int(np.argmin(np.abs(np.array(probe["efficiencies"]) - e))) for e in (0.2, 0.5, 1)
            ]
            self.results[set_label]["extra"] = {
                "probe": probe,
                "pol_l": {
                    str(pol["efficiency"][i]): float(pol["pol_l"][i]) for i in (0, 375, 1000)
                },
                "pol_r": {
                    str(pol["efficiency"][i]): float(pol["pol_r"][i]) for i in (0, 375, 1000)
                },
            }
            for name in ["net", "kinematic", "soft", "truth"]:
                LOGGER.info(
                    f"Probe {name} ({set_label}), efficiency 0.2 / 0.5 / 1.0: "
                    + " ".join(
                        f"{key}=" + "/".join(f"{probe[name][key][i]:.3f}" for i in at)
                        for key in probe[name]
                        if not key.endswith("_err") and key != "beta_truth_raw"
                    )
                )
        if self.cfg.save:
            self._save_results()

        # labels.txt holds the down tags that the tt_smeft evaluation reads (data.downtag_labels)
        if self.cfg.data.dataset_eval == "ttsmeft" and not self.cfg.evaluation.save_labels:
            self._save_metrics()

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        metrics = super()._evaluate_single(loader, title, mode, step)
        momenta = {}

        # predictions
        mom_hadtop, mom_b, mom_hard, mom_soft, mom_leptop, mom_lep, charge = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        down_hard = []
        self.model.eval()
        for batch in loader:
            fourmomenta, scalars, _, _, lep_vec, lep_charge = self._extract_batch(batch)
            # the true down tag; L/R datasets carry it separately, the others have it as label
            down_hard.append(getattr(batch, "down_hard", batch.label).flatten().cpu().double())
            mask_hadtop, mask_b, mask_hard, mask_soft = (
                (scalars[:, 0] == 1),
                (scalars[:, 1] + scalars[:, 2] == 2),
                (scalars[:, 1] + scalars[:, 3] == 2),
                (scalars[:, 1] + scalars[:, 4] == 2),
            )
            mom_hadtop.append(fourmomenta[mask_hadtop])
            mom_b.append(fourmomenta[mask_b])
            mom_hard.append(fourmomenta[mask_hard])
            mom_soft.append(fourmomenta[mask_soft])
            if lep_vec is None:
                mask_leptop, mask_lep = (
                    (scalars[:, 5] == 1),
                    (scalars[:, 6] != 0),
                )
                mom_leptop.append(fourmomenta[mask_leptop])
                mom_lep.append(fourmomenta[mask_lep])
                charge.append(scalars[scalars[:, 6] != 0][:, 6])
            else:
                mom_leptop.append(lep_vec[::2])
                mom_lep.append(lep_vec[1::2])
                charge.append(lep_charge)

        mom_hadtop, mom_b, mom_hard, mom_soft, mom_leptop, mom_lep, charge = (
            torch.cat(mom_hadtop).cpu(),
            torch.cat(mom_b).cpu(),
            torch.cat(mom_hard).cpu(),
            torch.cat(mom_soft).cpu(),
            torch.cat(mom_leptop).cpu(),
            torch.cat(mom_lep).cpu(),
            torch.cat(charge).cpu(),
        )

        if mode == "eval":
            (
                momenta["q_hadtop"],
                momenta["q_b"],
                momenta["q_hard"],
                momenta["q_soft"],
                momenta["q_leptop"],
                momenta["q_lep"],
                momenta["charge"],
            ) = (
                mom_hadtop,
                mom_b,
                mom_hard,
                mom_soft,
                mom_leptop,
                mom_lep,
                charge,
            )
            momenta["down_hard"] = torch.cat(down_hard)

        results = metrics | momenta

        return results


class PolarizationExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 5  # Differentiate between fatjet, subjets, and constituents
        if self.cfg.data.pid:
            self.extra_scalars += 5
        if self.cfg.data.use_ditop:
            self.extra_scalars += 2
        # L and R are CP mirrors, the trailing charge column is C-odd
        self.ps_task = True
        self.n_ps_data = int(self.cfg.data.pid or self.cfg.data.use_ditop)

    def _init_experiment(self, run_name_add=None):
        run_name_add = [
            "dataset=polarized",
            f"mass={self.cfg.data.mass_window}",
            f"ditop={self.cfg.data.use_ditop}",
            f"data={'all' if self.cfg.data.length == -1 else self.cfg.data.length}",
            f"lr={self.cfg.training.lr}",
            f"opt={self.cfg.training.optimizer}",
            f"bs={self.cfg.training.batchsize}",
        ]
        super()._init_experiment(run_name_add=run_name_add)

    def init_data(self, mode="train"):
        if self.data:
            return

        if self.cfg.data.mass_window == "strict":
            top_window = "165_190"
        elif self.cfg.data.mass_window == "loose":
            top_window = "120_220"
        else:
            raise ValueError(f"Unknown mass window definition '{self.cfg.data.mass_window}'")

        filenames = {
            "L": sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"top_pol_had_L*_{top_window}_train*.parquet",
                    )
                )
            ),
            "R": sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"top_pol_had_R*_{top_window}_train*.parquet",
                    )
                )
            ),
        }

        dataset_kwargs = {
            "length": getattr(self.cfg.data, "length", None),
            "pid": self.cfg.data.pid,
            "use_ditop": self.cfg.data.use_ditop,
            # the down tagger also evaluates on this data (dataset_eval=polarized), in its own layout
            "charge_last": self.ps_task,
        }
        self._init_data(PolarizationDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)

    def evaluate(self):
        super().evaluate()

        for set_label in self.cfg.evaluation.eval_set:
            pol = get_polarization(self.results[set_label], exp="polarization")
            LOGGER.info(
                f"Degree of polarization: "
                f"{pol['pol_l'][-1]:.3f} (L) and {pol['pol_r'][-1]:.3f} (R) at efficiency "
                f"{pol['efficiency'][-1]}"
            )
            self.results[set_label]["efficiencies"] = pol["efficiency"]
            self.results[set_label]["pol_l"] = pol["pol_l"]
            self.results[set_label]["pol_r"] = pol["pol_r"]
            self.results[set_label]["score_l"] = pol["score_l"][-1]
            self.results[set_label]["score_r"] = pol["score_r"][-1]
            self.results[set_label]["beta_l"] = pol["beta_l"]
            self.results[set_label]["beta_r"] = pol["beta_r"]


class TTSmeftExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = TTSmeftLayout(pid=self.cfg.data.pid, include_ps=self.cfg.data.pseudoscalars)
        self.extra_scalars = self.layout.n_scalars
        self.ps_task = True
        self.n_ps_data = self.layout.n_ps
        assert self.cfg.data.max_particles is None, (
            "data.max_particles crops the leptonic top and the lepton, set it to null"
        )

        # `_cpflip_batch` relabels the CP-conjugate copy as the opposite class
        if self.cfg.data.cpflip_batch and (
            self.cfg.data.lbl_0 != "neg" or self.cfg.data.bsm != "im"
        ):
            raise ValueError(
                "data.cpflip_batch=true is only valid for lbl_0='neg' with bsm='im', where the "
                "two classes are exact CP mirror images. Got "
                f"lbl_0='{self.cfg.data.lbl_0}', bsm='{self.cfg.data.bsm}'."
            )

    @staticmethod
    def _dataset_kwargs(cfg_data):
        """Dataset options that define the tt_smeft graphs."""
        return {
            "length": cfg_data.length,
            "pid": cfg_data.pid,
            "include_ps": cfg_data.pseudoscalars,
            "tag_down": cfg_data.tag_down,
        }

    def _init_experiment(self, run_name_add=None):
        run_name_add = [
            "dataset=ttsmeft",
            f"mass={self.cfg.data.mass_window}",
            f"data={'all' if self.cfg.data.length == -1 else self.cfg.data.length}",
            f"lr={self.cfg.training.lr}",
            f"opt={self.cfg.training.optimizer}",
            f"bs={self.cfg.training.batchsize}",
        ]
        super()._init_experiment(run_name_add=run_name_add)

    @staticmethod
    def _select_filenames(data_dir, mass_window, lbl_0, bsm, c_value=None):
        if mass_window == "strict":
            top_window = "165_190"
        elif mass_window == "loose":
            top_window = "120_220"
        else:
            raise ValueError(f"Unknown mass window definition '{mass_window}'")

        all_files = sorted(
            glob.glob(os.path.join(data_dir, f"tt_ctg_*_{top_window}_train*.parquet"))
        )

        def _coeffs(filename):
            m = re.search(
                r"_re_(-?[\d]+(?:p[\d]+)?)_im_(-?[\d]+(?:p[\d]+)?)_", os.path.basename(filename)
            )
            if m is None:
                return None, None
            return float(m.group(1).replace("p", ".")), float(m.group(2).replace("p", "."))

        if bsm not in ("re", "im", "mix"):
            raise ValueError(
                f"Unknown type for tt SMEFT data '{bsm}' (expected 're', 'im', or 'mix')"
            )

        def _select(sign):
            """Files whose active coefficient(s) have the given sign and |c| == c_value."""
            selected = []
            for f in all_files:
                re_c, im_c = _coeffs(f)
                if re_c is None:
                    continue
                if bsm == "re":
                    active, inactive = (re_c,), (im_c,)
                elif bsm == "im":
                    active, inactive = (im_c,), (re_c,)
                else:
                    active, inactive = (re_c, im_c), ()
                if any(a == 0 for a in active) or any(sign * a < 0 for a in active):
                    continue
                if any(i != 0 for i in inactive):
                    continue
                if c_value is not None and any(abs(a) != float(c_value) for a in active):
                    continue
                selected.append(f)
            return selected

        if lbl_0 == "sm":
            null_files = [f for f in all_files if "re_0_im_0_" in f]
        elif lbl_0 == "neg":
            null_files = _select(-1)
        else:
            raise ValueError(f"Unknown lbl_0 '{lbl_0}' for tt SMEFT data (expected 'sm' or 'neg')")
        pos_files = _select(+1)
        assert null_files and pos_files, (
            f"No tt_smeft files for one label with lbl_0={lbl_0}, bsm={bsm}, c_value={c_value} "
            f"in {data_dir}"
        )
        return {0: null_files, 1: pos_files}

    def init_data(self, mode="train"):
        if self.data:
            return

        filenames = self._select_filenames(
            self.cfg.data.data_dir,
            self.cfg.data.mass_window,
            self.cfg.data.lbl_0,
            self.cfg.data.bsm,
            self.cfg.data.c_value,
        )

        LOGGER.info(
            f"Using tt SMEFT dataset from {self.cfg.data.data_dir} for '{self.cfg.data.bsm}' tagging "
            f"with {len(filenames[0])} files labeled 0 and {len(filenames[1])} files labeled 1."
        )

        dataset_kwargs = self._dataset_kwargs(self.cfg.data)
        self._init_data(TTSmeftDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)

        if not self.cfg.train and self.cfg.data.dataset_eval == "downtag":
            self._relabel_test_data()

    def _relabel_test_data(self):
        """Replace the true down tag of the test split by the down-tagger prediction."""
        # labels.txt of a downtagging run with data.dataset_eval='ttsmeft': one row per test
        # event, (down subjet is the leading one, predicted probability of that)
        assert self.cfg.data.tag_down, "data.dataset_eval=downtag requires data.tag_down=true"
        path = self.cfg.data.downtag_labels
        labels = np.loadtxt(path)
        true_leading, pred_leading = labels[:, 0] == 1, labels[:, 1].round() == 1

        network_dtype = torch.float64 if self.cfg.use_float64 else torch.float32
        test_graphs = [self.data[i] for i in self.data.get_subset("test").tolist()]
        # guards against labels from a differently built test split
        own_leading = [
            ttsmeft_graph_to_downtag_input(graph, self.layout, False, network_dtype)[1]
            for graph in test_graphs
        ]
        if not np.array_equal(own_leading, true_leading):
            raise ValueError(
                f"Down tags in {path} do not match the tt_smeft test split. Run the downtagging "
                "experiment with data.dataset_eval='ttsmeft' and the same tt_smeft settings."
            )

        self.data_test = TaggingDataset(
            relabel_ttsmeft_graph(graph, swap, self.layout, network_dtype, self.momentum_dtype)
            for graph, swap in zip(test_graphs, true_leading != pred_leading, strict=True)
        )
        self.downtag_labels = labels
        self.downtag_accuracy = float((true_leading == pred_leading).mean())
        LOGGER.info(
            f"Relabeled the down tag of {(true_leading != pred_leading).sum()}/{len(test_graphs)} "
            f"tt_smeft test events with the down-tagger predictions from {path}"
        )

    def _eval_output_suffix(self):
        suffix = super()._eval_output_suffix()
        if not self.cfg.train and self.cfg.data.c_value != 2.6:  # e.g. 0.1 -> _c0p1
            suffix += f"_c{self.cfg.data.c_value}".replace(".", "p")
        return suffix

    def evaluate(self):
        super().evaluate()
        if "test" in self.cfg.evaluation.eval_set and self.cfg.data.bsm == "im":
            self._evaluate_asymmetry()
        if self.cfg.save:
            self._save_results()

    def _event_jets(self, idx):
        """Momenta (n, 4, 6) as in `stack_jets`, lepton charge and the truth down-is-harder flag."""
        data = self.data
        start, end = data.ptr[idx], data.ptr[idx + 1]
        down, other = data.x[start + 2], data.x[start + 3]  # tag_down slots
        harder = rest_frame_energy(down, data.x[start]) >= rest_frame_energy(other, data.x[start])
        hard, soft = (torch.where(harder[:, None], a, b) for a, b in [(down, other), (other, down)])
        jets = torch.stack(
            [data.x[start], data.x[start + 1], hard, soft, data.x[end - 2], data.x[end - 1]], dim=-1
        )
        return jets, data.scalars[end - 1, self.layout.charge_col].double(), harder.double()

    def _evaluate_asymmetry(self):
        """Sign asymmetries of O1 and of the network output on the test events, per coupling."""
        assert self.cfg.data.tag_down, "O1 with the truth down quark needs data.tag_down=true"
        cfg, res = self.cfg.data, self.results["test"]
        jets, charge, harder = self._event_jets(self.data.get_subset("test"))
        arrays = {
            "labels": res["labels_true"].numpy(),
            "omega": (2 * res.get("labels_predict_pseudo", res["labels_predict"]) - 1).numpy(),
            "o1_truth": o1(jets, charge, harder).numpy(),
            "c0": 0.0 if cfg.lbl_0 == "sm" else -float(cfg.c_value),
            "c1": float(cfg.c_value),
        }
        if cfg.dataset_eval == "downtag":
            labels = self.downtag_labels
            assert np.array_equal(labels[:, 0], harder.numpy()), "labels.txt misaligned with test"
            arrays["o1_tagger"] = o1(jets, charge, torch.from_numpy(labels[:, 1])).numpy()
            res.setdefault("extra", {})["downtag_label_accuracy"] = self.downtag_accuracy
        asym = {}
        for key in [k for k in arrays if k.startswith(("o1_", "omega"))]:
            for label in (0, 1):
                a, sigma = asymmetry(arrays[key][arrays["labels"] == label])
                asym[f"{key}_c={arrays[f'c{label}']:g}"] = {"A": a, "sigma": sigma}
                LOGGER.info(
                    f"Asymmetry of {key} at c = {arrays[f'c{label}']:g}: {a:+.5f} +- {sigma:.5f}"
                )
        res.setdefault("extra", {})["asymmetry"] = asym
        if self.cfg.save:
            path = os.path.join(
                self.cfg.run_dir, f"preds_{self.cfg.run_idx}{self._eval_output_suffix()}"
            )
            os.makedirs(path, exist_ok=True)
            np.savez(os.path.join(path, "o1.npz"), **arrays)

    def _cpflip_batch(self, fourmomenta, scalars, ptr, label):
        new_fourmomenta = torch.cat(
            [fourmomenta, fourmomenta * fourmomenta.new_tensor([1, -1, -1, -1])], dim=0
        )  # Flip momentum
        # the trailing columns are parity-odd ones
        scalar_signs = scalars.new_ones(scalars.shape[1])
        scalar_signs[-self.layout.n_ps :] = -1
        new_scalars = torch.cat([scalars, scalars * scalar_signs], dim=0)
        new_label = torch.cat([label, 1 - label], dim=0)
        new_ptr = torch.cat([ptr, ptr[1:] + ptr[-1]])
        return new_fourmomenta, new_scalars, new_ptr, new_label

    def _extract_batch(self, batch):
        fourmomenta, scalars, ptr, label = super()._extract_batch(batch)
        if self.cfg.data.cpflip_batch and self.model.training:
            return self._cpflip_batch(fourmomenta, scalars, ptr, label)
        return fourmomenta, scalars, ptr, label

    def _batch_ref(self, batch):
        ref = batch.get("ref")
        if self.cfg.data.cpflip_batch and self.model.training:
            # the CP-conjugate copy keeps the reference tokens of the original
            ref = torch.cat([ref, ref])
        return ref


class PseudoscalarExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = (
            10  # Differentiate between tops, higgs, photons, subjets, and constituents
        )
        if self.cfg.data.pid:
            self.extra_scalars += 5
        self.ps_task = True
        self.n_ps_data = 1  # trailing C-odd charge column

    def _init_experiment(self, run_name_add=None):
        run_name_add = [
            f"exp={self.cfg.exp_type}",
            f"mass={self.cfg.data.mass_window}",
            f"data={'all' if self.cfg.data.length == -1 else self.cfg.data.length}",
            f"lr={self.cfg.training.lr}",
            f"opt={self.cfg.training.optimizer}",
            f"bs={self.cfg.training.batchsize}",
        ]
        super()._init_experiment(run_name_add=run_name_add)

    def init_data(self, mode="train"):
        if self.data:
            return

        if self.cfg.data.mass_window == "strict":
            top_window = "165_190"
        elif self.cfg.data.mass_window == "loose":
            top_window = "120_220"
        else:
            raise ValueError(f"Unknown mass window definition '{self.cfg.data.mass_window}'")

        filenames = {
            0: sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"ttH_alpha_45_*_{top_window}_train_*.parquet",
                    )
                )
            ),
            1: sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"ttH_alpha_-45_*_{top_window}_train_*.parquet",
                    )
                )
            ),
        }

        dataset_kwargs = {
            "length": getattr(self.cfg.data, "length", None),
            "pid": self.cfg.data.pid,
        }
        self._init_data(PseudoscalarDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)


class VBFExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 3
        self.ps_task = True

    def _init_experiment(self):
        BaseExperiment._init_experiment(self)

    def init_physics(self):
        super().init_physics()
        self.cfg.exp_type += f"_{self.cfg.data.exp_type}"

    def init_data(self, mode="train"):
        if self.data:
            return

        if self.cfg.data.exp_type == "pseudoscalar":
            filenames = {
                0: os.path.join(
                    self.cfg.data.data_dir,
                    "cHWtil_pos.npy",
                ),
                1: os.path.join(
                    self.cfg.data.data_dir,
                    "cHWtil_neg.npy",
                ),
            }
        elif self.cfg.data.exp_type == "cp":
            filenames = {
                0: os.path.join(
                    self.cfg.data.data_dir,
                    "cHWtil_zero.npy",
                ),
                1: os.path.join(
                    self.cfg.data.data_dir,
                    "cHWtil_pos.npy",
                ),
            }
        else:
            raise ValueError(
                f"Unknown experiment type '{self.cfg.data.exp_type}'"
                " (expected 'pseudoscalar' or 'cp')"
            )

        dataset_kwargs = {
            "length": getattr(self.cfg.data, "length", None),
        }
        self._init_data(VBFDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)
