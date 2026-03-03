import glob
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from experiments.base_experiment import BaseExperiment
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.dataset import QuarkTaggingDataset, TestDataset
from experiments.tagging.embedding import embed_tagging_data, get_num_tagging_features
from experiments.tagging.likelihood import get_polarization
from experiments.tagging.plots import plot_mixer


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

    def init_physics(self):
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        self.momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else torch.float32

        self.cfg.model.out_channels = self.num_outputs
        if modelname in ["LGATr", "LGATrSlim"]:
            # Lorentz-equivariance by internal representations
            in_s_channels = self.extra_scalars
            in_s_channels += get_num_tagging_features(
                tagging_features=self.cfg.data.tagging_features
            )
            self.cfg.model.net.in_s_channels = 0 if self.cfg.model.mean_aggregation else 1
            self.cfg.model.net.in_s_channels += in_s_channels

            # doesn't affect results and never needed
            self.cfg.data.boost_jet = False
        elif modelname == "Transformer":
            # Non-equivariant or canonicalization
            self.cfg.model.in_channels = 7 + self.extra_scalars
            if self.cfg.model.add_fourmomenta_backbone:
                self.cfg.model.in_channels += 4

            if modelname == "Transformer":
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
            raise NotImplementedError(f"Model {modelname} not implemented")

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
            )
            self.model.init_standardization(embedding["fourmomenta"], embedding["ptr"])

    def _init_experiment(self, run_name_add=None):
        run_name_add = [
            f"dataset={self.cfg.data.dataset}",
            f"mass={self.cfg.data.mass_window}",
            f"ditop={self.cfg.data.use_ditop}",
            f"data={'all' if self.cfg.data.length == -1 else self.cfg.data.length}",
            f"lr={self.cfg.training.lr}",
            f"opt={self.cfg.training.optimizer}",
            f"bs={self.cfg.training.batchsize}",
        ]
        super()._init_experiment(run_name_add=run_name_add)

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
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label)
            labels_predict.append(y_pred)
        labels_true, labels_predict = (
            torch.cat(labels_true).cpu().float(),
            torch.cat(labels_predict).cpu().float(),
        )

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
            labels_predict, labels_true
        ).item()
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

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

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

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
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

    def _save_metrics(self):
        metric_path = os.path.join(self.cfg.run_dir, f"preds_{self.cfg.run_idx}")
        os.makedirs(metric_path, exist_ok=True)
        LOGGER.info(f"Saving labels in {metric_path}")

        file = f"{metric_path}/labels.txt"
        labels = np.stack(
            (self.results["test"]["labels_true"], self.results["test"]["labels_predict"]), axis=-1
        )
        np.savetxt(file, labels)

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        title = f"{type(self.model.net).__name__} - {self.cfg.data.dataset}"
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
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics, _ = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        else:
            metrics, _ = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker, _ = self._get_ypred_and_label(batch)
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

    def _get_ypred_and_label(self, batch):
        fourmomenta, scalars, ptr, label, *_ = self._extract_batch(batch)
        embedding = embed_tagging_data(
            fourmomenta,
            scalars,
            ptr,
            self.cfg.data,
        )
        embedding["num_graphs"] = label.shape[0]
        y_pred, tracker, frames = self.model(embedding)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]
        return y_pred, label, tracker, frames

    def _init_metrics(self):
        return {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }


class TestExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = TestDataset.n_token_types

    def init_data(self):
        data_path = self.cfg.data.npy_path
        if not os.path.isabs(data_path):
            data_path = os.path.join(self.cfg.base_dir, data_path)
        dataset_kwargs = dict(
            seed=self.cfg.data.seed,
            split_fracs=tuple(self.cfg.data.split_fracs),
            shuffle=self.cfg.data.shuffle,
            length=self.cfg.data.length,
        )
        self._init_data(TestDataset, data_path, dataset_kwargs)


class QuarkTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 5  # Differentiate between fatjet, subjets, and constituents
        if self.cfg.data.pid:
            self.extra_scalars += 6
        if self.cfg.data.use_ditop:
            self.extra_scalars += 2

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
        if self.cfg.data.dataset_eval and mode != "train":
            dataset = self.cfg.data.dataset_eval
        else:
            dataset = self.cfg.data.dataset

        if self.data and (dataset == self.cfg.data.dataset):
            return

        if self.cfg.data.mass_window == "strict":
            top_window = "165_190"
        elif self.cfg.data.mass_window == "loose":
            top_window = "120_220"
        else:
            raise ValueError(f"Unknown mass window definition '{self.cfg.data.mass_window}'")

        if dataset == "unpolarized":
            tag_id = 1
            filenames = sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.data.data_dir,
                        f"top_pol_0_*_{top_window}_train_*.parquet",
                    )
                )
            )
        elif dataset == "polarized":
            tag_id = -1
            filenames = {
                "L": sorted(
                    glob.glob(
                        os.path.join(
                            self.cfg.data.data_dir,
                            f"top_pol_L_*_{top_window}_train_*.parquet",
                        )
                    )
                ),
                "R": sorted(
                    glob.glob(
                        os.path.join(
                            self.cfg.data.data_dir,
                            f"top_pol_R_*_{top_window}_train_*.parquet",
                        )
                    )
                ),
            }
        else:
            raise ValueError(f"Unknown dataset type '{self.cfg.data.dataset}'")

        dataset_kwargs = {
            "length": getattr(self.cfg.data, "length", None),
            "tag": tag_id,
            "pid": self.cfg.data.pid,
            "use_ditop": self.cfg.data.use_ditop,
            "use_cache": self.cfg.data.use_cache,
        }
        self._init_data(QuarkTaggingDataset, self.cfg.data.data_dir, filenames, dataset_kwargs)

    def evaluate(self):
        super().evaluate()

        for set_label in self.cfg.evaluation.eval_set:
            pol = get_polarization(self.cfg, self.results[set_label])
            LOGGER.info(
                f'Degrees of polarization: '
                f'{pol["pol_l"][0]} (L) and {pol["pol_r"][0]} (R) at efficiency {pol["efficiency"][0]}, '
                f'{pol["pol_l"][-1]} (L) and {pol["pol_r"][-1]} (R) at efficiency {pol["efficiency"][-1]}'
            )
            self.results[set_label]["score_l"] = pol["score_l"][0]
            self.results[set_label]["score_r"] = pol["score_r"][0]

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        metrics = super()._evaluate_single(loader, title, mode, step)
        momenta = {}

        # predictions
        mom_hadtop, mom_b, mom_hard, mom_soft, mom_leptop, mom_lep, charge = [], [], [], [], [], [], []
        self.model.eval()
        for batch in loader:
            fourmomenta, scalars, _, _, lep_vec, lep_charge = self._extract_batch(batch)
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
            momenta["q_hadtop"], momenta["q_b"], momenta["q_hard"], momenta["q_soft"], momenta["q_leptop"], momenta["q_lep"], momenta["charge"] = (
                mom_hadtop,
                mom_b,
                mom_hard,
                mom_soft,
                mom_leptop,
                mom_lep,
                charge,
            )

        results = metrics | momenta

        return results