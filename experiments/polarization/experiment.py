import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.dataset import (
    Dataset,
    load_dataset,
    positional_encoding,
)
from experiments.polarization.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"LGATr": "L-GATr", "Transformer": "Tr"}


class PolarizationExperiment(BaseExperiment):
    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

            if self.cfg.evaluation.load_samples:
                self.cfg.train = False
                self.cfg.evaluation.sample = False
                self.cfg.evaluation.save_samples = False

            max_num_particles, diff, pt_min, jet_pt_min, masked_dims, load_fn = (
                load_dataset(self.cfg.data.dataset)
            )

            self.cfg.data.max_num_particles = max_num_particles
            self.cfg.data.diff = diff
            self.load_fn = load_fn

            self.cfg.wrapper_cfg.const_coordinates_options.pt_min = pt_min
            self.cfg.wrapper_cfg.const_coordinates_options.fixed_dims = masked_dims
            self.cfg.wrapper_cfg.jet_coordinates_options.pt_min = jet_pt_min

            self.cfg.model.wrapper_cfg = self.cfg.wrapper_cfg
            self.cfg.model.range = (
                self.cfg.data.min_mult,
                self.cfg.data.max_num_particles,
            )

            if self.cfg.modelname == "Transformer":
                self.cfg.model.net.in_channels = 4
                if self.cfg.wrapper_cfg.add_jet:
                    self.cfg.model.net.in_channels += 1
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_channels += 6
                if self.cfg.data.pos_encoding_dim > 0:
                    self.cfg.model.net.in_channels += self.cfg.data.pos_encoding_dim
            elif self.cfg.modelname == "LGATr":
                # scalar channels
                self.cfg.model.net.in_s_channels = len(self.cfg.model.scalar_inputs)
                if self.cfg.wrapper_cfg.add_jet:
                    self.cfg.model.net.in_s_channels += 1
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_s_channels += 6
                if self.cfg.data.pos_encoding_dim > 0:
                    self.cfg.model.net.in_s_channels += self.cfg.data.pos_encoding_dim

                # mv channels for beam_reference and time_reference
                self.cfg.model.net.in_mv_channels = 1

            else:
                raise ValueError(f"Model not implemented: {self.cfg.modelname}")

    def init_data(self):
        t0 = time.time()
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating {self.cfg.data.dataset} from {data_path}")
        if self.cfg.data.dataset == "ttbar":
            raise NotImplementedError
            # self._init_ttbar_data(data_path)
        else:
            self._init_data(data_path)
        LOGGER.info(
            f"Created {self.cfg.data.dataset} with {len(self.train_data)} training events, {len(self.val_data)} validation events, and {len(self.test_data)} test events in {time.time() - t0:.2f} seconds"
        )

    def _init_data(self, data_path):
        t0 = time.time()
        data = self.load_fn(data_path, self.cfg.data, self.dtype)
        particles = data["det_particles"]
        mults = data["det_mults"]
        pids = data["det_pids"]
        jets = data["det_jets"]
        labels = data["labels"]
        size = len(particles)

        LOGGER.info(f"Loaded {size} events in {time.time() - t0:.2f} seconds")

        if self.cfg.data.max_constituents > 0:
            mults = torch.clamp(mults, max=self.cfg.data.max_constituents)
            size = len(particles)

        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        # initialize cfm (might require data)
        self.model.init_coordinates()

        train_mask = torch.arange(particles.shape[1])[None, :] < mults[:train_idx, None]
        self.model.condition_const_coordinates.init_fit(
            particles[:train_idx],
            mask=train_mask,
            jet=torch.repeat_interleave(jets[:train_idx], mults[:train_idx], dim=0),
        )

        self.model.condition_jet_coordinates.init_fit(jets[:train_idx])

        gen_mask = torch.arange(particles.shape[1])[None, :] < mults[:, None]
        particles[gen_mask] = self.model.const_coordinates.fourmomenta_to_x(
            particles[gen_mask],
            jet=torch.repeat_interleave(jets, mults, dim=0),
            ptr=torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int64), mults], dim=0), dim=0
            ),
        )

        jets = self.model.jet_coordinates.fourmomenta_to_x(jets)

        pos_encoding = positional_encoding(pe_dim=self.cfg.data.pos_encoding_dim)

        if self.cfg.data.mult_encoding_dim > 0:
            mult_encoding = self.model.mult_encoding.to(pos_encoding.device)
        else:
            mult_encoding = None

        self.train_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.val_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.test_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )

        self.train_data.append(
            particles=particles[:train_idx],
            pids=pids[:train_idx],
            mults=mults[:train_idx],
            jets=jets[:train_idx],
            labels=labels[:train_idx],
        )
        self.val_data.append(
            particles=particles[train_idx:val_idx],
            pids=pids[train_idx:val_idx],
            mults=mults[train_idx:val_idx],
            jets=jets[train_idx:val_idx],
            labels=labels[train_idx:val_idx],
        )
        self.test_data.append(
            particles=particles[val_idx:test_idx],
            pids=pids[val_idx:test_idx],
            mults=mults[val_idx:test_idx],
            jets=jets[val_idx:test_idx],
            labels=labels[val_idx:test_idx],
        )

    def _init_dataloader(self):
        if self.cfg.evaluation.load_samples:
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            return

        train_sampler = torch.utils.data.DistributedSampler(
            self.train_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.val_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
        )
        test_sampler = torch.utils.data.DistributedSampler(
            self.test_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=test_sampler,
            num_workers=2,
            pin_memory=True,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self):
        if self.ema is not None:
            with self.ema.average_parameters():
                # self.results_train = self._evaluate_single(
                #     self.train_loader, "train"
                # )
                # self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(
                    self.test_loader, "test", "eval"
                )

            # also evaluate without ema to see the effect
            # self._evaluate_single(self.train_loader, "train_noema")
            # self._evaluate_single(self.val_loader, "val_noema")
            self._evaluate_single(self.test_loader, "test_noema", "eval")

        else:
            # self.results_train = self._evaluate_single(self.train_loader, "train")
            # self.results_val = self._evaluate_single(self.val_loader, "val")
            self.results_test = self._evaluate_single(self.test_loader, "test", "eval")

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
            y_pred, label = self._get_ypred_and_label(batch)
            y_pred = torch.nn.functional.sigmoid(y_pred)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy(
            labels_predict, labels_true
        ).item()
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
            num_parameters = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            LOGGER.info(
                f"table {title}: {framesString} ({self.cfg.training.iterations} iterations)"
                f" & {num_parameters} & {metrics['accuracy']:.4f} & {metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f} & {metrics['rej05']:.0f} & {metrics['rej08']:.0f} \\\\"
            )
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        title = type(self.model.net).__name__
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            # plot_dict["results_train"] = self.results_train
            # plot_dict["results_val"] = self.results_val
            plot_dict["results_test"] = self.results_test
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = self.train_grad_norm
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _batch_loss(self, batch):
        y_pred, label = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)

        metrics = {}
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        y_pred = self.model(batch.to(self.device))
        return y_pred, batch.label

    def _init_metrics(self):
        return {}
