import datetime
import os

import hydra
import optuna
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from experiments.tagging.experiment import (
    DownTaggingExperiment,
    PolarizationExperiment,
    PseudoscalarExperiment,
    VBFExperiment,
)

GLOBAL_METRIC = None


@hydra.main(config_path="config", config_name="tagging", version_base=None)
def main(cfg):
    if cfg.optuna:
        study_name = f"{cfg.exp_name}"
        storage_name = "sqlite:///optuna.db"
        cfg.run_name = f"optuna_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=pruner,
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(cfg, trial),
            n_trials=20,
            gc_after_trial=True,
        )
        print(f"Best trial: {study.best_trial.params}")

    else:
        objective(cfg)


def objective(base_cfg, trial=None):
    cfg = OmegaConf.merge(base_cfg)
    if trial:
        cfg.run_name = os.path.join(cfg.run_name, f"trial_{trial.number}")
        cfg.optuna_trial_id = trial.number
        cfg.training.lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        cfg.training.scheduler = trial.suggest_categorical(
            "scheduler", ["CosineAnnealingLR", "OneCycleLR", "ReduceLROnPlateau"]
        )
        cfg.training.batchsize = trial.suggest_categorical("batchsize", [32, 64, 128, 256])
        cfg.training.optimizer = trial.suggest_categorical("optimizer", ["Lion", "AdamW"])
        cfg.training.weight_decay = trial.suggest_categorical(
            "weight_decay", [0.2, 0.5, 1.0, 2.0, 5.0]
        )
        if cfg.model.net._target_.startswith("lgatr."):
            cfg.model.net.num_blocks = trial.suggest_categorical("num_blocks", [6, 12])
            cfg.model.net.dropout_prob = trial.suggest_categorical("dropout_prob", [None, 0.1, 0.2])

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    try:
        if world_size > 1:
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            os.environ.setdefault("NCCL_DEBUG", "WARN")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")

            _set_common_env(world_size)
            mp.spawn(ddp_worker, nprocs=world_size, args=(cfg, trial))
            metric = GLOBAL_METRIC
        else:
            # no GPU or only one GPU -> run on main process
            metric = ddp_worker(rank=0, cfg=cfg, trial=trial)
    except optuna.TrialPruned:
        raise

    return metric


def ddp_worker(rank, cfg, trial=None):
    global GLOBAL_METRIC

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    if world_size > 1:
        # set up communication between processes
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(rank)
        print(
            "Rank",
            rank,
            "sees devices:",
            torch.cuda.device_count(),
            "current device:",
            torch.cuda.current_device(),
        )

    if cfg.exp_type == "tagging":
        raise ValueError(
            "exp_type tagging was deprecated. For polarization studies, choose 'downtagging' or 'polarization' instead."
        )
    elif cfg.exp_type == "polarization":
        constructor = PolarizationExperiment
    elif cfg.exp_type == "downtagging":
        constructor = DownTaggingExperiment
    elif cfg.exp_type == "pseudoscalar":
        constructor = PseudoscalarExperiment
    elif cfg.exp_type == "vbf":
        constructor = VBFExperiment
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp = constructor(cfg, rank, world_size, trial)
    metric = exp()
    if world_size > 1:
        metric_tensor = torch.tensor(metric if rank == 0 else 0.0, device=rank)
        dist.broadcast(metric_tensor, src=0)
        metric = metric_tensor.item()

        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()

    if rank == 0:
        GLOBAL_METRIC = float(metric) if metric else None
        return metric


def _set_common_env(world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["WORLD_SIZE"] = str(world_size)


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
