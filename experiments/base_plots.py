import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12


def plot_loss(file, losses, lr=None, labels=None, logy=True):
    if len(losses[1]) == 0:  # catch no-validations case
        losses = [losses[0]]
        labels = [labels[0]]
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0]) + 1)
    fig, ax = plt.subplots()
    for loss, label in zip(losses, labels, strict=False):
        if len(loss) == len(iterations):
            its = iterations
        else:
            # the val series starts with the pre-training validation, i.e. at iteration 0
            frac = len(losses[0]) / max(len(loss) - 1, 1)
            its = np.arange(len(loss)) * frac
        ax.plot(its, loss, label=label)

    if logy:
        ax.set_yscale("log")
    if lr is not None:
        axright = ax.twinx()
        axright.plot(iterations, lr, label="learning rate", color="crimson")
        axright.set_ylabel("Learning rate", fontsize=FONTSIZE)

    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metric(
    file,
    metrics,
    metric_label,
    labels=None,
    logy=False,
    val_metric=False,
    cfg=None,
    metrics_std=None,
    ema=False,
    yrange=None,
):
    labels = [None for _ in range(len(metrics))] if labels is None else labels
    fig, ax = plt.subplots()
    its_scale = cfg.training.validate_every_n_steps if val_metric else 1

    def _its(metric):
        # validation series start with the pre-training validation, i.e. at iteration 0
        offset = 0 if val_metric else 1
        return (np.arange(len(metric)) + offset) * its_scale

    if metrics_std is not None:
        for metric, metric_std, label in zip(metrics, metrics_std, labels, strict=False):
            its = _its(metric)
            ax.plot(its, metric, label=label)
            metric_low = np.array(metric) - np.array(metric_std)
            metric_high = np.array(metric) + np.array(metric_std)
            ax.fill_between(its, metric_low, metric_high, alpha=0.3)
    else:
        for metric, label in zip(metrics, labels, strict=False):
            its = _its(metric)
            if ema:
                metric = np.array(metric)
                mean = np.mean(metric)
                std = np.std(metric)
                metric = np.clip(metric, mean - 5 * std, mean + 5 * std)
                alpha = 0.01
                for i in range(1, len(metric)):
                    metric[i] = alpha * metric[i] + (1 - alpha) * metric[i - 1]
            ax.plot(its, metric, label=label)
            if yrange is not None:
                ax.set_ylim(yrange)

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric_label, fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper left")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()
