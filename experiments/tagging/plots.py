import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss, plot_metric

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, preds_path, title, plot_dict):
    os.makedirs(preds_path, exist_ok=True)
    dataset_eval_active = not cfg.train and cfg.data.dataset_eval == "polarized"

    if cfg.plotting.loss and cfg.train:
        preds_data = {
            "train_loss": np.asarray(plot_dict["train_loss"]),
            "val_loss": np.asarray(plot_dict["val_loss"]),
            "train_lr": np.asarray(plot_dict["train_lr"]),
            "grad_norm": np.asarray(plot_dict["grad_norm"]),
        }
        for output in ["scalar", "pseudoscalar"]:
            for lbl in [0, 1]:
                for key in [f"{output}_output_{lbl}", f"{output}_output_{lbl}_std"]:
                    if key in plot_dict:
                        preds_data[key] = np.asarray(plot_dict[key])
        np.savez(f"{preds_path}/training.npz", **preds_data)

        file = f"{plot_path}/training.pdf"
        with PdfPages(file) as out:
            plot_loss(
                out,
                [plot_dict["train_loss"], plot_dict["val_loss"]],
                plot_dict["train_lr"],
                labels=["train loss", "val loss"],
                logy=True,
            )
            plot_metric(
                out,
                [plot_dict["grad_norm"]],
                "Gradient norm",
                logy=True,
            )

            for output in ["scalar", "pseudoscalar"]:
                if f"{output}_output_0" not in plot_dict:
                    continue
                stds = [f"{output}_output_{lbl}_std" for lbl in [0, 1]]
                plot_metric(
                    out,
                    [plot_dict[f"{output}_output_{lbl}"] for lbl in [0, 1]],
                    f"{output} output (val)",
                    labels=["label=0", "label=1"],
                    val_metric=True,
                    cfg=cfg,
                    metrics_std=[plot_dict[k] for k in stds]
                    if all(k in plot_dict for k in stds)
                    else None,
                )

    if cfg.plotting.score and cfg.evaluate and not dataset_eval_active:
        preds_data = {
            "labels_true": np.asarray(plot_dict["results_test"]["labels_true"]),
            "labels_predict": np.asarray(plot_dict["results_test"]["labels_predict"]),
        }
        for output in ["scalar", "pseudo"]:
            if f"labels_predict_{output}" in plot_dict["results_test"]:
                preds_data[f"labels_predict_{output}"] = np.asarray(
                    plot_dict["results_test"][f"labels_predict_{output}"]
                )
        np.savez(f"{preds_path}/score.npz", **preds_data)

        file = f"{plot_path}/score.pdf"
        with PdfPages(file) as out:
            plot_score(
                cfg,
                out,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
            )
            plot_score(
                cfg,
                out,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
                logy=True,
            )
            for output in ["scalar", "pseudo"]:
                if f"labels_predict_{output}" in plot_dict["results_test"]:
                    lp = plot_dict["results_test"][f"labels_predict_{output}"]
                    lt = plot_dict["results_test"]["labels_true"]
                    plot_score(cfg, out, lt, lp, title=f"{title} ({output})")
                    plot_score(cfg, out, lt, lp, title=f"{title} ({output})", logy=True)

    if cfg.plotting.calibration and cfg.evaluate and not dataset_eval_active:
        lt = np.asarray(plot_dict["results_test"]["labels_true"])
        preds_data, file = {}, f"{plot_path}/calibration.pdf"
        with PdfPages(file) as out:
            for output in [None, "scalar", "pseudo"]:
                key = "labels_predict" if output is None else f"labels_predict_{output}"
                if key not in plot_dict["results_test"]:
                    continue
                lp = np.asarray(plot_dict["results_test"][key])
                net = cfg.model.net._target_.split(".")[-1]
                name = net if output is None else f"{net} ({output})"
                calib = plot_calibration(out, lt, lp, label=name)
                suffix = "" if output is None else f"_{output}"
                preds_data.update({f"{k}{suffix}": v for k, v in calib.items()})
        np.savez(f"{preds_path}/calibration.npz", **preds_data)

    if cfg.plotting.roc and cfg.evaluate and not dataset_eval_active:
        preds_data = {
            "fpr": np.asarray(plot_dict["results_test"]["fpr"]),
            "tpr": np.asarray(plot_dict["results_test"]["tpr"]),
            "auc": np.asarray(plot_dict["results_test"]["auc"]),
        }
        for output in ["scalar", "pseudo"]:
            if f"fpr_{output}" in plot_dict["results_test"]:
                preds_data[f"fpr_{output}"] = np.asarray(plot_dict["results_test"][f"fpr_{output}"])
                preds_data[f"tpr_{output}"] = np.asarray(plot_dict["results_test"][f"tpr_{output}"])
                preds_data[f"auc_{output}"] = np.asarray(plot_dict["results_test"][f"auc_{output}"])
        np.savez(f"{preds_path}/roc.npz", **preds_data)

        file = f"{plot_path}/roc.pdf"
        with PdfPages(file) as out:
            plot_roc(
                out,
                plot_dict["results_test"]["fpr"],
                plot_dict["results_test"]["tpr"],
                plot_dict["results_test"]["auc"],
                title=title,
            )
            for output in ["scalar", "pseudo"]:
                if f"fpr_{output}" in plot_dict["results_test"]:
                    plot_roc(
                        out,
                        plot_dict["results_test"][f"fpr_{output}"],
                        plot_dict["results_test"][f"tpr_{output}"],
                        plot_dict["results_test"][f"auc_{output}"],
                        title=f"{title} ({output})",
                    )

    if cfg.plotting.polarization and cfg.evaluate:
        preds_data = {
            "score_l": np.asarray(plot_dict["results_test"]["score_l"]),
            "score_r": np.asarray(plot_dict["results_test"]["score_r"]),
            "efficiencies": np.asarray(plot_dict["results_test"]["efficiencies"]),
            "pol_l": np.asarray(plot_dict["results_test"]["pol_l"]),
            "pol_r": np.asarray(plot_dict["results_test"]["pol_r"]),
            "beta_l": np.asarray(plot_dict["results_test"]["beta_l"]),
            "beta_r": np.asarray(plot_dict["results_test"]["beta_r"]),
        }
        for key in ("cos_true_l", "cos_true_r"):  # downtagging only
            if key in plot_dict["results_test"]:
                preds_data[key] = np.asarray(plot_dict["results_test"][key])
        np.savez(f"{preds_path}/polarization.npz", **preds_data)

        file = f"{plot_path}/polarization.pdf"
        with PdfPages(file) as out:
            plot_pol(
                out,
                plot_dict["results_test"]["score_l"],
                plot_dict["results_test"]["score_r"],
                title=title,
            )
            plot_eff(
                out,
                plot_dict["results_test"]["efficiencies"],
                plot_dict["results_test"]["pol_l"],
                plot_dict["results_test"]["pol_r"],
                title=title,
            )


def plot_roc(out, fpr, tpr, auc, title=None):
    color = colors[2]
    rnd = np.linspace(1e-3, 1, 100)

    # usual roc
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_B$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd, "k--")
    ax.plot(fpr, tpr, color=color)
    ax.text(
        0.95,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # physicists roc
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 / \epsilon_B$", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.plot(rnd, 1 / rnd, "k--")
    ax.plot(tpr, 1 / fpr, color=color)
    ax.text(
        0.05,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # sic
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S / \sqrt{\epsilon_B}$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd**0.5, "k--")
    ax.plot(tpr, tpr / fpr**0.5, color=color)
    ax.text(
        0.95,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()


def plot_score(
    cfg, out, labels_true, labels_predicted, title=None, xrange=[0, 1], bins=100, logy=False
):
    cols = [colors[1], colors[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    if cfg.exp_type == "polarization":
        label_0, label_1 = "L", "R"
    elif cfg.exp_type == "downtagging":
        label_0, label_1 = "Up-type", "Down-type"
    elif cfg.exp_type in ["pseudoscalar", "vbf_pseudoscalar"]:
        label_0, label_1 = "Pos", "Neg"
    elif cfg.exp_type == "vbf_cp":
        label_0, label_1 = "SM", "Pos"
    elif cfg.exp_type == "tt_smeft":
        if cfg.data.lbl_0 == "sm":
            label_0 = r"$c_{tG} = 0$"
        elif cfg.data.lbl_0 == "neg":
            label_0 = r"$c_{tG} < 0$"
        label_1 = r"$c_{tG} > 0$"
    else:
        label_0, label_1 = "label=0", "label=1"
    ax.set_xlabel(f"Classifier score (0={label_0}, 1={label_1})", fontsize=FONTSIZE)
    _, bins, _ = ax.hist(
        labels_predicted[labels_true == 0],
        range=xrange,
        bins=bins,
        alpha=0.5,
        label=label_0,
        density=True,
        color=cols[0],
    )
    ax.hist(
        labels_predicted[labels_true == 1],
        bins=bins,
        label=label_1,
        alpha=0.5,
        density=True,
        color=cols[1],
    )
    ax.set_xlim(xrange)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper left")
    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()


def plot_calibration(out, labels_true, labels_predicted, label=None, bins=20):
    # reliability diagram: fraction of label=1 per score bin, binomial errors
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(labels_predicted, edges) - 1, 0, bins - 1)
    count = np.bincount(idx, minlength=bins)
    n = np.maximum(count, 1)
    score_mean = np.bincount(idx, weights=labels_predicted, minlength=bins) / n
    frac = np.bincount(idx, weights=labels_true, minlength=bins) / n
    frac_err = np.sqrt(frac * (1 - frac) / n)
    ece = np.sum(count * np.abs(frac - score_mean)) / count.sum()
    mask = count > 0
    x, y, dy = score_mean[mask], frac[mask], frac_err[mask]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("Predicted probability", fontsize=FONTSIZE)
    ax.set_ylabel("Fraction of label=1", fontsize=FONTSIZE)
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(x, y, color=colors[2], label=label)
    ax.fill_between(x, y - dy, y + dy, color=colors[2], alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="lower right")
    ax.text(
        0.05,
        0.95,
        s=f"ECE = {ece:.4f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()
    return {"edges": edges, "count": count, "score_mean": score_mean, "frac": frac, "ece": ece}


def plot_pol(out, score_l, score_r, title=None, xrange=[-1, 1], bins=20):
    cols = [colors[1], colors[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    ax.set_xlabel("Polarization distribution", fontsize=FONTSIZE)
    ax.hist(
        score_l,
        range=xrange,
        bins=bins,
        label="L",
        histtype="step",
        weights=np.ones_like(score_l) / len(score_l),
        color=cols[0],
    )
    ax.hist(
        score_r,
        range=xrange,
        bins=bins,
        label="R",
        histtype="step",
        weights=np.ones_like(score_r) / len(score_r),
        color=cols[1],
    )
    ax.set_xlim(xrange)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper left")
    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()


def plot_eff(out, efficiencies, pol_l, pol_r, title=None, xrange=[0.2, 1.0], yrange=[0.6, 1.0]):
    cols = [colors[1], colors[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylabel(r"$\beta_{\mathrm{opt}}$", fontsize=FONTSIZE)
    ax.set_xlabel("Fraction of most confident events", fontsize=FONTSIZE)
    ax.plot(
        efficiencies,
        pol_l,
        label="L",
        color=cols[0],
    )
    ax.plot(
        efficiencies,
        pol_r,
        label="R",
        color=cols[1],
    )
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper left")

    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()
