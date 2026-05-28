import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss, plot_metric

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, title, plot_dict):
    if cfg.plotting.loss and cfg.train:
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

            if "scalar_output_0" in plot_dict:
                plot_metric(
                    out,
                    [plot_dict["scalar_output_0"], plot_dict["scalar_output_1"]],
                    "Scalar output",
                    labels=["label=0", "label=1"],
                    ema=True,
                    # yrange=[-0.5, 0.5],
                )
            if "pseudoscalar_output_0" in plot_dict:
                plot_metric(
                    out,
                    [plot_dict["pseudoscalar_output_0"], plot_dict["pseudoscalar_output_1"]],
                    "Pseudoscalar output",
                    labels=["label=0", "label=1"],
                    ema=True,
                    # yrange=[-0.5, 0.5],
                )

    if cfg.plotting.score and cfg.evaluate:
        file = f"{plot_path}/score.pdf"
        with PdfPages(file) as out:
            plot_score(
                out,
                cfg.exp_type,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
            )
            plot_score(
                out,
                cfg.exp_type,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
                logy=True,
            )
            for output in ["scalar", "pseudo"]:
                if f"labels_predict_{output}" in plot_dict["results_test"]:
                    lp = plot_dict["results_test"][f"labels_predict_{output}"]
                    lt = plot_dict["results_test"]["labels_true"]
                    plot_score(out, cfg.exp_type, lt, lp, title=f"{title} ({output})")
                    plot_score(out, cfg.exp_type, lt, lp, title=f"{title} ({output})", logy=True)

    if cfg.plotting.roc and cfg.evaluate:
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
    out, exp, labels_true, labels_predicted, title=None, xrange=[0, 1], bins=100, logy=False
):
    cols = [colors[1], colors[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    if exp == "polarization":
        label_0, label_1 = "L", "R"
    elif exp == "downtagging":
        label_0, label_1 = "Up-type", "Down-type"
    elif exp in ["pseudoscalar", "vbf_pseudoscalar"]:
        label_0, label_1 = "Pos", "Neg"
    elif exp == "vbf_cp":
        label_0, label_1 = "SM", "Pos"
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
    ax.set_xlabel("Efficiency", fontsize=FONTSIZE)
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
