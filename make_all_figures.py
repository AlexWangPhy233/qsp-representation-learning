"""Make the two manuscript figures. Edit parameters here, then run this file."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qsp_figure_tools import (
    beta_32_12_density,
    diagonal_coefficient_of_variation,
    first_crossing,
    reconstruct_initial_data,
    sample_diagonal_kernel,
    scalar_loss_ratios,
)


# =============================================================================
# Parameters
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dynamics"
OUTPUT = ROOT / "generated"

# Figure 1
INPUT_X = 0.60
BETA_DEGREES = (16, 64, 256)
BETA_SAMPLES = 4000
BETA_SEED = 20260812
CV_INPUTS = (0.60, 0.75, 0.90)
CV_MAX_DEGREE = 4096
CORRELATION_DEGREES = (16, 64, 256, 1024)

# Figure 2: (degree, number of samples, title)
DYNAMICS_CELLS = (
    (1024, 4, "(a) Sparse Regime"),
    (128, 64, "(b) Dense Regime"),
)
DYNAMICS_SEED = 20260718
TEACHER_DEGREE = 8
LOSS_THRESHOLD = 1.0e-4

COLORS = ("#2b6cb0", "#dd6b20", "#319795", "#805ad5")
DYNAMICS_COLORS = {
    "qsp": "#4053d3",
    "scalar": "#d62728",
    "frozen": "#ddb310",
    "limit": "0.55",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(figure, name):
    """Save one figure as PDF and PNG."""
    figure.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_figure_1():
    """Initialization beta law, diagonal fluctuations, and angular correlation."""
    rng = np.random.default_rng(BETA_SEED)
    bins = np.linspace(0, 1.65, 55)
    beta_samples = {}
    for degree in BETA_DEGREES:
        _, kernel = sample_diagonal_kernel(
            x=INPUT_X, degree=degree, samples=BETA_SAMPLES, rng=rng
        )
        beta_samples[degree] = 3 * kernel / (degree + 1)

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))

    # (a) Beta law
    ax = axes[0]
    for color, degree in zip(COLORS, BETA_DEGREES):
        ax.hist(
            beta_samples[degree], bins=bins, density=True, histtype="step",
            linewidth=1.25, color=color, label=rf"$d={degree}$",
        )
    z = np.linspace(0.002, 0.99, 500)
    ax.plot(z, beta_32_12_density(z), "k", lw=1.6,
            label=r"$\mathrm{Beta}(3/2,1/2)$")
    ax.axvline(1, color="0.55", lw=0.7, ls=":")
    ax.set(xlim=(0, 1.65), ylim=(0, 7.2),
           xlabel=r"$3K_d(x,x)/(d+1)$", ylabel="density")
    ax.set_title(rf"Beta law at $x={INPUT_X}$")
    ax.legend(frameon=False, loc="upper left")

    # (b) Exact coefficient of variation
    ax = axes[1]
    degrees = np.unique(np.rint(np.geomspace(4, CV_MAX_DEGREE, 150)).astype(int))
    for color, x in zip(COLORS, CV_INPUTS):
        cv = [diagonal_coefficient_of_variation([x], int(d))[0] for d in degrees]
        ax.plot(degrees, cv, color=color, lw=1.35, label=rf"$x={x:.2f}$")
    ax.axhline(1 / 3, color="k", lw=1, ls="--", label=r"$1/3$")
    ax.set_xscale("log", base=2)
    ax.set(xlim=(4, CV_MAX_DEGREE), ylim=(0.27, 0.78), xlabel="depth $d$")
    ax.set_ylabel(r"$\sqrt{\operatorname{Var}K_d(x,x)}/\mathbb{E}K_d(x,x)$")
    ax.set_title("Diagonal Fluctuations")
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # (c) Angular correlation
    ax = axes[2]
    u = np.linspace(0, 4.5, 450)
    for color, degree in zip(COLORS, CORRELATION_DEGREES):
        ax.plot(u, np.cos(u / np.sqrt(degree)) ** degree,
                color=color, lw=1.35, label=rf"$d={degree}$")
    ax.plot(u, np.exp(-u**2 / 2), "k--", lw=1.2, label=r"$e^{-u^2/2}$")
    ax.set(xlim=(0, 4.5), ylim=(0, 1.03),
           xlabel=r"$u=\sqrt{d}\,|\alpha-\beta|$",
           ylabel=r"$4\,\mathbb{E}K_d(x,y)/(d+1)$")
    ax.set_title("Angular Correlation")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, r"$\alpha+\beta=\pi/2$", transform=ax.transAxes,
            fontsize=7.5, color="0.35")

    inset = ax.inset_axes([0.55, 0.31, 0.41, 0.27])
    delta = np.linspace(0, 0.8, 300)
    for color, degree in zip(COLORS, CORRELATION_DEGREES):
        inset.plot(delta, np.cos(delta) ** degree, color=color, lw=0.9)
    inset.set(xlim=(0, 0.8), ylim=(0, 1.02), xlabel=r"$|\alpha-\beta|$")
    inset.tick_params(labelsize=6)
    inset.xaxis.label.set_size(6.5)
    inset.spines["top"].set_visible(True)
    inset.spines["right"].set_visible(True)

    for label, ax in zip("abc", axes):
        ax.text(-0.16, 1.05, f"({label})", transform=ax.transAxes,
                fontweight="bold", fontsize=11)
    figure.subplots_adjust(wspace=0.34)
    save(figure, "FIG1_initialization_angular")


def make_figure_2():
    """Sparse and dense QSP training dynamics."""
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)

    for panel, (ax, (degree, samples, title)) in enumerate(zip(axes, DYNAMICS_CELLS)):
        archive = np.load(DATA / f"cell_d{degree}_n{samples}.npz")
        metadata = json.loads((DATA / f"cell_d{degree}_n{samples}.json").read_text())
        eta = metadata["cell"]["eta"]
        seeds = metadata["cell"]["n_seeds"]

        time = archive["loss_idx"] * eta
        qsp = archive["loss_dec"] / archive["loss_dec"][:, :1]
        qsp_hit = archive["t_full"] * eta
        representative = np.nanargmin(abs(qsp_hit - np.nanmedian(qsp_hit)))

        output0, target = reconstruct_initial_data(
            degree, samples, seeds, DYNAMICS_SEED, TEACHER_DEGREE
        )
        scalar = scalar_loss_ratios(degree, samples, output0, target, time)
        scalar_hit = first_crossing(time, scalar, LOSS_THRESHOLD)
        valid = np.isfinite(qsp_hit) & np.isfinite(scalar_hit)

        q25, q75 = np.nanquantile(qsp, [0.25, 0.75], axis=0)
        eigenvalues = archive[f"lam_{representative}"]
        weights = archive[f"w_{representative}"]
        frozen = (weights * np.exp(-2 * np.outer(time, eigenvalues) / samples)).sum(1)
        frozen /= weights.sum()

        c = DYNAMICS_COLORS
        ax.fill_between(time[1:], q25[1:], q75[1:], color=c["qsp"], alpha=0.15,
                        lw=0, label="QSP GD, 25–75% band")
        ax.plot(time[1:], qsp[representative, 1:], color=c["qsp"], lw=1.8,
                label="QSP GD, representative run")
        ax.plot(time[1:], scalar[representative, 1:], color=c["scalar"], lw=2,
                ls=":", label="scalar flow, matched initialization")
        ax.plot(time[1:], frozen[1:], color=c["frozen"], lw=1.5, ls="--",
                label=r"frozen $K_0$, matched initialization")
        ax.plot(time[1:], np.exp(-2 * (degree + 1) * time[1:]),
                color=c["limit"], lw=1, label=r"speed limit $e^{-2(d+1)t}$")

        if panel == 0:
            scalar_ratio = np.median(scalar_hit[valid] / qsp_hit[valid])
            frozen_ratio = np.nanmedian(archive["t_frozen"] / archive["t_full"])
            text = (rf"median $t_{{\rm sc}}/t_{{\rm QSP}}={scalar_ratio:.2f}$" "\n"
                    rf"median $t_{{K_0}}/t_{{\rm QSP}}={frozen_ratio:.2f}$")
            ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="0.7",
                              alpha=0.88, boxstyle="round,pad=0.3"))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(time[1] * 0.8, time[-1] * 1.3), ylim=(3e-5, 2),
               xlabel=r"$t=\mathrm{step}\times\eta$")
        ax.set_title(
            rf"{title}" "\n"
            rf"$d={degree},\ n={samples},\ w={samples/np.sqrt(degree):.3g}$"
        )

    axes[0].set_ylabel(r"$L(t)/L(0)$")
    axes[1].legend(fontsize=9.5, loc="center left",
                   bbox_to_anchor=(1.02, 0.52), borderaxespad=0)
    figure.tight_layout()
    save(figure, "FIG2_two_regimes")


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    make_figure_1()
    make_figure_2()
    print(f"Wrote manuscript figures to {OUTPUT}")
