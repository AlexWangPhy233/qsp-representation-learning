"""Draw Figs. 1 and 2 of the paper from saved arrays; no simulations are run here."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

# =============================================================================
# EDIT HERE: files, figures, appearance and captions
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data/figure_data.npz"
OUTPUT_DIR = ROOT / "figures"
FIGURES = ("FIG1_initialization_angular", "FIG2_dynamics_scan")
FORMATS = ("pdf", "png")
DPI = 300
FIGURE_SIZES = {
    "FIG1_initialization_angular": (10.8, 3.25),
    "FIG2_dynamics_scan": (8, 3.3),
}
FONT_FAMILY = "DejaVu Sans"
FIG1_FONT_SIZE = 9
FIG2_FONT_SIZE = 7
INSET_FONT_SIZE = 7
COLORS = ("#2b6cb0", "#dd6b20", "#319795", "#805ad5")
DYNAMICS_COLORS = {
    "qsp": "#4053d3", "scalar": "#d62728",
    "frozen": "#ddb310", "limit": "0.55",
}
DYNAMICS_STYLES = {"qsp": "-", "scalar": ":", "frozen": "--",
                   "limit": "-"}
DEPTH_CMAP = "viridis"
ISO_CMAP = "plasma"
DEPTH_MARKERS = ("o", "s", "^", "D", "v")
LINE_WIDTH = 0.8
MARKER_SIZE = 3
JITTER_SEED, JITTER_WIDTH = 0, 0.11  # Scatter placement only.
FIG2_LAYOUT = dict(left=0.065, right=0.995, bottom=0.16, top=0.83,
                   wspace=0.48)
FIG2_WIDTH_RATIOS = (1, 1, 1.1, 1.1)
FIG2_LOSS_TICKS = (1e-4, 1e-3, 1e-2, 1e-1, 1)
# The loss axes extend above unity to leave room for the ratio insets.
RATIO_INSET_POSITIONS = {"a": (0.45, 0.75, 0.51, 0.25),
                         "b": (0.45, 0.75, 0.51, 0.25)}
RATIO_INSET_LIMITS = {"a": (0.2, 6), "b": (0.01, 500)}
RATIO_INSET_TICKS = {"a": (0.25, 0.5, 1, 2, 4),
                     "b": (0.01, 0.1, 1, 10, 100)}
RATIO_INSET_FONT_SIZE = 6
RATIO_INSET_MARKER_SIZE = 1.8
RATIO_BOX_WIDTH = 0.35
RATIO_Y_LABEL = r"$t_{\rm ref}/t_{\rm QSP}$"
RATIO_X_LABELS = ("Scalar", "$K(0)$")
ISO_INSET_POSITION = (0.28, 0.7, 0.7, 0.3)
ISO_REFERENCE_COLORS = ("0.65", "0.45", "0.25")
ISO_REFERENCE_LINE_WIDTH = 0.6
EPS_GUIDE_LEVEL = 1e-1
EPS_LINE_WIDTH = 0.6
RTEST_LINE_WIDTH = 0.6
AXIS_LIMITS = {
    "beta_x": (0, 1.65), "beta_y": (0, 7.2),
    "cv_y": (0.27, 0.78), "correlation_y": (0, 1.03),
    "loss_y": (3e-5, 1e3),
    "eps_y": (0.003, 8),
    "Rtest_y": (0, 1.75),
    "iso_x": (3, 400), "iso_y": (0.035, 1.0),
}
OCCUPANCY_TICKS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8)
OCCUPANCY_LABELS = ("1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8")
CAPTIONS = {
    "FIG1_initialization_angular": (
        "Initialization fluctuations and angular locality of the mean QNTK. "
        "(a) Histograms of z=3 K_d(x,x)/(d+1), compared with the "
        "Beta(3/2,1/2) density. The input and depths are shown in the panel. "
        "Finite-depth samples can extend above z=1. (b) Exact coefficient "
        "of variation of the diagonal kernel at the indicated inputs; the "
        "dashed line is 1/3. (c) Along alpha+beta=pi/2, the normalized mean "
        "kernel cos^d(alpha-beta) is shown against u=sqrt(d)|alpha-beta|, "
        "with exp(-u^2/2) as a reference. The inset uses the unscaled angle."
    ),
    "FIG2_dynamics_scan": (
        "Training dynamics and the occupancy scan. (a,b) For each cell, "
        "the QSP run whose hitting time is closest to the median (or the "
        "first run when all hitting times are censored), with its matched "
        "scalar and frozen-kernel references. Time is the gradient-descent "
        "step multiplied by the step size. The gray Lower bound exp[-2(d+1)t] is the "
        "continuous-gradient-flow bound on normalized loss, shown as a "
        "reference for the discrete gradient-descent trajectories; it is "
        "not asserted as a bound for the discrete iterates. Each upper "
        "inset shows that cell's per-run scalar/QSP and frozen-kernel/QSP "
        "convergence-time ratios on a logarithmic scale. Boxes show the "
        "quartiles and median, whiskers span the retained minimum and "
        "maximum, and points are individual paired runs. Labels give "
        "valid pairs/total runs; each reference uses its own finite, "
        "positive pairs, so the distributions are conditional on "
        "uncensored QSP/reference pairs. Frozen-kernel crossing times "
        "come from the analytic reference and may exceed the observation "
        "cap used for QSP gradient descent. The loss axes extend above "
        "unity to accommodate the insets. "
        "(c) Median scalar-closure error epsilon versus occupancy "
        "w=n/sqrt(d), with interquartile error bars. Each seed's maximum "
        "over sampled times through tau=T={horizon} is taken before the "
        "ensemble "
        "summary. The horizontal dotted line marks epsilon={eps_guide:g} "
        "as a visual reference. The inset compares fixed-w sequences with "
        "sqrt(n)(1+ln n) shape references, each anchored at its smallest n "
        "and restricted to that sequence's n range; ln is the natural "
        "logarithm. (d) Final/initial "
        "RMS target error on the between-node test grid at tau=T={horizon}, "
        "again "
        "with medians and interquartile ranges. R_test measures an error "
        "norm ratio, not output-vector displacement. "
        "At fixed w, d=(n/w)^2; changing n also changes "
        "finite depth."
    ),
}


def _style(size):
    return {"font.family": FONT_FAMILY, "font.size": size,
            "axes.labelsize": size, "axes.titlesize": size,
            "xtick.labelsize": size, "ytick.labelsize": size,
            "legend.fontsize": size, "pdf.fonttype": 42,
            "ps.fonttype": 42, "axes.spines.top": False,
            "axes.spines.right": False, "axes.linewidth": LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH, "savefig.bbox": None}


def _available(arrays, keys, label):
    missing = [key for key in keys if key not in arrays]
    if missing:
        print(f"Skipped {label}: missing {', '.join(missing)}")
    return not missing


def _empty(ax, label):
    print(f"No saved series for {label}.")
    ax.text(0.5, 0.5, "No saved data", ha="center", va="center",
            transform=ax.transAxes)


def _save(fig, name, output, tight=False):
    for extension in FORMATS:
        fig.savefig(output / f"{name}.{extension}", dpi=DPI,
                    bbox_inches="tight" if tight else None)
    plt.close(fig)


def _caption(name, arrays):
    horizon = (f"{float(arrays['scan/T_main']):g}"
               if "scan/T_main" in arrays else "not available")
    text = CAPTIONS[name].format(horizon=horizon,
                                 eps_guide=EPS_GUIDE_LEVEL)
    if name == "FIG2_dynamics_scan":
        for panel in ("a", "b"):
            if all(f"{panel}/{key}" in arrays for key in ("d", "n", "w")):
                d, n, w = [arrays[f"{panel}/{key}"] for key in ("d", "n", "w")]
                text += (f" Cell ({panel}): d={int(d)}, n={int(n)}, "
                         f"w={float(w):.6g}.")
            threshold_key = panel + "/loss_threshold"
            if threshold_key in arrays:
                text += (f" Cell ({panel}) hitting threshold: "
                         f"L/L0 <= {float(arrays[threshold_key]):g}.")
            key = panel + "/representative_has_finite_hit"
            if key in arrays and not bool(arrays[key]):
                text += f" Cell ({panel}) has no finite QSP hitting time; "
                text += "the displayed trajectory is the first run."
            prefix = panel + "/ratio/"
            keys = ("t_sc_count", "t_k0_count", "total")
            if all(prefix + key in arrays for key in keys):
                sc, k0, total = [int(arrays[prefix + key]) for key in keys]
                text += (f" Cell ({panel}) retains {sc}/{total} scalar "
                         f"pairs and {k0}/{total} frozen-kernel pairs.")
    return text


def _depths(arrays, metric):
    return sorted({int(key.split("/")[2]) for key in arrays
                   if key.startswith(f"scan/{metric}/")
                   and key.endswith("/w")})


def _marked(ax, x, y, color, marker="o"):
    ax.plot(x, y, marker, ms=MARKER_SIZE, color=color,
            mfc=color, mew=LINE_WIDTH, ls="none")


def _series(ax, arrays, metric, linewidth):
    depths = _depths(arrays, metric)
    handles = []
    for index, depth in enumerate(depths):
        keys = [f"scan/{metric}/{depth}/{key}"
                for key in ("w", "median", "err")]
        if not _available(arrays, keys, f"{metric}, d={depth}"):
            continue
        x, y, err = [arrays[key] for key in keys]
        if not x.size:
            continue
        color = plt.get_cmap(DEPTH_CMAP)(index / max(1, len(depths) - 1))
        marker = DEPTH_MARKERS[index % len(DEPTH_MARKERS)]
        ax.errorbar(x, y, yerr=err, color=color, lw=linewidth,
                    ls="-", capsize=0)
        _marked(ax, x, y, color, marker)
        handles.append(Line2D([], [], color=color, marker=marker,
                              ms=MARKER_SIZE, lw=linewidth,
                              label=str(depth)))
    if not handles:
        _empty(ax, metric)
    return handles


def _occupancy_axis(ax, maximum=4):
    ax.set_xscale("log")
    ax.set_xlim(0.054, maximum * 1.12)
    ax.set_xticks([x for x in OCCUPANCY_TICKS if x <= maximum],
                  [s for x, s in zip(OCCUPANCY_TICKS, OCCUPANCY_LABELS)
                   if x <= maximum])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=45, pad=1)
    ax.tick_params(which="both", length=2)
    ax.set_xlabel(r"$w=n/\sqrt{d}$", labelpad=0)


def make_figure_1(arrays, output):
    """Draw saved histogram heights, exact CV and angular correlations."""
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES[
        "FIG1_initialization_angular"])
    ax = axes[0]
    if _available(arrays, ("init/beta_degrees", "init/bins"), "beta panel"):
        for index, degree in enumerate(arrays["init/beta_degrees"]):
            key = f"init/beta_hist/{int(degree)}"
            if _available(arrays, (key,), f"beta d={degree}"):
                ax.stairs(arrays[key], arrays["init/bins"],
                          color=COLORS[index % len(COLORS)], lw=1.25,
                          label=rf"$d={degree}$")
    if _available(arrays, ("init/beta_z", "init/beta_density"), "beta law"):
        ax.plot(arrays["init/beta_z"], arrays["init/beta_density"],
                "k", lw=1.6, label=r"$\mathrm{Beta}(3/2,1/2)$")
    ax.axvline(1, color="0.55", lw=0.7, ls=":")
    ax.set(xlim=AXIS_LIMITS["beta_x"], ylim=AXIS_LIMITS["beta_y"],
           xlabel=r"$3K_d(x,x)/(d+1)$", ylabel="density")
    title = "Beta law"
    if "init/input_x" in arrays:
        title += rf" at $x={float(arrays['init/input_x']):g}$"
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    if _available(arrays, ("init/cv_degrees", "init/cv_inputs"), "CV panel"):
        for index, x in enumerate(arrays["init/cv_inputs"]):
            key = f"init/cv/{x:g}"
            if _available(arrays, (key,), f"CV x={x:g}"):
                ax.plot(arrays["init/cv_degrees"], arrays[key], lw=1.35,
                        color=COLORS[index % len(COLORS)],
                        label=rf"$x={x:.2f}$")
    ax.axhline(1 / 3, color="k", lw=1, ls="--", label=r"$1/3$")
    ax.set_xscale("log", base=2)
    ax.set(ylim=AXIS_LIMITS["cv_y"], xlabel="depth $d$")
    if "init/cv_degrees" in arrays and arrays["init/cv_degrees"].size:
        degrees = arrays["init/cv_degrees"]
        ax.set_xlim(degrees[0], degrees[-1])
    ax.set_ylabel(r"$\sqrt{\operatorname{Var}K_d(x,x)}/\mathbb{E}K_d(x,x)$")
    ax.set_title("Diagonal Fluctuations")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax = axes[2]
    inset = ax.inset_axes([0.55, 0.31, 0.41, 0.27])
    if _available(arrays, ("init/correlation_degrees", "init/u"),
                  "angular panel"):
        for index, degree in enumerate(arrays["init/correlation_degrees"]):
            color = COLORS[index % len(COLORS)]
            key = f"init/correlation/{int(degree)}"
            if _available(arrays, (key,), f"correlation d={degree}"):
                ax.plot(arrays["init/u"], arrays[key], color=color,
                        lw=1.35, label=rf"$d={degree}$")
            key = f"init/correlation_inset/{int(degree)}"
            if _available(arrays, ("init/delta", key), "angular inset"):
                inset.plot(arrays["init/delta"], arrays[key],
                           color=color, lw=0.9)
    if _available(arrays, ("init/u", "init/gaussian"), "Gaussian reference"):
        ax.plot(arrays["init/u"], arrays["init/gaussian"], "k--",
                lw=1.2, label=r"$e^{-u^2/2}$")
    ax.set(ylim=AXIS_LIMITS["correlation_y"],
           xlabel=r"$u=\sqrt{d}\,|\alpha-\beta|$",
           ylabel=r"$4\,\mathbb{E}K_d(x,y)/(d+1)$")
    ax.set_title("Angular Correlation")
    if "init/u" in arrays and arrays["init/u"].size:
        ax.set_xlim(0, arrays["init/u"][-1])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, r"$\alpha+\beta=\pi/2$", transform=ax.transAxes,
            fontsize=INSET_FONT_SIZE, color="0.35")
    inset.set(ylim=(0, 1.02), xlabel=r"$|\alpha-\beta|$")
    if "init/delta" in arrays and arrays["init/delta"].size:
        inset.set_xlim(0, arrays["init/delta"][-1])
    inset.tick_params(labelsize=INSET_FONT_SIZE)
    inset.xaxis.label.set_size(INSET_FONT_SIZE)
    for side in ("top", "right"):
        inset.spines[side].set_visible(True)
    for label, ax in zip("abc", axes):
        ax.text(-0.16, 1.05, f"({label})", transform=ax.transAxes,
                fontweight="bold", fontsize=FIG1_FONT_SIZE + 2)
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "FIG1_initialization_angular", output, tight=True)


def _dynamics(ax, arrays, panel):
    keys = [f"{panel}/{key}" for key in (
        "time", "qsp", "scalar", "frozen", "representative", "envelope")]
    if not _available(arrays, keys, f"dynamics panel {panel}"):
        _empty(ax, f"dynamics panel {panel}")
        return False
    time, qsp, scalar, frozen, representative, envelope = [
        arrays[key] for key in keys]
    if time.size < 2:
        _empty(ax, f"dynamics panel {panel}: fewer than two saved times")
        return False
    representative = int(representative)
    for key, y in (
        ("qsp", qsp[representative, 1:]),
        ("scalar", scalar[representative, 1:]),
        ("frozen", frozen[1:]),
        ("limit", envelope),
    ):
        ax.plot(time[1:], y, color=DYNAMICS_COLORS[key],
                ls=DYNAMICS_STYLES[key], lw=0.9)
    ax.set(xscale="log", yscale="log", ylim=AXIS_LIMITS["loss_y"],
           xlim=(time[1] * 0.8, time[-1] * 1.3), xlabel="$t$")
    title = f"Cell ({panel})"
    if all(f"{panel}/{key}" in arrays for key in ("d", "n")):
        d, n = [int(arrays[f"{panel}/{key}"]) for key in ("d", "n")]
        title = {(1024, 4): "Sparse", (128, 64): "Dense"}.get((d, n), title)
    ax.set_title(title)
    return True


def _iso_inset(ax, arrays):
    labels = sorted({key.split("/")[1] for key in arrays
                     if key.startswith("iso/") and key.endswith("/n")},
                    key=float)
    if not labels:
        print("Skipped fixed-w inset: no saved sequences.")
        return
    inset = ax.inset_axes(ISO_INSET_POSITION)
    for index, label in enumerate(labels):
        keys = [f"iso/{label}/{key}" for key in ("n", "median", "sqrt_n_log_n")]
        if not _available(arrays, keys, f"fixed-w sequence {label}"):
            continue
        n, median, reference = [arrays[key] for key in keys]
        if not n.size:
            continue
        color = plt.get_cmap(ISO_CMAP)(
            0.15 + 0.6 * index / max(1, len(labels) - 1))
        inset.plot(n, reference, ":",
                    color=ISO_REFERENCE_COLORS[index % len(ISO_REFERENCE_COLORS)],
                    lw=ISO_REFERENCE_LINE_WIDTH, zorder=1)
        inset.plot(n, median, color=color, lw=LINE_WIDTH, zorder=2)
        _marked(inset, n, median, color)
        text = {"0.25": "1/4", "0.5": "1/2"}.get(label, label)
        inset.annotate(rf"$w = {text}$", (n[-1], median[-1]), xytext=(2, 0),
                       textcoords="offset points", fontsize=INSET_FONT_SIZE,
                       color=color, va="center")
    inset.set(xscale="log", yscale="log", xlim=AXIS_LIMITS["iso_x"],
              ylim=AXIS_LIMITS["iso_y"])
    inset.set_xticks([4, 16, 64], ["4", "16", "64"])
    inset.set_yticks([0.05, 0.2], ["0.05", "0.2"])
    inset.tick_params(which="both", labelsize=INSET_FONT_SIZE,
                      length=1, pad=0)
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.yaxis.set_minor_formatter(NullFormatter())
    inset.set_xlabel("$n$", fontsize=INSET_FONT_SIZE, labelpad=0)
    inset.xaxis.set_label_position("bottom")


def make_figure_2(arrays, output):
    """Draw dynamics, closure error and between-node target-error ratios."""
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZES[
        "FIG2_dynamics_scan"], gridspec_kw={
            "width_ratios": FIG2_WIDTH_RATIOS})
    fig.subplots_adjust(**FIG2_LAYOUT)
    for panel, ax in zip(("a", "b"), axes[:2]):
        _dynamics(ax, arrays, panel)
        ax.set_yticks(FIG2_LOSS_TICKS)
        minor_ticks = ax.yaxis.get_minorticklocs()
        ax.yaxis.set_minor_locator(FixedLocator(minor_ticks[minor_ticks < 1]))
        ax.tick_params(which="both", length=2, pad=1)
        ax.yaxis.set_minor_formatter(NullFormatter())
        _ratio_inset(ax, arrays, panel)
    axes[0].set_ylabel("$L(t)/L(0)$", labelpad=1)
    handles = [Line2D([], [], color=DYNAMICS_COLORS[key],
                      ls=DYNAMICS_STYLES[key], lw=0.9, label=label)
               for key, label in (("qsp", "QSP"), ("scalar", "Scalar"),
                                  ("frozen", "$K(0)$"),
                                  ("limit", "Lower bound"))]
    fig.legend(handles=handles, ncol=4, loc="upper left",
               bbox_to_anchor=(0.05, 1.01), frameon=False, handlelength=1.2,
               columnspacing=0.7, handletextpad=0.3)
    _series(axes[2], arrays, "eps", linewidth=EPS_LINE_WIDTH)
    depths = _series(axes[3], arrays, "Rtest", linewidth=RTEST_LINE_WIDTH)
    axes[2].axhline(EPS_GUIDE_LEVEL, color="0.45", ls=":",
                    lw=LINE_WIDTH, zorder=0)
    axes[2].set(yscale="log", ylim=AXIS_LIMITS["eps_y"],
                ylabel=r"$\epsilon$")
    axes[3].set(ylim=AXIS_LIMITS["Rtest_y"], ylabel=r"$R_{\mathrm{test}}$",
                yticks=[0, 0.5, 1, 1.5])
    for guide in (1, 0.5):
        axes[3].axhline(guide, color=".55", ls=":", lw=LINE_WIDTH)
    for ax in axes[2:]:
        _occupancy_axis(ax)
    if depths:
        axes[3].legend(handles=depths, title="$d$", loc="upper right",
                       frameon=False, ncol=2, handlelength=0.8,
                       columnspacing=0.35, handletextpad=0.2,
                       borderaxespad=0.15, labelspacing=0.2)
    _iso_inset(axes[2], arrays)
    for label, ax in zip("abcd", axes):
        ax.text(0.02, 0.97, f"({label})", transform=ax.transAxes,
                va="top", weight="bold", fontsize=FIG2_FONT_SIZE)
    _save(fig, "FIG2_dynamics_scan", output)


def _ratio_boxes(ax, arrays, prefix, marker_size):
    """Draw stored observations and box statistics without recomputing them."""
    rng = np.random.default_rng(JITTER_SEED)
    retained = False
    for index, key in enumerate(("t_sc", "t_k0"), start=1):
        keys = (prefix + key, prefix + key + "_box")
        if not _available(arrays, keys, f"convergence ratios {key}"):
            continue
        values, box = [arrays[k] for k in keys]
        if not values.size:
            print(f"Skipped convergence ratios {key}: no retained pairs.")
            continue
        retained = True
        q1, median, q3, low, high = box
        ax.bxp([dict(q1=q1, med=median, q3=q3, whislo=low, whishi=high)],
               positions=[index], widths=RATIO_BOX_WIDTH, showfliers=False,
               manage_ticks=False,
               boxprops={"linewidth": LINE_WIDTH},
               medianprops={"color": "black", "linewidth": 0.9})
        ax.plot(index + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, len(values)),
                values, "o", ms=marker_size, alpha=0.8,
                color=DYNAMICS_COLORS["scalar" if index == 1 else "frozen"])
    ax.set_xlim(0.45, 2.55)
    return retained


def _ratio_inset(ax, arrays, panel):
    prefix = panel + "/ratio/"
    keys = [prefix + key for key in ("t_sc", "t_sc_box", "t_sc_count",
                                     "t_k0", "t_k0_box", "t_k0_count", "total")]
    if not _available(arrays, keys, f"cell {panel} convergence inset"):
        return
    inset = ax.inset_axes(RATIO_INSET_POSITIONS[panel])
    retained = _ratio_boxes(inset, arrays, prefix, RATIO_INSET_MARKER_SIZE)
    total = int(arrays[prefix + "total"])
    labels = [f"{label}\n{int(arrays[prefix + key + '_count'])}/{total}"
              for label, key in zip(RATIO_X_LABELS, ("t_sc", "t_k0"))]
    inset.axhline(1, ls="--", color=".5", lw=LINE_WIDTH)
    inset.set(yscale="log", ylim=RATIO_INSET_LIMITS[panel],
              xticks=[1, 2], xticklabels=labels)
    ticks = RATIO_INSET_TICKS[panel]
    inset.set_yticks(ticks, [f"{value:g}" for value in ticks])
    inset.set_ylabel(RATIO_Y_LABEL, fontsize=RATIO_INSET_FONT_SIZE, labelpad=1)
    inset.tick_params(which="both", labelsize=RATIO_INSET_FONT_SIZE,
                      length=2, pad=1)
    inset.yaxis.set_minor_formatter(NullFormatter())
    if not retained:
        _empty(inset, f"cell {panel} finite positive convergence-time pairs")


def main():
    with np.load(DATA_FILE, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    functions = {
        "FIG1_initialization_angular": make_figure_1,
        "FIG2_dynamics_scan": make_figure_2,
    }
    for name in FIGURES:
        size = (FIG1_FONT_SIZE if name == "FIG1_initialization_angular"
                else FIG2_FONT_SIZE)
        with plt.rc_context(_style(size)):
            functions[name](arrays, OUTPUT_DIR)
        (OUTPUT_DIR / f"{name}.txt").write_text(
            _caption(name, arrays) + "\n", encoding="utf-8")
    captions = "\n\n".join(
        f"## {name}\n\n{_caption(name, arrays)}" for name in FIGURES)
    (OUTPUT_DIR / "captions.md").write_text(captions + "\n", encoding="utf-8")
    print(f"Saved figures/captions to {OUTPUT_DIR.resolve()}; "
          "see messages for unavailable series.")


if __name__ == "__main__":
    main()
