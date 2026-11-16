"""Generate every manuscript figure's data from the QSP model.

Edit the parameters below, then run: python generate_data.py
Each run starts from its seeds and replaces the selected output files.
No supplied data files are read. Full production settings take hours.
"""
from pathlib import Path
import json
import math
import os
import sys

# Fix BLAS thread counts before importing NumPy; the experiment loop is serial.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import scipy
from qsp import (
    sample_diagonal_kernel, beta_32_12_density,
    diagonal_coefficient_of_variation, sparse_lambda, run_point,
    dynamics_cell,
)

# =============================================================================
# EDIT HERE: output location and numerical experiment ranges
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GENERATE_INITIALIZATION = True
GENERATE_DYNAMICS = True
SCAN_STAGES = ("static", "dtcheck", "dynamic", "robustness")

# Figure 1: fixed-seed samples, exact finite-depth CV, angular correlation.
INPUT_X = 0.60
BETA_DEGREES = (16, 64, 256)
BETA_SAMPLES = 4000
BETA_SEED = 20260812
CV_INPUTS = (0.60, 0.75, 0.90)
CV_MIN_DEGREE, CV_MAX_DEGREE, CV_POINTS = 4, 4096, 150
CORRELATION_DEGREES = (16, 64, 256, 1024)
CORRELATION_U_MAX, CORRELATION_U_POINTS = 4.5, 450
CORRELATION_DELTA_MAX, CORRELATION_DELTA_POINTS = 0.8, 300

# Figure 2(a,b): discrete GD; physical time t = step * eta.
# The first cell also supplies the paired-time statistics in summary.json.
DYNAMICS_CELLS = ((1024, 4), (128, 64))
DYNAMICS_SEEDS = 30
DYNAMICS_BASE_SEED = 20260718
TEACHER_DEGREE = 8
DYNAMICS_STEP_FACTOR = 0.1  # eta = factor / (d + 1)
DYNAMICS_MAX_STEPS = 500_000
LOSS_THRESHOLD = 1.0e-4
DYNAMICS_CHECKPOINTS = 256

# Occupancy scan: seeded integer node grids, RK4 in rescaled time tau.
# At d=16384 the scan is limited to w <= 1 with 32 seeds.
# TEACHER_DEGREE above applies to both the dynamics cells and teacher scans.
NODE_COUNTS = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181,
               256, 362, 512, 724, 1024]
SCAN = {
    "base_seed": 20261008,
    "max_elems": 2_000_000,
    "max_kernel_elems": 40_000_000,
    "dynamic": {
        "sqrt_d": [16, 32, 64, 128], "n_sequence": NODE_COUNTS[:-2],
        "w_min": 0.0625, "w_max": 4.0, "d_min": 256,
        "seeds": 32, "seeds_crossover": 64, "crossover_w": [0.25, 2.0],
        "T_list": [1, 2, 3], "dt": 0.05, "targets": "teacher",
        "per_depth": {"16384": {"w_max": 1.0, "seeds_crossover": 32}},
    },
    "static": {
        "sqrt_d": [16, 32, 64, 128, 256], "n_sequence": NODE_COUNTS,
        "w_min": 0.0625, "w_max": 8.0, "seeds": 32,
    },
    "dt_check": {
        "points": [[32, 16], [64, 64]], "seeds": 8,
        "dt_main": 0.05, "dt_fine": 0.025, "T_list": [1, 2, 3],
    },
    "robustness": {
        "sqrt_d": [32, 64], "w": [0.25, 1.0, 2.0], "seeds": 32,
        "T_list": [1, 2, 3], "dt": 0.05, "targets": "zero",
    },
    "analysis": {
        "T_main": 2, "fit_w_min": 0.125, "lin_w_min": 0.125,
        "fit_w_max": 1.0, "fit_d_min": 1024, "fit_d_min_alt": 256,
        "collapse_w_max": 2.0, "b_window": [-0.15, 0.25],
        "lin_tol": 0.35, "R_mem": 0.95, "R_share": 0.5,
    },
}
BOOTSTRAP_REPEATS = 400
BOOTSTRAP_SEED = 0
EXPLORATORY_MIN_N = 16
ISO_W_VALUES = (0.25, 0.5, 1.0)

# =============================================================================
# Numerical generation and analysis
# =============================================================================
STAGES = ("static", "dtcheck", "dynamic", "robustness")

def point_seed(base_seed, tag, d, n):
    """Deterministic seed from the base seed, stage tag, d and n."""
    h = int(d) * 1_000_003 + int(n) * 7919 + sum(map(ord, tag)) * 104729
    return int((base_seed + h) % (2 ** 32 - 1))


def plan(cfg, stage):
    """List one stage's scan points with seed counts, tags and memory exclusions."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    c = cfg["dt_check" if stage == "dtcheck" else stage]
    points = []

    def add(d, n, count, times, dt, targets, dynamic=True):
        p = dict(d=d, n=n, seeds=count, T_list=times, dt=dt,
                 targets=targets, dynamic=dynamic, what=stage)
        tag = f"{stage}-{targets}" if stage != "dtcheck" else "dtcheck"
        p.update(point_seed=point_seed(
            cfg["base_seed"], tag, d, n), seed_indices=list(range(count)))
        if n * (d + 1) > cfg["max_kernel_elems"]:
            p["skip_reason"] = "n*(d+1) exceeds max_kernel_elems"
        if d + 1 > cfg["max_elems"]:
            p["skip_reason"] = "one Jacobian row exceeds max_elems"
        points.append(p)
    if stage in ("static", "dynamic"):
        for sd in c["sqrt_d"]:
            d = sd * sd
            row = dict(c, **c.get("per_depth", {}).get(str(d), {})
                       ) if stage == "dynamic" else c
            if stage == "dynamic" and d < row["d_min"]:
                continue
            for n in row["n_sequence"]:
                w = n / sd
                if n < 2 or not row["w_min"] * (
                        1 - 1e-9) <= w <= row["w_max"] * (1 + 1e-9):
                    continue
                count = row["seeds"]
                if stage == "dynamic":
                    if row["crossover_w"][0] <= w <= row["crossover_w"][1]:
                        count = row["seeds_crossover"]
                    add(d, n, count, row["T_list"], row["dt"], row["targets"])
                else:
                    add(d, n, count, [1.0], 1.0, "teacher", False)
    elif stage == "robustness":
        for sd in c["sqrt_d"]:
            for w in c["w"]:
                add(sd * sd, int(round(w * sd)),
                    c["seeds"], c["T_list"], c["dt"], c["targets"])
    else:
        for sd, n in c["points"]:
            for dt in (c["dt_main"], c["dt_fine"]):
                add(sd * sd, n, c["seeds"], c["T_list"], dt, "teacher")
    if len({_key(p) for p in points}) != len(points):
        raise ValueError("Configuration contains duplicate points")
    return points


def _key(p):
    return (p["what"], p["d"], p["n"], p["seeds"], p["targets"],
            p["dt"] if p["what"] != "static" else 0.0, tuple(p["T_list"]))


def _values(values, seeds):
    values = np.asarray(values, float)
    if seeds < 1 or values.shape != (seeds,) or not np.isfinite(values).all():
        raise ValueError("Observables need one finite value per seed")
    return values


def _points(rows, time):
    table, key = [], f"{time:g}"
    for row in rows:
        if (row.get("status", "complete") != "complete"
                or row.get("targets") != "teacher"):
            continue
        for name in ("eps", "Rtest", "dclos", "dlogL_sc", "dlogL_K0"):
            _values(row[name][key], row["seeds"])
        _values(row["kappa0"], row["seeds"])
        point = {k: row[k] for k in ("d", "n", "w", "Lambda")}
        point.update(eps=float(np.median(row["eps"][key])),
                     Rtest=float(np.median(row["Rtest"][key])),
                     _eps=row["eps"][key])
        table.append(point)
    return sorted(table, key=lambda p: (p["d"], p["n"]))


def _fit(points, bootstrap):
    result = {"n_points": len(points)}
    if len(points) < 4:
        return result
    x = np.array([[1., math.log(p["w"]), math.log(p["n"])] for p in points])
    if np.linalg.matrix_rank(x) < 3:
        return result
    beta = np.linalg.lstsq(x, [math.log(p["eps"]) for p in points],
                           rcond=None)[0]
    result.update(c0=float(beta[0]), p=float(beta[1]), b=float(beta[2]))
    if bootstrap:
        if bootstrap < 2:
            raise ValueError("bootstrap must be at least two")
        rng, samples = np.random.default_rng(BOOTSTRAP_SEED), []
        for _ in range(bootstrap):
            y = np.array([math.log(max(np.median(rng.choice(
                p["_eps"], len(p["_eps"]))), 1e-12)) for p in points])
            samples.append(np.linalg.lstsq(x, y, rcond=None)[0][2])
        result["b_ci95"] = np.percentile(samples, [2.5, 97.5]).tolist()
    return result


def _domain(table, a, depth):
    return [p for p in table if a.get("fit_w_min", 0) * (1 - 1e-9)
            <= p["w"] <= a["fit_w_max"] * (1 + 1e-9)
            and p["d"] >= depth and p["eps"] > 0]


def main_analysis(rows, config, bootstrap=400):
    """Main fit log eps = c0 + p log w + b log n, with a seeded bootstrap for b.

    W_CONTROLLED requires CI strictly inside b_window and RMS(w)<=RMS(Lambda).
    N_DEPENDENT requires CI_low above b_window; otherwise INCONCLUSIVE.
    Point order, resampling order, window tolerances and tie breaks are fixed.
    """
    a = config["analysis"]
    table = _points(rows, a["T_main"])
    fit = _fit(_domain(table, a, a["fit_d_min"]), bootstrap)
    alt = _fit(_domain(table, a, a.get("fit_d_min_alt", a["fit_d_min"])), 0)
    if "b" in alt:
        fit["b_alt_all_depths"] = alt["b"]
    collapse = [p for p in table if p["w"] <= a["collapse_w_max"] * (1 + 1e-9)
                and p["eps"] > 0]
    for variable in ("w", "Lambda"):
        residual = None
        if len(collapse) >= 4:
            x = np.array([math.log(p[variable]) for p in collapse])
            y = np.array([math.log(p["eps"]) for p in collapse])
            residual = float(np.sqrt(np.mean(
                (np.polyval(np.polyfit(x, y, 2), x) - y) ** 2)))
        fit["resid_" + variable] = residual
    outcome, reason = "INCONCLUSIVE", ""
    if "b_ci95" in fit and fit["resid_w"] is not None:
        low, high = fit["b_ci95"]
        if (a["b_window"][0] < low and high < a["b_window"][1]
                and fit["resid_w"] <= fit["resid_Lambda"]):
            outcome = "W_CONTROLLED"
        elif low > a["b_window"][1]:
            outcome = "N_DEPENDENT"
            reason = "closure error grows with n at fixed w"
        else:
            reason = ("CI of b not inside the window or collapse in w "
                      "not better than in Lambda")
    else:
        reason = ("fewer than 4 fit points" if fit["n_points"] < 4 else
                  "design matrix rank < 3 (need several depths at fixed w)")
    big = [p for p in table if p["d"] >= a["fit_d_min"] and p["eps"] > 0]
    c_lin = w_lin = w_mem = w_share = None
    for bound in (.25, .5, 1.):
        subset = [p for p in big if a.get("lin_w_min", .125) * (1 - 1e-9)
                  <= p["w"] <= bound * (1 + 1e-9)]
        if len(subset) < 2:
            continue
        ratios = np.array([p["eps"] / p["w"] for p in subset])
        c = float(np.median(ratios))
        if np.all(np.abs(ratios / c - 1) <= a["lin_tol"]):
            c_lin, w_lin = c, bound
        elif w_lin is not None:
            break
    pairs = sorted([(p["w"], p["Rtest"]) for p in big], key=lambda p: p[0])
    memory = [w for w, _ in pairs
              if all(r >= a["R_mem"] for ww, r in pairs if ww <= w)]
    sharing = [w for w, _ in pairs
               if all(r <= a["R_share"] for ww, r in pairs if ww >= w)]
    if memory:
        w_mem = 2. ** math.floor(math.log2(max(memory)) + 1e-9)
    if sharing:
        w_share = 2. ** math.ceil(math.log2(min(sharing)) - 1e-9)
    best = None
    for w in sorted({p["w"] for p in table}):
        line = sorted([p for p in table if abs(p["w"] - w) < 1e-9],
                      key=lambda p: p["n"])
        if (len(line) >= 2 and w <= 1 + 1e-9
                and abs(math.log2(w) - round(math.log2(w))) < 1e-9):
            factor = line[-1]["n"] / line[0]["n"]
            if best is None or factor > best[0] or (
                    factor == best[0] and w > best[1]):
                best = factor, w, line[-1]["Lambda"] / line[0]["Lambda"]
    return dict(outcome=outcome, reason=reason, fit=fit, c_lin=c_lin,
                w_lin=w_lin, w_mem=w_mem, w_share=w_share,
                n_factor=int(round(best[0])) if best else None,
                Lambda_factor=float(best[2]) if best else None)


def exploratory_scope(dynamic_rows, static_rows, config,
                      min_n=16, bootstrap=400):
    """Secondary analysis: the main fit restricted to n>=16, and fixed-w spreads.

    It does not change the outcome of the main fit.

    Iso-w spreads use n>=16,w<=2, at least two depths and no depth cut.
    Definitions below distinguish range/mean, population CV and max deviation.
    """
    a = config["analysis"]
    table = _points(dynamic_rows, a["T_main"])
    fit = _fit([p for p in _domain(table, a, a["fit_d_min"])
                if p["n"] >= min_n], bootstrap)
    spreads = {}
    for name, rows in (("eps", table), ("Rtest", table),
                       ("kappa0", static_rows)):
        selected = [p for p in rows
                    if p.get("status", "complete") == "complete"
                    and p["n"] >= min_n and p["w"] <= 2 * (1 + 1e-9)]
        groups = []
        for w in sorted({p["w"] for p in selected}):
            line = sorted([p for p in selected if abs(p["w"] - w) < 1e-9],
                          key=lambda p: (p["d"], p["n"]))
            if len(line) < 2:
                continue
            values = np.array([float(np.median(_values(p[name], p["seeds"])))
                               if name == "kappa0" else p[name] for p in line])
            mean = float(np.mean(values))
            groups.append(dict(
                w=w, depths=[p["d"] for p in line], values=values.tolist(),
                range_over_mean=float(np.ptp(values) / mean),
                cv=float(np.std(values) / mean),
                max_abs_deviation_over_mean=float(
                    np.max(np.abs(values - mean)) / mean)))
        spreads[name] = groups
    return dict(label="exploratory", min_n=min_n, fit=fit,
                iso_w_spreads=spreads, spread_definitions={
                    "range_over_mean": "(max-min)/mean",
                    "cv": "population std/mean",
                    "max_abs_deviation_over_mean": "max(abs(x-mean))/mean"})


def _describe(values):
    return dict(quartiles=np.percentile(values, [25, 50, 75]).tolist()
                if len(values) else None,
                range=[float(np.min(values)), float(np.max(values))]
                if len(values) else None)


def fig2_statistics(data):
    """Fig. 2(a): hitting-time ratios over runs with finite positive times."""
    times = {k: np.asarray(data["t_" + k], float) for k in ("qsp", "sc", "k0")}
    if any(v.ndim != 1 or not v.size or v.size != times["qsp"].size
           for v in times.values()):
        raise ValueError("Time arrays must be nonempty, 1D and equal lengths")
    valid = {k: np.isfinite(v) & (v > 0) for k, v in times.items()}
    masks = {k: valid["qsp"] & valid[k] for k in ("sc", "k0")}
    result = dict(n_runs=len(times["qsp"]),
                  n_used=int(np.sum(masks["sc"] & masks["k0"])))
    for k, mask in masks.items():
        stats = _describe(times[k][mask] / times["qsp"][mask])
        result.update({"n_" + k + "_used": int(np.sum(mask)),
                       k + "_quartiles": stats["quartiles"],
                       k + "_range": stats["range"]})
    result["censoring"] = {k: dict(
        missing_or_nonfinite=int(np.sum(~np.isfinite(v))),
        nonpositive=int(np.sum(np.isfinite(v) & (v <= 0))),
        finite_positive=int(np.sum(valid[k]))) for k, v in times.items()}
    result["time_stats"] = {
        k: _describe(v[valid[k]]) for k, v in times.items()}
    return result


def write_json(path, value):
    """Write strict JSON; censored or undefined scalar values become null."""
    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(item) for key, item in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [clean(item) for item in obj]
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def generate_initialization(directory):
    """Persist raw kernel samples and all deterministic Figure 1 curves."""
    rng = np.random.default_rng(BETA_SEED)
    raw, arrays = {}, {}
    for degree in BETA_DEGREES:
        output, kernel = sample_diagonal_kernel(
            x=INPUT_X, degree=degree, samples=BETA_SAMPLES, rng=rng)
        raw[f"output_{degree}"] = output
        raw[f"kernel_{degree}"] = kernel
        samples = 3 * kernel / (degree + 1)
        arrays[f"init/beta_hist/{degree}"] = np.histogram(
            samples, bins=np.linspace(0, 1.65, 55),
            density=True)[0]
    degrees = np.unique(np.rint(np.geomspace(
        CV_MIN_DEGREE, CV_MAX_DEGREE, CV_POINTS)).astype(int))
    for x in CV_INPUTS:
        arrays[f"init/cv/{x:g}"] = np.array([
            diagonal_coefficient_of_variation([x], int(d))[0] for d in degrees])
    z = np.linspace(0.002, 0.99, 500)
    u = np.linspace(0, CORRELATION_U_MAX, CORRELATION_U_POINTS)
    delta = np.linspace(0, CORRELATION_DELTA_MAX, CORRELATION_DELTA_POINTS)
    arrays.update({
        "init/input_x": INPUT_X, "init/beta_degrees": BETA_DEGREES,
        "init/bins": np.linspace(0, 1.65, 55),
        "init/beta_z": z, "init/beta_density": beta_32_12_density(z),
        "init/cv_degrees": degrees, "init/cv_inputs": CV_INPUTS,
        "init/correlation_degrees": CORRELATION_DEGREES,
        "init/u": u, "init/gaussian": np.exp(-u**2 / 2), "init/delta": delta,
    })
    for d in CORRELATION_DEGREES:
        arrays[f"init/correlation/{d}"] = np.cos(u / np.sqrt(d)) ** d
        arrays[f"init/correlation_inset/{d}"] = np.cos(delta) ** d
    np.savez_compressed(Path(directory) / "initialization.npz", **raw)
    return arrays


def dynamics_plot_arrays(raw, metadata):
    """Normalized loss curves, references and hitting-time ratios of one cell."""
    cell = metadata["cell"]
    eta, n, d = cell["eta"], cell["n"], cell["d"]
    time = raw["loss_idx"] * eta
    qsp = raw["loss_dec"] / raw["loss_dec"][:, :1]
    scalar = raw["scalar_loss_ratio"]
    times = {"t_qsp": raw["t_full"] * eta,
             "t_sc": raw["t_scalar"] * eta,
             "t_k0": raw["t_frozen"] * eta}
    hits = times["t_qsp"]
    # A small test may censor every run; use run 0 and record that selection.
    representative = (int(np.nanargmin(abs(hits - np.nanmedian(hits))))
                      if np.any(np.isfinite(hits)) else 0)
    lam = raw[f"lam_{representative}"]
    weights = raw[f"w_{representative}"]
    frozen = (weights * np.exp(-2 * np.outer(time, lam) / n)).sum(1)
    frozen /= weights.sum()
    q25, q75 = np.nanquantile(qsp, [0.25, 0.75], axis=0)
    result = dict(time=time, qsp=qsp, scalar=scalar, frozen=frozen,
                  representative=representative, q25=q25, q75=q75,
                  envelope=np.exp(-2 * (d + 1) * time[1:]), d=d, n=n, w=n / np.sqrt(d),
                  representative_has_finite_hit=bool(np.any(np.isfinite(hits))),
                  loss_threshold=cell["loss_threshold"])
    result["ratio/total"] = len(hits)
    for key, label in (("t_sc", "scalar"), ("t_k0", "frozen")):
        valid = np.isfinite(hits) & (hits > 0) & np.isfinite(times[key]) & (times[key] > 0)
        ratios = times[key][valid] / hits[valid]
        result[f"ratio_{label}_median"] = np.median(ratios) if len(ratios) else np.nan
        result[f"ratio/{key}"] = ratios
        result[f"ratio/{key}_box"] = (
            np.r_[np.percentile(ratios, [25, 50, 75]), ratios.min(), ratios.max()]
            if len(ratios) else np.full(5, np.nan))
        result[f"ratio/{key}_count"] = int(np.sum(valid))
        result[f"ratio/{key}_seed_indices"] = np.flatnonzero(valid)
    return result, times


def scan_plot_arrays(dynamic):
    """Persist per-seed summaries and reference curves consumed by the plots."""
    dynamic = [r for r in dynamic if r["status"] == "complete" and r["targets"] == "teacher"]
    arrays = {"scan/T_main": SCAN["analysis"]["T_main"]}
    time = f'{SCAN["analysis"]["T_main"]:g}'
    for metric in ("eps", "Rtest"):
        for d in sorted({r["d"] for r in dynamic}):
            points = sorted([r for r in dynamic if r["d"] == d], key=lambda r: r["w"])
            values = [r[metric][time] for r in points]
            med = np.array([np.median(v) for v in values])
            err = np.array([[m - np.percentile(v, 25), np.percentile(v, 75) - m]
                            for m, v in zip(med, values)]).T
            prefix = f"scan/{metric}/{d}/"
            arrays.update({prefix + "w": np.array([r["w"] for r in points]),
                           prefix + "n": np.array([r["n"] for r in points]),
                           prefix + "median": med, prefix + "err": err})
    for w in ISO_W_VALUES:
        points = sorted([r for r in dynamic if abs(r["w"] - w) < 1e-9], key=lambda r: r["n"])
        if not points:
            continue
        n = np.array([r["n"] for r in points])
        med = np.array([np.median(r["eps"][time]) for r in points])
        arrays.update({f"iso/{w:g}/n": n, f"iso/{w:g}/median": med,
                       f"iso/{w:g}/sqrt_n_log_n": (
                           med[0] * np.sqrt(n / n[0])
                           * (1 + np.log(n)) / (1 + np.log(n[0])))})
    return arrays


def generate_scan(directory):
    """Run the selected stages from seeds, with one raw JSONL per stage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = {stage: [] for stage in STAGES}
    for stage in SCAN_STAGES:
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for point in plan(SCAN, stage):
                row = dict(point, M=point["d"] + 1,
                           w=point["n"] / np.sqrt(point["d"]),
                           Lambda=float(sparse_lambda(point["n"], point["d"] + 1)))
                if "skip_reason" in point:
                    row.update(status="skipped", reason=point["skip_reason"])
                else:
                    options = {key: point[key] for key in
                               ("d", "n", "seeds", "T_list", "dt", "targets", "dynamic")}
                    row.update(run_point(**options, max_elems=SCAN["max_elems"],
                                         teacher_degree=TEACHER_DEGREE,
                                         rng=np.random.default_rng(point["point_seed"])),
                               status="complete")
                    if not point["dynamic"]:
                        row["dt"] = 0.0
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows[stage].append(row)
                print(f"[{stage}] {row['status']} d={row['d']} n={row['n']}", flush=True)
    return rows


def main():
    """Generate raw data, statistics and a complete data-only plotting bundle."""
    # Check plot horizons before starting the potentially multi-hour calculation.
    for stage in SCAN_STAGES:
        for point in plan(SCAN, stage):
            if (stage == "dynamic" and point["targets"] == "teacher"
                    and "skip_reason" not in point
                    and SCAN["analysis"]["T_main"] not in point["T_list"]):
                raise ValueError("Every dynamic T_list must include analysis.T_main")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = {name: value for name, value in globals().items()
              if name.isupper() and name not in ("ROOT", "DATA_DIR", "STAGES")}
    config["numpy_version"] = np.__version__
    config["scipy_version"] = scipy.__version__
    config["python_version"] = sys.version.split()[0]
    write_json(DATA_DIR / "protocol.json", config)
    arrays, times = {}, None
    if GENERATE_INITIALIZATION:
        arrays.update(generate_initialization(DATA_DIR))
        print("[initialization] complete", flush=True)
    if GENERATE_DYNAMICS:
        directory = DATA_DIR / "dynamics"
        directory.mkdir(parents=True, exist_ok=True)
        for index, (d, n) in enumerate(DYNAMICS_CELLS):
            print(f"[dynamics] starting d={d} n={n}", flush=True)
            raw, metadata = dynamics_cell(
                d, n, seeds=DYNAMICS_SEEDS, cap=DYNAMICS_MAX_STEPS,
                base_seed=DYNAMICS_BASE_SEED, teacher_degree=TEACHER_DEGREE,
                eta_scale=DYNAMICS_STEP_FACTOR, rho=LOSS_THRESHOLD,
                checkpoints=DYNAMICS_CHECKPOINTS)
            np.savez_compressed(directory / f"cell_d{d}_n{n}.npz", **raw)
            write_json(directory / f"cell_d{d}_n{n}.json", metadata)
            plot, cell_times = dynamics_plot_arrays(raw, metadata)
            panel = chr(ord("a") + index)
            arrays.update({panel + "/" + key: value for key, value in plot.items()})
            if index == 0:
                times = cell_times
            print(f"[dynamics] complete d={d} n={n}", flush=True)
    rows = generate_scan(DATA_DIR / "occupancy_scan")
    summary = main_analysis(rows["dynamic"], SCAN, BOOTSTRAP_REPEATS)
    summary["exploratory"] = exploratory_scope(
        rows["dynamic"], rows["static"], SCAN, EXPLORATORY_MIN_N, BOOTSTRAP_REPEATS)
    if times is not None:
        write_json(DATA_DIR / "occupancy_scan/fig2a_hitting_times.json", times)
        summary["fig2"] = fig2_statistics(times)
    write_json(DATA_DIR / "occupancy_scan/summary.json", summary)
    arrays.update(scan_plot_arrays(rows["dynamic"]))
    np.savez_compressed(DATA_DIR / "figure_data.npz", **arrays)
    write_json(DATA_DIR / "provenance.json", {
        "source": "Generated from the QSP model without reading supplied data",
        "initialization_generated": GENERATE_INITIALIZATION,
        "dynamics_cells_generated": DYNAMICS_CELLS if GENERATE_DYNAMICS else [],
        "scan_stages_generated": SCAN_STAGES,
        "scan_rows": {stage: len(rows[stage]) for stage in SCAN_STAGES},
        "figure_data": "Only the experiment families generated by this invocation",
    })
    print(f"Wrote raw data and figure_data.npz to {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
