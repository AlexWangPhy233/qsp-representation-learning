# Numerical conventions

`qsp.py` defines the model and the numerical solvers. `generate_data.py`
chooses the experiment grids, runs the solvers, analyses the results and saves
raw data, summaries and plot arrays. `plot_figures.py` only reads the plot
arrays.

## Model and observables

Primes denote derivatives with respect to the rescaled time `tau`. RMS is
taken over the relevant input nodes, separately for each seed. Trajectory
maxima are evaluated at RK4 step endpoints, including `tau=0` and every
requested horizon; they are not certified continuous-time suprema.

| Quantity | Convention |
|---|---|
| Phase count and occupancy | `M=d+1`, `w=n/sqrt(d)`. |
| Training grid | `theta_a=(2a+1)pi/(4n)`, `x_a=cos(theta_a)`, `a=0,...,n-1`. |
| Test grid | `theta_a=(a+1)pi/(2n)`, `a=0,...,n-2`: the `n-1` angular midpoints between adjacent training nodes. |
| Time and phase flow | `tau=M*t/(3n)`, residual `e=g-f`, `phi'=-(3/M)J^T e`. |
| Scalar reference | Same initial output and target as QSP, evolved by `u'=-(1-u^2)(u-f)`. |
| Frozen-kernel reference | The realized `K(0)` of each seed; its eigendecomposition gives the loss. This is not an ensemble-averaged kernel. |
| `Lambda` | Sparse-data control parameter `n^(3/2)(1+ln n)/sqrt(M)`. |
| `kappa0` | Initial RMS of the cross-input forcing divided by the RMS of the diagonal forcing. |
| `dclos0` | Initial RMS of `3 K_aa/M-(1-g_a^2)`. |
| `eps[T]` | Maximum over sampled times through `T` of the RMS difference between the QSP output and its matched scalar solution. |
| `dclos[T]` | Maximum over sampled times through `T` of the RMS of `3 K_aa/M-(1-g_a^2)`. |
| `Rtest[T]` | Final divided by initial RMS target error on the test grid. |
| `dlogL_sc[T]`, `dlogL_K0[T]` | Maximum absolute log-loss difference between QSP and the scalar or frozen-kernel reference. Losses are floored at `1e-300` before taking logarithms. |
| `Lratio[T]` | Final training loss divided by initial training loss. |

## Dynamics cells, Fig. 2(a,b)

The two cells are `(d,n)=(1024,4)` (sparse) and `(128,64)` (dense), with 30
runs each, on the training grid above. Student phases and degree-8 teacher
phases of run `k` use the independent NumPy seed sequences
`[20260718,1,d,n,k]` and `[20260718,11,d,n,k]`.

The update is `phi <- phi - (eta/n) J.T @ (g-f)` with `eta=0.1/(d+1)`, and the
loss is `sum((g-f)**2)/(2n)`. Each run records its first crossing of
`L/L0 <= 1e-4`, using log-linear interpolation between adjacent float64
losses, and keeps evolving afterwards. The batch stops when every run has
crossed or at 500,000 steps. Missing crossings are NaN in NPZ files and null in
JSON. The sparse cell stops after 2260 steps. The dense cell reaches the cap,
and runs 14 and 23 (zero-based) do not cross.

Loss histories are stored as float32 at the steps
`[0] + unique(geomspace(1, steps_run, 256).astype(int))`. The scalar curves
are integrated with `solve_ivp` (`rtol=1e-9`, `atol=1e-11`) on the same time
grid, and their crossing uses log-linear interpolation on that grid.
Frozen-kernel crossings solve the spectral loss equation with Brent's method
(`xtol=1e-14`, `rtol=1e-13`) on physical time `[0,1e14]`.

`t_full`, `t_frozen` and `t_scalar` in the cell NPZ files are in **GD steps**;
multiply by `eta` for physical time. The occupancy scan instead integrates the
flow with RK4 in rescaled time; the two protocols are deliberately distinct.

## Occupancy scan, Fig. 2(c,d)

The scan has four stages: `static` (initial observables only), `dynamic`
(teacher targets, horizons `T=1,2,3`, RK4 step `0.05`), `dtcheck` (two points
rerun with step `0.025`) and `robustness` (zero targets). It contains 69 static,
46 dynamic, four step-size and six zero-target points. Two static points,
`(d,n)=(65536,724)` and `(65536,1024)`, are skipped because their kernels
exceed the memory setting; their rows record the reason.

Each row records `what, d, n, M, w, Lambda, seeds, point_seed, seed_indices,
targets, teacher_degree`, the horizons and RK4 step where applicable, and
`status`. Per-seed initial observables are `kappa0` and `dclos0`. Dynamic
observables `eps, dclos, Rtest, dlogL_sc, dlogL_K0, Lratio` are dictionaries
keyed by the horizon.

### Analysis in `summary.json`

All quantities use the dynamic stage at `T=2`, with point medians over seeds.

| Field | Definition |
|---|---|
| `fit` | Least-squares fit `log eps = c0 + p log w + b log n` over points with `1/8 <= w <= 1` and `d >= 1024`. `b_ci95` comes from 400 bootstrap resamples of the seeds within each point (seed 0). `b_alt_all_depths` repeats the fit with `d >= 256`. |
| `resid_w`, `resid_Lambda` | RMS residual of a quadratic fit of `log eps` against `log w` or `log Lambda`, over all points with `w <= 2`. |
| `outcome` | `W_CONTROLLED` if `b_ci95` lies inside `(-0.15, 0.25)` and `resid_w <= resid_Lambda`; `N_DEPENDENT` if its lower end exceeds `0.25`; otherwise `INCONCLUSIVE`. |
| `c_lin`, `w_lin` | Largest `w_lin` in `{1/4, 1/2, 1}` for which every `eps/w` with `1/8 <= w <= w_lin` and `d >= 1024` lies within 35% of its median `c_lin`. |
| `w_mem`, `w_share` | Powers of two bounding the region where every `Rtest >= 0.95` (below `w_mem`) and where every `Rtest <= 0.5` (above `w_share`), for `d >= 1024`. |
| `n_factor`, `Lambda_factor` | Along the fixed-`w` line with the longest range of `n` (`w <= 1`, powers of two), the ratio of its largest to smallest `n` and the corresponding ratio of `Lambda`. |
| `exploratory.fit` | The same fit restricted to `n >= 16`. |
| `exploratory.iso_w_spreads` | For each fixed `w <= 2` with points at two or more depths and `n >= 16`, the point medians of `eps`, `Rtest` and `kappa0` and their spreads: `(max-min)/mean`, population `std/mean`, and `max(abs(x-mean))/mean`. |
| `fig2` | Scalar/QSP and frozen/QSP hitting-time ratios of the sparse cell, with quartiles, ranges and censoring counts. |

## Plot arrays

`figure_data.npz` holds every array drawn in Figs. 1 and 2; all medians,
quartiles, box summaries and data-dependent reference curves are computed by
the generator. The displayed run of each cell is the one whose finite QSP
hitting time is closest to the median; if every run is censored, run 0 is shown
and `representative_has_finite_hit` is false. The ratio insets use, for each
reference separately, only runs where both hitting times are finite and
positive: 30/30 pairs in the sparse cell and 28/30 in the dense cell. The
interquartile loss bands (`q25`, `q75`) and median ratios (`ratio_*_median`)
are saved for reference but not drawn.

The dotted references in the inset of Fig. 2(c) are
`median(n0)*sqrt(n/n0)*(1+ln n)/(1+ln n0)`, with natural logarithms, each
anchored at the smallest `n0` of its fixed-`w` sequence and drawn only over
that sequence: `n` from 4 to 32, 8 to 64 and 16 to 128 for `w=1/4, 1/2, 1`.

## Randomness and memory

Each scan point's seed is determined by the base seed, the stage tag, `d` and
`n`. A point first draws the student phases of all seeds and then one
independent degree-8 teacher per seed, so the teacher stream depends on the
configured seed count: reducing a 64-seed point to eight seeds does not
reproduce its first eight student/teacher pairs.

`max_elems` limits the number of entries in each Jacobian block; it is not a
byte count or a hard limit on memory. `max_kernel_elems` sets the two static
exclusions above.

## Reading the results

- Medians and interquartile ranges summarize variation across seeds. For a
  trajectory maximum, the time maximum is taken for each seed before the
  median. A bound on the medians does not hold for every run.
- At fixed occupancy `d=(n/w)^2`, so the fitted `n` exponent mixes the grid
  size with finite-depth effects. The near-zero exponent of the `n >= 16` fit
  is exploratory and does not isolate a finite-depth correction.
- `W_CONTROLLED` describes the finite scan window. It does not assert an
  asymptotic result or `b=0`.
- `Rtest` compares target-error norms; a value near one does not mean the
  outputs at the test points stayed fixed.
- The gray `Lower bound`, `exp[-2(d+1)t]`, bounds the normalized loss of the
  continuous gradient flow. It is shown as a reference for the discrete
  iterates, not as a rigorous bound on them.
- Frozen-kernel crossing times are computed analytically from the initial
  spectrum and can lie beyond the QSP step cap; in the dense cell the
  frozen/QSP ratio reaches about 280.
- The scan uses even `d` only, teacher targets of degree 8 and one zero-target
  check. It does not test other targets, nonuniform grids or longer times.

## Reproducibility

`python generate_data.py` regenerates every data family from the seeds without
reading the supplied files. The supplied data come from the production runs with
the settings in `data/protocol.json`. Regenerating the sparse cell reproduces
all 30 QSP hitting times to a relative difference of `1e-12` or better. For
the dense cell, whose full run takes several hours, the first 20 steps are
reproduced exactly and the frozen-kernel hitting times to a relative
difference of `1e-10`. Regenerated scan points and initialization samples
agree with the supplied values to within `1e-11`. Differences at the level of
floating-point rounding can arise across numerical-library versions.
