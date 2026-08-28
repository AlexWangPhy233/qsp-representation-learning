# Figure 2 data schema

Each cell has one JSON metadata file and one NumPy NPZ archive.

## Metadata consumed by the plot

- `cell.d`, `cell.n`, `cell.w`: QSP degree, number of training points, and density variable.
- `cell.eta`: discrete gradient-descent step size.
- `cell.n_seeds`: number of independent student--teacher pairs.
- `cell.seeds`: deterministic seed construction used to reconstruct the matched initial outputs and targets.

The `cell.fig` value is the identifier used by the upstream numerical campaign; it is not the manuscript figure number.

## Archive fields consumed by the plot

- `loss_idx`: stored iteration indices.
- `loss_dec`: QSP loss trajectories for all runs.
- `t_full`: QSP threshold-crossing iteration for each run.
- `t_frozen`: frozen-kernel threshold-crossing iteration for each run.
- `lam_<k>` and `w_<k>`: eigenvalues and residual weights for run `k`, used for the representative frozen-kernel curve.

The plotting script selects the representative run by the median finite QSP hitting time, reconstructs the matched scalar flow from the frozen seed protocol, and retains censored trajectories in the trajectory statistics.
