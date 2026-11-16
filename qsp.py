"""QSP physics and numerical routines used by the data generator.

Use M=d+1, loss L=||g-f||^2/(2*n), and tau=M*t/(3*n).
The rescaled phase flow is phi'=-(3/M) J.T @ (g-f).
The occupancy scan uses RK4 and records every step endpoint for its suprema.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


TWO_PI = 2.0 * np.pi
TEACHER_DEGREE = 8


def beta_32_12_density(value: np.ndarray) -> np.ndarray:
    """Density of Beta(3/2, 1/2), with support handling."""

    x = np.asarray(value, dtype=float)
    out = np.zeros_like(x)
    inside = (x > 0.0) & (x < 1.0)
    out[inside] = 2.0 / np.pi * np.sqrt(x[inside] / (1.0 - x[inside]))
    return out


def mean_diagonal(x: np.ndarray | list[float], degree: int) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return (degree + 1.0) / 4.0 * (
        1.0 + (2.0 * values * values - 1.0) ** degree
    )


def diagonal_second_moment(
    x: np.ndarray | list[float], degree: int
) -> np.ndarray:
    """Exact finite-depth raw second moment E[K_d(x,x)^2]."""

    values = np.asarray(x, dtype=float)
    result = np.empty_like(values)
    at_zero = np.isclose(values, 0.0, atol=1.0e-12, rtol=0.0)
    at_one = np.isclose(np.abs(values), 1.0, atol=1.0e-12, rtol=0.0)
    interior = ~(at_zero | at_one)

    result[at_zero] = (
        3.0 / 16.0 * (1.0 + (-1.0) ** degree) * (degree + 1.0) ** 2
    )
    result[at_one] = 3.0 / 8.0 * (degree + 1.0) ** 2
    if np.any(interior):
        z = values[interior]
        delta_1 = (2.0 * z * z - 1.0) ** degree
        delta_2 = (6.0 * z**4 - 6.0 * z * z + 1.0) ** degree
        d = float(degree)
        result[interior] = (
            3.0 * d**2 * (27.0 * delta_1 + delta_2 + 10.0)
            + 6.0 * d * (27.0 * delta_1 + 5.0 * delta_2 + 6.0)
            + (8.0 * d + 16.0)
            * (delta_2 - 1.0)
            / (z * z * (z * z - 1.0))
            + 81.0 * delta_1
            + 75.0 * delta_2
            + 6.0
        ) / 432.0
    return result


def diagonal_coefficient_of_variation(
    x: np.ndarray | list[float], degree: int
) -> np.ndarray:
    mean = mean_diagonal(x, degree)
    variance = diagonal_second_moment(x, degree) - mean**2
    return np.sqrt(np.maximum(0.0, variance)) / mean


def _phase_matrix_batch(phases: np.ndarray) -> np.ndarray:
    result = np.zeros((phases.size, 2, 2), dtype=np.complex128)
    result[:, 0, 0] = np.exp(1j * phases)
    result[:, 1, 1] = np.exp(-1j * phases)
    return result


def _frame_axis_batch(unitaries: np.ndarray) -> np.ndarray:
    uz = unitaries.copy()
    uz[:, :, 1] *= -1.0
    axes = uz @ np.swapaxes(unitaries.conj(), 1, 2)
    return np.stack(
        [axes[:, 0, 1].real, -axes[:, 0, 1].imag, axes[:, 0, 0].real],
        axis=1,
    )


def sample_diagonal_kernel(
    *, x: float, degree: int, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample g and K(x,x) using the exact frame identity K=c^T S c."""

    phases = rng.uniform(0.0, 2.0 * np.pi, size=(samples, degree + 1))
    unitary = _phase_matrix_batch(phases[:, 0])
    frame_sum = np.zeros((samples, 3, 3), dtype=float)
    axis = _frame_axis_batch(unitary)
    frame_sum += axis[:, :, None] * axis[:, None, :]

    signal = np.array(
        [[x, 1j * np.sqrt(1.0 - x * x)],
         [1j * np.sqrt(1.0 - x * x), x]],
        dtype=np.complex128,
    )
    for index in range(1, degree + 1):
        phase = _phase_matrix_batch(phases[:, index])
        unitary = unitary @ (signal[None, :, :] @ phase)
        axis = _frame_axis_batch(unitary)
        frame_sum += axis[:, :, None] * axis[:, None, :]

    output = unitary[:, 0, 0].real
    c_vector = np.stack(
        [unitary[:, 0, 1].imag,
         unitary[:, 0, 1].real,
         unitary[:, 0, 0].imag],
        axis=1,
    )
    kernel = np.einsum("ri,rij,rj->r", c_vector, frame_sum, c_vector)
    return output, kernel


def midpoint_inputs(samples: int) -> np.ndarray:
    theta = (np.arange(samples) + 0.5) * (np.pi / 2.0) / samples
    return np.cos(theta)


def qsp_outputs(phases: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Evaluate batched QSP responses for phases (B,M) and inputs (n,)."""

    phases = np.asarray(phases, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    batch, phase_count = phases.shape
    sample_count = inputs.size

    row = np.zeros((batch, sample_count, 2), dtype=np.complex128)
    row[:, :, 0] = np.exp(1j * phases[:, 0])[:, None]
    sine = np.sqrt(np.clip(1.0 - inputs * inputs, 0.0, None))
    signal = np.zeros((sample_count, 2, 2), dtype=np.complex128)
    signal[:, 0, 0] = inputs
    signal[:, 1, 1] = inputs
    signal[:, 0, 1] = 1j * sine
    signal[:, 1, 0] = 1j * sine

    for index in range(1, phase_count):
        phase = np.exp(1j * phases[:, index])
        block = np.empty((batch, sample_count, 2, 2), dtype=np.complex128)
        block[:, :, :, 0] = signal[None, :, :, 0] * phase[:, None, None]
        block[:, :, :, 1] = signal[None, :, :, 1] * np.conj(phase)[:, None, None]
        row = np.einsum("bni,bnij->bnj", row, block)
    return row[:, :, 0].real


def qsp_outputs_and_jacobian(
    phases: np.ndarray, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate QSP outputs and all phase derivatives."""

    phases = np.asarray(phases, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    batch, phase_count = phases.shape
    sample_count = inputs.size
    sine = np.sqrt(np.clip(1.0 - inputs * inputs, 0.0, None))
    signal = np.array(
        [[[x, 1j * s], [1j * s, x]] for x, s in zip(inputs, sine)],
        dtype=np.complex128,
    )

    blocks = []
    for index in range(phase_count):
        phase = np.exp(1j * phases[:, index])
        block = np.zeros(
            (batch, sample_count, 2, 2), dtype=np.complex128
        )
        if index == 0:
            block[:, :, 0, 0] = phase[:, None]
            block[:, :, 1, 1] = np.conj(phase)[:, None]
        else:
            block[:, :, :, 0] = signal[None, :, :, 0] * phase[:, None, None]
            block[:, :, :, 1] = signal[None, :, :, 1] * np.conj(phase)[:, None, None]
        blocks.append(block)

    prefixes = np.empty(
        (batch, sample_count, phase_count, 2), dtype=np.complex128
    )
    row = None
    for index, block in enumerate(blocks):
        row = block[:, :, 0, :] if row is None else np.einsum(
            "bni,bnij->bnj", row, block
        )
        prefixes[:, :, index, :] = row

    suffix = np.zeros(
        (batch, sample_count, phase_count + 1, 2), dtype=np.complex128
    )
    suffix[:, :, phase_count, 0] = 1.0
    column = suffix[:, :, phase_count, :]
    for index in range(phase_count - 1, 0, -1):
        column = np.einsum("bnij,bnj->bni", blocks[index], column)
        suffix[:, :, index, :] = column

    output = prefixes[:, :, -1, 0].real
    jacobian = (
        1j
        * (
            prefixes[..., 0] * suffix[:, :, 1:, 0]
            - prefixes[..., 1] * suffix[:, :, 1:, 1]
        )
    ).real
    return output, jacobian


def scalar_loss_ratios(
    degree: int,
    samples: int,
    outputs_initial: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Integrate all realization-matched scalar flows in physical time."""

    shape = outputs_initial.shape
    scale = (degree + 1.0) / (3.0 * samples)

    def right_hand_side(_time: float, flat: np.ndarray) -> np.ndarray:
        outputs = flat.reshape(shape)
        return (-scale * (1.0 - outputs**2) * (outputs - targets)).ravel()

    solution = solve_ivp(
        right_hand_side,
        (0.0, float(times[-1])),
        outputs_initial.ravel(),
        t_eval=times,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    outputs = solution.y.reshape(*shape, times.size)
    loss = np.sum((outputs - targets[..., None]) ** 2, axis=1)
    return loss / loss[:, :1]


def first_crossing(
    times: np.ndarray, ratios: np.ndarray, threshold: float
) -> np.ndarray:
    """First crossing with log-linear interpolation of the loss ratio."""

    hits = np.full(ratios.shape[0], np.nan)
    for seed, row in enumerate(ratios):
        crossed = np.flatnonzero(row <= threshold)
        if crossed.size == 0:
            continue
        high = int(crossed[0])
        if high == 0:
            hits[seed] = times[0]
            continue
        low = high - 1
        y_low, y_high = np.log(row[low]), np.log(row[high])
        weight = (np.log(threshold) - y_low) / (y_high - y_low)
        hits[seed] = times[low] + weight * (times[high] - times[low])
    return hits


def nodes(n):
    """Training grid theta_a=(2a+1)pi/(4n), x_a=cos(theta_a); arrays (n,)."""
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    th = (2 * np.arange(n) + 1) * np.pi / (4 * n)
    return th, np.cos(th)


def midpoints(n):
    """Test grid between training nodes, theta=(a+1)pi/(2n); arrays (n-1,)."""
    th = np.arange(1, n) * np.pi / (2 * n)
    return th, np.cos(th)


def sparse_lambda(n, M):
    """Sparse-data control parameter n**1.5*(1+log n)/sqrt(M)."""
    return n ** 1.5 * (1.0 + np.log(n)) / np.sqrt(M)


def _chunks(S, n, M, max_elems):
    if max_elems < M:
        raise ValueError(f"max_elems={max_elems} cannot hold one row of M={M}")
    width = min(n, max_elems // M)
    if width == 0:
        return
    seeds = max(1, max_elems // (width * M))
    for s in range(0, S, seeds):
        for a in range(0, n, width):
            yield slice(s, min(S, s + seeds)), slice(a, min(n, a + width))


def eval_outputs(x, phi, max_elems=2_000_000):
    """QSP outputs: x(n,), phi(S,M) -> g(S,n), evaluated in rectangular chunks."""
    S, M = phi.shape
    n = len(x)
    g = np.empty((S, n))
    for ss, aa in _chunks(S, n, M, max_elems):
        g[ss, aa] = qsp_outputs(phi[ss], x[aa])
    return g


def flow_rhs(x, f, phi, max_elems=2_000_000):
    """Rescaled-time phase flow phi'=-(3/M)J^T(g-f).

    x(n,), targets f(S,n), phases phi(S,M) -> g(S,n), diagK(S,n), dphi(S,M).
    Each evaluator call has at most max_elems entries in its Jacobian.
    """
    S, M = phi.shape
    n = len(x)
    g = np.empty((S, n))
    dK = np.empty((S, n))
    acc = np.zeros((S, M))
    for ss, aa in _chunks(S, n, M, max_elems):
        gg, J = qsp_outputs_and_jacobian(phi[ss], x[aa])
        g[ss, aa] = gg
        dK[ss, aa] = np.einsum("saj,saj->sa", J, J)
        e = gg - f[ss, aa]
        for local, seed in enumerate(range(ss.start, ss.stop)):
            acc[seed] += e[local] @ J[local]
    return g, dK, -(3.0 / M) * acc


def kernel_matrix(x, phi_row, max_elems=2_000_000):
    """Realized kernel: x(n,), phi_row(M,) -> g(n,), K(n,n)."""
    n, M = len(x), len(phi_row)
    g = np.empty(n)
    J = np.empty((n, M))
    for ss, aa in _chunks(1, n, M, max_elems):
        gg, JJ = qsp_outputs_and_jacobian(phi_row[None], x[aa])
        g[aa], J[aa] = gg[0], JJ[0]
    return g, J @ J.T


def teacher_targets(rng, S, x, degree=TEACHER_DEGREE, max_elems=2_000_000):
    """S independent uniform-phase QSP teachers of the given degree."""
    tph = rng.uniform(0.0, TWO_PI, (S, degree + 1))
    return tph, eval_outputs(x, tph, max_elems)


def scalar_rhs(u, f):
    """Right-hand side of the scalar flow, -(1-u**2)*(u-f)."""
    return -(1.0 - u * u) * (u - f)


def rms(v, axis=-1):
    """Root mean square along the input axis."""
    return np.sqrt(np.mean(v * v, axis=axis))


def run_point(d, n, seeds, rng, T_list=(1.0, 2.0, 3.0), dt=0.05,
              targets="teacher", max_elems=2_000_000, dynamic=True,
              teacher_degree=TEACHER_DEGREE):
    """Run one seeded ensemble of the occupancy scan.

    Returns per-seed lists:
      kappa0         RMS(cross forcing) / RMS(diagonal forcing) at tau = 0
      dclos0         RMS_a |3 K_aa/M - (1 - g_a^2)| at tau = 0
      eps[T]         sup_{tau<=T} RMS_a |g_a - gbar_a| (scalar closure)
      dclos[T]       sup_{tau<=T} RMS_a |3 K_aa/M - (1-g_a^2)|
      Rtest[T]       RMS_mid(g - f)(T) / RMS_mid(g - f)(0), between nodes
      dlogL_sc[T]    sup_{tau<=T} |ln L_sc - ln L_QSP|
      dlogL_K0[T]    sup_{tau<=T} |ln L_K0 - ln L_QSP|
      Lratio[T]      L_QSP(T) / L_QSP(0)
    """
    if any(int(v) != v or v < 1 for v in (d, n, seeds)):
        raise ValueError("d, n and seeds must be positive integers")
    T_list = sorted(float(T) for T in T_list)
    if not T_list or len(set(T_list)) != len(
            T_list) or not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "T_list must be nonempty and unique; "
            "dt must be positive and finite")
    if any(not np.isfinite(T) or T <= 0 or not np.isclose(
            T / dt, round(T / dt), atol=1e-9, rtol=0) for T in T_list):
        raise ValueError(
            "Every T must be positive, finite and an integer multiple of dt")
    M = d + 1
    _, x = nodes(n)
    _, xm = midpoints(n)
    phi = rng.uniform(0.0, TWO_PI, (seeds, M))
    if targets == "teacher":
        tph, f = teacher_targets(
            rng, seeds, x, degree=teacher_degree, max_elems=max_elems)
        fm = eval_outputs(
            xm, tph, max_elems) if n > 1 else np.zeros((seeds, 0))
    elif targets == "zero":
        f = np.zeros((seeds, n))
        fm = np.zeros((seeds, max(n - 1, 0)))
    else:
        raise ValueError(targets)

    # initialization: realized kernel per seed
    kappa0 = np.empty(seeds)
    dclos0 = np.empty(seeds)
    lam = []
    wts = []
    L0 = np.empty(seeds)
    for s in range(seeds):
        g0, K0 = kernel_matrix(x, phi[s], max_elems)
        e0 = g0 - f[s]
        dg = np.diag(K0) * e0
        off = K0 @ e0 - dg
        kappa0[s] = np.linalg.norm(off) / max(np.linalg.norm(dg), 1e-300)
        dclos0[s] = rms(3.0 * np.diag(K0) / M - (1.0 - g0 ** 2))
        ev, V = np.linalg.eigh(K0)
        lam.append(np.clip(ev, 0.0, None))
        wts.append((V.T @ e0) ** 2)
        L0[s] = 0.5 * np.mean(e0 ** 2)
    res = {
        "d": d, "n": n, "M": M, "w": n / np.sqrt(d), "seeds": seeds,
        "targets": targets, "Lambda": float(sparse_lambda(n, M)),
        "kappa0": kappa0.tolist(), "dclos0": dclos0.tolist(),
        "teacher_degree": teacher_degree if targets == "teacher" else None,
        "sup_sampling": (
            "Every RK4 step endpoint including tau=0 and each T; "
            "not a certified continuous-time supremum"),
        "time_convention": "tau=(d+1)*t/(3*n)",
        "seed_indices": list(range(seeds))}
    if not dynamic:
        return res

    gm0 = eval_outputs(xm, phi, max_elems) if n > 1 else None
    steps = int(round(T_list[-1] / dt))
    rec_steps = {int(round(T / dt)): T for T in T_list}
    lam = np.array(lam)
    wts = np.array(wts)
    eps = np.zeros(seeds)
    dcl = np.zeros(seeds)
    dsc = np.zeros(seeds)
    dk0 = np.zeros(seeds)
    out = {k: {} for k in (
        "eps", "dclos", "Rtest", "dlogL_sc", "dlogL_K0", "Lratio")}
    gbar = None
    for k in range(steps + 1):
        tau = k * dt
        g, dK, k1 = flow_rhs(x, f, phi, max_elems)
        if not all(np.all(np.isfinite(v)) for v in (g, dK, k1)):
            raise FloatingPointError(f"Nonfinite QSP state at tau={tau:g}")
        if gbar is None:
            gbar = g.copy()
        eps = np.maximum(eps, rms(g - gbar))
        dcl = np.maximum(dcl, rms(3.0 * dK / M - (1.0 - g ** 2)))
        Lq = np.maximum(0.5 * np.mean((g - f) ** 2, 1), 1e-300)
        Ls = np.maximum(0.5 * np.mean((gbar - f) ** 2, 1), 1e-300)
        Lk = np.maximum(
            0.5 * np.sum(wts * np.exp(-2.0 * (3.0 / M) * lam * tau),
                         1) / n, 1e-300)
        dsc = np.maximum(dsc, np.abs(np.log(Ls) - np.log(Lq)))
        dk0 = np.maximum(dk0, np.abs(np.log(Lk) - np.log(Lq)))
        if k in rec_steps:
            T = rec_steps[k]
            key = f"{T:g}"
            out["eps"][key] = eps.tolist()
            out["dclos"][key] = dcl.tolist()
            out["dlogL_sc"][key] = dsc.tolist()
            out["dlogL_K0"][key] = dk0.tolist()
            out["Lratio"][key] = (Lq / L0).tolist()
            if n > 1:
                gm = eval_outputs(xm, phi, max_elems)
                out["Rtest"][key] = (
                    rms(gm - fm) / np.maximum(rms(gm0 - fm), 1e-300)).tolist()
        if k == steps:
            break
        _, _, k2 = flow_rhs(x, f, phi + 0.5 * dt * k1, max_elems)
        _, _, k3 = flow_rhs(x, f, phi + 0.5 * dt * k2, max_elems)
        _, _, k4 = flow_rhs(x, f, phi + dt * k3, max_elems)
        phi = phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        a1 = scalar_rhs(gbar, f)
        a2 = scalar_rhs(gbar + 0.5 * dt * a1, f)
        a3 = scalar_rhs(gbar + 0.5 * dt * a2, f)
        a4 = scalar_rhs(gbar + dt * a3, f)
        gbar = gbar + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    res.update(out)
    res["dt"] = dt
    res["T_list"] = T_list
    return res


def train_gd(
    phases0, inputs, targets, eta, rho_stop=1.0e-4, max_steps=500_000
):
    """Run batched Euler gradient descent and record every loss in float32.

    A first threshold crossing records a hitting time, but the corresponding
    phases keep evolving until all runs cross or the common step cap is reached.
    Hitting times use float64 losses before storage and adjacent-step log-linear
    interpolation. The loss is sum((output - target)**2) / (2 * n).
    """
    phases = phases0.copy()
    samples = inputs.size
    output, jacobian = qsp_outputs_and_jacobian(phases, inputs)
    residual = output - targets
    loss0 = 0.5 * (residual * residual).sum(-1) / samples
    history = np.empty((phases.shape[0], max_steps + 1), dtype=np.float32)
    history[:, 0] = loss0
    hit = np.full(phases.shape[0], np.nan)
    previous_ratio = np.ones(phases.shape[0])
    active = np.ones(phases.shape[0], dtype=bool)
    step = 0
    while step < max_steps and active.any():
        phases = phases - (eta / samples) * np.einsum(
            "bnp,bn->bp", jacobian, residual
        )
        step += 1
        output, jacobian = qsp_outputs_and_jacobian(phases, inputs)
        residual = output - targets
        loss = 0.5 * (residual * residual).sum(-1) / samples
        history[:, step] = loss
        ratio = loss / loss0
        newly = active & (ratio <= rho_stop)
        if newly.any():
            low = np.log(np.maximum(previous_ratio[newly], 1.0e-300))
            high = np.log(np.maximum(ratio[newly], 1.0e-300))
            fraction = (low - np.log(rho_stop)) / np.maximum(
                low - high, 1.0e-300
            )
            hit[newly] = step - 1 + np.clip(fraction, 0.0, 1.0)
            active[newly] = False
        previous_ratio = ratio
    return {
        "loss0": loss0,
        "loss_hist": history[:, : step + 1],
        "hit_step": hit,
        "steps_run": step,
    }


def frozen_hitting_time(kernel0, residual0, samples, rho=1.0e-4, t_max=1.0e14):
    """Return the realized frozen-kernel crossing in physical time."""
    from scipy.optimize import brentq

    eigenvalues, eigenvectors = np.linalg.eigh(kernel0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    weights = (eigenvectors.T @ residual0) ** 2
    weight_sum = weights.sum()

    def ratio(time):
        return float(
            (weights * np.exp(-2.0 * eigenvalues * time / samples)).sum()
            / weight_sum
        )

    if ratio(t_max) > rho:
        return np.nan
    return brentq(
        lambda time: ratio(time) - rho,
        0.0,
        t_max,
        xtol=1.0e-14,
        rtol=1.0e-13,
    )


def dynamics_cell(
    degree,
    samples,
    seeds=30,
    cap=500_000,
    base_seed=20260718,
    teacher_degree=TEACHER_DEGREE,
    eta_scale=0.1,
    rho=1.0e-4,
    checkpoints=256,
):
    """Generate one Fig. 2 dynamics cell with its scalar and frozen references.

    All t_* fields are measured in GD steps; physical time equals eta times
    the stored value.
    """
    if any(int(value) != value or value < 1
           for value in (degree, samples, seeds, cap, teacher_degree, checkpoints)):
        raise ValueError("Degrees, samples, seeds, cap and checkpoints must be positive integers")
    if not np.isfinite(eta_scale) or eta_scale <= 0 or not 0 < rho < 1:
        raise ValueError("eta_scale must be positive and finite; rho must lie in (0, 1)")
    inputs = midpoint_inputs(samples)
    eta = eta_scale / (degree + 1)
    phases0 = np.stack(
        [
            np.random.default_rng([base_seed, 1, degree, samples, run]).uniform(
                0.0, 2.0 * np.pi, degree + 1
            )
            for run in range(seeds)
        ]
    )
    teacher_phases = np.stack(
        [
            np.random.default_rng([base_seed, 11, degree, samples, run]).uniform(
                0.0, 2.0 * np.pi, teacher_degree + 1
            )
            for run in range(seeds)
        ]
    )
    # Teachers are evaluated one run at a time; this fixes the floating-point
    # rounding of the stored targets.
    targets = np.stack(
        [qsp_outputs(phases[None, :], inputs)[0] for phases in teacher_phases]
    )
    result = train_gd(phases0, inputs, targets, eta, rho, cap)
    indices = np.concatenate(
        [
            [0],
            np.unique(
                np.geomspace(1, result["steps_run"], checkpoints).astype(int)
            ),
        ]
    )
    initial_output, initial_jacobian = qsp_outputs_and_jacobian(phases0, inputs)
    frozen_steps = np.full(seeds, np.nan)
    arrays = {
        "inputs": inputs,
        "initial_phases": phases0,
        "teacher_phases": teacher_phases,
        "initial_outputs": qsp_outputs(phases0, inputs),
        "targets": targets,
        "loss_idx": indices,
        "loss_dec": result["loss_hist"][:, indices],
        "t_full": result["hit_step"],
        "t_frozen": frozen_steps,
    }
    for run in range(seeds):
        jacobian = initial_jacobian[run][None, :, :]
        kernel0 = np.einsum("bnp,bmp->bnm", jacobian, jacobian)[0]
        residual0 = initial_output[run] - targets[run]
        frozen_steps[run] = frozen_hitting_time(
            kernel0, residual0, samples, rho
        ) / eta
        eigenvalues, eigenvectors = np.linalg.eigh(kernel0)
        arrays[f"lam_{run}"] = np.clip(eigenvalues, 0.0, None)
        arrays[f"w_{run}"] = (eigenvectors.T @ residual0) ** 2

    # Initial outputs and targets of the scalar reference, evaluated in batch.
    scalar_initial = qsp_outputs(phases0, inputs)
    scalar_targets = qsp_outputs(teacher_phases, inputs)
    time = indices * eta
    scalar = scalar_loss_ratios(
        degree, samples, scalar_initial, scalar_targets, time
    )
    scalar_steps = first_crossing(time, scalar, rho) / eta
    arrays["scalar_loss_ratio"] = scalar
    arrays["t_scalar"] = scalar_steps
    metadata = {
        "cell": {
            "d": degree,
            "n": samples,
            "w": samples / np.sqrt(degree),
            "eta": eta,
            "cap": cap,
            "n_seeds": seeds,
            "teacher_degree": teacher_degree,
            "loss_threshold": rho,
            "checkpoints": checkpoints,
            "seeds": {
                "init": f"[{base_seed},1,{degree},{samples},k]",
                "teacher": f"[{base_seed},11,{degree},{samples},k]",
            },
        },
        "steps_run": int(result["steps_run"]),
        "censored_full": (~np.isfinite(result["hit_step"])).tolist(),
        "censored_scalar": (~np.isfinite(scalar_steps)).tolist(),
        "time_unit": "GD steps; physical time = eta * steps",
        "training": {
            "integrator": "Euler gradient descent",
            "loss": "sum((output - target)**2) / (2 * n)",
            "step_size": "eta_scale / (degree + 1)",
            "eta_scale": eta_scale,
            "hitting_time": "first crossing; adjacent-step log-linear interpolation",
            "stopping": "all runs cross or common cap; crossed runs keep evolving",
            "loss_storage": "float32; hitting-time calculation uses float64",
        },
        "scalar": {
            "integrator": "scipy.integrate.solve_ivp (RK45)",
            "rtol": 1.0e-9,
            "atol": 1.0e-11,
            "hitting_time": "first crossing; log-linear interpolation on loss_idx * eta",
        },
    }
    return arrays, metadata
