"""Minimal QSP and analytic routines consumed by the manuscript figures.

This module deliberately excludes training campaigns, work-package analysis,
and paper-independent utilities.  It contains only computations called by
``make_all_figures.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

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


def reconstruct_initial_data(
    degree: int,
    samples: int,
    seeds: int,
    base_seed: int,
    teacher_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the student and QSP-teacher outputs used in Fig. 2."""

    inputs = midpoint_inputs(samples)
    student_phases = np.stack(
        [
            np.random.default_rng(
                [base_seed, 1, degree, samples, seed]
            ).uniform(0.0, 2.0 * np.pi, degree + 1)
            for seed in range(seeds)
        ]
    )
    teacher_phases = np.stack(
        [
            np.random.default_rng(
                [base_seed, 11, degree, samples, seed]
            ).uniform(0.0, 2.0 * np.pi, teacher_degree + 1)
            for seed in range(seeds)
        ]
    )
    return qsp_outputs(student_phases, inputs), qsp_outputs(teacher_phases, inputs)


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
