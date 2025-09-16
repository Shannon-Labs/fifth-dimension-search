#!/usr/bin/env python3
"""Simple deterministic convergence experiment for contributors.

The original project claimed to run convergence studies on the full numerical
relativity code, but the required classes were never implemented. This module
provides a lightweight, reproducible example that demonstrates the workflow we
expect new contributions to follow:

1. Define an analytic "truth" solution.
2. Compute discrete approximations at different resolutions.
3. Estimate the observed convergence order using Richardson extrapolation.

The toy problem below evaluates the phase of a chirping complex exponential.
Sampling at finite resolution introduces an error ~O(Δt), therefore the script
should report first-order convergence.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class ConvergenceResult:
    resolution: int
    dt: float
    phase_estimate: float
    phase_error: float


TRUE_FINAL_TIME = 1.0


def analytic_phase(time: np.ndarray) -> np.ndarray:
    """Analytic phase model used for the convergence experiment."""

    # A gently chirping signal: phi(t) = 20 t + 4 t^2 radians
    return 20.0 * time + 4.0 * time ** 2


def sample_phase(resolution: int) -> ConvergenceResult:
    """Return the phase estimate obtained with `resolution` samples."""

    if resolution <= 1:
        raise ValueError("Resolution must be greater than 1 for a convergence test")

    dt = TRUE_FINAL_TIME / resolution
    sample_time = np.linspace(0.0, TRUE_FINAL_TIME - dt, resolution)
    complex_signal = np.exp(1j * analytic_phase(sample_time))
    phase_estimate = float(np.angle(complex_signal[-1]))
    true_phase = float(analytic_phase(np.array([TRUE_FINAL_TIME]))[0])
    phase_error = abs(true_phase - phase_estimate)
    return ConvergenceResult(resolution=resolution, dt=dt, phase_estimate=phase_estimate, phase_error=phase_error)


def run_convergence_study(resolutions: Iterable[int]) -> List[ConvergenceResult]:
    results = [sample_phase(res) for res in resolutions]
    results.sort(key=lambda item: item.resolution)
    return results


def estimate_convergence_order(results: List[ConvergenceResult]) -> float | None:
    """Estimate the observed convergence order using consecutive resolutions."""

    if len(results) < 3:
        return None

    # Assume the grid spacing halves between the last three resolutions
    e1 = results[-3].phase_error
    e2 = results[-2].phase_error
    e3 = results[-1].phase_error

    if e1 == 0 or e2 == 0 or e3 == 0:
        return None

    # Standard Richardson estimate
    return np.log((e1 - e2) / (e2 - e3)) / np.log(2)


def save_results(results: List[ConvergenceResult], order: float | None, path: Path) -> None:
    payload = {
        "results": [asdict(item) for item in results],
        "estimated_order": order,
        "analytic_phase": "phi(t) = 20 t + 4 t^2",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved convergence report to {path}")


def main() -> None:
    resolutions = (32, 64, 128)
    results = run_convergence_study(resolutions)
    order = estimate_convergence_order(results)

    print("Convergence study for toy chirp phase")
    for entry in results:
        print(f"  N={entry.resolution:<4d} phase={entry.phase_estimate:+.5f} rad error={entry.phase_error:.3e}")
    if order is not None:
        print(f"Observed convergence order ≈ {order:.2f}")
    else:
        print("Unable to estimate convergence order – need at least three non-zero errors")

    save_results(results, order, Path("artifacts/convergence_results.json"))


if __name__ == "__main__":
    main()
