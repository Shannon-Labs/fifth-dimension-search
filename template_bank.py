"""Utilities to pre-generate BSSN+KK waveform templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .bssn_kk_evolver import (
    BSSNParameters,
    bssn_rhs,
    enforce_periodic_state,
    extract_waveform,
    integrate_psi4_series,
    scale_waveform_to_observer,
    solve_initial_conditions,
    state_linear_combination,
)


def evolve_to_waveform(
    params: BSSNParameters,
    shape: Sequence[int],
    spacing: Sequence[float],
    steps: int,
    dt: float,
    radius: float,
    device: torch.device | str = "mps",
) -> dict:
    base_device = torch.device(device) if isinstance(device, str) else device
    if base_device.type not in ("mps", "cuda"):
        base_device = torch.device("cpu")

    if base_device.type == "cuda":
        state = solve_initial_conditions(shape, spacing, params, device=base_device, dtype=torch.float64)
        state = state.to(device=base_device, dtype=torch.float64)
    else:
        state = solve_initial_conditions(shape, spacing, params, device="cpu", dtype=torch.float64)

    psi4_history: List[complex] = []

    for _ in range(steps):
        if base_device.type == "cuda":
            rhs = bssn_rhs(state, spacing, params)
            state = state_linear_combination(state, rhs, dt)
            enforce_periodic_state(state)
            snapshot = state.to("cpu", dtype=torch.float64)
        else:
            work = state.to(device=base_device, dtype=torch.float32 if base_device.type == "mps" else torch.float64)
            rhs = bssn_rhs(work, spacing, params).to(device="cpu", dtype=torch.float64)
            state = state_linear_combination(state, rhs, dt)
            enforce_periodic_state(state)
            snapshot = state

        waveform = extract_waveform(snapshot, spacing, radius=radius, projection="plus", params=params)
        psi4_history.append(complex(waveform["psi4"].item()))

    time_geom, h_plus_geom, h_cross_geom = integrate_psi4_series(psi4_history, dt, target_rate=None)
    return {
        "time_geom": time_geom,
        "h_plus_geom": h_plus_geom,
        "h_cross_geom": h_cross_geom,
        "psi4": psi4_history,
    }


def generate_template_bank(
    q_values: Iterable[float],
    m5_values: Iterable[float],
    L5_values: Iterable[float],
    shape: Sequence[int],
    spacing: Sequence[float],
    steps: int,
    dt: float,
    radius: float,
    output: Path,
    device: torch.device | str = "mps",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    for q in q_values:
        for m5 in m5_values:
            for L5 in L5_values:
                params = BSSNParameters(q=q, m5=m5, L5=L5)
                waveform = evolve_to_waveform(params, shape, spacing, steps, dt, radius, device=device)
                h_plus_geom = waveform["h_plus_geom"]
                peak_geom = float(np.max(np.abs(h_plus_geom))) if h_plus_geom.size else 0.0
                scale_factor = scale_waveform_to_observer(1.0, radius, params)
                peak_strain = peak_geom * scale_factor
                records.append({
                    "q": q,
                    "m5": m5,
                    "L5": L5,
                    "h_plus": peak_strain,
                    "h_plus_geom": peak_geom,
                    "steps": steps,
                    "dt": dt,
                    "shape": list(shape),
                    "spacing": list(spacing),
                })
                with open(output, "w", encoding="utf-8") as fp:
                    json.dump(records, fp, indent=2)


__all__ = [
    "generate_template_bank",
    "evolve_to_waveform",
]
