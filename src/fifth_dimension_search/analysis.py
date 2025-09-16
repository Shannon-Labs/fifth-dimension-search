"""Helper utilities for analysing the packaged toy datasets.

None of the numbers produced here correspond to validated physics. The module
exists purely to make it easier for contributors to reproduce current results,
understand where the model is broken, and build improved analyses on top of a
clean interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from .datasets import data_path


@dataclass(frozen=True)
class PhaseDeviation:
    """Observed deviation relative to the GR baseline for a configuration."""

    label: str
    delta_phase: float
    percent_change: float


def load_phase_summary() -> pd.DataFrame:
    """Return the phase/amplitude summary table packaged with the project."""

    return pd.read_csv(data_path("phase_amp_summary.csv"))


def load_kk_metrics() -> pd.DataFrame:
    """Return the KK parameter sweep metrics bundled with the project."""

    return pd.read_csv(data_path("kk_sweep_metrics.csv"))


def compute_phase_deviations(table: pd.DataFrame) -> list[PhaseDeviation]:
    """Compute relative phase deviations with respect to the GR baseline."""

    if "label" not in table or "phase_span" not in table:
        raise ValueError("Expected columns 'label' and 'phase_span' to be present")

    try:
        baseline = float(table.loc[table["label"] == "GR", "phase_span"].iloc[0])
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Phase summary table must contain a 'GR' baseline row") from exc

    deviations: list[PhaseDeviation] = []
    for _, row in table.iterrows():
        label = str(row["label"])
        if label == "GR":
            continue
        phase_span = float(row["phase_span"])
        delta = phase_span - baseline
        percent = 0.0 if baseline == 0 else 100.0 * delta / baseline
        deviations.append(PhaseDeviation(label=label, delta_phase=delta, percent_change=percent))
    return deviations


def ligo_phase_precision(snr: float) -> float:
    """Very rough phase uncertainty estimate for a compact binary signal.

    This uses the scaling sigma_phi ≈ 0.1 rad × (30 / SNR) following public LIGO
    documentation. It deliberately errs on the optimistic side to highlight how
    far the current toy model is from detectability.
    """

    snr = max(snr, 1.0)
    return 0.1 * (30.0 / snr)


def detectability_assessment(
    metrics: pd.DataFrame,
    snr_grid: Iterable[float] = (10.0, 20.0, 30.0, 60.0),
) -> Dict[str, Mapping[str, float]]:
    """Return a dictionary summarising detectability across an SNR grid."""

    if "phase_diff_max" not in metrics:
        raise ValueError("KK metrics table missing 'phase_diff_max' column")

    result: Dict[str, Mapping[str, float]] = {}
    for snr in snr_grid:
        sigma = ligo_phase_precision(snr)
        ratio = float(metrics["phase_diff_max"].max()) / sigma if sigma else np.nan
        result[f"SNR={snr:g}"] = {
            "phase_sigma": sigma,
            "max_phase_diff": float(metrics["phase_diff_max"].max()),
            "ratio_to_noise": ratio,
        }
    return result


def normalise_labels(labels: Iterable[str]) -> list[str]:
    """Return contributor-friendly labels for plotting/summary output."""

    friendly = []
    for label in labels:
        label = str(label).strip()
        if label.upper() == "GR":
            friendly.append("General Relativity")
        else:
            friendly.append(label.replace("_", " ").replace("KK", "Toy KK"))
    return friendly


__all__ = [
    "PhaseDeviation",
    "compute_phase_deviations",
    "detectability_assessment",
    "ligo_phase_precision",
    "load_phase_summary",
    "load_kk_metrics",
    "normalise_labels",
]
