"""Convenience plotting functions used by the toy analysis scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as fds_analysis


def create_toy_diagnostics_plot(
    phase_data: pd.DataFrame,
    kk_metrics: pd.DataFrame,
    output: Path,
    *,
    dpi: int = 150,
) -> Path:
    """Render the standard two-panel diagnostic plot for the toy datasets."""

    output.parent.mkdir(parents=True, exist_ok=True)

    deviations = fds_analysis.compute_phase_deviations(phase_data)
    labels = [row.label for row in deviations]
    values = [row.percent_change for row in deviations]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if labels:
        colors = plt.cm.cividis(np.linspace(0.3, 0.9, len(labels)))
        bars = ax1.bar(range(len(labels)), values, color=colors)
        ax1.bar_label(bars, fmt="{:.1f}%", padding=3)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(fds_analysis.normalise_labels(labels), rotation=30, ha="right")
    else:
        ax1.text(0.5, 0.5, "No non-GR entries", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_ylabel("Phase deviation from GR (%)")
    ax1.set_title("Toy waveform phase differences")
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    scatter = ax2.scatter(
        kk_metrics["L5"].values,
        kk_metrics["phase_diff_max"].values,
        c=kk_metrics["q"].values,
        cmap="plasma",
        s=70,
        linewidth=0.8,
        edgecolors="black",
    )
    ax2.set_xlabel("Compactification radius L₅ (toy units)")
    ax2.set_ylabel("Max phase difference (rad)")
    ax2.set_yscale("log")
    ax2.axhline(
        y=fds_analysis.ligo_phase_precision(30.0),
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Optimistic LIGO σφ",
    )
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label="Toy KK charge q")

    plt.suptitle("Bundled toy results – use only as regression data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output


__all__ = ["create_toy_diagnostics_plot"]
