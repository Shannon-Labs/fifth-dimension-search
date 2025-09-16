#!/usr/bin/env python3
"""Generate honest quick-look summaries for the packaged toy datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from fifth_dimension_search import analysis as fds_analysis
from fifth_dimension_search.visualisation import create_toy_diagnostics_plot


def load_and_analyze_results():
    """Load the packaged CSV tables and print a transparent summary."""

    phase_data = fds_analysis.load_phase_summary()
    kk_metrics = fds_analysis.load_kk_metrics()

    print("=" * 68)
    print("TOY EXTRA-DIMENSION ANALYSIS (BROKEN MODEL - FOR COLLABORATION ONLY)")
    print("=" * 68)
    print("\nThe following numbers come from the bundled toy simulation results.")
    print("They are not physically meaningful, but they provide common ground for")
    print("debugging and validating future fixes.\n")

    deviations = fds_analysis.compute_phase_deviations(phase_data)
    if deviations:
        print("Phase span deviations relative to the GR baseline:")
        for entry in deviations:
            print(f"  - {entry.label:20s} Δφ = {entry.delta_phase:+.3e} rad ({entry.percent_change:+6.1f}%)")
    else:
        print("No non-GR configurations found in the phase summary table.")

    print("\nApproximate detectability assuming optimistic LIGO/Virgo phase errors:")
    detectability = fds_analysis.detectability_assessment(kk_metrics)
    for label, stats in detectability.items():
        ratio = stats["ratio_to_noise"]
        verdict = "undetectable" if ratio < 1 else "potentially detectable"
        print(
            f"  - {label:<8s}: σφ≈{stats['phase_sigma']:.2e} rad, "
            f"max Δφ≈{stats['max_phase_diff']:.2e} rad (ratio {ratio:.1e}, {verdict})"
        )

    return phase_data, kk_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/toy_diagnostics.png"),
        help="Where to write the generated plot",
    )
    args = parser.parse_args()

    phase_data, kk_metrics = load_and_analyze_results()
    path = create_toy_diagnostics_plot(phase_data, kk_metrics, args.output)
    print(f"Saved diagnostic plot to {path}")


if __name__ == "__main__":
    main()
