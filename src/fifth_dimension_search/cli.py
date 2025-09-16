"""Command line tools for the fifth-dimension sandbox."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from . import analysis as fds_analysis
from .datasets import data_path, list_datasets
from .visualisation import create_toy_diagnostics_plot

app = typer.Typer(help="Utility commands for the exploratory sandbox.")
datasets_app = typer.Typer(help="Inspect the packaged toy datasets.")
app.add_typer(datasets_app, name="datasets")


@app.command()
def info() -> None:
    """Print a short project status summary using the packaged data."""

    phase_data = fds_analysis.load_phase_summary()
    kk_metrics = fds_analysis.load_kk_metrics()

    typer.echo("Fifth Dimension Sandbox – current toy data status:\n")

    deviations = fds_analysis.compute_phase_deviations(phase_data)
    if deviations:
        typer.echo("Phase span deviations relative to GR:")
        for row in deviations:
            typer.echo(
                f"  · {row.label:18s} Δφ = {row.delta_phase:+.3e} rad "
                f"({row.percent_change:+6.1f}%)"
            )
    else:
        typer.echo("No deviations recorded – check the phase summary table.")

    typer.echo("\nDetectability sanity check:")
    detectability = fds_analysis.detectability_assessment(kk_metrics)
    for label, stats in detectability.items():
        ratio = stats["ratio_to_noise"]
        verdict = "undetectable" if ratio < 1 else "potentially detectable"
        typer.echo(
            f"  · {label:<8s}: σφ≈{stats['phase_sigma']:.2e} rad, "
            f"max Δφ≈{stats['max_phase_diff']:.2e} rad (ratio {ratio:.1e}, {verdict})"
        )

    typer.echo("\nRemember: the bundled data come from a broken toy model governed by\n"
               "incorrect physics. Use them only as regression fixtures or"
               " starting points for improved models.")


@datasets_app.command("list")
def list_available() -> None:
    """List the dataset files that ship with the package."""

    entries = sorted(list_datasets())
    if not entries:
        typer.echo("No datasets bundled with the package yet.")
        raise typer.Exit(code=1)
    for name in entries:
        typer.echo(f"- {name}")


@datasets_app.command("show")
def show_dataset(name: str, limit: int = typer.Option(5, help="Number of rows to display")) -> None:
    """Preview the first few rows of a packaged CSV dataset."""

    path = data_path(name)
    if path.suffix.lower() != ".csv":
        typer.echo(f"Dataset {name} is not a CSV file – unable to preview.")
        raise typer.Exit(code=1)
    frame = pd.read_csv(path)
    typer.echo(frame.head(limit).to_string(index=False))


@app.command()
def plot(
    output: Path = typer.Option(Path("artifacts/toy_diagnostics.png"), help="Output PNG path"),
    dpi: int = typer.Option(160, min=72, max=600, help="Plot resolution in DPI"),
    show: bool = typer.Option(False, help="Open the plot in a window after saving"),
) -> None:
    """Render the standard diagnostic plot for the toy datasets."""

    phase_data = fds_analysis.load_phase_summary()
    kk_metrics = fds_analysis.load_kk_metrics()
    output = create_toy_diagnostics_plot(phase_data, kk_metrics, output, dpi=dpi)
    typer.echo(f"Saved diagnostic plot to {output}")
    if show:
        try:
            import webbrowser

            webbrowser.open(output.resolve().as_uri())
        except Exception as exc:  # pragma: no cover - best effort only
            typer.echo(f"Unable to open browser automatically: {exc}")


def run() -> None:  # pragma: no cover - thin wrapper for console script
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
