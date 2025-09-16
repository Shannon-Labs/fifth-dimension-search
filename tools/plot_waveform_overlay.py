#!/usr/bin/env python3
"""Overlay a simulated strain waveform with a whitened LIGO segment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_waveform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "time_strain" in data and "h_plus" in data:
        return data["time_strain"], data["h_plus"]
    raise ValueError(f"Waveform file {path} missing 'time_strain'/'h_plus' entries")


def estimate_alignment(sim_time: np.ndarray, sim_strain: np.ndarray,
                       det_time: np.ndarray, det_strain: np.ndarray) -> Tuple[float, float]:
    """Return (time_offset, amplitude_scale) that best matches whitened strain."""

    if det_time.size < 2 or sim_time.size < 2:
        return 0.0, 1.0

    dt = det_time[1] - det_time[0]
    resampled = np.interp(det_time, sim_time, sim_strain, left=0.0, right=0.0)

    det_zero = det_strain - det_strain.mean()
    sim_zero = resampled - resampled.mean()

    if np.allclose(sim_zero, 0.0):
        return 0.0, 1.0

    corr = np.correlate(det_zero, sim_zero, mode="full")
    lag_index = corr.argmax() - (sim_zero.size - 1)
    time_offset = lag_index * dt

    scale_num = np.dot(det_zero, sim_zero)
    scale_den = np.dot(sim_zero, sim_zero)
    amplitude_scale = scale_num / scale_den if scale_den > 0 else 1.0

    return time_offset, amplitude_scale


def load_whitened_segment(path: Path, detector: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    time_key = f"{detector}_time"
    strain_key = f"{detector}_strain"
    if time_key not in data or strain_key not in data:
        raise ValueError(f"Whitened segment {path} missing keys for detector {detector}")
    times = data[time_key]
    strain = data[strain_key]
    times = times - times[0]
    return times, strain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform", type=Path, required=True, help="Path to waveform_timeseries.npz")
    parser.add_argument("--whitened", type=Path, required=True, help="Path to whitened segment .npz")
    parser.add_argument("--detector", type=str, default="H1", help="Detector to overlay (e.g., H1)")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="Shift applied to simulated waveform (seconds)")
    parser.add_argument("--amplitude-scale", type=float, default=1.0,
                        help="Scale factor applied to simulated strain")
    parser.add_argument("--auto-align", action="store_true",
                        help="Estimate time offset and amplitude scale via cross-correlation")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    sim_time, sim_strain = load_waveform(args.waveform)
    whiten_time, whiten_strain = load_whitened_segment(args.whitened, args.detector)

    time_offset = args.time_offset
    amplitude_scale = args.amplitude_scale

    if args.auto_align:
        auto_offset, auto_scale = estimate_alignment(sim_time, sim_strain, whiten_time, whiten_strain)
        time_offset += auto_offset
        amplitude_scale *= auto_scale
        print(f"Auto alignment -> time_offset={auto_offset:.6f}s, amplitude_scale={auto_scale:.3e}")

    sim_time = sim_time + time_offset
    sim_strain = sim_strain * amplitude_scale

    plt.figure(figsize=(10, 5))
    plt.plot(whiten_time, whiten_strain, label=f"Whitened {args.detector}", linewidth=1.0, alpha=0.7)
    plt.plot(sim_time, sim_strain, label="Simulated h_plus", linewidth=1.5)
    plt.xlabel("Time since segment start [s]")
    plt.ylabel("Strain")
    plt.title(f"Waveform overlay: {args.detector}")
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()
