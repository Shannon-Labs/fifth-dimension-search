import numpy as np

from fifth_dimension_search import analysis
from fifth_dimension_search.brane_world import BSSNParameters, C_SI, scale_waveform_to_observer


def test_phase_summary_loads_with_gr_baseline():
    table = analysis.load_phase_summary()
    assert "GR" in set(table["label"])


def test_compute_phase_deviations_ignores_baseline():
    table = analysis.load_phase_summary()
    deviations = analysis.compute_phase_deviations(table)
    labels = {entry.label for entry in deviations}
    assert "GR" not in labels
    assert labels  # at least one non-GR entry bundled


def test_detectability_assessment_reports_expected_keys():
    metrics = analysis.load_kk_metrics()
    summary = analysis.detectability_assessment(metrics, snr_grid=(10,))
    assert "SNR=10" in summary
    stats = summary["SNR=10"]
    assert {"phase_sigma", "max_phase_diff", "ratio_to_noise"} <= set(stats)


def test_scale_waveform_to_observer_matches_manual_conversion():
    params = BSSNParameters()
    amplitude = 4.0e-4
    radius = 12.0
    expected = amplitude / radius * params.lattice_length_m * C_SI / params.observation_distance_m
    observed = scale_waveform_to_observer(amplitude, radius, params)
    assert np.isclose(observed, expected, rtol=1e-12, atol=0.0)


def test_packaged_peak_strain_has_physical_magnitude():
    table = analysis.load_phase_summary()
    peak = table["peak_h_plus"].max()
    assert 1e-22 < peak < 1e-20
