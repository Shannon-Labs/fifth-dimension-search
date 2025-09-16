from fifth_dimension_search import analysis


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
