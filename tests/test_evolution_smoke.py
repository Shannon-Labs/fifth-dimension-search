from fifth_dimension_search import template_bank
from fifth_dimension_search.brane_world import BSSNParameters


def test_evolve_to_waveform_smoke():
    params = BSSNParameters(q=1e-4, m5=1e-3, L5=10.0)
    shape = (6, 6, 6, 2)
    spacing = (0.5, 0.5, 0.5, 1.0)
    result = template_bank.evolve_to_waveform(
        params=params,
        shape=shape,
        spacing=spacing,
        steps=3,
        dt=0.01,
        radius=1.0,
        device="cpu",
    )

    assert set(result) == {"time_geom", "h_plus_geom", "h_cross_geom", "psi4"}
    assert len(result["psi4"]) == 3
    assert result["h_plus_geom"].shape == result["h_cross_geom"].shape
    assert result["time_geom"].shape[0] == result["h_plus_geom"].shape[0]
