import numpy as np
import torch

from fifth_dimension_search.brane_world import (
    BSSNParameters,
    coordinate_grids,
    solve_initial_conditions,
)


def test_yukawa_initial_data_matches_profile():
    params = BSSNParameters(q=5e-3, m5=0.2, L5=0.0)
    shape = (8, 8, 8, 1)
    spacing = (0.8, 0.8, 0.8, 1.0)
    state = solve_initial_conditions(shape, spacing, params, device="cpu", dtype=torch.float64)

    phi = state.phi_brane.cpu().numpy()
    X, Y, Z, _ = coordinate_grids(shape, spacing)
    r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2).cpu().numpy()

    interaction_length = 1.0 / max(params.m5, 1e-3)
    amplitude = params.q if params.q != 0.0 else 1e-2
    expected = amplitude * np.exp(-r / interaction_length) / (1.0 + r)

    mask = expected > 1e-8
    relative_error = np.abs(phi[mask] - expected[mask]) / expected[mask]
    assert np.median(relative_error) < 5e-2
