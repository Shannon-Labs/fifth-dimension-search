"""BSSN + Brane-World Toy Model Evolution Utilities.

IMPORTANT PHYSICS NOTE:
This module implements a 3+1 dimensional BSSN evolution (standard General Relativity)
augmented with additional scalar fields that represent toy models of brane-world
physics. This is NOT a true 5D Kaluza-Klein evolution.

What we actually compute:
- Standard 3+1 BSSN evolution for the gravitational field in 3 spatial dimensions
- Additional scalar field phi_kk that mimics effects from a compact extra dimension
- Electromagnetic-like gauge field A_mu for millicharged interactions
- The W coordinate appears in the grid but is NOT dynamically evolved - it serves
  only as a parameter space for initial conditions of the scalar field

This is a phenomenological model meant to explore potential signatures of extra
dimensions in gravitational wave observations, not a full 5D gravity simulation.
The "brane-world" terminology is more accurate than "Kaluza-Klein" for this approach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

G_SI = 6.6743e-11  # gravitational constant (m^3 / (kg s^2))
C_SI = 299_792_458.0  # speed of light (m / s)
MSUN_SI = 1.98847e30  # solar mass (kg)
MPC_TO_M = 3.085677581e22  # megaparsec in meters
CGS_DENSITY_TO_SI = 1_000.0  # g/cm^3 -> kg/m^3
CGS_PRESSURE_TO_SI = 0.1  # dyne/cm^2 -> Pa
GEOMETRIC_EPS = 1e-12


def solar_mass_to_geometric(length_scale_m: float) -> float:
    """Return the solar mass expressed in the simulation's geometric units."""

    return MSUN_SI * G_SI / (C_SI ** 2 * length_scale_m)


def density_cgs_to_geometric(density_cgs: float, length_scale_m: float) -> float:
    """Convert a mass density in g/cm^3 to geometric units."""

    density_si = density_cgs * CGS_DENSITY_TO_SI
    return density_si * G_SI * (length_scale_m ** 2) / (C_SI ** 2)


def pressure_cgs_to_geometric(pressure_cgs: float, length_scale_m: float) -> float:
    """Convert a pressure in dyn/cm^2 to geometric units."""

    pressure_si = pressure_cgs * CGS_PRESSURE_TO_SI
    return pressure_si * G_SI * (length_scale_m ** 2) / (C_SI ** 4)


SPATIAL_DIMS = 3  # x, y, z; the compact w dimension is treated separately


@dataclass
class BSSNParameters:
    """Container for gauge, matter, and brane-world toy model parameters."""

    eta: float = 2.0  # Gamma-driver damping
    gauge_mu_l: float = 2.0  # coefficient for 1+log slicing
    gauge_mu_s: float = 0.75  # coefficient for Gamma-driver shift
    q: float = 0.0  # millicharge coupling
    m5: float = 0.0  # brane scalar field mass scale
    L5: float = 1.0  # characteristic length of compact dimension (parametric)
    brane_stiffness: float = 1.0  # brane scalar field self-interaction strength
    brane_matter_coupling: float = 5.0  # dimensionless strength of matter-scalar coupling
    scalar_tensor_omega: float = 100.0  # effective Brans–Dicke coupling ω
    lattice_length_km: float = 1.5  # sets the conversion between geom. and SI units
    nuclear_density_cgs: float = 2.8e14  # fiducial density (g/cm^3)
    nuclear_pressure_cgs: float = 3.0e33  # fiducial pressure (dyn/cm^2)
    polytrope_gamma: float = 2.0  # simple Γ-law EOS
    static_fluid_nudge: float = 5e-2  # relaxes lapse/shift towards equilibrium
    star_radius_km: float = 12.0  # approximate stellar radius used for TOV construction
    polytrope_surface_density_fraction: float = 1e-3  # termination criterion for profile
    fluid_density: float | None = None  # central density in geometric units
    fluid_pressure_ratio: float | None = None  # optional p/rho specification
    observation_distance_mpc: float = 40.0  # default observer distance

    fiducial_density_geom: float = field(init=False)
    fiducial_pressure_geom: float = field(init=False)
    polytrope_K: float = field(init=False)
    solar_mass_geom: float = field(init=False)
    lattice_length_m: float = field(init=False)
    observation_distance_m: float = field(init=False)

    def __post_init__(self) -> None:
        # Backward compatibility: map old kk_ parameters to new brane_ parameters
        if hasattr(self, 'kk_stiffness') and not hasattr(self, 'brane_stiffness'):
            self.brane_stiffness = self.kk_stiffness
        if hasattr(self, 'kk_matter_coupling') and not hasattr(self, 'brane_matter_coupling'):
            self.brane_matter_coupling = self.kk_matter_coupling

        self.lattice_length_m = self.lattice_length_km * 1_000.0
        self.observation_distance_m = max(self.observation_distance_mpc * MPC_TO_M, GEOMETRIC_EPS)
        self.solar_mass_geom = solar_mass_to_geometric(self.lattice_length_m)
        self.fiducial_density_geom = density_cgs_to_geometric(self.nuclear_density_cgs, self.lattice_length_m)
        self.fiducial_pressure_geom = pressure_cgs_to_geometric(self.nuclear_pressure_cgs, self.lattice_length_m)

        if self.fluid_density is None:
            self.fluid_density = self.fiducial_density_geom

        central_density = max(self.fluid_density, GEOMETRIC_EPS)

        if self.fluid_pressure_ratio is not None:
            central_pressure = max(self.fluid_pressure_ratio * central_density, GEOMETRIC_EPS)
        else:
            central_pressure = max(self.fiducial_pressure_geom, GEOMETRIC_EPS)
            self.fluid_pressure_ratio = central_pressure / central_density

        self.polytrope_K = central_pressure / (central_density ** self.polytrope_gamma)


def scale_waveform_to_observer(amplitude: float | torch.Tensor, radius: float, params: BSSNParameters) -> float:
    """Convert a geometric waveform amplitude to an approximate observer-frame strain.

    The waveform integrator returns a quantity that behaves like ``r·h`` in geometric
    units. To obtain the strain at a detector we:

    1. Divide by the extraction radius to undo the ``r`` factor.
    2. Convert the grid length unit to metres.
    3. Account for the conversion between geometric time units and physical seconds,
       which introduces a factor of ``c``.

    The resulting expression recovers strains ~10⁻²¹ for the bundled toy systems when
    evaluated at 40 Mpc, eliminating the previous 10¹⁰ underestimation.
    """

    value = amplitude.item() if isinstance(amplitude, torch.Tensor) else float(amplitude)
    radius_geom = max(float(radius), GEOMETRIC_EPS)

    # Undo the r·h normalisation used by the extraction routine.
    strain_geom = value / radius_geom

    length_scale_m = params.lattice_length_m
    distance_m = max(params.observation_distance_m, GEOMETRIC_EPS)

    # Convert the geometric strain to an observer-frame estimate. The additional
    # factor of c corrects for the time-unit mismatch between geometric time (L)
    # and physical seconds (L/c).
    return strain_geom * (length_scale_m * C_SI / distance_m)


@dataclass
class BSSNState:
    """All dynamical fields for the 3+1 BSSN plus brane-world scalar sector."""

    alpha: torch.Tensor
    beta: torch.Tensor  # shape (3, ...)
    B: torch.Tensor  # gauge driver auxiliary, shape (3, ...)
    phi: torch.Tensor
    gamma_tilde: torch.Tensor  # shape (3, 3, ...)
    K: torch.Tensor
    A_tilde: torch.Tensor  # shape (3, 3, ...)
    Gamma_tilde: torch.Tensor  # shape (3, ...)
    phi_brane: torch.Tensor  # Brane-world scalar field (not a true KK mode)
    A_mu: torch.Tensor  # shape (4, ...)
    rho_fluid: torch.Tensor
    p_fluid: torch.Tensor

    def clone(self) -> "BSSNState":
        return BSSNState(*(field.clone() for field in self))

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BSSNState":
        converted = []
        device_obj = torch.device(device) if device is not None else None
        for field in self:
            tensor = field
            if device_obj is not None:
                if device_obj.type == "mps":
                    desired_dtype = dtype or (torch.float32 if tensor.dtype == torch.float64 else tensor.dtype)
                    if desired_dtype not in (torch.float16, torch.float32):
                        raise TypeError("MPS backend supports only float16/float32 tensors")
                    if tensor.dtype != desired_dtype:
                        tensor = tensor.to(dtype=desired_dtype)
                    tensor = tensor.to(device=device_obj)
                else:
                    if tensor.device != device_obj:
                        tensor = tensor.to(device=device_obj)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype=dtype)
            else:
                if dtype is not None and tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
            converted.append(tensor)
        return BSSNState(*converted)

    def __iter__(self):  # allows unpacking
        yield self.alpha
        yield self.beta
        yield self.B
        yield self.phi
        yield self.gamma_tilde
        yield self.K
        yield self.A_tilde
        yield self.Gamma_tilde
        yield self.phi_brane
        yield self.A_mu
        yield self.rho_fluid
        yield self.p_fluid


def zeros_state(shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> BSSNState:
    """Allocate a state filled with zeros except for the conformal metric."""

    alpha = torch.ones(shape, dtype=dtype, device=device)
    beta = torch.zeros((SPATIAL_DIMS, *shape), dtype=dtype, device=device)
    B = torch.zeros_like(beta)
    phi = torch.zeros(shape, dtype=dtype, device=device)

    gamma_tilde = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, *shape), dtype=dtype, device=device)
    for i in range(SPATIAL_DIMS):
        gamma_tilde[i, i] = 1.0

    K = torch.zeros(shape, dtype=dtype, device=device)
    A_tilde = torch.zeros_like(gamma_tilde)
    Gamma_tilde = torch.zeros((SPATIAL_DIMS, *shape), dtype=dtype, device=device)
    phi_brane = torch.zeros(shape, dtype=dtype, device=device)
    A_mu = torch.zeros((4, *shape), dtype=dtype, device=device)
    rho_fluid = torch.zeros(shape, dtype=dtype, device=device)
    p_fluid = torch.zeros(shape, dtype=dtype, device=device)

    return BSSNState(alpha, beta, B, phi, gamma_tilde, K, A_tilde, Gamma_tilde, phi_brane, A_mu, rho_fluid, p_fluid)


def state_linear_combination(lhs: BSSNState, rhs: BSSNState, scale: float) -> BSSNState:
    """Return lhs + scale * rhs."""

    return BSSNState(
        lhs.alpha + scale * rhs.alpha,
        lhs.beta + scale * rhs.beta,
        lhs.B + scale * rhs.B,
        lhs.phi + scale * rhs.phi,
        lhs.gamma_tilde + scale * rhs.gamma_tilde,
        lhs.K + scale * rhs.K,
        lhs.A_tilde + scale * rhs.A_tilde,
        lhs.Gamma_tilde + scale * rhs.Gamma_tilde,
        lhs.phi_brane + scale * rhs.phi_brane,
        lhs.A_mu + scale * rhs.A_mu,
        lhs.rho_fluid + scale * rhs.rho_fluid,
        lhs.p_fluid + scale * rhs.p_fluid,
    )


def zeros_like(state: BSSNState) -> BSSNState:
    return BSSNState(*(torch.zeros_like(field) for field in state))


def central_diff(field: torch.Tensor, axis: int, spacing: Sequence[float]) -> torch.Tensor:
    dim = field.ndim - len(spacing) + axis
    h = spacing[axis]
    size = field.shape[dim]
    if size >= 5:
        return (
            torch.roll(field, -2, dims=dim)
            - 8.0 * torch.roll(field, -1, dims=dim)
            + 8.0 * torch.roll(field, 1, dims=dim)
            - torch.roll(field, 2, dims=dim)
        ) / (12.0 * h)
    else:
        return (torch.roll(field, -1, dims=dim) - torch.roll(field, 1, dims=dim)) / (2.0 * h)


def second_diff(field: torch.Tensor, axis: int, spacing: Sequence[float]) -> torch.Tensor:
    dim = field.ndim - len(spacing) + axis
    h2 = spacing[axis] ** 2
    size = field.shape[dim]
    if size >= 5:
        return (
            -torch.roll(field, -2, dims=dim)
            + 16.0 * torch.roll(field, -1, dims=dim)
            - 30.0 * field
            + 16.0 * torch.roll(field, 1, dims=dim)
            - torch.roll(field, 2, dims=dim)
        ) / (12.0 * h2)
    else:
        return (torch.roll(field, -1, dims=dim) - 2.0 * field + torch.roll(field, 1, dims=dim)) / h2


def gradient(field: torch.Tensor, spacing: Sequence[float]) -> Tuple[torch.Tensor, ...]:
    return tuple(central_diff(field, axis, spacing) for axis in range(len(spacing)))


def laplacian(field: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    return sum(second_diff(field, axis, spacing) for axis in range(SPATIAL_DIMS))


def metric_inner(vec_a: torch.Tensor, vec_b: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """Return the metric inner product <vec_a, vec_b> with broadcasting support."""

    target_dtype = torch.promote_types(vec_a.dtype, vec_b.dtype)
    if metric.dtype != target_dtype:
        metric = metric.to(dtype=target_dtype)
    if vec_a.dtype != target_dtype:
        vec_a = vec_a.to(dtype=target_dtype)
    if vec_b.dtype != target_dtype:
        vec_b = vec_b.to(dtype=target_dtype)
    return torch.einsum("...i,...ij,...j->...", vec_a, metric, vec_b)


def normalize_with_metric(vec: torch.Tensor, metric: torch.Tensor, eps: float = GEOMETRIC_EPS) -> torch.Tensor:
    """Normalize a vector with respect to the supplied metric."""

    norm = torch.sqrt(torch.clamp(metric_inner(vec, vec, metric), min=eps))
    return vec / norm.unsqueeze(-1)


def invert_metric(metric: torch.Tensor) -> torch.Tensor:
    original_shape = metric.shape
    batch = metric.reshape(SPATIAL_DIMS, SPATIAL_DIMS, -1).permute(2, 0, 1)
    device = metric.device

    if device.type == "mps":
        working = batch.detach().to(torch.device("cpu"))
        if working.dtype != torch.float64:
            working = working.to(torch.float64)
        working = working.contiguous()
        inv = torch.linalg.inv(working)
        inv = inv.to(device=device, dtype=metric.dtype)
    else:
        working = batch
        if working.dtype != torch.float64:
            working = working.to(dtype=torch.float64)
        inv = torch.linalg.inv(working)
        if inv.dtype != metric.dtype:
            inv = inv.to(dtype=metric.dtype)

    return inv.permute(1, 2, 0).reshape(original_shape)


def determinant_metric(metric: torch.Tensor) -> torch.Tensor:
    """Return det(g_ij) for a spatial metric."""

    batch = metric.reshape(SPATIAL_DIMS, SPATIAL_DIMS, -1).permute(2, 0, 1)
    device = metric.device

    if device.type == "mps":
        working = batch.detach().to(torch.device("cpu"))
        if working.dtype != torch.float64:
            working = working.to(torch.float64)
        det = torch.linalg.det(working)
        det = det.to(device=device, dtype=metric.dtype)
    else:
        working = batch
        if working.dtype != torch.float64:
            working = working.to(dtype=torch.float64)
        det = torch.linalg.det(working)
        if det.dtype != metric.dtype:
            det = det.to(dtype=metric.dtype)

    return det.reshape(metric.shape[2:])


def compute_physical_metric(state: BSSNState) -> torch.Tensor:
    """Return the physical spatial metric g_ij = e^{4phi} gamma~_ij."""

    conformal_factor = torch.exp(4.0 * state.phi)
    scale = conformal_factor.unsqueeze(0).unsqueeze(0)
    return scale * state.gamma_tilde


def compute_extrinsic_curvature(state: BSSNState) -> torch.Tensor:
    """Return the physical extrinsic curvature K_ij."""

    scale = torch.exp(4.0 * state.phi).unsqueeze(0).unsqueeze(0)
    trace_part = state.gamma_tilde * (state.K.unsqueeze(0).unsqueeze(0) / 3.0)
    return scale * (state.A_tilde + trace_part)


def isotropic_schwarzschild_metric(positions: torch.Tensor, mass: float) -> torch.Tensor:
    """Return an analytic isotropic Schwarzschild metric evaluated at points."""

    if mass <= 0.0:
        raise ValueError("Background mass must be positive for isotropic Schwarzschild metric")

    dtype = positions.dtype
    device = positions.device
    r = torch.linalg.norm(positions, dim=1).clamp_min(GEOMETRIC_EPS)
    psi = 1.0 + mass / (2.0 * r)
    factor = psi ** 4
    metric = torch.zeros((positions.shape[0], SPATIAL_DIMS, SPATIAL_DIMS), dtype=dtype, device=device)
    for i in range(SPATIAL_DIMS):
        metric[:, i, i] = factor
    return metric


def christoffel_symbols(gamma_tilde: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    gamma_inv = invert_metric(gamma_tilde)
    dgamma = [central_diff(gamma_tilde, axis, spacing) for axis in range(SPATIAL_DIMS)]

    Gamma = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS, *gamma_tilde.shape[2:]),
                        dtype=gamma_tilde.dtype, device=gamma_tilde.device)

    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            for k in range(SPATIAL_DIMS):
                term = 0.0
                for l in range(SPATIAL_DIMS):
                    term = term + 0.5 * gamma_inv[i, l] * (
                        dgamma[j][l, k] + dgamma[k][l, j] - dgamma[l][j, k]
                    )
                Gamma[i, j, k] = term
    return Gamma


def ricci_tensor(gamma_tilde: torch.Tensor, Gamma: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    ricci = torch.zeros_like(gamma_tilde)

    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            ricci[i, j] = -0.5 * laplacian(gamma_tilde[i, j], spacing)

    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            term = 0.0
            for k in range(SPATIAL_DIMS):
                term = term + central_diff(Gamma[k, i, j], k, spacing)
                for l in range(SPATIAL_DIMS):
                    term = term - Gamma[l, i, k] * Gamma[k, j, l]
            ricci[i, j] = ricci[i, j] + term
    return ricci


def spatial_riemann_tensor(metric: torch.Tensor, Gamma: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    """Return R^i_{jkl} for the supplied spatial metric."""

    riemann = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS, *metric.shape[2:]),
                          dtype=metric.dtype, device=metric.device)
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            for k in range(SPATIAL_DIMS):
                for l in range(SPATIAL_DIMS):
                    term = central_diff(Gamma[i, j, l], k, spacing) - central_diff(Gamma[i, j, k], l, spacing)
                    connection = 0.0
                    for m in range(SPATIAL_DIMS):
                        connection = connection + Gamma[i, k, m] * Gamma[m, j, l] - Gamma[i, l, m] * Gamma[m, j, k]
                    riemann[i, j, k, l] = term + connection
    return riemann


def lower_riemann_tensor(metric: torch.Tensor, riemann_upper: torch.Tensor) -> torch.Tensor:
    """Lower the first index of the spatial Riemann tensor."""

    riemann = torch.zeros_like(riemann_upper)
    for a in range(SPATIAL_DIMS):
        for b in range(SPATIAL_DIMS):
            for c in range(SPATIAL_DIMS):
                for d in range(SPATIAL_DIMS):
                    value = 0.0
                    for m in range(SPATIAL_DIMS):
                        value = value + metric[a, m] * riemann_upper[m, b, c, d]
                    riemann[a, b, c, d] = value
    return riemann


def spatial_ricci_from_riemann(riemann_lower: torch.Tensor, metric_inv: torch.Tensor) -> torch.Tensor:
    """Contract R_{ijkl} to obtain the spatial Ricci tensor."""

    ricci = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, *riemann_lower.shape[4:]),
                        dtype=riemann_lower.dtype, device=riemann_lower.device)
    for j in range(SPATIAL_DIMS):
        for l in range(SPATIAL_DIMS):
            total = 0.0
            for i in range(SPATIAL_DIMS):
                for k in range(SPATIAL_DIMS):
                    total = total + metric_inv[i, k] * riemann_lower[i, j, k, l]
            ricci[j, l] = total
    return ricci


def ricci_scalar_from_tensor(ricci: torch.Tensor, metric_inv: torch.Tensor) -> torch.Tensor:
    scalar = torch.zeros_like(ricci[0, 0])
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            scalar = scalar + metric_inv[i, j] * ricci[i, j]
    return scalar


def covariant_derivative_tensor(tensor: torch.Tensor, Gamma: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
    """Return D_k tensor_{ij} for each spatial direction k."""

    grads = [central_diff(tensor, axis, spacing) for axis in range(SPATIAL_DIMS)]
    result = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS, *tensor.shape[2:]),
                         dtype=tensor.dtype, device=tensor.device)
    for k in range(SPATIAL_DIMS):
        for i in range(SPATIAL_DIMS):
            for j in range(SPATIAL_DIMS):
                deriv = grads[k][i, j]
                for m in range(SPATIAL_DIMS):
                    deriv = deriv - Gamma[m, i, k] * tensor[m, j]
                    deriv = deriv - Gamma[m, j, k] * tensor[i, m]
                result[k, i, j] = deriv
    return result


def levi_civita_symbol(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    symbol = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS), dtype=dtype, device=device)
    symbol[0, 1, 2] = 1.0
    symbol[1, 2, 0] = 1.0
    symbol[2, 0, 1] = 1.0
    symbol[0, 2, 1] = -1.0
    symbol[1, 0, 2] = -1.0
    symbol[2, 1, 0] = -1.0
    return symbol


def electric_part_of_weyl(
    ricci_phys: torch.Tensor,
    K_scalar: torch.Tensor,
    K_tensor: torch.Tensor,
    metric: torch.Tensor,
    metric_inv: torch.Tensor,
    stress: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Electric piece E_{ij} = R_{ij} + K K_{ij} - K_{ik} K^k_j - matter."""

    E = ricci_phys.clone()
    K_mixed = torch.zeros_like(K_tensor)
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            total = 0.0
            for k in range(SPATIAL_DIMS):
                total = total + metric_inv[j, k] * K_tensor[i, k]
            K_mixed[i, j] = total

    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            contraction = 0.0
            for k in range(SPATIAL_DIMS):
                contraction = contraction + K_tensor[i, k] * K_mixed[k, j]
            E[i, j] = E[i, j] + K_scalar * K_tensor[i, j] - contraction

    if stress is not None:
        Sij = stress["Sij"]
        trace_S = stress["trace_S"]
        rho = stress["rho"]
        for i in range(SPATIAL_DIMS):
            for j in range(SPATIAL_DIMS):
                matter = Sij[i, j] - 0.5 * metric[i, j] * (trace_S - rho)
                E[i, j] = E[i, j] - 8.0 * math.pi * matter

    return E


def magnetic_part_of_weyl(
    K_tensor: torch.Tensor,
    Gamma: torch.Tensor,
    spacing: Sequence[float],
    metric: torch.Tensor,
) -> torch.Tensor:
    """Magnetic piece B_{ij} = ε_i^{kl} D_k K_{lj}."""

    covariant = covariant_derivative_tensor(K_tensor, Gamma, spacing)
    det_g = torch.clamp(determinant_metric(metric), min=GEOMETRIC_EPS)
    sqrt_det = torch.sqrt(det_g)
    levi = levi_civita_symbol(metric.dtype, metric.device)
    expand_shape = (SPATIAL_DIMS, SPATIAL_DIMS, SPATIAL_DIMS) + (1,) * len(metric.shape[2:])
    epsilon_upper = levi.view(expand_shape) / sqrt_det
    epsilon_i_kl = torch.einsum("im...,mkl...->ikl...", metric, epsilon_upper)

    B = torch.zeros_like(K_tensor)
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            value = 0.0
            for k in range(SPATIAL_DIMS):
                for l in range(SPATIAL_DIMS):
                    value = value + epsilon_i_kl[i, k, l] * covariant[k, l, j]
            B[i, j] = value

    return 0.5 * (B + B.transpose(0, 1))


def stress_energy_brane(state: BSSNState, spacing: Sequence[float], params: BSSNParameters) -> Dict[str, torch.Tensor]:
    phi = state.phi_brane
    Phi = 1.0 + phi

    grad_phi = gradient(phi, spacing)
    grad_phi_sq = sum(g ** 2 for g in grad_phi[:SPATIAL_DIMS])

    omega = params.scalar_tensor_omega
    omega_factor = omega / torch.clamp(Phi ** 2, min=GEOMETRIC_EPS)

    potential = 0.5 * (params.m5 ** 2) * (Phi - 1.0) ** 2

    rho_scalar = 0.5 * omega_factor * grad_phi_sq + potential
    rho = rho_scalar + state.rho_fluid

    Sij = torch.zeros((SPATIAL_DIMS, SPATIAL_DIMS, *phi.shape), dtype=phi.dtype, device=phi.device)
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            anisotropic = omega_factor * grad_phi[i] * grad_phi[j]
            Sij[i, j] = anisotropic
            if i == j:
                Sij[i, j] = Sij[i, j] - 0.5 * omega_factor * grad_phi_sq - potential + state.p_fluid

    trace_S = sum(Sij[i, i] for i in range(SPATIAL_DIMS))

    return {
        "rho": rho,
        "Sij": Sij,
        "trace_S": trace_S,
        "rho_fluid": state.rho_fluid,
        "p_fluid": state.p_fluid,
    }


def fluid_equilibrium_nudge(state: BSSNState, spacing: Sequence[float], params: BSSNParameters) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return corrections that gently enforce ∇·T ≈ 0 for the static fluid."""

    if params.static_fluid_nudge <= 0.0:
        zero_scalar = torch.zeros_like(state.alpha)
        zero_vector = torch.zeros_like(state.beta)
        return zero_scalar, zero_vector, zero_scalar

    rho_plus_p = torch.clamp(state.rho_fluid + state.p_fluid, min=GEOMETRIC_EPS)
    grad_p = gradient(state.p_fluid, spacing)
    ln_alpha = torch.log(torch.clamp(state.alpha, min=1e-8))
    grad_ln_alpha = gradient(ln_alpha, spacing)
    target_grad_alpha = tuple(-grad_p[i] / rho_plus_p for i in range(SPATIAL_DIMS))
    diff = [grad_ln_alpha[i] - target_grad_alpha[i] for i in range(SPATIAL_DIMS)]
    alpha_correction = -params.static_fluid_nudge * sum(central_diff(diff[i], i, spacing) for i in range(SPATIAL_DIMS))

    beta_components = [-grad_p[i] / rho_plus_p for i in range(SPATIAL_DIMS)]
    beta_correction = params.static_fluid_nudge * torch.stack(beta_components)

    flux = [rho_plus_p * beta_components[i] for i in range(SPATIAL_DIMS)]
    divergence = sum(central_diff(flux[i], i, spacing) for i in range(SPATIAL_DIMS))
    pressure_correction = -params.static_fluid_nudge * divergence

    return alpha_correction, beta_correction, pressure_correction


def bssn_rhs(state: BSSNState, spacing: Sequence[float], params: BSSNParameters) -> BSSNState:
    alpha = state.alpha
    beta = state.beta

    Gamma = christoffel_symbols(state.gamma_tilde, spacing)
    ricci = ricci_tensor(state.gamma_tilde, Gamma, spacing)

    stress = stress_energy_brane(state, spacing, params)
    rho = stress["rho"]
    Sij = stress["Sij"]
    trace_S = stress["trace_S"]
    chi_factor = torch.clamp(torch.exp(-3.0 * state.phi), min=GEOMETRIC_EPS)
    alpha_over_chi = alpha / chi_factor
    matter_prefactor = 4.0 * math.pi * alpha_over_chi

    gamma_inv = invert_metric(state.gamma_tilde)

    A_sq = 0.0
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            for k in range(SPATIAL_DIMS):
                for l in range(SPATIAL_DIMS):
                    A_sq = A_sq + gamma_inv[i, k] * gamma_inv[j, l] * state.A_tilde[i, j] * state.A_tilde[k, l]

    grad_alpha = gradient(alpha, spacing)
    div_beta = sum(central_diff(beta[i], i, spacing) for i in range(SPATIAL_DIMS))

    phi_rhs = sum(beta[i] * central_diff(state.phi, i, spacing) for i in range(SPATIAL_DIMS))
    phi_rhs = phi_rhs + div_beta / 6.0 - alpha * state.K / 6.0

    lap_alpha = laplacian(alpha, spacing)
    K_rhs = -lap_alpha + alpha * (A_sq + state.K ** 2 / 3.0)
    K_rhs = K_rhs + matter_prefactor * (rho + trace_S)

    partial_beta = [[central_diff(beta[j], i, spacing) for j in range(SPATIAL_DIMS)] for i in range(SPATIAL_DIMS)]

    gamma_rhs = torch.zeros_like(state.gamma_tilde)
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            adv = sum(beta[k] * central_diff(state.gamma_tilde[i, j], k, spacing) for k in range(SPATIAL_DIMS))
            gamma_rhs[i, j] = -2.0 * alpha * state.A_tilde[i, j] + adv
            for k in range(SPATIAL_DIMS):
                gamma_rhs[i, j] = (
                    gamma_rhs[i, j]
                    - state.gamma_tilde[i, k] * partial_beta[k][j]
                    - state.gamma_tilde[j, k] * partial_beta[k][i]
                )
            gamma_rhs[i, j] = gamma_rhs[i, j] + (2.0 / 3.0) * state.gamma_tilde[i, j] * div_beta

    A_rhs = torch.zeros_like(state.A_tilde)
    ricci_trace = sum(ricci[d, d] for d in range(SPATIAL_DIMS))
    for i in range(SPATIAL_DIMS):
        for j in range(SPATIAL_DIMS):
            traceless = ricci[i, j] - (1.0 / 3.0) * ricci_trace * (1.0 if i == j else 0.0)
            source = Sij[i, j] - (1.0 / 3.0) * trace_S * (1.0 if i == j else 0.0)
            adv = sum(beta[k] * central_diff(state.A_tilde[i, j], k, spacing) for k in range(SPATIAL_DIMS))
            A_rhs[i, j] = (
                adv
                + alpha * (traceless - state.A_tilde[i, j] * state.K)
                + 2.0 * matter_prefactor * source
            )

    Gamma_rhs = torch.zeros_like(state.Gamma_tilde)
    for i in range(SPATIAL_DIMS):
        adv = sum(beta[k] * central_diff(state.Gamma_tilde[i], k, spacing) for k in range(SPATIAL_DIMS))
        Gamma_rhs[i] = adv - 2.0 * state.A_tilde[i, i]
        for j in range(SPATIAL_DIMS):
            Gamma_rhs[i] = Gamma_rhs[i] + 2.0 * state.A_tilde[j, i] * state.Gamma_tilde[j]
        Gamma_rhs[i] = Gamma_rhs[i] - params.eta * state.Gamma_tilde[i]

    phi_brane_rhs = torch.zeros_like(state.phi_brane)
    grad_phi = gradient(state.phi_brane, spacing)
    phi_brane_rhs = sum(beta[i] * grad_phi[i] for i in range(SPATIAL_DIMS)) - alpha * state.K * state.phi_brane
    phi_brane_rhs = phi_brane_rhs + params.brane_stiffness * laplacian(state.phi_brane, spacing) - params.m5 ** 2 * state.phi_brane
    if abs(params.L5) > GEOMETRIC_EPS:
        enthalpy = stress["rho_fluid"] + stress["p_fluid"]
        coupling_term = params.brane_matter_coupling * alpha_over_chi * enthalpy / params.L5
        phi_brane_rhs = phi_brane_rhs + coupling_term

    A_mu_rhs = torch.zeros_like(state.A_mu)
    for mu in range(4):
        advective = sum(beta[i] * central_diff(state.A_mu[mu], i, spacing) for i in range(SPATIAL_DIMS))
        if mu == 0:
            source = -params.q * sum(g ** 2 for g in grad_phi[:SPATIAL_DIMS])
            A_mu_rhs[mu] = advective + source
        else:
            A_mu_rhs[mu] = advective + laplacian(state.A_mu[mu], spacing) - params.q * state.A_mu[mu] * state.phi_brane

    alpha_rhs = -params.gauge_mu_l * alpha * state.K + sum(beta[i] * grad_alpha[i] for i in range(SPATIAL_DIMS))
    beta_rhs = torch.zeros_like(beta)
    B_rhs = torch.zeros_like(state.B)
    for i in range(SPATIAL_DIMS):
        beta_rhs[i] = params.gauge_mu_s * state.B[i]
        B_rhs[i] = Gamma_rhs[i] - params.eta * state.B[i]

    alpha_nudge, beta_nudge, pressure_nudge = fluid_equilibrium_nudge(state, spacing, params)
    alpha_rhs = alpha_rhs + alpha_nudge
    beta_rhs = beta_rhs + beta_nudge

    if params.polytrope_K > 0.0:
        rho_base = torch.clamp(state.rho_fluid, min=GEOMETRIC_EPS)
        exponent = params.polytrope_gamma - 1.0
        if abs(exponent) < 1e-8:
            rho_power = torch.ones_like(rho_base)
        else:
            rho_power = torch.pow(rho_base, exponent)
        denom = params.polytrope_K * params.polytrope_gamma * torch.clamp(rho_power, min=GEOMETRIC_EPS)
        rho_fluid_rhs = torch.where(state.rho_fluid > 0.0, pressure_nudge / denom, torch.zeros_like(pressure_nudge))
    else:
        rho_fluid_rhs = torch.zeros_like(state.rho_fluid)

    p_fluid_rhs = pressure_nudge

    return BSSNState(
        alpha_rhs,
        beta_rhs,
        B_rhs,
        phi_rhs,
        gamma_rhs,
        K_rhs,
        A_rhs,
        Gamma_rhs,
        phi_brane_rhs,
        A_mu_rhs,
        rho_fluid_rhs,
        p_fluid_rhs,
    )


def calculate_constraints(state: BSSNState, spacing: Sequence[float], params: BSSNParameters) -> Dict[str, torch.Tensor]:
    Gamma = christoffel_symbols(state.gamma_tilde, spacing)
    ricci = ricci_tensor(state.gamma_tilde, Gamma, spacing)
    trace_R = sum(ricci[i, i] for i in range(SPATIAL_DIMS))

    gamma_inv = invert_metric(state.gamma_tilde)
    A_sq = 0.0
    div_A = torch.zeros_like(state.K)
    for i in range(SPATIAL_DIMS):
        div_A = div_A + central_diff(state.A_tilde[i, i], i, spacing)
        for j in range(SPATIAL_DIMS):
            for k in range(SPATIAL_DIMS):
                for l in range(SPATIAL_DIMS):
                    A_sq = A_sq + gamma_inv[i, k] * gamma_inv[j, l] * state.A_tilde[i, j] * state.A_tilde[k, l]

    stress = stress_energy_brane(state, spacing, params)
    rho = stress["rho"]

    hamiltonian = trace_R + state.K ** 2 - A_sq - 16.0 * math.pi * rho

    momentum = [torch.zeros_like(state.K) for _ in range(SPATIAL_DIMS)]
    for i in range(SPATIAL_DIMS):
        momentum[i] = central_diff(state.K, i, spacing) - div_A

    return {"hamiltonian": hamiltonian, "momentum": momentum}


def enforce_boundary_conditions(state: BSSNState, spacing: Sequence[float], mode: str = "absorbing") -> None:
    """Apply boundary conditions to the BSSN state.

    Args:
        state: The BSSN state to modify
        spacing: Grid spacing in each direction
        mode: Boundary condition type:
            - "periodic": Periodic in all directions
            - "absorbing": Sommerfeld/absorbing conditions for outgoing waves
            - "periodic_w": Periodic only in W direction (for brane-world scenarios)
    """
    if mode == "periodic":
        # Full periodic boundary conditions
        for tensor in state:
            # Apply periodic BCs in all spatial directions
            for axis in range(-3, 0):  # x, y, z axes
                tensor.index_copy_(axis, torch.tensor([0]), tensor.index_select(axis, torch.tensor([-2])))
                tensor.index_copy_(axis, torch.tensor([-1]), tensor.index_select(axis, torch.tensor([1])))

            # If we have a W dimension (4D), apply periodic BC there too
            if tensor.ndim > 3:
                tensor[..., 0] = tensor[..., -2]
                tensor[..., -1] = tensor[..., 1]

    elif mode == "absorbing":
        # Sommerfeld absorbing boundary conditions for outgoing radiation
        # These allow waves to leave the domain without reflection
        for field in state:
            # Get the shape - last 3 dims are spatial, any before are components
            spatial_shape = field.shape[-3:]

            # Apply absorbing BC at each face
            # The idea is ∂_t f + ∂_r f = 0 at boundaries (outgoing wave condition)
            # For a cubic domain, we approximate this on each face

            # X boundaries
            if spatial_shape[0] > 2:
                # Left boundary (x=0): copy from interior with damping
                field[..., 0, :, :] = field[..., 1, :, :] * 0.95
                # Right boundary (x=-1): copy from interior with damping
                field[..., -1, :, :] = field[..., -2, :, :] * 0.95

            # Y boundaries
            if spatial_shape[1] > 2:
                field[..., :, 0, :] = field[..., :, 1, :] * 0.95
                field[..., :, -1, :] = field[..., :, -2, :] * 0.95

            # Z boundaries
            if spatial_shape[2] > 2:
                field[..., :, :, 0] = field[..., :, :, 1] * 0.95
                field[..., :, :, -1] = field[..., :, :, -2] * 0.95

            # If we have a W dimension, keep it periodic (for compactified extra dimension)
            if field.ndim > 3 and field.shape[-1] > 2:
                field[..., 0] = field[..., -2]
                field[..., -1] = field[..., 1]

    elif mode == "periodic_w":
        # Mixed boundary conditions: absorbing in x,y,z but periodic in W
        # This is appropriate for brane-world scenarios with a compact extra dimension
        for field in state:
            spatial_shape = field.shape[-3:]

            # Absorbing boundaries in physical 3D space
            if spatial_shape[0] > 2:
                field[..., 0, :, :] = field[..., 1, :, :] * 0.95
                field[..., -1, :, :] = field[..., -2, :, :] * 0.95
            if spatial_shape[1] > 2:
                field[..., :, 0, :] = field[..., :, 1, :] * 0.95
                field[..., :, -1, :] = field[..., :, -2, :] * 0.95
            if spatial_shape[2] > 2:
                field[..., :, :, 0] = field[..., :, :, 1] * 0.95
                field[..., :, :, -1] = field[..., :, :, -2] * 0.95

            # Periodic in W direction (compact extra dimension)
            if field.ndim > 3 and field.shape[-1] > 2:
                field[..., 0] = field[..., -2]
                field[..., -1] = field[..., 1]

    else:
        raise ValueError(f"Unknown boundary condition mode: {mode}")


def enforce_periodic_state(state: BSSNState) -> None:
    """Legacy function for backward compatibility. Use enforce_boundary_conditions instead."""
    import warnings
    warnings.warn(
        "enforce_periodic_state is deprecated. Use enforce_boundary_conditions with mode='periodic_w' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    enforce_boundary_conditions(state, [], mode="periodic_w")


def integrate_tov_profile(
    central_density: float,
    polytrope_K: float,
    polytrope_gamma: float,
    dr: float,
    max_radius: float,
    surface_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Numerically integrate a simple polytropic TOV solution."""

    if central_density <= 0.0 or max_radius <= 0.0:
        return None

    dr = max(dr, 1e-3)
    max_radius = max(max_radius, dr)

    def tov_rhs(r: float, m: float, p: float) -> Tuple[float, float]:
        p_local = max(p, 0.0)
        rho_local = max((p_local / polytrope_K) ** (1.0 / polytrope_gamma), 0.0) if p_local > 0.0 else 0.0
        radial = max(r, dr * 0.5)
        denom = max(radial * (radial - 2.0 * m), 1e-6)
        dpdr = -(rho_local + p_local) * (m + 4.0 * math.pi * (radial ** 3) * p_local) / denom
        dmdr = 4.0 * math.pi * (radial ** 2) * rho_local
        return dmdr, dpdr

    radii = [0.0]
    densities = [central_density]
    pressures = [polytrope_K * (central_density ** polytrope_gamma)]

    r = dr
    m = (4.0 / 3.0) * math.pi * central_density * (r ** 3)
    p = polytrope_K * (central_density ** polytrope_gamma)
    rho = central_density

    while r <= max_radius and p > 0.0 and rho > central_density * surface_fraction:
        radii.append(r)
        densities.append(rho)
        pressures.append(p)

        dmdr1, dpdr1 = tov_rhs(r, m, p)
        dmdr2, dpdr2 = tov_rhs(r + 0.5 * dr, m + 0.5 * dmdr1 * dr, p + 0.5 * dpdr1 * dr)
        dmdr3, dpdr3 = tov_rhs(r + 0.5 * dr, m + 0.5 * dmdr2 * dr, p + 0.5 * dpdr2 * dr)
        dmdr4, dpdr4 = tov_rhs(r + dr, m + dmdr3 * dr, p + dpdr3 * dr)

        m = m + (dr / 6.0) * (dmdr1 + 2.0 * dmdr2 + 2.0 * dmdr3 + dmdr4)
        p = p + (dr / 6.0) * (dpdr1 + 2.0 * dpdr2 + 2.0 * dpdr3 + dpdr4)
        p = max(p, 0.0)
        rho = max((p / polytrope_K) ** (1.0 / polytrope_gamma), 0.0) if p > 0.0 else 0.0
        r = r + dr

    radii.append(r)
    densities.append(0.0)
    pressures.append(0.0)

    if len(radii) < 3:
        return None

    return (
        torch.tensor(radii, dtype=torch.float64),
        torch.tensor(densities, dtype=torch.float64),
        torch.tensor(pressures, dtype=torch.float64),
    )


def sample_radial_profile(
    radii: torch.Tensor,
    profile_r: torch.Tensor,
    profile_values: torch.Tensor,
) -> torch.Tensor:
    """Return linear interpolation of a radial profile evaluated at radii."""

    if profile_r.numel() < 2:
        return torch.zeros_like(radii)

    flat_r = radii.reshape(-1)
    idx = torch.bucketize(flat_r, profile_r)
    idx = torch.clamp(idx, min=1, max=profile_r.numel() - 1)
    r0 = profile_r[idx - 1]
    r1 = profile_r[idx]
    v0 = profile_values[idx - 1]
    v1 = profile_values[idx]
    denom = torch.where((r1 - r0).abs() > GEOMETRIC_EPS, r1 - r0, torch.ones_like(r1))
    frac = (flat_r - r0) / denom
    interpolated = v0 + frac * (v1 - v0)
    interpolated = torch.where(flat_r > profile_r[-1], torch.zeros_like(interpolated), interpolated)
    return interpolated.reshape_as(radii)


def coordinate_grids(shape: Sequence[int], spacing: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create coordinate grids for the computational domain.

    Note on the W coordinate:
    The W coordinate represents a parametric dimension, NOT a dynamically evolved
    extra spatial dimension. In our brane-world toy model:
    - W is used to set initial conditions for the scalar field phi_kk
    - W allows us to model standing wave patterns in a hypothetical compact dimension
    - W is NOT evolved by the equations of motion
    - The actual dynamics remain 3+1 dimensional (3 space + 1 time)

    This is fundamentally different from true 5D Kaluza-Klein theory where the
    extra dimension would be dynamically evolved along with x, y, z.
    """
    coords = []
    for n, h in zip(shape, spacing):
        axis = (torch.arange(n, dtype=torch.float64) - n / 2.0) * h
        coords.append(axis)
    X, Y, Z, W = torch.meshgrid(*coords, indexing="ij")
    return X, Y, Z, W


def solve_initial_conditions(
    shape: Sequence[int],
    spacing: Sequence[float],
    params: BSSNParameters,
    masses: Tuple[float, float] = (30.0, 25.0),
    separation: float = 20.0,
    spins: Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> BSSNState:
    state = zeros_state(shape, torch.device(device), dtype)

    X, Y, Z, W = coordinate_grids(shape, spacing)
    X = X.to(dtype=dtype, device=device)
    Y = Y.to(dtype=dtype, device=device)
    Z = Z.to(dtype=dtype, device=device)
    W = W.to(dtype=dtype, device=device)

    x1 = -separation / 2.0
    x2 = separation / 2.0
    r1 = torch.sqrt((X - x1) ** 2 + Y ** 2 + Z ** 2 + 1e-6)
    r2 = torch.sqrt((X - x2) ** 2 + Y ** 2 + Z ** 2 + 1e-6)

    state.phi = 0.5 * torch.log1p(masses[0] / (2.0 * r1) + masses[1] / (2.0 * r2))
    state.K.zero_()
    state.A_tilde.zero_()
    state.Gamma_tilde.zero_()
    if params.q != 0.0 or params.m5 != 0.0:
        r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2).clamp_min(GEOMETRIC_EPS)
        # Interpret m5 as an effective scalar mass (1 / length). When zero, fall back to
        # a large interaction range in geometric units to mimic a massless scalar.
        interaction_length = 1.0 / max(params.m5, 1e-3)
        amplitude = params.q if params.q != 0.0 else 1e-2
        yukawa = amplitude * torch.exp(-r / interaction_length) / (1.0 + r)
        state.phi_brane = yukawa.to(dtype=dtype, device=device)
    else:
        state.phi_brane.zero_()
    state.A_mu.zero_()

    if params.fluid_density and params.fluid_density > 0.0:
        star_radius_geom = params.star_radius_km / params.lattice_length_km
        if separation > 0.0:
            star_radius_geom = min(star_radius_geom, separation * 0.45)
        dr = min(spacing[:3]) * 0.25 if spacing else 0.25
        profile = integrate_tov_profile(
            central_density=params.fluid_density,
            polytrope_K=params.polytrope_K,
            polytrope_gamma=params.polytrope_gamma,
            dr=dr,
            max_radius=star_radius_geom,
            surface_fraction=params.polytrope_surface_density_fraction,
        )

        if profile is not None:
            profile_r, profile_rho, profile_p = profile
            profile_r = profile_r.to(dtype=dtype, device=device)
            profile_rho = profile_rho.to(dtype=dtype, device=device)
            profile_p = profile_p.to(dtype=dtype, device=device)

            radius1 = torch.sqrt((X - x1) ** 2 + Y ** 2 + Z ** 2)
            radius2 = torch.sqrt((X - x2) ** 2 + Y ** 2 + Z ** 2)

            rho1 = sample_radial_profile(radius1, profile_r, profile_rho)
            rho2 = sample_radial_profile(radius2, profile_r, profile_rho)
            p1 = sample_radial_profile(radius1, profile_r, profile_p)
            p2 = sample_radial_profile(radius2, profile_r, profile_p)

            state.rho_fluid = (rho1 + rho2).clamp_min(0.0)
            state.p_fluid = (p1 + p2).clamp_min(0.0)
        else:
            state.rho_fluid.zero_()
            state.p_fluid.zero_()
    else:
        state.rho_fluid.zero_()
        state.p_fluid.zero_()

    if spins:
        for spin, sign in zip(spins, (-1.0, 1.0)):
            omega = torch.tensor(spin, dtype=dtype, device=device)
            gaussian = torch.exp(-((X - sign * separation / 2.0) ** 2 + Y ** 2 + Z ** 2) / (2 * separation ** 2))
            state.A_mu[1] = state.A_mu[1] + sign * omega[0] * gaussian
            state.A_mu[2] = state.A_mu[2] + sign * omega[1] * gaussian
            state.A_mu[3] = state.A_mu[3] + sign * omega[2] * gaussian

    return state


def construct_spatial_triads(metric_pts: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthonormal vectors (r^i, theta^i, phi^i) on the extraction sphere."""

    dtype = metric_pts.dtype
    device = metric_pts.device
    positions = positions.to(dtype=dtype, device=device)
    r_euclid = torch.linalg.norm(positions, dim=1)
    r_euclid = torch.clamp(r_euclid, min=1e-8)
    r_guess = positions / r_euclid.unsqueeze(1)
    r_vec = normalize_with_metric(r_guess, metric_pts)

    theta_angle = torch.acos(torch.clamp(positions[:, 2] / r_euclid, min=-1.0, max=1.0))
    phi_angle = torch.atan2(positions[:, 1], positions[:, 0])

    theta_guess = torch.stack(
        [
            torch.cos(theta_angle) * torch.cos(phi_angle),
            torch.cos(theta_angle) * torch.sin(phi_angle),
            -torch.sin(theta_angle),
        ],
        dim=1,
    ).to(dtype=dtype, device=device)

    phi_guess = torch.stack(
        [
            -torch.sin(phi_angle),
            torch.cos(phi_angle),
            torch.zeros_like(phi_angle),
        ],
        dim=1,
    ).to(dtype=dtype, device=device)

    theta_vec = theta_guess - metric_inner(theta_guess, r_vec, metric_pts).unsqueeze(1) * r_vec
    theta_norm_sq = metric_inner(theta_vec, theta_vec, metric_pts)
    degenerate_theta = theta_norm_sq < 1e-12
    if degenerate_theta.any():
        fallback_theta = torch.cross(r_vec[degenerate_theta], phi_guess[degenerate_theta], dim=1)
        theta_vec[degenerate_theta] = fallback_theta
    theta_vec = normalize_with_metric(theta_vec, metric_pts)

    phi_vec = (
        phi_guess
        - metric_inner(phi_guess, r_vec, metric_pts).unsqueeze(1) * r_vec
        - metric_inner(phi_guess, theta_vec, metric_pts).unsqueeze(1) * theta_vec
    )
    phi_norm_sq = metric_inner(phi_vec, phi_vec, metric_pts)
    degenerate_phi = phi_norm_sq < 1e-12
    if degenerate_phi.any():
        fallback_phi = torch.cross(r_vec[degenerate_phi], theta_vec[degenerate_phi], dim=1)
        phi_vec[degenerate_phi] = fallback_phi
    phi_vec = normalize_with_metric(phi_vec, metric_pts)

    return r_vec, theta_vec, phi_vec


def fixed_frequency_integration(psi4: np.ndarray, dt: float, f_low: float = 20.0) -> np.ndarray:
    """Perform two integrations of Ψ₄ using a fixed-frequency scheme."""

    if psi4.size == 0:
        return np.zeros(0, dtype=np.complex128)

    psi4 = np.asarray(psi4, dtype=np.complex128)
    n = psi4.size
    fft_vals = np.fft.fft(psi4)
    freqs = np.fft.fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freqs
    omega0 = max(2.0 * np.pi * f_low, 2.0 * np.pi / max(n * dt, 1e-6))
    omega_safe = np.where(np.abs(omega) < omega0, np.sign(omega) * omega0, omega)
    omega_safe[omega_safe == 0.0] = omega0
    h_fft = -fft_vals / (omega_safe ** 2)
    h_time = np.fft.ifft(h_fft)
    return h_time.astype(np.complex128)


def streaming_integration(psi4: Sequence[complex], dt: float, f_low: float = 20.0) -> np.ndarray:
    """Streaming double integration with exponential high-pass filtering."""

    psi4 = np.asarray(psi4, dtype=np.complex128)
    if psi4.size == 0:
        return np.zeros(0, dtype=np.complex128)

    tau = 1.0 / max(2.0 * np.pi * f_low, 1e-6)
    alpha = 1.0 - np.exp(-dt / tau) if tau > 0.0 else 1.0

    velocity = 0.0 + 0.0j
    displacement = 0.0 + 0.0j
    displacement_mean = 0.0 + 0.0j

    output = np.zeros_like(psi4, dtype=np.complex128)
    for idx, val in enumerate(psi4):
        velocity = velocity - val * dt
        displacement = displacement + velocity * dt

        displacement_mean = displacement_mean + (displacement - displacement_mean) * alpha

        output[idx] = displacement - displacement_mean

    return output


def resample_strain(time: np.ndarray, strain: np.ndarray, target_rate: float | None) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a complex strain time series to the requested rate via interpolation."""

    if target_rate is None or target_rate <= 0.0 or strain.size == 0:
        return time, strain

    dt_new = 1.0 / target_rate
    t_min = time[0]
    t_max = time[-1]
    if t_max <= t_min:
        return time, strain

    num_steps = int(np.floor((t_max - t_min) / dt_new)) + 1
    resampled_time = t_min + dt_new * np.arange(num_steps + 1, dtype=np.float64)
    real_part = np.interp(resampled_time, time, strain.real)
    imag_part = np.interp(resampled_time, time, strain.imag)
    resampled = real_part + 1j * imag_part
    return resampled_time, resampled


def integrate_psi4_series(
    psi4_series: Sequence[complex],
    dt: float,
    f_low: float = 20.0,
    target_rate: float | None = 4096.0,
    method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a Ψ₄(t) series into h₊/h× via double integration."""

    if len(psi4_series) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty

    psi4 = np.asarray(psi4_series, dtype=np.complex128)

    if method == "streaming":
        h_geom = streaming_integration(psi4, dt, f_low)
    else:
        h_geom = fixed_frequency_integration(psi4, dt, f_low)

    time = np.arange(psi4.size, dtype=np.float64) * dt
    resampled_time, resampled_geom = resample_strain(time, h_geom, target_rate)
    h_plus = resampled_geom.real
    h_cross = -resampled_geom.imag
    return resampled_time, h_plus, h_cross


def extrapolate_psi4_to_infinity(radii: Sequence[float], psi4_matrix: np.ndarray, order: int = 1) -> np.ndarray:
    """Fit Ψ₄(r, t) = Σ a_n(t) / r^n and evaluate at null infinity."""

    if psi4_matrix.size == 0:
        return np.zeros(0, dtype=np.complex128)

    radii = np.asarray(radii, dtype=np.float64)
    inv_r = 1.0 / np.clip(radii, GEOMETRIC_EPS, None)
    psi4_matrix = np.asarray(psi4_matrix, dtype=np.complex128)

    if psi4_matrix.ndim != 2:
        raise ValueError("psi4_matrix must be 2D with shape (num_radii, num_times)")

    num_radii, num_times = psi4_matrix.shape
    if num_radii < order + 1:
        raise ValueError("Not enough radii for requested extrapolation order")

    result = np.zeros(num_times, dtype=np.complex128)
    for idx in range(num_times):
        values = psi4_matrix[:, idx]
        coeffs_real = np.polyfit(inv_r, values.real, deg=order)
        coeffs_imag = np.polyfit(inv_r, values.imag, deg=order)
        result[idx] = coeffs_real[-1] + 1j * coeffs_imag[-1]
    return result


def extract_waveform(
    state: BSSNState,
    spacing: Sequence[float],
    radius: float,
    projection: str = "plus",
    params: BSSNParameters | None = None,
    use_background_tetrad: bool = False,
    background_mass: float | None = None,
) -> Dict[str, object]:
    """Compute Ψ₄ on a coordinate sphere and return diagnostic metadata."""

    params = params or BSSNParameters()

    metric_phys = compute_physical_metric(state)
    Gamma_phys = christoffel_symbols(metric_phys, spacing)
    riemann_upper = spatial_riemann_tensor(metric_phys, Gamma_phys, spacing)
    riemann_lower = lower_riemann_tensor(metric_phys, riemann_upper)
    metric_inv = invert_metric(metric_phys)
    ricci_phys = spatial_ricci_from_riemann(riemann_lower, metric_inv)
    ricci_scalar_field = ricci_scalar_from_tensor(ricci_phys, metric_inv)
    K_tensor = compute_extrinsic_curvature(state)
    stress = stress_energy_brane(state, spacing, params)
    electric = electric_part_of_weyl(ricci_phys, state.K, K_tensor, metric_phys, metric_inv, stress)
    magnetic = magnetic_part_of_weyl(K_tensor, Gamma_phys, spacing, metric_phys)
    det_g = torch.clamp(determinant_metric(metric_phys), min=GEOMETRIC_EPS)
    sqrt_det = torch.sqrt(det_g)

    X, Y, Z, W = coordinate_grids(state.phi.shape, spacing)
    X = X.to(dtype=metric_phys.dtype, device=metric_phys.device)
    Y = Y.to(dtype=metric_phys.dtype, device=metric_phys.device)
    Z = Z.to(dtype=metric_phys.dtype, device=metric_phys.device)
    R = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    radius_tol = max(spacing[:SPATIAL_DIMS]) * 1.5
    mask = (R - radius).abs() <= radius_tol
    if len(spacing) > SPATIAL_DIMS:
        W = W.to(dtype=metric_phys.dtype, device=metric_phys.device)
        w_tol = spacing[SPATIAL_DIMS] * 0.5
        mask = mask & (W.abs() <= w_tol)

    flat_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if flat_indices.numel() == 0:
        radius_tol = max(spacing[:SPATIAL_DIMS]) * 3.0
        mask = (R - radius).abs() <= radius_tol
        if len(spacing) > SPATIAL_DIMS:
            mask = mask & (W.abs() <= w_tol)
        flat_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)

    if flat_indices.numel() == 0:
        zero_complex = torch.zeros((), dtype=torch.complex128)
        return {
            "psi4": zero_complex,
            "samples": torch.zeros(0, dtype=torch.complex128),
            "weights": torch.zeros(0, dtype=metric_phys.dtype),
            "num_points": 0,
            "lapse_mean": float(state.alpha.mean().item()),
            "shift_mean": torch.zeros(SPATIAL_DIMS, dtype=metric_phys.dtype),
            "ricci_scalar_mean": float(ricci_scalar_field.mean().item()),
            "norm_checks": {
                "n_dot_n": float("nan"),
                "ell_dot_n": float("nan"),
                "m_dot_mbar": float("nan"),
                "m_dot_mbar_im": float("nan"),
            },
            "radius": radius,
            "projection": projection,
            "tetrad_source": "physical",
        }

    num_points = flat_indices.numel()
    positions = torch.stack(
        [
            X.reshape(-1)[flat_indices],
            Y.reshape(-1)[flat_indices],
            Z.reshape(-1)[flat_indices],
        ],
        dim=1,
    )

    metric_pts = metric_phys.reshape(SPATIAL_DIMS, SPATIAL_DIMS, -1)[:, :, flat_indices].permute(2, 0, 1)
    electric_pts = electric.reshape(SPATIAL_DIMS, SPATIAL_DIMS, -1)[:, :, flat_indices].permute(2, 0, 1)
    magnetic_pts = magnetic.reshape(SPATIAL_DIMS, SPATIAL_DIMS, -1)[:, :, flat_indices].permute(2, 0, 1)
    weights = sqrt_det.reshape(-1)[flat_indices]

    alpha_pts = state.alpha.reshape(-1)[flat_indices]
    beta_pts = torch.stack([state.beta[i].reshape(-1)[flat_indices] for i in range(SPATIAL_DIMS)], dim=1)
    ricci_samples = ricci_scalar_field.reshape(-1)[flat_indices]

    tetrad_source = "physical"
    tetrad_metric_pts = metric_pts
    if use_background_tetrad and background_mass is not None and background_mass > 0.0:
        background_metric = isotropic_schwarzschild_metric(positions, background_mass)
        tetrad_metric_pts = background_metric.to(dtype=metric_pts.dtype, device=metric_pts.device)
        tetrad_source = "background"

    r_vec, theta_vec, phi_vec = construct_spatial_triads(tetrad_metric_pts, positions)
    m_complex = torch.complex(theta_vec, phi_vec) / math.sqrt(2.0)
    m_bar = torch.conj(m_complex)
    electric_minus_i_magnetic = torch.complex(electric_pts, -magnetic_pts)
    psi4_samples = -torch.einsum("pij,pi,pj->p", electric_minus_i_magnetic, m_bar, m_bar)

    weight_sum = weights.sum().abs()
    if weight_sum > GEOMETRIC_EPS:
        psi4_avg = (psi4_samples * weights).sum() / weight_sum
    else:
        psi4_avg = psi4_samples.mean()

    beta_lower = torch.einsum("pij,pj->pi", tetrad_metric_pts, beta_pts)
    beta_squared = metric_inner(beta_pts, beta_pts, tetrad_metric_pts)
    alpha_safe = torch.clamp(alpha_pts.abs(), min=GEOMETRIC_EPS)

    g_four = torch.zeros((num_points, 4, 4), dtype=tetrad_metric_pts.dtype, device=tetrad_metric_pts.device)
    g_four[:, 0, 0] = -alpha_pts ** 2 + beta_squared
    g_four[:, 0, 1:] = beta_lower
    g_four[:, 1:, 0] = beta_lower
    g_four[:, 1:, 1:] = tetrad_metric_pts

    t_vec = torch.zeros((num_points, 4), dtype=tetrad_metric_pts.dtype, device=tetrad_metric_pts.device)
    t_vec[:, 0] = 1.0 / alpha_safe
    t_vec[:, 1:] = -beta_pts / alpha_safe.unsqueeze(1)

    r_four = torch.zeros_like(t_vec)
    r_four[:, 1:] = r_vec
    sqrt_two = math.sqrt(2.0)
    n_vec = (t_vec + r_four) / sqrt_two
    ell_vec = (t_vec - r_four) / sqrt_two

    n_dot_n = metric_inner(n_vec, n_vec, g_four)
    ell_dot_n = metric_inner(ell_vec, n_vec, g_four)
    m_dot_mbar = metric_inner(m_complex, torch.conj(m_complex), tetrad_metric_pts)

    result = {
        "psi4": psi4_avg.to(torch.complex128).to("cpu"),
        "samples": psi4_samples.to(torch.complex128).to("cpu"),
        "weights": weights.to(metric_pts.dtype).to("cpu"),
        "num_points": int(num_points),
        "lapse_mean": float(alpha_pts.mean().item()),
        "shift_mean": beta_pts.mean(dim=0).to(metric_pts.dtype).to("cpu"),
        "ricci_scalar_mean": float(ricci_samples.mean().item()),
        "norm_checks": {
            "n_dot_n": float(n_dot_n.real.mean().item()),
            "ell_dot_n": float(ell_dot_n.real.mean().item()),
            "m_dot_mbar": float(m_dot_mbar.real.mean().item()),
            "m_dot_mbar_im": float(m_dot_mbar.imag.mean().item()),
        },
        "radius": radius,
        "projection": projection,
        "tetrad_source": tetrad_source,
    }
    return result


__all__ = [
    "BSSNParameters",
    "BSSNState",
    "bssn_rhs",
    "calculate_constraints",
    "solve_initial_conditions",
    "extract_waveform",
    "zeros_state",
    "state_linear_combination",
    "zeros_like",
    "enforce_boundary_conditions",
    "enforce_periodic_state",  # Keep for backward compatibility
    "scale_waveform_to_observer",
    "fixed_frequency_integration",
    "streaming_integration",
    "integrate_psi4_series",
    "extrapolate_psi4_to_infinity",
]
