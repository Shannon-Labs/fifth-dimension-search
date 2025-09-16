# Scalar–Tensor Sandbox Notes

_This note captures the working equations and assumptions for the scalar–tensor upgrade path (Option B). It is a living document—add references or derivations as they become available._

## 1. Action and Field Content

We adopt a Jordan-frame action of the form

\[
S = \frac{1}{16 \pi} \int d^4x \sqrt{-g} \left[ \Phi R - \frac{\omega(\Phi)}{\Phi} (\nabla \Phi)^2 - 2 V(\Phi) \right] + S_\text{matter}[g_{\mu\nu}, \Psi] ,
\]

where:

- \(\Phi\) is the effective scalar field (implemented numerically as `phi_brane`).
- \(\omega(\Phi)\) controls the coupling strength; we currently assume it is constant and expose it as `BSSNParameters.scalar_tensor_omega`.
- \(V(\Phi)\) maps onto the mass parameter `m5` through a simple quadratic potential \(V = \frac{1}{2} m_\phi^2 (\Phi - \Phi_0)^2\).

For now we linearise around \(\Phi_0 = 1\) so that the Einstein equations reduce to:

\[
G_{\mu\nu} = 8 \pi (T_{\mu\nu}^{\text{matter}} + T_{\mu\nu}^{\Phi})
\]

with

\[
T_{\mu\nu}^{\Phi} = \frac{\omega}{\Phi^2} \left( \nabla_{\mu} \Phi \nabla_{\nu} \Phi - \frac{1}{2} g_{\mu\nu} (\nabla \Phi)^2 \right) + \frac{1}{\Phi} \left( \nabla_{\mu} \nabla_{\nu} \Phi - g_{\mu\nu} \Box \Phi \right) - g_{\mu\nu} V(\Phi).
\]

## 2. 3+1 Split in the BSSN Formalism

Under the conformal/BSSN decomposition, the scalar field contributes additional source terms:

- Energy density: \(\rho_\Phi = n^\mu n^\nu T_{\mu\nu}^{\Phi}\)
- Momentum density: \(S_i^\Phi = -\gamma_{i}^{\ \mu} n^{\nu} T_{\mu\nu}^{\Phi}\)
- Stress tensor: \(S_{ij}^\Phi = \gamma_i^{\ \mu} \gamma_j^{\ \nu} T_{\mu\nu}^{\Phi}\)

The current implementation (`stress_energy_brane`) already follows this structure. The next step is to substitute the explicit scalar–tensor expressions above and audit each term for dimensional consistency.

## 3. Scalar Field Evolution

The scalar obeys a damped wave equation sourced by the trace of the matter stress tensor:

\[
\Box \Phi = \frac{1}{2\omega + 3} \left(8\pi T - \frac{d\omega}{d\Phi} (\nabla \Phi)^2 + 2 \Phi \frac{dV}{d\Phi} - 4 V \right).
\]

With constant \(\omega\) and quadratic \(V\), this simplifies to

\[
\Box \Phi = \frac{8\pi}{2\omega + 3} T - m_\phi^2 (\Phi - \Phi_0).
\]

In the code we implement this via the Yukawa-style initial data and the evolution terms inside `bssn_rhs`. Future work: derive the precise factors of lapse/shift for the BSSN form and hook them into the RHS routine.

## 4. Initial Data Prescription

To remain consistent with the scalar–tensor interpretation we:

1. Treat `m5` as the effective mass \(m_\phi\), so the interaction length is \(\lambda = 1/m_\phi\).
2. Seed the scalar with a spherically symmetric Yukawa profile \(\Phi - \Phi_0 \propto e^{-r/\lambda} / (1 + r)\), matching the changes in `solve_initial_conditions`.
3. Avoid any dependence on the auxiliary `W` coordinate; it is retained only for backward compatibility with the previous dataset.

## 5. Validation Checklist

- [ ] Derive explicit source terms for \(\rho_\Phi\), \(S_i^\Phi\), and \(S_{ij}^\Phi\) and cross-check with `stress_energy_brane`.
- [ ] Implement a spectral or analytic solution (e.g., static Yukawa field in flat space) and compare against the numerical data.
- [ ] Benchmark the scalar dynamics against a known Jordan–Brans–Dicke solution in the weak-field limit.

✅ **Progress:** `tests/test_scalar_tensor.py` verifies that the new Yukawa initial data matches the analytic profile at the percent level.

Contributions welcome—append derivations or references directly to this file.
