# Theory Framework Decision Log

_Last updated: 2025-09-16_

## Summary

We are intentionally repositioning the sandbox as a **brane-inspired scalar extension of 3+1 General Relativity** while we work towards a more rigorous extra-dimensional model. The previous README and commit messages oscillated between "Kaluza–Klein", "brane world", and generic modified gravity claims, which made it difficult for contributors to understand what they were fixing. This document records the current consensus and outlines possible upgrade paths.

## Current Baseline (v0.1)

- **Geometry**: Standard 3+1 BSSN formulation with conformal variables.
- **Extra Field**: A single scalar `phi_brane` confined to the 3D hypersurface, with optional standing-wave dependence on a parametric `W` coordinate used only for initial data.
- **Gauge Sector**: Toy electromagnetic-like vector potential `A_mu` coupled to the scalar via a millicharge `q`.
- **Matter**: Simple polytropic fluid primarily used to stabilise neutron-star-like initial conditions.
- **Wave Extraction**: Standard `Ψ₄` extraction with spherical averaging; new strain-scaling fix brings amplitudes to ≈10⁻²¹ at 40 Mpc.

> 📌 **Naming convention**: We refer to this setup as the *brane scalar sandbox* in docs and code comments. Avoid calling it "Kaluza–Klein" unless the extra dimension is dynamically evolved.

## Near-Term Goals

1. **Decide on the next physics upgrade**
   - Option A: Promote the scalar to a bona fide 5D metric degree of freedom (true KK).
   - Option B: Embrace a 4D scalar–tensor theory and drop extra-dimensional language.
   - Option C: Implement a Randall–Sundrum-style brane with Israel junction conditions.
2. **Document required equations of motion** for the chosen path, including constraints and gauge conditions.
3. **Assess computational feasibility** of the selected model (grid structure, boundary conditions, GPU requirements).

## Decision (2025-09-16)

We are committing the collaboration to **Option B: developing a consistent 4D scalar–tensor model** as the next milestone. The rationale:

- It keeps the numerical infrastructure close to the existing BSSN + scalar code.
- It avoids over-claiming 5D physics while still letting us explore modified gravity phenomenology.
- It provides a concrete path to validation against well-studied Jordan–Brans–Dicke–like systems.

### Immediate action items

- Draft a technical note deriving the scalar–tensor equations in BSSN variables (assigned: open issue `#theory-derivation`).
- Replace the ad-hoc standing-wave initial data with profiles consistent with the chosen scalar–tensor theory.
- Audit uses of the parametric `W` coordinate and document whether it remains necessary; if not, plan its removal.

Option A (full 5D) stays on the long-term roadmap once we have confidence in the scalar–tensor baseline.

## Guidelines for Contributors

- When opening issues or PRs, explicitly state which framework your proposal targets.
- If you prototype equations beyond the current scalar toy, place them under `experiments/` or a draft branch until we agree on the theory scope.
- Reference this decision log in PR descriptions to keep the terminology aligned, and consult [`scalar_tensor_notes.md`](scalar_tensor_notes.md) for the evolving set of equations.

## Open Questions

- How do we validate the scalar sector against any published brane-world calculations?
- What observational signatures are even measurable once the theory is consistent?
- How should we treat the parametric `W` coordinate if we remain in 3+1 dimensions?

If you want to champion one of the upgrade paths, comment on the GitHub issue `#framework-roadmap` (to be opened) or start a discussion in `docs/ROADMAP.md`.
