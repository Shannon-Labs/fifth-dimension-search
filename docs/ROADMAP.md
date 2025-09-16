# Roadmap

This document expands on the roadmap bullets in the README and tracks medium-term goals. Pair it with [`THEORY_FRAMEWORK.md`](THEORY_FRAMEWORK.md) for the latest decision on which physics model we are targeting. Feel free to open a pull request to update the roadmap as you make progress.

## Stabilise the Numerical Core

- [x] Audit the strain scaling in `brane_world.py` and add unit/CLI smoke coverage for the corrected conversion.
- [x] Commit to the scalar–tensor (Option B) upgrade path and capture the decision in [`THEORY_FRAMEWORK.md`](THEORY_FRAMEWORK.md).
- [ ] Replace the ad-hoc scalar field evolution with an implementation consistent with the chosen theory.
- [ ] Add convergence tests that exercise the real evolution code (the current `convergence_test.py` is a deterministic placeholder).

## Tooling and Developer Experience

- [x] Introduce a `pytest`-based unit test suite targeting conversion helpers, waveform extraction, and CLI commands.
- [ ] Provide GitHub Actions workflows for linting, tests, and documentation builds once the test suite is stable.
- [ ] Package example notebooks demonstrating best practices for analysing simulation outputs.

## Data & Visualisation

- [ ] Curate a small library of validated benchmark waveforms (numerically simple but physically meaningful) with metadata on their limitations.
- [ ] Expand the CLI with comparison utilities (`fds compare --baseline …`).
- [ ] Offer interactive visualisations (Altair/Plotly) that highlight deviations from General Relativity.

## Community & Documentation

- [x] Capture the theoretical background in a living design document (see [`THEORY_FRAMEWORK.md`](THEORY_FRAMEWORK.md)); continue expanding with peer-reviewed references.
- [ ] Collect starter issues suitable for students or first-time contributors.
- [ ] Host regular office hours or recorded walkthroughs once there is sufficient interest.

If you want to lead any of these items—or propose a new direction—open an issue so we can coordinate. The roadmap is intentionally flexible.
