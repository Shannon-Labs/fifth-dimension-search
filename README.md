# 🧪 Fifth Dimension Research Sandbox

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Exploratory research sandbox. The bundled models are broken on purpose so the community can help us fix them.**

## Project Status at a Glance

- 🔬 **Toy model only** – the code evolves a brane-inspired scalar field coupled to 3+1 GR. No real 5D Kaluza–Klein physics is implemented yet.
- 📉 **Unphysical results** – waveform amplitudes and phase shifts are orders of magnitude off. Use the data as regression fixtures, not scientific evidence.
- 🧠 **Collaboration-first** – everything is structured so new contributors can spin up quickly, understand the shortcomings, and propose improvements.
- 🔁 **Transparent failure log** – see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and [`FINDINGS.md`](FINDINGS.md) for the candid list of what is wrong today.

If you have expertise in numerical relativity, modified gravity, detector modelling, or just want to help clean up the code, **you are in the right place**.

## Quick Start

```bash
# 1. Clone the sandbox
git clone https://github.com/shannon-labs/fifth-dimension-search.git
cd fifth-dimension-search

# 2. Install in editable mode (includes CLI and plotting extras)
pip install -e .[plot]

# 3. Inspect the current toy data summary
fds info

# 4. Generate the baseline diagnostic plot
fds plot --output artifacts/toy_diagnostics.png

# 5. Run the deterministic convergence toy example
python convergence_test.py
```

The CLI is intentionally lightweight so it can run on laptops without GPUs. When you are ready to dive deeper, look at `src/fifth_dimension_search/` for the full brane-world toy evolution code.

## Visualising the Bundled Toy Data

Two quick options are provided out of the box:

- `fds plot` – CLI command that writes `artifacts/toy_diagnostics.png` summarising the packaged CSV tables.
- `python tools/quick_analysis.py` – prints a transparent textual summary and produces the same diagnostic figure.

The generated plot makes it obvious how far the current toy results are from detector sensitivity, which is the whole point of the sandbox.

## Repository Layout

```
├── src/fifth_dimension_search/
│   ├── brane_world.py          # 3+1 BSSN + brane scalar toy implementation
│   ├── template_bank.py        # Minimal template generation helpers
│   ├── analysis.py             # Helpers for inspecting packaged results
│   ├── visualisation.py        # Shared plotting routines
│   └── datasets/               # Bundled CSV fixtures for reproducibility
├── tools/                      # CLI-friendly scripts and plotting utilities
├── convergence_test.py         # Deterministic convergence demonstration
├── KNOWN_ISSUES.md             # Honest list of physics/numerical failures
├── FINDINGS.md                 # What we learned (and where we failed)
└── docs/                       # Onboarding, roadmap, and theory decision logs
```

## How to Contribute

1. **Read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** – short guide covering branching, testing, and coding style.
2. **Skim [`docs/THEORY_FRAMEWORK.md`](docs/THEORY_FRAMEWORK.md)** – understand the brane-scalar sandbox scope before proposing physics changes.
3. Pick an item from [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) or the open issues tab and start a discussion. Even confirming existing bugs is valuable.
4. Use `fds datasets list` and `fds datasets show <name>` to inspect the shared fixtures, then build your improvement in a dedicated branch.
5. Submit a pull request with before/after plots or numbers. We value transparent failure analysis over unverified claims.

Not sure where to begin? Check the **starter issues** section in the contributing guide for ideas ranging from documentation clean-up to designing unit tests around the current conversion utilities.

## Roadmap Highlights

- ✅ Reorganised into an installable package with a contributor-friendly CLI
- ✅ Bundled deterministic visualisations and convergence examples
- ⏳ Implement physically consistent strain scaling and unit conversions
- ⏳ Decide on a single extra-dimensional theory to pursue (KK vs brane-world vs scalar-tensor)
- ⏳ Add automated tests and CI once the numerical core stabilises

If you want to champion any of the ⏳ items—or propose new ones—open an issue and we will happily collaborate.

## Acknowledgements & Licence

Shared under the [MIT Licence](LICENSE). Massive thanks to everyone willing to help turn this broken idea into a rigorous research project. Let’s make future gravitational-wave catalogues extra-dimensional ready!
