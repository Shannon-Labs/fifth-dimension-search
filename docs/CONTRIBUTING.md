# Contributing Guide

Thanks for your interest in helping with the Fifth Dimension Research Sandbox! This project is intentionally open, imperfect, and collaborative. The code currently produces unphysical results; our goal is to turn the sandbox into a rigorous extra-dimensional modelling playground step by step.

## Getting Set Up

1. **Fork and clone** the repository.
2. Create a virtual environment and install the package in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[plot]
   ```
3. Run a few quick diagnostics to make sure your environment works:
   ```bash
   fds info
   python tools/quick_analysis.py --output artifacts/quick.png
   python convergence_test.py
   ```

## Development Workflow

- Create a feature branch from `main` (`git checkout -b fix-strain-scaling`).
- Keep pull requests focused. If you are fixing strain scaling and improving docs, submit them separately.
- Update or add tests for the functionality you touch. Lightweight tests live in `tests/` (coming soon) and we also accept executable documentation (Jupyter/Quarto) as supplementary evidence.
- Run formatting tools before committing if you install them locally (`black` with `line-length 100` and `isort` with the Black profile).
- Document any new command-line options or scripts in the README or `docs/`.

## Filing Issues

Useful issue reports include:
- The command you ran and the environment (Python version, CPU/GPU availability).
- What you expected to happen.
- What actually happened, including stack traces.
- Any specific physics references or requirements relevant to the change.

If you are proposing a feature, please outline:
- Why the feature is necessary.
- How it fits within the roadmap (even if it updates the roadmap).
- Any dependencies or potential blockers.

## Pull Request Checklist

Before submitting:

- [ ] Code is formatted and linted locally.
- [ ] New or modified functionality is covered by tests or a reproducible notebook.
- [ ] Docs/README snippets explain how to use the change.
- [ ] Large data files are not committed. Instead, add download instructions.
- [ ] PR description references relevant issues or motivates the change clearly.

## Starter Contributions

Not sure where to start? Consider tackling one of the following:

- **Unit conversion sanity checks** – add tests for the conversion utilities in `brane_world.py`.
- **Waveform visualisations** – improve `create_toy_diagnostics_plot` or design a new diagnostic view.
- **Physics notes** – help us document what a correct 5D implementation should include.
- **Infrastructure** – wire up GitHub Actions to run smoke tests and linting once the suite stabilises.

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). This is a friendly citizen-science project—please be respectful.

Happy hacking! Together we can transform this broken toy model into a robust research tool.
