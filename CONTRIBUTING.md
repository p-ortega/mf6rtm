# Contributing to mf6rtm

Thanks for your interest in mf6rtm. This page explains how to report a problem, ask a question, and contribute code or documentation.

## Report a bug

Open an issue at https://github.com/p-ortega/mf6rtm/issues. Please include:

- the mf6rtm version (`python -c "import mf6rtm; print(mf6rtm.__version__)"`), your OS, and how you installed it (pip or pixi);
- what you ran and what happened, with the full error message;
- what you expected to happen;
- if possible, a small model that reproduces the problem. A cut-down script or the model folder (PHREEQC input, database, and MODFLOW 6 files) is the fastest way to a fix.

If a model runs but the results look wrong, say how you checked them (for example, against PHREEQC, PHT3D, or measured data).

## Ask a question

Open an issue and start the title with `Question:`. Questions about setting up a model, choosing PHREEQC options, or linking mf6rtm to PEST++ are welcome. The documentation at https://mf6rtm.readthedocs.io covers installation, the tutorials, and the API, and may already answer it.

## Suggest a feature

Open an issue describing the use case: what you are trying to model and what is missing. Larger changes are easier to review if we agree on the approach in the issue before you start coding.

## Contribute code or documentation

1. Fork the repository and create a branch from `develop`. `main` holds released versions only.
2. Set up the development environment with [pixi](https://pixi.sh):

   ```commandline
   git clone https://github.com/YOUR-USERNAME/mf6rtm.git
   cd mf6rtm
   pixi install
   ```

   This installs all dependencies and mf6rtm itself in editable mode.
3. Make your change, and add or update tests in `autotest/`. Bug fixes should come with a test that fails without the fix.
4. Run the checks before opening a pull request:

   ```commandline
   pixi run test      # fetches the MODFLOW 6 binaries the first time
   pixi run lint
   pixi run format
   ```

5. Open a pull request into `develop`. Describe what changed and why, and link the issue it addresses.

Some conventions:

- Public functions and classes have numpydoc docstrings, which feed the API pages on the documentation site.
- Changes that affect results should be checked against the benchmarks in `benchmark/` and the benchmark tests in `autotest/test_benchmark.py`. If a reference result has to change, explain why in the pull request.
- Tutorials live in `docs/tutorials/` as jupytext `.py` files. Build the documentation locally with `pixi run build-docs`.

All pull requests are reviewed by a maintainer and must pass CI on Linux, macOS, and Windows before they are merged.

## Code of conduct

Please be respectful and constructive in issues and pull requests. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

