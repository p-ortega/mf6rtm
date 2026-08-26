# AGENTS.md

Guidance for AI coding agents (and human contributors) working in this repo. This file covers
**how to work here** -- packaging, environment, and CI mechanics; for *what the project is* --
scope, science, and installation for end users -- see [README.md](README.md) and
[docs/development.rst](docs/development.rst). When they disagree, this file wins on process,
README wins on scope. Keep this file updated as conventions change -- it's a shared resource, not
a one-time snapshot.

## ⚠️ Critical guardrails -- read before any work

- **Only the user commits and merges -- never the agent.** Do **not** run `git commit`,
  `git merge`, or `git push`. Make and verify changes, leave them **staged / on-disk**, and let
  the user review and commit. Creating a branch (`git checkout -b`) is fine -- never commit
  directly to `main` or `develop`. *This written rule is the only enforcement: there is no hook
  stopping you. Treat it as absolute.*
- **Never dispatch the release workflow.** `.github/workflows/python-publish.yml` creates a real
  git tag and publishes to PyPI the moment it runs -- see [Releasing](#releasing). Only a human
  triggers it, from the GitHub Actions UI, never from a local shell or agent session.
- **Multi-step work pauses for review before each commit.** Implement one coherent change, verify
  it (tests/lint -- see [Commands](#commands)), then stop and leave it staged for the user to
  review and commit -- don't batch multiple unrelated changes into one uncommitted pile, and don't
  commit on your own initiative even when verification passes.

## Packaging: pyproject.toml is the single source of truth

There is no `setup.cfg`, `setup.py`, `pixi.toml`, `.bumpversion.toml`, or `MANIFEST.in` -- all of
it is consolidated into `pyproject.toml`: build backend, runtime/dev/docs dependencies, the pixi
workspace (environments, features, tasks), pytest config, mypy config, and ruff config. If you're
about to create one of those files, stop -- the section you need already has a home in
`pyproject.toml`.

### Dependencies are declared twice, on purpose

`[project.dependencies]` is the authoritative runtime contract in **PyPI names** -- this is what
`pip install mf6rtm` resolves and what a conda-forge recipe is generated from. It should exactly
match what `mf6rtm/` actually imports (not autotest/benchmark/docs) -- check with imports, not
assumptions; this list has drifted from reality before (`numpy`, `pandas`, `pillow`, `pyyaml` were
all used but undeclared until this was audited).

`[tool.pixi.dependencies]` mirrors that same list under **conda-forge names**, plus `sqlite`
(pinned for phreeqcrm's underlying requirement; no meaningful PyPI equivalent -- a documented
exception, not drift). `autotest/test_dependency_parity.py` enforces both directions: every
`[project.dependencies]` entry must appear in `[tool.pixi.dependencies]`, and every
`[tool.pixi.dependencies]` entry must be either a `[project.dependencies]` entry or in that test's
`CONDA_ONLY` allow-list. Add a runtime dependency to one list and not the other, and this test
fails -- that's the point.

Optional features (`hdf5`, `testing`, `docs` in `[project.optional-dependencies]`) don't get this
same enforcement; they're deliberately looser.

### Pixi features don't compose the way you'd guess

Each pixi environment (`default`, `py311`, `py312`, `py313`, `docs`) is an explicit list of
features, and **nothing is shared between them except what's in `[tool.pixi.dependencies]`** (the
implicit default feature, included everywhere) and the `editable` feature (the `mf6rtm`
self-install, added to every environment's feature list so `pixi install` alone gives a working
`import mf6rtm` -- no `pip install -e .` needed).

Concretely: the `dev` feature (pytest, ruff, jupytext, geopandas, matplotlib, ...) is **not**
included in the `docs` environment. If a docs/tutorial notebook needs a package, it has to be
added to `[tool.pixi.feature.docs.dependencies]` explicitly, even if that exact package is already
in `dev`. This bit us three times in one pass: `jupytext`/`nbconvert`/`ipykernel` (needed by
`exec-tutorials`'s `jupytext --execute` step itself), `shapely` (an optional flopy dependency,
lazily imported by flopy's plotting code, which one tutorial exercises), and `pytables` (one
tutorial exercises mf6rtm's `hdf5` output path). Each was only found by actually running
`pixi run -e docs build-docs` locally until it stopped erroring -- don't assume a docs-environment
change is safe just because `pixi install`/`pixi run test` succeed; run `build-docs` too.

`dev`'s own list is intentionally broader than `[project.optional-dependencies].testing`: it also
covers packages `mf6rtm/` never imports but autotest/benchmark do (`geopandas`, `matplotlib`), plus
tools with real PyPI packages that are simply outside mf6rtm's runtime contract, not
PyPI-unavailable like `sqlite` (`modflow-devtools`, `coverage`, `coveralls`, `python-build`,
`grayskull`).

### Wheel packaging needs explicit `force-include` for non-`.py` assets

`[tool.hatch.build.targets.wheel] packages = ["mf6rtm"]` only picks up `*.py` files and
`py.typed` by default -- it silently drops `mf6rtm/assets/*.png`, which
`mf6rtm/assets/__init__.py`'s `mrbeaker_path()` loads at runtime via `importlib.resources`. Found
by actually unzipping a locally-built wheel and checking, not by reasoning about hatchling's
defaults. If you add another non-`.py` file that needs to ship in the wheel, add it to
`[tool.hatch.build.targets.wheel.force-include]` too, and verify with
`python -m build --wheel && python -m zipfile -l dist/*.whl`.

### Jupytext pairing is scoped to `docs/tutorials/` only

`[tool.jupytext] formats = "docs/tutorials//py:percent,docs/tutorials//ipynb"` declares the
`docs/tutorials/*.py` <-> `*.ipynb` pairing explicitly, so `jupytext --sync` and editor
Jupytext-Sync extensions can find it. Before this existed, `jupytext --paired-paths` reported
*nothing* for either file -- the pairing only "worked" because `exec-tutorials` calls a one-way
`jupytext --to notebook --execute`, never relying on stored pairing metadata. The path prefix on
both sides of `formats` scopes this to `docs/tutorials/` deliberately: `benchmark/*.ipynb` have no
paired `.py` and aren't meant to (they're edited directly as notebooks) -- a bare, unscoped
`formats = "ipynb,py:percent"` would incorrectly imply pairing intent there too.

### Lint is scoped to `mf6rtm/` only

`[tool.ruff] include = ["pyproject.toml", "mf6rtm/**/*.py"]` -- deliberately, matching what the
old `flake8 mf6rtm` task covered. `autotest/` and `docs/tutorials/` contain jupytext-paired `.py`
files; running `ruff check --fix`/`ruff format` unscoped over the whole repo can trigger an
editor's Jupytext Sync extension to silently rewrite the paired `.ipynb` on save. If you widen
ruff's scope, check `git status` afterward for unexpected notebook diffs.

### Versioning is git-tag-driven (hatch-vcs)

`mf6rtm/__init__.py` imports `__version__` from `mf6rtm/_version.py`, which is generated at build
time by the hatch-vcs hook -- never hand-edit a version string anywhere. Any commit without a tag
on top of it gets an automatic `X.Y.Z.devN+g<hash>` version; only a real release (see below) gets a
clean `X.Y.Z`. Existing tags have no `v` prefix (`0.5.2`, not `v0.5.2`) -- keep that convention.

### Releasing

Dispatch `.github/workflows/python-publish.yml` with a bare `X.Y.Z` version (no `v` prefix). It
validates the format and that the tag doesn't already exist, creates and pushes the tag, builds,
verifies the built wheel's filename matches, publishes to PyPI (Trusted Publishing), and cuts a
GitHub Release. There's no "just bump for the next cycle" mode anymore -- every dispatch is a real
release. Never dispatch this from a local shell/agent session; it's a real, outward-facing publish.

### CI's pinned pixi version must track `pixi.lock`'s schema version

`pixi.lock` has its own schema version (currently 7, in its `version:` field), written by whatever
pixi CLI last ran `pixi install`/`pixi update`. Every workflow that runs pixi (`ci-pixi.yml`,
`ci-pixi-macos.yml`, `docs.yml`) pins an exact `pixi-version:`. If your local pixi is newer than
that pin and you run `pixi install`, the lock file's schema version can outrun what the pinned CI
version supports -- every job then fails in "Setup pixi" in under 20 seconds with `lock file
version is 7, but only up to including version 6 is supported`. If you bump your local pixi (or
just notice `pixi.lock`'s diff is enormous), bump the `pixi-version:` pin in all three workflow
files to match.

### Conda-forge: prep only, not submitted

`conda-recipe/meta.yaml` is a local starting point for a future `conda-forge/staged-recipes` PR --
regenerate it with `pixi run build-conda-recipe` (grayskull against a local sdist). Grayskull gets
two things wrong every time, both documented in the file's own header comment and needing a manual
fix after each regeneration: it points `source` at a local `file://` sdist path (repoint at a real
PyPI release instead), and it adds a `mf6rtm --help` test command that doesn't work (see below).

## A known bug, not yet fixed: `mf6rtm`'s CLI ignores arguments

The `mf6rtm` console-script entry point (`mf6rtm:run_cmd`) takes an optional `cwd` parameter and
otherwise ignores `sys.argv` entirely -- `mf6rtm --help` doesn't print help, it attempts a full
`solve()` in the current directory. Don't assume `mf6rtm <anything>` on the command line does what
the flag name suggests until this is fixed.

## Commands

```bash
pixi install                    # dev environment, mf6rtm already editable-installed
pixi run test                   # pytest suite (auto-fetches MODFLOW binaries it needs)
pixi run test-cov               # pytest with coverage
pixi run lint                   # ruff check . (scoped to mf6rtm/, see above)
pixi run format                 # ruff format .
pixi run -e py311 test          # also py312, py313
pixi run -e docs build-docs     # re-executes docs/tutorials/*.py, builds Sphinx HTML
pixi run build-conda-recipe     # regenerate conda-recipe/meta.yaml (see caveats above)
pixi run install-modflow        # fetch MF6 executables directly, without running tests
pixi run install-benchmark-bins # fetch the benchmark/bin/<os> executables directly
```

## Repository structure

| Path | Contents |
| --- | --- |
| `mf6rtm/` | The package. `[tool.ruff]`'s lint scope; every runtime import here must be in `[project.dependencies]`. |
| `autotest/` | Pytest suite (`test_notebooks.py` executes `benchmark/*.ipynb` directly via nbconvert, not jupytext). |
| `benchmark/` | PHT3D-comparison Jupyter notebooks, edited directly as `.ipynb` -- no paired `.py`, unlike `docs/tutorials/` -- plus their bundled executables. |
| `docs/` | Sphinx source, including `docs/tutorials/*.py` (jupytext-paired with `*.ipynb`, re-executed by `build-docs` as a smoke test). |
| `conda-recipe/` | Unsubmitted conda-forge recipe prep (see above). |
