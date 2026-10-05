"""Guards against `[project.dependencies]` (PyPI) and `[tool.pixi.dependencies]`
(conda-forge) drifting apart -- see https://github.com/p-ortega/mf6rtm/issues/81.
"""
from pathlib import Path

import toml

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Packages with no meaningful PyPI equivalent that are only ever declared on
# the conda side.
CONDA_ONLY = {"python", "sqlite"}


def _pyproject():
    return toml.load(PYPROJECT)


def test_every_pypi_dependency_has_a_conda_mirror():
    data = _pyproject()
    pypi_deps = {name.lower() for name in data["project"]["dependencies"]}
    pixi_deps = {name.lower() for name in data["tool"]["pixi"]["dependencies"]}

    missing = pypi_deps - pixi_deps
    assert not missing, (
        f"{missing} declared in [project.dependencies] but missing from "
        "[tool.pixi.dependencies]"
    )


def test_no_undocumented_conda_only_dependencies():
    data = _pyproject()
    pypi_deps = {name.lower() for name in data["project"]["dependencies"]}
    pixi_deps = {name.lower() for name in data["tool"]["pixi"]["dependencies"]}

    undocumented = pixi_deps - pypi_deps - CONDA_ONLY
    assert not undocumented, (
        f"{undocumented} declared in [tool.pixi.dependencies] but not in "
        "[project.dependencies] or the CONDA_ONLY allow-list"
    )
