import pytest
import re
from pprint import pprint
from flaky import flaky
from autotest.conftest import get_project_root_path, run_cmd

def get_notebooks(pattern=None, exclude=None):
    prjroot = get_project_root_path()
    nbpaths = [
        str(p)
        for p in (prjroot / "benchmark").glob("*.ipynb")
        if pattern is None or pattern in p.name
    ]

    # sort for pytest-xdist: workers must collect tests in the same order
    return sorted(
        [p for p in nbpaths if not exclude or not any(e in p for e in exclude)]
    )

@flaky(max_runs=3)
@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize(
    "notebook",
    get_notebooks(pattern="ex", exclude=["mf6_lgr"])
    # + get_notebooks(pattern="ex"),
)
def test_notebooks(notebook):
    args = ["jupytext", "--from", "ipynb", "--to", "py", "--execute", notebook]
    stdout, stderr, returncode = run_cmd(*args, verbose=True)

    if returncode != 0:
        if "Missing optional dependency" in stderr:
            pkg = re.findall("Missing optional dependency '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires optional dependency {pkg!r}")
        elif "No module named " in stderr:
            pkg = re.findall("No module named '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires package {pkg!r}")

    assert returncode == 0, f"could not run {notebook}"
    pprint(stdout)
    pprint(stderr)