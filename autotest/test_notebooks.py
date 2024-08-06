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
def test_get_notebooks():
    assert len(get_notebooks()) > 0
@flaky(max_runs=3)
@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize(
    "notebook",
    get_notebooks(pattern="ex", exclude=["mf6_lgr"])
    # + get_notebooks(pattern="ex"),
)
def test_notebooks(notebook):
    # "--from", "ipynb", "--to", "py",
    args = ["jupytext",  "--execute", notebook]
    stdout, stderr, returncode = run_cmd(*args, verbose=True)

    if returncode != 0:
        if "Missing optional dependency" in stderr:
            pkg = re.findall("Missing optional dependency '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires optional dependency {pkg!r}")
        elif "No module named " in stderr:
            pkg = re.findall("No module named '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires package {pkg!r}")

    assert returncode == 0, f"could not run {notebook} {stderr}"
    pprint(stdout)
    pprint(stderr)

# def test_one_note():
#     from pathlib import Path
#     # notebook = get_notebooks(pattern="ex", exclude=["mf6_lgr"])[0]
#     notebook = Path('C://Users//portega//intera//rd//mf6rtm//benchmark//ex1.ipynb')
#     args = ["jupytext",  "--execute", notebook]

#     assert args[-1] == notebook
#     stdout, stderr, returncode = run_cmd(*args, verbose=True)
    # if returncode != 0:
    #     if "Missing optional dependency" in stderr:
    #         pkg = re.findall("Missing optional dependency '(.*)'", stderr)[0]
    #         pytest.skip(f"notebook requires optional dependency {pkg!r}")
    #     elif "No module named " in stderr:
    #         pkg = re.findall("No module named '(.*)'", stderr)[0]
    #         pytest.skip(f"notebook requires package {pkg!r}")

    # assert returncode == 0, f"could not run {notebook} cus of returncode {returncode} and {stderr}"
    # pprint(stdout)
    # pprint(stderr)