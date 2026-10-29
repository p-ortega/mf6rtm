import pytest
import re
from pprint import pprint
from flaky import flaky
from autotest.conftest import get_project_root_path, run_cmd
import sys
from pathlib import Path
import nbformat
import shutil

# Add the autotest directory to path so we can import comment_conda
sys.path.append(str(Path(__file__).parent))
# from comment_conda import comment_conda_develop

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

def comment_conda_develop(notebook_path):
    """
    Comments out any conda develop lines in a Jupyter notebook and returns the original content.
    
    Args:
        notebook_path (str or Path): Path to the notebook file
        
    Returns:
        dict: Original notebook content
    """
    notebook_path = Path(notebook_path)
    
    # Read the original notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        original_nb = nbformat.read(f, as_version=4)
    
    # Create a copy of the notebook to modify
    modified_nb = nbformat.from_dict(original_nb)
    
    # Process each cell
    for cell in modified_nb.cells:
        if cell.cell_type == 'code':
            # Find and comment out conda develop lines
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if '%conda develop' in line:
                    # Comment out the line
                    new_lines.append(f"# {line}")
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)
    
    # Write the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(modified_nb, f)
    
    return original_nb

def restore_notebook(notebook_path, original_nb):
    """
    Restores a notebook to its original state.
    
    Args:
        notebook_path (str or Path): Path to the notebook file
        original_nb (dict): Original notebook content
    """
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(original_nb, f)

def test_get_notebooks():
    assert len(get_notebooks()) > 0

@flaky(max_runs=3)
@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize(
    "notebook",
    get_notebooks(exclude=["mf6_lgr"])
)
def test_notebooks(notebook):
    # Comment out conda develop lines and get original content
    original_nb = comment_conda_develop(notebook)
    
    try:
        args = ["jupytext", "--from", "ipynb", "--to", "py", "--execute", notebook]
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
    
    finally:
        # Always restore the notebook to its original state
        restore_notebook(notebook, original_nb)

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