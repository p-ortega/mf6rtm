from pathlib import Path
from subprocess import PIPE, Popen
import os, stat

def run_cmd(*args, verbose=False, **kwargs):
    """
    Run any command, return tuple (stdout, stderr, returncode).

    Originally written by Mike Toews (mwtoews@gmail.com) for FloPy.
    """
    args = [str(g) for g in args]
    if verbose:
        print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE, **kwargs)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    if verbose:
        print(f"stdout:\n{stdout}")
        print(f"stderr:\n{stderr}")
        print(f"returncode: {returncode}")
    return stdout, stderr, returncode


def get_project_root_path() -> Path:
    return Path(__file__).parent.parent

def make_dir_writable(function, path, exception):
    """The path on Windows cannot be gracefully removed due to being read-only,
    so we make the directory writable on a failure and retry the original function.
    """
    os.chmod(path, stat.S_IWRITE)
    function(path)