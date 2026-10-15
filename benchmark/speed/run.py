"""Release speed test: time benchmark models under the previous release and the current code.

    python benchmark/speed/run.py --previous 0.5.2
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

import timing

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(REPO, ".speedtest")
MODELS = {"ex7": ("ex7.ipynb", None), "ex8": ("ex8.ipynb", None), "ex5_large": ("ex5.ipynb", "ncol")}
_BIN = {"Linux": "linux", "Darwin": "mac", "Windows": "win"}[platform.system()]


def run_once(python, workdir, notebook, ncol, timeout=3600):
    cmd = [python, os.path.join(HERE, "timed_run.py"), notebook]
    if ncol:
        cmd += ["--ncol", str(ncol)]
    try:
        out = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  failed ({notebook}): no result after {timeout} s", file=sys.stderr)
        return None
    if out.returncode != 0:
        print(f"  failed ({notebook}):\n{out.stderr[-1500:]}", file=sys.stderr)
        return None
    return timing.parse_result(out.stdout)


def check_previous(previous, current):
    """Stop before any run if the previous release cannot be compared."""
    if previous == current:
        raise SystemExit(f"--previous {previous} is the same as the current version")
    found = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "-q", f"refs/tags/{previous}"],
                           capture_output=True)
    if found.returncode != 0:
        raise SystemExit(f"no git tag {previous}; check the version or fetch tags")


def old_python(version):
    """Python of a cached venv with mf6rtm==version installed, or None if the install fails."""
    venv = os.path.join(CACHE, f"venv-{version}")
    py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
    # marker written only after pip succeeds, so a half-built venv is rebuilt next time
    marker = os.path.join(venv, "installed")
    if not os.path.exists(marker):
        try:
            subprocess.run([sys.executable, "-m", "venv", "--clear", venv], check=True)
            subprocess.run([py, "-m", "pip", "install", "-q", f"mf6rtm=={version}", "matplotlib"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"could not install mf6rtm=={version}: {e}", file=sys.stderr)
            return None
        open(marker, "w").close()
    return py


def benchmark_copy(dest, tag=None):
    """Fresh copy of benchmark/ (from a git tag if given), with the current libmf6 in bin/."""
    if tag is None:
        shutil.copytree(os.path.join(REPO, "benchmark"), dest,
                        ignore=shutil.ignore_patterns("speed", "*.png"))
    else:
        tar = dest + ".tar"
        subprocess.run(["git", "-C", REPO, "archive", "-o", tar, tag, "benchmark"], check=True)
        with tarfile.open(tar) as t:
            t.extractall(os.path.dirname(dest), filter="data")
        os.remove(tar)
    shutil.copytree(os.path.join(REPO, "benchmark", "bin", _BIN),
                    os.path.join(dest, "bin", _BIN), dirs_exist_ok=True)
    return dest


def header(previous, current, repeats):
    mf6 = os.path.join(REPO, "benchmark", "bin", _BIN, "mf6")
    try:
        mf6v = subprocess.run([mf6, "-v"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except (OSError, IndexError):
        mf6v = "unknown"
    return (f"**mf6rtm {previous} vs {current}** — {platform.system()} {platform.machine()}, "
            f"Python {platform.python_version()}, {mf6v}, median of {repeats} runs.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--previous", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--models", default="ex7,ex8,ex5_large")
    p.add_argument("--ex5-ncol", type=int, default=400)
    p.add_argument("--out", default="speedtest.md")
    a = p.parse_args()

    from mf6rtm import __version__ as current
    check_previous(a.previous, current)
    versions = {a.previous: old_python(a.previous), current: sys.executable}
    rows = []
    for model in a.models.split(","):
        notebook, scaled = MODELS[model]
        ncol = a.ex5_ncol if scaled else None
        medians = []
        for version, python in versions.items():
            results = []
            for i in range(a.repeats):
                print(f"{model} {version} run {i + 1}/{a.repeats}", flush=True)
                if python is None:
                    results.append(None)
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    tag = a.previous if version == a.previous else None
                    work = benchmark_copy(os.path.join(tmp, "benchmark"), tag)
                    results.append(run_once(python, work, notebook, ncol))
            medians.append(timing.median_result(results))
        rows.append((model, medians[0], medians[1]))

    md = timing.format_table(rows, a.previous, current, header(a.previous, current, a.repeats))
    with open(a.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
