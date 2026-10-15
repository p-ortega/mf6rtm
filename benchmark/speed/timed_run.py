"""Run one benchmark notebook once with stopwatches and print a SPEEDTEST line.

Imports only the standard library and mf6rtm, so it also runs in the old-release venv.
"""
import argparse
import functools
import json
import re
import time

MARKER = "SPEEDTEST "  # keep in sync with timing.MARKER
_NCOL_RE = re.compile(r"^ncol\s*=\s*\d+", re.MULTILINE)


def code_until_last_run(nb):
    """Code of the notebook's code cells up to and including the last ``.run(`` cell."""
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    last = max((i for i, c in enumerate(cells) if ".run(" in c), default=None)
    if last is None:
        raise ValueError("notebook has no .run( call")
    keep = [ln for c in cells[: last + 1] for ln in c.splitlines()
            if not ln.lstrip().startswith(("%", "!"))]
    return "\n".join(keep) + "\n"


def set_ncol(code, ncol):
    if not _NCOL_RE.search(code):
        raise ValueError("no `ncol = <int>` line to scale")
    return _NCOL_RE.sub(f"ncol = {ncol}", code, count=1)


class Stopwatch:
    def __init__(self):
        self.times = {"total": 0.0, "chemistry": 0.0, "transport": 0.0}
        self.steps = 0
        self.failed = False

    def wrap(self, cls, name, key, counts_steps=False, resets_steps=False):
        orig = getattr(cls, name, None)
        if orig is None:
            self.times[key] = None
            return

        @functools.wraps(orig)
        def timed(*args, **kwargs):
            if resets_steps:
                self.steps = 0
            if counts_steps:
                self.steps += 1
            t0 = time.perf_counter()
            try:
                out = orig(*args, **kwargs)
                # Mup3d.run returns False instead of raising when the run fails
                if out is False:
                    self.failed = True
                return out
            finally:
                self.times[key] += time.perf_counter() - t0

        setattr(cls, name, timed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--ncol", type=int)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from mf6rtm.mup3d.base import Mup3d
    from mf6rtm.simulation.mf6api import Mf6API
    from mf6rtm.simulation.phreeqcbmi import PhreeqcBMI

    sw = Stopwatch()
    sw.wrap(Mup3d, "run", "total", resets_steps=True)
    sw.wrap(PhreeqcBMI, "_solve_phreeqcrm", "chemistry")
    sw.wrap(Mf6API, "_solve_gwt", "transport", counts_steps=True)

    with open(args.notebook) as f:
        code = code_until_last_run(json.load(f))
    if args.ncol:
        code = set_ncol(code, args.ncol)
    ns = {"__name__": "__main__"}
    exec(compile(code, args.notebook, "exec"), ns)
    if sw.failed:
        raise SystemExit("a model run returned False")

    model = ns.get("model")
    result = dict(sw.times, cells=getattr(model, "nxyz", None), steps=sw.steps or None)
    print(MARKER + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
