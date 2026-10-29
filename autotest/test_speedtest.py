import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark", "speed"))

import run as speed_run  # noqa: E402
import timed_run  # noqa: E402
import timing  # noqa: E402

OLD = {"total": 100.0, "chemistry": 70.0, "transport": 20.0, "cells": 400, "steps": 164}
NEW = {"total": 80.0, "chemistry": 50.0, "transport": 20.0, "cells": 400, "steps": 164}


class TestParseResult:
    def test_takes_last_marker_line(self):
        out = 'noise\nSPEEDTEST {"total": 1.0}\nmore\nSPEEDTEST {"total": 2.0}\n'
        assert timing.parse_result(out) == {"total": 2.0}

    def test_none_without_marker(self):
        assert timing.parse_result("model crashed\n") is None


class TestMedianResult:
    def test_median_per_key(self):
        rs = [dict(OLD, total=t) for t in (90.0, 100.0, 300.0)]
        assert timing.median_result(rs)["total"] == 100.0

    def test_skips_failed_repeats(self):
        assert timing.median_result([None, OLD, OLD])["total"] == 100.0

    def test_all_failed_is_none(self):
        assert timing.median_result([None, None]) is None

    def test_missing_split_stays_none(self):
        r = dict(OLD, chemistry=None, transport=None)
        m = timing.median_result([r, r])
        assert m["chemistry"] is None and m["total"] == 100.0


class TestPctChange:
    def test_faster_is_negative(self):
        assert timing.pct_change(100.0, 80.0) == pytest.approx(-20.0)

    def test_none_when_side_missing(self):
        assert timing.pct_change(None, 80.0) is None


class TestFormatTable:
    def test_row_has_old_and_new(self):
        md = timing.format_table([("ex5_large", OLD, NEW)], "0.5.2", "0.6.0", "Linux")
        assert "| ex5_large | 400 | 164 | 0.5.2 / 0.6.0 | 100.0 / 80.0 | 70.0 / 50.0 | 20.0 / 20.0 | 10.0 / 10.0 | -20% |" in md

    def test_missing_split_prints_na(self):
        old = dict(OLD, chemistry=None, transport=None)
        md = timing.format_table([("ex7", old, NEW)], "0.5.2", "0.6.0", "Linux")
        assert "| n/a / 50.0 |" in md
        assert "| n/a / 10.0 |" in md

    def test_failed_version_prints_failed(self):
        md = timing.format_table([("ex8", None, NEW)], "0.5.2", "0.6.0", "Linux")
        assert "failed / 80.0" in md
        assert md.rstrip().endswith("| n/a |")

    def test_header_line_kept(self):
        md = timing.format_table([], "0.5.2", "0.6.0", "Linux x86_64, Python 3.13")
        assert "Linux x86_64, Python 3.13" in md


def _nb(*cells):
    return {"cells": [{"cell_type": "code", "source": c} for c in cells]}


class TestCodeUntilLastRun:
    def test_stops_after_last_run_cell(self):
        nb = _nb("a = 1\n", "model.run(reactive=False)\n", "model.run()\n", "plot()\n")
        code = timed_run.code_until_last_run(nb)
        assert "model.run()" in code and "plot()" not in code

    def test_drops_magics_and_shell_lines(self):
        nb = _nb("%matplotlib inline\n!pip install x\na = 1\n", "model.run()\n")
        code = timed_run.code_until_last_run(nb)
        assert "%matplotlib" not in code and "!pip" not in code and "a = 1" in code

    def test_skips_markdown_cells(self):
        nb = {"cells": [{"cell_type": "markdown", "source": "model.run()"},
                        {"cell_type": "code", "source": "model.run()\n"}]}
        assert timed_run.code_until_last_run(nb).count("model.run()") == 1

    def test_no_run_call_raises(self):
        with pytest.raises(ValueError, match="no .run\\("):
            timed_run.code_until_last_run(_nb("a = 1\n"))


class TestSetNcol:
    def test_replaces_assignment(self):
        assert timed_run.set_ncol("ncol = 16 # Number of columns\n", 400) == "ncol = 400 # Number of columns\n"

    def test_missing_assignment_raises(self):
        with pytest.raises(ValueError, match="ncol"):
            timed_run.set_ncol("nrow = 1\n", 400)


class TestStopwatch:
    def test_sums_calls_and_counts_steps_of_last_run(self):
        class Model:
            def run(self):
                for _ in range(3):
                    Solver().step()

        class Solver:
            def step(self):
                pass

        sw = timed_run.Stopwatch()
        sw.wrap(Model, "run", "total", resets_steps=True)
        sw.wrap(Solver, "step", "transport", counts_steps=True)
        Model().run()
        Model().run()
        assert sw.times["total"] > 0 and sw.steps == 3

    def test_missing_method_is_none(self):
        sw = timed_run.Stopwatch()
        sw.wrap(object, "nope", "chemistry")
        assert sw.times["chemistry"] is None


def test_timed_run_on_ex5(tmp_path):
    bench = os.path.join(os.path.dirname(__file__), "..", "benchmark")
    work = tmp_path / "benchmark"
    shutil.copytree(bench, work, ignore=shutil.ignore_patterns("*.png"))
    script = os.path.join(bench, "speed", "timed_run.py")
    out = subprocess.run([sys.executable, script, "ex5.ipynb"], cwd=work,
                         capture_output=True, text=True, timeout=600)
    r = timing.parse_result(out.stdout)
    assert r is not None, out.stderr[-2000:]
    assert r["cells"] == 16 and r["steps"] == 164
    assert 0 < r["chemistry"] + r["transport"] <= r["total"]


def test_run_once_returns_none_on_crash(tmp_path):
    (tmp_path / "bad.ipynb").write_text('{"cells": [{"cell_type": "code", "source": "raise SystemExit(3)\\nmodel.run()"}]}')
    assert speed_run.run_once(sys.executable, str(tmp_path), "bad.ipynb", None) is None


def test_run_once_returns_none_on_timeout(tmp_path):
    (tmp_path / "slow.ipynb").write_text('{"cells": [{"cell_type": "code", "source": "import time\\ntime.sleep(30)\\nmodel.run()"}]}')
    assert speed_run.run_once(sys.executable, str(tmp_path), "slow.ipynb", None, timeout=1) is None


def test_stopwatch_flags_a_run_that_returns_false():
    class Model:
        def run(self):
            return False

    sw = timed_run.Stopwatch()
    sw.wrap(Model, "run", "total", resets_steps=True)
    Model().run()
    assert sw.failed


def test_old_python_is_none_when_pip_fails(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if "pip" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        os.makedirs(os.path.join(cmd[-1], "bin"), exist_ok=True)

    monkeypatch.setattr(speed_run, "CACHE", str(tmp_path))
    monkeypatch.setattr(speed_run.subprocess, "run", fake_run)
    assert speed_run.old_python("9.9.9") is None
    assert not os.path.exists(os.path.join(tmp_path, "venv-9.9.9", "installed"))


def test_check_previous_rejects_current_version():
    with pytest.raises(SystemExit, match="same"):
        speed_run.check_previous("0.6.0", "0.6.0")


def test_check_previous_rejects_missing_tag(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128)

    monkeypatch.setattr(speed_run.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="no git tag"):
        speed_run.check_previous("0.0.1", "0.6.0")
