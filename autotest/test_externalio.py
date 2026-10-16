"""Unit tests for externalio.py.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import phreeqcrm
import pytest
from unittest.mock import MagicMock, patch

from mf6rtm.io.externalio import SelectedOutput, Regenerator
from mf6rtm.io.yaml_reader import read_output_vars


# Shared fixtures

@pytest.fixture
def mock_mf6rtm():
    m = MagicMock()
    m.wd = tempfile.mkdtemp()
    m.phreeqcbmi.sout_headers = ["time_d", "cell", "pH", "pe"]
    m.phreeqcbmi.soutdf.columns = ["time_d", "cell", "pH", "pe"]
    yield m
    shutil.rmtree(m.wd, ignore_errors=True)


# configurable sout filename and format

class TestSelectedOutputInit:
    def test_default_params(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        assert so.sout_fname == "sout.csv"
        assert so.output_format == "csv"

    def test_custom_csv_fname(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm, sout_fname="results.csv")
        assert so.sout_fname == "results.csv"
        assert so.output_format == "csv"

    def test_explicit_output_format_csv(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.csv", output_format="csv")
        assert so.output_format == "csv"

    def test_h5_extension_auto_detected(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.h5")
        assert so.output_format == "hdf5"

    def test_hdf5_extension_auto_detected(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.hdf5")
        assert so.output_format == "hdf5"

    def test_explicit_hdf5_format(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.h5", output_format="hdf5")
        assert so.output_format == "hdf5"

    def test_hdf5_missing_tables_raises(self, mock_mf6rtm):
        with patch.dict("sys.modules", {"tables": None}):
            with pytest.raises(ImportError, match="pip install tables"):
                SelectedOutput(mock_mf6rtm, output_format="hdf5")

    def test_csv_extension_with_h5_name_no_conflict(self, mock_mf6rtm):
        """Explicit output_format='csv' overrides extension detection."""
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.csv", output_format="csv")
        assert so.output_format == "csv"


# spatial cell identifier columns

class TestAddSpatialColumns:
    def test_dis_adds_cell_layer_row_col(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 24, "pH": range(24), "pe": range(24)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            result = so._add_spatial_columns(df)
        assert list(result.columns[1:5]) == ["cell", "layer", "row", "col"]
        assert result["cell"].iloc[0] == 1
        assert result["layer"].iloc[0] == 1
        assert result["row"].iloc[0] == 1
        assert result["col"].iloc[0] == 1
        assert result["cell"].iloc[-1] == 24
        assert result["layer"].iloc[-1] == 2

    def test_disv_adds_cell_layer_cell2d(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 30, "pH": range(30)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(3, 10)):
            result = so._add_spatial_columns(df)
        assert list(result.columns[1:4]) == ["cell", "layer", "cell2d"]
        assert "col" not in result.columns
        assert result["layer"].iloc[10] == 2   # 11th cell (0-indexed=10) → layer 2

    def test_cell_not_duplicated_if_phreeqc_has_it(self, mock_mf6rtm):
        """PHREEQC cell column is dropped and replaced by our spatial cell."""
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "cell": [1, 2, 3, 4, 5, 6], "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert list(result.columns).count("cell") == 1
        assert result.columns[0] == "time_d"
        assert result.columns[1] == "cell"

    def test_dis_indices_are_one_based(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert result["cell"].min() == 1
        assert result["layer"].min() == 1
        assert result["row"].min() == 1
        assert result["col"].min() == 1

    def test_time_column_remains_first(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert result.columns[0] == "time_d"

    def test_write_sout_headers_dis_includes_spatial(self, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH", "pe"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "row"
        assert cols[4] == "col"
        assert "pH" in cols
        assert "pe" in cols

    def test_write_sout_headers_disv_includes_spatial(self, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(3, 10)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "cell2d"
        assert "col" not in cols

    def test_write_sout_headers_no_duplicate_cell_when_phreeqc_has_it(self, mock_mf6rtm):
        """PHREEQC 'cell' is excluded — our spatial cell replaces it, time_d stays first."""
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "cell", "pH"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols.count("cell") == 1
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "row"
        assert cols[4] == "col"


# HDF5 output and CSV append

class TestAppendToSoutFile:
    def test_csv_append(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        so._current_soutdf = pd.DataFrame({"time_d": [1.0], "pH": [7.0]})
        so._append_to_soutdf_file()
        path = os.path.join(mock_mf6rtm.wd, "sout.csv")
        assert os.path.exists(path)
        assert "7.0" in open(path).read()

    def test_hdf5_append(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="sout.h5")
        so._current_soutdf = pd.DataFrame({"time_d": [1.0], "pH": [7.0]})
        so._append_to_soutdf_file()
        df = pd.read_hdf(os.path.join(mock_mf6rtm.wd, "sout.h5"), key="sout")
        assert len(df) == 1
        assert "pH" in df.columns

    def test_hdf5_multiple_appends_accumulate(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.hdf5")
        for t in [1.0, 2.0, 3.0]:
            so._current_soutdf = pd.DataFrame({"time_d": [t], "pH": [7.0]})
            so._append_to_soutdf_file()
        df = pd.read_hdf(os.path.join(mock_mf6rtm.wd, "out.hdf5"), key="sout")
        assert len(df) == 3

    def test_rm_sout_file(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        path = os.path.join(mock_mf6rtm.wd, "sout.csv")
        open(path, "w").close()
        so._rm_sout_file()
        assert not os.path.exists(path)

    def test_rm_sout_file_missing_no_error(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        so._rm_sout_file()  # must not raise

    def test_write_sout_headers_skipped_for_hdf5(self, mock_mf6rtm):
        pytest.importorskip("tables")
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH"]
        so = SelectedOutput(mock_mf6rtm, sout_fname="sout.h5")
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            so._write_sout_headers()
        assert not os.path.exists(os.path.join(mock_mf6rtm.wd, "sout.h5"))


# Regenerator: PHREEQC keyword-block parsing / classification / reassembly

# A synthetic phinp exercising every category, including the tricky cases:
#   - SOLUTION_SPECIES / KNOBS with no trailing END (definitions)
#   - SOLUTION vs SOLUTION_SPECIES, EXCHANGE vs EXCHANGE_SPECIES (exact-token match)
#   - SURFACE preserved verbatim; EQUILIBRIUM_PHASES/KINETICS/EXCHANGE dropped
#   - SELECTED_OUTPUT/USER_PUNCH with column-0 data lines (-reset, numbered PUNCH)
_SYNTHETIC_PHINP = """\
SOLUTION_MASTER_SPECIES
    Foo Foo 0.0 Foo 1.0
SOLUTION_SPECIES
    H2O + 0.01e- = H2O-0.01
    log_k -9.0
KNOBS
    iterations 500
SOLUTION 1
    temp 25.0
    pH 7.0
END
SOLUTION 2
    temp 10.0
END
EQUILIBRIUM_PHASES 1
    Calcite 0.0 1.0
END
EXCHANGE 1
    X 0.1
END
KINETICS 1
    Pyrite
        -m0 1.0
END
SURFACE 1
    Hfo_w Goethite equilibrium_phase 0.2 600.0
    -equilibrate 1
END
PRINT
    -reset false
END
SELECTED_OUTPUT
-reset         false
-file          out.dat
USER_PUNCH
    -headings time_d cell pH
1   PUNCH sim_time/86400
2   PUNCH cell_no
END
"""


def _make_regenerator(wd, phinp_text, config=None,
                      nlay=1, nxyz=None, grid_shape=None, file_data=None):
    """Build a Regenerator that bypasses the heavy __init__ (no live model/dll).

    Optional ``nlay``/``nxyz``/``grid_shape``/``file_data`` let tests exercise the
    per-cell block generators and external-file readers without a real model.
    """
    reg = Regenerator.__new__(Regenerator)
    reg.wd = wd
    reg.phinp = "phinp.dat"
    reg.config = config if config is not None else {"reactive": {}, "output": {}}
    reg.nlay = nlay
    if nxyz is not None:
        reg.nxyz = nxyz
    if grid_shape is not None:
        reg.grid_shape = grid_shape
    if file_data is not None:
        reg.file_data = file_data
    with open(os.path.join(wd, "phinp.dat"), "w") as f:
        f.write(phinp_text)
    return reg


@pytest.fixture
def tmp_wd():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestRegeneratorBlockParsing:
    def test_is_block_start_distinguishes_exact_keywords(self):
        f = Regenerator._is_block_start
        assert f("SOLUTION 1\n") == "SOLUTION"
        assert f("SOLUTION_SPECIES\n") == "SOLUTION_SPECIES"
        assert f("EXCHANGE 1\n") == "EXCHANGE"
        assert f("EXCHANGE_SPECIES\n") == "EXCHANGE_SPECIES"
        assert f("SURFACE 1\n") == "SURFACE"

    def test_is_block_start_rejects_data_end_blank(self):
        f = Regenerator._is_block_start
        assert f("    pH 7.0\n") is None        # indented data
        assert f("END\n") is None               # END attaches to current block
        assert f("\n") is None                  # blank
        assert f("-reset false\n") is None      # leading dash
        assert f("1   PUNCH x\n") is None        # leading digit

    def test_classify_block(self):
        c = Regenerator._classify_block
        assert c("EQUILIBRIUM_PHASES") == "regenerated"
        assert c("KINETICS") == "regenerated"
        assert c("EXCHANGE") == "regenerated"
        assert c("SOLUTION") == "solution"
        assert c("SURFACE") == "preserved"
        assert c("GAS_PHASE") == "preserved"
        assert c("SELECTED_OUTPUT") == "output"
        assert c("USER_PUNCH") == "output"
        assert c("KNOBS") == "definition"
        assert c("SOLUTION_SPECIES") == "definition"
        assert c("SOLUTION_MASTER_SPECIES") == "definition"

    def test_split_into_blocks_carries_trailing_end(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        blocks = reg._split_into_blocks(_SYNTHETIC_PHINP.splitlines(keepends=True))
        kws = [kw for kw, _ in blocks]
        assert kws == [
            "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "KNOBS",
            "SOLUTION", "SOLUTION",
            "EQUILIBRIUM_PHASES", "EXCHANGE", "KINETICS",
            "SURFACE", "PRINT", "SELECTED_OUTPUT", "USER_PUNCH",
        ]
        # SOLUTION 1 block keeps its own END
        sol1 = next(lines for kw, lines in blocks if kw == "SOLUTION")
        assert any(l.strip() == "END" for l in sol1)


class TestRegeneratorReassembly:
    def test_definitions_before_solution_and_output_once(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        starts = [
            l.split()[0] for l in script.splitlines()
            if Regenerator._is_block_start(l) is not None
        ]
        # definitions emitted at the top, before the first SOLUTION
        assert starts.index("SOLUTION_SPECIES") < starts.index("SOLUTION")
        assert starts.index("KNOBS") < starts.index("SOLUTION")
        assert starts.index("SOLUTION_MASTER_SPECIES") < starts.index("SOLUTION")
        # output blocks appear exactly once (no duplication)
        assert starts.count("SELECTED_OUTPUT") == 1
        assert starts.count("USER_PUNCH") == 1

    def test_regenerated_blocks_dropped_preserved_kept(self, tmp_wd):
        # config has no *_phases keys -> generators are skipped, so any
        # EQUILIBRIUM_PHASES/KINETICS/EXCHANGE in the output would be leftovers.
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        starts = [
            l.split()[0] for l in script.splitlines()
            if Regenerator._is_block_start(l) is not None
        ]
        assert "EQUILIBRIUM_PHASES" not in starts
        assert "KINETICS" not in starts
        assert "EXCHANGE" not in starts
        # SURFACE preserved verbatim, in the reaction section
        assert "SURFACE" in starts
        assert "-equilibrate 1" in script
        assert "Hfo_w" in script
        last_solution = max(i for i, s in enumerate(starts) if s == "SOLUTION")
        assert starts.index("SURFACE") > last_solution
        assert starts.index("SURFACE") < starts.index("SELECTED_OUTPUT")

    def test_definition_group_terminated_by_end(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        lines = [l for l in script.splitlines()]
        # the defensive END must sit between the last definition and the first SOLUTION
        knobs_i = next(i for i, l in enumerate(lines) if l.strip() == "KNOBS")
        sol_i = next(i for i, l in enumerate(lines) if l.strip().startswith("SOLUTION ")
                     and "SPECIES" not in l and "MASTER" not in l)
        assert any(lines[i].strip() == "END" for i in range(knobs_i, sol_i))

    def test_no_merge_when_last_block_lacks_trailing_newline(self, tmp_wd):
        """A definition block at EOF with no trailing newline must not merge with the
        defensive END (regression for '-diagonal_scale trueEND')."""
        phinp = (
            "SOLUTION 1\n    pH 7.0\nEND\n"
            "SELECTED_OUTPUT\n-file out.dat\n"
            "KNOBS\n    iterations 500\n    -diagonal_scale true"  # no trailing newline
        )
        reg = _make_regenerator(tmp_wd, phinp)
        script = reg.generate_new_script()
        assert "trueEND" not in script
        lines = [l.strip() for l in script.splitlines()]
        # KNOBS moved to the top as a definition, terminated by its own END line
        assert lines[0] == "KNOBS"
        assert "-diagonal_scale true" in lines
        diag_i = lines.index("-diagonal_scale true")
        assert lines[diag_i + 1] == "END"

    def test_leading_comment_kept_as_definition(self, tmp_wd):
        """A comment before the first keyword is emitted at the top (keyword is None)."""
        phinp = "# header comment\nSOLUTION 1\n    pH 7.0\nEND\n"
        reg = _make_regenerator(tmp_wd, phinp)
        script = reg.generate_new_script()
        assert "# header comment" in script


def _make_so_for_ml(mock_mf6rtm, components, nxyz, ctime=1.0):
    """Configure mock_mf6rtm + build a SelectedOutput for write_ml_arrays tests."""
    mock_mf6rtm.nxyz = nxyz
    mock_mf6rtm.ctime = ctime
    mock_mf6rtm.get_saturation_from_mf6.return_value = np.ones(nxyz)
    mock_mf6rtm.phreeqcbmi.components = components
    mock_mf6rtm.phreeqcbmi.ncomps = len(components)
    return SelectedOutput(mock_mf6rtm)


class TestWriteMlArrays:
    def test_writes_header_and_rows_on_first_iter(self, mock_mf6rtm):
        nxyz, comps = 3, ["Ca", "Cl"]
        so = _make_so_for_ml(mock_mf6rtm, comps, nxyz)
        conc = np.arange(len(comps) * nxyz, dtype=float)
        so.write_ml_arrays(conc, iter=0, fname="_features.csv")

        path = os.path.join(mock_mf6rtm.wd, "_features.csv")
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0] == "time,cell,saturation,Ca,Cl"
        assert len(lines) == 1 + nxyz  # header + one row per cell

    def test_append_on_later_iter_has_no_header(self, mock_mf6rtm):
        nxyz, comps = 3, ["Ca", "Cl"]
        so = _make_so_for_ml(mock_mf6rtm, comps, nxyz)
        conc = np.arange(len(comps) * nxyz, dtype=float)
        so.write_ml_arrays(conc, iter=0, fname="_t.csv")
        so.write_ml_arrays(conc, iter=1, fname="_t.csv")

        with open(os.path.join(mock_mf6rtm.wd, "_t.csv")) as f:
            lines = f.read().splitlines()
        # one header + two appended blocks of nxyz rows
        assert lines[0] == "time,cell,saturation,Ca,Cl"
        assert len(lines) == 1 + 2 * nxyz

    def test_additional_variables_appended(self, mock_mf6rtm):
        nxyz, comps = 3, ["Ca", "Cl"]
        so = _make_so_for_ml(mock_mf6rtm, comps, nxyz)
        mock_mf6rtm.phreeqcbmi.soutdf.columns = pd.Index(["time_d", "cell", "pH", "pe"])
        mock_mf6rtm.phreeqcbmi.GetSelectedOutput.return_value = np.arange(4 * nxyz, dtype=float)
        conc = np.arange(len(comps) * nxyz, dtype=float)
        so.write_ml_arrays(conc, iter=0, add_var_names=["pH"], fname="_feat.csv")

        with open(os.path.join(mock_mf6rtm.wd, "_feat.csv")) as f:
            header = f.read().splitlines()[0]
        assert header == "time,cell,saturation,Ca,Cl,pH"


class TestSoutHeaders:
    @patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 1, 3))
    def test_prepends_time_d_when_no_time_header(self, _gd, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["cell", "pH", "pe"]
        so = SelectedOutput(mock_mf6rtm)
        so._write_sout_headers()
        with open(os.path.join(mock_mf6rtm.wd, "sout.csv")) as f:
            header = f.read().splitlines()[0]
        assert header.split(",")[0] == "time_d"
        assert header == "time_d,cell,layer,row,col,pH,pe"

    @patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 1, 3))
    def test_uses_existing_time_header(self, _gd, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time", "cell", "pH"]
        so = SelectedOutput(mock_mf6rtm)
        so._write_sout_headers()
        with open(os.path.join(mock_mf6rtm.wd, "sout.csv")) as f:
            header = f.read().splitlines()[0]
        assert header == "time,cell,layer,row,col,pH"


class TestRegeneratorBlockGeneration:
    def test_equilibrium_phases_blocks(self, tmp_wd):
        config = {"equilibrium_phases": {"names": ["Calcite"], "si": {"Calcite": 0.0}}}
        file_data = {"equilibrium_phases": {"Calcite": np.array([0.0, 0.1, 0.2])}}
        reg = _make_regenerator(tmp_wd, "", config=config, nxyz=3, file_data=file_data)
        blocks = reg.generate_equilibrium_phases_blocks()
        assert len(blocks) == 3
        assert blocks[1].startswith("EQUILIBRIUM_PHASES 2")
        assert "Calcite" in blocks[1]
        assert blocks[0].strip().endswith("END")

    def test_kinetic_phases_blocks(self, tmp_wd):
        config = {"kinetic_phases": {
            "names": ["Pyrite"],
            "parms": {"Pyrite": [1e-5, 2.0]},
            "formula": {"Pyrite": "FeS2"},
        }}
        file_data = {"kinetic_phases": {"Pyrite": np.array([0.1, 0.2, 0.3])}}
        reg = _make_regenerator(tmp_wd, "", config=config, nxyz=3, file_data=file_data)
        blocks = reg.generate_kinetic_phases_blocks()
        assert len(blocks) == 3
        joined = blocks[0]
        assert "KINETICS 1" in joined
        assert "Pyrite" in joined
        assert "-m0" in joined
        assert "-parms" in joined
        assert "-formula FeS2" in joined

    def test_exchange_phases_blocks(self, tmp_wd):
        config = {"exchange_phases": {"names": ["X"]}}
        file_data = {"exchange_phases": {"X": np.array([1.5e-3, 1.6e-3, 1.7e-3])}}
        reg = _make_regenerator(tmp_wd, "", config=config, nxyz=3, file_data=file_data)
        blocks = reg.generate_exchange_phases_blocks()
        assert len(blocks) == 3
        assert "EXCHANGE 1" in blocks[0]
        assert "X" in blocks[0]
        assert "-equilibrate 1" in blocks[0]


class TestRegeneratorFileIO:
    def test_validate_missing_names_raises(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, "", config={"equilibrium_phases": {}}, nlay=1)
        with pytest.raises(ValueError):
            reg.validate_external_files()

    def test_validate_missing_file_raises(self, tmp_wd):
        config = {"equilibrium_phases": {"names": ["Calcite"]}}
        reg = _make_regenerator(tmp_wd, "", config=config, nlay=1)
        with pytest.raises(FileNotFoundError):
            reg.validate_external_files()

    def test_read_external_files_skips_missing_names(self, tmp_wd, capsys):
        reg = _make_regenerator(tmp_wd, "", config={"equilibrium_phases": {}},
                                nlay=1, grid_shape=(1, 1, 3))
        data = reg.read_external_files()
        assert "equilibrium_phases" not in data
        assert "Warning" in capsys.readouterr().out

    def test_read_external_files_loads_layer_file(self, tmp_wd):
        config = {"equilibrium_phases": {"names": ["Calcite"]}}
        np.savetxt(os.path.join(tmp_wd, "equilibrium_phases.Calcite.m0.layer1.txt"),
                   np.array([0.0, 0.1, 0.2]))
        reg = _make_regenerator(tmp_wd, "", config=config, nlay=1, grid_shape=(1, 1, 3))
        data = reg.read_external_files()
        assert data["equilibrium_phases"]["Calcite"] is not None
        assert data["equilibrium_phases"]["Calcite"].shape == (1, 1, 3)

    def test_read_external_files_handles_unreadable_file(self, tmp_wd, capsys):
        """A file numpy cannot parse is warned about and yields a None entry."""
        config = {"equilibrium_phases": {"names": ["Calcite"]}}
        with open(os.path.join(tmp_wd, "equilibrium_phases.Calcite.m0.layer1.txt"), "w") as f:
            f.write("not a number\nfoo bar baz\n")
        reg = _make_regenerator(tmp_wd, "", config=config, nlay=1, grid_shape=(1, 1, 3))
        data = reg.read_external_files()
        assert data["equilibrium_phases"]["Calcite"] is None
        assert "Warning" in capsys.readouterr().out


class TestReadOutputVars:
    """read_output_vars must return the YAML's AddOutputVars entries unchanged."""

    def test_reads_entries_in_order(self, tmp_path):
        """Entries come back in file order with their own true/false values."""
        yrm = phreeqcrm.YAMLPhreeqcRM()
        yrm.YAMLSetGridCellCount(2)
        yrm.YAMLAddOutputVars("SolutionProperties", "true")
        yrm.YAMLAddOutputVars("KineticReactants", "false")
        fname = str(tmp_path / "rm.yaml")
        yrm.WriteYAMLDoc(fname)

        assert read_output_vars(fname) == [
            ("SolutionProperties", "true"),
            ("KineticReactants", "false"),
        ]

    def test_empty_when_yaml_has_no_entries(self, tmp_path):
        """A YAML without AddOutputVars gives an empty list, not an error."""
        yrm = phreeqcrm.YAMLPhreeqcRM()
        yrm.YAMLSetGridCellCount(2)
        fname = str(tmp_path / "rm.yaml")
        yrm.WriteYAMLDoc(fname)

        assert read_output_vars(fname) == []
