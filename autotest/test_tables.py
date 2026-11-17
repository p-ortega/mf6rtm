"""Tests for table input to Solutions and EquilibriumPhases (#38, #90)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mf6rtm import utils
from mf6rtm.mup3d import ChemStress, EquilibriumPhases, Mup3d, Solutions, tables

DATA = Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "data"
DATABASE = str(DATA / "tut01_datab.dat")


def tut01_solution_tables(with_ph=True):
    """Tutorial 01 solutions as components and settings tables.

    With ``with_ph=False`` the components table has only the four element
    rows, which keeps per-row test changes short.
    """
    components = pd.DataFrame({
        "block": [1, 1, 2, 2],
        "element": ["C(+4)", "Ca", "Cl", "Mg"],
        "conc": [1.23e-4, 1.23e-4, 2e-3, 1e-3],
    })
    if with_ph:
        components = pd.concat([components, pd.DataFrame({
            "block": [1, 1, 2, 2],
            "element": ["pH", "pe", "pH", "pe"],
            "conc": [9.91, 4.0, 7.0, 4.0],
        })], ignore_index=True)
    settings = pd.DataFrame({
        "block": [1, 2],
        "note": ["background", "inflow"],
    })
    return components, settings


def tut01_eq_table():
    return pd.DataFrame({
        "block": [1, 1],
        "phase": ["Calcite", "Dolomite"],
        "si": [0.0, 0.0],
        "m0": [1.220625e-4, 0.0],
    })


def build_model(wd, solutions, eq_phases=None, ncol=4):
    model = Mup3d("tables", solutions, nlay=1, nrow=1, ncol=ncol)
    model.set_wd(str(wd))
    model.set_database(DATABASE)
    if eq_phases is not None:
        model.set_equilibrium_phases(eq_phases)
    return model


def script_blocks(script, keyword):
    """Return the lines of each ``keyword`` block, keyed by its header line."""
    blocks, current = {}, None
    for line in script.splitlines():
        if line.startswith(keyword + " "):
            current = line
            blocks[current] = []
        elif line.strip() == "END":
            current = None
        elif current is not None:
            blocks[current].append(line.strip())
    return blocks


# ==================== Solutions ====================

class TestSolutionsTable:

    def test_table_matches_dict_input(self, tmp_path):
        """Tutorial 01 chemistry gives the same initial state from both inputs."""
        components, settings = tut01_solution_tables()
        table_model = build_model(tmp_path / "table", Solutions(components, settings, ic=1),
                                  EquilibriumPhases(tut01_eq_table(), ic=1))
        table_model.initialize()

        with pytest.warns(DeprecationWarning):
            solutions = Solutions({
                "pH": [9.91, 7.0], "pe": [4.0, 4.0],
                "C(+4)": [1.23e-4, 0.0], "Ca": [1.23e-4, 0.0],
                "Cl": [0.0, 2e-3], "Mg": [0.0, 1e-3],
            })
            eq_phases = EquilibriumPhases({1: {
                "Calcite": {"si": 0.0, "m0": 1.220625e-4},
                "Dolomite": {"si": 0.0, "m0": 0.0},
            }})
        solutions.set_ic(1)
        eq_phases.set_ic(1)
        dict_model = build_model(tmp_path / "dict", solutions, eq_phases)
        dict_model.initialize()

        assert table_model.components == dict_model.components
        for name in dict_model.components:
            np.testing.assert_allclose(table_model.sconc[name], dict_model.sconc[name], rtol=1e-12)

    def test_block_numbers_written_as_given(self, tmp_path):
        components, settings = tut01_solution_tables()
        components["block"] = components["block"].map({1: 10, 2: 20})
        settings["block"] = settings["block"].map({1: 10, 2: 20})
        solutions = Solutions(components, settings)
        assert solutions.block_numbers == [10, 20]
        assert solutions.settings[10]["note"] == "background"

        ic = np.array([[[10, 10, 20, 20]]])
        solutions.set_ic(ic)
        model = build_model(tmp_path, solutions)
        model.initialize()

        script = open(model.phinp).read()
        assert "SOLUTION 10 background\n" in script
        assert "SOLUTION 20 inflow\n" in script
        # cells take the chemistry of the solution number in ic
        assert np.all(model.sconc["Ca"][0, 0, :2] > 0)
        assert np.all(model.sconc["Ca"][0, 0, 2:] == 0)
        assert np.all(model.sconc["Mg"][0, 0, 2:] > 0)

    def test_default_ic_is_first_number(self, tmp_path):
        components, settings = tut01_solution_tables()
        components["block"] = components["block"] + 4
        settings["block"] = settings["block"] + 4
        model = build_model(tmp_path, Solutions(components, settings))
        assert np.all(model.solutions.ic == 5)

    def test_undefined_ic_number_raises(self, tmp_path):
        components, settings = tut01_solution_tables()
        model = build_model(tmp_path, Solutions(components, settings, ic=3))
        with pytest.raises(ValueError, match=r"undefined block\(s\) \[3\]"):
            model.initialize()

    @pytest.mark.parametrize("spd", [[3], {1: [2], 2: [3]}])
    def test_chem_stress_undefined_solution_raises(self, tmp_path, spd):
        components, settings = tut01_solution_tables()
        components["block"] = components["block"] * 10        # solutions 10 and 20
        model = build_model(tmp_path, Solutions(components, settings.assign(block=[10, 20]), ic=10))
        model.initialize()
        wel = ChemStress("wel")
        wel.set_spd(spd)
        with pytest.raises(ValueError, match=r"ChemStress 'wel' refers to undefined solution\(s\) \[(2, )?3\]; "
                                             r"defined solutions are \[10, 20\]"):
            model.set_chem_stress(wel)

        wel.set_spd([20])
        model.set_chem_stress(wel)
        assert model.wel.data

    def test_add_charge_flag_name_must_match(self, tmp_path, capsys):
        """A name matches an element exactly: "C" is not "C(+4)"."""
        components, settings = tut01_solution_tables()
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        script = model.generate_phreeqc_script(add_charge_flag="C")
        assert "charge flag component 'C' not found" in capsys.readouterr().out
        assert " charge" not in script

    def test_line_options(self, tmp_path):
        components = pd.DataFrame({
            "block": [1, 1, 1, 1],
            "element": ["Ca", "C(+4)", "Cl", "pH"],
            "conc": [1.0, 2.0, 1.0, 7.0],
            "units": ["mmol/kgw", np.nan, np.nan, np.nan],
            "as": [np.nan, "HCO3", np.nan, np.nan],
            "charge": [np.nan, np.nan, np.nan, True],
            "phase": [np.nan, "CO2(g)", np.nan, np.nan],
            "phase_si": [np.nan, -3.5, np.nan, np.nan],
        })
        model = build_model(tmp_path, Solutions(components, ic=1, units="mmol/kgw", temp=10.0))
        script = model.generate_phreeqc_script()
        lines = script_blocks(script, "SOLUTION")["SOLUTION 1"]
        assert lines[:2] == ["units mmol/kgw", "temp 10.0"]
        assert "Ca 1.00000e+00 mmol/kgw" in lines
        assert "C(+4) 2.00000e+00 as HCO3 CO2(g) -3.50000e+00" in lines
        assert "Cl 1.00000e+00" in lines
        assert "pH 7.00000e+00 charge" in lines

    def test_add_charge_flag_on_rows(self, tmp_path):
        """add_charge_flag marks one row per solution, also a keyword pH."""
        components, settings = tut01_solution_tables(with_ph=False)
        model = build_model(tmp_path, Solutions(components, settings, ic=1, pH=7.0))
        blocks = script_blocks(model.generate_phreeqc_script(add_charge_flag=True), "SOLUTION")
        for lines in blocks.values():
            assert [line for line in lines if line.startswith("pH")] == ["pH 7.00000e+00 charge"]
            assert sum(line.endswith(" charge") for line in lines) == 1

        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        blocks = script_blocks(model.generate_phreeqc_script(add_charge_flag="Cl"), "SOLUTION")
        assert "Cl 2.00000e-03 charge" in blocks["SOLUTION 2 inflow"]
        assert not any(line.endswith(" charge") for line in blocks["SOLUTION 1 background"])
        # the stored table entries are not changed
        assert not model.solutions.entries[2]["Cl"]["charge"]

    def test_add_charge_flag_same_row_as_charge_column(self, tmp_path):
        components, settings = tut01_solution_tables()
        components["charge"] = components["element"] == "Cl"
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        blocks = script_blocks(model.generate_phreeqc_script(add_charge_flag="Cl"), "SOLUTION")
        assert sum(line.endswith(" charge") for line in blocks["SOLUTION 2 inflow"]) == 1

    @pytest.mark.parametrize("change, options, match", [
        ({"charge": [False, False, True, False]}, {"pH": 7.0}, "more than one charge balance"),
        ({"charge": [False, False, True, False, False, False, False, False]}, {}, "more than one charge balance"),
        ({"phase": [np.nan] * 6 + ["Calcite", np.nan], "phase_si": [np.nan] * 6 + [0.0, np.nan]}, {},
         "add_charge_flag cannot go on a row with 'phase'"),
    ])
    def test_add_charge_flag_conflicts_raise(self, tmp_path, change, options, match):
        """The flag goes through the table's charge checks instead of editing the script text."""
        components, settings = tut01_solution_tables(with_ph="pH" not in options)
        model = build_model(tmp_path, Solutions(components.assign(**change), settings, ic=1, **options))
        with pytest.raises(ValueError, match=match):
            model.generate_phreeqc_script(add_charge_flag=True)

    def test_add_charge_flag_missing_species_warns(self, tmp_path, capsys):
        components, settings = tut01_solution_tables(with_ph=False)
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        script = model.generate_phreeqc_script(add_charge_flag=True)
        assert "charge flag component 'pH' not found" in capsys.readouterr().out
        assert " charge" not in script

    def test_gfw_and_line_redox(self, tmp_path):
        components = pd.DataFrame({
            "block": [1, 1],
            "element": ["S(6)", "Fe"],
            "conc": [96.0, 0.1],
            "units": ["mg/l", np.nan],
            "gfw": [96.06, np.nan],
            "redox": [np.nan, "O(-2)/O(0)"],
        })
        model = build_model(tmp_path, Solutions(components, ic=1))
        lines = script_blocks(model.generate_phreeqc_script(), "SOLUTION")["SOLUTION 1"]
        assert "S(6) 9.60000e+01 mg/l gfw 9.60600e+01" in lines
        assert "Fe 1.00000e-01 O(-2)/O(0)" in lines

    def test_temperature_priority(self, tmp_path):
        components, settings = tut01_solution_tables()
        settings["temp"] = [np.nan, 12.0]

        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        model.set_initial_temp(20.0)
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert "temp 20.0" in blocks["SOLUTION 1 background"]
        assert "temp 12.0" in blocks["SOLUTION 2 inflow"]

        model = build_model(tmp_path, Solutions(components, settings, ic=1, temp=15.0))
        model.set_initial_temp(20.0)
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert "temp 15.0" in blocks["SOLUTION 1 background"]
        assert "temp 12.0" in blocks["SOLUTION 2 inflow"]

    def test_nan_omitted_zero_written(self, tmp_path):
        components, settings = tut01_solution_tables()
        settings["pressure"] = [np.nan, 0.0]
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert not any(line.startswith("pressure") for line in blocks["SOLUTION 1 background"])
        assert "pressure 0.00000e+00" in blocks["SOLUTION 2 inflow"]

    def test_density_calculate(self, tmp_path):
        components, settings = tut01_solution_tables()
        settings = settings.assign(density=[1.023, 1.0], density_calculate=[True, np.nan])
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert "density 1.02300e+00 calculate" in blocks["SOLUTION 1 background"]
        assert "density 1.00000e+00" in blocks["SOLUTION 2 inflow"]

        # keyword options, with a per-solution density
        model = build_model(tmp_path, Solutions(components, settings.drop(columns="density_calculate"),
                                                ic=1, density_calculate=True))
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert "density 1.00000e+00 calculate" in blocks["SOLUTION 2 inflow"]

    @pytest.mark.parametrize("flag, written", [
        ("TRUE", True), (" false ", False), (1, True), (0.0, False), (np.True_, True),
    ])
    def test_density_calculate_flag_values(self, tmp_path, flag, written):
        """Flags read from CSV may be strings or 0/1 rather than bools."""
        components, settings = tut01_solution_tables()
        settings = settings.assign(density=[1.0, 1.0],
                                   density_calculate=pd.Series([flag, np.nan], dtype=object))
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        lines = script_blocks(model.generate_phreeqc_script(), "SOLUTION")["SOLUTION 1 background"]
        assert ("density 1.00000e+00 calculate" in lines) is written

    @pytest.mark.parametrize("settings_change, options, match", [
        ({"density_calculate": [True, np.nan]}, {}, "Solution 1: 'density_calculate' needs a 'density' value"),
        ({}, {"density_calculate": True}, "Solution 1: 'density_calculate' needs a 'density' value"),
        ({"density": [1.0, 1.0], "density_calculate": ["often", np.nan]}, {}, "must be True or False"),
    ])
    def test_invalid_density_calculate(self, settings_change, options, match):
        components, settings = tut01_solution_tables()
        with pytest.raises(ValueError, match=match):
            Solutions(components, settings.assign(**settings_change), **options)

    def test_ph_rows_and_option(self, tmp_path):
        """A solution's pH row overrides the pH option; others use the option."""
        components, settings = tut01_solution_tables(with_ph=False)
        components.loc[len(components)] = [1, "pH", 9.91]
        model = build_model(tmp_path, Solutions(components, settings, ic=1, pH=7.5))
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert [line for line in blocks["SOLUTION 1 background"] if line.startswith("pH")] == ["pH 9.91000e+00"]
        assert "pH 7.50000e+00" in blocks["SOLUTION 2 inflow"]

    def test_no_ph_writes_no_line(self, tmp_path):
        """Without a pH row or option PHREEQC uses its default."""
        components, settings = tut01_solution_tables(with_ph=False)
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        blocks = script_blocks(model.generate_phreeqc_script(), "SOLUTION")
        assert not any(line.startswith(("pH", "pe")) for lines in blocks.values() for line in lines)

    @pytest.mark.parametrize("change, match", [
        ({"foo": [1, 1, 2, 2]}, "Unknown column"),
        ({"charge": [True, True, False, False]}, "more than one charge balance"),
        ({"element": ["Ca", "Ca", "Cl", "Mg"]}, "more than once"),
        ({"block": [1, 1, 2, -2]}, "non-negative integers"),
        ({"block": [1, 1, 2, np.nan]}, "non-negative integers, got nan"),
        ({"block": [1, 1, 2, "a"]}, "non-negative integers, got 'a'"),
        ({"conc": pd.Series([1e-4, True, 2e-3, 1e-3], dtype=object)},
         "'conc' for Ca in solution 1 must be a number, got True"),
        ({"phase_si": [0.0, np.nan, np.nan, np.nan]}, "'phase_si' needs 'phase'"),
        ({"as": ["HCO3", np.nan, np.nan, np.nan], "gfw": [61.0, np.nan, np.nan, np.nan]},
         "either 'as' or 'gfw'"),
        ({"charge": [True, False, False, False], "phase": ["Calcite", np.nan, np.nan, np.nan]},
         "either 'charge' or 'phase'"),
        ({"element": ["C(+4)", np.nan, "Cl", "Mg"]}, "Empty 'element'"),
        ({"conc": [1e-4, np.nan, 2e-3, 1e-3]}, "Empty 'conc' for Ca in solution 1"),
        ({"conc": [1e-4, "abc", 2e-3, 1e-3]}, "'conc' for Ca in solution 1 must be a number"),
        ({"gfw": [np.nan, "x", np.nan, np.nan]}, "'gfw' for Ca in solution 1 must be a number"),
    ])
    def test_invalid_components(self, change, match):
        components, settings = tut01_solution_tables(with_ph=False)
        components = components.assign(**change)
        with pytest.raises(ValueError, match=match):
            Solutions(components, settings)

    def test_missing_required_column(self):
        components, _ = tut01_solution_tables()
        with pytest.raises(ValueError, match=r"Missing required column\(s\) \['conc'\]"):
            Solutions(components.drop(columns="conc"))

    def test_ph_in_settings_raises(self):
        components, settings = tut01_solution_tables(with_ph=False)
        with pytest.raises(ValueError, match=r"Unknown column\(s\) \['pH'\] in Solutions settings"):
            Solutions(components, settings.assign(pH=[9.91, 7.0]))

    def test_unknown_option_raises(self):
        components, settings = tut01_solution_tables()
        with pytest.raises(TypeError, match="unknown option"):
            Solutions(components, settings, water=1.0)

    def test_ph_row_with_units_raises(self):
        components, settings = tut01_solution_tables(with_ph=False)
        components = pd.concat([components, pd.DataFrame(
            {"block": [2], "element": ["pH"], "conc": [7.0], "units": ["mol/kgw"]})], ignore_index=True)
        with pytest.raises(ValueError, match=r"pH in solution 2 cannot have \['units'\]"):
            Solutions(components, settings)

    @pytest.mark.parametrize("change, match", [
        ({"block": [1, 1]}, "Block 1 appears more than once in Solutions settings"),
        ({"temp": [25.0, "warm"]}, "'temp' for block 2 in Solutions settings must be a number"),
        ({"pressure": ["high", 1.0]}, "'pressure' for block 1 in Solutions settings must be a number"),
    ])
    def test_invalid_settings(self, change, match):
        components, settings = tut01_solution_tables()
        with pytest.raises(ValueError, match=match):
            Solutions(components, settings.assign(**change))

    def test_non_numeric_option_raises(self):
        components, settings = tut01_solution_tables()
        with pytest.raises(ValueError, match="Solutions option 'temp' must be a number"):
            Solutions(components, settings, temp="warm")

    def test_empty_tables_raise(self):
        components, _ = tut01_solution_tables()
        with pytest.raises(ValueError, match="define no solutions"):
            Solutions(components.iloc[0:0])

    def test_settings_only_solution(self, tmp_path):
        """A solution may have settings but no components (pure water)."""
        components, settings = tut01_solution_tables()
        settings.loc[len(settings)] = [3, "water"]
        solutions = Solutions(components, settings, ic=1)
        assert solutions.block_numbers == [1, 2, 3]
        blocks = script_blocks(build_model(tmp_path, solutions).generate_phreeqc_script(), "SOLUTION")
        assert blocks["SOLUTION 3 water"] == ["units mol/kgw", "temp 25.0"]

    def test_phase_not_in_database_raises(self, tmp_path):
        components, settings = tut01_solution_tables(with_ph=False)
        components["phase"] = ["Unobtainium", np.nan, np.nan, np.nan]
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        with pytest.raises(AssertionError, match="Following phases are not in database: Unobtainium"):
            model.generate_phreeqc_script()

    def test_dict_input_warns(self):
        with pytest.warns(DeprecationWarning, match="Solutions: dict input is deprecated"):
            solutions = Solutions({"Ca": [1e-3], "Cl": [2e-3]})
        assert solutions.block_numbers is None

    @pytest.mark.parametrize("settings, options", [
        (pd.DataFrame({"block": [1], "temp": [10.0]}), {}),
        (None, {"temp": 10.0}),
    ])
    def test_dict_input_rejects_settings_and_options(self, settings, options):
        with pytest.raises(TypeError, match="settings and options require a components DataFrame"):
            Solutions({"Ca": [1e-3]}, settings, **options)

    @pytest.mark.parametrize("value, given", [
        (None, False), (np.nan, False), (0.0, True), ("", True), ([1, 2], True),
    ])
    def test_is_given(self, value, given):
        """A cell is given unless it is None or NaN; list-like cells count as given."""
        assert tables.is_given(value) is given

    def test_components_must_be_dataframe(self):
        with pytest.raises(TypeError, match="Solutions components must be a pandas DataFrame, got list"):
            Solutions([["Ca", 1e-3]])


# ==================== EquilibriumPhases ====================

class TestEquilibriumPhasesTable:

    def test_defaults_and_options(self, tmp_path):
        components, settings = tut01_solution_tables()
        eq = pd.DataFrame({
            "block": [5, 5, 5],
            "phase": ["Calcite", "Dolomite", "CO2(g)"],
            "si": [np.nan, 0.0, -3.5],
            "m0": [1e-3, 0.0, np.nan],
            "alternative": [np.nan, "Calcite", np.nan],
            "constraint": [np.nan, "precipitate_only", np.nan],
            "force_equality": [np.nan, True, np.nan],
        })
        eq_settings = pd.DataFrame({"block": [5], "note": ["aquifer"]})
        eq_phases = EquilibriumPhases(eq, eq_settings, ic=5)
        model = build_model(tmp_path, Solutions(components, settings, ic=1), eq_phases)
        model.initialize()

        script = open(model.phinp).read()
        assert script_blocks(script, "EQUILIBRIUM_PHASES") == {
            "EQUILIBRIUM_PHASES 5 aquifer": [
                "Calcite 0.00000e+00 1.00000e-03",
                "Dolomite 0.00000e+00 Calcite 0.00000e+00 precipitate_only",
                "-force_equality true",
                "CO2(g) -3.50000e+00 1.00000e+01",
            ]
        }

    def test_block_numbers_kept_after_set(self, tmp_path):
        components, settings = tut01_solution_tables()
        eq = tut01_eq_table()
        eq = pd.concat([eq, eq.assign(block=7, m0=[0.0, 0.0])], ignore_index=True)
        eq["block"] = eq["block"].replace({1: 3})
        eq_phases = EquilibriumPhases(eq, ic=np.array([[[3, 3, 7, 7]]]))
        model = build_model(tmp_path, Solutions(components, settings, ic=1), eq_phases)
        assert list(model.equilibrium_phases.data) == [0, 1]
        assert model.equilibrium_phases.block_numbers == [3, 7]

        model.initialize()
        headers = list(script_blocks(open(model.phinp).read(), "EQUILIBRIUM_PHASES"))
        assert headers == ["EQUILIBRIUM_PHASES 3", "EQUILIBRIUM_PHASES 7"]

        m0 = utils.map_species_property_to_grid(
            model.equilibrium_phases.data, model.equilibrium_phases.ic, "Calcite", "m0",
            block_numbers=model.equilibrium_phases.block_numbers)
        np.testing.assert_allclose(m0[0, 0], [1.220625e-4, 1.220625e-4, 0.0, 0.0])

    @pytest.mark.parametrize("change, match", [
        ({"sat_index": [0.0, 0.0]}, "Unknown column"),
        ({"phase": ["Calcite", "Calcite"]}, "more than once"),
        ({"constraint": ["dissolve", np.nan]}, "'constraint' must be one of"),
        ({"force_equality": ["maybe", np.nan]}, "must be True or False"),
        ({"m0": ["lots", 0.0]}, "'m0' for Calcite in EQUILIBRIUM_PHASES 1 must be a number"),
        ({"phase": ["Calcite", np.nan]}, "Empty 'phase'"),
    ])
    def test_invalid_components(self, change, match):
        with pytest.raises(ValueError, match=match):
            EquilibriumPhases(tut01_eq_table().assign(**change))

    def test_missing_required_column(self):
        with pytest.raises(ValueError, match=r"Missing required column\(s\) \['phase'\]"):
            EquilibriumPhases(tut01_eq_table().drop(columns="phase"))

    def test_empty_tables_raise(self):
        with pytest.raises(ValueError, match="define no blocks"):
            EquilibriumPhases(tut01_eq_table().iloc[0:0])

    def test_note_stored_in_settings(self):
        eq_phases = EquilibriumPhases(tut01_eq_table(), pd.DataFrame({"block": [1], "note": ["two\nlines"]}))
        assert eq_phases.settings == {1: {"note": "two lines"}}

    def test_externalio_warns_about_options(self, tmp_path):
        components, settings = tut01_solution_tables()
        eq = tut01_eq_table().assign(constraint=[np.nan, "precipitate_only"])
        model = build_model(tmp_path, Solutions(components, settings, ic=1),
                            EquilibriumPhases(eq, ic=1))
        model.config.reactive["externalio"] = True
        with pytest.warns(UserWarning, match="externalio rewrites EQUILIBRIUM_PHASES"):
            model.initialize()
        assert os.path.exists(os.path.join(model.wd, "equilibrium_phases.Calcite.m0.layer1.txt"))

    def test_dict_input_warns(self):
        with pytest.warns(DeprecationWarning, match="EquilibriumPhases: dict input is deprecated"):
            eq_phases = EquilibriumPhases({1: {"Calcite": {"si": 0.0, "m0": 1.0}}})
        assert eq_phases.block_numbers is None

    def test_missing_ic_raises(self, tmp_path):
        components, settings = tut01_solution_tables()
        model = build_model(tmp_path, Solutions(components, settings, ic=1))
        with pytest.raises(ValueError, match="EquilibriumPhases has no ic"):
            model.set_equilibrium_phases(EquilibriumPhases(tut01_eq_table()))
        assert model.equilibrium_phases is None

        eq_phases = EquilibriumPhases(tut01_eq_table())
        eq_phases.set_ic(1)
        model.set_equilibrium_phases(eq_phases)
        assert model.equilibrium_phases.ic.shape == model.grid_shape

    def test_dict_input_rejects_settings(self):
        with pytest.raises(TypeError, match="settings requires a components DataFrame"):
            EquilibriumPhases({1: {"Calcite": {"si": 0.0, "m0": 1.0}}},
                              pd.DataFrame({"block": [1], "note": ["sand"]}))
