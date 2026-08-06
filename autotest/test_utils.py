import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from mf6rtm.utils import utils


class TestConversionFunctions:
    """Test suite for concentration and rate conversion functions."""
    
    def test_concentration_l_to_m3(self):
        """Test conversion from M/L to M/m³."""
        assert utils.concentration_l_to_m3(1.0) == 1000.0
        assert utils.concentration_l_to_m3(0.001) == 1.0
        assert utils.concentration_l_to_m3(0) == 0
    
    def test_concentration_m3_to_l(self):
        """Test conversion from M/m³ to M/L."""
        assert utils.concentration_m3_to_l(1000.0) == 1.0
        assert utils.concentration_m3_to_l(1.0) == 0.001
        assert utils.concentration_m3_to_l(0) == 0
    
    def test_concentration_to_massrate(self):
        """Test mass rate calculation from flow rate and concentration."""
        # q = 10 L³/T, conc = 2 M/L³ -> mrate = 20 M/T
        assert utils.concentration_to_massrate(10, 2) == 20
        assert utils.concentration_to_massrate(0, 100) == 0
        assert utils.concentration_to_massrate(5.5, 2.0) == 11.0
    
    def test_concentration_volbulk_to_volwater(self):
        """Test conversion from bulk volume to pore water concentration."""
        # conc_bulk = 10, porosity = 0.2 -> conc_water = 50
        assert utils.concentration_volbulk_to_volwater(10, 0.2) == 50.0
        assert utils.concentration_volbulk_to_volwater(5, 0.5) == 10.0
        # Edge case: porosity = 1.0
        assert utils.concentration_volbulk_to_volwater(10, 1.0) == 10.0


class TestListOperations:
    """Test suite for list manipulation functions."""
    
    def test_flatten_list_simple(self):
        """Test flattening a simple nested list."""
        nested = [[1, 2], [3, 4], [5]]
        assert utils.flatten_list(nested) == [1, 2, 3, 4, 5]
    
    def test_flatten_list_empty(self):
        """Test flattening empty lists."""
        assert utils.flatten_list([]) == []
        assert utils.flatten_list([[], []]) == []
    
    def test_flatten_list_mixed(self):
        """Test flattening lists with different types."""
        nested = [['a', 'b'], ['c'], ['d', 'e', 'f']]
        assert utils.flatten_list(nested) == ['a', 'b', 'c', 'd', 'e', 'f']


class TestPhreeqcScriptGeneration:
    """Test suite for PHREEQC script generation functions."""
    
    def test_generate_solution_block(self):
        """Test generation of SOLUTION block."""
        species_dict = {'Ca': 1.0, 'Cl': 2.0, 'pH': 7.0}
        script = utils.generate_solution_block(species_dict, 0, temp=25.0, water=1.0)
        
        assert 'SOLUTION 1' in script
        assert 'temp 25.0' in script
        assert 'Ca' in script and '1.00000e+00' in script
        assert 'Cl' in script and '2.00000e+00' in script
        assert 'pH' in script and '7.00000e+00' in script
        assert 'END' in script
    
    def test_generate_exchange_block(self):
        """Test generation of EXCHANGE block."""
        exchange_dict = {'X': {'m0': 1.5e-3}}
        script = utils.generate_exchange_block(exchange_dict, 0, equilibrate_solutions=1)
        
        assert 'EXCHANGE 1' in script
        assert 'X' in script
        assert '1.50000e-03' in script
        assert '-equilibrate 1' in script
        assert 'END' in script
    
    def test_generate_equ_phases_block(self):
        """Test generation of EQUILIBRIUM_PHASES block."""
        phases_dict = {'Calcite': {'m0': 0.0, 'si': 10.0}}
        script = utils.generate_equ_phases_block(phases_dict, 0)

        assert 'EQUILIBRIUM_PHASES 1' in script
        assert 'Calcite' in script
        assert '0.00000e+00' in script
        assert '1.00000e+01' in script
        assert 'END' in script

    def test_generate_kinetics_block(self):
        """Test generation of KINETICS block."""
        kinetics_dict = {
            'Pyrite': {
                'm0': 0.1,
                'parms': [1e-5, 2.0],
                'formula': 'FeS2',
                'steps': [100, 200]
            }
        }
        script = utils.generate_kinetics_block(kinetics_dict, 0)

        assert 'KINETICS 1' in script
        assert 'Pyrite' in script
        assert '-m0' in script
        assert '-parms' in script
        assert 'END' in script

    def test_generate_surface_block(self):
        """Test generation of SURFACE block."""
        surface_dict = {'Hfo': [0.1, 600]}
        script = utils.generate_surface_block(surface_dict, 0, options=[])

        assert 'SURFACE 1' in script
        assert 'Hfo' in script
        assert '-equilibrate 1' in script  # default equilibrate target
        assert 'END' in script

    def test_generate_surface_block_equilibrate_target(self):
        """SURFACE block equilibrates with the given solution number."""
        surface_dict = {'Hfo': [0.1, 600]}
        script = utils.generate_surface_block(surface_dict, 0, equilibrate_solutions=5)

        assert '-equilibrate 5' in script
        assert '-equilibrate 1' not in script


class TestCheckPhreeqcrmStatus:
    """Test suite for the PhreeqcRM status guard."""

    def test_negative_status_raises_with_error_string(self):
        """A negative status raises RuntimeError surfacing GetErrorString()."""
        rm = MagicMock()
        rm.GetErrorString.return_value = "EQUILIBRIUM_PHASES 1 not found."
        with pytest.raises(RuntimeError) as exc:
            utils.check_phreeqcrm_status(rm, -3, "RunFile")

        assert "RunFile" in str(exc.value)
        assert "EQUILIBRIUM_PHASES 1 not found." in str(exc.value)

    def test_ok_status_does_not_raise(self):
        """A zero/positive status is a no-op."""
        rm = MagicMock()
        utils.check_phreeqcrm_status(rm, 0, "RunFile")
        utils.check_phreeqcrm_status(rm, 12, "FindComponents")  # component count
        rm.GetErrorString.assert_not_called()

    def test_none_status_does_not_raise(self):
        """A None status (call returned nothing) is tolerated."""
        rm = MagicMock()
        utils.check_phreeqcrm_status(rm, None, "RunFile")
        rm.GetErrorString.assert_not_called()


class TestAddChargeFlag:
    """Test suite for adding charge flags to PHREEQC scripts."""

    def test_add_charge_flag_to_ph(self):
        """Test adding charge flag to pH in SOLUTION block."""
        script = """SOLUTION 1
    pH 7.0
    Ca 1.0
END"""
        modified = utils.add_charge_flag_to_species_in_solution(script, ["pH"])

        assert 'pH 7.0 charge' in modified
        assert 'Ca 1.0' in modified  # Should not be modified
    
    def test_add_charge_flag_multiple_solutions(self):
        """Test adding charge flag across multiple SOLUTION blocks."""
        script = """SOLUTION 1
    pH 7.0
END
SOLUTION 2
    pH 8.0
END"""
        modified = utils.add_charge_flag_to_species_in_solution(script, ["pH"])

        assert modified.count('pH') == 2
        assert modified.count('charge') == 2

    def test_add_charge_flag_no_solutions(self):
        """Test script with no SOLUTION blocks."""
        script = """EQUILIBRIUM_PHASES 1
    Calcite 0.0 10.0
END"""
        modified = utils.add_charge_flag_to_species_in_solution(script, ["pH"])
        
        assert modified == script  # Should be unchanged


class TestDataFrameConversions:
    """Test suite for DataFrame to dictionary conversions."""

    def test_solution_df_to_dict(self):
        """Test converting solution DataFrame to dictionary."""
        df = pd.DataFrame({
            0: [1.0, 2.0, 3.0],
            1: [2.0, 4.0, 6.0],
            2: [7.0, 7.5, 8.0]
        }, index=['Ca', 'Cl', 'pH'])
        result = utils.solution_df_to_dict(df)
        print(df)
        assert 'Ca' in result
        assert 'Cl' in result
        assert 'pH' in result
        assert result['Ca'] == [1.0, 2.0, 7.0]
        assert result['Cl'] == [2.0, 4.0, 7.5]
        assert result['pH'] == [3.0, 6.0, 8.0]

    def test_kinetics_df_to_dict(self):
        """Test converting kinetics DataFrame to dictionary."""
        df = pd.DataFrame({
            'm0': [0.1, 0.2],
            'parm1': [1e-5, 2e-5],
            'parm2': [2.0, 3.0],
            'num': [1, 2]
        }, index=['Pyrite', 'Calcite'])
        df['phase'] = df.index
        result = utils.parse_kinetics_dataframe(df)

        assert 'Pyrite' in result[1]
        assert 'Calcite' in result[2]
        assert 'm0' in result[1]['Pyrite']
        assert 'parms' in result[1]['Pyrite']
        assert result[1]['Pyrite']['m0'] == 0.1
        assert result[1]['Pyrite']['parms'] == [1e-5, 2.]
        assert result[2]['Calcite']['m0'] == 0.2
        assert result[2]['Calcite']['parms'] == [2e-5, 3.]


class TestCSVReaders:
    """Test suite for CSV reading functions."""
    
    def test_solution_csv_to_dict(self, tmp_path):
        """Test reading solution CSV file."""
        csv_file = tmp_path / "solution.csv"
        csv_file.write_text("""comp,0,1
Ca,1.0,2.0
Cl,2.0,4.0
pH,3.0,6.0""")
        
        result = utils.solution_csv_to_dict(str(csv_file))
        print(result)
        
        assert 'Ca' in result
        assert len(result['Ca']) == 2
        assert result['pH'][0] == 3.0


class TestHandleBlock:
    """Test suite for block handling functions."""
    
    def test_handle_block_solution(self):
        """Test handling solution blocks."""
        current_items = {'Ca': 1.0, 'Cl': 2.0}
        script = utils.handle_block(
            current_items, 
            utils.generate_solution_block, 
            0, 
            temp=25.0, 
            water=1.0
        )
        
        assert 'SOLUTION 1' in script
        assert 'Ca' in script
        assert 'END' in script
    
    def test_handle_block_exchange(self):
        """Test handling exchange blocks."""
        current_items = {'X': {'m0':1e-3}}
        script = utils.handle_block(
            current_items,
            utils.generate_exchange_block,
            0,
            equilibrate_solutions=1
        )
        
        assert 'EXCHANGE 1' in script
        assert 'X' in script


class TestGetCompoundNames:
    """Test suite for compound name extraction from database."""
    
    def test_get_compound_names_basic(self, tmp_path):
        """Test extracting compound names from database file."""
        db_file = tmp_path / "test.db"
        db_file.write_text("""SOLUTION_MASTER_SPECIES
Ca    Ca+2    0.0    40.08    40.08
Cl    Cl-     0.0    35.45    35.45
END
PHASES
Calcite    CaCO3 = Ca+2 + CO3-2
Gypsum     CaSO4:2H2O = Ca+2 + SO4-2 + 2H2O
END""")
        
        # Test SOLUTION_MASTER_SPECIES
        species = utils.get_compound_names(str(db_file), 'SOLUTION_MASTER_SPECIES')
        assert 'Ca' in species
        assert 'Cl' in species
        
        # Test PHASES
        phases = utils.get_compound_names(str(db_file), 'PHASES')
        assert 'Calcite' in phases
        assert 'Gypsum' in phases


class TestMapSpeciesProperty:
    """Test suite for mapping species properties to grid."""
    
    def test_map_species_property_to_grid(self):
        """Test mapping species properties to model grid."""
        data_dict = {
            1: {'Goethite': {'m0': 1.0, 'si': 0.9}},
            2: {'Pyrite': {'m0': 2.0, 'si': 1.8}}
        }
        # data_dict = {0: {'Goethite': {'si': 3.0, 'm0': 0.027}}}
        ic_array = np.array([[[1, 1, 2, 1]]])


        filled_dict = utils.fill_missing_minerals(data_dict)

        result = utils.map_species_property_to_grid(
            filled_dict, 
            ic_array, 
            'Goethite', 
            'm0'
                )

        expected = np.array([[[0.0, 0.0, 1.0, 0.0]]])
        np.testing.assert_array_equal(result, expected)


# Fixtures for testing
@pytest.fixture
def sample_solution_dict():
    """Fixture providing sample solution dictionary."""
    return {
        'Ca': [1.0, 2.0, 3.0],
        'Cl': [2.0, 4.0, 6.0],
        'pH': [7.0, 7.5, 8.0]
    }


@pytest.fixture
def sample_kinetics_dict():
    """Fixture providing sample kinetics dictionary."""
    return {
        'Pyrite': {
            'm0': 0.1,
            'parms': [1e-5, 2.0],
            'formula': 'FeS2',
            'steps': [100, 200]
        }
    }