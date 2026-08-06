import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mf6rtm.simulation.solver import Mf6RTM
from mf6rtm.mup3d.base import (
    Block,
    Solutions,
    EquilibriumPhases,
    ExchangePhases,
    KineticPhases,
    Surfaces,
    GasPhase,
    ChemStress,
    Mup3d,
    working_dir
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_solutions_data():
    """Sample solutions data dictionary."""
    return {
        # 'Ca': [1.0, 2.0, 3.0],
        'Cl': [2.0, 4.0, 6.0],
        'pH': [7.0, 7.5, 8.0]
    }


@pytest.fixture
def sample_equilibrium_data():
    """Sample equilibrium phases data."""
    return {
        0: {'Calcite': {'m0': 0.0, 'si': 0.5}},
        1: {'Gypsum': {'m0': 1.0, 'si': -0.5}}
    }


@pytest.fixture
def sample_kinetic_data():
    """Sample kinetic phases data."""
    return {
        0: {
            'Pyrite': {
                'm0': 0.1,
                'parms': [1e-5, 2.0],
                'formula': 'FeS2',
                'steps': [100, 200]
            }
        }
    }


@pytest.fixture
def sample_exchange_data():
    """Sample exchange phases data."""
    return {
        0: {'X': {'m0': 1.5e-3}},
        1: {'X': {'m0': 2.0e-3}}
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def db_temp_dir():
    """Create a temporary directory for database testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def mock_database(db_temp_dir):
    """Create a mock database file."""
    db_path = os.path.join(db_temp_dir, 'pht3d_datab.dat')
    db_content = """SOLUTION_MASTER_SPECIES
H		H+	-1.0	H		1.008
H(0)		H2	0	H
H(1)		H+	-1.0	0
Ca		Ca+2	0	Ca		40.08
Cl		Cl-	0	Cl		35.453

SOLUTION_SPECIES
H+ = H+
	-gamma	9.0	0
	-dw	9.31e-9  1000  0.46  1e-10 # The dw parameters are defined in ref. 3.
# Dw(TK) = 9.31e-9 * exp(1000 / TK - 1000 / 298.15) * TK * 0.89 / (298.15 * viscos)
# Dw(I) = Dw(TK) * exp(-0.46 * DH_A * |z_H+| * I^0.5 / (1 + DH_B * I^0.5 * 1e-10 / (1 + I^0.75)))
e- = e-
H2O = H2O
# H2O + 0.01e- = H2O-0.01; -log_k -9 # aids convergence
Ca+2 = Ca+2
	-gamma	5.0	0.1650
	-dw	0.793e-9  97  3.4  24.6
	-Vm  -0.3456  -7.252  6.149  -2.479  1.239  5  1.60  -57.1  -6.12e-3  1 # ref. 1
Cl- = Cl-
	-gamma	3.5	  0.015
	-gamma	3.63  0.017 # cf. pitzer.dat
	-dw	2.03e-9  194  1.6  6.9
	-Vm  4.465  4.801  4.325  -2.847  1.748  0  -0.331  20.16  0  1 # ref. 1

PHASES
Calcite    CaCO3 = Ca+2 + CO3-2
Gypsum     CaSO4:2H2O = Ca+2 + SO4-2 + 2H2O
Pyrite     FeS2 + 3.5O2 + H2O = Fe+2 + 2SO4-2 + 2H+
END
EXCHANGE_MASTER_SPECIES
X    X-
END
SURFACE_MASTER_SPECIES
Hfo_w    Hfo_wOH
END"""
    with open(db_path, 'w') as f:
        f.write(db_content)
    return db_path


# ==================== Specialized Block Tests ====================

class TestSolutions:
    """Test suite for Solutions class."""
    
    def test_solutions_initialization(self, sample_solutions_data):
        """Test Solutions initialization."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        assert solutions.data == sample_solutions_data
        assert 'pH' in solutions.names
        assert 'Cl' in solutions.names


class TestEquilibriumPhases:
    """Test suite for EquilibriumPhases class."""
    
    def test_equilibrium_phases_initialization(self, sample_equilibrium_data):
        """Test EquilibriumPhases initialization."""
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        assert 'Calcite' in eq_phases.names
        assert 'Gypsum' in eq_phases.names
    
    @patch('mf6rtm.mup3d.base.utils.fill_missing_minerals')
    def test_fill_missing_minerals_called(self, mock_fill, sample_equilibrium_data):
        """Test that fill_missing_minerals is called."""
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        mock_fill.assert_called_once()


class TestKineticPhases:
    """Test suite for KineticPhases class."""
    
    def test_kinetic_phases_initialization(self, sample_kinetic_data):
        """Test KineticPhases initialization."""
        kinetic = KineticPhases(sample_kinetic_data)
        assert 'Pyrite' in kinetic.names
        assert kinetic.parameters is None
    
    def test_set_parameters(self, sample_kinetic_data):
        """Test setting kinetic parameters."""
        kinetic = KineticPhases(sample_kinetic_data)
        params = {'rate': 1e-5}
        kinetic.set_parameters(params)
        assert kinetic.parameters == params


class TestExchangePhases:
    """Test suite for ExchangePhases class."""
    
    def test_exchange_phases_initialization(self, sample_exchange_data):
        """Test ExchangePhases initialization."""
        exchange = ExchangePhases(sample_exchange_data)
        assert 'X' in exchange.names


class TestSurfaces:
    """Test suite for Surfaces class."""
    
    def test_surfaces_initialization(self):
        """Test Surfaces initialization."""
        surface_data = {0: {'Hfo': [0.1, 600]}}
        surfaces = Surfaces(surface_data)
        assert 'Hfo' in surfaces.names


# ==================== ChemStress Tests ====================

class TestChemStress:
    """Test suite for ChemStress class."""

    def test_chem_stress_initialization(self):
        """Test ChemStress defaults to aux type."""
        stress = ChemStress('WEL-1')
        assert stress.packnme == 'WEL-1'
        assert stress.type == 'aux'
        assert stress.sol_spd is None
        assert stress.cells is None

    def test_chem_stress_cnc_type(self):
        """Test explicit cnc type."""
        stress = ChemStress('inflow', type='cnc')
        assert stress.type == 'cnc'

    def test_set_spd(self):
        """Test setting stress period data."""
        stress = ChemStress('WEL-1')
        spd = [1, 2, 3, 4, 5]
        stress.set_spd(spd)
        assert stress.sol_spd == spd

    def test_set_type(self):
        """Test set_type changes the coupling type."""
        stress = ChemStress('inflow')
        stress.set_type('src')
        assert stress.type == 'src'

    def test_set_cells(self):
        """Test set_cells stores cellids."""
        stress = ChemStress('inflow', type='cnc')
        cells = [(0, 0, 0), (0, 0, 1)]
        stress.set_cells(cells)
        assert stress.cells == cells


# ==================== Mup3d Tests ====================

class TestMup3dInitialization:
    """Test suite for Mup3d initialization."""
    
    def test_mup3d_initialization_dis(self, sample_solutions_data):
        """Test Mup3d initialization with DIS grid."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 2, 2))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
            
        model = Mup3d(
            solutions=solutions,
            nlay=1,
            nrow=2,
            ncol=2
        )
        assert model.nlay == 1
        assert model.nrow == 2
        assert model.ncol == 2
        assert model.nxyz == 4
        assert model.grid_shape == (1, 2, 2)
    
    def test_mup3d_initialization_disv(self, sample_solutions_data):
        """Test Mup3d initialization with DISV grid."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 4))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
        model = Mup3d(
            solutions=solutions,
            nlay=1,
            ncpl=4
        )
        assert model.nlay == 1
        assert model.ncpl == 4
        assert model.nxyz == 4
        assert model.grid_shape == (1, 4)
    
    # def test_mup3d_initialization_disu(self, sample_solutions_data):
    #     """Test Mup3d initialization with DISU grid."""
    #     solutions = Solutions(sample_solutions_data)
    #     model = Mup3d(
    #         solutions=solutions,
    #         nxyz=100
    #     )
    #     assert model.nxyz == 100
    #     assert model.grid_shape == (100,)
    
    def test_mup3d_no_solutions_error(self):
        """Test error when solutions not provided."""
        with pytest.raises(ValueError, match="solutions parameter is required"):
            Mup3d(nlay=1, nrow=1, ncol=1)
    
    def test_mup3d_no_grid_params_error(self, sample_solutions_data):
        """Test error when grid parameters not provided."""
        solutions = Solutions(sample_solutions_data)
        with pytest.raises(ValueError):
            Mup3d(solutions=solutions)
    
    def test_mup3d_old_style_initialization(self, sample_solutions_data):
        """Test old-style initialization (solutions as first arg)."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 1, 4))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions, nlay=1, nrow=1, ncol=4)
        assert model.solutions == solutions
        assert model.nxyz == 4

class TestMup3dSetters:
    """Test suite for Mup3d setter methods."""
    
    def test_set_wd(self, sample_solutions_data, temp_dir):
        """Test setting working directory."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_wd(temp_dir)
        assert model.wd == Path(temp_dir)
        assert os.path.exists(temp_dir)
    
    def test_set_database(self, sample_solutions_data, temp_dir, mock_database):
        """Test setting database."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        model.set_wd(temp_dir)
        model.set_database(mock_database)
        # Database should be copied to wd
        assert os.path.exists(os.path.join(temp_dir, os.path.basename(mock_database)))
    
    def test_set_initial_temp(self, sample_solutions_data):
        """Test setting initial temperature."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_initial_temp(30.0)
        assert model.init_temp == 30.0
    
    def test_set_initial_temp_invalid(self, sample_solutions_data):
        """Test setting initial temperature with invalid type."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        with pytest.raises(AssertionError):
            model.set_initial_temp("invalid")
    
    def test_set_componenth2o(self, sample_solutions_data):
        """Test setting componentH2O flag."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        result = model.set_componenth2o(True)
        assert result is True
        assert model.componenth2o is True
    
    def test_set_componenth2o_invalid(self, sample_solutions_data):
        """Test set_componenth2o with invalid type."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        with pytest.raises(AssertionError):
            model.set_componenth2o("invalid")
    
    def test_get_componenth2o(self, sample_solutions_data):
        """Test getting componentH2O flag."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_componenth2o(True)
        assert model.get_componenth2o() is True
    
    def test_set_charge_offset(self, sample_solutions_data):
        """Test setting charge offset."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_charge_offset(10.0)
        assert model.charge_offset == 10.0
    
    def test_set_fixed_components(self, sample_solutions_data):
        """Test setting fixed components."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        fixed = ['H', 'O']
        model.set_fixed_components(fixed)
        assert model.fixed_components == fixed

    def test_set_diffusion_coeff(self, sample_solutions_data):
        """Test setting per-component diffusion coefficients."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        coeffs = {'Ca': 0.792e-9, 'Cl': 2.032e-9}
        model.set_diffusion_coeff(coeffs)
        assert model._diffusion_coeff == coeffs

    def test_set_diffusion_coeff_overwrites(self, sample_solutions_data):
        """Calling set_diffusion_coeff again replaces the previous mapping."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_diffusion_coeff({'Ca': 1e-9})
        model.set_diffusion_coeff({'Cl': 2e-9})
        assert model._diffusion_coeff == {'Cl': 2e-9}


class TestResolveChargeSpecies:
    """Test suite for Mup3d._resolve_charge_species (charge-flag resolution)."""

    def test_true_defaults_to_ph(self):
        assert Mup3d._resolve_charge_species(True) == "pH"

    def test_string_returned_as_is(self):
        assert Mup3d._resolve_charge_species("Cl") == "Cl"

    def test_single_element_list_unwrapped(self):
        assert Mup3d._resolve_charge_species(["Na"]) == "Na"

    def test_single_element_tuple_unwrapped(self):
        assert Mup3d._resolve_charge_species(("Na",)) == "Na"

    def test_multiple_components_raise(self):
        """Only one component may carry the charge flag."""
        with pytest.raises(ValueError):
            Mup3d._resolve_charge_species(["Na", "Cl"])

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            Mup3d._resolve_charge_species(123)


class TestMup3dPhases:
    """Test suite for Mup3d phase-related methods."""
    
    def test_set_equilibrium_phases(self, sample_solutions_data, sample_equilibrium_data):
        """Test setting equilibrium phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        eq_phases.set_ic(np.ones((1, 2, 5), dtype=int))
        model.set_equilibrium_phases(eq_phases)
        
        assert model.equilibrium_phases is not None
        assert model.equilibrium_phases.ic.shape == (1, 2, 5)
    
    def test_set_phases_kinetic(self, sample_solutions_data, sample_kinetic_data):
        """Test setting kinetic phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        kinetic = KineticPhases(sample_kinetic_data)
        kinetic.set_ic(np.zeros((1, 2, 5), dtype=int))
        model.set_phases(kinetic)
        
        assert model.kinetic_phases is not None
    
    def test_set_phases_invalid_ic_shape(self, sample_solutions_data, sample_equilibrium_data):
        """Test set_phases with invalid IC shape."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        eq_phases.set_ic(np.ones((2, 3, 4), dtype=int))  # Wrong shape
        
        with pytest.raises(AssertionError):
            model.set_equilibrium_phases(eq_phases)
    
    def test_set_exchange_phases(self, sample_solutions_data, sample_exchange_data):
        """Test setting exchange phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        exchange = ExchangePhases(sample_exchange_data)
        exchange.set_ic(np.ones((1, 2, 5), dtype=int))
        model.set_exchange_phases(exchange)
        
        assert model.exchange_phases is not None


# class TestMup3dChemStress:
#     """Test suite for ChemStress in Mup3d."""
    
#     def test_set_chem_stress(self, sample_solutions_data):
#         """Test setting ChemStress."""
#         solutions = Solutions(sample_solutions_data)
#         sol_ic = 1
#         solutions.set_ic(sol_ic)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)

#         model.set_wd(temp_dir)
#         model.set_database(mock_database)
        
#         stress = ChemStress('WEL-1')
#         stress.set_spd([2])
#         stress.set_packtype('WEL')
        
#         model.set_chem_stress(stress)
#         assert hasattr(model, 'WEL-1')
#         assert getattr(model, 'WEL-1') == stress
    
#     def test_set_chem_stress_invalid_type(self, sample_solutions_data):
#         """Test set_chem_stress with invalid type."""
#         solutions = Solutions(sample_solutions_data)
#         sol_ic = 1
#         solutions.set_ic(sol_ic)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        
#         with pytest.raises(AssertionError):
#             model.set_chem_stress("invalid")


# class TestMup3dConfig:
#     """Test suite for Mup3d configuration methods."""
    
#     def test_set_config(self, sample_solutions_data):
#         """Test setting configuration."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
#         config = model.set_config(
#             reactive_externalio=True,
#             nthreads=4
#         )
#         assert config.reactive_externalio is True
#         assert config.nthreads == 4
    
#     def test_get_config(self, sample_solutions_data):
#         """Test getting configuration."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
#         model.set_config(nthreads=2)
        
#         config_dict = model.get_config()
#         assert isinstance(config_dict, dict)
#         assert config_dict['nthreads'] == 2
    
#     def test_save_config(self, sample_solutions_data, temp_dir):
#         """Test saving configuration to file."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
#         model.set_wd(temp_dir)
#         model.set_config(nthreads=3)
        
#         config_path = model.save_config()
#         assert os.path.exists(config_path)
#         assert config_path.name == 'mf6rtm.toml'


class TestMup3dInitialConditions:
    """Test suite for initial conditions handling."""

    def test_solutions_none(self, sample_solutions_data):
        """Test error when solutions is None."""
        solutions = None
        
        with pytest.raises(ValueError):
            model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
    
    def test_solutions_ic_single_value(self, sample_solutions_data):
        """Test solutions IC with single value."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(2)
        model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
        
        assert model.solutions.ic.shape == (2, 3, 4)
        assert np.all(model.solutions.ic == 2)
    
    def test_solutions_ic_array(self, sample_solutions_data):
        """Test solutions IC with array."""
        solutions = Solutions(sample_solutions_data)
        ic = np.ones((2, 3, 4), dtype=int)
        ic[0, :, :] = 1
        ic[1, :, :] = 2
        solutions.set_ic(ic)
        
        model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
        np.testing.assert_array_equal(model.solutions.ic, ic)


class TestMup3dFileOperations:
    """Test suite for file operations."""
    
    def test_set_postfix(self, sample_solutions_data, temp_dir):
        """Test setting postfix file."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
        postfix_file = os.path.join(temp_dir, 'postfix.txt')
        with open(postfix_file, 'w') as f:
            f.write('# Postfix content')
        
        model.set_postfix(postfix_file)
        assert model.postfix == postfix_file
    
    def test_set_postfix_nonexistent(self, sample_solutions_data):
        """Test set_postfix with nonexistent file."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
        with pytest.raises(AssertionError):
            model.set_postfix('/nonexistent/file.txt')


class TestMup3dSaveLoad:
    """Test suite for save/load operations."""
    
    def test_save_mup3d(self, sample_solutions_data, temp_dir):
        """Test saving Mup3d object."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        model.set_wd(temp_dir)
        model.set_initial_temp(25.0)
        model.set_componenth2o(True)
        
        model.save_mup3d('test_model.pkl')
        
        pkl_file = os.path.join(temp_dir, 'test_model.pkl')
        assert os.path.exists(pkl_file)
    
    def test_load_mup3d(self, sample_solutions_data, temp_dir, mock_database):
        """Test loading Mup3d object."""
        # Create and save a model
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        model.set_wd(temp_dir)
        model.set_database(mock_database)
        model.set_initial_temp(30.0)
        model.set_componenth2o(True)
        model.set_charge_offset(5.0)

        model.save_mup3d('test_model.pkl')

        # Load the model
        loaded_model = Mup3d.load_mup3d('test_model.pkl', temp_dir)

        assert loaded_model.nlay == 1
        assert loaded_model.nrow == 2
        assert loaded_model.ncol == 5
        assert loaded_model.init_temp == 30.0
        assert loaded_model.componenth2o is True
        assert loaded_model.charge_offset == 5.0


# ==================== solve() kwargs Tests ====================

class TestSolveKwargs:
    """Tests for the **kwargs passthrough in solve() and Mup3d.run()."""

    def _make_mock_mf6rtm(self, **attrs):
        """Return a Mock with Mf6RTM-like attributes pre-set."""
        mock = Mock()
        defaults = dict(
            reactive=True,
            min_concentration=None,
            threshold=1e-10,
            charge_offset=0.0,
            fixed_components=None,
        )
        defaults.update(attrs)
        for k, v in defaults.items():
            setattr(mock, k, v)
        return mock

    @patch("mf6rtm.simulation.solver.initialize_interfaces")
    def test_kwarg_sets_attribute(self, mock_init):
        """Valid kwargs are applied to the Mf6RTM instance before solving."""
        from mf6rtm.simulation.solver import solve

        mock_mf6rtm = self._make_mock_mf6rtm()
        mock_init.return_value = mock_mf6rtm

        solve(Path("/fake/wd"), min_concentration=1e-6)

        assert mock_mf6rtm.min_concentration == 1e-6

    @patch("mf6rtm.simulation.solver.initialize_interfaces")
    def test_multiple_kwargs_set(self, mock_init):
        """Multiple valid kwargs are all applied."""
        from mf6rtm.simulation.solver import solve

        mock_mf6rtm = self._make_mock_mf6rtm()
        mock_init.return_value = mock_mf6rtm

        solve(Path("/fake/wd"), threshold=1e-8, charge_offset=0.5)

        assert mock_mf6rtm.threshold == 1e-8
        assert mock_mf6rtm.charge_offset == 0.5

    @patch("mf6rtm.simulation.solver.initialize_interfaces")
    def test_unknown_kwarg_raises(self, mock_init):
        """An unrecognised kwarg raises AttributeError before solving."""
        from mf6rtm.simulation.solver import solve
        from mf6rtm.simulation.solver import Mf6RTM

        mock_mf6rtm = Mock(spec=Mf6RTM)
        mock_init.return_value = mock_mf6rtm

        with pytest.raises(AttributeError, match="no attribute 'nonexistent_param'"):
            solve(Path("/fake/wd"), nonexistent_param=42)


# ==================== Mf6RTM concentration transfer Tests ====================

class TestTransferArrayToPhreeqcRM:
    """Unit tests for _transfer_array_to_phreeqcrm concentration clipping."""

    def _make_mock_self(self, components, concs_by_component, min_concentration=None, charge_offset=0.0):
        """Build a minimal mock Mf6RTM instance for _transfer_array_to_phreeqcrm."""
        nxyz = len(next(iter(concs_by_component.values())))
        mock = MagicMock()
        mock.nxyz = nxyz
        mock.min_concentration = min_concentration
        mock.charge_offset = charge_offset
        mock.phreeqcbmi.ncomps = len(components)
        mock.phreeqcbmi.components = components
        mock.component_model_dict = {c: f"gwt_{c.lower()}" for c in components}
        mock.mf6api.get_value.side_effect = lambda key: concs_by_component[
            key.split("/")[0].lower().replace("gwt_", "")
        ].copy()
        return mock

    def test_clipping_uses_m3_floor(self):
        """np.clip uses the mol/m³ floor, not the raw mol/L value."""
        # min_concentration = 1e-6 mol/L → floor in mol/m³ = 1e-3
        concs = np.array([-5e-4, 5e-4, 2e-3])  # mol/m³; first two are below floor
        mock = self._make_mock_self(
            components=["Na"],
            concs_by_component={"na": concs},
            min_concentration=1e-6,
        )

        Mf6RTM._transfer_array_to_phreeqcrm(mock)

        passed = mock.phreeqcbmi.SetConcentrations.call_args[0][0]
        result_m3 = passed * 1e3  # convert back from mol/L to mol/m³
        floor_m3 = 1e-3
        assert np.all(result_m3 >= floor_m3), "Values below floor_m3 were not clipped"

    def test_clipping_prints_when_below(self, capsys):
        """A warning is printed when cells fall below min_concentration."""
        concs = np.array([-1e-3, 2e-3])  # first cell below floor of 1e-3 mol/m³
        mock = self._make_mock_self(
            components=["Ca"],
            concs_by_component={"ca": concs},
            min_concentration=1e-6,
        )

        Mf6RTM._transfer_array_to_phreeqcrm(mock)

        captured = capsys.readouterr()
        assert "mol/L" in captured.out
        assert "Ca" in captured.out

    def test_no_clipping_when_min_concentration_none(self):
        """When min_concentration is None, negative values pass through unchanged."""
        concs = np.array([-1e-3, 2e-3])
        mock = self._make_mock_self(
            components=["Cl"],
            concs_by_component={"cl": concs},
            min_concentration=None,
        )

        Mf6RTM._transfer_array_to_phreeqcrm(mock)

        passed = mock.phreeqcbmi.SetConcentrations.call_args[0][0]
        result_m3 = passed * 1e3
        assert result_m3[0] < 0, "Negative value should not be clipped when min_concentration=None"

    def test_charge_component_uses_offset(self):
        """Charge component subtracts charge_offset instead of clipping."""
        concs = np.array([5e-3, 3e-3])
        mock = self._make_mock_self(
            components=["Charge"],
            concs_by_component={"charge": concs},
            min_concentration=1e-6,
            charge_offset=1e-3,
        )

        Mf6RTM._transfer_array_to_phreeqcrm(mock)

        passed = mock.phreeqcbmi.SetConcentrations.call_args[0][0]
        result_m3 = passed * 1e3
        np.testing.assert_allclose(result_m3, concs - 1e-3)


# ==================== from_mf6 Structural Tests ====================

_BENCHMARK_DB = Path(__file__).parent.parent / 'benchmark' / 'database' / 'pht3d_datab.dat'


@pytest.fixture
def benchmark_database():
    """Path to the benchmark pht3d_datab.dat — skipped if not present."""
    if not _BENCHMARK_DB.exists():
        pytest.skip("Benchmark database not found; skipping from_mf6 integration test")
    return str(_BENCHMARK_DB)


class TestFromMf6:
    """Structural tests for Mup3d.from_mf6 — no MF6 run required."""

    @staticmethod
    def _build_minimal_sim(sim_ws, nper=1):
        """Minimal flopy sim: 3-cell GWF + tracer GWT with CHD aux + SSM.

        ``nper`` stress periods are created; the inflow CHD (``chdin``) carries a
        ``tracer`` aux and is given SPD rows for every period so per-period aux
        expansion has something to attach to.
        """
        import flopy
        sim = flopy.mf6.MFSimulation(
            sim_name='test', sim_ws=str(sim_ws), exe_name='mf6'
        )
        flopy.mf6.ModflowTdis(sim, nper=nper,
                              perioddata=[(1.0, 1, 1.0)] * nper,
                              time_units='days')

        gwf = flopy.mf6.ModflowGwf(sim, modelname='gwf', save_flows=True)
        ims_gwf = flopy.mf6.ModflowIms(sim, filename='gwf.ims')
        sim.register_ims_package(ims_gwf, ['gwf'])

        flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=3,
                                 delr=1.0, delc=1.0, top=0.0, botm=-1.0,
                                 filename='gwf.dis')
        flopy.mf6.ModflowGwfnpf(gwf, k=1.0, filename='gwf.npf')
        flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename='gwf.ic')
        chdin_spd = {k: [[(0, 0, 0), 1.0, 1.0]] for k in range(nper)}
        flopy.mf6.ModflowGwfchd(
            gwf, stress_period_data=chdin_spd,
            auxiliary=['tracer'], pname='chdin', filename='gwf.chdin.chd',
        )
        flopy.mf6.ModflowGwfchd(
            gwf, stress_period_data=[[(0, 0, 2), 0.0]],
            pname='chdout', filename='gwf.chdout.chd',
        )
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord='gwf.hds',
                                budget_filerecord='gwf.cbb',
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])

        gwt = flopy.mf6.ModflowGwt(sim, modelname='gwt')
        ims_gwt = flopy.mf6.ModflowIms(
            sim, linear_acceleration='BICGSTAB', filename='gwt.ims'
        )
        sim.register_ims_package(ims_gwt, ['gwt'])

        flopy.mf6.ModflowGwtdis(gwt, nlay=1, nrow=1, ncol=3,
                                  delr=1.0, delc=1.0, top=0.0, botm=-1.0,
                                  filename='gwt.dis')
        flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename='gwt.ic')
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[['chdin', 'aux', 'tracer']], filename='gwt.ssm'
        )
        flopy.mf6.ModflowGwtadv(gwt, scheme='UPSTREAM')
        flopy.mf6.ModflowGwtmst(gwt, porosity=0.3, filename='gwt.mst')
        flopy.mf6.ModflowGwtoc(gwt, budget_filerecord='gwt.cbc',
                                 concentration_filerecord='gwt.ucn',
                                 saverecord=[('CONCENTRATION', 'ALL')])
        flopy.mf6.ModflowGwfgwt(sim, exgtype='GWF6-GWT6',
                                  exgmnamea='gwf', exgmnameb='gwt',
                                  filename='gwf-gwt.gwfgwt')
        return sim

    def test_tracer_component_name_collision_raises(self, tmp_path):
        """A tracer GWT named after a PHREEQC component fails loudly.

        from_mf6() clones the tracer into one GWT per component; if the tracer
        name is also a component name the clone collides and flopy fails with an
        opaque solutiongroup error. The guard turns that into a clear ValueError.
        No MF6 run or database required — the guard fires before any sim access.
        """
        sim_ws = tmp_path / 'collision'
        sim_ws.mkdir()
        sim = self._build_minimal_sim(sim_ws)

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        # Force the collision: the tracer GWT is named 'gwt', which is also a
        # (pretend) PHREEQC component. Normally self.components is set by
        # initialize(); set it directly to isolate the guard.
        model.components = ['gwt', 'Ca', 'Cl']

        with pytest.raises(ValueError, match="collides with a"):
            model._build_reactive_gwt_models()

    def test_aux_and_cnc_file_structure(self, tmp_path, benchmark_database):
        """write_simulation writes per-component CNC files and removes tracer GWT."""
        sim_ws = tmp_path / 'conservative'
        sim_ws.mkdir()
        sim = self._build_minimal_sim(sim_ws)

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        components = model.components
        assert len(components) > 0

        chdin_cs = ChemStress('chdin', type='aux')
        chdin_cs.set_spd([2])
        model.set_chem_stress(chdin_cs)

        cnc_cs = ChemStress('cncout', type='cnc')
        cnc_cs.set_spd([1])
        cnc_cs.set_cells([(0, 0, 2)])
        model.set_chem_stress(cnc_cs)

        model.write_simulation()

        written = set(os.listdir(model.wd))

        # Tracer GWT model files must be gone
        tracer_files = [f for f in written if f.startswith('gwt.') and not f.endswith('.ims')]
        assert len(tracer_files) == 0, f'Tracer model files remain: {tracer_files}'

        for c in components:
            assert any(f.startswith(f'{c}.') for f in written), f'No GWT files for {c}'
            assert f'{c}.ims' in written, f'Missing IMS for {c}'
            assert f'{c}.cncout.cnc' in written, f'Missing CNC file for {c}'
            assert f'{c}.gwfgwt' in written, f'Missing exchange for {c}'

    def test_dsp_options_and_mst_override(self, tmp_path, benchmark_database):
        """xt3d_off survives the clone and set_mst_override reaches the generated MST.

        Regression for two silent from_mf6 defects: the DSP OPTIONS block was
        dropped, so every component fell back to XT3D (the MF6 default) even when
        the reference model set xt3d_off; and the MST was cloned verbatim from the
        tracer template, so per-component sorption was lost.
        """
        import flopy
        sim_ws = tmp_path / 'dsp_mst'
        sim_ws.mkdir()
        sim = self._build_minimal_sim(sim_ws)
        # xt3d_off is an OPTIONS keyword, not griddata — the clone must carry it
        gwt_ref = sim.get_model('gwt')
        flopy.mf6.ModflowGwtdsp(gwt_ref, xt3d_off=True, alh=1.0, ath1=0.1,
                                filename='gwt.dsp')

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        assert 'Ca' in model.components
        model.set_diffusion_coeff({c: 0.0 for c in model.components})
        model.set_mst_override({'Ca': {'sorption': 'linear',
                                       'bulk_density': 1850.0,
                                       'distcoef': 2.1141e-4}})
        model.write_simulation()

        for c in model.components:
            dsp = (Path(model.wd) / f'{c}.dsp').read_text().upper()
            assert 'XT3D_OFF' in dsp, f'XT3D_OFF dropped for {c}'
            mst = (Path(model.wd) / f'{c}.mst').read_text().upper()
            if c == 'Ca':
                assert 'SORPTION' in mst, 'MST override did not reach Ca.mst'
            else:
                assert 'SORPTION' not in mst, f'sorption leaked into {c}.mst'

    def test_aux_per_period_switches_solution(self, tmp_path, benchmark_database):
        """type='aux' with a per-period dict writes distinct aux columns per period.

        Regression for from_mf6 aux ChemStress: set_spd({0:[2], 1:[3]}) must expand
        the tracer aux into one column per component *per stress period* (period 0 =
        solution 2, period 1 = solution 3). Previously the aux path indexed cs.data by
        cell only, so a per-period dict produced wrong-width rows (MFDataException) and
        never switched solution.
        """
        sim_ws = tmp_path / 'perperiod'
        sim_ws.mkdir()
        sim = self._build_minimal_sim(sim_ws, nper=2)

        # three solutions (1,2,3) so per-period numbers 2 and 3 are valid
        solutions = Solutions({'Ca': [1e-4, 1e-3, 5e-3], 'Cl': [2e-4, 2e-3, 5e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()
        ncomps = len(model.components)

        chdin_cs = ChemStress('chdin', type='aux')
        chdin_cs.set_spd({0: [2], 1: [3]})
        model.set_chem_stress(chdin_cs)

        # must not raise MFDataException about the wrong number of columns
        model.write_simulation()

        gwf_name = next(
            n for n in model._gwt_sim.model_names
            if model._gwt_sim.get_model(n).model_type == 'gwf6'
        )
        pkg = model._gwt_sim.get_model(gwf_name).get_package('chdin')
        spd = pkg.stress_period_data.get_data()

        # chdin record is (cellid, head, *concs) -> width = 2 base fields + ncomps,
        # and the trailing ncomps fields are the component aux columns.
        rec0 = tuple(spd[0][0])
        rec1 = tuple(spd[1][0])
        assert len(rec0) == 2 + ncomps
        assert len(rec1) == 2 + ncomps
        aux0 = rec0[-ncomps:]
        aux1 = rec1[-ncomps:]

        # each period carries its mapped solution's equilibrated concentrations
        np.testing.assert_allclose(aux0, tuple(model.chdin.data[0][0]))
        np.testing.assert_allclose(aux1, tuple(model.chdin.data[1][0]))

        # the solution actually switched between the two periods
        assert aux0 != aux1

    def test_cnc_missing_cells_raises(self, tmp_path, benchmark_database):
        """cnc ChemStress without set_cells raises ValueError at write_simulation."""
        sim_ws = tmp_path / 'conservative'
        sim_ws.mkdir()
        sim = self._build_minimal_sim(sim_ws)

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        cnc_cs = ChemStress('cncout', type='cnc')
        cnc_cs.set_spd([1])
        # intentionally omit set_cells
        model.set_chem_stress(cnc_cs)

        with pytest.raises(ValueError, match="cells is not set"):
            model.write_simulation()

    @staticmethod
    def _build_minimal_sim_rcha(sim_ws):
        """Minimal flopy sim: GWF with array recharge (RCHA) + tracer GWT.

        RCHA carries a single ``tracer`` aux array; GWT SSM sources it. Mirrors
        ``_build_minimal_sim`` but exercises the array-recharge aux path.
        """
        import flopy
        sim = flopy.mf6.MFSimulation(
            sim_name='test', sim_ws=str(sim_ws), exe_name='mf6'
        )
        flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)], time_units='days')

        gwf = flopy.mf6.ModflowGwf(sim, modelname='gwf', save_flows=True)
        ims_gwf = flopy.mf6.ModflowIms(sim, filename='gwf.ims')
        sim.register_ims_package(ims_gwf, ['gwf'])

        flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=3,
                                 delr=1.0, delc=1.0, top=0.0, botm=-1.0,
                                 filename='gwf.dis')
        flopy.mf6.ModflowGwfnpf(gwf, k=1.0, filename='gwf.npf')
        flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename='gwf.ic')
        flopy.mf6.ModflowGwfchd(
            gwf, stress_period_data=[[(0, 0, 2), 0.0]],
            pname='chdout', filename='gwf.chdout.chd',
        )
        flopy.mf6.ModflowGwfrcha(
            gwf, recharge=0.001, auxiliary=['tracer'], aux={0: 0.0},
            pname='rcha', filename='gwf.rcha',
        )
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord='gwf.hds',
                                budget_filerecord='gwf.cbb',
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])

        gwt = flopy.mf6.ModflowGwt(sim, modelname='gwt')
        ims_gwt = flopy.mf6.ModflowIms(
            sim, linear_acceleration='BICGSTAB', filename='gwt.ims'
        )
        sim.register_ims_package(ims_gwt, ['gwt'])

        flopy.mf6.ModflowGwtdis(gwt, nlay=1, nrow=1, ncol=3,
                                  delr=1.0, delc=1.0, top=0.0, botm=-1.0,
                                  filename='gwt.dis')
        flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename='gwt.ic')
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[['rcha', 'aux', 'tracer']], filename='gwt.ssm'
        )
        flopy.mf6.ModflowGwtadv(gwt, scheme='UPSTREAM')
        flopy.mf6.ModflowGwtmst(gwt, porosity=0.3, filename='gwt.mst')
        flopy.mf6.ModflowGwtoc(gwt, budget_filerecord='gwt.cbc',
                                 concentration_filerecord='gwt.ucn',
                                 saverecord=[('CONCENTRATION', 'ALL')])
        flopy.mf6.ModflowGwfgwt(sim, exgtype='GWF6-GWT6',
                                  exgmnamea='gwf', exgmnameb='gwt',
                                  filename='gwf-gwt.gwfgwt')
        return sim

    def test_rcha_aux_array_structure(self, tmp_path, benchmark_database):
        """RCHA aux becomes per-component grid arrays filled with equilibrated concs."""
        sim_ws = tmp_path / 'conservative'
        sim_ws.mkdir()
        sim = self._build_minimal_sim_rcha(sim_ws)

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        components = model.components
        assert len(components) > 0

        rcha_cs = ChemStress('rcha', type='aux')
        rcha_cs.set_spd([2])  # uniform: single source solution
        model.set_chem_stress(rcha_cs)

        model.write_simulation()

        # Inspect the reactive GWF rcha package in the written simulation.
        gwf = model._gwt_sim.get_model('gwf')
        rcha = gwf.get_package('rcha')

        # auxiliary names must be the components, not the original 'tracer'.
        # After write_simulation, get_data() returns a recarray whose single row
        # is ('auxiliary', comp0, comp1, ...); drop the leading tag field.
        aux_row = np.atleast_1d(rcha.auxiliary.get_data()).tolist()[0]
        aux_names = [str(c) for c in aux_row[1:]]
        assert aux_names == list(components)

        # aux is a per-component grid array (naux, nrow, ncol), uniform fill.
        aux_arr = rcha.aux.get_data()[0]
        assert aux_arr.shape == (len(components), 1, 3)
        for j in range(len(components)):
            assert np.allclose(aux_arr[j], aux_arr[j].flat[0])

        # Each reactive GWT sources rcha via SSM AUX.
        written = set(os.listdir(model.wd))
        assert 'gwf.rcha' in written
        for c in components:
            assert f'{c}.ssm' in written, f'Missing SSM for {c}'

    def test_rcha_multiple_solutions_raises(self, tmp_path, benchmark_database):
        """RCHA ChemStress with >1 source solution raises (uniform chemistry only)."""
        sim_ws = tmp_path / 'conservative'
        sim_ws.mkdir()
        sim = self._build_minimal_sim_rcha(sim_ws)

        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        rcha_cs = ChemStress('rcha', type='aux')
        rcha_cs.set_spd([1, 2])  # more than one solution is invalid for rcha
        model.set_chem_stress(rcha_cs)

        with pytest.raises(ValueError, match="uniform chemistry"):
            model.write_simulation()

    @staticmethod
    def _build_minimal_sim_disv(sim_ws, nper=1):
        """Minimal flopy sim on a **DISV** grid: GWF + tracer GWT with CHD aux.

        A hand-built 2x2 single-layer vertex grid (``ncpl=4``, ``nvert=9``) — no
        gridgen dependency. The inflow CHD (``chdin``, cell2d 0) carries a
        ``tracer`` aux for every stress period; ``chdout`` sits at cell2d 3. Mirrors
        :meth:`_build_minimal_sim` but exercises the DISV grid-copy path in
        ``_build_reactive_gwt_models``.
        """
        import flopy
        vertices = [
            [0, 0.0, 2.0], [1, 1.0, 2.0], [2, 2.0, 2.0],
            [3, 0.0, 1.0], [4, 1.0, 1.0], [5, 2.0, 1.0],
            [6, 0.0, 0.0], [7, 1.0, 0.0], [8, 2.0, 0.0],
        ]
        cell2d = [
            [0, 0.5, 1.5, 4, 0, 1, 4, 3],
            [1, 1.5, 1.5, 4, 1, 2, 5, 4],
            [2, 0.5, 0.5, 4, 3, 4, 7, 6],
            [3, 1.5, 0.5, 4, 4, 5, 8, 7],
        ]
        disv_kwargs = dict(nlay=1, ncpl=4, nvert=9, top=0.0, botm=-1.0,
                           vertices=vertices, cell2d=cell2d)

        sim = flopy.mf6.MFSimulation(
            sim_name='test', sim_ws=str(sim_ws), exe_name='mf6'
        )
        flopy.mf6.ModflowTdis(sim, nper=nper,
                              perioddata=[(1.0, 1, 1.0)] * nper,
                              time_units='days')

        gwf = flopy.mf6.ModflowGwf(sim, modelname='gwf', save_flows=True)
        ims_gwf = flopy.mf6.ModflowIms(sim, filename='gwf.ims')
        sim.register_ims_package(ims_gwf, ['gwf'])

        flopy.mf6.ModflowGwfdisv(gwf, filename='gwf.disv', **disv_kwargs)
        flopy.mf6.ModflowGwfnpf(gwf, k=1.0, filename='gwf.npf')
        flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename='gwf.ic')
        chdin_spd = {k: [[(0, 0), 1.0, 1.0]] for k in range(nper)}
        flopy.mf6.ModflowGwfchd(
            gwf, stress_period_data=chdin_spd,
            auxiliary=['tracer'], pname='chdin', filename='gwf.chdin.chd',
        )
        flopy.mf6.ModflowGwfchd(
            gwf, stress_period_data=[[(0, 3), 0.0]],
            pname='chdout', filename='gwf.chdout.chd',
        )
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord='gwf.hds',
                                budget_filerecord='gwf.cbb',
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])

        gwt = flopy.mf6.ModflowGwt(sim, modelname='gwt')
        ims_gwt = flopy.mf6.ModflowIms(
            sim, linear_acceleration='BICGSTAB', filename='gwt.ims'
        )
        sim.register_ims_package(ims_gwt, ['gwt'])

        flopy.mf6.ModflowGwtdisv(gwt, filename='gwt.disv', **disv_kwargs)
        flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename='gwt.ic')
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[['chdin', 'aux', 'tracer']], filename='gwt.ssm'
        )
        flopy.mf6.ModflowGwtadv(gwt, scheme='UPSTREAM')
        flopy.mf6.ModflowGwtmst(gwt, porosity=0.3, filename='gwt.mst')
        flopy.mf6.ModflowGwtoc(gwt, budget_filerecord='gwt.cbc',
                                 concentration_filerecord='gwt.ucn',
                                 saverecord=[('CONCENTRATION', 'ALL')])
        flopy.mf6.ModflowGwfgwt(sim, exgtype='GWF6-GWT6',
                                  exgmnamea='gwf', exgmnameb='gwt',
                                  filename='gwf-gwt.gwfgwt')
        return sim

    def test_from_mf6_disv_write_simulation(self, tmp_path, benchmark_database):
        """from_mf6 on a DISV grid writes reactive GWTs with a DISV package.

        Regression for the grid-type dispatch bug in ``_build_reactive_gwt_models``:
        it used ``gwf.get_package('dis')`` to detect the grid, but flopy prefix-
        matches ``'dis'`` to ``'disv'``, so a DISV model wrongly took the DIS branch
        and crashed with ``AttributeError: ... has no attribute 'nrow'`` at
        ``write_simulation``. The fix detects the grid via ``get_grid_type().name``.
        This asserts DISV survives write_simulation and every cloned reactive GWT
        carries a DISV package with the grid copied through.
        """
        sim_ws = tmp_path / 'disv'
        sim_ws.mkdir()
        sim = self._build_minimal_sim_disv(sim_ws)

        # two solutions so the aux ChemStress can map to solution 2
        solutions = Solutions({'Ca': [1e-4, 1e-3], 'Cl': [2e-4, 2e-3]})
        solutions.set_ic(1)

        model = Mup3d.from_mf6(sim, solutions, name='test', gwt_name='gwt')
        model.set_database(benchmark_database)
        model.initialize()

        chdin_cs = ChemStress('chdin', type='aux')
        chdin_cs.set_spd([2])
        model.set_chem_stress(chdin_cs)

        # pre-fix this raised AttributeError about 'nrow' on the DISV grid
        model.write_simulation()

        components = model.components
        assert len(components) > 0

        # tracer template GWT must be gone
        written = set(os.listdir(model.wd))
        tracer_files = [f for f in written
                        if f.startswith('gwt.') and not f.endswith('.ims')]
        assert len(tracer_files) == 0, f'Tracer model files remain: {tracer_files}'

        # every reactive GWT is DISV with the grid copied through (ncpl == 4)
        for c in components:
            gwt_model = model._gwt_sim.get_model(c)
            assert gwt_model.get_grid_type().name == 'DISV', \
                f'{c} GWT is not DISV'
            assert int(gwt_model.disv.ncpl.get_data()) == 4, \
                f'{c} DISV ncpl not copied'

