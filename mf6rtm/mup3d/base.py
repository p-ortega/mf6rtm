"""
Base module of the mup3d package
"""

import os
import warnings
import phreeqcrm
import shutil
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Union
from pathlib import Path
from contextlib import contextmanager

import flopy

from mf6rtm.simulation.solver import solve
from mf6rtm.utils import utils
from mf6rtm.config import MF6RTMConfig


class Block:
    """Base class for PHREEQC input "keyword data blocks".

    Attributes
    ----------
    data : dict
        Dictionary of geochemical components (keys) and their total concentrations
        (list) and other parameters, indexed by block number, similar to a .pqi file.
    names : list
        List of names of geochemical components that serve as keys to the data.
    ic : array
        Initial condition.
    eq_solutions : list
        List of equilibrium solutions.
    options : list
        List of options.
    """
    def __init__(
        self,
        data: dict,
        ic: Union[int, float, np.ndarray, None] = None,
    ) -> None:
        """Initialize a Block instance with inputs from a PHREEQC data block.

        Parameters
        ----------
        data
            PHREEQC components (keys) and their total concentrations (list) indexed by
            block number, similar to a .pqi file.
        ic, optional
            Initial condition concentrations. Default is None.
        """
        self.data = data
        self.ic = ic  #: None means no initial condition (-1)
        self.eq_solutions = None
        self.options = []
        self.get_names()

    def get_names(self):
        """Get the names of geochemical components or phases specified in the block.

        Returns
        -------
        list
            List of names of geochemical components that serve as keys to the data.
        """
        if isinstance(self, Solutions):
            self.names = sorted(self.data.keys())
        else:
            block_names = []
            for block_num in self.data:
                block_names.extend(list(self.data[block_num].keys()))
            self.names = sorted(list(set(block_names)))
        return self.names

    def set_ic(self, ic: Union[int, float, np.ndarray]):
        """Set the initial condition for the block.

        Parameters
        ----------
        ic
            Initial condition concentrations. Can be an int, float, or ndarray.
        Returns
        -------
        None
        """
        assert isinstance(ic, (int, float, np.ndarray)), 'ic must be an int, float or ndarray'
        self.ic = ic

    def set_equilibrate_solutions(self, eq_solutions) -> None:
        """Set the equilibrium solutions for the exchange phases.
        Array where index is the exchange phase number and value
        is the solution number to equilibrate with.

        Parameters
        ----------
        eq_solutions
            List of equilibrium solution indices for each exchange phase.

        Returns
        -------
        None
        """
        self.eq_solutions = eq_solutions

    def set_options(self, options) -> None:
        """Set the options for the block.

        Parameters
        ----------
        options
            List of options for the block.
        Returns
        -------
        None
        """
        self.options = options

class GasPhase(Block):
    """The GasPhase Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the gas phase. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)

class Solutions(Block):
    """The Solutions Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the solutions. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)

class EquilibriumPhases(Block):
    """The EquilibriumPhases Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the equilibrium phases. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        self.data = utils.fill_missing_minerals(data)

class ExchangePhases(Block):
    """The ExchangePhases Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the exchange phases. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)

class KineticPhases(Block):
    """The KineticPhases Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the kinetic phases. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        self.data = utils.fill_missing_minerals(data)
        self.parameters = None

    def set_parameters(self, parameters):
        """Set the kinetic phase parameters.

        Parameters
        ----------
        parameters : dict
            Parameters for the kinetic phases, keyed by mineral/phase name.
        """
        self.parameters = parameters

class Surfaces(Block):
    """The Surfaces Block.

    Attributes
    ----------
    parameters : dict, optional
        Dictionary of parameters for the surfaces. Default is None.
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)

class ChemStress():
    """The ChemStress class for handling stress period data.

    Attributes
    ----------
    packnme : str
        Name of the package.
    type : str
        Boundary coupling type: ``'aux'`` (GWF stress package + GWT SSM),
        ``'cnc'`` (GWT constant-concentration), or ``'src'`` (GWT mass-loading).
        Default is ``'aux'``.
    sol_spd : list, optional
        List of solution indices (one per boundary cell). Default is None.
    cells : list, optional
        List of cellids — required for cnc/src; for aux the cells come from
        the GWF package SPD. Default is None.

    Notes
    -----
    # TODO: advanced MF6 transport packages (LKT, MVT, UZT, SFT, MWT, IST) are
    # not yet supported. These require per-component re-equilibration analogous
    # to the aux/cnc path and will be added in a future release.
    """
    def __init__(self, packnme, type='aux') -> None:
        self.packnme = packnme
        self.type = type
        self.sol_spd = None
        self.cells = None

    def set_spd(self, sol_spd):
        """Set the solution stress period data.

        Parameters
        ----------
        sol_spd : list of int
            Solution indices, one per boundary cell, mapping each cell to a
            PHREEQC solution number.
        """
        self.sol_spd = sol_spd

    def set_type(self, type):
        """Set the boundary coupling type.

        Parameters
        ----------
        type : {'aux', 'cnc', 'src'}
            Boundary coupling type: ``'aux'`` (GWF stress package + GWT SSM),
            ``'cnc'`` (GWT constant-concentration), or ``'src'`` (GWT
            mass-loading).
        """
        self.type = type

    def set_cells(self, cells):
        """Set the boundary cell ids.

        Required for ``'cnc'`` and ``'src'`` types; for ``'aux'`` the cells
        are taken from the GWF package stress period data.

        Parameters
        ----------
        cells : list of tuple
            Cell ids, e.g. ``(layer, row, col)`` for DIS or
            ``(layer, cell2d)`` for DISV.
        """
        self.cells = cells


phase_types = {
    'KineticPhases': KineticPhases,
    'ExchangePhases': ExchangePhases, # TODO: Exchange has to be abstracted to be used with this methods
    'EquilibriumPhases': EquilibriumPhases,
    'Surfaces': Surfaces,
}


class Mup3d(object):
    """The Mup3d class wrapper and extension for a PhreeqcRM model class.

    This class extends the PhreeqcRM class to include additional methods
    that facilitate the coupling to Modflow6 via ModflowAPI.

    Attributes
    ----------
    name : str
        Name of the model.
    wd : str
        Working directory path.
    charge_offset : float
        Charge offset, initialized to 0.0.
    database : str
        Path to the PHREEQC database file.
    solutions : Solutions
        Solutions instance containing the geochemical data.
    init_temp : float, optional
        Initial temperature. Default is 25.0.
    equilibrium_phases : EquilibriumPhases
        Equilibrium phases in the model.
    kinetic_phases : KineticPhases
        Kinetic phases in the model.
    exchange_phases : ExchangePhases
        Exchange phases in the model.
    surfaces_phases : Surfaces
        Surface phases in the model.
    postfix : str
        Postfix for the output files.
    phreeqc_rm : object
        PHREEQC reactive transport model instance.
    init_conc_array_phreeqc : ndarray
        1D array of concentrations (mol/L) structured for PhreeqcRM, with each
        component concentration for each grid cell ordered by model.components.
    sconc : dict[str, np.ndarray]
        Dictionary of concentrations in units of moles per m^3 and structured to
        match the shape of Modflow's grid.
    phinp : object
        PHREEQC input instance.
    components : list
        List of chemical components.
    fixed_components : list
        List of fixed components.
    nlay : int
        Number of layers in the model grid.
    nrow : int
        Number of rows in the model grid.
    ncol : int
        Number of columns in the model grid.
    nxyz : int
        Total number of cells in the model grid, either
        (nlay * nrow * ncol) if DIS or
        (nlay * ncpl) if DISV or
        (nxyz) if DISU.
    grid_shape : tuple
        Shape of the model grid, either
        (nlay, nrow, ncol) if DIS or
        (nlay, ncpl) if DISV or
        (nxyz) if DISU.
    """
    def __init__(
        self,
        name: Union[str, None] = None,
        solutions: Union[Solutions, None] = None,
        nlay: Union[int, None] = None,
        nrow: Union[int, None] = None,
        ncol: Union[int, None] = None,
        ncpl: Union[int, None] = None,
        nxyz: Union[int, None] = None,
    ):
        """Initializes a Mup3d instance with the given parameters.

        Parameters
        ----------
        name : str, optional
            The name of the model. Default is None.
        solutions : Solutions, optional
            Instance of the Solutions class containing the geochemical data.
            Required if name is not an instance of Solutions.
        nlay : int, optional
            Number of layers in the model, if it has a layered grid
            discretization (DIS or DISV).
        nrow : int, optional
            Number of rows in the model, if it has a structured rectangular
            layered grid discretization (DIS).
        ncol : int, optional
            Number of columns in the model, if it has a structured rectangular
            layered grid discretization (DIS).
        ncpl : int, optional
            Number of cells per layer in the model, if it has an unstructured
            layered grid Discretization by Vertices (DISV).
        nxyz : int, optional
            Total number of cells in the model grid, either
            (nlay * nrow * ncol) if DIS or
            (nlay * ncpl) if DISV or
            (nxyz) if DISU.

        Raises
        ------
        ValueError
            If solutions is not provided.
        ValueError
            If any of nlay, nrow, or ncol is not provided.
        """
        if solutions is None and isinstance(name, Solutions):
            # New style: first argument is solutions
            solutions = name
            name = None
        # Validate required parameters
        if solutions is None:
            raise ValueError("solutions parameter is required")
        if (any(param is None for param in [nlay, nrow, ncol]) and
            any(param is None for param in [nlay, ncpl])):
            raise ValueError(("nlay, nrow, and ncol parameters are required "
                " for DIS, or nlay and ncpl parameters are required for DISV"))
        self.name = name
        self.wd = None
        self.charge_offset = 0.0
        self.database = os.path.join('pht3d_datab.dat')
        self.solutions = solutions
        self.init_temp = 25.0
        self.equilibrium_phases = None
        self.kinetic_phases = None
        self.exchange_phases = None
        self.surfaces_phases = None
        self.postfix = None
        # self.gas_phase = None
        # self.solid_solutions = None
        self.phreeqc_rm = None
        self.sconc = None
        self.phinp = None
        self.components = None
        self.fixed_components = None
        self.componenth2o = False
        self.config = MF6RTMConfig() #default config
        self._gwt_sim = None
        self._gwt_name = None
        self._diffusion_coeff = {}
        self._mst_overrides = {}

        # Set grid parameters for DIS
        if all(param is not None for param in [nlay, nrow, ncol]):
            self.nlay = int(nlay)
            self.nrow = int(nrow)
            self.ncol = int(ncol)
            self.nxyz = self.nlay * self.nrow * self.ncol
            self.grid_shape = (self.nlay, self.nrow, self.ncol)
        # Set grid parameters for DISV
        elif all(param is not None for param in [nlay, ncpl]):
            self.nlay = int(nlay)
            self.ncpl = int(ncpl)
            self.nxyz = self.nlay * self.ncpl
            self.grid_shape = (self.nlay, self.ncpl)
        # Set grid parameters for DISU
        elif nxyz is not None:
            self.nxyz = int(nxyz)
            self.grid_shape = (self.nxyz,)

        if self.solutions.ic is None:
            self.solutions.ic = [1]*self.nxyz
        if isinstance(self.solutions.ic, (int, float)):
            self.solutions.ic = np.reshape([self.solutions.ic]*self.nxyz, self.grid_shape)
            # print(self.solutions.ic.shape, self.nxyz, self.grid_shape)
        assert self.solutions.ic.shape == self.grid_shape, (
            f'Initial conditions array must be an array of the shape ({self.grid_shape})'
            f'not {self.solutions.ic.shape}'
        )
    def set_componenth2o(self, flag):
        """Set the component H2O to be True or False.
        if False Total H and Total O are transported
        if True H2O, Excess H and Excess O are transported

        Parameters
        ----------
        flag : bool
            True to include H2O as a component, False otherwise.

        Returns
        -------
        bool
            The value of the componenth2o flag.
        Raises
        ------
        AssertionError
            If flag is not a boolean.
        """

        assert isinstance(flag, bool), f"flag must be a boolean, got {type(flag).__name__}"
        self.componenth2o = flag
        return self.componenth2o

    def get_componenth2o(self):
        """Get componenth2o flag

        Returns
        -------
        bool
            The value of the componenth2o flag.
        """
        return getattr(self, 'componenth2o')

    def set_fixed_components(self, fixed_components):
        """Set the fixed components for the MF6RTM model.
        These are the components that are not transported during the simulation.

        Parameters
        ----------
        fixed_components : list
            List of component names to be fixed (not transported).
        Returns
        -------
        None
        """
        # FIXME: implemented but commented in main coupling loop
        self.fixed_components = fixed_components

    def set_initial_temp(self, temp):
        """Sets the initial temperature for the MF6RTM model.

        Parameters
        ----------
        temp : int, float, or list
            Initial temperature value(s). Can be a single value (int or float)
            for homogeneous temperature or a list for spatially variable temperature.
        Returns
        -------
        None
        """
        assert isinstance(temp, (int, float, list)), 'temp must be an int or float'
        # TODO: for non-homogeneous fields allow 3D and 2D arrays
        self.init_temp = temp

    def set_phases(self, phase):
        """Sets the phases for the MF6RTM model.

        Parameters
        ----------
        phase : KineticPhases, ExchangePhases, EquilibriumPhases, or Surfaces
            Instance of one of the phase classes containing geochemical data.
        Returns
        -------
        None
        """
        # Dynamically get the class of the phase object
        phase_class = phase.__class__

        # Check if the phase object's class is in the dictionary of phase types
        if phase_class not in phase_types.values():
            raise AssertionError(f'{phase_class.__name__} is not a recognized phase type')

        # Proceed with the common logic
        if isinstance(phase.ic, (int, float)):
            phase.ic = np.reshape([phase.ic]*self.nxyz, self.grid_shape)
        phase.data = {i: phase.data[key] for i, key in enumerate(phase.data.keys())}
        assert phase.ic.shape == self.grid_shape, f'Initial conditions array must be an array of the shape {self.grid_shape} not {phase.ic.shape}'

        # Dynamically set the phase attribute based on the class name
        setattr(self, f"{phase_class.__name__.lower().split('phases')[0]}_phases", phase)

    def set_exchange_phases(self, exchanger):
        """Sets the exchange phases for the MF6RTM model.

        Parameters
        ----------
        exchanger : ExchangePhases
            Instance of the ExchangePhases class containing geochemical data.
        Returns
        -------
        None
        """
        assert isinstance(exchanger, ExchangePhases), 'exchanger must be an instance of the Exchange class'
        # exchanger.data = {i: exchanger.data[key] for i, key in enumerate(exchanger.data.keys())}
        if isinstance(exchanger.ic, (int, float)):
            exchanger.ic = np.reshape([exchanger.ic]*self.nxyz, self.grid_shape)
        assert exchanger.ic.shape == self.grid_shape, f'Initial conditions array must be an array of the shape {self.grid_shape} not {exchanger.ic.shape}'
        self.exchange_phases = exchanger

    def set_equilibrium_phases(self, eq_phases):
        """Sets the equilibrium phases for the MF6RTM model.

        Parameters
        ----------
        eq_phases : EquilibriumPhases
            Instance of the EquilibriumPhases class containing geochemical data.
        Returns
        -------
        None
        """
        assert isinstance(eq_phases, EquilibriumPhases), 'eq_phases must be an instance of the EquilibriumPhases class'
        # change all keys from eq_phases so they start from 0
        eq_phases.data = {i: eq_phases.data[key] for i, key in enumerate(eq_phases.data.keys())}
        self.equilibrium_phases = eq_phases
        if isinstance(self.equilibrium_phases.ic, (int, float)):
            self.equilibrium_phases.ic = np.reshape([self.equilibrium_phases.ic]*self.nxyz, self.grid_shape)
        assert self.equilibrium_phases.ic.shape == self.grid_shape, f'Initial conditions array must be an array of the shape ({self.grid_shape}) not {self.equilibrium_phases.ic.shape}'

    def set_charge_offset(self, charge_offset):
        """
        Sets the charge offset for the MF6RTM model to handle negative charge values

        Parameters
        ----------
        charge_offset : float
            The charge offset value to be added to the charge concentration.
        Returns
        -------
        None
        """
        self.charge_offset = charge_offset

    def set_chem_stress(
            self,
            chem_stress: ChemStress,
    ) -> None:
        """
        Sets the ChemStress instance for the MF6RTM model.

        Parameters
        ----------
        chem_stress : ChemStress
            Instance of the ChemStress class containing stress period data.
        Returns
        -------
        None
        """
        assert isinstance(chem_stress, ChemStress), 'chem_stress must be an instance of the ChemStress class'
        attribute_name = chem_stress.packnme
        setattr(self, attribute_name, chem_stress)

        self.initialize_chem_stress(attribute_name)

    def set_wd(self, wd):
        """Set the working directory for the MF6RTM model.

        The directory is created if it does not already exist.

        Parameters
        ----------
        wd : str
            The path to the working directory.
        """
        # get absolute path of the working directory
        wd = Path(os.path.abspath(wd))
        # joint current directory with wd, check if exist, create if not
        if not wd.exists():
            wd.mkdir(parents=True, exist_ok=True)
        self.wd = wd

    def set_database(self, database):
        """Set the database for the MF6RTM model.

        Copies the database file into the working directory for
        self-containment.

        Parameters
        ----------
        database : str
            The path to the database file.
        """
        try:
            assert os.path.exists(database), f"{database} not found"
            database = os.path.abspath(database)
            # database not in wd so copy it there for self containment
            shutil.copy(database, os.path.join(self.wd, os.path.basename(database)))
        except AssertionError:
            try:
                alt_path = os.path.join(self.wd, database)
                assert os.path.exists(alt_path), f"{database} not found inside the model dir"
                database = alt_path  # update to the valid path
            except AssertionError:
                print(f"Couldn't find the database in '{database}' or '{alt_path}'")

        # get absolute path of the database
        self.database = database

    def set_postfix(self, postfix):
        """Set the postfix file for the MF6RTM model.

        Parameters
        ----------
        postfix : str
            The path to the postfix file.

        Raises
        ------
        AssertionError
            If the postfix file does not exist.
        """
        assert os.path.exists(postfix), f'{postfix} not found'
        self.postfix = postfix

    def set_reaction_temp(self):
        """Sets the reaction temperature for the MF6RTM model.

        Returns
        -------
        list
            List of reaction temperatures for each grid cell.
        """
        if isinstance(self.init_temp, (int, float)):
            rx_temp = [self.init_temp]*self.nxyz
            print('Using temperatue of {} for all cells'.format(rx_temp[0]))
        elif isinstance(self.init_temp, (list, np.ndarray)):
            rx_temp = [self.init_temp[0]]*self.nxyz
            print('Using temperatue of {} from SOLUTION 1 for all cells'.format(rx_temp[0]))
        self.reaction_temp = rx_temp
        return rx_temp

    @staticmethod
    def _resolve_charge_species(add_charge_flag):
        """Resolve the ``add_charge_flag`` argument to a single species name.

        The PHREEQC ``charge`` keyword balances a solution by adjusting exactly
        one species, so the flag may target only one component.

        Parameters
        ----------
        add_charge_flag : bool or str
            ``True`` applies the flag to ``"pH"`` (the default/legacy behavior).
            A string applies it to that single component instead, e.g. ``"Cl"``.

        Returns
        -------
        str
            The single species name that should receive the charge flag.
        """
        if add_charge_flag is True:
            return "pH"
        if isinstance(add_charge_flag, str):
            return add_charge_flag
        # accept a 1-element list/tuple for convenience, but never more
        if isinstance(add_charge_flag, (list, tuple)):
            if len(add_charge_flag) == 1 and isinstance(add_charge_flag[0], str):
                return add_charge_flag[0]
            raise ValueError(
                "The charge flag can only be applied to one component; "
                f"got {add_charge_flag!r}"
            )
        raise TypeError(
            "add_charge_flag must be True (defaults to 'pH') or a single component "
            f"name as a string; got {type(add_charge_flag).__name__}"
        )

    def generate_phreeqc_script(self, add_charge_flag=False):
        """
        Generates the phinp file for the MF6RTM model.

        Parameters
        ----------
        add_charge_flag : bool or str, optional
            Add the PHREEQC ``charge`` balancing flag to one component in every
            SOLUTION block. ``False`` (default) disables it. ``True`` applies it
            to ``"pH"`` (legacy behavior). Pass a single component name (e.g.
            ``"Cl"``) to balance on that component instead. Only one component
            may carry the flag.
        Returns
        -------
        str
            The generated PHREEQC script as a string.
        """

        # where to save the phinp file
        filename = os.path.join(self.wd, 'phinp.dat')
        self.phinp = filename
        # assert that database in self.database exists
        assert os.path.exists(self.database), f'{self.database} not found inside the model dir'

        # Check if all compounds are in the database
        names = utils.get_compound_names(self.database)
        assert all([key in names for key in self.solutions.data.keys() if key not in ["pH", "pe"]]), f'Not all compounds are in the database - check: {", ".join([key for key in self.solutions.data.keys() if key not in names and key not in ["pH", "pe"]])}'

        script = ""

        # Convert single values to lists
        for key, value in self.solutions.data.items():
            if not isinstance(value, list):
                self.solutions.data[key] = [value]
        # replace all values in self.solutinons.data that are 0.0 to a very small number
        for key, value in self.solutions.data.items():
            # self.solutions.data[key] = [1e-30 if val == 0.0 else val for val in value]
            self.solutions.data[key] = [val for val in value]

        # Get the number of solutions
        num_solutions = len(next(iter(self.solutions.data.values())))

        # Initialize the list of previous concentrations and phases

        for i in range(num_solutions):
            # Get the current concentrations and phases
            concentrations = {species: values[i] for species, values in self.solutions.data.items()}
            script += utils.handle_block(concentrations, utils.generate_solution_block, i, temp=self.init_temp, water=1)

        # check if self.equilibrium_phases is not None
        if self.equilibrium_phases is not None:
            for i in self.equilibrium_phases.data.keys():
                # Get the current   phases
                phases = self.equilibrium_phases.data[i]
                # check if all equilibrium phases are in the database
                names = utils.get_compound_names(self.database, 'PHASES')
                assert all([key in names for key in phases.keys()]), 'Following phases are not in database: '+', '.join(f'{key}' for key in phases.keys() if key not in names)

                # Handle the  EQUILIBRIUM_PHASES blocks
                script += utils.handle_block(phases, utils.generate_equ_phases_block, i)

        # check if self.exchange_phases is not None
        if self.exchange_phases is not None:
            for i in self.exchange_phases.data.keys():
                # Get the current   phases
                phases = self.exchange_phases.data[i]
                # check if all equilibrium phases are in the database
                names = utils.get_compound_names(self.database, 'EXCHANGE')
                assert all([key in names for key in phases.keys()]), 'Following phases are not in database: '+', '.join(f'{key}' for key in phases.keys() if key not in names)
                assert self.exchange_phases.eq_solutions is not None, 'No equilibrate solutions defined'
                assert isinstance(self.exchange_phases.eq_solutions, (list, np.ndarray)), "exchange_phases.eq_solutions must be a list or numpy array"
                assert len(self.exchange_phases.data.keys()) == len(self.exchange_phases.eq_solutions), "Mismatch between number of exchangers and eq_solutions"
                # Handle the  EQUILIBRIUM_PHASES blocks
                script += utils.handle_block(phases, utils.generate_exchange_block, i, equilibrate_solutions=self.exchange_phases.eq_solutions[i])

        # check if self.kinetic_phases is not None
        if self.kinetic_phases is not None:
            for i in self.kinetic_phases.data.keys():
                # Get the current   phases
                phases = self.kinetic_phases.data[i]
                # check if all kinetic phases are in the database
                names = []
                for blocknme in ['PHASES', 'SOLUTION_MASTER_SPECIES']:
                    names += utils.get_compound_names(self.database, blocknme)

                assert all([key in names for key in phases.keys()]), 'Following phases are not in database: '+', '.join(f'{key}' for key in phases.keys() if key not in names)

                script += utils.handle_block(phases, utils.generate_kinetics_block, i)

        if self.surfaces_phases is not None:
            for i in self.surfaces_phases.data.keys():
                # Get the current   phases
                phases = self.surfaces_phases.data[i]
                # check if all surfaces are in the database
                names = utils.get_compound_names(self.database, 'SURFACE_MASTER_SPECIES')
                assert all([key in names for key in phases.keys()]), 'Following phases are not in database: '+', '.join(f'{key}' for key in phases.keys() if key not in names)
                # Equilibrate each surface with its zone's solution (mirrors the
                # exchange path); fall back to solution 1 if no targets were set.
                eq_sol = 1 if self.surfaces_phases.eq_solutions is None else self.surfaces_phases.eq_solutions[i]
                script += utils.handle_block(phases, utils.generate_surface_block, i, equilibrate_solutions=eq_sol, options=self.surfaces_phases.options)

        # add end of line before postfix
        script += utils.endmainblock

        # Append the postfix file to the script
        if self.postfix is not None and os.path.isfile(self.postfix):
            with open(self.postfix, 'r') as source:  # Open the source file in read mode
                script += '\n'
                script += source.read()

        if add_charge_flag:
            charge_species = self._resolve_charge_species(add_charge_flag)
            if charge_species not in self.solutions.data:
                print(
                    f"WARNING: charge flag component '{charge_species}' not found in "
                    f"solution species {sorted(self.solutions.data.keys())}; "
                    "no 'charge' keyword will be added."
                )
            script = utils.add_charge_flag_to_species_in_solution(
                script, species=[charge_species]
            )

        with open(filename, 'w') as file:
            file.write(script)
        return script

    def initialize(self, nthreads=1, add_charge_flag=False):
        """Initialize a PhreeqcRM object and calculate initial concentrations.

        This method initializes a PhreeqcRM object using PHREEQC inputs and adds several
        key attributes to the Mup3d object for reactive transport modeling.

        Parameters
        ----------
        nthreads : int, optional
            Number of threads for parallel processing. Default is 1.
        add_charge_flag : bool or str, optional
            Add the PHREEQC ``charge`` balancing flag to one component. ``False``
            (default) disables it, ``True`` applies it to ``"pH"``, and a single
            component name (e.g. ``"Cl"``) balances on that component instead.

        Attributes Added
        ---------------
        components : list
            List of transportable chemical components.
        init_conc_array_phreeqc : ndarray
            1D array of concentrations (mol/L) structured for PhreeqcRM, with each
            component concentration for each grid cell ordered by model.components.
        sconc : dict
            Dictionary with components as keys and concentration arrays (mol/m^3) as values,
            structured to match the shape of the Modflow6 model domain grid.
        phreeqc_rm : PhreeqcRM
            Initialized PhreeqcRM object.
        nchem : int
            Number of chemistry cells.

        Returns
        -------
        None

        Notes
        -----
        This method performs several key initialization steps:
        1. Generates PHREEQC input script
        2. Initializes PhreeqcRM object
        3. Sets up initial conditions
        4. Calculates initial concentrations
        5. Converts concentrations to proper units and grid structure
        """
        # get model dis info
        # dis = sim.get_model(sim.model_names[0]).dis

        # create phinp
        # check if phinp.dat is in wd
        phinp = self.generate_phreeqc_script(add_charge_flag=add_charge_flag)

        # initialize phreeqccrm object
        self.phreeqc_rm = phreeqcrm.PhreeqcRM(self.nxyz, nthreads)
        status = self.phreeqc_rm.SetComponentH2O(self.componenth2o)
        self.phreeqc_rm.UseSolutionDensityVolume(False)

        # Open files for phreeqcrm logging
        status = self.phreeqc_rm.SetFilePrefix(os.path.join(self.wd, '_phreeqc'))
        self.phreeqc_rm.OpenFiles()

        # Set concentration units
        status = self.phreeqc_rm.SetUnitsSolution(2)
            # 1, mg/L; 2, mol/L; 3, mass fraction, kg/kgs
        # status = self.phreeqc_rm.SetUnitsExchange(1)
        # status = self.phreeqc_rm.SetUnitsSurface(1)
        # status = self.phreeqc_rm.SetUnitsKinetics(1)

        # mf6 handles poro . set to 1
        poro = np.full((self.nxyz), 1.)
        status = self.phreeqc_rm.SetPorosity(poro)

        print_chemistry_mask = np.full((self.nxyz), 1)
        status = self.phreeqc_rm.SetPrintChemistryMask(print_chemistry_mask)
        nchem = self.phreeqc_rm.GetChemistryCellCount()
        self.nchem = nchem

        # Set printing of chemistry file
        status = self.phreeqc_rm.SetPrintChemistryOn(False, True, False)  # workers, initial_phreeqc, utility

        # Load database
        status = self.phreeqc_rm.LoadDatabase(self.database)
        utils.check_phreeqcrm_status(self.phreeqc_rm, status, "LoadDatabase")
        status = self.phreeqc_rm.RunFile(True, True, True, self.phinp)
        utils.check_phreeqcrm_status(self.phreeqc_rm, status, "RunFile")

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = self.phreeqc_rm.RunString(True, False, True, input)
        utils.check_phreeqcrm_status(self.phreeqc_rm, status, "RunString")

        # Get component information - these two functions need to be invoked to find comps
        ncomps = self.phreeqc_rm.FindComponents()
        utils.check_phreeqcrm_status(self.phreeqc_rm, ncomps, "FindComponents")
        components = list(self.phreeqc_rm.GetComponents())
        self.ncomps = ncomps

        # set components as attribute
        self.components = components

        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = self.phreeqc_rm.SetTime(time)
        status = self.phreeqc_rm.SetTimeStep(time_step)

        ic1 = np.ones((self.nxyz, 7), dtype=int)*-1

        # this gets a column slice
        ic1[:, 0] = np.reshape(self.solutions.ic, self.nxyz)

        if isinstance(self.equilibrium_phases, EquilibriumPhases):
            ic1[:, 1] = np.reshape(self.equilibrium_phases.ic, self.nxyz)
        if isinstance(self.exchange_phases, ExchangePhases):
            ic1[:, 2] = np.reshape(self.exchange_phases.ic, self.nxyz)  # Exchange
        if isinstance(self.surfaces_phases, Surfaces):
            ic1[:, 3] = np.reshape(self.surfaces_phases.ic, self.nxyz)  # Surface
        ic1[:, 4] = -1  # Gas phase
        ic1[:, 5] = -1  # Solid solutions
        if isinstance(self.kinetic_phases, KineticPhases):
            ic1[:, 6] = np.reshape(self.kinetic_phases.ic, self.nxyz)  # Kinetics

        ic1_flatten = ic1.flatten('F')

        # set initial conditions as attribute but in a new sub class
        self.ic1 = ic1
        self.ic1_flatten = ic1_flatten

        # initialize ic1 phreeqc to module with phrreeqcrm
        status = self.phreeqc_rm.InitialPhreeqc2Module(ic1_flatten)
        utils.check_phreeqcrm_status(self.phreeqc_rm, status, "InitialPhreeqc2Module")

        # get initial concentrations from running phreeqc
        status = self.phreeqc_rm.RunCells()
        utils.check_phreeqcrm_status(self.phreeqc_rm, status, "RunCells")
        c_dbl_vect = self.phreeqc_rm.GetConcentrations()
        self.init_conc_array_phreeqc = c_dbl_vect

        conc = [c_dbl_vect[i:i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)]

        self.sconc = {}

        for i, c in enumerate(components):
            # where thelement is a component name (c)
            get_conc = np.reshape(conc[i], self.grid_shape)
            get_conc = utils.concentration_l_to_m3(get_conc)
            if c.lower() == 'charge':
                get_conc += self.charge_offset
            self.sconc[c] = get_conc

        self.set_reaction_temp()
        self._write_phreeqc_files()
        print('Phreeqc initialized')
        return

    def set_config(self, **kwargs) -> MF6RTMConfig:
        """Create and store a config object.

        Parameters
        ----------
        **kwargs : dict
            Configuration parameters for MF6RTMConfig.

        Returns
        -------
        MF6RTMConfig
            The created configuration object.
        """
        self.config = MF6RTMConfig(**kwargs)
        return self.config

    def get_config(self):
        """Retrieve config object

        Returns
        -------
        dict
            Configuration parameters as a dictionary.
        """
        return self.config.to_dict()

    def save_config(self):
        """Save config toml file

        Returns
        -------
        Path
            Path to the saved configuration file.
        """
        assert self.wd is not None, "Model directory not specified"
        config_path = self.wd / "mf6rtm.toml"
        print(self.config)
        self.config.save_to_file(filepath=config_path)
        return config_path

    def _update_gwt_stress_packages(self) -> None:
        """Wire equilibrated boundary chemistry into GWF stress package SPDs.

        For each ChemStress on the model, extends the corresponding flopy GWF
        stress package SPD records with component concentrations as auxiliary
        variables. Must be called after initialize() and set_chem_stress().
        """
        if self.components is None:
            raise RuntimeError(
                "Call model.initialize() before write_simulation()."
            )

        gwf_name = next(
            n for n in self._gwt_sim.model_names
            if self._gwt_sim.get_model(n).model_type == 'gwf6'
        )
        gwf = self._gwt_sim.get_model(gwf_name)

        chem_stresses = [
            v for v in vars(self).values()
            if isinstance(v, ChemStress) and getattr(v, 'data', None) is not None
            and v.type == 'aux'
        ]

        for cs in chem_stresses:
            pkg = gwf.get_package(cs.packnme.lower())
            if pkg is None:
                raise ValueError(
                    f"Package '{cs.packnme}' not found in GWF model '{gwf_name}'. "
                    "Ensure ChemStress name matches the flopy package pname exactly."
                )

            # Array recharge stores aux as full grid arrays (one per auxiliary
            # variable), not as per-cell records, so it needs its own path.
            if pkg.package_type == "rcha":
                self._set_rcha_aux(pkg, cs)
                continue

            # Count any existing auxiliary columns (e.g. a conservative tracer
            # aux var). These trail the base SPD fields and are stripped before
            # the component concentrations are appended.
            aux_data = pkg.auxiliary.get_data() if pkg.auxiliary is not None else None
            if aux_data is not None:
                n_existing_aux = sum(len(list(row)) - 1 for row in aux_data)
            else:
                n_existing_aux = 0
            # Detect BOUNDNAMES: if present the last field of each SPD record
            # is the bound name string and must be preserved around the aux strip.
            has_boundnames = (
                hasattr(pkg, 'boundnames')
                and pkg.boundnames is not None
                and pkg.boundnames.get_data() not in (None, False)
            )
            spd = pkg.stress_period_data.get_data()
            ncomps = len(self.components)
            # cs.data is either flat {cell_i: concs} or per-period nested {kper: {cell_i: concs}}
            # Detect once, then look up concentrations per period.
            per_period = bool(cs.data) and isinstance(
                next(iter(cs.data.values())), dict
            )
            updated_spd = {}
            for sp, records in spd.items():
                updated = []
                if records is None:
                    updated_spd[sp] = None
                    # for recharge some spds are none
                    continue
                # concentrations for this stress period keyed by cell index;
                # periods absent from a per-period dict get zero chemistry.
                cell_concs = cs.data.get(sp, {}) if per_period else cs.data
                for i, rec in enumerate(records):
                    base = tuple(rec)
                    if has_boundnames:
                        boundname = base[-1]
                        base = base[:-1]
                    if n_existing_aux > 0:
                        base = base[:-n_existing_aux]
                    concs = cell_concs.get(i, [0.0] * ncomps)
                    row = base + tuple(concs)
                    if has_boundnames:
                        row = row + (boundname,)
                    updated.append(row)
                updated_spd[sp] = updated
            # auxiliary must be set before the data so flopy rebuilds the SPD
            # dtype with the component columns before validating the records.
            pkg.auxiliary = self.components
            pkg.stress_period_data.set_data(updated_spd)

    def _set_rcha_aux(self, pkg, cs) -> None:
        """Write equilibrated recharge chemistry as per-component array aux on RCHA.

        Array recharge stores aux as full grid arrays (one per auxiliary variable),
        not as per-cell records, so each component concentration is broadcast across
        the recharge grid. Uniform recharge chemistry only: exactly one source solution.

        Parameters
        ----------
        pkg : flopy.mf6.ModflowGwfrcha
            The array-recharge package to receive component aux arrays.
        cs : ChemStress
            The equilibrated chem stress; ``cs.data`` must hold a single solution.
        """
        ncomps = len(self.components)
        if len(cs.data) != 1:
            raise ValueError(
                f"Array recharge ChemStress '{cs.packnme}' supports uniform chemistry "
                f"only (one source solution); got {len(cs.data)}. Use set_spd([solution])."
            )
        concs = next(iter(cs.data.values()))  # [c0, c1, ...] for the single solution
        aux_spd = {}
        for sp, rch_arr in pkg.recharge.get_data().items():
            if rch_arr is None:
                aux_spd[sp] = None  # reuse previous period
                continue
            aux_spd[sp] = np.stack(
                [np.full_like(rch_arr, concs[j], dtype=float) for j in range(ncomps)]
            )
        # auxiliary names must be set before the arrays so flopy sizes naux correctly.
        pkg.auxiliary = self.components
        pkg.aux.set_data(aux_spd)

    def _build_reactive_gwt_models(self) -> None:
        """Clone the conservative tracer GWT into N reactive GWT models.

        For each PHREEQC component, creates a flopy GWT model named after
        the component, populated from the reference GWT (ADV, DSP, OC) with
        component-specific IC (from self.sconc) and SSM (from ChemStress).
        The conservative tracer GWT is removed via sim.remove_model().
        """
        # Guard against a tracer/component name collision.
        if self.components is not None and self._gwt_name in self.components:
            raise ValueError(
                f"Conservative tracer GWT model name '{self._gwt_name}' collides with a "
                f"PHREEQC component of the same name. from_mf6() clones the tracer into one "
                f"GWT model per component, so the tracer name must not be a component name. "
                f"Rename the tracer GWT (gwt_name=...) to a non-component name such as 'tracer'."
            )
        gwt_ref = self._gwt_sim.get_model(self._gwt_name)
        gwf_name = next(
            n for n in self._gwt_sim.model_names
            if self._gwt_sim.get_model(n).model_type == 'gwf6'
        )

        all_chem_stresses = [
            v for v in vars(self).values()
            if isinstance(v, ChemStress) and getattr(v, 'data', None) is not None
        ]
        aux_stresses = [cs for cs in all_chem_stresses if cs.type == 'aux']
        nonaux_stresses = [cs for cs in all_chem_stresses if cs.type in ('cnc', 'src')]

        gwf = self._gwt_sim.get_model(gwf_name)

        # Warn once about GWT packages on the tracer model that won't be replicated.
        # Advanced packages (LKT/MVT/UZT/SFT/MWT/IST) carry component-specific
        # concentrations and need per-component re-equilibration — not a blind copy.
        # TODO: add support for LKT, MVT, UZT, SFT, MWT, IST in a future release.
        handled_pkg_types = {
            'dis', 'disv', 'disu', 'adv', 'dsp', 'ic', 'mst', 'ssm', 'oc', 'cnc', 'src'
        }
        unhandled = {p.package_type for p in gwt_ref.packagelist} - handled_pkg_types
        for ptype in unhandled:
            warnings.warn(
                f"GWT package '{ptype}' on the tracer model is not replicated to the "
                "reactive GWT models. Advanced-transport packages "
                "(LKT/MVT/UZT/SFT/MWT/IST) are not yet supported by from_mf6().",
                stacklevel=2,
            )

        # Find the conservative tracer's IMS to use as a settings template for
        # the per-component reactive IMS packages. The solutiongroup recarray
        # maps each IMS file to its models, e.g. ('ims6', 'gwt.ims', 'gwt').
        slngroup = self._gwt_sim.name_file.solutiongroup.get_data(0)
        ims_fname = None
        for row in slngroup:
            fields = [f for f in list(row) if f is not None]
            if self._gwt_name in fields[2:]:
                ims_fname = fields[1]
                break
        if ims_fname is None:
            raise ValueError(
                f"Could not find the IMS solution package for GWT model "
                f"'{self._gwt_name}'."
            )
        ims_settings = self._clone_ims_settings(
            self._gwt_sim.get_solution_package(ims_fname)
        )

        for component in self.components:
            gwt = flopy.mf6.ModflowGwt(
                self._gwt_sim, modelname=component,
            )

            # IMS — one per component, cloned from the tracer's solver settings
            ims = flopy.mf6.ModflowIms(
                self._gwt_sim, filename=f"{component}.ims", **ims_settings
            )
            self._gwt_sim.register_ims_package(ims, [component])

            # DIS/DISV — copy from GWF (shared grid). Detect the grid type with
            distype = gwf.get_grid_type().name
            if distype == 'DIS':
                d = gwf.dis
                flopy.mf6.ModflowGwtdis(
                    gwt,
                    nlay=d.nlay.get_data(),
                    nrow=d.nrow.get_data(),
                    ncol=d.ncol.get_data(),
                    delr=d.delr.get_data(),
                    delc=d.delc.get_data(),
                    top=d.top.get_data(),
                    botm=d.botm.get_data(),
                    idomain=d.idomain.get_data() if d.idomain is not None else None,
                    filename=f"{component}.dis",
                )
            elif distype == 'DISV':
                d = gwf.disv
                flopy.mf6.ModflowGwtdisv(
                    gwt,
                    nlay=d.nlay.get_data(),
                    ncpl=d.ncpl.get_data(),
                    nvert=d.nvert.get_data(),
                    vertices=d.vertices.get_data(),
                    cell2d=d.cell2d.get_data(),
                    top=d.top.get_data(),
                    botm=d.botm.get_data(),
                    idomain=d.idomain.get_data() if d.idomain is not None else None,
                    filename=f"{component}.disv",
                )

            # ADV — copy verbatim
            if gwt_ref.get_package('adv') is not None:
                flopy.mf6.ModflowGwtadv(
                    gwt, scheme=gwt_ref.adv.scheme.get_data()
                )

            # DSP — dispersivity from reference, diffc per component (default 0)
            if gwt_ref.get_package('dsp') is not None:
                diffc = self._diffusion_coeff.get(component, None)
                if diffc is None:
                    warnings.warn(
                        f"diffc not set for component '{component}', defaulting to 0. "
                        "Set via model.set_diffusion_coeff().",
                        stacklevel=2,
                    )
                    diffc = 0.0
                dsp_kwargs = dict(diffc=diffc)
                # xt3d_off/xt3d_rhs are OPTIONS to be carried otherwise get turned on
                for attr in ('alh', 'ath1', 'ath2', 'atv', 'xt3d_off', 'xt3d_rhs'):
                    pkg_attr = getattr(gwt_ref.dsp, attr, None)
                    if pkg_attr is not None:
                        try:
                            dsp_kwargs[attr] = pkg_attr.get_data()
                        except Exception:
                            pass
                flopy.mf6.ModflowGwtdsp(gwt, filename=f"{component}.dsp", **dsp_kwargs)

            # IC — equilibrated initial concentrations for this component
            flopy.mf6.ModflowGwtic(gwt, strt=self.sconc[component], filename=f"{component}.ic")

            # MST — copy porosity and decay from reference
            if gwt_ref.get_package('mst') is not None:
                mst_ref = gwt_ref.mst
                mst_kwargs = {}
                for attr in ('porosity', 'decay', 'decay_sorbed', 'bulk_density',
                             'distcoef', 'sp2', 'first_order_decay', 'zero_order_decay',
                             'sorption'):
                    pkg_attr = getattr(mst_ref, attr, None)
                    if pkg_attr is not None:
                        try:
                            mst_kwargs[attr] = pkg_attr.get_data()
                        except Exception:
                            pass
                # sorption is often per-component but the template has one MST; see set_mst_override()
                mst_kwargs.update(self._mst_overrides.get(component, {}))
                flopy.mf6.ModflowGwtmst(gwt, filename=f"{component}.mst", **mst_kwargs)

            # SSM — rebuilt from aux-type ChemStress objects only
            if aux_stresses:
                sources = [[cs.packnme, 'AUX', component] for cs in aux_stresses]
                flopy.mf6.ModflowGwtssm(gwt, sources=sources, filename=f"{component}.ssm")
            else: #if no aux we still need an empty ssm
                flopy.mf6.ModflowGwtssm(gwt, filename=f"{component}.ssm")

            # CNC / SRC — per-component constant-concentration or mass-loading packages
            for cs in nonaux_stresses:
                if cs.cells is None:
                    raise ValueError(
                        f"ChemStress '{cs.packnme}' has type='{cs.type}' but cells "
                        "is not set. Call cs.set_cells() with a list of cellids."
                    )
                j = self.components.index(component)
                first_val = next(iter(cs.data.values()))
                if isinstance(first_val, dict):
                    # per-stress-period dict path: {sp: {cell_i: [concs]}}
                    spd = {
                        sp: [(cs.cells[i], concs[j]) for i, concs in sorted(pd.items())]
                        for sp, pd in cs.data.items()
                    }
                else:
                    # list path: {cell_i: [concs]} — same for all stress periods
                    spd = [(cs.cells[i], cs.data[i][j]) for i in range(len(cs.cells))]
                pkg_cls = (flopy.mf6.ModflowGwtcnc if cs.type == 'cnc'
                           else flopy.mf6.ModflowGwtsrc)
                pkg_cls(
                    gwt,
                    stress_period_data=spd,
                    pname=cs.packnme,
                    filename=f"{component}.{cs.packnme}.{cs.type}",
                )

            # OC — copy verbatim if present
            if gwt_ref.get_package('oc') is not None:
                oc_ref = gwt_ref.oc
                flopy.mf6.ModflowGwtoc(
                    gwt,
                    budget_filerecord=f'{component}.cbc',
                    concentration_filerecord=f'{component}.ucn',
                    saverecord=oc_ref.saverecord.get_data(),
                    printrecord=(
                        oc_ref.printrecord.get_data()
                        if oc_ref.printrecord is not None else None
                    ),
                )

            # GWF-GWT flow model interface exchange
            flopy.mf6.ModflowGwfgwt(
                self._gwt_sim,
                exgtype='GWF6-GWT6',
                exgmnamea=gwf_name,
                exgmnameb=component,
                filename=f'{component}.gwfgwt',
            )

        # Drop the tracer's orphaned GWF-GWT exchange. remove_model() cleans the
        # mfsim.nam exchanges block but leaves the exchange package object, which
        # flopy tracks in both _exchange_files and _other_files and would
        # otherwise still write to disk as a stale file.
        for container in (self._gwt_sim._exchange_files,
                          self._gwt_sim._other_files):
            for fname, pkg in list(container.items()):
                if self._gwt_name in (getattr(pkg, 'exgmnamea', None),
                                      getattr(pkg, 'exgmnameb', None)):
                    del container[fname]

        self._gwt_sim.remove_model(self._gwt_name)

        # Remove the tracer's now-orphaned IMS: drop the package file and the
        # empty solutiongroup row that remove_model() leaves behind.
        if ims_fname in self._gwt_sim._solution_files:
            del self._gwt_sim._solution_files[ims_fname]
        sg = self._gwt_sim.name_file.solutiongroup.get_data(0)
        kept = [
            tuple(f for f in row if f is not None)
            for row in sg
            if tuple(row)[1] != ims_fname
        ]
        self._gwt_sim.name_file.solutiongroup.set_data(kept, 0)

    def _clone_ims_settings(self, ims_template) -> dict:
        """Extract solver settings from an IMS package into a kwargs dict.

        Used to replicate the conservative tracer's IMS configuration onto each
        per-component reactive IMS. rcloserecord comes back as a recarray, so it
        is special-cased to the scalar inner_rclose value.
        """
        settings = {}
        names = [
            'print_option', 'complexity', 'outer_dvclose', 'outer_maximum',
            'under_relaxation', 'under_relaxation_theta', 'under_relaxation_kappa',
            'under_relaxation_gamma', 'under_relaxation_momentum',
            'backtracking_number', 'inner_maximum', 'inner_dvclose',
            'linear_acceleration', 'relaxation_factor', 'scaling_method',
            'reordering_method', 'preconditioner_levels',
            'preconditioner_drop_tolerance', 'number_orthogonalizations',
        ]
        for name in names:
            v = getattr(ims_template, name, None)
            if v is None:
                continue
            try:
                d = v.get_data()
            except Exception:
                continue
            if d is not None:
                settings[name] = d
        rc = getattr(ims_template, 'rcloserecord', None)
        if rc is not None:
            try:
                d = rc.get_data()
                if d is not None and len(d) > 0:
                    settings['rcloserecord'] = float(d[0]['inner_rclose'])
            except Exception:
                pass
        return settings

    def write_simulation(self):
        """Write phreqcrm simulation and configuration files.

        When the instance was created via from_mf6(), also clones the
        conservative tracer GWT into reactive GWT models, wires stress package
        auxiliary variables, and writes the full MF6 simulation to self.wd.

        Returns
        -------
        None
        """
        if self._gwt_sim is not None:
            self._update_gwt_stress_packages()
            self._build_reactive_gwt_models()
            self._gwt_sim.set_sim_path(str(self.wd))
            self._gwt_sim.set_all_data_external()
            self._gwt_sim.write_simulation()
        self._write_phreeqc_files()
        print(f"Simulation saved in {self.wd}")

    def _write_phreeqc_files(self):
        """Write the PhreeqcRM init file, internal/external parameters and config.

        This is the chemistry-side output, independent of any MF6 flopy build.
        Called both by initialize() (which only needs the PHREEQC files) and by
        write_simulation() (which also builds the MF6 simulation).
        """
        self._write_phreeqc_init_file()
        if self.config.reactive['externalio']:
            self.write_internal_parameters()
            self.write_external_files_layered()
        self.save_config()
        return

    def initialize_chem_stress(
        self,
        attr: str,
        nthreads: int = 1,
    ) -> dict:
        """Initialize a PhreeqcRM object with boundary condition chemical
        concentrations for the specified Modflow Stress Period and Package.

        Parameters
        ----------
        attr : str
            The Modflow 6 Package name.
        nthreads : int, optional
            Number of threads to use for PhreeqcRM (default is 1).

        Returns
        -------
        dict
            Dictionary with component names as keys and concentration arrays in moles/m3 as values.

        Notes
        -----
        This function initializes a PhreeqcRM object, loads a database, runs a Phreeqc input file,
        and transfers solutions and reactants to the reaction-module workers. It then equilibrates
        the cells, gets the concentrations, and converts them to moles/m3.

        See Also
        --------
        phreeqcrm.PhreeqcRM : PhreeqcRM class documentation.
        """
        print('Initializing ChemStress')
        chem_stress = [attr for attr in dir(self) if isinstance(getattr(self, attr), ChemStress)]
        assert len(chem_stress) > 0, 'No ChemStress attribute found in self'

        sol_spd = getattr(self, attr).sol_spd

        if isinstance(sol_spd, dict):
            # Per-stress-period chemistry: {sp: [sol_per_cell]}
            # Collect unique solution numbers across all periods, run PHREEQC once
            all_solutions = []
            for sp_sols in sol_spd.values():
                all_solutions.extend(sp_sols)
            unique_sols = list(dict.fromkeys(all_solutions))  # ordered unique
            nxyz_spd = len(unique_sols)

            phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz_spd, nthreads)
            status = phreeqc_rm.SetComponentH2O(self.componenth2o)
            phreeqc_rm.UseSolutionDensityVolume(False)
            status = phreeqc_rm.SetUnitsSolution(2)
            poro = np.full((nxyz_spd), 1.)
            status = phreeqc_rm.SetPorosity(poro)
            status = phreeqc_rm.SetPrintChemistryMask(np.full((nxyz_spd), 1))
            status = phreeqc_rm.SetPrintChemistryOn(False, True, False)
            status = phreeqc_rm.LoadDatabase(self.database)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "LoadDatabase")
            status = phreeqc_rm.RunFile(True, True, True, self.phinp)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "RunFile")
            input = "DELETE; -all"
            status = phreeqc_rm.RunString(True, False, True, input)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "RunString")
            ncomps = phreeqc_rm.FindComponents()
            utils.check_phreeqcrm_status(phreeqc_rm, ncomps, "FindComponents")
            components = list(phreeqc_rm.GetComponents())

            ic1 = [-1] * nxyz_spd * 7
            for e, sol in enumerate(unique_sols):
                ic1[e] = sol
            status = phreeqc_rm.InitialPhreeqc2Module(ic1)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "InitialPhreeqc2Module")
            status = phreeqc_rm.SetTime(0.0)
            status = phreeqc_rm.SetTimeStep(0.0)

            c_dbl_vect = utils.concentration_l_to_m3(phreeqc_rm.GetConcentrations())
            c_dbl_vect = [c_dbl_vect[i:i + nxyz_spd] for i in range(0, len(c_dbl_vect), nxyz_spd)]
            for i, c in enumerate(components):
                if c.lower() == 'charge':
                    c_dbl_vect[i] += self.charge_offset

            # Map solution number → concentrations
            sol_to_concs = {sol: [arr[j] for arr in c_dbl_vect]
                            for j, sol in enumerate(unique_sols)}

            # Build nested {sp: {cell_i: concs}} data structure
            sconc = {
                sp: {cell_i: sol_to_concs[sol]
                     for cell_i, sol in enumerate(sols)}
                for sp, sols in sol_spd.items()
            }

            status = phreeqc_rm.CloseFiles()
            status = phreeqc_rm.MpiWorkerBreak()

        else:
            # Original list path — same chemistry for all stress periods
            nxyz_spd = len(sol_spd)

            phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz_spd, nthreads)
            status = phreeqc_rm.SetComponentH2O(self.componenth2o)
            phreeqc_rm.UseSolutionDensityVolume(False)
            status = phreeqc_rm.SetUnitsSolution(2)
            poro = np.full((nxyz_spd), 1.)
            status = phreeqc_rm.SetPorosity(poro)
            status = phreeqc_rm.SetPrintChemistryMask(np.full((nxyz_spd), 1))
            status = phreeqc_rm.SetPrintChemistryOn(False, True, False)
            status = phreeqc_rm.LoadDatabase(self.database)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "LoadDatabase")
            status = phreeqc_rm.RunFile(True, True, True, self.phinp)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "RunFile")
            input = "DELETE; -all"
            status = phreeqc_rm.RunString(True, False, True, input)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "RunString")
            ncomps = phreeqc_rm.FindComponents()
            utils.check_phreeqcrm_status(phreeqc_rm, ncomps, "FindComponents")
            components = list(phreeqc_rm.GetComponents())

            ic1 = [-1] * nxyz_spd * 7
            for e, i in enumerate(sol_spd):
                ic1[e] = i
            status = phreeqc_rm.InitialPhreeqc2Module(ic1)
            utils.check_phreeqcrm_status(phreeqc_rm, status, "InitialPhreeqc2Module")
            status = phreeqc_rm.SetTime(0.0)
            status = phreeqc_rm.SetTimeStep(0.0)

            c_dbl_vect = utils.concentration_l_to_m3(phreeqc_rm.GetConcentrations())
            c_dbl_vect = [c_dbl_vect[i:i + nxyz_spd] for i in range(0, len(c_dbl_vect), nxyz_spd)]
            for i, c in enumerate(components):
                if c.lower() == 'charge':
                    c_dbl_vect[i] += self.charge_offset

            sconc = {i: [arr[i] for arr in c_dbl_vect] for i in range(nxyz_spd)}

            status = phreeqc_rm.CloseFiles()
            status = phreeqc_rm.MpiWorkerBreak()

        setattr(getattr(self, attr), 'data', sconc)
        setattr(getattr(self, attr), 'auxiliary', components)
        print(f'ChemStress {attr} initialized')
        return sconc

    def _initialize_phreeqc_from_file(self, yamlfile):
        """Initialize phreeqc from a yaml file

        Parameters
        ----------
        yamlfile : str
            Path to the yaml file.
        Returns
        -------
        None
        """
        yamlfile = self.phreeqcyaml_file
        phreeqcrm_from_yaml = phreeqcrm.InitializeYAML(yamlfile)
        if phreeqcrm_from_yaml is None:
            raise RuntimeError(
                f"PhreeqcRM InitializeYAML failed for '{yamlfile}'."
            )
        if self.phreeqc_rm is None:
            self.phreeqc_rm = phreeqcrm_from_yaml
        return

    def write_internal_parameters(self, internals = {
                                                    "equilibrium_phases": ["si"],
                                                    "kinetic_phases": ["parms", "formula", "steps"],
                                                    "exchange_phases": ["dummy"]
                                                    }
                                    ):
        """Add non-external attributes to the config object.

        Parameters
        ----------
        internals : dict
            Dictionary with phase names as keys and list of attributes to add as values.
            Default is {
                "equilibrium_phases": ["si"],
                "kinetic_phases": ["parms", "formula", "steps"],
                "exchange_phases": ["dummy"]
            }.

        Returns
        -------
        None
        """
        valid_internals = {k: v for k, v in internals.items() if getattr(self, k, None) is not None}
        # self.add_read_external_files_flag_to_config(flag=True)
        for key in valid_internals.keys():
            # if key is ot defined continue
            attr_list = internals[key]
            phase_obj = getattr(self, key)
            data = phase_obj.data[0]

            for item in attr_list:
                if item == "dummy":
                    attr_name = f"{key}_names"
                    self.config.add_new_configuration(**{attr_name: list(phase_obj.names)})
                    # Skip dummy items, they are not real parameters
                    continue
                attr_name = f"{key}_names"
                if attr_name not in self.config.__dict__:
                    # Add the names of the phases to the config object
                    self.config.add_new_configuration(**{attr_name: list(phase_obj.names)})
                for name in phase_obj.names:
                    # print(f"Adding internal parameters for {key:<20}: {item} in {name}")
                    if item in data[name]:
                        # Create nested attribute name: equilibrium_phases_si_Goethite
                        attr_name = f"{key}_{item}_{name}"
                        if not hasattr(self.config, attr_name):
                            self.config.add_new_configuration(**{attr_name: data[name][item]})

    def save_mup3d(self, filename='mup3d.pkl'):
        """
        Save the Mup3d object to a pickle file.

        This method saves all non-private, non-callable attributes of the Mup3d object
        to a pickle file for later restoration using load_mup3d().

        Parameters
        ----------
        filename : str, optional
            Name of the pickle file. Default is 'mup3d.pkl'.
        """
        import pickle
        fname = os.path.join(self.wd, filename)

        # Attributes that cannot be pickled (SWIG objects, etc.)
        unpickleable_attrs = {
            'phreeqc_rm',           # SWIG PhreeqcRM object
            'phreeqcrm_yaml',       # SWIG YAMLPhreeqcRM object
        }

        # Create a dictionary of the object's attributes
        # Exclude private attributes, callable methods, and unpickleable objects
        attributes = {}
        skipped_attrs = []

        for attr in dir(self):
            if attr.startswith('_') or callable(getattr(self, attr)):
                continue
            if attr in unpickleable_attrs:
                skipped_attrs.append(attr)
                continue

            try:
                value = getattr(self, attr)
                # Test if the attribute can be pickled
                pickle.dumps(value)
                attributes[attr] = value
            except (TypeError, AttributeError) as e:
                # Skip attributes that can't be pickled
                skipped_attrs.append(f"{attr} ({str(e)})")
                continue

        # Save the object to a file
        with open(fname, "wb") as file:
            pickle.dump(attributes, file)
        print(f"Saved Mup3d model to {fname}")
        if skipped_attrs:
            print(f"Skipped unpickleable attributes: {skipped_attrs}")

    @classmethod
    def load_mup3d(cls, filename='mup3d.pkl', wd='.'):
        """
        Load a Mup3d object from a pickle file (class method).
        This creates a new Mup3d instance from a saved pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the pickle file. Default is 'mup3d.pkl'.
        working_dir : str, optional
            Directory containing the pickle file. Default is current directory.

        Returns
        -------
        Mup3d
            A new Mup3d instance loaded from the pickle file.

        Examples
        --------
        >>> # Create a new model from pickle file
        >>> model = Mup3d.load_mup3d('my_model.pkl', '/path/to/model/dir')
        """
        import pickle
        fname = os.path.join(wd, filename)

        # Load the object from a file
        with open(fname, "rb") as file:
            attributes = pickle.load(file)

        # Create a new Mup3d instance with core parameters
        instance = cls(
            name=attributes.get('name', None),
            solutions=attributes.get('solutions', None),
            nlay=attributes.get('nlay', None),
            nrow=attributes.get('nrow', None),
            ncol=attributes.get('ncol', None),
            ncpl=attributes.get('ncpl', None),
            nxyz=attributes.get('nxyz', None)
        )

        # Set the working directory if it exists
        if attributes.get('wd') is not None:
            instance.set_wd(attributes.get('wd'))

        # Set the database if it exists
        if attributes.get('database') is not None and os.path.exists(attributes.get('database')):
            instance.set_database(attributes.get('database'))

        # Set the postfix if it exists
        if attributes.get('postfix') is not None and os.path.exists(attributes.get('postfix')):
            instance.set_postfix(attributes.get('postfix'))

        # Set the componenth2o flag
        instance.set_componenth2o(attributes.get('componenth2o', False))

        # Set the initial temperature
        instance.set_initial_temp(attributes.get('init_temp', 25.0))

        # Set the charge offset
        instance.set_charge_offset(attributes.get('charge_offset', 0.0))

        # Set the config object if it exists
        if 'config' in attributes and attributes['config'] is not None:
            if hasattr(attributes['config'], 'to_dict'):
                # If config is a MF6RTMConfig object
                config_dict = attributes['config'].to_dict()
                instance.set_config(**config_dict)
            elif isinstance(attributes['config'], dict):
                # If config is already a dictionary
                instance.set_config(**attributes['config'])

        # Set the phases using the appropriate setter methods
        phase_types = ['equilibrium_phases', 'kinetic_phases', 'exchange_phases', 'surfaces_phases']
        for phase_type in phase_types:
            if phase_type in attributes and attributes[phase_type] is not None:
                if phase_type == 'exchange_phases':
                    instance.set_exchange_phases(attributes[phase_type])
                elif phase_type == 'equilibrium_phases':
                    instance.set_equilibrium_phases(attributes[phase_type])
                else:
                    # For kinetic_phases and surfaces_phases, use set_phases
                    instance.set_phases(attributes[phase_type])

        # Set remaining attributes that don't have specific setter methods
        skip_attrs = {
            'name', 'solutions', 'nlay', 'nrow', 'ncol', 'ncpl', 'nxyz',
            'wd', 'database', 'postfix', 'componenth2o', 'init_temp',
            'charge_offset', 'config', 'equilibrium_phases', 'kinetic_phases',
            'exchange_phases', 'surfaces_phases'
        }

        for attr, value in attributes.items():
            if attr not in skip_attrs and not callable(value):
                setattr(instance, attr, value)

        print(f"Loaded Mup3d model from {fname}")
        return instance

    def set_diffusion_coeff(self, diffusion_coeff: dict) -> None:
        """Set molecular diffusion coefficients per PHREEQC component.

        Parameters
        ----------
        diffusion_coeff : dict
            Mapping of component name to diffusion coefficient in m²/s,
            e.g. ``{'Ca': 0.792e-9, 'Cl': 2.032e-9}``. Components not listed
            default to 0 (no diffusion) with a warning at write time.
        """
        self._diffusion_coeff = diffusion_coeff

    def set_mst_override(self, overrides: dict) -> None:
        """Override MST keyword arguments for specific components (from_mf6 builds).

        Parameters
        ----------
        overrides : dict
            Mapping of component name to ``ModflowGwtmst`` keyword arguments, e.g.
            ``{'Tmp': {'sorption': 'Linear', 'bulk_density': 1850.0,
            'distcoef': 2.1141e-4}}``. Keys given here replace values copied from the
            template; components not listed are unaffected.
        """
        self._mst_overrides = overrides

    @classmethod
    def from_mf6(cls, sim, solutions, name=None, gwf_name=None, gwt_name=None):
        """Create a Mup3d instance from an existing flopy MFSimulation.

        Grid dimensions are extracted automatically from the GWF model.
        The conservative tracer GWT model is stored as a template and will
        be cloned into N reactive GWT models (one per PHREEQC component)
        when ``write_simulation()`` is called.

        Parameters
        ----------
        sim : flopy.mf6.MFSimulation
            Loaded flopy simulation containing at least one GWF and one GWT model.
        solutions : Solutions
            Geochemical solutions for the reactive transport model.
        name : str, optional
            Model name. Defaults to None.
        gwf_name : str, optional
            Name of the GWF model. Defaults to the first model in sim.
        gwt_name : str, optional
            Name of the conservative tracer GWT model to use as template.
            Required when the simulation contains more than one GWT model.

        Returns
        -------
        Mup3d
        """
        # Resolve GWF model
        gwf = sim.get_model(gwf_name) if gwf_name else sim.get_model(sim.model_names[0])

        # Detect grid type and extract dims
        distype = gwf.get_grid_type().name
        if distype == 'DIS':
            nlay = int(gwf.dis.nlay.data)
            nrow = int(gwf.dis.nrow.data)
            ncol = int(gwf.dis.ncol.data)
            instance = cls(name, solutions, nlay=nlay, nrow=nrow, ncol=ncol)
        elif distype == 'DISV':
            nlay = int(gwf.disv.nlay.data)
            ncpl = int(gwf.disv.ncpl.data)
            instance = cls(name, solutions, nlay=nlay, ncpl=ncpl)
        else:
            raise ValueError(
                f"Grid type '{distype}' is not supported by mf6rtm. "
                "Only DIS and DISV are supported."
            )

        # Resolve GWT template
        gwt_models = [n for n in sim.model_names if sim.get_model(n).model_type == 'gwt6']
        if len(gwt_models) == 0:
            raise ValueError("No GWT model found in the simulation.")
        elif len(gwt_models) == 1:
            gwt_name = gwt_models[0]
        elif gwt_name is None:
            raise ValueError(
                f"Multiple GWT models found {gwt_models}. "
                "Specify gwt_name to select the conservative tracer template."
            )

        # Auto-set working directory as sibling 'reactive/' of sim workspace
        sim_ws_abs = os.path.abspath(sim.sim_path)
        default_wd = os.path.join(os.path.dirname(sim_ws_abs), 'reactive')
        warnings.warn(
            f"Working directory not set. Defaulting to '{default_wd}'. "
            "Call set_wd() before write_simulation() to override.",
            stacklevel=2,
        )
        instance.set_wd(default_wd)

        instance._gwt_sim = sim
        instance._gwt_name = gwt_name
        return instance

    def write_external_files_layered(self,
                                     internals = [
                                                    "exchange_phases",
                                                    "equilibrium_phases",
                                                    "kinetic_phases"
                                                    ],
                                     property_to_write = ['m0']) -> None:
        """
        Write layered external text files for selected geochemical phases and properties.

        For each specified geochemical phase (e.g., exchange, equilibrium, kinetic), this method extracts
        the given properties (e.g., 'm0') for all defined species and writes a separate file per layer
        in the simulation domain. The files are saved in the model's working directory and follow the
        naming convention:

            {phase}.{species}.{property}.layer{n}.txt

        Parameters
        ----------
        internals : list of str, optional
            List of model attributes containing geochemical phase data.
            Default is ["exchange_phases", "equilibrium_phases", "kinetic_phases"].

        property_to_write : list of str, optional
            List of property names to extract and write per species and layer.
            Default is ['m0'].
        """
        valid_internals = [k for k in internals if getattr(self, k, None) is not None]
        for attr in valid_internals:
            phase_obj = getattr(self, attr)
            if phase_obj is None:
                print(f"Warning: model has no attribute '{attr}'. Skipping.")
                continue
            data = phase_obj.data
            ic = phase_obj.ic
            for name in phase_obj.names:
                print(f"Writing external files for {attr:<20}: {name}")
                for prop in property_to_write:
                    arr = utils.map_species_property_to_grid(
                        data, ic, name, prop
                    )
                    for ly in range(arr.shape[0]):
                        filepath = os.path.join(self.wd, f"{attr}.{name}.{prop}.layer{ly+1}.txt")
                        with open(filepath, "w") as fh:
                            fh.write("\n".join(f"{val:.10e}" for val in arr[ly].flatten()))

    def _write_phreeqc_init_file(self, filename='mf6rtm.yaml') -> None:
        """Write the phreeqc init yaml file.

        Parameters
        ----------
        filename : str, optional
            Name of the yaml file. Default is 'mf6rtm.yaml'.

        Returns
        -------
        None
        """
        fdir = os.path.join(self.wd, filename)
        phreeqcrm_yaml = phreeqcrm.YAMLPhreeqcRM()
        phreeqcrm_yaml.YAMLSetGridCellCount(self.nxyz)
        phreeqcrm_yaml.YAMLThreadCount(1)
        status = phreeqcrm_yaml.YAMLSetComponentH2O(self.componenth2o)
        status = phreeqcrm_yaml.YAMLUseSolutionDensityVolume(False)

        # Open files for phreeqcrm logging
        status = phreeqcrm_yaml.YAMLSetFilePrefix(os.path.join('_phreeqc'))
        status = phreeqcrm_yaml.YAMLOpenFiles()

        # set some properties
        phreeqcrm_yaml.YAMLSetErrorHandlerMode(1)
        phreeqcrm_yaml.YAMLSetRebalanceFraction(0.5) # Needed for multithreading
        phreeqcrm_yaml.YAMLSetRebalanceByCell(True) # Needed for multithreading
        phreeqcrm_yaml.YAMLSetPartitionUZSolids(False) # TODO: implement when UZF is turned on

        # Set concentration units
        phreeqcrm_yaml.YAMLSetUnitsSolution(2)       # 1, mg/L; 2, mol/L; 3, kg/kgs
        phreeqcrm_yaml.YAMLSetUnitsPPassemblage(1)   # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        phreeqcrm_yaml.YAMLSetUnitsExchange(1)       # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        phreeqcrm_yaml.YAMLSetUnitsSurface(1)        # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        phreeqcrm_yaml.YAMLSetUnitsGasPhase(1)       # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        phreeqcrm_yaml.YAMLSetUnitsSSassemblage(1)   # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        phreeqcrm_yaml.YAMLSetUnitsKinetics(1)       # 0, mol/L cell; 1, mol/L water; 2 mol/L rock

        # mf6 handles poro . set to 1
        poro = [1.0]*self.nxyz
        status = phreeqcrm_yaml.YAMLSetPorosity(list(poro))

        print_chemistry_mask = [1]*self.nxyz
        assert all(isinstance(i, int) for i in print_chemistry_mask), 'print_chemistry_mask length must be equal to the number of grid cells'
        status = phreeqcrm_yaml.YAMLSetPrintChemistryMask(print_chemistry_mask)
        status = phreeqcrm_yaml.YAMLSetPrintChemistryOn(False, True, False)  # workers, initial_phreeqc, utility

        rv = [1] * self.nxyz
        phreeqcrm_yaml.YAMLSetRepresentativeVolume(rv)

        # Load database
        status = phreeqcrm_yaml.YAMLLoadDatabase(os.path.basename(self.database))
        status = phreeqcrm_yaml.YAMLRunFile(True, True, True, os.path.basename(self.phinp))

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = phreeqcrm_yaml.YAMLRunString(True, False, True, input)
        if self.postfix is None:
            phreeqcrm_yaml.YAMLAddOutputVars("SolutionProperties", "true")
            phreeqcrm_yaml.YAMLAddOutputVars("SolutionTotalMolalities", "true")
            if self.equilibrium_phases is not None:
                phreeqcrm_yaml.YAMLAddOutputVars("EquilibriumPhases", "true")
            if self.kinetic_phases is not None:
                phreeqcrm_yaml.YAMLAddOutputVars("KineticReactants", "true")
        else:
            phreeqcrm_yaml.YAMLAddOutputVars("AddOutputVars", "true")

        status = phreeqcrm_yaml.YAMLFindComponents()
        # convert ic1 to a list
        ic1_flatten = self.ic1_flatten

        status = phreeqcrm_yaml.YAMLInitialPhreeqc2Module(ic1_flatten)
        status = phreeqcrm_yaml.YAMLRunCells()
        # Initial equilibration of cells
        time = 0.0
        status = phreeqcrm_yaml.YAMLSetTime(time)
        # status = phreeqcrm_yaml.YAMLSetTimeStep(time_step)
        status = phreeqcrm_yaml.WriteYAMLDoc(fdir)

        # create new attribute for phreeqc yaml file
        self.phreeqcyaml_file = fdir
        self.phreeqcrm_yaml = phreeqcrm_yaml
        return

    def run(self, reactive=None, nthread=1, libname=None, output_format=None, **kwargs) -> bool:
        """Wrapper function to run the MF6RTM model

        Parameters
        ----------
        reactive : bool, optional
            Whether to run the model in reactive mode. If None, uses the value from the config.
        nthread : int, optional
            Number of threads to use for the simulation. Default is 1.
        libname : str, optional
            Name of the MF6 shared library. If None, uses the default.
        output_format : str, optional
            Output format: "csv" (default) or "hdf5". Overrides mf6rtm.toml [output].
            Output file is named "sout.csv" or "sout.h5" accordingly.
        **kwargs
            Additional keyword arguments set as attributes on the Mf6RTM instance
            before solving. Valid keys are any settable attribute of Mf6RTM, e.g.:
            ``threshold``, ``min_concentration``, ``charge_offset``,
            ``fixed_components``.

        Returns
        -------
        bool
            True if the model ran successfully, False otherwise.
        """
        with working_dir(self.wd):
            print("Running mf6rtm", flush=True)
            success = solve(self.wd, reactive=reactive, nthread=nthread, libname=libname,
                            output_format=output_format, **kwargs)
            return success

@contextmanager
def working_dir(path):
    """Context manager for changing the current working directory.

    Parameters
    ----------
    path : str
        Path to the directory to change to.
    Yields
    ------
    None
    """
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)
