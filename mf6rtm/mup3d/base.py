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
    sol_spd : list, optional
        List of solution stress period data. Default is None.
    packtype : str, optional
        Type of the package. Default is None.
    """
    def __init__(self, packnme) -> None:
        self.packnme = packnme
        self.sol_spd = None
        self.packtype = None

    def set_spd(self, sol_spd):
        self.sol_spd = sol_spd

    def set_packtype(self, packtype):
        self.packtype = packtype


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
        name: Union[int, None] = None,
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
            print(self.solutions.ic.shape, self.nxyz, self.grid_shape)
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
        """
        Sets the working directory for the MF6RTM model.

        Parameters
        ----------
        wd (str): The path to the working directory.

        Returns
        -------
        None

        Raises
        ------
        AssertionError: If the working directory path is not a string.
        """
        # get absolute path of the working directory
        wd = Path(os.path.abspath(wd))
        # joint current directory with wd, check if exist, create if not
        if not wd.exists():
            wd.mkdir(parents=True, exist_ok=True)
        self.wd = wd

    def set_database(self, database):
        """
        Sets the database for the MF6RTM model.

        Parameters:
        ----------
        database (str): The path to the database file.

        Returns:
        -------
        None
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
        """
        Sets the postfix file for the MF6RTM model.

        Parameters:
        ----------
        postfix (str): The path to the postfix file.

        Returns:
        -------
        None
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

    def generate_phreeqc_script(self, add_charge_flag=False):
        """
        Generates the phinp file for the MF6RTM model.

        Parameters
        ----------
        add_charge_flag : bool, optional
            Whether to add a charge flag to species in the solution block.
            Default is False.
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
                script += utils.handle_block(phases, utils.generate_surface_block, i, options=self.surfaces_phases.options)

        # add end of line before postfix
        script += utils.endmainblock

        # Append the postfix file to the script
        if self.postfix is not None and os.path.isfile(self.postfix):
            with open(self.postfix, 'r') as source:  # Open the source file in read mode
                script += '\n'
                script += source.read()

        if add_charge_flag:
            script = utils.add_charge_flag_to_species_in_solution(script)

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
        add_charge_flag : bool, optional
            Whether to add charge flag to species. Default is False.

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
        status = self.phreeqc_rm.RunFile(True, True, True, self.phinp)

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = self.phreeqc_rm.RunString(True, False, True, input)

        # Get component information - these two functions need to be invoked to find comps
        ncomps = self.phreeqc_rm.FindComponents()
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

        # get initial concentrations from running phreeqc
        status = self.phreeqc_rm.RunCells()
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
        self.write_simulation()
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

    def write_simulation(self):
        """Write phreqcrm simulation and configuration files

        Returns
        -------
        None
        """
        self._write_phreeqc_init_file()
        if self.config.reactive['externalio']:
            self.write_internal_parameters()
            self.write_external_files_layered()
        self.save_config()
        print(f"Simulation saved in {self.wd}")
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
        # check if self has a an attribute that is a class ChemStress but without knowing the attribute name
        chem_stress = [attr for attr in dir(self) if isinstance(getattr(self, attr), ChemStress)]

        assert len(chem_stress) > 0, 'No ChemStress attribute found in self'

        # Get total number of grid cells affected by the stress period
        nxyz_spd = len(getattr(self, attr).sol_spd)

        phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz_spd, nthreads)
        status = phreeqc_rm.SetComponentH2O(self.componenth2o)
        phreeqc_rm.UseSolutionDensityVolume(False)

        # Set concentration units
        status = phreeqc_rm.SetUnitsSolution(2)

        poro = np.full((nxyz_spd), 1.)
        status = phreeqc_rm.SetPorosity(poro)
        print_chemistry_mask = np.full((nxyz_spd), 1)
        status = phreeqc_rm.SetPrintChemistryMask(print_chemistry_mask)
        nchem = phreeqc_rm.GetChemistryCellCount()

        # Set printing of chemistry file
        status = phreeqc_rm.SetPrintChemistryOn(False, True, False)  # workers, initial_phreeqc, utility

        # Load database
        status = phreeqc_rm.LoadDatabase(self.database)
        status = phreeqc_rm.RunFile(True, True, True, self.phinp)

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = phreeqc_rm.RunString(True, False, True, input)

        # Get component information - these two functions need to be invoked to find comps
        ncomps = phreeqc_rm.FindComponents()
        components = list(phreeqc_rm.GetComponents())

        # Transfer solutions and reactants from the InitialPhreeqc instance to
        # the reaction-module workers. See https://usgs-coupled.github.io/phreeqcrm/namespacephreeqcrm.html#ac3d7e7db76abda97a3d11b3ff1903322
        ic1 = [-1] * nxyz_spd * 7
        for e, i in enumerate(getattr(self, attr).sol_spd):
            # TODO: modify to conform to the index, element convention
            #       (i and e are reversed in line above)
            ic1[e] = i  # Solution 1
            # TODO: implment other ic1 blocks
            # ic1[nxyz_spd + i]     = -1  # Equilibrium phases none
            # ic1[2 * nxyz_spd + i] =  -1  # Exchange 1
            # ic1[3 * nxyz_spd + i] = -1  # Surface none
            # ic1[4 * nxyz_spd + i] = -1  # Gas phase none
            # ic1[5 * nxyz_spd + i] = -1  # Solid solutions none
            # ic1[6 * nxyz_spd + i] = -1  # Kinetics none
        status = phreeqc_rm.InitialPhreeqc2Module(ic1)

        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = phreeqc_rm.SetTime(time)
        status = phreeqc_rm.SetTimeStep(time_step)

        # status = phreeqc_rm.RunCells()
        c_dbl_vect = phreeqc_rm.GetConcentrations()
        c_dbl_vect = utils.concentration_l_to_m3(c_dbl_vect)

        c_dbl_vect = [c_dbl_vect[i:i + nxyz_spd] for i in range(0, len(c_dbl_vect), nxyz_spd)]

        # find charge in c_dbl_vect and add charge_offset
        for i, c in enumerate(components):
            if c.lower() == 'charge':
                c_dbl_vect[i] += self.charge_offset

        sconc = {}
        for i in range(nxyz_spd):
            sconc[i] = [array[i] for array in c_dbl_vect]

        status = phreeqc_rm.CloseFiles()
        status = phreeqc_rm.MpiWorkerBreak()

        # set as attribute
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

    def run(self, reactive = None, nthread=1):
        """Wrapper function to run the MF6RTM model

        Parameters
        ----------
        reactive : bool, optional
            Whether to run the model in reactive mode. If None, uses the value from the config
            nthread : int, optional
                Number of threads to use for the simulation. Default is 1.

        Returns
        -------
        bool
            True if the model ran successfully, False otherwise.
        """
        with working_dir(self.wd):
            print("Running mf6rtm", flush=True)
            success = solve(self.wd, reactive=reactive, nthread=nthread)
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
