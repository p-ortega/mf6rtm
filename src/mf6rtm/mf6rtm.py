from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image
import pandas as pd
import numpy as np
import flopy
import phreeqcrm
import modflowapi
# from modflowapi.extensions import ApiSimulation
from datetime import datetime
# from time import sleep
from . import utils
from phreeqcrm import yamlphreeqcrm
import yaml

# add representer to yaml to write np.array as list
def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(np.ndarray, ndarray_representer)


# global variables
DT_FMT = "%Y-%m-%d %H:%M:%S"

time_units_dic = {
    'second': 1,
    'minute': 60,
    'hour': 3600,
    'day': 86400,
    'week': 604800,
    'month': 2628000,
    'year': 31536000
}


class Block:
    def __init__(self, data, ic=None) -> None:
        self.data = data
        self.names = [key for key in data.keys()]
        self.ic = ic  # None means no initial condition (-1)
        self.eq_solutions = []
        self.options = []

    def set_ic(self, ic):
        '''Set the initial condition for the block.
        '''
        assert isinstance(ic, (int, float, np.ndarray)), 'ic must be an int, float or ndarray'
        self.ic = ic

    def set_equilibrate_solutions(self, eq_solutions):
        '''Set the equilibrium solutions for the exchange phases.
        Array where index is the exchange phase number and value is the solution number to equilibrate with.
        '''
        self.eq_solutions = eq_solutions

    def set_options(self, options):
        self.options = options


class GasPhase(Block):
    def __init__(self, data) -> None:
        super().__init__(data)


class Solutions(Block):
    def __init__(self, data) -> None:
        super().__init__(data)


class EquilibriumPhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)


class ExchangePhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)


class KineticPhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters


class Surfaces(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)


class ChemStress():
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
    'ExchangePhases': ExchangePhases,
    'EquilibriumPhases': EquilibriumPhases,
    'Surfaces': Surfaces,
}


class Mup3d(object):
    def __init__(self, name, solutions, nlay, nrow, ncol):
        self.name = name
        self.wd = None
        self.charge_offset = 0.0
        self.database = os.path.join('database', 'pht3d_datab.dat')
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
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.ncpl = self.nlay*self.nrow*self.ncol

        if self.solutions.ic is None:
            self.solutions.ic = [1]*self.ncpl
        if isinstance(self.solutions.ic, (int, float)):
            self.solutions.ic = np.reshape([self.solutions.ic]*self.ncpl, (self.nlay, self.nrow, self.ncol))
        assert self.solutions.ic.shape == (self.nlay, self.nrow, self.ncol), f'Initial conditions array must be an array of the shape ({nlay}, {nrow}, {ncol}) not {self.solutions.ic.shape}'

    def set_fixed_components(self, fixed_components):
        self.fixed_components = fixed_components

    def set_initial_temp(self, temp):
        assert isinstance(temp, (int, float, list)), 'temp must be an int or float'
        # TODO: for non-homogeneous fields allow 3D and 2D arrays
        self.init_temp = temp

    def set_phases(self, phase):
        # Dynamically get the class of the phase object
        phase_class = phase.__class__

        # Check if the phase object's class is in the dictionary of phase types
        if phase_class not in phase_types.values():
            raise AssertionError(f'{phase_class.__name__} is not a recognized phase type')

        # Proceed with the common logic
        if isinstance(phase.ic, (int, float)):
            phase.ic = np.reshape([phase.ic]*self.ncpl, (self.nlay, self.nrow, self.ncol))
        phase.data = {i: phase.data[key] for i, key in enumerate(phase.data.keys())}
        assert phase.ic.shape == (self.nlay, self.nrow, self.ncol), f'Initial conditions array must be an array of the shape ({self.nlay}, {self.nrow}, {self.ncol}) not {phase.ic.shape}'

        # Dynamically set the phase attribute based on the class name
        setattr(self, f"{phase_class.__name__.lower().split('phases')[0]}_phases", phase)

    def set_exchange_phases(self, exchanger):
        assert isinstance(exchanger, ExchangePhases), 'exchanger must be an instance of the Exchange class'
        # exchanger.data = {i: exchanger.data[key] for i, key in enumerate(exchanger.data.keys())}
        if isinstance(exchanger.ic, (int, float)):
            exchanger.ic = np.reshape([exchanger.ic]*self.ncpl, (self.nlay, self.nrow, self.ncol))
        assert exchanger.ic.shape == (self.nlay, self.nrow, self.ncol), f'Initial conditions array must be an array of the shape ({self.nlay}, {self.nrow}, {self.ncol}) not {exchanger.ic.shape}'
        self.exchange_phases = exchanger

    def set_equilibrium_phases(self, eq_phases):
        '''Sets the equilibrium phases for the MF6RTM model.
        '''
        assert isinstance(eq_phases, EquilibriumPhases), 'eq_phases must be an instance of the EquilibriumPhases class'
        # change all keys from eq_phases so they start from 0
        eq_phases.data = {i: eq_phases.data[key] for i, key in enumerate(eq_phases.data.keys())}
        self.equilibrium_phases = eq_phases
        if isinstance(self.equilibrium_phases.ic, (int, float)):
            self.equilibrium_phases.ic = np.reshape([self.equilibrium_phases.ic]*self.ncpl, (self.nlay, self.nrow, self.ncol))
        assert self.equilibrium_phases.ic.shape == (self.nlay, self.nrow, self.ncol), f'Initial conditions array must be an array of the shape ({self.nlay}, {self.nrow}, {self.ncol}) not {self.equilibrium_phases.ic.shape}'

    def set_charge_offset(self, charge_offset):
        """
        Sets the charge offset for the MF6RTM model to handle negative charge values
        """
        self.charge_offset = charge_offset

    def set_chem_stress(self, chem_stress):
        assert isinstance(chem_stress, ChemStress), 'chem_stress must be an instance of the ChemStress class'
        attribute_name = chem_stress.packnme
        setattr(self, attribute_name, chem_stress)

        self.initiliaze_chem_stress(attribute_name)

    # def initialize_chem_stress(self):
    #     #initialize phreeqc for chemstress
    #     pass

    def set_wd(self, wd):
        """
        Sets the working directory for the MF6RTM model.
        """
        # joint current directory with wd, check if exist, create if not
        self.wd = Path(wd)
        if not self.wd.exists():
            self.wd.mkdir(parents=True, exist_ok=True)

    def set_database(self, database):
        """
        Sets the database for the MF6RTM model.

        Parameters:
        database (str): The path to the database file.

        Returns:
        None
        """
        assert os.path.exists(database), f'{database} not found'
        self.database = database

    def set_postfix(self, postfix):
        """
        Sets the postfix file for the MF6RTM model.

        Parameters:
        postfix (str): The path to the postfix file.

        Returns:
        None
        """
        assert os.path.exists(postfix), f'{postfix} not found'
        self.postfix = postfix

    def set_reaction_temp(self):
        if isinstance(self.init_temp, (int, float)):
            rx_temp = [self.init_temp]*self.ncpl
            print('Using temperatue of {} for all cells'.format(rx_temp[0]))
        elif isinstance(self.init_temp, (list, np.ndarray)):
            rx_temp = [self.init_temp[0]]*self.ncpl
            print('Using temperatue of {} from SOLUTION 1 for all cells'.format(rx_temp[0]))
        self.reaction_temp = rx_temp
        return rx_temp

    def generate_phreeqc_script(self):
        """
        Generates the phinp file for the MF6RTM model.
        """

        # where to save the phinp file
        filename = os.path.join(self.wd, 'phinp.dat')
        self.phinp = filename
        # assert that database in self.database exists
        assert os.path.exists(self.database), f'{self.database} not found'

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
                script += utils.handle_block(phases, utils.generate_phases_block, i)

        # check if self.exchange_phases is not None
        if self.exchange_phases is not None:
            # Get the current   phases
            phases = self.exchange_phases.data
            # check if all exchange phases are in the database
            names = utils.get_compound_names(self.database, 'EXCHANGE')
            assert all([key in names for key in phases.keys()]), 'Following are not in database: '+', '.join(f'{key}' for key in phases.keys() if key not in names)

            num_exch = len(next(iter(self.exchange_phases.data.values())))
            for i in range(num_exch):
                # Get the current concentrations and phases
                concentrations = {species: values[i] for species, values in self.exchange_phases.data.items()}
                script += utils.handle_block(concentrations, utils.generate_exchange_block, i, equilibrate_solutions=self.exchange_phases.eq_solutions)

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

        with open(filename, 'w') as file:
            file.write(script)
        return script

    def initialize(self, nthreads=1):
        '''Initialize a solution with phreeqcrm and returns a dictionary with components as keys and
            concentration array in moles/m3 as items
        '''
        # get model dis info
        # dis = sim.get_model(sim.model_names[0]).dis

        # create phinp
        # check if phinp.dat is in wd
        phinp = self.generate_phreeqc_script()

        # initialize phreeqccrm object
        self.phreeqc_rm = phreeqcrm.PhreeqcRM(self.ncpl, nthreads)
        status = self.phreeqc_rm.SetComponentH2O(False)
        self.phreeqc_rm.UseSolutionDensityVolume(False)

        # Open files for phreeqcrm logging
        status = self.phreeqc_rm.SetFilePrefix(os.path.join(self.wd, '_phreeqc'))
        self.phreeqc_rm.OpenFiles()

        # Set concentration units
        status = self.phreeqc_rm.SetUnitsSolution(2)
        # status = self.phreeqc_rm.SetUnitsExchange(1)
        # status = self.phreeqc_rm.SetUnitsSurface(1)
        # status = self.phreeqc_rm.SetUnitsKinetics(1)

        # mf6 handles poro . set to 1
        poro = np.full((self.ncpl), 1.)
        status = self.phreeqc_rm.SetPorosity(poro)

        print_chemistry_mask = np.full((self.ncpl), 1)
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

        ic1 = np.ones((self.ncpl, 7), dtype=int)*-1

        # this gets a column slice
        ic1[:, 0] = np.reshape(self.solutions.ic, self.ncpl)

        if isinstance(self.equilibrium_phases, EquilibriumPhases):
            ic1[:, 1] = np.reshape(self.equilibrium_phases.ic, self.ncpl)
        if isinstance(self.exchange_phases, ExchangePhases):
            ic1[:, 2] = np.reshape(self.exchange_phases.ic, self.ncpl)  # Exchange
        if isinstance(self.surfaces_phases, Surfaces):
            ic1[:, 3] = np.reshape(self.surfaces_phases.ic, self.ncpl)  # Surface
        ic1[:, 4] = -1  # Gas phase
        ic1[:, 5] = -1  # Solid solutions

        if isinstance(self.kinetic_phases, KineticPhases):
            ic1[:, 6] = np.reshape(self.kinetic_phases.ic, self.ncpl)  # Kinetics

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

        conc = [c_dbl_vect[i:i + self.ncpl] for i in range(0, len(c_dbl_vect), self.ncpl)]

        self.sconc = {}

        for e, c in enumerate(components):
            get_conc = np.reshape(conc[e], (self.nlay, self.nrow, self.ncol))
            get_conc = concentration_l_to_m3(get_conc)
            if c.lower() == 'charge':
                get_conc += self.charge_offset
            self.sconc[c] = get_conc

        self.set_reaction_temp()
        self._write_phreeqc_init_file()
        print('Phreeqc initialized')
        return

    def initiliaze_chem_stress(self, attr, nthreads=1):
        '''Initialize a solution with phreeqcrm and returns a dictionary with components as keys and
            concentration array in moles/m3 as items
        '''
        print('Initializing ChemStress')
        # check if self has a an attribute that is a class ChemStress but without knowing the attribute name
        chem_stress = [attr for attr in dir(self) if isinstance(getattr(self, attr), ChemStress)]

        assert len(chem_stress) > 0, 'No ChemStress attribute found in self'

        nxyz = len(getattr(self, attr).sol_spd)
        phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)
        status = phreeqc_rm.SetComponentH2O(False)
        phreeqc_rm.UseSolutionDensityVolume(False)

        # Set concentration units
        status = phreeqc_rm.SetUnitsSolution(2)

        poro = np.full((nxyz), 1.)
        status = phreeqc_rm.SetPorosity(poro)
        print_chemistry_mask = np.full((nxyz), 1)
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

        ic1 = [-1] * nxyz * 7
        for e, i in enumerate(getattr(self, attr).sol_spd):
            ic1[e] = i  # Solution 1
            # ic1[nxyz + i]     = -1  # Equilibrium phases none
            # ic1[2 * nxyz + i] =  -1  # Exchange 1
            # ic1[3 * nxyz + i] = -1  # Surface none
            # ic1[4 * nxyz + i] = -1  # Gas phase none
            # ic1[5 * nxyz + i] = -1  # Solid solutions none
            # ic1[6 * nxyz + i] = -1  # Kinetics none
        status = phreeqc_rm.InitialPhreeqc2Module(ic1)

        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = phreeqc_rm.SetTime(time)
        status = phreeqc_rm.SetTimeStep(time_step)

        # status = phreeqc_rm.RunCells()
        c_dbl_vect = phreeqc_rm.GetConcentrations()
        c_dbl_vect = concentration_l_to_m3(c_dbl_vect)

        c_dbl_vect = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]

        # find charge in c_dbl_vect and add charge_offset
        for i, c in enumerate(components):
            if c.lower() == 'charge':
                c_dbl_vect[i] += self.charge_offset

        sconc = {}
        for i in range(nxyz):
            sconc[i] = [array[i] for array in c_dbl_vect]

        status = phreeqc_rm.CloseFiles()
        status = phreeqc_rm.MpiWorkerBreak()

        # set as attribute
        setattr(getattr(self, attr), 'data', sconc)
        setattr(getattr(self, attr), 'auxiliary', components)
        print(f'ChemStress {attr} initialized')
        return sconc

    def _initialize_phreeqc_from_file(self, yamlfile):
        yamlfile = self.phreeqcyaml_file
        phreeqcrm_from_yaml = phreeqcrm.InitializeYAML(yamlfile)
        if self.phreeqc_rm is None:
            self.phreeqc_rm = phreeqcrm_from_yaml
        return

    def _write_phreeqc_init_file(self, filename='mf6rtm.yaml'):
        fdir = os.path.join(self.wd, filename)
        phreeqcrm_yaml = yamlphreeqcrm.YAMLPhreeqcRM()
        phreeqcrm_yaml.YAMLSetGridCellCount(self.ncpl)
        phreeqcrm_yaml.YAMLThreadCount(1)
        status = phreeqcrm_yaml.YAMLSetComponentH2O(False)
        status = phreeqcrm_yaml.YAMLUseSolutionDensityVolume(False)

        # Open files for phreeqcrm logging
        # status = phreeqcrm_yaml.YAMLSetGridCellCount(self.ncpl)
        status = phreeqcrm_yaml.YAMLSetFilePrefix(os.path.join(self.wd, '_phreeqc'))
        status = phreeqcrm_yaml.YAMLOpenFiles()

        # set some properties
        # phreeqcrm_yaml.YAMLSetErrorHandlerMode(1)
        # phreeqcrm_yaml.YAMLSetRebalanceFraction(0.5)
        # phreeqcrm_yaml.YAMLSetRebalanceByCell(True)
        # phreeqcrm_yaml.YAMLSetPartitionUZSolids(False)
        # Set concentration units
        status = phreeqcrm_yaml.YAMLSetUnitsSolution(2)

        # mf6 handles poro . set to 1
        # poro = np.full((self.ncpl), 1.)
        poro = [1.0]*self.ncpl
        status = phreeqcrm_yaml.YAMLSetPorosity(list(poro))

        print_chemistry_mask = [1]*self.ncpl
        assert all(isinstance(i, int) for i in print_chemistry_mask), 'print_chemistry_mask length must be equal to the number of grid cells'
        status = phreeqcrm_yaml.YAMLSetPrintChemistryMask(print_chemistry_mask)
        status = phreeqcrm_yaml.YAMLSetPrintChemistryOn(False, True, False)  # workers, initial_phreeqc, utility
        # rv = [1] * self.ncpl
        # phreeqcrm_yaml.YAMLSetRepresentativeVolume(rv)
        # Load database
        status = phreeqcrm_yaml.YAMLLoadDatabase(self.database)
        status = phreeqcrm_yaml.YAMLRunFile(True, True, True, self.phinp)

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = phreeqcrm_yaml.YAMLRunString(True, False, True, input)
        status = phreeqcrm_yaml.YAMLFindComponents()
        # convert ic1 to a list
        ic1_flatten = self.ic1_flatten

        status = phreeqcrm_yaml.YAMLInitialPhreeqc2Module(ic1_flatten)
        status = phreeqcrm_yaml.YAMLRunCells()
        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0 # TODO: set time step from mf6 and convert to seconds
        status = phreeqcrm_yaml.YAMLSetTime(time)
        status = phreeqcrm_yaml.YAMLSetTimeStep(time_step)
        status = phreeqcrm_yaml.WriteYAMLDoc(fdir)

        # create new attribute for phreeqc yaml file
        self.phreeqcyaml_file = fdir
        self.phreeqcrm_yaml = phreeqcrm_yaml
        return

    def run_mup3d(self, sim, dll=None, reaction=True):
        """
        Modflow6 API and PhreeqcRM integration function to solve model.

        Parameters
        ----------
        sim(mf6.simulation): the MODFLOW-6 simulation object (from flopy)
        dll (Path like): the MODFLOW-6 shared/DLL library filename
        reaction (bool): to indicate wether to invoke phreeqcrm or not. Default is True
        """

        print('\n-----------------------------  WELCOME TO  MUP3D -----------------------------\n')

        success = False  # initialize success flag

        phreeqc_rm = self.phreeqc_rm
        sim_ws = sim.simulation_data.mfpath.get_sim_path()
        dll = os.path.join(self.wd, 'libmf6')

        nlay, nrow, ncol = get_mf6_dis(sim)
        nxyz = self.ncpl

        modelnmes = ['Flow'] + [nme.capitalize() for nme in sim.model_names if nme != 'gwf']  # revise later

        print(f"Transporting the following components: {', '.join([nme for nme in modelnmes])}")

        components = [nme.capitalize() for nme in sim.model_names[1:]]

        mf6 = modflowapi.ModflowApi(dll, working_directory=sim_ws)

        mf6.initialize()

        nsln = mf6.get_subcomponent_count()
        sim_start = datetime.now()

        print("Starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))

        # get the current sim time
        ctime = mf6.get_current_time()
        ctimes = [0.0]

        # get the ending sim time
        etime = mf6.get_end_time()

        num_fails = 0

        # some phreeqc output settings
        phreeqc_rm.SetScreenOn(True)
        sout_columns = phreeqc_rm.GetSelectedOutputHeadings()
        soutdf = pd.DataFrame(columns=sout_columns)

        # let's do it!
        while ctime < etime:
            sol_start = datetime.now()
            # length of the current solve time
            dt = mf6.get_time_step()
            # prep the current time step
            mf6.prepare_time_step(dt)

            status = phreeqc_rm.SetTimeStep(dt*86400)  # FIXME: generalize	with time_units_dic and mf6.sim.time_units
            kiter = 0

            # prep to solve
            for sln in range(1, nsln+1):
                mf6.prepare_solve(sln)

            # the one-based stress period number
            stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
            time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]

            # if ctime == 0.0:
            #     run_reactions(  self, mf6, ctime, time_step, sout_columns, dt)

            # mf6 transport loop block
            for sln in range(1, nsln+1):
                if self.fixed_components is not None and modelnmes[sln-1] in self.fixed_components:
                    print(f'not transporting {modelnmes[sln-1]}')
                    continue

                # max number of solution iterations
                max_iter = mf6.get_value(mf6.get_var_address("MXITER", f"SLN_{sln}"))  # FIXME: not sure to define this inside the loop
                mf6.prepare_solve(sln)

                print(f'\nSolving solution {sln} - Solving {modelnmes[sln-1]}')
                while kiter < max_iter:
                    convg = mf6.solve(sln)
                    if convg:
                        td = (datetime.now() - sol_start).total_seconds() / 60.0
                        print("Transport stress period: {0} --- time step: {1} --- converged with {2} iters --- took {3:10.5G} mins".format(stress_period, time_step, kiter, td))
                        break
                    kiter += 1

                if not convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    print("Transport stress period: {0} --- time step: {1} --- did not converge with {2} iters --- took {3:10.5G} mins".format(stress_period, time_step, kiter, td))
                    num_fails += 1
                try:
                    mf6.finalize_solve(sln)
                    print(f'\nSolution {sln} finalized')
                except:
                    pass

            mf6_conc_array = []
            for c in components:
                if c.lower() == 'charge':
                    mf6_conc_array.append(concentration_m3_to_l(mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) - self.charge_offset))

                else:
                    mf6_conc_array.append(concentration_m3_to_l(mf6.get_value(mf6.get_var_address("X", f'{c.upper()}'))))

            c_dbl_vect = np.reshape(mf6_conc_array, self.ncpl*self.ncomps)  # flatten array
            # Phreeqc loop block
            if reaction:
                # get arrays from mf6 and flatten for phreeqc
                print(f'\nGetting concentration arrays --- time step: {time_step} --- elapsed time: {ctime}')

                status = phreeqc_rm.SetTemperature(self.reaction_temp)
                # status = phreeqc_rm.SetPressure([2.0] * nxyz)

                # update phreeqc time and time steps
                status = phreeqc_rm.SetTime(ctime*86400)

                # allow phreeqc to print some info in the terminal
                print_selected_output_on = True
                print_chemistry_on = True
                status = phreeqc_rm.SetSelectedOutputOn(print_selected_output_on)
                status = phreeqc_rm.SetPrintChemistryOn(print_chemistry_on, True, True)
                # set concentrations for reactions
                status = phreeqc_rm.SetConcentrations(c_dbl_vect)

                # reactions loop
                message = '\nBeginning reaction calculation               {} days\n'.format(ctime)
                phreeqc_rm.LogMessage(message)
                phreeqc_rm.ScreenMessage(message)
                status = phreeqc_rm.RunCells()
                if status < 0:
                    print('Error in RunCells: {0}'.format(status))
                # selected ouput
                sout = phreeqc_rm.GetSelectedOutput()
                sout = [sout[i:i + nxyz] for i in range(0, len(sout), nxyz)]

                # add time to selected ouput
                sout[0] = np.ones_like(sout[0])*(ctime+dt)

                df = pd.DataFrame(columns=sout_columns)
                for col, arr in zip(df.columns, sout):
                    df[col] = arr
                soutdf = pd.concat([soutdf.astype(df.dtypes), df])  # avoid pandas warning by passing dtypes to empty df

                # Get concentrations from phreeqc
                c_dbl_vect = phreeqc_rm.GetConcentrations()
                conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]  # reshape array

                conc_dic = {}
                for e, c in enumerate(components):
                    # conc_dic[c] = np.reshape(conc[e], (nlay, nrow, ncol))
                    conc_dic[c] = conc[e]
                # Set concentrations in mf6
                    print(f'\nTransferring concentrations to mf6 for component: {c}')
                    if c.lower() == 'charge':
                        mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]) + self.charge_offset)

                    else:
                        mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]))

            mf6.finalize_time_step()
            ctime = mf6.get_current_time()  # update the current time tracking

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0
        if num_fails > 0:
            print("\nFailed to converge {0} times".format(num_fails))
        print("\n")

        # save selected ouput to csv
        soutdf.to_csv(os.path.join(sim_ws, 'sout.csv'), index=False)

        print("\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT), td))

        # Clean up and close api objs
        try:
            status = phreeqc_rm.CloseFiles()
            status = phreeqc_rm.MpiWorkerBreak()
            mf6.finalize()
            success = True
            print(mrbeaker())
            print('\nMR BEAKER IMPORTANT MESSAGE: MODEL RUN FINISHED BUT CHECK THE RESULTS .. THEY ARE PROLY RUBBISH\n')
        except:
            print('\nMR BEAKER IMPORTANT MESSAGE: SOMETHING WENT WRONG. BUMMER\n')
            pass
        return success

def prep_to_run(wd):
    '''Prepares the model to run by checking if the model directory contains the necessary files
    and returns the path to the yaml file (phreeqcrm) and the dll file (mf6 api)'''
    # check if wd exists
    assert os.path.exists(wd), f'{wd} not found'
    # check if file starting with libmf6 exists
    dll = [f for f in os.listdir(wd) if f.startswith('libmf6.')]
    assert len(dll) == 1, 'libmf6 dll not found in model directory'
    assert os.path.exists(os.path.join(wd, 'mf6rtm.yaml')), 'mf6rtm.yaml not found in model directory'

    nam = [f for f in os.listdir(wd) if f.endswith('.nam')]
    assert 'mfsim.nam' in nam, 'mfsim.nam file not found in model directory'
    assert 'gwf.nam' in nam, 'gwf.nam file not found in model directory'
    dll = os.path.join(wd, 'libmf6')
    yamlfile = os.path.join(wd, 'mf6rtm.yaml')

    return yamlfile, dll

def solve(wd, reactive=True):
    # success = check_model_files(wd)
    # if success:
    #     pass
    # else:
    #     raise Exception('Model files not found')
    mf6rtm = initialize_interfaces(wd)
    if not reactive:
        mf6rtm._set_reactive(reactive)
    mf6rtm._solve()
    print('done')

def initialize_interfaces(wd):
    '''Function to initialize the interfaces for modflowapi and phreeqcrm and returns the mf6rtm object
    '''
    yamlfile, dll = prep_to_run(wd)
    print(yamlfile, dll)
    mf6api = Mf6API(wd, dll)
    phreeqcrm = PhreeqcBMI(yamlfile)
    mf6rtm = Mf6RTM(wd, mf6api, phreeqcrm)
    return mf6rtm


def get_mf6_dis(sim):
    '''Function to extract dis from modflow6 sim object
    '''
    dis = sim.get_model(sim.model_names[0]).dis
    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()
    return nlay, nrow, ncol


def calc_nxyz_from_dis(sim):
    '''Function to calculate number of cells from dis object
    '''
    nlay, nrow, ncol = get_mf6_dis(sim)
    return nlay*nrow*ncol

def get_mf6_disv(sim):
    #TODO: implement this function
    ...

def determine_grid_type(sim):
    '''Function to determine the grid type of the model
    '''
    # get the grid type
    mf6 = sim.get_model(sim.model_names[0])
    distype = mf6.get_grid_type().name
    return distype

class PhreeqcBMI(phreeqcrm.BMIPhreeqcRM):

    def __init__(self, yaml="mf6rtm.yaml"):
        phreeqcrm.BMIPhreeqcRM.__init__(self)
        self.initialize(yaml)

    def get_grid_to_map(self):
        '''Function to get grid to map
        '''
        return self.GetGridToMap()

    def _prepare_phreeqcrm_bmi(self):
        '''Prepare phreeqc bmi for reaction calculations
        '''
        self.SetScreenOn(True)
        self.set_scalar("NthSelectedOutput", 0)
        sout_headers = self.GetSelectedOutputHeadings()
        soutdf = pd.DataFrame(columns=sout_headers)

        # set attributes
        self.components = self.get_value_ptr("Components")
        self.ncomps = len(self.components)
        self.soutdf = soutdf
        self.sout_headers = sout_headers

    def _set_ctime(self, ctime):
        # self.ctime = self.SetTime(ctime*86400)
        self.ctime = ctime

    def set_scalar(self, var_name, value):
        itemsize = self.get_var_itemsize(var_name)
        nbytes = self.get_var_nbytes(var_name)
        dim = nbytes // itemsize

        if dim != 1:
            raise ValueError(f"{var_name} is not a scalar")

        vtype = self.get_var_type(var_name)
        dest = np.empty(1, dtype=vtype)
        dest[0] = value
        x = self.set_value(var_name, dest)

    def _solve_phreeqcrm(self, dt, diffmask):
        '''Function to solve phreeqc bmi
        '''

        # status = phreeqc_rm.SetTemperature([self.init_temp[0]] * self.ncpl)
        # status = phreeqc_rm.SetPressure([2.0] * nxyz)
        self.SetTimeStep(dt*86400)

        # update which cells to run depending on conc change between tsteps
        sat = [1]*self.GetGridCellCount()
        self.SetSaturation(sat)
        if diffmask is not None:
            # get idx where diffmask is 0
            inact = get_indices(0, diffmask)
            if len(inact) > 0:
                for i in inact:
                    sat[i] = 0
            print(f"{'Cells sent to reactions':<25} | {self.GetGridCellCount()-len(inact):<0}/{self.GetGridCellCount():<15}")
            self.SetSaturation(sat)

        # allow phreeqc to print some info in the terminal
        print_selected_output_on = True
        print_chemistry_on = True
        status = self.SetSelectedOutputOn(print_selected_output_on)
        status = self.SetPrintChemistryOn(print_chemistry_on, False, True)
        # reactions loop
        sol_start = datetime.now()

        message = f"{'Reaction loop':<25} | {'Stress period:':<15} {self.kper:<5} | {'Time step:':<15} {self.kstp:<10} | {'Running ...':<10}"
        self.LogMessage(message+'\n')	# log message
        print(message)
        # self.ScreenMessage(message)
        # status = self.RunCells()
        # if status < 0:
        #     print('Error in RunCells: {0}'.format(status))
        self.update()
        td = (datetime.now() - sol_start).total_seconds() / 60.0
        message = f"{'Reaction loop':<25} | {'Stress period:':<15} {self.kper:<5} | {'Time step:':<15} {self.kstp:<10} | {'Completed in :':<10} {td//60:.0f} min {td%60:.4f} sec\n\n"
        self.LogMessage(message)
        print(message)
        # self.ScreenMessage(message)


    def _get_kper_kstp_from_mf6api(self, mf6api):
        assert isinstance(mf6api, Mf6API), 'mf6api must be an instance of Mf6API'
        self.kper = mf6api.kper
        self.kstp = mf6api.kstp
        return


class Mf6API(modflowapi.ModflowApi):
    def __init__(self, wd, dll):
        modflowapi.ModflowApi.__init__(self, dll, working_directory=wd)
        self.initialize()
        self.sim = flopy.mf6.MFSimulation.load(sim_ws=wd, verbosity_level=0)

    def _prepare_mf6(self):
        '''Prepare mf6 bmi for transport calculations
        '''
        self.modelnmes = ['Flow'] + [nme.capitalize() for nme in self.sim.model_names if nme != 'gwf']
        self.components = [nme.capitalize() for nme in self.sim.model_names[1:]]
        self.nsln = self.get_subcomponent_count()
        self.sim_start = datetime.now()
        self.ctimes = [0.0]
        self.num_fails = 0

    def _solve_gwt(self):
        # prep to solve
        for sln in range(1, self.nsln+1):
            self.prepare_solve(sln)
        # the one-based stress period number
        stress_period = self.get_value(self.get_var_address("KPER", "TDIS"))[0]
        time_step = self.get_value(self.get_var_address("KSTP", "TDIS"))[0]

        self.kper = stress_period
        self.kstp = time_step
        msg = f"{'Transport loop':<25} | {'Stress period:':<15} {stress_period:<5} | {'Time step:':<15} {time_step:<10} | {'Running ...':<10}"
        print(msg)
        # mf6 transport loop block
        for sln in range(1, self.nsln+1):
            # if self.fixed_components is not None and modelnmes[sln-1] in self.fixed_components:
            #     print(f'not transporting {modelnmes[sln-1]}')
            #     continue

            # set iteration counter
            kiter = 0
            # max number of solution iterations
            max_iter = self.get_value(self.get_var_address("MXITER", f"SLN_{sln}"))  # FIXME: not sure to define this inside the loop
            self.prepare_solve(sln)

            sol_start = datetime.now()
            while kiter < max_iter:
                convg = self.solve(sln)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    break
                kiter += 1
            if not convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                print("\nTransport stress period: {0} --- time step: {1} --- did not converge with {2} iters --- took {3:10.5G} mins".format(stress_period, time_step, kiter, td))
                self.num_fails += 1
            try:
                self.finalize_solve(sln)
            except:
                pass
        td = (datetime.now() - sol_start).total_seconds() / 60.0
        print(f"{'Transport loop':<25} | {'Stress period:':<15} {stress_period:<5} | {'Time step:':<15} {time_step:<10} | {'Completed in :':<10}  {td//60:.0f} min {td%60:.4f} sec")

    def _check_num_fails(self):
        if self.num_fails > 0:
            print("\nTransport failed to converge {0} times \n".format(self.num_fails))
        else:
            print("\nTransport converged successfully without any fails")

class Mf6RTM(object):
    def __init__(self, wd, mf6api, phreeqcbmi):
        assert isinstance(mf6api, Mf6API), 'MF6API must be an instance of Mf6API'
        assert isinstance(phreeqcbmi, PhreeqcBMI), 'PhreeqcBMI must be an instance of PhreeqcBMI'
        self.mf6api = mf6api
        self.phreeqcbmi = phreeqcbmi
        self.charge_offset = 0.0
        self.wd = wd
        self.sout_fname = 'sout.csv'
        self.reactive = True
        self.epsaqu = 0.0

        #set discretization
        self._set_dis()

    def _set_reactive(self, reactive):
        self.reactive = reactive

    def _set_dis(self):
        '''Set the model grid dimensions according to mf6 grid type
        '''
        if determine_grid_type(self.mf6api.sim) == 'DIS':
            self.nlay, self.nrow, self.ncol = get_mf6_dis(self.mf6api.sim)
            self.nxyz = calc_nxyz_from_dis(self.mf6api.sim)
        elif determine_grid_type(self.mf6api.sim) == 'DISV':
            self.nlay = self.mf6api.sim.nlay
            self.ncpl = self.mf6api.sim.ncpl

    def _prepare_to_solve(self):
        '''Prepare the model to solve
        '''
        # check if sout fname exists
        if self._check_sout_exist():
            self._rm_sout_file()

        self.mf6api._prepare_mf6()
        self.phreeqcbmi._prepare_phreeqcrm_bmi()

        # get and write sout headers
        self._write_sout_headers()

    def _set_ctime(self):
        self.ctime = self.mf6api.get_current_time()
        self.phreeqcbmi._set_ctime(self.ctime)
        return self.ctime

    def _set_etime(self):
        self.etime = self.mf6api.get_end_time()
        return self.etime

    def _set_time_step(self):
        self.time_step = self.mf6api.get_time_step()
        return self.time_step

    def _finalize(self):
        self._finalize_mf6api()
        self._finalize_phreeqcrm()
        return

    def _finalize_mf6api(self):
        self.mf6api.finalize()

    def _finalize_phreeqcrm(self):
        self.phreeqcbmi.finalize()

    def _get_cdlbl_vect(self):
        c_dbl_vect = self.phreeqcbmi.GetConcentrations()

        conc = [c_dbl_vect[i:i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)]  # reshape array
        return conc

    def _set_conc_at_current_kstep(self, c_dbl_vect):
        self.current_iteration_conc = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))

    def _set_conc_at_previous_kstep(self, c_dbl_vect):
        self.previous_iteration_conc = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))

    def _transfer_array_to_mf6(self):
        '''Transfer the concentration array to mf6
        '''
        c_dbl_vect = self._get_cdlbl_vect()

        # if self._check_previous_conc_exists() and self._check_inactive_cells_exist(c_dbl_vect):
        #     c_dbl_vect = self._replace_inactive_cells(c_dbl_vect)
        # else:
        #     pass
        conc_dic = {}
        for e, c in enumerate(self.phreeqcbmi.components):
            # conc_dic[c] = np.reshape(c_dbl_vect[e], (self.nlay, self.nrow, self.ncol))
            conc_dic[c] = c_dbl_vect[e]
        # Set concentrations in mf6
            if c.lower() == 'charge':
                self.mf6api.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]) + self.charge_offset)
            else:
                self.mf6api.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]))
        return c_dbl_vect

    def _check_previous_conc_exists(self):
        '''Function to replace inactive cells in the concentration array
        '''
        # check if self.previous_iteration_conc is a property
        if not hasattr(self, 'previous_iteration_conc'):
            return False
        else:
            return True

    def _check_inactive_cells_exist(self, c_dbl_vect):
        '''Function to check if inactive cells exist in the concentration array
        '''
        c_dbl_vect = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))
        inactive_idx = get_inactive_idx(c_dbl_vect)
        if len(inactive_idx) > 0:
            return True
        else:
            return False

    def _replace_inactive_cells(self, c_dbl_vect):
        '''Function to replace inactive cells in the concentration array
        '''
        c_dbl_vect = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))
        # get inactive cells
        inactive_idx = [get_inactive_idx(c_dbl_vect[k]) for k in range(self.phreeqcbmi.ncomps)]
        # replace inactive cells with previous conc
        c_dbl_vect[:, inactive_idx] = self.previous_iteration_conc[:, inactive_idx]
        c_dbl_vect = c_dbl_vect.flatten()
        conc = [c_dbl_vect[i:i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)]
        return conc

    def _transfer_array_to_phreeqcrm(self):
        mf6_conc_array = []
        for c in self.phreeqcbmi.components:
            if c.lower() == 'charge':
                mf6_conc_array.append(concentration_m3_to_l(self.mf6api.get_value(self.mf6api.get_var_address("X", f'{c.upper()}')) - self.charge_offset))

            else:
                mf6_conc_array.append(concentration_m3_to_l(self.mf6api.get_value(self.mf6api.get_var_address("X", f'{c.upper()}'))))

        c_dbl_vect = np.reshape(mf6_conc_array, self.nxyz*self.phreeqcbmi.ncomps)
        self.phreeqcbmi.SetConcentrations(c_dbl_vect)

        #set the kper and kstp
        self.phreeqcbmi._get_kper_kstp_from_mf6api(self.mf6api) # FIXME: calling this func here is not ideal

        return c_dbl_vect

    def _update_selected_output(self):
        self._get_selected_output()
        updf = pd.concat([self.phreeqcbmi.soutdf.astype(self._current_sout.dtypes), self._current_sout])
        self._update_soutdf(updf)

    def _get_selected_output(self):
        # selected ouput
        self.phreeqcbmi.set_scalar("NthSelectedOutput", 0)
        sout = self.phreeqcbmi.GetSelectedOutput()
        sout = [sout[i:i + self.nxyz] for i in range(0, len(sout), self.nxyz)]

        # add time to selected ouput
        sout[0] = np.ones_like(sout[0])*(self.ctime+self.time_step)
        df = pd.DataFrame(columns=self.phreeqcbmi.soutdf.columns)
        for col, arr in zip(df.columns, sout):
            df[col] = arr
        self._current_sout = df

    def _update_soutdf(self, df):
        self.phreeqcbmi.soutdf = df

    def _check_sout_exist(self):
        if os.path.exists(os.path.join(self.wd, self.sout_fname)):
            return True
        else :
            return False

    def _write_sout_headers(self):
        '''Write selected output headers to a file
        '''
        with open(os.path.join(self.wd,self.sout_fname), 'w') as f:
            f.write(','.join(self.phreeqcbmi.sout_headers))
            f.write('\n')

    def _rm_sout_file(self):
        '''Remove the selected output file
        '''
        try:
            os.remove(os.path.join(self.wd, self.sout_fname))
        except:
            pass

    def _append_to_soutdf_file(self):
        '''Append the current selected output to the selected output file
        '''
        assert not self._current_sout.empty, 'current sout is empty'
        self._current_sout.to_csv(os.path.join(self.wd, self.sout_fname), mode='a', index=False, header=False)

    def _export_soutdf(self):
        '''Export the selected output dataframe to a csv file
        '''
        self.phreeqcbmi.soutdf.to_csv(os.path.join(self.wd, self.sout_fname), index=False)

    def _solve(self):
        '''Solve the model
        '''
        success = False  # initialize success flag
        sim_start = datetime.now()
        self._prepare_to_solve()

        # check sout was created
        assert self._check_sout_exist(), f'{self.sout_fname} not found'

        print("Starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))
        print(f"Solving the following components: {', '.join([nme for nme in self.mf6api.modelnmes])}")
        ctime = self._set_ctime()
        etime = self._set_etime()
        while ctime < etime:
            temp_time = datetime.now()
            print(f"Starting solution at {temp_time.strftime(DT_FMT)}")
            # length of the current solve time
            dt = self._set_time_step()
            self.mf6api.prepare_time_step(dt)
            self.mf6api._solve_gwt()

            if self.reactive:
                # reaction block
                c_dbl_vect = self._transfer_array_to_phreeqcrm()
                self._set_conc_at_current_kstep(c_dbl_vect)
                if ctime > 0.0:
                    diffmask = get_conc_change_mask(self.current_iteration_conc,
                                                    self.previous_iteration_conc,
                                                    self.phreeqcbmi.ncomps, self.nxyz,
                                                    treshold=self.epsaqu)
                else:
                    diffmask = None
                # solve reactions
                self.phreeqcbmi._solve_phreeqcrm(dt, diffmask = diffmask)
                # get sout and update df
                self._update_selected_output()
                # append current sout rows to file
                self._append_to_soutdf_file()
                c_dbl_vect = self._transfer_array_to_mf6()
                self._set_conc_at_previous_kstep(c_dbl_vect)

            self.mf6api.finalize_time_step()
            ctime = self._set_ctime()  # update the current time tracking

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0

        self.mf6api._check_num_fails()

        print("\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT), td))

        # Clean up and close api objs
        try:
            self._finalize()
            success = True
            # save selected ouput to csv
            # self._export_soutdf()
            print(mrbeaker())
            print('\nMR BEAKER IMPORTANT MESSAGE: MODEL RUN FINISHED BUT CHECK THE RESULTS .. THEY ARE PROLY RUBBISH\n')
        except:
            print('MR BEAKER IMPORTANT MESSAGE: SOMETHING WENT WRONG. BUMMER\n')
            pass
        return success

def get_indices(element, lst):
    return [i for i, x in enumerate(lst) if x == element]

def get_less_than_zero_idx(arr):
    '''Function to get the index of all occurrences of <0 in an array
    '''
    idx = np.where(arr < 0)
    return idx

def get_inactive_idx(arr, val = 1e30):
    '''Function to get the index of all occurrences of <0 in an array
    '''
    idx = list(np.where(arr >= val)[0])
    return idx

def get_conc_change_mask(ci, ck, ncomp, nxyz, treshold=1e-5):
    '''Function to get the active-inactive cell mask for concentration change to inform phreeqc which cells to update
    '''
    # reshape arrays to 2D (nxyz, ncomp)
    ci = ci.reshape(nxyz, ncomp)
    ck = ck.reshape(nxyz, ncomp)+1e-30

    # get the difference between the two arrays and divide by ci
    diff = np.abs((ci - ck.reshape(-1*nxyz, ncomp))/ci) < treshold
    # print(diff)
    diff = np.where(diff, 0, 1)

    diff = diff.sum(axis=1)

    # where values <0 put -1 else 1
    diff = np.where(diff == 0, 0, 1)
    return diff

def mrbeaker():
    '''ASCII art of Mr. Beaker
    '''
    # get the path of this file
    whereismrbeaker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrbeaker.png')
    mr_beaker_image = Image.open(whereismrbeaker)

    # Resize the image to fit the terminal width
    terminal_width = 80  # Adjust this based on your terminal width
    aspect_ratio = mr_beaker_image.width / mr_beaker_image.height
    terminal_height = int(terminal_width / aspect_ratio*0.5)
    mr_beaker_image = mr_beaker_image.resize((terminal_width, terminal_height))

    # Convert the image to grayscale
    mr_beaker_image = mr_beaker_image.convert("L")

    # Convert the grayscale image to ASCII art
    ascii_chars = "%,.?>#*+=-:."

    mrbeaker = ""
    for y in range(int(mr_beaker_image.height)):
        mrbeaker += "\n"
        for x in range(int(mr_beaker_image.width)):
            pixel_value = mr_beaker_image.getpixel((x, y))
            mrbeaker += ascii_chars[pixel_value // 64]
        mrbeaker += "\n"
    return mrbeaker


def flatten_list(xss):
    return [x for xs in xss for x in xs]


def concentration_l_to_m3(x):
    '''Convert M/L to M/m3
    '''
    c = x*1e3
    return c


def concentration_m3_to_l(x):
    '''Convert M/L to M/m3
    '''
    c = x*1e-3
    return c


def concentration_to_massrate(q, conc):
    '''Calculate mass rate from rate (L3/T) and concentration (M/L3)
    '''
    mrate = q*conc  # M/T
    return mrate


def concentration_volbulk_to_volwater(conc_volbulk, porosity):
    '''Calculate concentrations as volume of pore water from bulk volume and porosity
    '''
    conc_volwater = conc_volbulk*(1/porosity)
    return conc_volwater
