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
    # 'ExchangePhases': ExchangePhases,
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
        phreeqcrm_yaml = phreeqcrm.YAMLPhreeqcRM()

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

        print_chemistry_mask = [1.0]*self.ncpl
        status = phreeqcrm_yaml.YAMLSetPrintChemistryMask(print_chemistry_mask)
        status = phreeqcrm_yaml.YAMLSetPrintChemistryOn(False, True, False)  # workers, initial_phreeqc, utility

        # Load database
        status = phreeqcrm_yaml.YAMLLoadDatabase(self.database)
        status = phreeqcrm_yaml.YAMLRunFile(True, True, True, self.phinp)

        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = phreeqcrm_yaml.YAMLRunString(True, False, True, input)
        status = phreeqcrm_yaml.YAMLFindComponents()

        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = phreeqcrm_yaml.YAMLSetTime(time)
        status = phreeqcrm_yaml.YAMLSetTimeStep(time_step)
        status = phreeqcrm_yaml.YAMLSetGridCellCount(self.nchem)

        # convert ic1 to a list
        ic1_flatten = self.ic1_flatten
        # assert isinstance(ic1_flatten, list), 'ic1_flatten is not a list'
        status = phreeqcrm_yaml.YAMLInitialPhreeqc2Module(ic1_flatten)
        status = phreeqcrm_yaml.YAMLRunCells()
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
                sout[0] = np.ones_like(sout[0])*(ctime+dt)  # TODO: generalize

                df = pd.DataFrame(columns=sout_columns)
                for col, arr in zip(df.columns, sout):
                    df[col] = arr
                soutdf = pd.concat([soutdf.astype(df.dtypes), df])  # avoid pandas warning by passing dtypes to empty df

                # Get concentrations from phreeqc
                c_dbl_vect = phreeqc_rm.GetConcentrations()
                conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]  # reshape array

                conc_dic = {}
                for e, c in enumerate(components):
                    conc_dic[c] = np.reshape(conc[e], (nlay, nrow, ncol))
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
        # status = phreeqc_rm.CloseFiles()
        # status = phreeqc_rm.MpiWorkerBreak()
        # mf6.finalize()

        return success


def prep_to_run(wd):
    # check if wd exists
    assert os.path.exists(wd), f'{wd} not found'

    # check if file starting with libmf6 exists
    dll = [f for f in os.listdir(wd) if f.startswith('libmf6.')]
    assert len(dll) == 1, 'libmf6 dll not found in model directory'
    assert os.path.exists(os.path.join(wd, 'mf6rtm.yaml')), 'mf6rtm.yaml not found in model directory'
    dll = os.path.join(wd, 'libmf6')
    yamlfile = os.path.join(wd, 'mf6rtm.yaml')

    return yamlfile, dll


def initialize_phreeqcrm_yaml(yamlfile, nxyz):

    # initialize phreeqc from yaml file
    phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, 1)
    # phreeqc_rm = phreeqcrm.BMIPhreeqcRM(yamlfile)
    status = phreeqc_rm.InitializeYAML(yamlfile)  # FIXME: phreeqcrm initialize yaml failing
    status = phreeqc_rm.SetScreenOn(True)
    sout_columns = phreeqc_rm.GetSelectedOutputHeadings()

    soutdf = pd.DataFrame(columns=sout_columns)
    return phreeqc_rm, soutdf, sout_columns


def initialize_gwt(dll, wd):
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=wd)
    modelnmes = ['Flow'] + [nme.capitalize() for nme in sim.model_names if nme != 'gwf']
    components = [nme.capitalize() for nme in sim.model_names[1:]]

    mf6 = modflowapi.ModflowApi(dll, working_directory=wd)
    mf6.initialize()

    return modelnmes, components, mf6, sim


def solve(wd, reaction=True):
    success = False  # initialize success flag
    yamlfile, dll = prep_to_run(wd)
    modelnmes, components, mf6, sim = initialize_gwt(dll, wd)
    ncpl = calc_nxyz_from_dis(sim)
    phreeqc_rm, soutdf, sout_columns = initialize_phreeqcrm_yaml(yamlfile, ncpl)
    ncomps = len(components)

    nsln = mf6.get_subcomponent_count()
    sim_start = datetime.now()
    # get the current sim time
    ctime = mf6.get_current_time()
    ctimes = [0.0]
    # get the ending sim time
    etime = mf6.get_end_time()

    num_fails = 0
    print(f"Transporting the following components: {', '.join([nme for nme in modelnmes])}")

    print("Starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))

    while ctime < etime:
        # length of the current solve time
        dt = mf6.get_time_step()
        mf6.prepare_time_step(dt)

        c_dbl_vect = solve_gwt(mf6, dt, components, phreeqc_rm, modelnmes, nsln, num_fails, ncpl, ncomps)

        conc_dic = solve_reactions(mf6, ctime, dt, sout_columns, soutdf, reaction)

        mf6.finalize_time_step()
        ctime = mf6.get_current_time()  # update the current time tracking

    sim_end = datetime.now()
    td = (sim_end - sim_start).total_seconds() / 60.0
    if num_fails > 0:
        print("\nFailed to converge {0} times".format(num_fails))
    print("\n")

    # save selected ouput to csv
    soutdf.to_csv(os.path.join(wd, 'sout.csv'), index=False)

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


def solve_gwt(mf6, dt, components, phreeqc_rm, modelnmes, nsln, num_fails, ncpl, ncomps):

    sol_start = datetime.now()
    status = phreeqc_rm.SetTimeStep(dt*86400)  # FIXME: generalize	with time_units_dic and mf6.sim.time_units
    kiter = 0

    # prep to solve
    for sln in range(1, nsln+1):
        mf6.prepare_solve(sln)

    # the one-based stress period number
    stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
    time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]

    # mf6 transport loop block
    for sln in range(1, nsln+1):
        # if self.fixed_components is not None and modelnmes[sln-1] in self.fixed_components:
        #     print(f'not transporting {modelnmes[sln-1]}')
        #     continue

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
            mf6_conc_array.append(concentration_m3_to_l(mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) - 0.0))  # TODO: code charge offset here

        else:
            mf6_conc_array.append(concentration_m3_to_l(mf6.get_value(mf6.get_var_address("X", f'{c.upper()}'))))

    c_dbl_vect = np.reshape(mf6_conc_array, ncpl*ncomps)  # flatten array
    return c_dbl_vect


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

# def get_mf6_disv(sim):
#     #TODO: implement this function
#     return icpl, nvert, vertices, cell2d, top, botm


def solve_reactions(self, mf6, ctime, time_step, sout_columns, soutdf, dt):

    phreeqc_rm = self.phreeqc_rm
    # get arrays from mf6 and flatten for phreeqc
    print(f'\nGetting concentration arrays --- time step: {time_step} --- elapsed time: {ctime}')

    status = phreeqc_rm.SetTemperature([self.init_temp[0]] * self.ncpl)
    # status = phreeqc_rm.SetPressure([2.0] * nxyz)

    # update phreeqc time and time steps
    status = phreeqc_rm.SetTime(ctime*86400)

    # allow phreeqc to print some info in the terminal
    print_selected_output_on = True
    print_chemistry_on = True
    status = phreeqc_rm.SetSelectedOutputOn(print_selected_output_on)
    status = phreeqc_rm.SetPrintChemistryOn(print_chemistry_on, True, True)

    # set concentrations for reactions
    # status = phreeqc_rm.SetConcentrations(c_dbl_vect)

    # reactions loop
    message = '\nBeginning reaction calculation               {} days\n'.format(ctime)
    phreeqc_rm.LogMessage(message)
    phreeqc_rm.ScreenMessage(message)
    status = phreeqc_rm.RunCells()
    if status < 0:
        print('Error in RunCells: {0}'.format(status))
    # selected ouput
    sout = phreeqc_rm.GetSelectedOutput()
    sout = [sout[i:i + self.ncpl] for i in range(0, len(sout), self.ncpl)]

    # add time to selected ouput
    sout[0] = np.ones_like(sout[0])*(ctime+dt)  # TODO: generalize

    df = pd.DataFrame(columns=sout_columns)
    for col, arr in zip(df.columns, sout):
        df[col] = arr
    soutdf = pd.concat([soutdf.astype(df.dtypes), df])  # avoid pandas warning by passing dtypes to empty df

    # Get concentrations from phreeqc
    c_dbl_vect = phreeqc_rm.GetConcentrations()
    conc = [c_dbl_vect[i:i + self.ncpl] for i in range(0, len(c_dbl_vect), self.ncpl)]  # reshape array

    conc_dic = {}
    for e, c in enumerate(self.components):
        conc_dic[c] = np.reshape(conc[e], (self.nlay, self.nrow, self.ncol))
        conc_dic[c] = conc[e]
    # Set concentrations in mf6
        print(f'\nTransferring concentrations to mf6 for component: {c}')
        if c.lower() == 'charge':
            mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]) + self.charge_offset)

        else:
            mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]))
    return conc_dic


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
