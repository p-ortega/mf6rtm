from pathlib import Path
import os
# import shutil
# import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image

import pandas as pd
import numpy as np
# import itertools
# import flopy
# import pyemu
import phreeqcrm
import modflowapi
# from modflowapi import Callbacks
from modflowapi.extensions import ApiSimulation
from utils import*
from datetime import datetime
from time import sleep

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
        self.ic = ic #None means no initial condition (-1)

    def set_ic(self, ic):
        assert isinstance(ic, (int, float, np.ndarray)), 'ic must be an int, float or ndarray'
        # if isinstance(ic, (int, float)):
        #     ic = [ic]
        self.ic = ic
        

class GasPhase(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)

class Solutions(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)

class EquilibriumPhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)

class ExchangePhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)

class KineticPhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters

class ChemStress():
    def __init__(self, packnme) -> None:
        self.packnme = packnme
        self.sol_spd = None
        self.packtype = None

    def set_spd(self, sol_spd):
        self.sol_spd = sol_spd

    def set_packtype(self, packtype):
        self.packtype = packtype

class Mup3d(object):
    def __init__(self, name, solutions, nlay, nrow, ncol):
        self.name = name
        self.wd = None
        self.charge_offset = 0.0
        self.database = os.path.join('database', 'pht3d_datab.dat')
        self.solutions = solutions
        self.equilibrium_phases = None
        self.kinetic_phases = None
        self.exchange_phases = None
        self.postfix = None
        # self.gas_phase = None
        # self.solid_solutions = None
        self.phreeqc_rm = None
        self.sconc = None
        self.phinp = None
        self.components = None
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.ncpl = self.nlay*self.nrow*self.ncol

        if self.solutions.ic is None:
            self.solutions.ic = [1]*self.ncpl
        if isinstance(self.solutions.ic, (int, float)):
            self.solutions.ic = np.reshape([self.solutions.ic]*self.ncpl, (self.nlay, self.nrow, self.ncol))
        assert self.solutions.ic.shape == (self.nlay, self.nrow, self.ncol), f'Initial conditions array must be an array of the shape ({nlay}, {nrow}, {ncol}) not {self.solutions.ic.shape}'


    # def set_kinetic_phases(self, kinetic_phases):
    #     assert isinstance(kinetic_phases, KineticPhases), 'kinetic_phases must be an instance of the KineticPhases class'
    #     attribute_name = kinetic_phases.packnme
    #     setattr(self, attribute_name, kinetic_phases)
    
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
        #change all keys from eq_phases so they start from 0
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
        #joint current directory with wd, check if exist, create if not
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
    
    def generate_phreeqc_script(self):
        
        #where to save the phinp file
        filename=os.path.join(self.wd,'phinp.dat')
        self.phinp = filename 
        #assert that database in self.database exists 
        assert os.path.exists(self.database), f'{self.database} not found'

        # Check if all compounds are in the database
        names = get_compound_names(self.database)
        assert all([key in names for key in self.solutions.data.keys() if key not in ["pH", "pe"]]), f'Not all compounds are in the database - check: {", ".join([key for key in self.solutions.data.keys() if key not in names and key not in ["pH", "pe"]])}'

        script = ""

        # Convert single values to lists
        for key, value in self.solutions.data.items():
            if not isinstance(value, list):
                self.solutions.data[key] = [value]
        # replace all values in self.solutinons.data that are 0.0 to a very small number
        for key, value in self.solutions.data.items():
            self.solutions.data[key] = [1e-30 if val == 0.0 else val for val in value]

        # Get the number of solutions
        num_solutions = len(next(iter(self.solutions.data.values())))

        # Initialize the list of previous concentrations and phases

        for i in range(num_solutions):
            # Get the current concentrations and phases
            concentrations = {species: values[i] for species, values in self.solutions.data.items()}
            script += handle_block(concentrations, generate_solution_block, i, temp=25, water =1)

        #check if self.equilibrium_phases is not None
        if self.equilibrium_phases is not None:
            for i in self.equilibrium_phases.data.keys():
                # Get the current   phases
                phases = self.equilibrium_phases.data[i]
                # check if all equilibrium phases are in the database
                names = get_compound_names(self.database, 'PHASES')
                assert all([key in names for key in phases.keys()]), 'not all compounds are in the database - check names'

                # Handle the  EQUILIBRIUM_PHASES blocks
                script += handle_block(phases, generate_phases_block, i)
                
        #check if self.exchange_phases is not None
        if self.exchange_phases is not None:
            # Get the current   phases
            phases = self.exchange_phases.data
            # check if all exchange phases are in the database
            names = get_compound_names(self.database, 'EXCHANGE')
            assert all([key in names for key in phases.keys()]), 'not all compounds are in the database - check names'
            
            num_exch = len(next(iter(self.exchange_phases.data.values())))
            for i in range(num_exch):
                # Get the current concentrations and phases
                concentrations = {species: values[i] for species, values in self.exchange_phases.data.items()}
                script += handle_block(concentrations, generate_exchange_block, i)

        # add end of line before postfix
        script += endmainblock

        # Append the postfix file to the script
        if os.path.isfile(self.postfix):
            with open(self.postfix, 'r') as source:  # Open the source file in read mode
                script += '\n'
                script += source.read()

        with open(filename, 'w') as file:
            file.write(script)
        return script

    def initialize(self, nthreads = 1):
        '''Initialize a solution with phreeqcrm and returns a dictionary with components as keys and 
            concentration array in moles/m3 as items
        '''
        #get model dis info
        # dis = sim.get_model(sim.model_names[0]).dis

        # create phinp
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
        # databasews = self.database  
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

        #this gets a column slice
        ic1[:, 0] = np.reshape(self.solutions.ic, self.ncpl)

        if isinstance(self.equilibrium_phases, EquilibriumPhases):
            ic1[:, 1] = np.reshape(self.equilibrium_phases.ic, self.ncpl)

        #     if len(self.equilibrium_phases.ic) == 1:
        #         self.equilibrium_phases.ic = [self.equilibrium_phases.ic[0]]*self.ncpl
        #     ic1[:, 1] = np.reshape(self.equilibrium_phases.ic, self.ncpl)  # Equilibrium phases
        # else:
        #     ic1[:, 1] = -1
        if isinstance(self.exchange_phases, ExchangePhases):
            ic1[:, 2] = np.reshape(self.exchange_phases.ic, self.ncpl)  # Exchange 1    

        ic1[:, 3] = -1  # Surface      
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

        print('Phreeqc initialized')

        return 
    
    def initiliaze_chem_stress(self, attr, nthreads = 1):
        '''Initialize a solution with phreeqcrm and returns a dictionary with components as keys and 
            concentration array in moles/m3 as items
        '''
        print('Initializing ChemStress')
        #check if self has a an attribute that is a class ChemStress but without knowing the attribute name
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
            ic1[e]    =  i  # Solution 1
            # ic1[nxyz + i]     = -1  # Equilibrium phases none
            # ic1[2 * nxyz + i] =  -1  # Exchange 1
            # ic1[3 * nxyz + i] = -1  # Surface none
            # ic1[4 * nxyz + i] = -1  # Gas phase none
            # ic1[5 * nxyz + i] = -1  # Solid solutions none
            # ic1[6 * nxyz + i] = -1  # Kinetics none
        status = phreeqc_rm.InitialPhreeqc2Module(ic1)
        # # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = phreeqc_rm.SetTime(time)
        status = phreeqc_rm.SetTimeStep(time_step)

        status = phreeqc_rm.RunCells()
        c_dbl_vect = phreeqc_rm.GetConcentrations()
        c_dbl_vect = concentration_l_to_m3(c_dbl_vect)

        c_dbl_vect = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]

        #find charge in c_dbl_vect and add charge_offset
        for i, c in enumerate(components):
            if c.lower() == 'charge':
                c_dbl_vect[i] += self.charge_offset

        sconc = {}
        for i in range(nxyz):
            sconc[i] = [array[i] for array in c_dbl_vect]

        status = phreeqc_rm.CloseFiles()
        status = phreeqc_rm.MpiWorkerBreak()

        #set as attribute
        setattr(getattr(self, attr), 'data', sconc)
        setattr(getattr(self, attr), 'auxiliary',components)
        print(f'ChemStress {attr} initialized')
        return sconc
    
    def run_mup3d(self, sim, dll = None, reaction = True):
        """
        Modflow6 API and PhreeqcRM integration function to solve model.

        Parameters
        ----------
        sim(mf6.simulation): the MODFLOW-6 simulation object (from flopy)
        dll (Path like): the MODFLOW-6 shared/DLL library filename
        reaction (bool): to indicate wether to invoke phreeqcrm or not. Default is True
        """

        print('\n-----------------------------  WELCOME TO  MUP3D -----------------------------\n')

        success = False #initialize success flag

        phreeqc_rm = self.phreeqc_rm
        sim_ws = sim.simulation_data.mfpath.get_sim_path()
        dll = os.path.join(self.wd, 'libmf6')

        nlay, nrow, ncol = get_mf6_dis(sim)
        nxyz = nlay*nrow*ncol
        
        modelnmes = ['Flow'] + [nme.capitalize() for nme in sim.model_names if nme != 'gwf'] #revise later

        print(f"Transporting the following components: {', '.join([nme for nme in modelnmes])}")

        components = [nme.capitalize() for nme in sim.model_names[1:]]

        mf6 = modflowapi.ModflowApi(dll, working_directory = sim_ws)
        
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
        
        #some phreeqc output settings
        phreeqc_rm.SetScreenOn(True)
        sout_columns = phreeqc_rm.GetSelectedOutputHeadings()
        soutdf = pd.DataFrame(columns = sout_columns)

        # let's do it!
        while ctime < etime:
            
            sol_start = datetime.now()
            # length of the current solve time
            dt = mf6.get_time_step()
            # prep the current time step
            mf6.prepare_time_step(dt)
    
            status = phreeqc_rm.SetTimeStep(dt*86400)#FIXME: generalize	with time_units_dic and mf6.sim.time_units
            kiter = 0

            # prep to solve
            for sln in range(1, nsln+1):
                mf6.prepare_solve(sln)

            # the one-based stress period number
            stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
            time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]
            
            ### mf6 transport loop block
            for sln in range(1, nsln+1):
                # max number of solution iterations
                max_iter = mf6.get_value(mf6.get_var_address("MXITER", f"SLN_{sln}")) #FIXME: not sure to define this inside the loop
                mf6.prepare_solve(sln)

                print(f'\nSolving solution {sln} - Solving {modelnmes[sln-1]}')
                while kiter < max_iter:
                    convg = mf6.solve(sln)
                    if convg:
                        td = (datetime.now() - sol_start).total_seconds() / 60.0
                        print("Transport stress period: {0} --- time step: {1} --- converged with {2} iters --- took {3:10.5G} mins".format(stress_period, time_step, kiter,td))
                        break
                    kiter += 1
                    
                if not convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    print("Transport stress period: {0} --- time step: {1} --- did not converge with {2} iters --- took {3:10.5G} mins".format(stress_period, time_step, kiter,td))
                    num_fails += 1
                try:
                    mf6.finalize_solve(sln)
                    print(f'\nSolution {sln} finalized')
                except:
                    pass

            mf6_conc_array = []
            for c in components:
                if c.lower() == 'charge':
                    mf6_conc_array.append(concentration_m3_to_l (mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) - self.charge_offset ) )
                    
                else:
                    mf6_conc_array.append( concentration_m3_to_l( mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) ) )

            c_dbl_vect = np.reshape(mf6_conc_array, self.ncpl*self.ncomps) #flatten array

           ### Phreeqc loop block
            if reaction:
                #get arrays from mf6 and flatten for phreeqc
                print(f'\nGetting concentration arrays --- time step: {time_step} --- elapsed time: {ctime}')

                # status = phreeqc_rm.SetTemperature([20.0] * nxyz)
                # status = phreeqc_rm.SetPressure([2.0] * nxyz)

                #update phreeqc time and time steps
                status = phreeqc_rm.SetTime(ctime*86400)
                
                #allow phreeqc to print some info in the terminal
                print_selected_output_on = True
                print_chemistry_on = True
                status = phreeqc_rm.SetSelectedOutputOn(True)
                status = phreeqc_rm.SetPrintChemistryOn(print_chemistry_on, True, True) 

                #set concentrations for reactions
                status = phreeqc_rm.SetConcentrations(c_dbl_vect)  

                #reactions loop
                message = '\nBeginning reaction calculation               {} days\n'.format(ctime)
                phreeqc_rm.LogMessage(message)
                phreeqc_rm.ScreenMessage(message)
                status = phreeqc_rm.RunCells()
                if status < 0:
                    print('Error in RunCells: {0}'.format(status))	
                #selected ouput
                sout = phreeqc_rm.GetSelectedOutput()
                sout = [sout[i:i + nxyz] for i in range(0, len(sout), nxyz)]

                #add time to selected ouput
                sout[0] = np.ones_like(sout[0])*(ctime+dt) #TODO: generalize

                df = pd.DataFrame(columns = sout_columns)
                for col, arr in zip(df.columns, sout):
                    df[col] = arr
                soutdf = pd.concat([soutdf.astype(df.dtypes), df]) #avoid pandas warning by passing dtypes to empty df

                # Get concentrations from phreeqc 
                c_dbl_vect = phreeqc_rm.GetConcentrations()
                conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)] #reshape array

                conc_dic = {} 
                for e, c in enumerate(components):
                    conc_dic[c] = np.reshape(conc[e], (nlay, nrow, ncol))
                    conc_dic[c] = conc[e]
                # Set concentrations in mf6
                    print(f'\nTransferring concentrations to mf6 for component: {c}')
                    if c.lower() == 'charge':
                        mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c] ) + self.charge_offset)

                    else:
                        mf6.set_value(f'{c.upper()}/X', concentration_l_to_m3(conc_dic[c]))


            mf6.finalize_time_step()
            ctime = mf6.get_current_time()  #update the current time tracking

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0
        if num_fails > 0:
            print("\nFailed to converge {0} times".format(num_fails))
        print("\n")

        #save selected ouput to csv
        soutdf.to_csv(os.path.join(sim_ws,'sout.csv'), index=False)
        
        print("\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT),td))

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
    
def get_mf6_dis(sim):
    # extract dis from modflow6 sim object 
    dis = sim.get_model(sim.model_names[0]).dis
    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()
    return nlay, nrow, ncol

# def get_mf6_disv(sim):
#     #TODO: implement this function
#     return icpl, nvert, vertices, cell2d, top, botm

def mrbeaker():
    #docstrings
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
    mrate = q*conc #M/T
    return mrate

def concentration_volbulk_to_volwater(conc_volbulk, porosity):
    '''Calculate concentrations as volume of pore water from bulk volume and porosity
    '''
    conc_volwater = conc_volbulk*(1/porosity)
    return conc_volwater