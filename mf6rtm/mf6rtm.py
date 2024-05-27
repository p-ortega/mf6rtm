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
from modflowapi import Callbacks
from modflowapi.extensions import ApiSimulation
from .utils import*
from datetime import datetime

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

endmainblock  = '''\nEND\n
PRINT
	-reset false
END\n'''

class Block:
    def __init__(self, data, ic=None) -> None:
        self.data = data
        self.names = [key for key in data.keys()]
        self.ic = ic #None means no initial condition

    def set_ic(self, ic):
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

class KineticPhases(Block):
    def __init__(self, data) -> None:
        super().__init__(data)
        # super().__init__(ic)
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters


class Mup3d(object):
    def __init__(self, solutions, nlay, nrow, ncol):
        self.database = os.path.join('database','pht3d_datab.dat')
        self.solutions = solutions
        self.equilibrium_phases = None
        self.kinetic_phases = None
        # self.exchange = None
        # self.gas_phase = None
        # self.solid_solutions = None
        self.phreeqc_rm = None
        self.sconc = None
        self.phinp = 'phinp.dat'
        self.components = None
        self.solutions = solutions
        self.equilibrium_phases = None
        self.kinetic_phases = None
        # self.exchange = None
        # self.gas_phase = None
        # self.solid_solutions = None
        self.phreeqc_rm = None
        self.sconc = None
        self.phinp = 'phinp.dat'
        self.components = None
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.ncpl = self.nlay*self.nrow*self.ncol
    
        #assert that database in self.database exists 
        assert os.path.exists(self.database), f'{self.database} not found'


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

    def set_equilibrium_phases(self, eq_phases):
        assert isinstance(eq_phases, EquilibriumPhases), 'eq_phases must be an instance of the EquilibriumPhases class'
        self.equilibrium_phases = eq_phases


    def generate_phreeqc_script(self, filename='phinp.dat'):

        # Check if all compounds are in the database
        names = get_compound_names(self.database)
        assert all([key in names for key in self.solutions.data.keys() if key not in ['pH', 'pe']]), 'not all compounds are in the database - check names'

        # check if all equilibrium phases are in the database
        names = get_compound_names(self.database, 'PHASES')
        assert all([key in names for key in self.equilibrium_phases.data.keys()]), 'not all compounds are in the database - check names'

        postfix = 'postfix.phqr'
        script = ""

        # Convert single values to lists
        for key, value in self.solutions.data.items():
            if not isinstance(value, list):
                self.solutions.data[key] = [value]

        # Get the number of solutions
        num_solutions = len(next(iter(self.solutions.data.values())))
        num_phases = max(len(phases) for phases in self.equilibrium_phases.data.values())

        # Iterate over the dataionary and fill arrays with zeros until they reach the maximum length
        for name, phases in self.equilibrium_phases.data.items():
            while len(self.equilibrium_phases.data) < num_phases:
                phases.append([0.0, 0.0])

        # Initialize the list of previous concentrations and phases

        for i in range(num_solutions):
            # Get the current concentrations and phases
            concentrations = {species: values[i] for species, values in self.solutions.data.items()}
            script += handle_block(concentrations, generate_solution_block, i)

        for i in range(num_phases):
            # Get the current   phases
            phases = {name: phase[i] for name, phase in self.equilibrium_phases.data.items() if i < len(self.equilibrium_phases.data)}
            # Handle the  EQUILIBRIUM_PHASES blocks
            script += handle_block(phases, generate_phases_block, i)

        # add end of line before postfix
        script += endmainblock

        # Append the postfix file to the script
        if os.path.isfile(postfix):
            with open(postfix, 'r') as source:  # Open the source file in read mode
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

        # initialize phreeqccrm object 
        self.phreeqc_rm = phreeqcrm.PhreeqcRM(self.ncpl, nthreads)
        status = self.phreeqc_rm.SetComponentH2O(False)
        self.phreeqc_rm.UseSolutionDensityVolume(False)
        self.phreeqc_rm.OpenFiles()

        # Set concentration units
        status = self.phreeqc_rm.SetUnitsSolution(2) 

        # mf6 handles poro . set to 1          
        poro = np.full((self.ncpl), 1.)
        status = self.phreeqc_rm.SetPorosity(poro)

        print_chemistry_mask = np.full((self.ncpl), 1)
        status = self.phreeqc_rm.SetPrintChemistryMask(print_chemistry_mask)
        nchem = self.phreeqc_rm.GetChemistryCellCount()

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
        components = self.phreeqc_rm.GetComponents()

        # set components as attribute
        self.components = components

        # Initial equilibration of cells
        time = 0.0
        time_step = 0.0
        status = self.phreeqc_rm.SetTime(time)
        status = self.phreeqc_rm.SetTimeStep(time_step)
        # eqphases = self.phreeqc_rm.GetEquilibriumPhases()

        #TODO: generalize this to have always at least solutions with ic 1. maybe an assert

        ic1 = np.ones((self.ncpl, 7), dtype=int)*-1
        if self.solutions.ic is None:
            self.solutions.ic = 1
        ic1[:, 0] = self.solutions.ic  

        if isinstance(self.equilibrium_phases, EquilibriumPhases):
            ic1[:, 1] = self.equilibrium_phases.ic  # Equilibrium phases
        else:
            ic1[:, 1] = -1

        ic1[:, 2] = -1  # Exchange 1
        
        ic1[:, 3] = -1  # Surface
        
        ic1[:, 4] = -1  # Gas phase
        
        ic1[:, 5] = -1  # Solid solutions
        
        if isinstance(self.kinetic_phases, KineticPhases):
            ic1[:, 6] = self.kinetic_phases.ic  # Kinetics
        else:
            ic1[:, 6] = -1

        ic1_flatten = ic1.flatten('F')

        # set initial conditions as attribute but in a new sub class
        self.ic1 = ic1
        self.ic1_flatten = ic1_flatten

        # initialize ic1 phreeqc to module with phrreeqcrm
        status = self.phreeqc_rm.InitialPhreeqc2Module(ic1_flatten)

        # get initial concentrations from running phreeqc
        status = self.phreeqc_rm.RunCells()
        c_dbl_vect = self.phreeqc_rm.GetConcentrations()
        conc = [c_dbl_vect[i:i + self.ncpl] for i in range(0, len(c_dbl_vect), self.ncpl)]

        self.sconc = {}
        for e, c in enumerate(components):
            get_conc = np.reshape(conc[e], (self.nlay, self.nrow, self.ncol))
            self.sconc[c] = get_conc
            self.sconc[c] = concentration_l_to_m3(get_conc)

        print('Phreeqc initialized')

        return 

def get_model_dis(sim):
    # extract dis from modflow6 sim object 
    dis = sim.get_model(sim.model_names[0]).dis
    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()
    return nlay, nrow, ncol

# def get_model_disv(sim):
#     #TODO: implement this function
#     return icpl, nvert, vertices, cell2d, top, botm

def mrbeaker():
    # Load the image of Mr. Beaker
    mr_beaker_image = Image.open("mrbeaker.png")

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

def get_dis_from_mf6(sim):

    return

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


def mf6rtm_run(dll, sim, phreeqc_rm, reaction = True):
    """
    Modflow6 API and PhreeqcRM integration function to solve model.

    Parameters
    ----------
    dll (Path like): the MODFLOW-6 shared/DLL library filename
    sim_ws (Path like): the model directory
    phreeqc_rm (phreeqcrm object): an initialized phreeqcrm object
    reaction (bool): to indicate wether to invoke phreeqcrm or not. Default is True
    """
    print('\n-----------------------------  WELCOME TO  MUPin3D -----------------------------\n')
    print(mrbeaker())
    print('\nTake your time to appreciate MR BEAKER!')
    
    sim_ws = sim.simulation_data.mfpath.get_sim_path()
    nlay, nrow, ncol = get_model_dis(sim)
    nxyz = nlay*nrow*ncol

    components = components = [nme.capitalize() for nme in sim.model_names[1:]]
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
    
    phreeqc_rm.SetScreenOn(True)
    sout_columns = phreeqc_rm.GetSelectedOutputHeadings()
    soutdf = pd.DataFrame(columns = sout_columns)

    results = []
    # let's do it!
    while ctime < etime:
            
        sol_start = datetime.now()
        # length of the current solve time
        dt = mf6.get_time_step()
        # prep the current time step
        mf6.prepare_time_step(dt)
 
        status = phreeqc_rm.SetTimeStep(dt*86400)

        kiter = 0

        # prep to solve
        for sln in range(1, nsln+1):
            mf6.prepare_solve(sln)

        # the one-based stress period number
        stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
        time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]

        #get arrays from mf6 and flatten for phreeqc
        print(f'\nGetting concentration arrays --- time step: {time_step} --- elapsed time: {ctime}')
        mf6_conc_array = [concentration_m3_to_l( mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) )for c in components]
        c_dbl_vect = flatten_list(mf6_conc_array)

        ### Phreeqc loop block
        if reaction:
            #update phreeqc time and time steps
            status = phreeqc_rm.SetTime(ctime*86400)
            
            #allow phreeqc to print some info in the terminal
            print_selected_output_on = True
            print_chemistry_on = True
            status = phreeqc_rm.SetSelectedOutputOn(True)
            status = phreeqc_rm.SetPrintChemistryOn(print_chemistry_on, False, False) 

            #set concentrations for reactions
            status = phreeqc_rm.SetConcentrations(c_dbl_vect)  

            #reactions loop
            message = '\nBeginning reaction calculation               {} days\n'.format(ctime)
            phreeqc_rm.LogMessage(message)
            phreeqc_rm.ScreenMessage(message)
            status = phreeqc_rm.RunCells()

            #selected ouput
            sout = phreeqc_rm.GetSelectedOutput()
            sout = [sout[i:i + nxyz] for i in range(0, len(sout), nxyz)]

            #add time to selected ouput
            sout[0] = np.ones_like(sout[0])*(ctime+dt) #TODO: generalize

            df = pd.DataFrame(columns = sout_columns)
            for col, arr in zip(df.columns, sout):
                df[col] = arr
            soutdf = pd.concat([soutdf, df])

            #TODO: merge the next two loops into one
            # Get concentrations from phreeqc 
            c_dbl_vect = phreeqc_rm.GetConcentrations()
            conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)] #reshape array
            sconc = {}
            for e, c in enumerate(components):
                sconc[c] = np.reshape(conc[e], (nlay, nrow, ncol))

            # Set concentrations in mf6
            for c in components:
                print(f'\nTransferring concentrations to mf6 for component: {c}')
                c_dbl_vect = concentration_l_to_m3(sconc[c]) #units to m3
                mf6.set_value(f'{c.upper()}/X', c_dbl_vect)

        # solve transport until converged
        for sln in range(1, nsln+1):
            # max number of solution iterations
            max_iter = mf6.get_value(mf6.get_var_address("MXITER", f"SLN_{sln}")) #TODO: not sure to define this inside the loop
            mf6.prepare_solve(sln)

            print(f'\nSolving solution {sln}') #TODO: Tie sln number with gwf or component
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

        mf6.finalize_time_step()
        ctime = mf6.get_current_time()  #update the current time tracking

    sim_end = datetime.now()
    td = (sim_end - sim_start).total_seconds() / 60.0
    print("\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT),td))
    if num_fails > 0:
        print("\nFailed to converge {0} times".format(num_fails))
    print("\n")

    #save selected ouput to csv
    soutdf.to_csv('sout.csv', index=False)

    # Clean up and close api objs
    status = phreeqc_rm.CloseFiles()
    status = phreeqc_rm.MpiWorkerBreak()
    mf6.finalize()