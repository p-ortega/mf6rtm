# imports
import platform
from pathlib import Path
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import math
import time
import itertools
#sys.path.insert(0,os.path.join("dependencies"))

import flopy
import pyemu
import phreeqcrm
import modflowapi
from modflowapi import Callbacks
from modflowapi.extensions import ApiSimulation

from time import sleep

from flopy.utils.gridintersect import GridIntersect
from datetime import datetime
import matplotlib.pyplot as plt

DT_FMT = "%Y-%m-%d %H:%M:%S"


### Model params and setup

nper = 1  # Number of periods
nlay = 1  # Number of layers
Lx = 0.5 #m
ncol = 50  # Number of columns
nrow = 1  # Number of rows
delr = Lx/ncol #10.0  # Column width ($m$)
delc = 1.0  # Row width ($m$)
top = 0.  # Top of the model ($m$)
botm = -1.0  # Layer bottom elevations ($m$)
prsity = 0.32  # Porosity
perlen = 0.24  # Simulation time ($days$)
k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)

k33 = k11  # Vertical hydraulic conductivity ($m/d$)
laytyp = 1
tstep = 0.01
nstp = perlen/tstep #100.0
dt0 = perlen / nstp

v = 0.24

strt = np.zeros((nlay, nrow, ncol), dtype=float)
strt[0, 0, :] = 1  # Starting head ($m$)

icelltype = 1  # Cell conversion type
ibound = np.ones((nlay, nrow, ncol), dtype=int)
ibound[0, 0, -1] = -1

dispersivity = 0.0067 
mixelm = -1  # TVD
rhob = 0.25
sp2 = 0.0  # read, but not used in this problem

# sconc = np.ones((nlay, nrow, ncol), dtype=float)*0.1
# dmcoef = 0.0  # Molecular diffusion coefficient

# Set solver parameter values (and related)
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0
ttsmult = 1.0
dceps = 1.0e-5  # HMOC parameters in case they are invoked
nplane = 1  # HMOC
npl = 0  # HMOC
nph = 4  # HMOC
npmin = 0  # HMOC
npmax = 8  # HMOC
nlsink = nplane  # HMOC
npsink = nph  # HMOC

length_units = "meters"
time_units = "days"


tdis_rc = []
tdis_rc.append((perlen, nstp, 1.0))

chdspd = [
          [(0, 0, ncol - 1), 1.]
          ]

def concentration_liters_to_m3(x):
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

def src_array(q, sconc, c):
    # for k,v in sconc.items():
    #     print(k,v)
    print(f'SRC stress period for component: {c}')
    c_selected = concentration_to_massrate(q, sconc[c])
    spd = [(0,0,0), c_selected]
    src_rec = [spd]
    print(src_rec)
    return src_rec

def wel_array(q, sconc, aux = True):
    # sconc = [v for k,v in sconc.items()]
    spd = [(0,0,0), q]
    if aux:
        spd.extend([v for k,v in sconc.items()])
    wel_rec = [spd]
    # print(wel_rec)
    return wel_rec

def flatten_list(xss):
    return [x for xs in xss for x in xs]

def init_solution(nthreads = 1, init_file = 'initsol.dat'):
    '''Initialize a solution with phreeqcrm and returns a dictionary with components as keys and 
        concentration array in moles/m3 as items
    '''
    nxyz = 1
    phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)
    status = phreeqc_rm.SetComponentH2O(False)
    phreeqc_rm.UseSolutionDensityVolume(False)
    status = phreeqc_rm.SetFilePrefix('initsol')
    phreeqc_rm.OpenFiles()
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
    databasews = os.path.join("database", "pht3d_datab.dat")
    status = phreeqc_rm.LoadDatabase(databasews)
    status = phreeqc_rm.RunFile(True, True, True, init_file)

    # Clear contents of workers and utility
    input = "DELETE; -all"
    status = phreeqc_rm.RunString(True, False, True, input)

    # Get component information - these two functions need to be invoked to find comps
    ncomps = phreeqc_rm.FindComponents()
    components = phreeqc_rm.GetComponents()

    for comp in components:
        phreeqc_rm.OutputMessage(comp)
    phreeqc_rm.OutputMessage("\n")

    ic1 = [-1] * nxyz * 7 
    for i in range(nxyz):
        ic1[i]            =  1  # Solution 1
        ic1[nxyz + i]     = -1  # Equilibrium phases none
        ic1[2 * nxyz + i] =  -1  # Exchange 1
        ic1[3 * nxyz + i] = -1  # Surface none
        ic1[4 * nxyz + i] = -1  # Gas phase none
        ic1[5 * nxyz + i] = -1  # Solid solutions none
        ic1[6 * nxyz + i] = -1  # Kinetics none

    status = phreeqc_rm.InitialPhreeqc2Module(ic1)

    # # Initial equilibration of cells
    time = 0.0
    time_step = 0.0
    status = phreeqc_rm.SetTime(time)
    status = phreeqc_rm.SetTimeStep(time_step)
    # eqphases = phreeqc_rm.GetEquilibriumPhases()
    
    status = phreeqc_rm.RunCells()
    c_dbl_vect = phreeqc_rm.GetConcentrations()
    c_dbl_vect = concentration_liters_to_m3(c_dbl_vect)

    sconc = {c: v for c,v in zip(components, c_dbl_vect)}
    print('solution ready')

    return components, sconc

def initialize_phreeqcrm(sim_name, nthreads = 1, init_file = 'phinp.dat'):

    nxyz = nlay*nrow*ncol
    phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)
    status = phreeqc_rm.SetComponentH2O(False)
    phreeqc_rm.UseSolutionDensityVolume(False)

    status = phreeqc_rm.SetFilePrefix(sim_name)

    phreeqc_rm.OpenFiles()

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
    databasews = os.path.join("database", "pht3d_datab.dat")
    status = phreeqc_rm.LoadDatabase(databasews)

    status = phreeqc_rm.RunFile(True, True, True, init_file)

    # Clear contents of workers and utility
    input = "DELETE; -all"
    status = phreeqc_rm.RunString(True, False, True, input)

    # Get component information - these two functions need to be invoked to find comps
    ncomps = phreeqc_rm.FindComponents()
    components = phreeqc_rm.GetComponents()

    for comp in components:
        phreeqc_rm.OutputMessage(comp)
    phreeqc_rm.OutputMessage("\n")

    ic1 = [-1] * nxyz * 7 
    for i in range(nxyz):
        ic1[i]            =  1  # Solution 1
        ic1[nxyz + i]     = 1  # Equilibrium phases none
        ic1[2 * nxyz + i] =  -1  # Exchange 1
        ic1[3 * nxyz + i] = -1  # Surface none
        ic1[4 * nxyz + i] = -1  # Gas phase none
        ic1[5 * nxyz + i] = -1  # Solid solutions none
        ic1[6 * nxyz + i] = -1  # Kinetics none

    status = phreeqc_rm.InitialPhreeqc2Module(ic1)

    # Initial equilibration of cells
    time = 0.0
    time_step = 0.0
    status = phreeqc_rm.SetTime(time)
    status = phreeqc_rm.SetTimeStep(time_step)
    eqphases = phreeqc_rm.GetEquilibriumPhases()
    
    status = phreeqc_rm.RunCells()
    c_dbl_vect = phreeqc_rm.GetConcentrations()
    conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]

    sconc = {}
    for e, c in enumerate(components):
        get_conc = np.reshape(conc[e], (nlay, nrow, ncol))
        sconc[c] = get_conc
        sconc[c] = concentration_liters_to_m3(get_conc)
        # print(sconc[c])
    print('initialize ready')

    return components, phreeqc_rm, sconc

def mf6rtm_api_test(dll, sim_ws, phreeqc_rm, components, reaction = True):

    mf6 = modflowapi.ModflowApi(dll, working_directory = sim_ws)
    mf6.initialize()

    nsln = mf6.get_subcomponent_count()
    nxyz = nlay*nrow*ncol

    sim_start = datetime.now()
    print("...starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))

    # reset the node tracking containers

    # get the current sim time
    ctime = mf6.get_current_time()
    ctimes = [0.0]

    # get the ending sim time
    etime = mf6.get_end_time()

    # max number of solution iterations
    num_fails = 0
    
    phreeqc_rm.SetScreenOn(True)
    columns = phreeqc_rm.GetSelectedOutputHeadings()
    stoutdf = pd.DataFrame(columns = columns)

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

        # array to store transported components
        print(f'\nGetting concentration arrays --- time step: {time_step} --- elapsed time: {ctime}')
        mf6_conc_array = [concentration_m3_to_l( mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) )for c in components]
        # print(mf6_conc_array)
        # mf6_conc_array = concentration_m3_to_l(mf6_conc_array)
        c_dbl_vect = flatten_list(mf6_conc_array)


        ### Phreeqc BLOCK
        if reaction:
            #update phreeqc time and time steps
            status = phreeqc_rm.SetTime(ctime*86400)
            
            print_selected_output_on = True
            print_chemistry_on = True
            status = phreeqc_rm.SetSelectedOutputOn(True)
            status = phreeqc_rm.SetPrintChemistryOn(print_chemistry_on, False, False) 

            status = phreeqc_rm.SetConcentrations(c_dbl_vect)  
            message = '\nBeginning reaction calculation               {} days\n'.format(ctime)
            phreeqc_rm.LogMessage(message)
            phreeqc_rm.ScreenMessage(message)
            status = phreeqc_rm.RunCells()

            print(phreeqc_rm.GetTimeStep())

            sout = phreeqc_rm.GetSelectedOutput()
            sout = [sout[i:i + nxyz] for i in range(0, len(sout), nxyz)]   
            df = pd.DataFrame(columns = columns)
            for col, arr in zip(df.columns, sout):
                df[col] = arr
            stoutdf = pd.concat([stoutdf, df])

            c_dbl_vect = phreeqc_rm.GetConcentrations()

            
            conc = [c_dbl_vect[i:i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]
            sconc = {}
            for e, c in enumerate(components):
                sconc[c] = np.reshape(conc[e], (nlay, nrow, ncol))

            for c in components:
                print(f'\nTransferring concentrations to mf6 for component: {c}')
                c_dbl_vect = concentration_liters_to_m3(sconc[c])
                mf6.set_value(f'{c.upper()}/X', c_dbl_vect)

        # solve transport until converged
        for sln in range(1, nsln+1):
            max_iter = mf6.get_value(mf6.get_var_address("MXITER", f"SLN_{sln}"))
            mf6.prepare_solve(sln)
            print(f'Solving solution {sln}')
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
                print(f'finalize sol {sln}')
                mf6.finalize_solve(sln)
            except:
                pass

        # mf6_conc_array = [mf6.get_value(mf6.get_var_address("X", f'{c.upper()}')) for c in components]
        # results.append(mf6_conc_array)

        mf6.finalize_time_step()
        # update the current time tracking
        ctime = mf6.get_current_time()

    sim_end = datetime.now()
    td = (sim_end - sim_start).total_seconds() / 60.0
    print("\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT),td))
    if num_fails > 0:
        print("...failed to converge {0} times".format(num_fails))
    print("\n")

    stoutdf.to_csv('sout.csv', index=False)

    # Clean up
    status = phreeqc_rm.CloseFiles()
    status = phreeqc_rm.MpiWorkerBreak()
    mf6.finalize()

    return

def api_test(dll, sim_ws):
    

    mf6 = modflowapi.ModflowApi(dll, working_directory = sim_ws)
    mf6.initialize()
    nsln = mf6.get_subcomponent_count()
    sim_start = datetime.now()
    print("...starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))
    # reset the node tracking containers

    # get the current sim time
    ctime = mf6.get_current_time()
    ctimes = [0.0]
    # get the ending sim time
    etime = mf6.get_end_time()
    # max number of solution iterations
    max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_2"))
    num_fails = 0
    # let's do it!
    while ctime < etime:
        sol_start = datetime.now()
        # length of the current solve time
        dt = mf6.get_time_step()
        # prep the current time step
        mf6.prepare_time_step(dt)
        kiter = 0
        # prep to solve
        # mf6.prepare_solve(1)
        # the one-based stress period number
        stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
        time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]

        # solve until converged
        for sln in range(1, nsln+1):
            mf6.prepare_solve(sln)
            while kiter < max_iter:
                convg = mf6.solve(sln)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    print("transport stress period {0}, time step {1}, converged with {2} iters, took {3:10.5G} mins".format(stress_period, time_step, kiter,td))
                    break
                kiter += 1
            try:
                mf6.finalize_solve(sln)
            except:
                pass
        if not convg:
            td = (datetime.now() - sol_start).total_seconds() / 60.0
            print("transport stress period {0}, time step {1}, did not converged, {2} iters, took {3:10.5G} mins".format(
                stress_period, time_step, kiter, td))
            num_fails += 1

        mf6.finalize_time_step()
        # update the current time tracking
        ctime = mf6.get_current_time()


    sim_end = datetime.now()
    td = (sim_end - sim_start).total_seconds() / 60.0
    print("\n...transport solution finished at {0}, took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT),td))
    if num_fails > 0:
        print("...failed to converge {0} times".format(num_fails))
    print("\n")
    mf6.finalize()

    return


def prep_bins(dest_path, src_path=os.path.join('bin'),  get_only=[]):
    if "linux" in platform.platform().lower():
        bin_path = os.path.join(src_path, "linux")
    elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
        bin_path = os.path.join(src_path, "mac")
    else:
        bin_path = os.path.join(src_path, "win")
    files = os.listdir(bin_path)
    if len(get_only)>0:
        files = [f for f in files if f.split(".")[0] in get_only]

    for f in files:
        if os.path.exists(os.path.join(dest_path,f)):
            try:
                os.remove(os.path.join(dest_path,f))
            except:
                continue
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dest_path,f))
    return

def run_mf6(sim):
    ws = sim.simulation_data.mfpath.get_sim_path()
    pyemu.os_utils.run('mf6', cwd  = ws)
    return 
def run_mt3dms(mt):
    ws = mt.model_ws 
    pyemu.os_utils.run('mf2005 gwf.nam', cwd  = ws)
    pyemu.os_utils.run('mt3dms gwt.nam', cwd  = ws)
    return 

def build_gwf_model(ws = 'model', sim_name = '1dtest'):
    '''Creates GWF model only, saving flows for an independent subsequent GWT model
    '''
    gwfname = "gwf_" + sim_name
    sim_ws = os.path.join(ws, sim_name)
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name='mf6')

    if os.path.exists(sim_ws):
        shutil.rmtree(sim_ws)
    os.makedirs(sim_ws)

    # Instantiating MODFLOW 6 time discretization
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)
    # Instantiating MODFLOW 6 groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        model_nam_file=f"{gwfname}.nam",
    )

    # Instantiating MODFLOW 6 solver for flow model
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{gwfname}.ims",
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    # Instantiating MODFLOW 6 discretization package
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=np.ones((nlay, nrow, ncol), dtype=int),
        filename=f"{gwfname}.dis",
    )
    dis.set_all_data_external()

    # Instantiating MODFLOW 6 node-property flow package
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_saturation = True,
        icelltype=icelltype,
        k=k11,
        k33=k33,
        save_specific_discharge=True,
        filename=f"{gwfname}.npf",
    )
    npf.set_all_data_external()
    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{gwfname}.ic")

    # Instantiating MODFLOW 6 constant head package
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        pname="CHD",
        filename=f"{gwfname}.chd",
    )
    chd.set_all_data_external()

    wel = flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=wel_rec,
            auxiliary = 'concentration',
            pname = 'wel',
            filename=f"{gwfname}.wel"
        )
    wel.set_all_data_external()
    # Instantiating MODFLOW 6 output control package for flow model
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwfname}.hds",
        budget_filerecord=f"{gwfname}.cbb",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
        #---write
    sim.write_simulation() 
    return sim

def build_gwt_model(sim_gwf, gwf_name = '1dtest', sp = 'chloride'):
    '''Creates an independent GWT linked to a parent GWF model
    '''
    model_name = f"{sp}" 
    ws = sim_gwf.simulation_data.mfpath.get_sim_path()
    gwt_ws = os.path.join(ws,f'{sp}')

    if os.path.exists(gwt_ws):
        shutil.rmtree(gwt_ws)
    os.makedirs(gwt_ws)

    # sim_gwf = flopy.mf6.MFSimulation.load(gwf_name, 'mf6', 'mf6' , ws)
    gwf_model = sim_gwf.get_model(sim_gwf.model_names[0])

    sim = flopy.mf6.MFSimulation(sim_ws=gwt_ws,
                                sim_name = model_name,
                            continue_=True) 
    
    # Instantiating MODFLOW 6 groundwater transport package
    gwt = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=model_name,
        model_nam_file=f"{model_name}.nam"
    )
    perioddata = sim_gwf.tdis.perioddata.array.tolist()
    nper = sim_gwf.tdis.nper.get_data()
    start_date_time = sim_gwf.tdis.start_date_time.get_data()
    print('--- Building TDIS package ---')
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                    nper=nper, 
                                    perioddata=perioddata, #gwt_perioddata,
                                    time_units='days', 
                                    start_date_time=start_date_time)
    
    # create iterative model solution and register the gwt model with it
    # nouter, ninner = 100, 300 
    # hclose, rclose, relax = 1e-6, 1e-6, 1.0 
    print('--- Building IMS package ---')
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{model_name}.ims",
    )
    sim.register_ims_package(ims, [gwt.name])

    print('--- Building DIS package ---')
    dis = gwf_model.dis

    # create grid object
    nlay=dis.nlay.get_data()
    nrow=dis.nrow.get_data()
    ncol=dis.ncol.get_data()
    dis = flopy.mf6.ModflowGwfdis(
                gwt,
                length_units=dis.length_units.get_data(),
                nlay=dis.nlay.get_data(),
                nrow=dis.nrow.get_data(),
                ncol=dis.ncol.get_data(),
                delr=dis.delr.get_data(),
                delc=dis.delc.get_data(),
                top=dis.top.get_data(),
                botm=dis.botm.get_data(),
                idomain=dis.idomain.get_data(),
                filename=f"{model_name}.dis",
            )
    
    ic = flopy.mf6.ModflowGwtic(gwt, strt=sconc, filename=f"{model_name}.ic")

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = ['wel', 'aux', 'concentration']
    flopy.mf6.ModflowGwtssm(
        gwt, 
        sources=sourcerecarray, 

        filename=f"{model_name}.ssm"
    )

    # Instantiating MODFLOW 6 transport adv package
    print('--- Building ADV package ---')
    adv = flopy.mf6.ModflowGwtadv(
        gwt,
        scheme="tvd",
    )

    # Instantiating MODFLOW 6 transport dispersion package
    alpha_l = np.ones(shape=(nlay, nrow, ncol))*dispersivity # Longitudinal dispersivity ($m$)
    alpha_th = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse horizontal dispersivity ($m$)
    alpha_tv = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse vertical dispersivity ($m$)


    print('--- Building DSP package ---')
    dsp = flopy.mf6.ModflowGwtdsp(
        gwt,
        xt3d_off=True,
        alh=alpha_l,
        ath1=alpha_th,
        atv = alpha_tv,
        # diffc = diffc,
        filename=f"{model_name}.dsp",
    )
    dsp.set_all_data_external()

    # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
    print('--- Building MST package ---')

    sorption = None
    zero_order_decay = None
    first_order_decay = None

    # cnc = flopy.mf6.ModflowGwtcnc(
    #         gwt,
    #         # maxbound=maxbounds_cnc,
    #         stress_period_data=cnc_spd,
    #         save_flows=True,
    #         pname="cnc",
    #         filename=f"{model_name}.cnc",
    #         boundnames = True
    #         )
    # cnc.set_all_data_external()
    mst = flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=prsity,
        first_order_decay=first_order_decay,
        # decay = decay,
        # decay_sorbed=decay_sorbed,
        # sorption= sorption,
        # bulk_density=bd_arr, 
        # distcoef=kd_arr, #Kd m3/mg
        filename=f"{model_name}.mst",
    )
    mst.set_all_data_external()

    print('building FMI and OC ...')
    flow_packagedata = [
        ("GWFHEAD", os.path.join("..", f"{gwf_name}.hds"), None),
        ("GWFBUDGET", os.path.join("..", f"{gwf_name}.cbb"), None, None),
        # ("maw-hopeland", os.path.join("..", f"{gwf_name}.maw.hopeland.bud"), None, None),
    ]
    fmi = flopy.mf6.ModflowGwtfmi(gwt, packagedata=flow_packagedata)

    # Instantiating MODFLOW 6 transport output control package
    oc = flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{model_name}.cbb",
        concentration_filerecord=f"{model_name}.ucn",
        concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                    ],
        saverecord=[("CONCENTRATION", "ALL"), 
                    # ("BUDGET", "ALL")
                    ],
        printrecord=[("CONCENTRATION", "LAST"), 
                        # ("BUDGET", "ALL")
                        ],
    )

    sim.write_simulation()
    prep_bins(gwt_ws, get_only=['mf6'])

    return sim

def build_model(ws = 'model', sim_name = '1dtest', spls = ['tracer'], sconc = None, wel_rec = None, init_comp = None):

    ##################### --- GWF model          --- #####################
    gwfname = 'gwf'
    sim_ws = os.path.join(ws, sim_name)
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name='mf6')

    if os.path.exists(sim_ws):
        shutil.rmtree(sim_ws)
    os.makedirs(sim_ws)

    # Instantiating MODFLOW 6 time discretization
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)

    # Instantiating MODFLOW 6 groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        model_nam_file=f"{gwfname}.nam",
    )

    # Instantiating MODFLOW 6 solver for flow model
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{gwfname}.ims",
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    # Instantiating MODFLOW 6 discretization package
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=np.ones((nlay, nrow, ncol), dtype=int),
        filename=f"{gwfname}.dis",
    )
    dis.set_all_data_external()

    # Instantiating MODFLOW 6 node-property flow package
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_saturation = True,
        icelltype=icelltype,
        k=k11,
        k33=k33,
        save_specific_discharge=True,
        filename=f"{gwfname}.npf",
    )
    npf.set_all_data_external()
    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{gwfname}.ic")

    # Instantiating MODFLOW 6 constant head package
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        pname="CHD",
        filename=f"{gwfname}.chd",
    )
    chd.set_all_data_external()

    if init_comp == None:
        auxiliary = 'tracer'
    else:
        auxiliary = [c for c in init_comp]

    if wel_rec == None:
        cin = 1.0e-3
        cin = concentration_liters_to_m3(cin)
        wel_rec = spd = [[(0,0,0), 0.259, cin]]

    wel = flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=wel_rec,
            # auxiliary = f'concentration',
            save_flows = True,
            auxiliary = auxiliary,
            pname = 'wel',
            filename=f"{gwfname}.wel"
        )
    wel.set_all_data_external()

    # Instantiating MODFLOW 6 output control package for flow model
    oc_gwf = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwfname}.hds",
        budget_filerecord=f"{gwfname}.cbb",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    
    ##################### --- GWT model          --- #####################
    for sp in spls:
        
        print(f'Setting model for component: {sp}')
        gwtname = sp
        
        # Instantiating MODFLOW 6 groundwater transport package
        gwt = flopy.mf6.MFModel(
            sim,
            model_type="gwt6",
            modelname=gwtname,
            model_nam_file=f"{gwtname}.nam"
        )

        # create iterative model solution and register the gwt model with it
        print('--- Building IMS package ---')
        imsgwt = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="NONE",
            inner_maximum=ninner,
            inner_dvclose=hclose,
            rcloserecord=rclose,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
            filename=f"{gwtname}.ims",
        )
        sim.register_ims_package(imsgwt, [gwt.name])

        print('--- Building DIS package ---')
        dis = gwf.dis

        # create grid object
        dis = flopy.mf6.ModflowGwtdis(
            gwt,
            length_units=length_units,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            idomain=np.ones((nlay, nrow, ncol), dtype=int),
            filename=f"{gwtname}.dis",
        )
        dis.set_all_data_external()

        if sconc == None:
            sconc = {}
            sconc[sp] = 0.
         
        ic = flopy.mf6.ModflowGwtic(gwt, strt=sconc[sp], filename=f"{gwtname}.ic")
        ic.set_all_data_external()

        # Instantiating MODFLOW 6 transport source-sink mixing package
        sourcerecarray = ['wel', 'aux', f'{sp}']
        ssm = flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 

            filename=f"{gwtname}.ssm"
        )
        ssm.set_all_data_external()
        # Instantiating MODFLOW 6 transport adv package
        print('--- Building ADV package ---')
        adv = flopy.mf6.ModflowGwtadv(
            gwt,
            scheme="tvd",
        )

        # Instantiating MODFLOW 6 transport dispersion package
        alpha_l = np.ones(shape=(nlay, nrow, ncol))*0.0067  # Longitudinal dispersivity ($m$)
        alpha_th = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse horizontal dispersivity ($m$)
        alpha_tv = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse vertical dispersivity ($m$)

        print('--- Building DSP package ---')
        dsp = flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alpha_l,
            ath1=alpha_th,
            atv = alpha_tv,
            # diffc = diffc,
            filename=f"{gwtname}.dsp",
        )
        dsp.set_all_data_external()

        # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
        print('--- Building MST package ---')

        sorption = None
        zero_order_decay = None
        first_order_decay = None

        # cnc = flopy.mf6.ModflowGwtcnc(
        #         gwt,
        #         # maxbound=maxbounds_cnc,
        #         stress_period_data=cnc_spd,
        #         save_flows=True,
        #         pname="cnc",
        #         filename=f"{model_name}.cnc",
        #         boundnames = True
        #         )
        # cnc.set_all_data_external()
        print('--- Building SRC package ---')
        # src_spd = src_array(q, init_comp, sp)
        # src = flopy.mf6.ModflowGwtsrc(
        #     gwt,
        #     stress_period_data = src_spd,
        #     save_flows = True,
        #     filename=f"{gwtname}.src"
        # )
        # src.set_all_data_external()

        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=prsity,
            first_order_decay=first_order_decay,
            filename=f"{gwtname}.mst",
        )
        mst.set_all_data_external()

        print('building FMI and OC ...')

        # Instantiating MODFLOW 6 transport output control package
        oc_gwt = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbb",
            concentration_filerecord=f"{gwtname}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                        ],
            saverecord=[("CONCENTRATION", "ALL"), 
                        # ("BUDGET", "ALL")
                        ],
            printrecord=[("CONCENTRATION", "LAST"), 
                            # ("BUDGET", "ALL")
                            ],
        )

        # Instantiating MODFLOW 6 flow-transport exchange mechanism
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwfname,
            exgmnameb=gwtname,
            filename=f"{gwtname}.gwfgwt",
        )

    sim.write_simulation()
    prep_bins(sim_ws, get_only=['mf6'])
    
    return sim

def plot_heads(sim, prefix = None):
    #create figures dir
    figures_dir = 'figures'

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    mf6_out_path = sim.simulation_data.mfpath.get_sim_path()
    model_name = list(sim.model_names)[0]
    gwf = sim.get_model(model_name)
    hdobj = flopy.utils.HeadFile(os.path.join(mf6_out_path, f'{model_name}.hds'), model=gwf)
    times = hdobj.get_times()
    heads = hdobj.get_alldata()

    fig, axs = plt.subplots(1, 1, figsize=(6.3, 3.2))
    ax = axs
    # mapview = flopy.plot.PlotMapView(model=gwf, ax=ax)
    mapview = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"Row": 0})

    if prefix != None:
        fname = f'{prefix}_{sim.name}_hds'
    else: fname = f'{sim.name}_hds'

    linecollection = mapview.plot_grid(ax = ax, alpha = 1, zorder=2, lw = 0.5)
    arr = mapview.plot_array(heads[-1,0,0,:])
    # # contours = mapview.contour_array(heads[-1,0,0,:], levels = np.arange(0,0.12, 0.01),colors="black", linewidths=0.5)
    # # ax.clabel(contours, colors="black", fontsize = 10)
    # # ax.set_ylim(-1.1,0)
    cbar = plt.colorbar(arr)
    cbar.ax.set_title('Head (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir,f'{fname}_map.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(6.3,3.2))
    ax = axs
    ax.plot(times, heads[:, 0, 0,-1])
    ax.set_ylabel('Head (m)')
    ax.set_xlabel('Time')
    fig.tight_layout()

    fig.savefig(os.path.join(figures_dir, f'{fname}_ts.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(heads[:, 0, 0,-1], times)

def plot_concentrations(sim, prefix = None):
    print(f'Saving figures for simulation: {sim.name}')
    #create figures dir
    figures_dir = 'figures'

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    mf6_out_path = sim.simulation_data.mfpath.get_sim_path()
    for model_name in list(sim.model_names[1:]):
        gwf = sim.get_model(model_name)

        if prefix != None:
            fname = f'{prefix}_{sim.name}_{model_name}'
        else: fname = f'{sim.name}_{model_name}'
        
        print(f'Saving figures for component: {model_name}')
        ucn = flopy.utils.HeadFile(os.path.join("model",sim.name,f"{model_name}.ucn"),text="concentration")
        results = ucn.get_alldata()
        # results = gwf.output.concentration().get_alldata()
        times = gwf.output.concentration().get_times()

        fig, axs = plt.subplots(1, 1, figsize=(6.3, 3.2))
        ax = axs
        # mapview = flopy.plot.PlotMapView(model=gwf, ax=ax)
        mapview = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"Row": 0})

        linecollection = mapview.plot_grid(ax = ax, alpha = 1, zorder=2, lw = 0.5)
        arr = mapview.plot_array(results[-1,0,0,:])
        # contours = mapview.contour_array(results[-1,0,0,:], levels = np.arange(0,0.12, 0.01),colors="black", linewidths=0.5)
        # ax.clabel(contours, colors="black", fontsize = 10)
        # ax.set_ylim(-1.1,0)
        cbar = plt.colorbar(arr)
        cbar.ax.set_title('Conc')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir,f'{fname}_conc_map.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6.3,3.2))
        ax = axs
        ax.plot(times, results[:, 0, 0,-1])
        ax.set_ylabel('Conc (mol/L)')
        ax.set_xlabel('Time')
        fig.tight_layout()

        fig.savefig(os.path.join(figures_dir,f'{fname}_ts.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    return

def build_mt3dms_model(sim_name,
    ws,
    dispersivity=dispersivity,
    mixelm=mixelm,
    silent=False,):

    sim_ws = os.path.join(ws, sim_name)
    if os.path.exists(sim_ws):
        shutil.rmtree(sim_ws)
    os.makedirs(sim_ws)
    modelname_mf = 'gwf'

    # Instantiate the MODFLOW model
    mf = flopy.modflow.Modflow(
        modelname=modelname_mf, model_ws=sim_ws, exe_name="mf2005"
    )

    # Instantiate discretization package
    # units: itmuni=4 (days), lenuni=2 (m)
    flopy.modflow.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        nstp=nstp,
        botm=botm,
        perlen=perlen,
        itmuni=4,
        lenuni=2,
    )
    q = 0.259
    welspd = [[0, 0, 0, q]]
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=welspd)

    # Instantiate basic package
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # Instantiate layer property flow package
    flopy.modflow.ModflowLpf(mf, hk=k11, laytyp=laytyp)

    # Instantiate solver package
    flopy.modflow.ModflowPcg(mf)

    # Instantiate link mass transport package (for writing linker file)
    flopy.modflow.ModflowLmt(mf)

    spd = {}
    # for kper in range(nper):
    for kstp in range(0,24+1):
        # print(kper, kstp)
        spd[(0, kstp)] = [
            "save head",
            "save budget",
            "print head",
            "print budget",
        ]
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
    # Transport
    modelname_mt = f"gwt"
    mt = flopy.mt3d.Mt3dms(
        modelname=modelname_mt,
        model_ws=sim_ws,
        exe_name="mt3dms",
        modflowmodel=mf,
        # ftlfree=True
    )

    c0 = 0.0
    icbund = np.ones((nlay, nrow, ncol), dtype=int)
    # icbund[0, 0, 0] = -1
    sconc = np.zeros((nlay, nrow, ncol), dtype=float)
    # sconc[0, 0, 0] = c0

    flopy.mt3d.Mt3dBtn(
        mt,
        # laycon=laytyp,
        icbund=icbund,
        prsity=prsity,
        sconc=sconc,
        dt0=dt0,
        nprs = -1
        # ifmtcn=1,
    )

    # Instatiate the advection package
    flopy.mt3d.Mt3dAdv(
        mt,
        mixelm=mixelm,
        dceps=dceps,
        nplane=nplane,
        npl=npl,
        nph=nph,
        npmin=npmin,
        npmax=npmax,
        nlsink=nlsink,
        npsink=npsink,
        percel=0.1,
    )

    # Instantiate the dispersion package
    flopy.mt3d.Mt3dDsp(mt,  al=dispersivity)

    rct = flopy.mt3d.Mt3dRct(
            mt,
            isothm=0, #no sorption
            ireact=0, #no reactions
            igetsc=0,
        )

    # Instantiate the source/sink mixing package
    cin = 1.0e-3
    cin = concentration_liters_to_m3(cin)
    ssm_data = {}
    ssm_data[0] = [(0, 0, 0, cin, 2)]
    flopy.mt3d.Mt3dSsm(mt, stress_period_data = ssm_data)

    # Instantiate the GCG solver in MT3DMS
    flopy.mt3d.Mt3dGcg(mt)

    mf.write_input()
    mt.write_input()
    fname = os.path.join(sim_ws, "MT3D001.UCN")
    if os.path.isfile(fname):
        os.remove(fname)
    prep_bins(sim_ws)

    return mf, mt

if __name__ == "__main__":
    sim_name = 'engesgaard1992'
    initsol_components, sconc_init = init_solution(init_file = 'initsol.dat')

    q = 0.259
    wel_rec = wel_array(q, sconc_init, aux = True)
    components, phreeqc_rm, sconc = initialize_phreeqcrm(sim_name)    

    
    # sim = build_model(ws = 'model', sim_name = sim_name, spls = components, 
    #                   sconc=sconc, wel_rec=wel_rec, init_comp=sconc_init)


    # run_mf6(sim)
    # plot_heads(sim)
    # plot_concentrations(sim)

    sim_name = 'engesgaard1992api'
    sim = build_model(ws = 'model', sim_name = sim_name, spls = components, 
                      sconc=sconc, wel_rec=wel_rec, init_comp=sconc_init)
    sim_ws = Path(f"model/{sim_name}/")
    dll = Path("bin/win/libmf6")
    print(components)
    results = mf6rtm_api_test(dll, sim_ws, components=components, phreeqc_rm=phreeqc_rm, reaction = True)
    # print(results)

    # plot_heads(sim, prefix = 'api')
    # plot_concentrations(sim, prefix = 'api')

    ##### Transport Benchmarks #####
    # sim_name = 'mt3dms'
    # ws = os.path.join('benchmark')
    # mf, mt = build_mt3dms_model(sim_name, ws = ws)
    # run_mt3dms(mt)

    # sim_name = 'mf6gwt'
    # ws = os.path.join('benchmark')
    # sim = build_model(ws = ws, sim_name = sim_name)
    # run_mf6(sim)

    # sim_name = 'mf6gwtapi'
    # ws = os.path.join('benchmark')
    # sim = build_model(ws = ws, sim_name = sim_name)
    # sim_ws = Path(f"benchmark/{sim_name}/")
    # results = api_test(dll, sim_ws)