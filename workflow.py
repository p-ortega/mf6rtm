# imports
import platform
from pathlib import Path
import os
import modflowapi
from modflowapi import Callbacks
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import math
import time
#sys.path.insert(0,os.path.join("dependencies"))
import flopy
# import pyemu
import itertools
# from pypestutils.pestutilslib import PestUtilsLib

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
top = 0.0  # Top of the model ($m$)
botm = -1.0  # Layer bottom elevations ($m$)
prsity = 0.32  # Porosity
perlen = 0.24  # Simulation time ($days$)
k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)

k33 = k11  # Vertical hydraulic conductivity ($m/d$)
laytyp = 1
tstep = 0.01
nstp = perlen/tstep #100.0
# dt0 = perlen / nstp
# Lx = (ncol - 1) * delr
v = 0.24

q = v * prsity
h1 = q * Lx
strt = np.zeros((nlay, nrow, ncol), dtype=float)
strt[0, 0, 0] = h1  # Starting head ($m$)

diffcdir = {'benzene':  9.392e-5, #m2/d
            'naphthalene': 6.471e-5 #m2/d,
            }

l = 1000.0  # Needed for plots
icelltype = 1  # Cell conversion type
ibound = np.ones((nlay, nrow, ncol), dtype=int)
ibound[0, 0, 0] = -1
ibound[0, 0, -1] = -1

mixelm = 0  # TVD
rhob = 0.25
sp2 = 0.0  # red, but not used in this problem
sconc = np.ones((nlay, nrow, ncol), dtype=float)*2
dmcoef = 0.0  # Molecular diffusion coefficient

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
        # [(0, 0, 0), h1], 
          [(0, 0, ncol - 1), 0.0]
          ]
c0 = 1.0
cnc_spd = [[(0, 0, 0), c0]]
conc = 2e-3 
q = 0.259
wel_rec = [[(0,0,0), q, conc]]

def api_test(dll, sim_ws):
    
    mf6 = modflowapi.ModflowApi(dll, working_directory = sim_ws)
    mf6.initialize()

    sim_start = datetime.now()
    print("...starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))
    # reset the node tracking containers

    # get the current sim time
    ctime = mf6.get_current_time()
    ctimes = [0.0]
    # get the ending sim time
    etime = mf6.get_end_time()
    # max number of solution iterations
    max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))
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
        mf6.prepare_solve(1)
        # the one-based stress period number
        stress_period = mf6.get_value(mf6.get_var_address("KPER", "TDIS"))[0]
        time_step = mf6.get_value(mf6.get_var_address("KSTP", "TDIS"))[0]

        # solve until converged
        while kiter < max_iter:
            # apply whatever change we want here
            val = mf6.get_value(mf6.get_var_address("x", 'SODIUM'))
            print(val)
            val += 1
            convg = mf6.solve(1)
            if convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                print("transport stress period,time step {0},{1} converged with {2} iters, took {3:10.5G} mins".format(stress_period, time_step, kiter,td))
                break
            kiter += 1

        if not convg:
            td = (datetime.now() - sol_start).total_seconds() / 60.0
            print("transport stress period,time step {0},{1} did not converged, {2} iters, took {3:10.5G} mins".format(
                stress_period, time_step, kiter, td))
            num_fails += 1
    mf6.finalize()


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

def run_model(sim, silent=False):
    success, buff = sim.run_simulation(silent=silent)
    if not success:
        print(buff)
    return success

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
    alpha_l = np.ones(shape=(nlay, nrow, ncol))*0.0067  # Longitudinal dispersivity ($m$)
    alpha_th = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse horizontal dispersivity ($m$)
    alpha_tv = np.ones(shape=(nlay, nrow, ncol))*1  # Transverse vertical dispersivity ($m$)

    # diffc =np.ones(shape=(nlay, nrow, ncol))* diffcdir[sp]# 82.94 m2/d = 9.6 cm2/s (Hilal et al2003) Diffusion coefficient ($m^2 d^{-1}$) 

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

def build_model(ws = 'model', sim_name = '1dtest', spls = ['chloride']):

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
    
    ##################### --- GWT model          --- #####################
    for sp in spls:
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

        
        ic = flopy.mf6.ModflowGwtic(gwt, strt=sconc, filename=f"{gwtname}.ic")

        # Instantiating MODFLOW 6 transport source-sink mixing package
        sourcerecarray = ['wel', 'aux', 'concentration']
        flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 

            filename=f"{gwtname}.ssm"
        )
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

        # diffc =np.ones(shape=(nlay, nrow, ncol))* diffcdir[sp]# 82.94 m2/d = 9.6 cm2/s (Hilal et al2003) Diffusion coefficient ($m^2 d^{-1}$) 

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

        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=prsity,
            first_order_decay=first_order_decay,
            filename=f"{gwtname}.mst",
        )
        mst.set_all_data_external()

        print('building FMI and OC ...')

        # Instantiating MODFLOW 6 transport output control package
        oc = flopy.mf6.ModflowGwtoc(
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
    prep_bins(ws, get_only=['mf6'])
    return sim

def plot_heads(sim):
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

    linecollection = mapview.plot_grid(ax = ax, alpha = 1, zorder=2, lw = 0.5)
    arr = mapview.plot_array(heads[-1,0,0,:])
    contours = mapview.contour_array(heads[-1,0,0,:], levels = np.arange(0,0.12, 0.01),colors="black", linewidths=0.5)
    ax.clabel(contours, colors="black", fontsize = 10)
    ax.set_ylim(-1.1,0)
    cbar = plt.colorbar(arr)
    cbar.ax.set_title('Head (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir,f'{sim.name}_hds_map.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(6.3,3.2))
    ax = axs
    ax.plot(times, heads[:, 0, 0,-1])
    ax.set_ylabel('Head (m)')
    ax.set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir,f'{sim.name}_hds_ts.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(heads[:, 0, 0,-1], times)

def plot_concentrations(sim):
    #create figures dir
    figures_dir = 'figures'

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    mf6_out_path = sim.simulation_data.mfpath.get_sim_path()
    for model_name in list(sim.model_names[1:]):
        gwf = sim.get_model(model_name)
        print(list(sim.model_names))
        results = gwf.output.concentration().get_alldata()
        times = gwf.output.concentration().get_times()

        fig, axs = plt.subplots(1, 1, figsize=(6.3, 3.2))
        ax = axs
        # mapview = flopy.plot.PlotMapView(model=gwf, ax=ax)
        mapview = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"Row": 0})

        linecollection = mapview.plot_grid(ax = ax, alpha = 1, zorder=2, lw = 0.5)
        arr = mapview.plot_array(results[-1,0,0,:])
        contours = mapview.contour_array(results[-1,0,0,:], levels = np.arange(0,0.12, 0.01),colors="black", linewidths=0.5)
        ax.clabel(contours, colors="black", fontsize = 10)
        ax.set_ylim(-1.1,0)
        cbar = plt.colorbar(arr)
        cbar.ax.set_title('Conc')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir,f'{sim.name}_{model_name}_conc_map.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6.3,3.2))
        ax = axs
        ax.plot(times, results[:, 0, 0,-1])
        ax.set_ylabel('Conc (mol/L)')
        ax.set_xlabel('Time')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir,f'{sim.name}_{model_name}_ts.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return

if __name__ == "__main__":

    ##################### --- Setup base GWF model          --- #####################
    sim = build_model(ws = 'model', sim_name = 'engesgaard1992', spls = ['sodium'])
    # run_model(sim)
    sim_ws = Path("model/engesgaard1992/")
    dll = Path("bin/win/libmf6")
    api_test(dll, sim_ws)
    # simgwt = build_gwt_model(sim, gwf_name = 'gwf_1dtest', sp = 'chloride')
    # run_model(simgwt)
    plot_heads(sim)
    plot_concentrations(sim)

    # sim_ws = Path("model/1dtest/chloride/")
    # dll = Path("bin/win/libmf6")
    # modflowapi.run_simulation(dll, sim_ws, callback_function, verbose=True)
