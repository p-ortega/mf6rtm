import os
import shutil 
import pytest
import platform
import flopy
import pandas as pd
import numpy as np
from mf6rtm import utils, mup3d
from pathlib import Path

cwd = os.path.abspath(os.path.dirname(__file__))
dataws = os.path.join(cwd, "data")
databasews = os.path.join(cwd, "database")

bin_path = "bin"
exe_ext = ""
env_path = Path(os.environ.get("CONDA_PREFIX", None))
assert env_path is not None, (
    "autotest script must be run from the mf6rt environment"
)

if "linux" in platform.platform().lower():
    lib_ext = ".so"
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_ext = ".dylib"
else:
    bin_path = "Scripts"
    lib_ext = ".dll"
    exe_ext = ".exe"

lib_name = env_path / f"{bin_path}/libmf6{lib_ext}"
src_path = env_path / f"{bin_path}"

def build_mf6_1d_injection_model(mup3d, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                 top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                 strt, rclose, relax, nouter, ninner):

    #####################        GWF model           #####################
    gwfname = 'gwf'
    sim_ws = mup3d.wd
    sim = flopy.mf6.MFSimulation(sim_name=mup3d.name, sim_ws=sim_ws, exe_name='mf6')

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
        complexity="complex",
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
    # sto = flopy.mf6.ModflowGwfsto(gwf, ss=1e-6, sy=0.25)

    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{gwfname}.ic")
    
    wel = flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=wel_spd,
            save_flows = True,
            auxiliary = mup3d.components,
            pname = 'wel',
            filename=f"{gwfname}.wel"
        )
    wel.set_all_data_external()

    # Instantiating MODFLOW 6 constant head package
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        # auxiliary=mup3d.components,
        save_flows=False,
        pname="CHD",
        filename=f"{gwfname}.chd",
    )
    chd.set_all_data_external()

    # Instantiating MODFLOW 6 output control package for flow model
    oc_gwf = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwfname}.hds",
        budget_filerecord=f"{gwfname}.cbb",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    
    #####################           GWT model          #####################
    for c in mup3d.components:
        print(f'Setting model for component: {c}')
        gwtname = c
        
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

         
        ic = flopy.mf6.ModflowGwtic(gwt, strt=mup3d.sconc[c], filename=f"{gwtname}.ic")
        ic.set_all_data_external()
        
        # Instantiating MODFLOW 6 transport source-sink mixing package
        sourcerecarray = ['wel', 'aux', f'{c}']
        # sourcerecarray = [()]
        ssm = flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 
            save_flows=True,
            print_flows=True,

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
        alpha_l = np.ones(shape=(nlay, nrow, ncol))*dispersivity  # Longitudinal dispersivity ($m$)
        ath1 = np.ones(shape=(nlay, nrow, ncol))*dispersivity*0.1 # Transverse horizontal dispersivity ($m$)
        atv = np.ones(shape=(nlay, nrow, ncol))*dispersivity*0.1   # Transverse vertical dispersivity ($m$)

        print('--- Building DSP package ---')
        dsp = flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alpha_l,
            ath1=ath1,
            atv = atv,
            # diffc = diffc,
            filename=f"{gwtname}.dsp",
        )
        dsp.set_all_data_external()

        # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
        print('--- Building MST package ---')

        first_order_decay = None

        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=prsity,
            first_order_decay=first_order_decay,
            filename=f"{gwtname}.mst",
        )
        mst.set_all_data_external()

        print('--- Building OC package ---')

        # Instantiating MODFLOW 6 transport output control package
        oc_gwt = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbb",
            concentration_filerecord=f"{gwtname}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                        ],
            saverecord=[("CONCENTRATION", "ALL"), 
                        ("BUDGET", "ALL")
                        ],
            printrecord=[("CONCENTRATION", "ALL"), 
                            ("BUDGET", "ALL")
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
    # utils.prep_bins(sim_ws, src_path=src_path, get_only=['mf6', 'libmf6'], add_platform=False)

    return sim

def build_mf6_1d_transport_first_model(sim_ws, name, nper, tdis_rc, length_units, time_units,
                                       nlay, nrow, ncol, delr, delc, top, botm, wel_spd, chdspd,
                                       prsity, k11, k33, dispersivity, icelltype, strt,
                                       hclose, rclose, relax, nouter, ninner):
    """Build a transport-first sim for Mup3d.from_mf6: GWF + one tracer GWT.

    Same physics as ``build_mf6_1d_injection_model`` but the WEL carries a single
    ``tracer`` auxiliary and there is exactly one conservative tracer GWT (SSM
    sourcing that aux). ``from_mf6`` clones this GWT into one reactive GWT per
    PHREEQC component at ``write_simulation()``.
    """
    gwfname = 'gwf'
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_ws, exe_name='mf6')
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)

    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True,
                               model_nam_file=f"{gwfname}.nam")
    imsgwf = flopy.mf6.ModflowIms(
        sim, complexity="complex", print_option="SUMMARY", outer_dvclose=hclose,
        outer_maximum=nouter, under_relaxation="NONE", inner_maximum=ninner,
        inner_dvclose=hclose, rcloserecord=rclose, linear_acceleration="CG",
        scaling_method="NONE", reordering_method="NONE", relaxation_factor=relax,
        filename=f"{gwfname}.ims",
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf, length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
        delc=delc, top=top, botm=botm, idomain=np.ones((nlay, nrow, ncol), dtype=int),
        filename=f"{gwfname}.dis",
    )
    flopy.mf6.ModflowGwfnpf(
        gwf, save_flows=True, save_saturation=True, icelltype=icelltype, k=k11, k33=k33,
        save_specific_discharge=True, filename=f"{gwfname}.npf",
    )
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{gwfname}.ic")
    # single 'tracer' aux; from_mf6 strips it and writes one aux per component
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=wel_spd, save_flows=True, auxiliary=['tracer'],
        pname='wel', filename=f"{gwfname}.wel",
    )
    flopy.mf6.ModflowGwfchd(
        gwf, maxbound=len(chdspd), stress_period_data=chdspd, save_flows=False,
        pname="CHD", filename=f"{gwfname}.chd",
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{gwfname}.hds", budget_filerecord=f"{gwfname}.cbb",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # one conservative tracer GWT used as the from_mf6 template
    gwtname = 'tracer'
    gwt = flopy.mf6.MFModel(sim, model_type="gwt6", modelname=gwtname,
                            model_nam_file=f"{gwtname}.nam")
    imsgwt = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", outer_dvclose=hclose, outer_maximum=nouter,
        under_relaxation="NONE", inner_maximum=ninner, inner_dvclose=hclose,
        rcloserecord=rclose, linear_acceleration="BICGSTAB", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=relax, filename=f"{gwtname}.ims",
    )
    sim.register_ims_package(imsgwt, [gwt.name])
    flopy.mf6.ModflowGwtdis(
        gwt, length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
        delc=delc, top=top, botm=botm, idomain=np.ones((nlay, nrow, ncol), dtype=int),
        filename=f"{gwtname}.dis",
    )
    flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename=f"{gwtname}.ic")
    flopy.mf6.ModflowGwtssm(gwt, sources=['wel', 'aux', 'tracer'], save_flows=True,
                            filename=f"{gwtname}.ssm")
    flopy.mf6.ModflowGwtadv(gwt, scheme="tvd")
    alpha_l = np.ones((nlay, nrow, ncol)) * dispersivity
    ath1 = np.ones((nlay, nrow, ncol)) * dispersivity * 0.1
    atv = np.ones((nlay, nrow, ncol)) * dispersivity * 0.1
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=alpha_l, ath1=ath1, atv=atv,
                            filename=f"{gwtname}.dsp")
    flopy.mf6.ModflowGwtmst(gwt, porosity=prsity, first_order_decay=None,
                            filename=f"{gwtname}.mst")
    flopy.mf6.ModflowGwtoc(
        gwt, budget_filerecord=f"{gwtname}.cbb", concentration_filerecord=f"{gwtname}.ucn",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea=gwfname,
                            exgmnameb=gwtname, filename=f"{gwtname}.gwfgwt")
    return sim

def build_mf6_2d_model(mup3d, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                 top, botm, chdspd, prsity, k11, k33, dispersivity, disp_tr_vert,icelltype, hclose,
                                 strt, rclose, relax, nouter, ninner):

    #####################        GWF model           #####################
    gwfname = 'gwf'
    sim_ws = mup3d.wd
    sim = flopy.mf6.MFSimulation(sim_name=mup3d.name, sim_ws=sim_ws, exe_name='mf6')

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
        complexity="complex",
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
    # sto = flopy.mf6.ModflowGwfsto(gwf, ss=1e-6, sy=0.25)

    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{gwfname}.ic")

    # Instantiating MODFLOW 6 constant head package
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        auxiliary=mup3d.components,
        save_flows=False,
        pname="CHD",
        filename=f"{gwfname}.chd",
    )
    chd.set_all_data_external()

    # Instantiating MODFLOW 6 output control package for flow model
    oc_gwf = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwfname}.hds",
        budget_filerecord=f"{gwfname}.cbb",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    
    #####################           GWT model          #####################
    for c in mup3d.components:
        print(f'Setting model for component: {c}')
        gwtname = c
        
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

         
        ic = flopy.mf6.ModflowGwtic(gwt, strt=mup3d.sconc[c], filename=f"{gwtname}.ic")
        ic.set_all_data_external()

        # cncspd = {0: [[(0, 0, col), conc] for col, conc in zip(range(ncol), model.sconc[c][0,0,:])]}
        cncspd = {0: [[(ly, 0, 0), mup3d.sconc[c][ly,0,0]] for ly in range(3,nlay)]}

        # print(cncspd)
        cnc = flopy.mf6.ModflowGwtcnc(gwt,
                                        # maxbound=len(cncspd),
                                        stress_period_data=cncspd,
                                        save_flows=True,
                                        print_flows = True,
                                        pname="CNC",
                                        filename=f"{gwtname}.cnc",
                                        )
        cnc.set_all_data_external()
        # Instantiating MODFLOW 6 transport source-sink mixing package
        sourcerecarray = ['chd', 'aux', f'{c}']
        # sourcerecarray = [()]
        ssm = flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 
            save_flows=True,
            print_flows=True,

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
        alpha_l = np.ones(shape=(nlay, nrow, ncol))*dispersivity  # Longitudinal dispersivity ($m$)
        ath1 = np.ones(shape=(nlay, nrow, ncol))*dispersivity  # Transverse horizontal dispersivity ($m$)
        atv = np.ones(shape=(nlay, nrow, ncol))*disp_tr_vert  # Transverse vertical dispersivity ($m$)

        print('--- Building DSP package ---')
        dsp = flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alpha_l,
            ath1=ath1,
            atv = atv,
            # diffc = diffc,
            filename=f"{gwtname}.dsp",
        )
        dsp.set_all_data_external()

        # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
        print('--- Building MST package ---')

        first_order_decay = None

        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=prsity,
            first_order_decay=first_order_decay,
            filename=f"{gwtname}.mst",
        )
        mst.set_all_data_external()

        print('--- Building OC package ---')

        # Instantiating MODFLOW 6 transport output control package
        oc_gwt = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbb",
            concentration_filerecord=f"{gwtname}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                        ],
            saverecord=[("CONCENTRATION", "ALL"), 
                        ("BUDGET", "ALL")
                        ],
            printrecord=[("CONCENTRATION", "ALL"), 
                            ("BUDGET", "ALL")
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
    # utils.prep_bins(sim_ws, src_path=src_path, get_only=['mf6', 'libmf6'], add_platform=False)
    
    return sim

def test01(request, prefix = 'test01'):

    '''Test 1: Simple 1D injection test with equilibrium phases'''	
    ### Model params and setup
    length_units = "meters"
    time_units = "days"

    nper = 1  # Number of periods
    nlay = 1  # Number of layers
    Lx = 0.5 #m
    ncol = 50 # Number of columns
    nrow = 1  # Number of rows
    delr = Lx/ncol #10.0  # Column width ($m$)
    delc = 1.0  # Row width ($m$)
    top = 0.  # Top of the model ($m$)
    botm = -1.0  # Layer bottom elevations ($m$)
    prsity = 0.32  # Porosity
    k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)

    tstep = 0.01  # Time step ($days$)
    perlen = 0.24  # Simulation time ($days$)
    nstp = perlen/tstep #100.0
    dt0 = perlen / nstp

    chdspd = [[(0, 0, ncol - 1), 1.]]  # Constant head boundary $m$
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    strt[0, 0, :] = 1  # Starting head ($m$)

    tdis_rc = []
    tdis_rc.append((perlen, nstp, 1.0))

    icelltype = 1  # Cell conversion type
    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    ibound[0, 0, -1] = -1

    q=0.259 #m3/d

    wel_spd = [[(0,0,0), q]]

    #transport
    dispersivity = 0.0067 # Longitudinal dispersivity ($m$)

    # Set solver parameter values (and related)
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    solutionsdf = pd.read_csv(os.path.join(dataws,f"{prefix}_solutions.csv"), comment = '#',  index_col = 0)
    solutions = utils.solution_df_to_dict(solutionsdf)
    #get postfix file
    equilibrium_phases = pd.read_csv(os.path.join(dataws,f"{prefix}_equilibrium_phases.csv"))
    equilibrium_phases = utils.parse_equilibriums_dataframe(equilibrium_phases)

    sol_ic = 1
    #add solutions to clss
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)
    #create equilibrium phases class
    equilibrium_phases = mup3d.EquilibriumPhases(equilibrium_phases)
    equilibrium_phases.set_ic(1)

    #create model class
    model = mup3d.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    modelwd = os.path.join(cwd, f'{prefix}')
    model.set_wd(os.path.join(modelwd))
                 
    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    #set database
    database = os.path.join(databasews, f'pht3d_datab.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_phases(equilibrium_phases)

    model.initialize()

    wellchem = mup3d.ChemStress('wel')
    sol_spd = [2]
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)

    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    run_test(prefix, model, request=request, libname=lib_name, treshold = 0.1)

    return 

def test02(request, prefix = 'test02'):
    # General
    length_units = "meters"
    time_units = "days"

    # Model discretization
    nlay = 1  # Number of layers
    Lx = 0.4 #m
    ncol = 80 # Number of columns
    nrow = 1  # Number of rows
    delr = Lx/ncol #10.0  # Column width ($m$)
    delc = 1.0  # Row width ($m$)
    top = 1.  # Top of the model ($m$)
    # botm = 0.0  # Layer bottom elevations ($m$)
    zbotm = 0.
    botm = np.linspace(top, zbotm, nlay + 1)[1:]

    #tdis
    nper = 1  # Number of periods
    tstep = 1  # Time step ($days$)
    perlen = 24  # Simulation time ($days$)
    nstp = perlen/tstep #100.0
    dt0 = perlen / nstp
    tdis_rc = []
    tdis_rc.append((perlen, nstp, 1.0))

    #injection
    q = 0.007 #injection rate m3/d
    wel_spd = [[(0,0,0), q]]

    #hydraulic properties
    prsity = 0.35  # Porosity
    k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)
    strt = np.ones((nlay, nrow, ncol), dtype=float)*1

    # two chd one for tailings and conc and other one for hds 
    r_hd = 1
    strt = np.ones((nlay, nrow, ncol), dtype=float)

    chdspd = [[(i, 0, ncol-1), r_hd] for i in range(nlay)] # Constant head boundary $m$

    #transport
    dispersivity = 0.005 # Longitudinal dispersivity ($m$)
    disp_tr_vert = dispersivity*0.1 # Transverse vertical dispersivity ($m$)

    icelltype = 1  # Cell conversion type

    # Set solver parameter values (and related)
    nouter, ninner = 300, 600
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    solutionsdf = pd.read_csv(os.path.join(dataws,f"{prefix}_solutions.csv"), comment = '#',  index_col = 0)

    # solutions = utils.solution_csv_to_dict(os.path.join(dataws,f"{prefix}_solutions.csv"))
    solutions = utils.solution_df_to_dict(solutionsdf)

    # get equilibrium phases file
    equilibrium_phases = pd.read_csv(os.path.join(dataws,f"{prefix}_equilibrium_phases.csv"))
    equilibrium_phases["conc_mol_lb"] = [utils.concentration_volbulk_to_volwater(i, prsity)
                                         for i in equilibrium_phases["conc_mol_l"].values]
    equilibrium_phases = utils.parse_equilibriums_dataframe(equilibrium_phases)

    #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)

    #add solutions to clss
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    #create equilibrium phases class
    equilibrium_phases = mup3d.EquilibriumPhases(equilibrium_phases)
    eqp_ic = 1
    # eqp_ic[3:,:,0]= -1 #boundary condation in layer 0 of no eq phases
    equilibrium_phases.set_ic(eqp_ic)

    #create model class
    model = mup3d.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    modelwd = os.path.join(cwd, f'{prefix}')
    model.set_wd(os.path.join(modelwd))

    #set database
    database = os.path.join(databasews, f'pht3d_datab_walter1994.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_phases(equilibrium_phases)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)
    model.set_config(
        reactive={
            "externalio": True
        }
    )
    model.initialize()

    wellchem = mup3d.ChemStress('wel')
    sol_spd = [2]

    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)


    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    run_test(prefix, model, request=request, libname=lib_name, treshold = 0.1)

def test03(request, prefix = 'test03'):
    length_units = "meters"
    time_units = "days"

    # Model discretization
    nlay = 10  # Number of layers
    Lx = 100 #m
    ncol = 25 # Number of columns
    nrow = 1  # Number of rows
    delr = Lx/ncol #10.0  # Column width ($m$)
    delc = 1.0  # Row width ($m$)
    top = 10.  # Top of the model ($m$)
    # botm = 0.0  # Layer bottom elevations ($m$)
    zbotm = 0.
    botm = np.linspace(top, zbotm, nlay + 1)[1:]

    #tdis
    nper = 1  # Number of periods
    tstep = 20  # Time step ($days$)
    perlen = 2000  # Simulation time ($days$)
    nstp = perlen/tstep #100.0
    dt0 = perlen / nstp
    tdis_rc = []
    tdis_rc.append((perlen, nstp, 1.0))

    #hydraulic properties
    prsity = 0.35  # Porosity
    k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)
    strt = np.ones((nlay, nrow, ncol), dtype=float)*10
    # two chd one for tailings and conc and other one for hds 

    l_hd = 12
    r_hd = 10
    strt = np.ones((nlay, nrow, ncol), dtype=float)*10
    strt[:, 0, 0] = l_hd  # Starting head ($m$)

    chdspd = [[(i, 0, 0), l_hd] for i in range(nlay)] # Constant head boundary $m$
    chdspd.extend([(i, 0, ncol - 1), r_hd] for i in range(nlay))


    #transport
    dispersivity = 2.5 # Longitudinal dispersivity ($m$)
    disp_tr_vert = 0.025 # Transverse vertical dispersivity ($m$)

    icelltype = 0  # Cell conversion type

    # Set solver parameter values (and related)
    nouter, ninner = 300, 600
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    solutionsdf = pd.read_csv(os.path.join(dataws,f"{prefix}_solutions.csv"), comment = '#',  index_col = 0)

    # solutions = utils.solution_csv_to_dict(os.path.join(dataws,f"{prefix}_solutions.csv"))
    solutions = utils.solution_df_to_dict(solutionsdf)
    # get equilibrium phases file
    equilibrium_phases = pd.read_csv(os.path.join(dataws,f"{prefix}_equilibrium_phases.csv"))
    equilibrium_phases["conc_mol_lb"] = [utils.concentration_volbulk_to_volwater(i, prsity)
                                         for i in equilibrium_phases["conc_mol_l"].values]
    equilibrium_phases = utils.parse_equilibriums_dataframe(equilibrium_phases)

    #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=int)

    #add solutions to clss
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    #create equilibrium phases class
    equilibrium_phases = mup3d.EquilibriumPhases(equilibrium_phases)
    eqp_ic = np.ones((nlay, nrow, ncol), dtype=int)*1
    eqp_ic[3:,:,0]= -1 #boundary condation in layer 0 of no eq phases
    equilibrium_phases.set_ic(eqp_ic)

    #create model class
    model = mup3d.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    modelwd = os.path.join(cwd, f'{prefix}')
    model.set_wd(os.path.join(modelwd))

    #set database
    database = os.path.join(databasews, f'pht3d_datab_walter1994.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_equilibrium_phases(equilibrium_phases)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)
    model.initialize()

    wellchem = mup3d.ChemStress('chdtail')
    sol_spd = [2]
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)

    for i in range(len(chdspd)):
        if i<3:
            chdspd[i].extend(model.chdtail.data[0])
        else:
            chdspd[i].extend(np.zeros_like(model.chdtail.data[0]))

    mf6sim = build_mf6_2d_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                 top, botm, chdspd, prsity, k11, k33, dispersivity, disp_tr_vert,icelltype, hclose,
                                 strt, rclose, relax, nouter, ninner)
    
    run_test(prefix, model, request=request, libname=lib_name, treshold = 0.1)


def test04(request, prefix = 'test04'):
    '''Test 4: Test 1: Simple 1D injection test with cation exchange from phreeqc'''
    # General
    length_units = "meters"
    time_units = "days"

    # Model discretization
    nlay = 1  # Number of layers
    Lx = 0.08 #m
    ncol = 40 # Number of columns
    nrow = 1  # Number of rows
    delr = Lx/ncol #10.0  # Column width ($m$)
    delc = 1.0  # Row width ($m$)
    top = 1.  # Top of the model ($m$)
    # botm = 0.0  # Layer bottom elevations ($m$)
    zbotm = 0.
    botm = np.linspace(top, zbotm, nlay + 1)[1:]

    #tdis
    nper = 1  # Number of periods
    tstep = 0.002  # Time step ($days$)
    perlen = 0.24  # Simulation time ($days$)
    nstp = perlen/tstep #100.0
    dt0 = perlen / nstp
    tdis_rc = []
    tdis_rc.append((perlen, nstp, 1.0))

    #injection
    q = 1 #injection rate m3/d
    wel_spd = [[(0,0,0), q]]

    #hydraulic properties
    prsity = 1 # Porosity
    k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)
    strt = np.ones((nlay, nrow, ncol), dtype=float)*1

    # two chd one for tailings and conc and other one for hds 
    r_hd = 1
    strt = np.ones((nlay, nrow, ncol), dtype=float)
    chdspd = [[(i, 0, ncol-1), r_hd] for i in range(nlay)] # Constant head boundary $m$
    #transport
    dispersivity = 0.002 # Longitudinal dispersivity ($m$)
    disp_tr_vert = dispersivity*0.1 # Transverse vertical dispersivity ($m$)
    icelltype = 1  # Cell conversion type

    # Set solver parameter values (and related)
    nouter, ninner = 300, 600
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    solutionsdf = pd.read_csv(os.path.join(dataws,f"{prefix}_solutions.csv"), comment = '#',  index_col = 0)
    # solutions = utils.solution_csv_to_dict(os.path.join(dataws,f"{prefix}_solutions.csv"))
    solutions = utils.solution_df_to_dict(solutionsdf)
    #get postfix file
    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')

    #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)
    #add solutions to clss
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    excdf = pd.read_csv(os.path.join(dataws,f"{prefix}_exchange.csv"), comment = '#',  index_col = 0)
    exchange_dict = {0:excdf.T.to_dict(index='comp')}
    exchanger = mup3d.ExchangePhases(exchange_dict)
    exchanger.set_equilibrate_solutions([1])
    exchanger.set_ic(np.ones((nlay, nrow, ncol), dtype=float))

    #create model class
    model = mup3d.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    modelwd = os.path.join(cwd, f'{prefix}')
    model.set_wd(os.path.join(modelwd))

    #set database
    database = os.path.join(databasews, f'pht3d_datab.dat')
    model.set_database(database)
    model.set_exchange_phases(exchanger)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    model.initialize()

    wellchem = mup3d.ChemStress('wel')
    sol_spd = [2]
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)

    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    
    run_test(prefix, model, request=request, libname=lib_name, treshold = 0.02)



def test05(request, prefix = 'test05'):
    '''Test 5: oxidation with pyrite 1D test
    This tests equilibrum phases, scm, kinetics and exchange    
    '''
    # General
    length_units = "meters"
    time_units = "days"

    # Model discretization
    nlay = 1  # Number of layers
    Lx = 0.053 #m
    ncol = 16 # Number of columns
    nrow = 1  # Number of rows
    delr = Lx/ncol #10.0  # Column width ($m$)
    delc = 1 # Row width ($m$)
    top = 2.87433E-03  # Top of the model ($m$)
    # botm = 0.0  # Layer bottom elevations ($m$)
    zbotm = 0.
    botm = np.linspace(top, zbotm, nlay + 1)[1:]

    #tdis
    nper = 2  # Number of periods
    nstp = [64, 100]  # Number of time steps
    # nstp = [i*10 for i in nstp]
    perlen = [ 0.9333, 1.45833]  # Simulation time ($days$)#100.0
    # dt0 = perlen / nstp
    tsmult = [1.0, 1.0]  # Time step multiplier
    tdis_rc = [(kper, kstep, ts) for kper, kstep, ts in zip(perlen, nstp, tsmult)]

    #injection
    q = 2.4e-4 #injection rate m3/d
    wel_spd = {i: [[(0,0,0), q]] for i in range(0, len(perlen))}


    #hydraulic properties
    prsity = 0.376 # Porosity
    k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)
    strt = np.ones((nlay, nrow, ncol), dtype=float)*1
    # two chd one for tailings and conc and other one for hds 

    # two chd one for tailings and conc and other one for hds 
    r_hd = 1
    strt = np.ones((nlay, nrow, ncol), dtype=float)

    chdspd = [[(i, 0, ncol-1), r_hd] for i in range(nlay)] # Constant head boundary $m$


    #transport
    dispersivity = 0.00537 #7.5e-5 Longitudinal dispersivity ($m$)

    icelltype = 1  # Cell conversion type

    # Set solver parameter values (and related)
    nouter, ninner = 300, 600
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    solutionsdf = pd.read_csv(os.path.join(dataws,f"{prefix}_solutions.csv"), comment = '#',  index_col = 0)

    solutions = utils.solution_df_to_dict(solutionsdf)
    solutions
    # #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)
    # sol_ic = 1
    #add solutions to clss
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    #exchange
    excdf = pd.read_csv(os.path.join(dataws,f"{prefix}_exchange.csv"), comment = '#',  index_col = 0)
    excdf = pd.read_csv(os.path.join(dataws,f"{prefix}_exchange.csv"), comment = '#',  index_col = 0)
    # exchangerdic = utils.solution_df_to_dict(excdf)
    excdf.columns=[0,1,2,3]
    exchanger_dict = excdf.to_dict()
    for k, subdict in exchanger_dict.items():
        for key in subdict:
            subdict[key] = {'m0': subdict[key]}

    exchanger = mup3d.ExchangePhases(exchanger_dict)
    exchanger_ic = np.ones((nlay, nrow, ncol), dtype=float)
    exchanger_ic[0,0,:4] = 1
    exchanger_ic[0,0,4:8] = 2
    exchanger_ic[0,0,8:12] = 3
    exchanger_ic[0,0,12:] = 4


    exchanger.set_ic(exchanger_ic)
    eq_solutions = [1,1,1,1]
    exchanger.set_equilibrate_solutions(eq_solutions)

    #kinetics
    df = pd.read_csv(os.path.join(dataws,f"{prefix}_kinetic_phases.csv"))
    kin_phases=utils.parse_kinetics_dataframe(df)
    orgsed_form = 'Orgc_sed -1.0 C 1.0' 
    kin_phases[1]['Orgc_sed']['formula'] = orgsed_form
    kinetics = mup3d.KineticPhases(kin_phases)
    kinetics.set_ic(1)

    #equilibrium phases
    df = pd.read_csv(os.path.join(dataws,f"{prefix}_equilibrium_phases.csv"))
    equ_phases = utils.parse_equilibriums_dataframe(df)

    equilibriums = mup3d.EquilibriumPhases(equ_phases)
    equilibriums.set_ic(1)

    #surfaces
    surfdic = utils.surfaces_csv_to_dict(os.path.join(dataws,f"{prefix}_surfaces.csv"))
    surfaces = mup3d.Surfaces(surfdic)
    surfaces.set_ic(1)
    # surfaces.set_options(['no_edl'])

    #create model class
    model = mup3d.Mup3d(prefix,solution, nlay, nrow, ncol)

    #set model workspace
    modelwd = os.path.join(cwd, f'{prefix}')
    model.set_wd(modelwd)

    # #set database
    database = os.path.join(databasews, f'ex5.dat')
    model.set_database(database)


    model.set_initial_temp([7., 7., 7.])
    # #get postfix file
    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)
    model.set_phases(surfaces)

    model.initialize()

    wellchem = mup3d.ChemStress('wel')
    sol_spd = [2,3]
    sol_spd
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)


    for key in wel_spd.keys():
        for i in range(len(wel_spd[key])):
            wel_spd[key][i].extend(model.wel.data[key])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                        top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                        strt, rclose, relax, nouter, ninner)
    
    run_test(prefix, model, request=request, test_cli=True, libname=lib_name, treshold = 0.02)

def test05_from_mf6(request, prefix = 'test05'):
    '''Test 5 via the transport-first from_mf6 workflow.

    Same pyrite 1D oxidation column as test05 (equilibrium phases, SCM, kinetics,
    exchange), but built by wrapping an existing flopy sim (GWF + one conservative
    tracer GWT) with Mup3d.from_mf6 and switching the injected solution per stress
    period through ChemStress('wel', type='aux').set_spd({0:[2], 1:[3]}). The output
    must match the same benchmark as test05 (benchmark/test05_benchmark.csv).
    '''
    # General
    length_units = "meters"
    time_units = "days"

    # Model discretization (identical to test05)
    nlay = 1
    Lx = 0.053
    ncol = 16
    nrow = 1
    delr = Lx/ncol
    delc = 1
    top = 2.87433E-03
    zbotm = 0.
    botm = np.linspace(top, zbotm, nlay + 1)[1:]

    # tdis
    nper = 2
    nstp = [64, 100]
    perlen = [0.9333, 1.45833]
    tsmult = [1.0, 1.0]
    tdis_rc = [(kper, kstep, ts) for kper, kstep, ts in zip(perlen, nstp, tsmult)]

    # injection — WEL carries one 'tracer' aux (placeholder value); from_mf6
    # replaces it with one aux column per component using the ChemStress mapping.
    q = 2.4e-4
    wel_spd = {i: [[(0, 0, 0), q, 1.0]] for i in range(0, len(perlen))}

    # hydraulic properties
    prsity = 0.376
    k11 = 1.0
    k33 = k11
    r_hd = 1
    strt = np.ones((nlay, nrow, ncol), dtype=float)
    chdspd = [[(i, 0, ncol-1), r_hd] for i in range(nlay)]

    # transport
    dispersivity = 0.00537
    icelltype = 1
    nouter, ninner = 300, 600
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    # chemistry (identical to test05)
    solutionsdf = pd.read_csv(os.path.join(dataws, f"{prefix}_solutions.csv"), comment='#', index_col=0)
    solutions = utils.solution_df_to_dict(solutionsdf)
    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)
    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)

    excdf = pd.read_csv(os.path.join(dataws, f"{prefix}_exchange.csv"), comment='#', index_col=0)
    excdf.columns = [0, 1, 2, 3]
    exchanger_dict = excdf.to_dict()
    for k, subdict in exchanger_dict.items():
        for key in subdict:
            subdict[key] = {'m0': subdict[key]}
    exchanger = mup3d.ExchangePhases(exchanger_dict)
    exchanger_ic = np.ones((nlay, nrow, ncol), dtype=float)
    exchanger_ic[0, 0, :4] = 1
    exchanger_ic[0, 0, 4:8] = 2
    exchanger_ic[0, 0, 8:12] = 3
    exchanger_ic[0, 0, 12:] = 4
    exchanger.set_ic(exchanger_ic)
    exchanger.set_equilibrate_solutions([1, 1, 1, 1])

    df = pd.read_csv(os.path.join(dataws, f"{prefix}_kinetic_phases.csv"))
    kin_phases = utils.parse_kinetics_dataframe(df)
    kin_phases[1]['Orgc_sed']['formula'] = 'Orgc_sed -1.0 C 1.0'
    kinetics = mup3d.KineticPhases(kin_phases)
    kinetics.set_ic(1)

    df = pd.read_csv(os.path.join(dataws, f"{prefix}_equilibrium_phases.csv"))
    equ_phases = utils.parse_equilibriums_dataframe(df)
    equilibriums = mup3d.EquilibriumPhases(equ_phases)
    equilibriums.set_ic(1)

    surfdic = utils.surfaces_csv_to_dict(os.path.join(dataws, f"{prefix}_surfaces.csv"))
    surfaces = mup3d.Surfaces(surfdic)
    surfaces.set_ic(1)

    # transport-first flopy sim: GWF + single conservative tracer GWT
    src_ws = os.path.join(cwd, f'{prefix}_from_mf6_src')
    sim = build_mf6_1d_transport_first_model(
        src_ws, prefix, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol,
        delr, delc, top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity,
        icelltype, strt, hclose, rclose, relax, nouter, ninner,
    )

    # build the reactive model by cloning the tracer GWT per component
    model = mup3d.Mup3d.from_mf6(sim, solution, name=prefix, gwt_name='tracer')
    model.set_wd(os.path.join(cwd, f'{prefix}_from_mf6'))
    model.set_database(os.path.join(databasews, 'ex5.dat'))
    model.set_initial_temp([7., 7., 7.])
    model.set_postfix(os.path.join(dataws, f'{prefix}_postfix.phqr'))
    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)
    model.set_phases(surfaces)
    model.initialize()

    wellchem = mup3d.ChemStress('wel', type='aux')
    wellchem.set_spd({0: [2], 1: [3]})  # period 0 -> solution 2, period 1 -> solution 3
    model.set_chem_stress(wellchem)

    model.write_simulation()

    # compare against the SAME benchmark as test05
    run_test(prefix, model, request=request, test_cli=True, libname=lib_name, treshold=0.02)


def decay_analytical(x, t, v, D, k, C0, Ci):
    '''van Genuchten & Alves (1982): semi-infinite column, flux inlet at C0, initial Ci,
    first-order decay k.'''
    from scipy.special import erfc, erfcx

    def e_erfc(a, z):  # exp(a) * erfc(z) without overflow
        z = np.asarray(z, float)
        return np.where(z > 0, np.exp(a - z**2) * erfcx(np.abs(z)), np.exp(a) * erfc(z))

    s = 2 * np.sqrt(D * t)
    u = v * np.sqrt(1 + 4 * k * D / v**2)
    A = (v / (v + u) * e_erfc((v - u) * x / (2 * D), (x - u * t) / s)
         + v / (v - u) * e_erfc((v + u) * x / (2 * D), (x + u * t) / s)
         + v**2 / (2 * k * D) * e_erfc(v * x / D - k * t, (x + v * t) / s))
    B = (1 - 0.5 * erfc((x - v * t) / s)
         - np.sqrt(v**2 * t / (np.pi * D)) * np.exp(-(x - v * t)**2 / (4 * D * t))
         + 0.5 * (1 + v * x / D + v**2 * t / D) * e_erfc(v * x / D, (x + v * t) / s))
    return C0 * A + Ci * np.exp(-k * t) * B


@pytest.mark.parametrize("nstp, tsmult", [(60, 1.0), (20, 1.1)], ids=["const_dt", "tsmult"])
def test06_decay(nstp, tsmult, prefix='test06'):
    '''Test 6: 1D transport with first-order KINETICS decay vs. analytical solution.

    Checks the reaction time step handed to PhreeqcRM: ahead of the front the column stays
    uniform, transport does nothing, and the exact answer is Ci*exp(-k*t). This catches a
    lagged dt (previous step's dt) and skipping of kinetic cells by the reaction mask.
    Same setup as benchmark/decay1d.
    '''
    length_units, time_units = "meters", "days"
    nper, nlay, nrow, ncol = 1, 1, 1, 100
    delr, delc, top, botm = 0.1, 1.0, 1.0, 0.0
    prsity = 0.3
    v = 0.1                          # pore velocity (m/d)
    dispersivity = 0.05              # m
    k = 0.03                         # first-order decay (1/d)
    C0, Ci = 1e-3, 5e-4              # inflow / initial Tr (mol/kgw)
    T = 60.0                         # d
    tdis_rc = [(T, nstp, tsmult)]

    solution = mup3d.Solutions({'pH': [7.0, 7.0], 'Na': [1e-3, 1e-3], 'Cl': [1e-3, 1e-3], 'Tr': [Ci, C0]})
    solution.set_ic(1)
    kinetics = mup3d.KineticPhases({1: {'Decay': {'m0': 1.0, 'parms': [k / 86400], 'formula': 'Tr 1'}}})
    kinetics.set_ic(1)

    model = mup3d.Mup3d(prefix, solution, nlay, nrow, ncol)
    model.set_wd(os.path.join(cwd, f'{prefix}_{"const" if tsmult == 1.0 else "tsmult"}'))
    model.set_postfix(os.path.join(dataws, f'{prefix}_postfix.phqr'))
    model.set_database(os.path.join(databasews, 'decay1d_datab.dat'))
    model.set_phases(kinetics)
    model.initialize()

    wellchem = mup3d.ChemStress('wel')
    wellchem.set_spd([2])
    model.set_chem_stress(wellchem)
    wel_spd = [[(0, 0, 0), v * prsity] + list(model.wel.data[0])]
    chdspd = [[(0, 0, ncol - 1), 1.0]]
    strt = np.ones((nlay, nrow, ncol))

    build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                 top, botm, wel_spd, chdspd, prsity, 1.0, 1.0, dispersivity, 0, 1e-10,
                                 strt, 1e-10, 1.0, 100, 300)
    assert model.run(libname=lib_name)

    out = pd.read_csv(os.path.join(model.wd, 'sout.csv'))
    dts = np.full(nstp, T / nstp) if tsmult == 1.0 else T * (tsmult - 1) / (tsmult**nstp - 1) * tsmult**np.arange(nstp)
    out['t'] = np.repeat(np.cumsum(dts), ncol)       # end of each MF6 step
    x = (out.cell.values[:ncol] - 0.5) * delr

    # uniform zone (x = 9.05 m, front still >= 5 m away for t <= 40 d): exact Ci*exp(-k*t)
    s = out[(out.cell == 91) & (out.t <= 40.0)]
    err = np.abs(s.Tr / (Ci * np.exp(-k * s.t)) - 1)
    assert err.max() < 1e-3, f"uniform-zone decay off by {err.max():.2%}"

    # behind the front at T: operator-splitting error ~ k*dt/2 (PHREEQC TRANSPORT: 1.5 %)
    if tsmult == 1.0:
        end = out[np.isclose(out.t, T)].sort_values('cell')
        behind = x < 4.0
        exact = decay_analytical(x[behind], T, v, dispersivity * v, k, C0, Ci)
        err = np.abs(end.Tr.values[behind] / exact - 1)
        assert err.max() < 0.05, f"profile behind front off by {err.max():.2%}"


def test_mf6_bin():
    '''Test that mf6 binary is available'''
    import subprocess as sp
    try:
        result = sp.run(['mf6', '--version'], capture_output=True, text=True, check=True)
        print("MODFLOW 6 binary is available.")
        print("Version info:", result.stdout)
    except FileNotFoundError:
        pytest.skip("MODFLOW 6 binary 'mf6' not found. Skipping test.")
    except sp.CalledProcessError as e:
        pytest.fail(f"Error occurred while checking MODFLOW 6 binary: {e}")

def get_test_dirs():
    '''Get test directories'''
    testdirs = [f for f in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, f)) and f.startswith('test') and not f.endswith('yaml')]
    # assert len(testdirs) > 0
    # assert testdirs[0] == 'test01'
    return testdirs

@pytest.mark.parametrize(
    'prefix',
    get_test_dirs()
)
@pytest.mark.skip
def test_yaml(prefix):
    '''tests running form yaml files
    '''	
    wd = os.path.join(cwd, f'{prefix}_yaml')
    #copy files to prefix_yaml with shutil
    if not os.path.exists(wd):
        os.makedirs(wd)
    for file in os.listdir(os.path.join(cwd, prefix)):
        shutil.copy(os.path.join(cwd, prefix, file), wd)
    run_yaml(wd)
    benchmarkdf = get_benchmark_results(prefix)
    testdf = pd.read_csv(os.path.join(cwd, wd,f"sout.csv"), index_col = 0)
    compare_results(benchmarkdf, testdf)

@pytest.mark.skip
def test01_yaml(prefix = 'test01'):

    '''Test 1: Simple 1D injection test with equilibrium phases
        Running from files insted of python memory
    '''	
    wd = os.path.join(cwd, f'{prefix}_yaml')
    #copy files to prefix_yaml with shutil
    if not os.path.exists(wd):
        os.makedirs(wd)
    for file in os.listdir(os.path.join(cwd, prefix)):
        shutil.copy(os.path.join(cwd, prefix, file), wd)
    run_yaml(wd)
    benchmarkdf = get_benchmark_results(prefix)
    testdf = pd.read_csv(os.path.join(cwd, wd,f"sout.csv"), index_col = 0)
    compare_results(benchmarkdf, testdf)


def get_benchmark_results(prefix):
    '''Get benchmark results'''
    benchwd = os.path.join(cwd, "benchmark")
    benchmarkdf = pd.read_csv(os.path.join(benchwd,f"{prefix}_benchmark.csv"), index_col = 0)
    return benchmarkdf

def get_benchmark_path(prefix):
    return Path(cwd) / "benchmark" / f"{prefix}_benchmark.csv"

def get_test_results(model):
    '''Get test results'''
    testdf = pd.read_csv(os.path.join(model.wd,f"sout.csv"), index_col = 0)
    return testdf

def compare_results(benchmarkdf, testdf, treshold = 0.01):
    '''Compare benchmark and test results'''

    # Align testdf to benchmark columns — testdf may have extra spatial columns
    # added by the SelectedOutput improvements (#26)
    shared_cols = [c for c in benchmarkdf.columns if c in testdf.columns]
    testdf = testdf[shared_cols]
    benchmarkdf = benchmarkdf[shared_cols]

    # assert both dataframes have the same shape
    assert benchmarkdf.shape == testdf.shape
    # assert both dataframes have the same columns
    assert benchmarkdf.columns.tolist() == testdf.columns.tolist()
    # assert both dataframes have the same indices (allow float64 platform differences)
    assert np.allclose(benchmarkdf.index.tolist(), testdf.index.tolist(), rtol=1e-10, atol=0)
    # iterate each column and each index and assert an absolute difference less than 0.01
    # skip spatial metadata columns
    spatial_cols = {"cell", "layer", "row", "col", "cell2d"}
    for col in [c for c in benchmarkdf.columns if c not in spatial_cols]:
        checkerarr = [i < treshold for i in np.abs((benchmarkdf.loc[:, col].values - testdf.loc[:, col].values)/testdf.loc[:, col].values)]
        lenarr = len(checkerarr)
        #get percentage of True
        perc = sum(checkerarr)/lenarr
        assert perc >= 0.99
        # assert all(i < treshold for i in np.abs(benchmarkdf.loc[:, col].values - testdf.loc[:, col].values))

def run_yaml(prefix):
    '''Run model from yaml file'''
    wd = os.path.join(prefix)
    #run the model
    mup3d.solve(wd)
    return

def run_test(prefix, model, request=None, test_cli = False, libname = None, *args, **kwargs):
    # for nthread in [1]:
        #try to run the model if success print test passed
    nthread = 1
    if test_cli:
        #run from cli
        bwd = os.getcwd()
        os.chdir(model.wd)
        #copy from env
        shutil.copy2(lib_name, f"./libmf6{lib_ext}")
        import subprocess as sp
        sp.run(['mf6rtm'])
        os.chdir(bwd)
    else:
        model.run(reactive=False, libname=libname)
        success = model.run(reactive=True, nthread=nthread, libname=libname)
        assert success

    testdf = get_test_results(model)

    if request is not None and request.config.getoption("--update-benchmarks"):
        testdf.to_csv(get_benchmark_path(prefix))
        return

    benchmarkdf = get_benchmark_results(prefix)
    compare_results(benchmarkdf, testdf, *args, **kwargs)

    return

# @pytest.fixture
# def cleanup():
#     '''Cleanup test files'''
#     #get all folders that start with 'test_'
#     testdirs = [f for f in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, f)) and f.startswith('test')]
#     # delete all test folders
#     for folder in testdirs:
#         shutil.rmtree(os.path.join(cwd, folder))


    # if os.path.exists(prefix):
    #     pass
    #     # shutil.rmtree(prefix, onerror=make_dir_writable)
    # return


if __name__ == '__main__':
    test01()
#     test01_yaml()


