from pathlib import Path
import os
import sys
import platform

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tempfile
import shutil 

#add mf6rtm path to the system
sys.path.insert(0,os.path.join("..","mf6rtm"))
import flopy
import mf6rtm
import utils


dataws = os.path.join("data")
databasews = os.path.join("database")

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
    utils.prep_bins(sim_ws, src_path=os.path.join('..','bin'), get_only=['mf6', 'libmf6'])
    
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
    utils.prep_bins(sim_ws, src_path=os.path.join('..','bin'), get_only=['mf6', 'libmf6'])
    
    return sim

def test01(prefix = 'test01'):

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
    equilibrium_phases = utils.equilibrium_phases_csv_to_dict(os.path.join(dataws, f'{prefix}_equilibrium_phases.csv'))

    sol_ic = 1
    #add solutions to clss
    solution = mf6rtm.Solutions(solutions)
    solution.set_ic(sol_ic)
    #create equilibrium phases class
    equilibrium_phases = mf6rtm.EquilibriumPhases(equilibrium_phases)
    equilibrium_phases.set_ic(1)

    #create model class
    model = mf6rtm.Mup3d(prefix,solution, nlay, nrow, ncol)

    #set model workspace
    model.set_wd(os.path.join(f'{prefix}'))
    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    #set database
    database = os.path.join(databasews, f'pht3d_datab.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_phases(equilibrium_phases)

    model.initialize()

    wellchem = mf6rtm.ChemStress('wel')
    sol_spd = [2]
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)

    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    run_test(prefix, model, mf6sim)
    try:
        cleanup(prefix)
    except:
        pass

    return 

def test02(prefix = 'test02'):
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
    equilibrium_phases = utils.equilibrium_phases_csv_to_dict(os.path.join(dataws, f'{prefix}_equilibrium_phases.csv'))

    for key, value in equilibrium_phases.items():
        for k, v in value.items():
            v[-1] = mf6rtm.concentration_volbulk_to_volwater( v[-1], prsity)
    #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)

    #add solutions to clss
    solution = mf6rtm.Solutions(solutions)
    solution.set_ic(sol_ic)

    #create equilibrium phases class
    equilibrium_phases = mf6rtm.EquilibriumPhases(equilibrium_phases)
    eqp_ic = 1
    # eqp_ic[3:,:,0]= -1 #boundary condation in layer 0 of no eq phases
    equilibrium_phases.set_ic(eqp_ic)

    #create model class
    model = mf6rtm.Mup3d(prefix,solution, nlay, nrow, ncol)

    #set model workspace
    model.set_wd(os.path.join(f'{prefix}'))

    #set database
    database = os.path.join(databasews, f'pht3d_datab_walter1994.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_phases(equilibrium_phases)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    model.initialize()

    wellchem = mf6rtm.ChemStress('wel')
    sol_spd = [2]

    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)


    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    run_test(prefix, model, mf6sim)
    try:
        cleanup(prefix)
    except:
        pass
    return

def test03(prefix = 'test03'):
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

    equilibrium_phases = utils.equilibrium_phases_csv_to_dict(os.path.join(dataws, f'{prefix}_equilibrium_phases.csv'))

    # equlibrium phases is a dictionary with keys as the phase number, values is another dictionary with phase name and an array of saturation indices as element 0 and concentrations as element 1. multiply the concentrations by 2
    for key, value in equilibrium_phases.items():
        for k, v in value.items():
            v[-1] = mf6rtm.concentration_volbulk_to_volwater( v[-1], prsity)
    #assign solutions to grid
    sol_ic = np.ones((nlay, nrow, ncol), dtype=int)

    #add solutions to clss
    solution = mf6rtm.Solutions(solutions)
    solution.set_ic(sol_ic)

    #create equilibrium phases class
    equilibrium_phases = mf6rtm.EquilibriumPhases(equilibrium_phases)
    eqp_ic = np.ones((nlay, nrow, ncol), dtype=int)*1
    eqp_ic[3:,:,0]= -1 #boundary condation in layer 0 of no eq phases
    equilibrium_phases.set_ic(eqp_ic)

    #create model class
    model = mf6rtm.Mup3d(prefix,solution, nlay, nrow, ncol)

    #set model workspace
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    model.set_wd(os.path.join(f'{prefix}'))

    #set database
    database = os.path.join(databasews, f'pht3d_datab_walter1994.dat')
    model.set_database(database)

    #include equilibrium phases in model class
    model.set_equilibrium_phases(equilibrium_phases)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)
    model.initialize()

    wellchem = mf6rtm.ChemStress('chdtail')
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
    
    run_test(prefix, model, mf6sim)

    try:
        cleanup(prefix)
    except:
        pass
    return 

def test04(prefix = 'test04'):
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
    solution = mf6rtm.Solutions(solutions)
    solution.set_ic(sol_ic)

    excdf = pd.read_csv(os.path.join(dataws,f"{prefix}_exchange.csv"), comment = '#',  index_col = 0)
    exchangerdic = utils.solution_df_to_dict(excdf)

    exchanger = mf6rtm.ExchangePhases(exchangerdic)
    exchanger.set_ic(np.ones((nlay, nrow, ncol), dtype=float))

    #create model class
    model = mf6rtm.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # temp_dir = tempfile.TemporaryDirectory()

    model.set_wd(os.path.join(f'{prefix}'))
    # model.set_wd(temp_dir.name)

    #set database
    database = os.path.join(databasews, f'pht3d_datab.dat')
    model.set_database(database)
    model.set_exchange_phases(exchanger)

    postfix = os.path.join(dataws, f'{prefix}_postfix.phqr')
    model.set_postfix(postfix)

    model.initialize()

    wellchem = mf6rtm.ChemStress('wel')
    sol_spd = [2]
    wellchem.set_spd(sol_spd)
    model.set_chem_stress(wellchem)

    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])

    mf6sim = build_mf6_1d_injection_model(model, nper, tdis_rc, length_units, time_units, nlay, nrow, ncol, delr, delc,
                                    top, botm, wel_spd, chdspd, prsity, k11, k33, dispersivity, icelltype, hclose, 
                                    strt, rclose, relax, nouter, ninner)
    
    run_test(prefix, model, mf6sim)

    try:
        cleanup(prefix)
    except:
        pass

    return 


def test05(prefix = 'test05'):
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
    solution = mf6rtm.Solutions(solutions)
    solution.set_ic(sol_ic)

    #exchange
    excdf = pd.read_csv(os.path.join(dataws,f"{prefix}_exchange.csv"), comment = '#',  index_col = 0)
    exchangerdic = utils.solution_df_to_dict(excdf)

    exchanger = mf6rtm.ExchangePhases(exchangerdic)
    exchanger_ic = np.ones((nlay, nrow, ncol), dtype=float)
    exchanger_ic[0,0,:4] = 1
    exchanger_ic[0,0,4:8] = 2
    exchanger_ic[0,0,8:12] = 3
    exchanger_ic[0,0,12:] = 4


    exchanger.set_ic(exchanger_ic)
    eq_solutions = [1,1,1,1]
    exchanger.set_equilibrate_solutions(eq_solutions)

    #kinetics
    kinedic = utils.kinetics_phases_csv_to_dict(os.path.join(dataws,f"{prefix}_kinetic_phases.csv"))
    orgsed_form = 'Orgc_sed -1.0 C 1.0' 
    kinedic[1]['Orgc_sed'].append(orgsed_form)
    kinetics = mf6rtm.KineticPhases(kinedic)
    kinetics.set_ic(1)

    #eq phases
    equilibriums = utils.equilibrium_phases_csv_to_dict(os.path.join(dataws,f"{prefix}_equilibrium_phases.csv"))
    equilibriums = mf6rtm.EquilibriumPhases(equilibriums)
    equilibriums.set_ic(1)

    #surfaces
    surfdic = utils.surfaces_csv_to_dict(os.path.join(dataws,f"{prefix}_surfaces.csv"))
    surfaces = mf6rtm.Surfaces(surfdic)
    surfaces.set_ic(1)
    # surfaces.set_options(['no_edl'])

    #create model class
    model = mf6rtm.Mup3d(prefix,solution, nlay, nrow, ncol)

    # set model workspace
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model.set_wd(os.path.join(f'{prefix}'))

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

    wellchem = mf6rtm.ChemStress('wel')
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
    
    run_test(prefix, model, mf6sim, treshold = 0.02)

    try:
        cleanup(prefix)
    except:
        pass

    return 

def get_benchmark_results(prefix):
    '''Get benchmark results'''
    dataws = os.path.join("benchmark")
    benchmarkdf = pd.read_csv(os.path.join(dataws,f"{prefix}_benchmark.csv"), index_col = 0)
    return benchmarkdf

def get_test_results(model):
    '''Get test results'''
    testdf = pd.read_csv(os.path.join(model.wd,f"sout.csv"), index_col = 0)
    return testdf

def compare_results(benchmarkdf, testdf, treshold = 0.01):
    '''Compare benchmark and test results'''
    # assert both dataframes have the same shape
    assert benchmarkdf.shape == testdf.shape
    # assert both dataframes have the same columns
    assert benchmarkdf.columns.tolist() == testdf.columns.tolist()
    # assert both dataframes have the same indices
    assert benchmarkdf.index.tolist() == testdf.index.tolist()
    # iterate each column and each index and assert an absolute difference less than 0.01 
    for col in benchmarkdf.columns:
        checkerarr = [i < treshold for i in np.abs(benchmarkdf.loc[:, col].values - testdf.loc[:, col].values)]
        lenarr = len(checkerarr)
        #get percentage of True
        perc = sum(checkerarr)/lenarr
        assert perc >= 0.99
        # assert all(i < treshold for i in np.abs(benchmarkdf.loc[:, col].values - testdf.loc[:, col].values))

def run_test(prefix, model, mf6sim, *args, **kwargs):
    #try to run the model if success print test passed
    success = model.run_mup3d(mf6sim, reaction=True)
    assert success

    # treshold = args.get('treshold', 0.01)
    benchmarkdf = get_benchmark_results(prefix)
    testdf = get_test_results(model)

    compare_results(benchmarkdf, testdf, *args, **kwargs)

    return

def cleanup(prefix):
    '''Cleanup test files'''
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    return

# def run_autotest():
    # test_01()
    # test_02()
    # test_04()
    # test_05()

# if __name__ == '__main__':
    # run_autotest()


