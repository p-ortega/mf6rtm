# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart: 1D Dissolution Front
#
# This quickstart reproduces the classic Engesgaard & Kipp (1992) benchmark: a
# one-dimensional column initially filled with water in equilibrium with
# **calcite** is flushed by an inflowing **MgCl$_2$** solution. The incoming
# water dissolves calcite and briefly precipitates **dolomite**, producing a set
# of sharp, migrating mineral fronts.
#
# It also shows the recommended **transport-first workflow**: you build an
# ordinary MODFLOW 6 flow + (single, conservative) transport model with FloPy,
# then hand it to :meth:`~mf6rtm.mup3d.base.Mup3d.from_mf6`, which clones the
# tracer transport model into one reactive transport model per chemical
# component and couples it to PhreeqcRM.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy

from mf6rtm import utils, mup3d

# Resolve paths whether executed from the notebook dir or the repo root.
BASE = "." if os.path.isdir("data") else os.path.join("docs", "tutorials")
DATA = os.path.join(BASE, "data")

# %% [markdown]
# ## 1. Spatial and temporal discretization
#
# A 0.5 m column split into 50 cells, simulated for 0.24 days.

# %%
length_units = "meters"
time_units = "days"

nlay, nrow, ncol = 1, 1, 50
Lx = 0.5
delr = Lx / ncol
delc = 1.0
top, botm = 0.0, -1.0

k11 = 1.0            # hydraulic conductivity (m/d)
prsity = 0.32        # porosity
dispersivity = 0.0067  # longitudinal dispersivity (m)

q = 0.259            # injection rate at the inlet (m3/d)

perlen = 0.24        # simulation length (days)
tstep = 0.01
nstp = int(perlen / tstep)
tdis_rc = [(perlen, nstp, 1.0)]

# %% [markdown]
# ## 2. Build the flow + tracer transport model with FloPy
#
# We build a single **conservative tracer** GWT model. The inlet well carries a
# `tracer` auxiliary variable; ``from_mf6`` uses that boundary to inject the
# inflow solution's chemistry once the model becomes reactive.

# %%
sim_ws = os.path.join(BASE, "_tutorial01_run")
sim = flopy.mf6.MFSimulation(sim_name="quickstart", sim_ws=sim_ws, exe_name="mf6")
flopy.mf6.ModflowTdis(sim, nper=1, perioddata=tdis_rc, time_units=time_units)

# --- GWF flow model ---
gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
ims_gwf = flopy.mf6.ModflowIms(sim, linear_acceleration="CG", filename="gwf.ims")
sim.register_ims_package(ims_gwf, ["gwf"])

flopy.mf6.ModflowGwfdis(
    gwf, length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc, top=top, botm=botm, filename="gwf.dis",
)
flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True, icelltype=1, k=k11,
                        save_saturation=True, filename="gwf.npf")
flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename="gwf.ic")

# constant head at the outlet (last column)
flopy.mf6.ModflowGwfchd(
    gwf, stress_period_data=[[(0, 0, ncol - 1), 1.0]],
    pname="chd", filename="gwf.chd",
)
# injection well at the inlet, carrying a tracer auxiliary
flopy.mf6.ModflowGwfwel(
    gwf, stress_period_data=[[(0, 0, 0), q, 1.0]],
    auxiliary=["tracer"], pname="wel", filename="gwf.wel",
)
flopy.mf6.ModflowGwfoc(
    gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbb",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)

# --- GWT conservative tracer model (template for from_mf6) ---
gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt")
ims_gwt = flopy.mf6.ModflowIms(sim, linear_acceleration="BICGSTAB", filename="gwt.ims")
sim.register_ims_package(ims_gwt, ["gwt"])

flopy.mf6.ModflowGwtdis(
    gwt, length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc, top=top, botm=botm, filename="gwt.dis",
)
flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename="gwt.ic")
flopy.mf6.ModflowGwtssm(gwt, sources=[["wel", "aux", "tracer"]], filename="gwt.ssm")
flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=dispersivity, ath1=0.0,
                        filename="gwt.dsp")
flopy.mf6.ModflowGwtmst(gwt, porosity=prsity, filename="gwt.mst")
flopy.mf6.ModflowGwtoc(
    gwt, concentration_filerecord="gwt.ucn",
    saverecord=[("CONCENTRATION", "ALL")],
)
flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf",
                        exgmnameb="gwt", filename="gwf-gwt.gwfgwt")

# %% [markdown]
# ## 3. Define the geochemistry
#
# mf6rtm expresses PHREEQC chemistry as ``mup3d`` classes. Each class takes a
# ``components`` table with one row per component per block, and optionally a
# ``settings`` table with one row per block. Both are pandas DataFrames, read
# here from CSV with ``pd.read_csv``. The ``block`` column gives the block
# number written to the PHREEQC script. ``ic`` places the blocks on the grid:
# an integer block number for uniform chemistry, or a ``(nlay, nrow, ncol)``
# array to vary it in space. ``Solutions`` goes into ``from_mf6``; the phases
# are attached with ``set_*`` methods.
#
# **Solutions** — the aqueous waters. Solution ``1`` is the background pore
# water (placed everywhere with ``ic=1``); solution ``2`` is the inflow water
# the boundary ``ChemStress`` injects. ``pH`` and ``pe`` are rows of the
# components table, like the elements. The settings table gives each solution
# an optional ``note``. Concentrations are in mol/kgw unless a ``units`` option
# or column says otherwise.

# %%
sol_components = pd.read_csv(os.path.join(DATA, "tut01_solutions.csv"))
sol_settings = pd.read_csv(os.path.join(DATA, "tut01_solution_settings.csv"))
solution = mup3d.Solutions(sol_components, sol_settings, ic=1)
sol_components

# %%
sol_settings

# %% [markdown]
# **Equilibrium phases** — minerals held at equilibrium each step (PHREEQC
# ``EQUILIBRIUM_PHASES``): calcite and dolomite, with saturation index ``si``
# and initial moles ``m0``. Dolomite starts at ``m0 = 0`` so it can only
# precipitate. ``ic=1`` applies the same assemblage to every cell.

# %%
eqp_components = pd.read_csv(os.path.join(DATA, "tut01_equilibrium_phases.csv"))
equilibrium_phases = mup3d.EquilibriumPhases(eqp_components, ic=1)
eqp_components

# %% [markdown]
# ## 4. Couple chemistry to the MF6 model with `from_mf6`
#
# ``from_mf6`` reads the grid from the GWF model and stores the tracer GWT model
# as the template that will be cloned per component.

# %%
model = mup3d.Mup3d.from_mf6(sim, solution, name="quickstart", gwt_name="gwt")
model.set_wd(sim_ws)
model.set_database(os.path.join(DATA, "tut01_datab.dat"))
model.set_equilibrium_phases(equilibrium_phases)
model.set_postfix(os.path.join(DATA, "tut01_postfix.phqr"))
model.initialize()

# %% [markdown]
# ## 5. Assign the inflow chemistry to the boundary
#
# The `wel` package injects solution 2 (the MgCl$_2$ water). ``type='aux'``
# routes it through the GWF stress package + GWT SSM.

# %%
wellchem = mup3d.ChemStress("wel", type="aux")
wellchem.set_spd([2])
model.set_chem_stress(wellchem)

# %% [markdown]
# ## 6. Write and run the reactive transport simulation

# %%
model.write_simulation()

# Stage the platform's MODFLOW 6 binaries (mf6 + libmf6) into the run directory
# so the solver finds the shared library locally. pixi fetches them into
# ``benchmark/bin`` via the ``install-benchmark-bins`` task.
utils.prep_bins(model.wd, src_path=os.path.join(BASE, "..", "..", "benchmark", "bin"))

model.run()

# %% [markdown]
# ## 7. Results
#
# Read the selected-output table and plot the aqueous species and mineral
# amounts along the column at the final time.

# %%
sout = pd.read_csv(os.path.join(model.wd, "sout.csv"))
sout["x"] = (sout["col"] - 0.5) * delr           # cell-center distance (m)
final = sout[sout["time_d"] == sout["time_d"].max()]

plot_vars = ["Ca", "Mg", "Cl", "C4", "Calcite", "Dolomite", "pH"]

fig, axs = plt.subplots(len(plot_vars), 1, figsize=(7, 1.9 * len(plot_vars)),
                        sharex=True)
for ax, var in zip(axs, plot_vars):
    ax.plot(final["x"], final[var], color="tab:blue")
    ax.set_ylabel(var)
    ax.grid(alpha=0.3)
axs[-1].set_xlabel("distance (m)")
fig.suptitle("Dissolution fronts at t = %.2f d" % sout["time_d"].max())
fig.tight_layout()
