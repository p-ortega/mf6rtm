# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1D Pyrite Oxidation with BYO(Build your own model files)
#
# This tutorial reproduces the marine-sediment oxidation experiment of
# Appelo et al. (1998): a column of pyrite-bearing sediment is first flushed
# with a dilute MgCl\ :sub:`2` solution and then with an oxidising solution,
# driving pyrite oxidation and a cascade of secondary reactions (calcite
# dissolution, cation/proton exchange, organic-matter oxidation, surface
# complexation).
#
# It uses the **classic workflow (BYO)**: build the :class:`~mf6rtm.mup3d.base.Mup3d`
# model first, then construct the MODFLOW 6 flow + per-component transport models
# yourself. The classic workflow injects the inflow chemistry directly into the
# well as per-period auxiliary concentrations, so the injected water can change
# between stress periods (dilute in period 0, oxidising in period 1).

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy

from mf6rtm import utils, mup3d

BASE = "." if os.path.isdir("data") else os.path.join("docs", "tutorials")
DATA = os.path.join(BASE, "data")

# %% [markdown]
# ## 1. Discretization and flow
#
# A 5.3 cm column in 16 cells, run in two stress periods: a dilute flush
# (period 0) followed by the oxidising flush (period 1). The injection well at
# the inlet runs at the same rate in both periods; only its chemistry changes.

# %%
length_units = "meters"
time_units = "days"

nlay, nrow, ncol = 1, 1, 16
Lx = 0.053
delr = Lx / ncol
delc = 1.0
top = 2.87433e-03
botm = np.linspace(top, 0.0, nlay + 1)[1:]

k11 = 1.0
prsity = 0.376
dispersivity = 0.00537

q = 2.4e-4  # injection rate (m3/d)

nper = 2
nstp = [64, 100]
perlen = [0.9333, 1.45833]
tdis_rc = [(pl, ns, 1.0) for pl, ns in zip(perlen, nstp)]

# constant head at the outlet; injection well at the inlet in both periods
chdspd = [[(0, 0, ncol - 1), 1.0]]
wel_spd = {kper: [[(0, 0, 0), q]] for kper in range(nper)}

# %% [markdown]
# ## 2. Geochemistry — defining the reaction network
#
# mf6rtm mirrors PHREEQC's chemistry blocks as ``mup3d`` classes. Every input
# type follows the same pattern: **load** the data (here from CSV via a
# ``utils`` helper), **wrap** it in the matching class, and **place** it with
# ``.set_ic(...)`` — an integer *zone number* for uniform chemistry, or a
# ``(nlay, nrow, ncol)`` array of zone numbers for spatially varying chemistry.
# The objects are attached to the ``Mup3d`` model in the next section.

# %% [markdown]
# **Solutions** — the aqueous waters (pH, pe, element totals), one per column.
# Column ``1`` is the initial pore water (applied everywhere); columns ``2`` and
# ``3`` are the dilute and oxidising inflow waters injected in periods 0 and 1.

# %%
solutionsdf = pd.read_csv(os.path.join(DATA, "tut02_solutions.csv"),
                          comment="#", index_col=0)
solutions = utils.solution_df_to_dict(solutionsdf)
solution = mup3d.Solutions(solutions)
solution.set_ic(np.ones((nlay, nrow, ncol), dtype=float))
solutionsdf

# %% [markdown]
# **Exchange phases** — cation-exchange sites (PHREEQC ``EXCHANGE``). The column
# is split into four zones with different exchanger amounts; the ``set_ic`` array
# maps cells 0–3 → zone 1, 4–7 → zone 2, and so on.
# ``set_equilibrate_solutions`` pre-equilibrates each zone's exchanger with the
# given solution number before the run.

# %%
excdf = pd.read_csv(os.path.join(DATA, "tut02_exchange.csv"), comment="#", index_col=0)
excdf.columns = [0, 1, 2, 3]
exchanger_dict = excdf.to_dict()
for _z, subdict in exchanger_dict.items():
    for key in subdict:
        subdict[key] = {"m0": subdict[key]}
exchanger = mup3d.ExchangePhases(exchanger_dict)

exchanger_ic = np.ones((nlay, nrow, ncol), dtype=float)
exchanger_ic[0, 0, :4] = 1
exchanger_ic[0, 0, 4:8] = 2
exchanger_ic[0, 0, 8:12] = 3
exchanger_ic[0, 0, 12:] = 4
exchanger.set_ic(exchanger_ic)
exchanger.set_equilibrate_solutions([1, 1, 1, 1])

# %% [markdown]
# **Kinetic phases** — rate-controlled reactions (PHREEQC ``KINETICS``): pyrite
# and organic-matter oxidation, the drivers of this experiment.
# ``parse_kinetics_dataframe`` builds the per-phase parameter dict; ``set_ic(1)``
# applies the same kinetics to every cell.

# %%
kin_df = pd.read_csv(os.path.join(DATA, "tut02_kinetic_phases.csv"))
kin_phases = utils.parse_kinetics_dataframe(kin_df)
kin_phases[1]["Orgc_sed"]["formula"] = "Orgc_sed -1.0 C 1.0"
kinetics = mup3d.KineticPhases(kin_phases)
kinetics.set_ic(1)

# %% [markdown]
# **Equilibrium phases & surfaces** — ``EquilibriumPhases`` are minerals kept at
# equilibrium each step (PHREEQC ``EQUILIBRIUM_PHASES``, here calcite);
# ``Surfaces`` are surface-complexation sites (PHREEQC ``SURFACE``). Both are
# uniform, so ``set_ic(1)``.

# %%
eqp_df = pd.read_csv(os.path.join(DATA, "tut02_equilibrium_phases.csv"))
equilibriums = mup3d.EquilibriumPhases(utils.parse_equilibriums_dataframe(eqp_df))
equilibriums.set_ic(1)

surfaces = mup3d.Surfaces(utils.surfaces_csv_to_dict(
    os.path.join(DATA, "tut02_surfaces.csv")))
surfaces.set_ic(1)

# %% [markdown]
# ## 3. Build the `Mup3d` model and assign inflow chemistry
#
# Create the model, attach every phase type, and initialize PhreeqcRM. Then map
# the inflow solutions to the well **per stress period** with ``set_spd([2, 3])``
# (solution 2 in period 0, solution 3 in period 1). ``set_chem_stress`` stores
# the per-period component concentrations in ``model.wel.data``.

# %%
sim_ws = os.path.join(BASE, "_tutorial02_run")
model = mup3d.Mup3d("pyrite_oxidation", solution, nlay, nrow, ncol)
model.set_wd(sim_ws)
model.set_database(os.path.join(DATA, "tut02_datab.dat"))
model.set_initial_temp([7.0, 7.0, 7.0])
model.set_postfix(os.path.join(DATA, "tut02_postfix.phqr"))

model.set_exchange_phases(exchanger)
model.set_phases(kinetics)
model.set_phases(equilibriums)
model.set_phases(surfaces)

model.initialize()

# inflow chemistry: solution 2 (dilute) in period 0, solution 3 (oxidising) in period 1
wellchem = mup3d.ChemStress("wel")
wellchem.set_spd([2, 3])
model.set_chem_stress(wellchem)

# append the per-period component concentrations to the well stress period data
for kper in wel_spd:
    for i in range(len(wel_spd[kper])):
        wel_spd[kper][i].extend(model.wel.data[kper])

# %% [markdown]
# ## 4. Build the flow + per-component transport models
#
# The well injects the components as auxiliary variables (per period); one GWT
# model is then built per chemical component, each sourcing the well auxiliary.

# %%
def build_model(model):
    sim = flopy.mf6.MFSimulation(sim_name=model.name, sim_ws=model.wd, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)

    # --- GWF flow model ---
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    imsgwf = flopy.mf6.ModflowIms(sim, complexity="complex", linear_acceleration="CG",
                                  filename="gwf.ims")
    sim.register_ims_package(imsgwf, [gwf.name])

    flopy.mf6.ModflowGwfdis(gwf, length_units=length_units, nlay=nlay, nrow=nrow,
                            ncol=ncol, delr=delr, delc=delc, top=top, botm=botm,
                            filename="gwf.dis")
    flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True, save_saturation=True,
                            icelltype=1, k=k11, filename="gwf.npf")
    flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename="gwf.ic")
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd, pname="chd",
                            filename="gwf.chd")
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, auxiliary=model.components,
                            pname="wel", filename="gwf.wel")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbb",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])

    # --- one GWT model per component ---
    for c in model.components:
        gwt = flopy.mf6.MFModel(sim, model_type="gwt6", modelname=c,
                                model_nam_file=f"{c}.nam")
        imsgwt = flopy.mf6.ModflowIms(sim, linear_acceleration="BICGSTAB",
                                      filename=f"{c}.ims")
        sim.register_ims_package(imsgwt, [gwt.name])

        flopy.mf6.ModflowGwtdis(gwt, length_units=length_units, nlay=nlay, nrow=nrow,
                                ncol=ncol, delr=delr, delc=delc, top=top, botm=botm,
                                filename=f"{c}.dis")
        flopy.mf6.ModflowGwtic(gwt, strt=model.sconc[c], filename=f"{c}.ic")
        flopy.mf6.ModflowGwtssm(gwt, sources=[["wel", "aux", c]], filename=f"{c}.ssm")
        flopy.mf6.ModflowGwtadv(gwt, scheme="tvd")
        flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=dispersivity,
                                ath1=dispersivity * 0.1, filename=f"{c}.dsp")
        flopy.mf6.ModflowGwtmst(gwt, porosity=prsity, filename=f"{c}.mst")
        flopy.mf6.ModflowGwtoc(gwt, concentration_filerecord=f"{c}.ucn",
                               saverecord=[("CONCENTRATION", "ALL")])
        flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf",
                                exgmnameb=c, filename=f"{c}.gwfgwt")

    sim.write_simulation()
    return sim


sim = build_model(model)

# %% [markdown]
# ## 5. Run

# %%
# Stage the platform's MODFLOW 6 binaries into the run directory (pixi fetches
# them into ``benchmark/bin``) so the solver finds libmf6 locally.
utils.prep_bins(model.wd, src_path=os.path.join(BASE, "..", "..", "benchmark", "bin"))

model.run(min_concentration=0.0)

# %% [markdown]
# ## 6. Results: effluent breakthrough curves
#
# Concentrations at the **outlet cell** over the experiment, plotted against
# cumulative effluent volume and compared with the measured effluent data of
# Appelo et al. (1998). (These are real laboratory measurements, not another
# model.)

# %%
sout = pd.read_csv(os.path.join(model.wd, "sout.csv"))
qml = q * 1.0e6  # m3/d -> mL/d, so time (d) * qml = effluent volume (mL)
outlet = sout[sout["col"] == sout["col"].max()].copy()
outlet["vol"] = outlet["time_d"] * qml

# Measured effluent data (real experiment, not a model benchmark).
obs = pd.read_csv(os.path.join(DATA, "tut02_obs.txt"), sep="\t", skipinitialspace=True)
obs.replace(-9999, np.nan, inplace=True)
obs.dropna(axis=0, how="all", inplace=True)
obs.columns = ["vol", "Mg", "Ca", "Alk", "Cl", "S(6)", "pH"]
obs.set_index("vol", inplace=True)
obs = obs.iloc[3:, :]

plot_vars = [v for v in ["Mg", "Ca", "Alk", "S(6)", "pH"] if v in outlet.columns]

fig, axs = plt.subplots(len(plot_vars), 1, figsize=(7, 2.0 * len(plot_vars)),
                        sharex=True)
for ax, var in zip(np.atleast_1d(axs), plot_vars):
    ax.plot(outlet["vol"], outlet[var], color="tab:red", zorder=10, label="mf6rtm")
    if var in obs.columns:
        ax.scatter(obs.index, obs[var], s=18, facecolors="none",
                   edgecolors="green", label="Appelo (1998), observed")
    ax.set_ylabel(var)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
np.atleast_1d(axs)[-1].set_xlabel("effluent volume (mL)")
fig.suptitle("Pyrite oxidation column: outlet breakthrough")
fig.tight_layout()
