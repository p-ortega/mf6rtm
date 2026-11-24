"""Spike for the design question in the review of PR #80: let MF6 IST do the mobile-immobile exchange and let
mf6rtm react the immobile water by reading/writing the IST immobile concentrations through the API.

1D column (40 cells), IST with theta_m 0.32, theta_im 0.08, zeta 2/d, continuous inflow.
  reference : IST with built-in first-order decay in the immobile domain (k = 3/d)
  writeback : IST without decay; after each time step CIMNEW *= exp(-k dt) through the API
              (stand-in for a PhreeqcRM step on the immobile water)
  control   : IST without decay, nothing written
Also tests writing CIM or CIMOLD instead of CIMNEW.

Result (MODFLOW 6.6.2, modflowapi 0.2.0): writing CIMNEW after each time step is carried forward; the outlet
deviation from the reference shrinks first-order with dt (1.30/0.68/0.35/0.18 % of the peak for dt 0.02-0.0025 d)
while the immobile reaction changes the outlet by 57 % of the peak. Writing CIM or CIMOLD is not carried forward.
The MF6 budget closes (0.00 %); the IST budget term shows the uptake into the immobile domain.

    MF6RTM_LIBMF6=/path/to/libmf6 python ist_cimnew_spike.py
"""
import os

import numpy as np
import flopy
from pathlib import Path
N, L = 40, 0.4; DX = L / N
TM, TI, ZETA = 0.32, 0.08, 2.0          # mobile/immobile water content, exchange coefficient (1/d)
Q, C0 = 0.1, 1.0                         # m3/d per unit area, inflow concentration (mol/m3)
DT, TEND = 0.01, 4.0


def build(ws, decay_im=None, dt=DT):
    ws = Path(ws); ws.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name="sim", sim_ws=str(ws), exe_name="mf6", verbosity_level=0)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(TEND, int(round(TEND / dt)), 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf")
    ims = flopy.mf6.ModflowIms(sim, complexity="simple", filename="gwf.ims"); sim.register_ims_package(ims, ["gwf"])
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=N, delr=DX, delc=1.0, top=1.0, botm=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, save_specific_discharge=True, save_saturation=True)
    flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, N - 1), 1.0]])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[(0, 0, 0), Q, C0]], auxiliary=["CONCENTRATION"], pname="wel")
    flopy.mf6.ModflowGwfoc(gwf)
    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt")
    imt = flopy.mf6.ModflowIms(sim, complexity="simple", linear_acceleration="BICGSTAB", filename="gwt.ims",
                               inner_dvclose=1e-10, outer_dvclose=1e-10)
    sim.register_ims_package(imt, ["gwt"])
    flopy.mf6.ModflowGwtdis(gwt, nlay=1, nrow=1, ncol=N, delr=DX, delc=1.0, top=1.0, botm=0.0)
    flopy.mf6.ModflowGwtic(gwt, strt=0.0)
    flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=0.005, ath1=0.0005, diffc=0.0)
    dec = decay_im is not None
    flopy.mf6.ModflowGwtmst(gwt, porosity=TM + TI, **(dict(first_order_decay=True, decay=0.0) if dec else {}))
    flopy.mf6.ModflowGwtist(gwt, porosity=TM + TI, volfrac=TI / (TM + TI), zetaim=ZETA, cim=0.0,
                            **(dict(first_order_decay=True, decay=decay_im) if dec else {}))
    flopy.mf6.ModflowGwtssm(gwt, sources=[["wel", "AUX", "CONCENTRATION"]])
    flopy.mf6.ModflowGwtoc(gwt, budget_filerecord="gwt.cbc", saverecord=[("BUDGET", "ALL")])
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf", exgmnameb="gwt")
    sim.write_simulation(silent=True)
    return ws


import sys, numpy as np
from modflowapi import ModflowApi
LIB = os.environ.get("MF6RTM_LIBMF6", "libmf6")
K = 3.0


def run(tag, dt, decay_im=None, write=None, var="CIMNEW"):
    ws = build(Path(os.environ.get("SPIKE_WS", "ist_cimnew_spike_runs")) / f"{tag}_{dt}", decay_im=decay_im, dt=dt)
    mf = ModflowApi(LIB, working_directory=str(ws)); mf.initialize()
    x = mf.get_value_ptr(mf.get_var_address("X", "GWT"))
    cim = {v: mf.get_value_ptr(mf.get_var_address(v, "GWT", "IST_0")) for v in ("CIM", "CIMNEW", "CIMOLD")}
    t, te, out, mim = mf.get_current_time(), mf.get_end_time(), [], []
    while t < te:
        mf.update()                                   # prepare + solve + finalize one time step
        if write is not None:
            cim[var][:] = write(cim[var], dt)         # operator-split "reaction" in the immobile water
        t = mf.get_current_time()
        out.append((t, x[-1])); mim.append(float(np.sum(cim["CIMNEW"]) * TI * DX))
    mf.finalize()
    a = np.array(out); return a[:, 0], a[:, 1], np.array(mim)


if __name__ == "__main__":
    decay = lambda c, dt: c * np.exp(-K * dt)
    rows = []
    for dt in (0.02, 0.01, 0.005, 0.0025):
        tr, cr, mr = run("ref", dt, decay_im=K)
        tw, cw, mw = run("wb", dt, write=decay)
        tc, cc, mc = run("ctrl", dt)
        pk = cc.max()
        rows.append((dt, 100 * np.abs(cw - cr).max() / pk, 100 * np.abs(cc - cr).max() / pk,
                     100 * abs(mw[-1] - mr[-1]) / mr[-1]))
        print(f"dt={dt:<7} write-back vs IST-decay: max dev {rows[-1][1]:.3f} % of peak | "
              f"effect of the immobile reaction (control vs reference) {rows[-1][2]:.2f} % | immobile mass end dev {rows[-1][3]:.3f} %", flush=True)
    # which array has to be written? (same test with CIM and CIMOLD at dt = 0.01)
    tr, cr, _ = run("ref", 0.01, decay_im=K)
    for v in ("CIM", "CIMOLD"):
        tw, cw, _ = run(f"wb_{v}", 0.01, write=decay, var=v)
        print(f"writing {v:7s} instead: max dev vs reference {100 * np.abs(cw - cr).max() / cr.max():.2f} % of peak")
