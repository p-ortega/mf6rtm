"""Tests for reactions in the immobile domain of the dual-domain mass transfer (DDMT) of feature-ddmt
(branch feature-ddmt-reactive-immobile).

Unit tests (no MODFLOW needed):
  * exact first-order exchange (feature-ddmt): mass conservation and the analytic two-box solution,
    also with parameters that change between steps
  * the immobile reaction step reacts all immobile cells (not the mobile diffmask)

Integration tests (need libmf6; set MF6RTM_LIBMF6 or have libmf6 in $CONDA_PREFIX/bin), marked slow:
  1D column, Cl breakthrough. mf6rtm DDMT (operator-split exchange in mf6rtm) is compared with an independent
  reference: the same MODFLOW 6 model with the built-in IST package (exchange solved implicitly by MODFLOW),
  run through modflowapi.
  * storage mode, constant parameters
  * reactive mode: first-order Cl decay (PHREEQC KINETICS) only in the immobile water, with an immobile chemistry
    of its own (YAML in another folder), vs. MF6 IST with first-order decay in the immobile domain
"""
import os
import shutil
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mf6rtm.simulation.ddmt import exchange_first_order_single_rate

cwd = Path(__file__).parent


# ---------------------------------------------------------------------------- unit tests
def _two_box(cm, ci, tm, ti, a, t):
    """analytic solution of tm dcm/dt = -a (cm - ci), ti dci/dt = a (cm - ci)"""
    tot = tm * cm + ti * ci
    d = (cm - ci) * np.exp(-a * (1 / tm + 1 / ti) * t)
    return (tot + ti * d) / (tm + ti), (tot - tm * d) / (tm + ti)


def test_exchange_mass_conservation_and_analytic():
    cm = np.array([[1.0, 0.0, 2.0]]); ci = np.array([[0.0, 1.0, 2.0]])
    tm, ti, a = np.array([0.3, 0.3, 0.3]), np.array([0.1, 0.2, 0.1]), np.array([2.0, 0.5, 1.0])
    m_new, i_new = exchange_first_order_single_rate(cm, ci, tm, ti, a, dt=0.1)
    np.testing.assert_allclose(tm * m_new + ti * i_new, tm * cm + ti * ci, rtol=1e-14)
    for j in range(3):
        em, ei = _two_box(cm[0, j], ci[0, j], tm[j], ti[j], a[j], 0.1)
        assert m_new[0, j] == pytest.approx(em) and i_new[0, j] == pytest.approx(ei)


def test_exchange_piecewise_time_varying():
    """step-wise exchange with parameters that change between steps = product of the exact step solutions"""
    cm, ci = np.array([[1.0]]), np.array([[0.0]])
    steps = [(0.30, 0.05, 1.0), (0.28, 0.07, 3.0), (0.26, 0.09, 9.0)]
    em, ei = 1.0, 0.0
    for tm, ti, a in steps:
        # mf6rtm keeps the concentrations when theta changes (no re-scaling of mass)
        cm, ci = exchange_first_order_single_rate(cm, ci, np.array([tm]), np.array([ti]), np.array([a]), dt=0.05)
        em, ei = _two_box(em, ei, tm, ti, a, 0.05)
    assert cm[0, 0] == pytest.approx(em) and ci[0, 0] == pytest.approx(ei)


class _FakeAPI:
    def __init__(self, nxyz, names):
        self.arrays = {}
        for n in names:
            for v in ("POROSITY", "THETAM"):
                self.arrays[f"{n}/MST/{v}"] = np.full(nxyz, 0.5)

    def get_var_address(self, var, model, pkg):
        return f"{model}/{pkg}/{var}"

    def get_value_ptr(self, addr):
        return self.arrays[addr]


def _bare_solver(nxyz=4):
    from mf6rtm.simulation.solver import Mf6RTM
    s = object.__new__(Mf6RTM)                     # no MODFLOW/PHREEQC initialization
    s.nxyz = nxyz
    s.ddmt_enabled = True
    s.component_model_dict = {"Cl": "gwt_cl", "Na": "gwt_na"}
    s.mf6api = _FakeAPI(nxyz, ["GWT_CL", "GWT_NA"])
    return s


def test_immobile_reactions_use_immobile_mask():
    s = _bare_solver()
    s.theta_immobile = np.array([0.1, 0.0, 0.1, 0.1])
    s.no_react_idx = np.array([3])
    s.immobile_conc_m3 = np.zeros((1, 4))
    s.ctime = 0.0

    class _FakeRM:
        ncomps = 1

        def SetConcentrations(self, c): self.c = np.asarray(c)
        def GetConcentrations(self): return self.c
        def _set_ctime(self, t): pass
        def _get_kper_kstp_from_mf6api(self, api): pass
        def _solve_phreeqcrm(self, dt, diffmask): self.mask = np.asarray(diffmask)
    s.phreeqcbmi_immobile = _FakeRM()
    mobile_diffmask = np.array([0.0, 0.0, 0.0, 1.0])          # mobile cells 0-2 unchanged
    s._solve_immobile_phreeqcrm(0.1, diffmask=mobile_diffmask)
    np.testing.assert_array_equal(s.phreeqcbmi_immobile.mask, [1.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------- integration tests
def _libmf6():
    ext = ".dylib" if "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() else \
        ".so" if "linux" in platform.platform().lower() else ".dll"
    cands = [os.environ.get("MF6RTM_LIBMF6")]
    if os.environ.get("CONDA_PREFIX"):
        cands.append(str(Path(os.environ["CONDA_PREFIX"]) / ("Scripts" if ext == ".dll" else "bin") / f"libmf6{ext}"))
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


LIB = _libmf6()
needs_mf6 = pytest.mark.skipif(LIB is None, reason="libmf6 not found (set MF6RTM_LIBMF6)")

NCOL, LX, Q, DISP = 40, 0.4, 0.2, 0.004          # m, m, m3/d (1 m2 cross-section), m
PERLEN, NSTP = 1.6, 160
TM0, TI0, Z0 = 0.32, 0.08, 2.0                    # constant case


def _build(ws, porosity, kinetics_rate=None):
    """1D column, Cl pulse-free step injection (test01 solutions, no phases), GWT model per component"""
    import flopy
    from mf6rtm import utils, mup3d
    from test_benchmark import build_mf6_1d_injection_model

    sol = utils.solution_df_to_dict(pd.read_csv(cwd / "data" / "test01_solutions.csv", comment="#", index_col=0))
    solution = mup3d.Solutions(sol); solution.set_ic(1)
    model = mup3d.Mup3d("ddmt", solution, 1, 1, NCOL)
    model.set_wd(str(ws))
    model.set_database(str(cwd / "database" / "pht3d_datab.dat"))
    if kinetics_rate is None:
        model.set_postfix(str(cwd / "data" / "test01_postfix.phqr"))
    else:
        # first-order removal of Cl (as MgCl2, charge neutral): dCl/dt = -k Cl; k in 1/d, PHREEQC TIME in s
        Path(ws).mkdir(parents=True, exist_ok=True)
        post = Path(ws) / "decay_postfix.phqr"
        post.write_text(f"""RATES
Cl
-start
10 k = {kinetics_rate} / 86400
20 moles = -k * TOT("Cl") / 2 * TIME
30 SAVE moles
-end
""" + (cwd / "data" / "test01_postfix.phqr").read_text())
        model.set_postfix(str(post))
        kin = mup3d.KineticPhases({0: {"Cl": {"formula": "Mg 1.0 Cl 2.0", "m": 1.0, "tol": 1e-10}}})
        kin.set_ic(1)
        model.set_phases(kin)
    model.initialize()
    wellchem = mup3d.ChemStress("wel"); wellchem.set_spd([2]); model.set_chem_stress(wellchem)
    wel_spd = [[(0, 0, 0), Q]]
    for i in range(len(wel_spd)):
        wel_spd[i].extend(model.wel.data[i])
    strt = np.ones((1, 1, NCOL))
    build_mf6_1d_injection_model(model, 1, [(PERLEN, NSTP, 1.0)], "meters", "days", 1, 1, NCOL, LX / NCOL, 1.0,
                                 0.0, -1.0, wel_spd, [[(0, 0, NCOL - 1), 1.0]], porosity, 1.0, 1.0, DISP, 1,
                                 1e-8, strt, 1e-8, 1.0, 100, 300)
    return model


KDEC = 3.0                                        # 1/d, first-order Cl decay in the immobile water only


def _cl_outlet(ws):
    import flopy
    ucn = flopy.utils.HeadFile(Path(ws) / "Cl.ucn", text="CONCENTRATION")
    times = np.array(ucn.get_times())
    return times, np.array([ucn.get_data(totim=t)[0, 0, -1] for t in times])


def _reference_ist(src_ws, ws, params, decay_cl=None):
    """copy of the MF6 model with the built-in IST package; parameters updated through modflowapi per step"""
    import flopy
    import modflowapi
    if Path(ws).exists():
        shutil.rmtree(ws)
    shutil.copytree(src_ws, ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    tm, ti, z = params(0.0)
    gwts = [m for m in sim.model_names if sim.get_model(m).model_type == "gwt6"]
    for name in gwts:
        g = sim.get_model(name)
        # decay_sorbed: required by MF6 >= 6.8 whenever decay is on (no SORPTION here, so it has no effect)
        dec = (dict(first_order_decay=True, decay=decay_cl, decay_sorbed=decay_cl)
               if (decay_cl is not None and name.lower() == "cl") else {})
        mst_file = g.mst.filename
        g.remove_package("mst")                       # MF6 needs DECAY in MST too if IST has it (mobile rate 0)
        flopy.mf6.ModflowGwtmst(g, porosity=tm + ti, filename=mst_file,
                                **(dict(first_order_decay=True, decay=0.0) if dec else {}))
        flopy.mf6.ModflowGwtist(g, porosity=tm + ti, volfrac=ti / (tm + ti), zetaim=z,
                                cim=g.ic.strt.array, filename=f"{name}.ist", **dec)
    sim.write_simulation(silent=True)
    mf6 = modflowapi.ModflowApi(LIB, working_directory=str(ws))
    mf6.initialize()
    ptr = []
    for name in gwts:
        N = name.upper()
        ptr.append({v: mf6.get_value_ptr(mf6.get_var_address(v, N, pk)) for v, pk in
                    (("POROSITY", "MST"), ("THETAM", "MST"))})
        ist = [p for p in sim.get_model(name).package_names if p.lower().startswith("ist")][0].upper()
        ptr[-1].update({f"IST_{v}": mf6.get_value_ptr(mf6.get_var_address(v, N, ist))
                        for v in ("POROSITY", "VOLFRAC", "ZETAIM")})
    nsln = mf6.get_subcomponent_count()
    t, tend = mf6.get_current_time(), mf6.get_end_time()
    while t < tend:
        dt = mf6.get_time_step()
        mf6.prepare_time_step(dt)
        tm, ti, z = params(t)
        for p in ptr:
            p["POROSITY"][:] = tm + ti; p["THETAM"][:] = tm
            p["IST_POROSITY"][:] = tm + ti; p["IST_VOLFRAC"][:] = ti / (tm + ti); p["IST_ZETAIM"][:] = z
        for sln in range(1, nsln + 1):
            mf6.prepare_solve(sln)
            for _ in range(100):
                if mf6.solve(sln):
                    break
            mf6.finalize_solve(sln)
        mf6.finalize_time_step()
        t = mf6.get_current_time()
    mf6.finalize()
    return _cl_outlet(ws)


def _run_ddmt(tmp_path, tag, theta_m, theta_i, alpham, **ddmt_kwargs):
    ws = tmp_path / f"mf6rtm_{tag}"
    model = _build(ws, theta_m)
    model.set_ddmt(theta_mobile=theta_m, theta_immobile=theta_i, alpham=alpham, mode="storage",
                   immobile_initial="mobile", output=False, output_format="csv", **ddmt_kwargs)
    model.config.output["output_format"] = "csv"
    model.save_config()
    assert model.run(reactive=False, libname=LIB)
    return ws, _cl_outlet(ws)


def _compare(a, b, tol):
    (ta, ca), (tb, cb) = a, b
    np.testing.assert_allclose(ta, tb)
    cmax = max(np.max(np.abs(ca)), 1e-30)
    err = np.max(np.abs(ca - cb)) / cmax
    print(f"max deviation mf6rtm-DDMT vs MF6-IST: {err:.3%} of the peak")
    assert err < tol, f"max deviation {err:.3%} of the peak (tolerance {tol:.1%})"
    return err


@needs_mf6
@pytest.mark.slow
def test_storage_constant_vs_mf6_ist(tmp_path):
    ws, out = _run_ddmt(tmp_path, "const", TM0, TI0, Z0)
    ref = _reference_ist(ws, tmp_path / "ist_const", lambda t: (TM0, TI0, Z0))
    nodd = _reference_ist(ws, tmp_path / "ist_none", lambda t: (TM0, 1e-6, 1e-9))   # no immobile domain
    eff = np.max(np.abs(ref[1] - nodd[1])) / np.max(ref[1])
    print(f"effect of the immobile domain on the BTC: {eff:.1%} of the peak")
    assert eff > 0.05                                                                # the exchange matters ...
    _compare(out, ref, tol=0.02)                                                     # ... and mf6rtm matches IST




@needs_mf6
@pytest.mark.slow
def test_reactive_immobile_decay_vs_mf6_ist(tmp_path):
    """reactive mode: PHREEQC first-order decay in the immobile water only (second PhreeqcRM instance),
    no reaction in the mobile water; reference = MF6 IST with first-order decay in the immobile domain"""
    im = _build(tmp_path / "immobile_chem", TM0, kinetics_rate=KDEC)       # chemistry of the immobile domain
    ws = tmp_path / "mf6rtm_react"
    model = _build(ws, TM0)
    model.set_ddmt(theta_mobile=TM0, theta_immobile=TI0, alpham=Z0, mode="reactive", immobile_initial="mobile",
                   immobile_yaml=im.phreeqcyaml_file, output=False, output_format="csv")
    model.config.output["output_format"] = "csv"
    model.save_config()
    assert model.run(reactive=True, libname=LIB)
    out = _cl_outlet(ws)
    ref = _reference_ist(ws, tmp_path / "ist_decay", lambda t: (TM0, TI0, Z0), decay_cl=KDEC)
    nodec = _reference_ist(ws, tmp_path / "ist_nodecay", lambda t: (TM0, TI0, Z0))
    eff = np.max(np.abs(ref[1] - nodec[1])) / np.max(nodec[1])
    print(f"effect of the immobile decay on the BTC: {eff:.1%} of the peak")
    assert eff > 0.05
    _compare(out, ref, tol=0.03)
