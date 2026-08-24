"""MODFLOW 6 API wrapper for mf6rtm.

Defines :class:`Mf6API`, a subclass of ``modflowapi.ModflowApi`` that loads
the coupled flow/transport simulation and drives its solve loop inside the
mf6rtm coupling.
"""
from datetime import datetime

import flopy
import modflowapi


class Mf6API(modflowapi.ModflowApi):
    """MODFLOW 6 API wrapper driving the flow and transport solve.

    Extends ``modflowapi.ModflowApi`` to load the coupled simulation and run
    the per-timestep GWF/GWT solve loop that :class:`Mf6RTM` alternates with
    the PhreeqcRM reaction step.

    Parameters
    ----------
    wd : os.PathLike
        Working directory containing the MODFLOW 6 simulation.
    dll : os.PathLike
        Path to the MODFLOW 6 shared library (libmf6).
    """

    def __init__(self, wd, dll):
        # TODO: reverse the order of args to match modflowapi?
        modflowapi.ModflowApi.__init__(self, dll, working_directory=wd)
        self.initialize()
        self.sim = flopy.mf6.MFSimulation.load(sim_ws=wd,
                                               verbosity_level=0,
                                               load_only=["dis", "disv", "tdis"])
        self.fmi = False

    def _prepare_mf6(self):
        """Prepare mf6 bmi for transport calculations"""
        self.modelnmes = [nme.capitalize() for nme in self.sim.model_names]
        # self.components = [nme.capitalize() for nme in self.sim.model_names[1:]]
        self.nsln = self.get_subcomponent_count() # i.e. Modflow6 models
        self.sim_start = datetime.now()
        self.ctimes = [0.0]
        self.num_fails = 0

    def _check_fmi(self):
        """Check if fmi is in the nam file"""
        ...
        return

    def _set_simtype_gwt(self):
        """Set the gwt sim type as sequential or flow interface"""
        ...

    def _solve_gwt(self):
        """Function to solve the transport loop"""
        # prep to solve each Modflow6 model "solution" (i.e. sln)
        for sln in range(1, self.nsln + 1):
            self.prepare_solve(sln)
        # the one-based stress period number
        stress_period = self.get_value(self.get_var_address("KPER", "TDIS"))[0]
        time_step = self.get_value(self.get_var_address("KSTP", "TDIS"))[0]

        self.kper = stress_period
        self.kstp = time_step
        msg = f"{'Transport':<15} | {'Stress period:':<15} {stress_period:<5} | {'Time step:':<15} {time_step:<10} | {'Running ...':<10}"
        # print(msg)
        # mf6 transport loop block
        for sln in range(1, self.nsln + 1):
            # if self.fixed_components is not None and modelnmes[sln-1] in self.fixed_components:
            #     print(f'not transporting {modelnmes[sln-1]}')
            #     continue

            # set iteration counter
            kiter = 0
            # max number of solution iterations
            max_iter = self.get_value(
                self.get_var_address("MXITER", f"SLN_{sln}")
            )  # FIXME: not sure to define this inside the loop
            self.prepare_solve(sln)

            sol_start = datetime.now()
            while kiter < max_iter:
                convg = self.solve(sln)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    break
                kiter += 1
            if not convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                print(
                    "\nTransport stress period: {0} --- time step: {1} --- did not converge with {2} iters --- took {3:10.5G} mins".format(
                        stress_period, time_step, kiter, td
                    )
                )
                self.num_fails += 1
            try:
                self.finalize_solve(sln)
            except:
                pass
        td = (datetime.now() - sol_start).total_seconds() / 60.0
        print(
            f"{'Transport':<15} | {'Stress period:':<15} {stress_period:<5} | "
            f"{'Time step:':<15} {time_step:<10} | {'Completed in :':<10}  {td//60:.0f} min {td%60:10.2e} sec"
        )

    def _check_num_fails(self):
        if self.num_fails > 0:
            print("\nMODFLOW 6 failed to converge {0} times \n".format(self.num_fails))
        else:
            print("\nMODFLOW 6 converged successfully without any fails")

    @property
    def grid_type(self) -> str:
        """Return the grid type of the MODFLOW 6 model."""
        mf6 = self.sim.get_model(self.sim.model_names[0])
        distype = mf6.get_grid_type().name
        return distype
