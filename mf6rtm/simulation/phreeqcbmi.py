"""PHREEQC BMI wrapper for mf6rtm.

Defines :class:`PhreeqcBMI`, a thin subclass of ``phreeqcrm.BMIPhreeqcRM``
that adapts the PhreeqcRM Basic Model Interface for use inside the mf6rtm
coupling loop.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import phreeqcrm

from mf6rtm.simulation.mf6api import Mf6API
from mf6rtm.utils import utils


class PhreeqcBMI(phreeqcrm.BMIPhreeqcRM):
    """Basic Model Interface wrapper around PhreeqcRM.

    Extends ``phreeqcrm.BMIPhreeqcRM`` with helpers used by :class:`Mf6RTM`
    to run the geochemical reaction step: scalar setters, selected-output
    handling and per-timestep solution updates.

    Parameters
    ----------
    yaml : str, optional
        Path to the PhreeqcRM YAML configuration file used to initialize the
        chemistry. Default is ``"mf6rtm.yaml"``.
    """

    def __init__(self, yaml="mf6rtm.yaml"):
        phreeqcrm.BMIPhreeqcRM.__init__(self)
        print("Processing initial chemistry configuration")
        self.initialize(yaml)
        self.sat_now = None

    def get_grid_to_map(self):
        """Get the PhreeqcRM grid-to-map mapping.

        Returns
        -------
        array-like
            The grid-to-map mapping returned by ``GetGridToMap``.
        """
        return self.GetGridToMap()

    def _prepare_phreeqcrm_bmi(self):
        """Prepare phreeqc bmi for reaction calculations"""
        self.SetScreenOn(False)
        self.set_scalar("NthSelectedOutput", 0)
        sout_headers = self.GetSelectedOutputHeadings()
        soutdf = pd.DataFrame(columns=sout_headers)

        # set attributes
        self.components = self.get_value_ptr("Components")
        self.ncomps = len(self.components)
        self.soutdf = soutdf
        self.sout_headers = sout_headers

    def _set_ctime(self, ctime):
        """Set the current time in phreeqc bmi"""
        # self.ctime = self.SetTime(ctime*86400)
        self.ctime = ctime

    def set_scalar(self, var_name, value):
        """Set a scalar PhreeqcRM BMI variable.

        Parameters
        ----------
        var_name : str
            Name of the BMI variable to set.
        value : Any
            Scalar value to assign; cast to the variable's declared dtype.

        Raises
        ------
        ValueError
            If ``var_name`` does not refer to a scalar (dimension != 1).
        """
        itemsize = self.get_var_itemsize(var_name)
        nbytes = self.get_var_nbytes(var_name)
        dim = nbytes // itemsize

        if dim != 1:
            raise ValueError(f"{var_name} is not a scalar")

        vtype = self.get_var_type(var_name)
        dest = np.empty(1, dtype=vtype)
        dest[0] = value
        x = self.set_value(var_name, dest)

    def _solve_phreeqcrm(self, dt, diffmask):
        """Function to solve phreeqc bmi"""
        # status = phreeqc_rm.SetTemperature([self.init_temp[0]] * self.ncpl)
        # status = phreeqc_rm.SetPressure([2.0] * nxyz)
        self.SetTimeStep(dt * 1.0 / self.GetTimeConversion())

        if self.sat_now is not None:
            sat = self.sat_now
        else:
            sat = [1] * self.GetGridCellCount()

        # self.SetSaturation(sat)

        # update which cells to run depending on conc change between tsteps
        if diffmask is not None:
            # get idx where diffmask is 0
            inact = utils.get_indices(0, diffmask)
            if len(inact) > 0:
                for i in inact:
                    sat[i] = 0
            # print(
            #     f"{'Cells sent to reactions':<25} | {self.GetGridCellCount()-len(inact):<0}/{self.GetGridCellCount():<15}"
            # )
            self.SetSaturation(sat)

        print_selected_output_on = True
        # print_chemistry_on = False
        status = self.SetSelectedOutputOn(print_selected_output_on)
        status = self.SetPrintChemistryOn(False, False, False)
        # reactions loop
        sol_start = datetime.now()

        message = f"{'Reactions':<15} | {'Stress period:':<15} {self.kper:<5} | {'Time step:':<15} {self.kstp:<10} | {'Running ...':<10}"
        self.LogMessage(message + "\n")  # log message
        # print(message)
        # self.ScreenMessage(message)
        # status = self.RunCells()
        # if status < 0:
        #     print('Error in RunCells: {0}'.format(status))
        # Suppress PhreeqcRM screen output during RunCells to avoid per-thread
        # timing lines printed for each timestep when nthread > 1.
        self.SetScreenOn(False)  # pragma: no cover
        self.update()  # pragma: no cover
        self.SetScreenOn(True)  # pragma: no cover
        td = (datetime.now() - sol_start).total_seconds() / 60.0
        message = (
            f"{'Reactions':<15} | {'Stress period:':<15} {self.kper:<5} | "
            f"{'Time step:':<15} {self.kstp:<10} | {'Completed in :':<10}  {td // 60:.0f} min {td % 60:10.2e} sec"
        )
        self.LogMessage(message)
        print(message)
        # self.ScreenMessage(message)

    def _get_kper_kstp_from_mf6api(self, mf6api: Mf6API):
        """Function to get the kper and kstp from mf6api"""
        assert isinstance(mf6api, Mf6API), "mf6api must be an instance of Mf6API"
        self.kper = mf6api.kper
        self.kstp = mf6api.kstp
        return
