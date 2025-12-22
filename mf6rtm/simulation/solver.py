"""The solver module provides the Mf6RTM class that couples modflowapi and
phreeqcrm, along with functions to run the coupled simulations.
"""
import os
# import warnings
import numpy as np

from datetime import datetime
from typing import Any, Union
from pathlib import Path

from PIL import Image
from mf6rtm.simulation.mf6api import Mf6API
from mf6rtm.simulation.phreeqcbmi import PhreeqcBMI
from mf6rtm.simulation.discretization import total_cells_in_grid
from mf6rtm.config.config import MF6RTMConfig
from mf6rtm.io.externalio import SelectedOutput
from mf6rtm.utils import utils

# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# global variables
DT_FMT = "%Y-%m-%d %H:%M:%S"

time_units_dict = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
    "years": 31536000,
    "unknown": 1,  # if unknown assume seconds
}

def check_config_file(wd: os.PathLike) -> tuple[os.PathLike, os.PathLike]:
    assert os.path.exists(
        os.path.join(wd, "mf6rtm.toml")
        ), "mf6rtm.toml not found in model directory"
    config_file= os.path.join(wd, "mf6rtm.toml")
    config = MF6RTMConfig.from_toml_file(config_file)
    return config

def check_nam_files(wd:os.PathLike) -> tuple[os.PathLike,os.PathLike]:
    """Check if the nam files are present in the model directory"""
    nam = [f for f in os.listdir(wd) if f.endswith(".nam")]
    assert "mfsim.nam" in nam, "mfsim.nam file not found in model directory"
    # assert "gwf.nam" in nam, "gwf.nam file not found in model directory"
    return os.path.join(wd, "mfsim.nam") #, os.path.join(wd, "gwf.nam")

def prep_to_run(wd:os.PathLike) -> tuple[os.PathLike,os.PathLike]:
    """
    Prepares the model to run by checking if the model directory (wd) contains the necessary files
    and returns the path to the yaml file (phreeqcrm) and the dll file (mf6 api)

    Parameters
    ----------
    wd :os.PathLike
        The path to the working directory of model directory
    Returns
    -------
    tuple[PathLike,os.PathLike]
        The path to the phreeqcrm model file (yaml) and the path to the MODFLOW 6 dll (associated with mf6api).
    """
    # check if wd exists
    assert os.path.exists(wd), f"Path {wd} not found"
    # check if file starting with libmf6 exists
    dll = [f for f in os.listdir(wd) if f.startswith("libmf6")]
    assert len(dll) == 1, "libmf6 dll not found in model directory"
    dll = os.path.join(wd, "libmf6")

    config = check_config_file(wd)
    check_nam_files(wd)
    if config.reactive['externalio']:
        from mf6rtm.io.externalio import Regenerator
        print("WARNING: Flag for external IO mode is active")
        regcls = Regenerator.regenerate_from_external_files(wd=wd,
                                                phinpfile='phinp.dat',
                                                yamlfile='mf6rtm.yaml',
                                                dllfile=dll
                                                )
        yamlfile = regcls.yamlfile
    else:
        yamlfile = os.path.join(wd, "mf6rtm.yaml")
    assert os.path.exists(
        os.path.join(wd, yamlfile)
    ), f"{yamlfile} not found in model directory {wd}"
    return yamlfile, dll

def solve(wd:os.PathLike, reactive: Union[bool, None] = None, nthread: int = 1) -> bool:
    """Wrapper to prepare and call solve functions"""

    mf6rtm = initialize_interfaces(wd, nthread=nthread)
    if reactive is not None and isinstance(reactive, bool) and reactive != mf6rtm.reactive:
        print(
                f"Mode changed from "
                f"{'reactive' if mf6rtm.reactive else 'non-reactive'} to "
                f"{'reactive' if reactive else 'non-reactive'}\n"
            )
        mf6rtm._set_reactive(reactive)
        # let mf6 manage this for conservative runs
        mf6rtm.selected_output.get_selected_output_on = False
    mf6rtm.print_warning_user_active()
    success = mf6rtm._solve()
    return success


# TODO: we should maybe move this into the Mf6API as an alternative constructor
def initialize_interfaces(wd:os.PathLike, nthread: int = 1) -> Mf6API:
    """Function to initialize the interfaces for modflowapi and phreeqcrm and returns the mf6rtm object"""

    yamlfile, dll = prep_to_run(wd)

    if nthread > 1:
        # set nthreds to nthread
        set_nthread_yaml(yamlfile, nthread=nthread)

    # initialize the interfaces
    mf6api = Mf6API(wd, dll)
    phreeqcrm = PhreeqcBMI(yamlfile) #FIXME: Does not work with path like
    mf6rtm = Mf6RTM(wd, mf6api, phreeqcrm)
    return mf6rtm


def set_nthread_yaml(yamlfile:os.PathLike, nthread: int = 1) -> None:
    """Function to set the number of threads in the yaml file"""
    with open(yamlfile, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "nthreads" in line:
            lines[i] = f"  nthreads: {nthread}\n"
    with open(yamlfile, "w") as f:
        f.writelines(lines)
    return


class Mf6RTM(object):
    def __init__(
        self,
        wd:os.PathLike,
        mf6api: Mf6API,
        phreeqcbmi: PhreeqcBMI,
    ) -> None:
        """
        Initialize the Mf6RTM instance with specified working directory, MF6API,
        and PhreeqcBMI instances.

        Parameters
        ----------
        wd :os.PathLike
            The working directory path for the model.
        mf6api : Mf6API
            An instance of the Mf6API class, representing the Modflow 6 API.
        phreeqcbmi : PhreeqcBMI
            An instance of the PhreeqcBMI class, representing the PHREEQC BMI.

        Attributes
        ----------
        mf6api : Mf6API
            The Modflow 6 API instance.
        phreeqcbmi : PhreeqcBMI
            The PHREEQC BMI instance.
        charge_offset : float
            Offset for charge, initialized to 0.0.
        wd :os.PathLike
            The working directory path.
        sout_fname : str
            Filename for the output, default is "sout.csv".
        reactive : bool
            Flag indicating if the model is reactive, default is True.
        epsaqu : float
            ??Epsaqueous value??, initialized to 0.0.
        fixed_components : Any
            Fixed components, default is None.
        get_selected_output_on : bool
            Flag indicating if selected output is on, default is True.
        component_model_dict : dict[str, str]
            Dictionary mapping PHREEQC aqueous chemical components to their
            corresponding Modflow 6 groundwater transport (gwt6) model names.
        conservative_transport_models: list[str]
            List of Modflow 6 groundwater transport (gwt6) model not coupled
            with PhreeqcRM
        nxyz : int
            Total number of cells in the grid.
        """
        assert isinstance(mf6api, Mf6API), "MF6API must be an instance of Mf6API"
        assert isinstance(
            phreeqcbmi, PhreeqcBMI
        ), "PhreeqcBMI must be an instance of PhreeqcBMI"
        self.mf6api = mf6api
        self.phreeqcbmi = phreeqcbmi
        self.charge_offset = 0.0
        self.wd = Path(wd)
        self.epsaqu = 0.0
        self.fixed_components = None
        self.selected_output = SelectedOutput(self)

        # set component model dictionary & list of conservative_transport_models
        self.component_model_dict, self.conservative_transport_models = self._create_component_model_dict()

        # set discretization
        self.nxyz = total_cells_in_grid(self.mf6api)
        # set time conversion factor
        self.set_time_conversion()

        self.config = MF6RTMConfig.from_toml_file(self.wd/"mf6rtm.toml")
        self.reactive = self.config.reactive_enabled
        self.set_emulator_training()

    def set_emulator_training(self) -> None:
        """
        Configure emulator training output.

        Reads ``emulator_training_data`` from the configuration. If enabled,
        sets up emulator output variables; otherwise disables training data.

        Attributes
        ----------
        ml_output : bool
            Whether emulator training data output is enabled.

        Returns
        -------
        None
        """
        self.ml_output = bool(getattr(self.config, "emulator_training_data", False))

        if self.ml_output:
            self.set_emulator_output_add_variables()
            print("Saving emulator training data for surrogating")


    def set_emulator_output_add_variables(self) -> None:
        """
        Add emulator target and feature variables to the output.

        Updates ``selected_output`` with variables defined in the configuration.
        Defaults to empty lists if not provided.

        Attributes
        ----------
        selected_output.target_var : list of str
            Target variables for emulator training.
        selected_output.feat_var : list of str
            Feature variables for emulator training.

        Returns
        -------
        None
        """
        self.selected_output.target_var = getattr(
            self.config, "emulator_target_variables", []
        )
        self.selected_output.feat_var = getattr(
            self.config, "emulator_feature_variables", []
        )

    def print_warning_user_active(self):
        """
        Prints a warning if reaction timing is set to 'user'.
        """
        if self.config.reactive_timing == 'user':
            print(f"WARNING: Running reaction only in the following periods and time steps:")
            for period, timestep in self.config.reactive_tsteps:
                print(f"  Period {period}, Time step {timestep}")
        else:
            return

    def get_saturation_from_mf6(self) -> dict[Any, np.ndarray]:
        """
        Get the saturation

        Parameters
        ----------
        mf6 (modflowapi): the modflow api object

        Returns
        -------
        array: the saturation
        """
        sat = {
            component: self.mf6api.get_value(
                self.mf6api.get_var_address(
                    "FMI/GWFSAT",
                    f"{self.component_model_dict[component]}"
                )
            )
            for component in self.phreeqcbmi.components
        }
        # select the first component to get the length of the array
        sat = sat[
            self.phreeqcbmi.components[0]
        ]  # saturation is the same for all components
        self.phreeqcbmi.sat_now = sat  # set phreeqcmbi saturation
        return sat

    def get_time_units_from_mf6(self) -> str:
        """Function to get the time units from mf6"""
        return self.mf6api.sim.tdis.time_units.get_data()

    def set_time_conversion(self) -> None:
        """Function to set the time conversion factor"""
        time_units = self.get_time_units_from_mf6()
        self.time_conversion = 1.0 / time_units_dict[time_units]
        self.phreeqcbmi.SetTimeConversion(self.time_conversion)

    def _create_component_model_dict(self)-> tuple[dict[str, str], list[str]]:
        """
        Create a dictionary of PHREEQC aqueous chemical component names and
        their corresponding Modflow 6 Groundwater Transport (GWT) model names.

        Returns
        -------
        component_model_dict : dict[str, str]
            A dictionary where the keys are the component names and the values are
            the corresponding transport model names.
        """
        components = self.phreeqcbmi.get_value_ptr("Components")
        # convert np.array to list of pure python strings
        components = [str(component) for component in components]

        gwt_model_names = [
            name for name in self.mf6api.sim.model_names
            if (self.mf6api.sim.get_model(name).model_type == 'gwt6')
        ]
        gwt_name_prefix = longest_common_substring(gwt_model_names)

        component_model_dict = dict(zip(components, [None]*len(components)))
        for component in components:
            for model_name in gwt_model_names:
                if model_name.replace(gwt_name_prefix, "").lower() == component.lower():
                    component_model_dict[component] = model_name
            if (component.lower() == 'charge') and (component_model_dict[component] == None):
                for model_name in gwt_model_names:
                    if model_name.replace(gwt_name_prefix, "").lower() == 'ch':
                        component_model_dict[component] = model_name
            assert (component_model_dict[component] != None,
                f"Component {component} is not matched with a transport model"
            )

        conservative_transport_models = list(
            set(gwt_model_names) - set(component_model_dict.values())
        )

        return component_model_dict, conservative_transport_models

    # TODO: remove or have raise not implemented error
    def _set_fixed_components(self, fixed_components): ...

    # TODO: make reactive a property
    def _set_reactive(self, reactive: bool) -> None:
        """Set the model to run only transport or transport and reactions"""
        self.reactive = reactive

    def _prepare_to_solve(self) -> None:
        """Prepare the model to solve"""
        # check if sout fname exists
        if self.selected_output._check_sout_exist():
            # if found remove it
            self.selected_output._rm_sout_file()

        self.mf6api._prepare_mf6()
        self.phreeqcbmi._prepare_phreeqcrm_bmi()

        # get and write sout headers
        self.selected_output._write_sout_headers()

    def _set_ctime(self) -> float:
        """Set the current time of the simulation from mf6api"""
        self.ctime = self.mf6api.get_current_time()
        self.phreeqcbmi._set_ctime(self.ctime)
        return self.ctime

    def _set_etime(self) -> float:
        """Set the end time of the simulation from mf6api"""
        self.etime = self.mf6api.get_end_time()
        return self.etime

    def _set_time_step(self) -> float:
        self.time_step = self.mf6api.get_time_step()
        return self.time_step

    def _finalize(self) -> None:
        """Finalize the APIs"""
        self._finalize_mf6api()
        self._finalize_phreeqcrm()

    def _finalize_mf6api(self) -> None:
        """Finalize the mf6api"""
        self.mf6api.finalize()

    def _finalize_phreeqcrm(self) -> None:
        """Finalize the phreeqcrm api"""
        self.phreeqcbmi.finalize()


    def _get_cdlbl_vect(self) -> np.ndarray[np.float64]:
        """Get the concentration array from phreeqc bmi reshape to (ncomps, nxyz)"""
        c_dbl_vect = self.phreeqcbmi.GetConcentrations()

        conc = [
            c_dbl_vect[i : i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)
        ]  # reshape array
        # TODO: refactor to use np.reshape(), which is 2x faster
        return conc

    def _set_conc_at_current_kstep(self, c_dbl_vect: np.ndarray[np.float64]):
        """Saves the current concentration array to the object"""
        self.current_iteration_conc = np.reshape(
            c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz)
        )

    def _set_conc_at_previous_kstep(self, c_dbl_vect: np.ndarray[np.float64]):
        """Saves the current concentration array to the object"""
        self.previous_iteration_conc = np.reshape(
            c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz)
        )

    def _transfer_array_to_mf6(self) -> np.ndarray[np.float64]:
        """Transfer the concentration array to mf6"""
        c_dbl_vect = self._get_cdlbl_vect()

        # check if reactive cells were skipped due to small changes from transport and replace with previous conc
        if self._check_previous_conc_exists() and self._check_inactive_cells_exist(
            self.diffmask
        ):
            c_dbl_vect = self._replace_inactive_cells(c_dbl_vect, self.diffmask)
        else:
            pass

        conc_dict = {}
        for i, c in enumerate(self.phreeqcbmi.components):
            conc_dict[c] = c_dbl_vect[i]
            # Set concentrations in mf6
            gwt_model_name = self.component_model_dict[c]
            if gwt_model_name.lower() == "charge":
                self.mf6api.set_value(
                    f"{gwt_model_name.upper()}/X",
                    utils.concentration_l_to_m3(conc_dict[c]) + self.charge_offset,
                )
            else:
                self.mf6api.set_value(
                    f"{gwt_model_name.upper()}/X",
                    utils.concentration_l_to_m3(conc_dict[c]),
                )
        return c_dbl_vect

    def _check_previous_conc_exists(self) -> bool:
        """Function to replace inactive cells in the concentration array"""
        # check if self.previous_iteration_conc is a property
        return hasattr(self, "previous_iteration_conc")

    def _check_inactive_cells_exist(self, diffmask: np.ndarray[np.float64]) -> bool:
        """Function to check if inactive cells exist in the concentration array"""
        inact = utils.get_indices(0, diffmask)
        return len(inact) > 0

    def _replace_inactive_cells(
        self,
        c_dbl_vect: np.ndarray[np.float64],
        diffmask: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        """Function to replace inactive cells in the concentration array"""
        c_dbl_vect = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))
        # get inactive cells
        inactive_idx = [
            utils.get_indices(0, diffmask) for k in range(self.phreeqcbmi.ncomps)
        ]
        c_dbl_vect[:, inactive_idx] = self.previous_iteration_conc[:, inactive_idx]
        c_dbl_vect = c_dbl_vect.flatten()
        conc = [
            c_dbl_vect[i : i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)
        ]
        return conc

    def _transfer_array_to_phreeqcrm(self) -> np.ndarray[np.float64]:
        """Transfer the concentration array to phreeqc bmi"""
        mf6_conc_array = []
        for c in self.phreeqcbmi.components:
            if c.lower() == "charge":
                mf6_conc_array.append(
                    utils.concentration_m3_to_l(
                        self.mf6api.get_value(
                            self.mf6api.get_var_address(
                                "X",
                                f"{self.component_model_dict[c].upper()}",
                            )
                        )
                        - self.charge_offset
                    )
                )

            else:
                mf6_conc_array.append(
                    utils.concentration_m3_to_l(
                        self.mf6api.get_value(
                            self.mf6api.get_var_address(
                                "X",
                                f"{self.component_model_dict[c].upper()}",
                            )
                        )
                    )
                )
        c_dbl_vect = np.reshape(mf6_conc_array, self.nxyz * self.phreeqcbmi.ncomps)
        self.phreeqcbmi.SetConcentrations(c_dbl_vect)

        # set the kper and kstp
        self.phreeqcbmi._get_kper_kstp_from_mf6api(
            self.mf6api
        )  # FIXME: calling this func here is not ideal

        return c_dbl_vect

    def _solve(self) -> bool:
        """Alias for the solve method to provide backward compatibility"""
        return self.solve()

    def is_reactive_tstep(self) -> bool:
        """
        Check if the current timestep should be reactive based on configuration.

        Returns:
            bool: True if current timestep should be reactive, False otherwise
        """
        # Early return if not in reactive mode

        if not self.reactive:
            return False

        # Get current timestep
        current_tstep = [self.mf6api.kper, self.mf6api.kstp]

        # Check strategy
        if self.config.reactive_timing == 'all':
            return True
        elif self.config.reactive_timing == 'user':
            return current_tstep in self.config.reactive_tsteps
        else:
            # Handle unknown strategy
            print(f"Warning: Unknown strategy '{self.config.reactive_timing}'. Defaulting to reactive.")
            return True
    def set_kiter(self) -> int:
        if hasattr(self, "kiter"):
            self.kiter += 1
        else:
            self.kiter = 0
        return self.kiter
    def solve(self) -> bool:
        """Solve the model"""
        success = False  # initialize success flag
        sim_start = datetime.now()
        self._prepare_to_solve()

        # check sout was created
        assert self.selected_output._check_sout_exist(), f"{self.selected_output.sout_fname} not found"

        print("Starting Solution at {0}".format(sim_start.strftime(DT_FMT)))
        ctime = self._set_ctime()
        etime = self._set_etime()
        while ctime < etime:
            # self iteration counter
            self.set_kiter()
            # length of the current solve time
            dt = self._set_time_step()
            self.mf6api.prepare_time_step(dt)
            self.mf6api._solve_gwt()

            # get saturation
            self.get_saturation_from_mf6()
            # check_reactive_kstp()
            if self.is_reactive_tstep():
                c_dbl_vect = self._transfer_array_to_phreeqcrm()
                self._set_conc_at_current_kstep(c_dbl_vect)

                # Export ML feature arrays if option is on
                if self.ml_output:
                    self.selected_output.write_ml_arrays(self.current_iteration_conc,
                                                    self.kiter,
                                                    add_var_names=self.selected_output.feat_var,
                                                    fname='_features.csv'
                                                )

                if ctime == 0.0:
                    self.diffmask = np.ones(self.nxyz)
                else:
                    diffmask = get_conc_change_mask(
                        self.current_iteration_conc,
                        self.previous_iteration_conc,
                        self.phreeqcbmi.ncomps,
                        self.nxyz,
                        treshold=self.epsaqu,
                    )
                    self.diffmask = diffmask
                # solve reactions
                self.phreeqcbmi._solve_phreeqcrm(dt, diffmask=self.diffmask)
                c_dbl_vect = self._transfer_array_to_mf6()

                self._set_conc_at_previous_kstep(c_dbl_vect)

            self.mf6api.finalize_time_step()
            ctime = self._set_ctime()  # update the current time tracking
            if self.selected_output.get_selected_output_on:
                # get sout and update df
                self.selected_output._update_selected_output()
                # append current sout rows to file
                self.selected_output._append_to_soutdf_file()
                # Export ML target arrays if option is on
                if self.ml_output:
                    self.selected_output.write_ml_arrays(self.previous_iteration_conc,
                                            self.kiter,
                                            add_var_names=self.selected_output.target_var,
                                            fname='_targets.csv'
                                                )

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0

        self.mf6api._check_num_fails()

        # Clean up and close api objs
        try:
            self._finalize()
            success = True
            print(mrbeaker())
            print(
                "\nMR BEAKER IMPORTANT MESSAGE: MODEL RUN FINISHED BUT CHECK THE RESULTS .. THEY ARE PROLY RUBBISH\n"
            )
        except:
            print("MR BEAKER IMPORTANT MESSAGE: SOMETHING WENT WRONG. BUMMER\n")
            pass
        print(
            "Solution finished at {0}. Running time: {1:10.5G} mins".format(
                sim_end.strftime(DT_FMT), td
            )
        )
        return success


def get_less_than_zero_idx(arr):
    """Function to get the index of all occurrences of <0 in an array"""
    idx = np.where(arr < 0)
    return idx


def get_inactive_idx(arr: np.ndarray, val: float = 1e30):
    """Function to get the index of all occurrences of <0 in an array"""
    idx = list(np.where(arr >= val)[0])
    return idx


def get_conc_change_mask(
    ci: np.ndarray[np.float64],
    ck: np.ndarray[np.float64],
    ncomp: int,
    nxyz: int,
    treshold: float = 1e-10,
) -> np.ndarray[np.float64]:
    """Function to get the active-inactive cell mask for concentration change to inform phreeqc which cells to update"""
    # reshape arrays to 2D (nxyz, ncomp)
    ci = ci.reshape(nxyz, ncomp)
    ck = ck.reshape(nxyz, ncomp) + 1e-30

    # get the difference between the two arrays and divide by ci
    diff = np.abs((ci - ck.reshape(-1 * nxyz, ncomp)) / ci) < treshold
    diff = np.where(diff, 0, 1)
    diff = diff.sum(axis=1)

    # where values <0 put -1 else 1
    diff = np.where(diff == 0, 0, 1)
    return diff


def longest_common_substring(strings):
    """Function to find the longest common substring of a list of strings
    Used here to find the common "stem" of the GWT model names for matching
    with PhreeqcRM components.
    """
    if not strings:
        return ""

    # Start with the first string as a reference
    reference_string = strings[0]
    longest_lcs = ""

    # Iterate through all possible substrings of the reference string
    for i in range(len(reference_string)):
        for j in range(i + 1, len(reference_string) + 1):
            current_substring = reference_string[i:j]

            # Check if this substring exists in all other strings
            is_common = True
            for other_string in strings[1:]:
                if current_substring not in other_string:
                    is_common = False
                    break

            # If it's common and longer than the current longest, update
            if is_common and len(current_substring) > len(longest_lcs):
                longest_lcs = current_substring

    return longest_lcs


def mrbeaker() -> str:
    """ASCII art of Mr. Beaker"""

    from mf6rtm.assets import mrbeaker_path

    mr_beaker_image = Image.open(mrbeaker_path())

    # Resize the image to fit the terminal width
    terminal_width = 70  # Adjust this based on your terminal width
    aspect_ratio = mr_beaker_image.width / mr_beaker_image.height
    terminal_height = int(terminal_width / aspect_ratio * 0.5)
    mr_beaker_image = mr_beaker_image.resize((terminal_width, terminal_height))

    # Convert the image to grayscale
    mr_beaker_image = mr_beaker_image.convert("L")

    # Convert the grayscale image to ASCII art
    ascii_chars = "%,.?>#*+=-:."

    mrbeaker = ""
    for y in range(int(mr_beaker_image.height)):
        mrbeaker += "\n"
        for x in range(int(mr_beaker_image.width)):
            pixel_value = mr_beaker_image.getpixel((x, y))
            mrbeaker += ascii_chars[pixel_value // 64]
        # mrbeaker += "\n"

    return mrbeaker

def run_cmd():
    # get the current directory
    cwd = os.getcwd()
    # run the solve function
    solve(cwd)
