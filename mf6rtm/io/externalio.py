"""externalio to write outputs and to write and read phreeqcrm
inputs from layered txt files.
"""
import os
import numpy as np
import pandas as pd
from mf6rtm.simulation.mf6api import Mf6API
from mf6rtm.simulation.discretization import grid_dimensions, total_cells_in_grid
from mf6rtm.config.yaml_reader import load_yaml_to_phreeqcrm
from mf6rtm.config.config import MF6RTMConfig
from mf6rtm.utils.utils import get_indices

ic_position = {
    'equilibrium_phases': 1,
    'exchange_phases': 2,
    'surface_phases': 3,
    'gas_phases': 4,
    'solid_solution_phases': 5,
     'kinetic_phases':6,
}

class Regenerator:
    """
    A class to regenerate a Mup3d object from a script file.
    """
    def __init__(self, wd='.', phinp='phinp.dat',
                 yamlfile='mf6rtm.yaml', dllfile='libmf6.dll'):
        """
        Initialize the Regenerator with the working directory and phinp file.

        Parameters:
            wd (str): Working directory where the phinp file is located.
            phinp (str): Name of the phinp file.
            yamlfile (str): Name of the YAML file to be used.
        """
        self.wd = os.path.abspath(wd)
        self.yamlfile = os.path.join(self.wd, yamlfile)
        self.phinp = phinp
        self.config = MF6RTMConfig.from_toml_file(os.path.join(self.wd, 'mf6rtm.toml')).to_dict()
        self.grid_shape = grid_dimensions(Mf6API(self.wd, os.path.join(self.wd, dllfile)))
        self.nlay = self.grid_shape[0]
        self.nxyz = total_cells_in_grid(Mf6API(self.wd, os.path.join(self.wd, dllfile)))

        # self.validate_external_files()

    @classmethod
    def regenerate_from_external_files(cls, wd='.',
                                       phinpfile='phinp.dat',
                                       yamlfile='mf6rtm.yaml',
                                       dllfile='libmf6.dll',
                                       prefix='_'):
        """
        Class method to execute the regeneration process.
        """
        instance = cls(
            wd=wd,
            phinp=phinpfile,
            yamlfile=yamlfile,
            dllfile=dllfile

        )
        instance.write_new_script(filename=f"{prefix}{phinpfile}")
        instance.update_yaml(filename=f"{prefix}{yamlfile}")
        return instance

    def validate_external_files(self):
        """
        Validate the existence of external files required for regeneration.
        """
        phinp_path = os.path.join(self.wd, self.phinp)
        if not os.path.exists(phinp_path):
            raise FileNotFoundError(f"Required file '{self.phinp}' not found in working directory '{self.wd}'.")

        for key, value in self.config.items():
            if key not in ['reactive', 'emulator']:
                print(key)
                if 'names' in self.config[key]:
                    names = self.config[key]['names']
                else:
                    raise ValueError(f"Key '{key}' does not have 'names' attribute.")
                for nme in names:
                    for lay in range(self.nlay):
                        file_path = os.path.join(self.wd, f"{key}.{nme}.m0.layer{lay+1}.txt")
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Required file '{file_path}' for key '{key}' not found in working directory '{self.wd}'.")

    def read_phinp(self):
        with open(os.path.join(self.wd, self.phinp), 'r') as f:
            script = f.readlines()
        return script

    def get_solution_blocks(self, script):
        """
        Extract solution blocks from the script.
        """
        block = []
        in_postfix = False
        for line in script:
            if line.startswith('EQUILIBRIUM') or line.startswith('KINETIC') or line.startswith('EXCHANGE'):
                in_postfix = False
            if line.startswith('SOLUTION'):
                in_postfix = True
            if in_postfix:
                block.append(line)
        # set new attibute named self.solution_blocks
        self.solution_blocks = block
        return block

    def update_yaml(self, filename='_mf6rtm.yaml'):
        """Update the YAML file with the regenerated script and initial conditions.
        """
        yamlphreeqcrm, ic1 = load_yaml_to_phreeqcrm(self.yamlfile)
        ic1 = ic1.reshape(7, self.nxyz).T
        ic1_phases = np.reshape(np.arange(1, self.nxyz + 1), self.nxyz)

        phases = [i for i in self.config.keys() if 'phases' in i]

        for phase in phases:
            i = ic_position[phase]
            ic1[:, i] = ic1_phases

        ic1_flatten = ic1.flatten('F')

        status = yamlphreeqcrm.YAMLRunFile(True, True, True, os.path.basename(self.regenerated_phinp))
        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = yamlphreeqcrm.YAMLRunString(True, False, True, input)
        yamlphreeqcrm.YAMLAddOutputVars("AddOutputVars", "true")

        status = yamlphreeqcrm.YAMLFindComponents()
        status = yamlphreeqcrm.YAMLInitialPhreeqc2Module(ic1_flatten)
        status = yamlphreeqcrm.YAMLRunCells()
        # Initial equilibration of cells
        time = 0.0
        status = yamlphreeqcrm.YAMLSetTime(time)

        fdir = os.path.join(self.wd, filename)
        status = yamlphreeqcrm.WriteYAMLDoc(fdir)

        self.yamlfile = filename
        return ic1_flatten

    def get_postfix_block(self, script):
        """
        Extract the postfix block from the script.
        """
        postfix_block = []
        in_postfix = False
        for line in script:
            if line.startswith('SELECTED_OUTPUT'):
                in_postfix = True
            if in_postfix:
                postfix_block.append(line)
            # if line.strip() == 'END':
                # in_postfix = False
        postfix_block = ['PRINT\n'] + postfix_block  # Ensure it starts with 'PRINT\n'
        self.postfix_blocks = postfix_block
        # + ''.join(postfix_block).strip()
        return postfix_block

    def generate_new_script(self):
        """
        Generate a new script based on the existing script and configuration.
        """
        script = self.read_phinp()
        solution_blocks = self.get_solution_blocks(script)
        postfix_block = self.get_postfix_block(script)

        # Create a new script with the solution blocks and postfix block
        new_script = []
        new_script.extend(solution_blocks)

        # Add equilibrium phases, kinetic phases, and exchange blocks
        sim_blocks = [key for key in self.config.keys() if key != 'reactive']
        block_generators = {
            "equilibrium_phases": self.generate_equilibrium_phases_blocks,
            "kinetic_phases": self.generate_kinetic_phases_blocks,
            "exchange_phases": self.generate_exchange_phases_blocks
        }
        for block in sim_blocks:
            generator = block_generators.get(block)
            if generator is not None:
                new_script.extend(generator())
        new_script.extend(postfix_block)
        self.regenerated_script = ''.join(new_script).strip()
        return self.regenerated_script

    def write_new_script(self, filename='_phinp.dat'):
        """
        Write the regenerated script to a file.
        """
        if not hasattr(self, 'regenerated_script'):
            self.generate_new_script()
        with open(os.path.join(self.wd, filename), 'w') as f:
            f.write(self.regenerated_script)
        # print(f"New script written to {os.path.join(self.wd, filename)}")
        self.regenerated_phinp = os.path.join(self.wd, filename)
        return self.regenerated_phinp

    def generate_equilibrium_phases_blocks(self):
        """
        Generate equilibrium phases blocks from the config.
        """
        self.add_m0_to_config()
        equilibrium_phases = self.config.get('equilibrium_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"EQUILIBRIUM_PHASES {i_phase}\n"
            for nme in equilibrium_phases['names']:
                si = equilibrium_phases.get(f'si', None).get(nme, None)
                m0 = equilibrium_phases.get(f'm0', None).get(nme, None).flatten()
                block += f"    {nme} {si:.5e} {m0[i_phase-1]:.5e}\n"
            block += "END\n"
            blocks.append(block)
        self.equilibrium_phases_blocks = blocks
        return blocks

    def generate_kinetic_phases_blocks(self):
        """
        Generate kinetic phases blocks from the config.
        """
        self.add_m0_to_config()
        kinetic_phases = self.config.get('kinetic_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"KINETICS {i_phase}\n"
            for nme in kinetic_phases['names']:
                # Get parameters for this kinetic phase
                parms = kinetic_phases.get('parms', {}).get(nme, [])
                m0 = kinetic_phases.get('m0', {}).get(nme, None).flatten()
                # Start the kinetic phase line with name and initial moles
                block += f"    {nme}\n"
                block += f"        -m0 {m0[i_phase-1]:.5e}\n"
                # Add parameters if they exist
                if parms:
                    parms_str = " ".join([f"{p:.5e}" for p in parms])
                    block += f"        -parms {parms_str}\n"
                # Add formula if it exists
                formula = kinetic_phases.get('formula', {}).get(nme, None)
                if formula:
                    block += f"        -formula {formula}\n"
            block += "END\n"
            blocks.append(block)
        self.kinetic_phases_blocks = blocks
        return blocks

    def generate_exchange_phases_blocks(self):
        """
        Generate exchange blocks from the config.
        """
        self.add_m0_to_config()
        exchange = self.config.get('exchange_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"EXCHANGE {i_phase}\n"
            for nme in exchange['names']:
                m0 = exchange.get('m0', {}).get(nme, None).flatten()
                block += f"    {nme} {m0[i_phase-1]:.5e}\n"
            # Hard code equilibrate 1 as requested
            block += "    -equilibrate 1\n"
            block += "END\n"
            blocks.append(block)
        self.exchange_blocks = blocks
        return blocks

    def read_external_files(self):
        """
        Read the external files required for regeneration using numpy.
        Returns a dictionary with the loaded arrays organized by key, name, and layer.
        """
        grid_type = ...

        file_data = {}
        # Read phase files following the same logic as validate_external_files
        for key, value in self.config.items():
            if key not in ['reactive', 'emulator']:
                if 'names' not in self.config[key]:
                    print(f"Warning: Key '{key}' does not have 'names' attribute, skipping.")
                    continue
                names = self.config[key]['names']
                file_data[key] = {}

                for nme in names:
                    layer_arrays = []

                    # Load all layers for this name
                    for lay in range(self.nlay):
                        file_path = os.path.join(self.wd, f"{key}.{nme}.m0.layer{lay+1}.txt")

                        if os.path.exists(file_path):
                            try:
                                # Load the array using numpy
                                array_data = np.loadtxt(file_path)
                                layer_arrays.append(array_data)
                            except Exception as e:
                                print(f"Warning: Could not load file {file_path}: {e}")
                                layer_arrays.append(None)
                        else:
                            print(f"Warning: File {file_path} does not exist")
                            layer_arrays.append(None)

                    # Merge layers and reshape using grid dimensions
                    if any(arr is not None for arr in layer_arrays):
                        try:
                            # Filter out None values and stack the arrays
                            valid_arrays = [arr for arr in layer_arrays if arr is not None]
                            if valid_arrays:
                                # Stack arrays along the first axis (layers)
                                merged_array = np.stack(valid_arrays, axis=0)

                                # Reshape to grid dimensions
                                reshaped_array = merged_array.reshape(self.grid_shape)

                                file_data[key][nme] = reshaped_array
                            else:
                                file_data[key][nme] = None
                                print(f"Warning: No valid arrays found for {nme}")
                        except Exception as e:
                            raise RuntimeError(
                                f"Could not merge/reshape arrays for {nme}"
                            ) from e
                    else:
                        file_data[key][nme] = None
                        print(f"Warning: No arrays loaded for {nme}")

        # Store the loaded data as an instance attribute
        self.file_data = file_data
        return file_data

    def add_m0_to_config(self):
        """
        Add the loaded array data to the config dictionary.
        This method should be called after read_external_files().
        """
        if not hasattr(self, 'file_data'):
            self.read_external_files()

        # Add phase array data to config
        for key in self.file_data:
            if key != 'phinp' and key in self.config:
                # Add arrays section to each phase type
                if 'm0' not in self.config[key]:
                    self.config[key]['m0'] = {}

                self.config[key]['m0'] = self.file_data[key]
                # print(f"Added m0 data for {key} to config")

        return self.config

class SelectedOutput:
    def __init__(self, mf6rtm):
        self.mf6rtm = mf6rtm
        self.phreeqcbmi = mf6rtm.phreeqcbmi
        self.mf6api = mf6rtm.mf6api
        self.sout_fname = "sout.csv"
        self.get_selected_output_on = True

    def write_ml_arrays(self, conc_array, iter,
                        add_var_names=None,
                        fname="_features.csv") -> None:
        """
        Write total transported component concentrations in mol/L
        (+ optional vars in mol/L) to CSV for Machine Learning arrays.

        Parameters
        ----------
        conc_array : array-like
            Main concentrations (ncomps x nxyz).
        add_var_names : list of str, optional
            Extra PHREEQC variables to include.
        fname : str
            Output filename (relative to model wd).
        """

        # Base arrays and labels
        cols = ["time", "cell", "saturation"] + list(self.phreeqcbmi.components)
        arrays = [
            np.full((self.mf6rtm.nxyz, 1), self.mf6rtm.ctime),
            np.arange(self.mf6rtm.nxyz).reshape(-1, 1),
            self.mf6rtm.get_saturation_from_mf6().reshape(-1, 1),
            np.reshape(conc_array, (self.phreeqcbmi.ncomps, self.mf6rtm.nxyz)).T
        ]

        # Optional PHREEQC selected outputs
        if add_var_names:
            col_idx = [self.phreeqcbmi.soutdf.columns.get_loc(c) for c in add_var_names]
            sout = self.phreeqcbmi.GetSelectedOutput().reshape(-1, self.mf6rtm.nxyz)
            arrays.append(sout[col_idx, :].T)
            cols.extend(add_var_names)

        arr = np.hstack(arrays)
        header_str = ",".join(cols)

        # Write
        fmt = ["%.6f", "%d"] + ["%.10e"] * (arr.shape[1] - 2)
        fname = os.path.join(self.mf6rtm.wd, fname)

        # flag for writing headers
        write_header = False
        if iter == 0:
            write_header = True
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

        with open(fname, "a") as f:
            np.savetxt(f, arr, delimiter=",",
                    header=header_str if write_header else "",
                    comments="", fmt=fmt)

    def _update_selected_output(self) -> None:
        """Update the selected output dataframe and save to attribute"""
        self._get_selected_output()
        updf = pd.concat(
            [
                self.phreeqcbmi.soutdf.astype(self._current_soutdf.dtypes),
                self._current_soutdf,
            ]
        )
        self._update_soutdf(updf)

    def __replace_inactive_cells_in_sout(self, sout, diffmask):
        """Function to replace inactive cells in the selected output dataframe"""
        # match headers in components closest string

        inactive_idx = get_indices(0, diffmask)

        sout[:, inactive_idx] = self._sout_k[:, inactive_idx]
        return sout

    def _get_selected_output(self) -> None:
        """Get the selected output from phreeqc bmi and replace skipped reactive cells with previous conc"""
        # selected ouput
        self.phreeqcbmi.set_scalar("NthSelectedOutput", 0)
        sout = self.phreeqcbmi.GetSelectedOutput()
        sout = sout.reshape(-1, self.mf6rtm.nxyz)

        if self.mf6rtm._check_inactive_cells_exist(self.mf6rtm.diffmask) and hasattr(self, "_sout_k"):
            sout = self.__replace_inactive_cells_in_sout(sout, self.mf6rtm.diffmask)
        self._sout_k = sout  # save sout to a private attribute
        # add time to selected ouput
        sout[0] = np.ones_like(sout[0]) * (self.mf6rtm.ctime + self.mf6rtm.time_step)
        df = pd.DataFrame(columns=self.phreeqcbmi.soutdf.columns)
        for col, arr in zip(df.columns, sout):
            df[col] = arr
        self._current_soutdf = df

    def _update_soutdf(self, df: pd.DataFrame) -> None:
        """Update the selected output dataframe to phreeqcrm object"""
        self.phreeqcbmi.soutdf = df

    def _check_sout_exist(self) -> bool:
        """Check if selected output file exists"""
        return os.path.exists(os.path.join(self.mf6rtm.wd, self.sout_fname))

    def _write_sout_headers(self) -> None:
        """Write selected output headers to a file"""
        with open(os.path.join(self.mf6rtm.wd, self.sout_fname), "w") as f:
            f.write(",".join(self.phreeqcbmi.sout_headers))
            f.write("\n")

    def _rm_sout_file(self) -> None:
        """Remove the selected output file"""
        try:
            os.remove(os.path.join(self.mf6rtm.wd, self.sout_fname))
        except:
            pass

    def _append_to_soutdf_file(self) -> None:
        """Append the current selected output to the selected output file"""
        assert not self._current_soutdf.empty, "current sout is empty"
        self._current_soutdf.to_csv(
            os.path.join(self.mf6rtm.wd, self.sout_fname), mode="a", index=False, header=False
        )

    def _export_soutdf(self) -> None:
        """Export the selected output dataframe to a csv file"""
        self.phreeqcbmi.soutdf.to_csv(
            os.path.join(self.mf6rtm.wd, self.sout_fname), index=False
        )
