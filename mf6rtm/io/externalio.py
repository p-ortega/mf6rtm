"""externalio to write outputs and to write and read phreeqcrm
inputs from layered txt files.
"""
import os
import re

import numpy as np
import pandas as pd

from mf6rtm.config.config import MF6RTMConfig
from mf6rtm.config.yaml_reader import load_yaml_to_phreeqcrm
from mf6rtm.simulation.discretization import grid_dimensions, total_cells_in_grid
from mf6rtm.simulation.mf6api import Mf6API
from mf6rtm.utils.utils import get_indices

ic_position = {
    'equilibrium_phases': 1,
    'exchange_phases': 2,
    'surface_phases': 3,
    'gas_phases': 4,
    'solid_solution_phases': 5,
     'kinetic_phases':6,
}

# PHREEQC top-level keyword classification used when regenerating _phinp.dat.
# Matched against the EXACT first token of a column-0 keyword line (not startswith),
# so e.g. SOLUTION != SOLUTION_SPECIES and EXCHANGE != EXCHANGE_SPECIES.
_REGENERATED_KEYWORDS = {  # dropped from source; rebuilt per-cell from the config m0 files
    "EQUILIBRIUM_PHASES", "PURE_PHASES", "KINETICS", "EXCHANGE",
}
_SOLUTION_KEYWORDS = {"SOLUTION"}  # preserved verbatim
_PRESERVED_REACTION_KEYWORDS = {  # reaction-state blocks we don't regenerate -> preserved verbatim
    "SURFACE", "GAS_PHASE", "SOLID_SOLUTIONS", "REACTION",
    "REACTION_TEMPERATURE", "REACTION_PRESSURE", "MIX",
    "USE", "SAVE", "COPY", "TITLE",
}
_OUTPUT_KEYWORDS = {  # emitted last, verbatim, exactly once
    "SELECTED_OUTPUT", "USER_PUNCH", "USER_GRAPH", "USER_PRINT", "PRINT", "DUMP",
}
# Anything else that starts a block (KNOBS, RATES, PHASES, SOLUTION_SPECIES,
# *_MASTER_SPECIES, EXCHANGE_SPECIES, SURFACE_SPECIES, unknown uppercase keywords)
# is treated as a DEFINITION block and emitted at the top, in original order.

# A column-0 PHREEQC keyword: uppercase letters/underscores/digits, not indented.
_KEYWORD_RE = re.compile(r"^[A-Z][A-Z_0-9]*$")


def _ends_with_end(lines):
    """True if the last non-blank line of ``lines`` is a PHREEQC ``END``."""
    for line in reversed(lines):
        if line.strip():
            return line.strip().upper() == "END"
    return False


class Regenerator:
    """
    A class to regenerate a Mup3d object from a script file.
    """
    def __init__(self, wd='.', phinp='phinp.dat',
                 yamlfile='mf6rtm.yaml', dllfile='libmf6.dll'):
        """Initialize the Regenerator with the working directory and input files.

        Parameters
        ----------
        wd : str, optional
            Working directory where the phinp file is located. Default is ``"."``.
        phinp : str, optional
            Name of the phinp file. Default is ``"phinp.dat"``.
        yamlfile : str, optional
            Name of the PhreeqcRM YAML file to use. Default is ``"mf6rtm.yaml"``.
        dllfile : str, optional
            Name of the MODFLOW 6 shared library. Default is ``"libmf6.dll"``.
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
        """Regenerate the PHREEQC script and YAML from external files.

        Parameters
        ----------
        wd : str, optional
            Working directory. Default is ``"."``.
        phinpfile : str, optional
            Name of the source phinp file. Default is ``"phinp.dat"``.
        yamlfile : str, optional
            Name of the PhreeqcRM YAML file. Default is ``"mf6rtm.yaml"``.
        dllfile : str, optional
            Name of the MODFLOW 6 shared library. Default is ``"libmf6.dll"``.
        prefix : str, optional
            Prefix for the regenerated output files. Default is ``"_"``.

        Returns
        -------
        Regenerator
            The instance after writing the regenerated script and YAML.
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
            # only chemistry-phase sections (e.g. equilibrium_phases, kinetic_phases)
            # carry external m0 files; skip config sections like reactive/emulator/output/solver
            if key.endswith('_phases'):
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
        """Read the PHREEQC input script from disk.

        Returns
        -------
        list of str
            Lines of the ``phinp`` file in the working directory.
        """
        with open(os.path.join(self.wd, self.phinp), 'r') as f:
            script = f.readlines()
        return script

    @staticmethod
    def _is_block_start(line):
        """A column-0, non-blank, non-END line whose first token is a PHREEQC keyword.

        PHREEQC data lines are indented or lowercase, so they never match; this lets
        ``END`` and data attach to the current block instead of starting a new one.
        """
        if not line or line[0].isspace() or not line.strip():
            return None
        token = line.split()[0].upper()
        if token == "END":
            return None
        if _KEYWORD_RE.match(token):
            return token
        return None

    def _split_into_blocks(self, script):
        """Split a phinp script (list of lines) into ordered ``(keyword, lines)`` blocks.

        A block starts at a column-0 keyword line and runs until the next one; ``END``,
        blank lines, and indented/lowercase data attach to the current block (so each
        block carries its own trailing ``END`` when the source had one). Any leading
        lines before the first keyword (comments, etc.) are returned under key ``None``.
        """
        blocks = []
        current_kw = None
        current_lines = []
        for line in script:
            kw = self._is_block_start(line)
            if kw is not None:
                if current_lines:
                    blocks.append((current_kw, current_lines))
                current_kw = kw
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            blocks.append((current_kw, current_lines))
        return blocks

    @staticmethod
    def _classify_block(keyword):
        """Map a block keyword to one of: regenerated, solution, preserved, output, definition."""
        if keyword in _REGENERATED_KEYWORDS:
            return "regenerated"
        if keyword in _SOLUTION_KEYWORDS:
            return "solution"
        if keyword in _PRESERVED_REACTION_KEYWORDS:
            return "preserved"
        if keyword in _OUTPUT_KEYWORDS:
            return "output"
        return "definition"

    def update_yaml(self, filename='_mf6rtm.yaml'):
        """Update the YAML file with the regenerated script and initial conditions.

        Parameters
        ----------
        filename : str, optional
            Name of the YAML file to write. Default is ``"_mf6rtm.yaml"``.

        Returns
        -------
        numpy.ndarray
            The flattened initial-condition array written to PhreeqcRM.
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

    def generate_new_script(self):
        """
        Generate a new ``_phinp.dat`` script from the source phinp and the config.

        The source is parsed into PHREEQC keyword blocks and bucketed by category, then
        reassembled in an order that respects PHREEQC semantics:

            DEFINITION blocks (KNOBS, RATES, PHASES, *_SPECIES, ...) at the top
            -> SOLUTION blocks (verbatim)
            -> preserved reaction blocks (SURFACE/GAS_PHASE/SOLID_SOLUTIONS, verbatim)
            -> per-cell phase blocks regenerated from the config m0 files
            -> OUTPUT blocks (SELECTED_OUTPUT/USER_PUNCH/PRINT, verbatim, exactly once)

        Source EQUILIBRIUM_PHASES/KINETICS/EXCHANGE blocks are dropped because they are
        rebuilt per cell from the config.
        """
        script = self.read_phinp()
        blocks = self._split_into_blocks(script)

        definitions, solutions, preserved, output = [], [], [], []
        for keyword, lines in blocks:
            if keyword is None:
                # leading comments / preamble before the first keyword -> keep at top
                definitions.extend(lines)
                continue
            category = self._classify_block(keyword)
            if category == "regenerated":
                continue  # rebuilt from config below
            elif category == "solution":
                solutions.extend(lines)
            elif category == "preserved":
                preserved.extend(lines)
            elif category == "output":
                output.extend(lines)
            else:  # definition
                definitions.extend(lines)

        self.solution_blocks = solutions
        self.postfix_blocks = output

        new_script = []
        new_script.extend(definitions)
        if definitions and not _ends_with_end(definitions):
            # terminate the definition group so it commits before the SOLUTION blocks
            new_script.append("END\n")
        new_script.extend(solutions)
        new_script.extend(preserved)

        # Per-cell phase blocks regenerated from the config m0 files.
        block_generators = {
            "equilibrium_phases": self.generate_equilibrium_phases_blocks,
            "kinetic_phases": self.generate_kinetic_phases_blocks,
            "exchange_phases": self.generate_exchange_phases_blocks,
        }
        for key in self.config.keys():
            generator = block_generators.get(key)
            if generator is not None:
                new_script.extend(generator())

        new_script.extend(output)
        # Guarantee every piece ends with a newline so adjacent blocks never merge
        # (e.g. a source block whose last line lacks a trailing '\n' -> "...trueEND").
        normalized = [p if p.endswith("\n") else p + "\n" for p in new_script]
        self.regenerated_script = ''.join(normalized).strip()
        return self.regenerated_script

    def write_new_script(self, filename='_phinp.dat'):
        """Write the regenerated script to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the output script file. Default is ``"_phinp.dat"``.

        Returns
        -------
        str
            Full path to the written script file.
        """
        if not hasattr(self, 'regenerated_script'):
            self.generate_new_script()
        with open(os.path.join(self.wd, filename), 'w') as f:
            f.write(self.regenerated_script)
        # print(f"New script written to {os.path.join(self.wd, filename)}")
        self.regenerated_phinp = os.path.join(self.wd, filename)
        return self.regenerated_phinp

    def generate_equilibrium_phases_blocks(self):
        """Generate per-cell EQUILIBRIUM_PHASES blocks from the config.

        Returns
        -------
        list of str
            One ``EQUILIBRIUM_PHASES`` block string per grid cell.
        """
        self.add_m0_to_config()
        equilibrium_phases = self.config.get('equilibrium_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"EQUILIBRIUM_PHASES {i_phase}\n"
            for nme in equilibrium_phases['names']:
                si = equilibrium_phases.get('si', None).get(nme, None)
                m0 = equilibrium_phases.get('m0', None).get(nme, None).flatten()
                block += f"    {nme} {si:.5e} {m0[i_phase-1]:.5e}\n"
            block += "END\n"
            blocks.append(block)
        self.equilibrium_phases_blocks = blocks
        return blocks

    def generate_kinetic_phases_blocks(self):
        """Generate per-cell KINETICS blocks from the config.

        Returns
        -------
        list of str
            One ``KINETICS`` block string per grid cell.
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
        """Generate per-cell EXCHANGE blocks from the config.

        Returns
        -------
        list of str
            One ``EXCHANGE`` block string per grid cell.
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
        """Read the external phase files required for regeneration.

        Returns
        -------
        dict
            Loaded arrays organized by phase key and name, reshaped to the
            grid dimensions; also stored on ``self.file_data``.
        """
        grid_type = ...

        file_data = {}
        # Read phase files following the same logic as validate_external_files
        for key, value in self.config.items():
            # only chemistry-phase sections (e.g. equilibrium_phases, kinetic_phases)
            # carry external m0 files; skip config sections like reactive/emulator/output/solver
            if key.endswith('_phases'):
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
        """Add the loaded array data to the config dictionary.

        Reads the external files first if they have not been loaded yet.

        Returns
        -------
        dict
            The configuration dictionary with ``m0`` arrays added per phase.
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
    """Collects and writes PhreeqcRM selected output for a run.

    Reads the PhreeqcRM selected-output block each reactive time step and
    writes it (with cell ids and saturation) to CSV or HDF5, and optionally
    exports machine-learning feature arrays.

    Parameters
    ----------
    mf6rtm : Mf6RTM
        The parent reactive transport instance, providing the MF6 and
        PhreeqcRM interfaces.
    sout_fname : str, optional
        Output filename. Default is ``"sout.csv"``. A ``.h5``/``.hdf5``
        extension switches the format to HDF5.
    output_format : {'csv', 'hdf5'}, optional
        Output format. Default is ``"csv"``. HDF5 requires PyTables.
    """

    def __init__(self, mf6rtm, sout_fname: str = "sout.csv", output_format: str = "csv"):
        self.mf6rtm = mf6rtm
        self.phreeqcbmi = mf6rtm.phreeqcbmi
        self.mf6api = mf6rtm.mf6api
        self.sout_fname = sout_fname
        if output_format == "csv" and sout_fname.endswith((".h5", ".hdf5")):
            output_format = "hdf5"
        if output_format == "hdf5":
            try:
                import tables  # noqa: F401
            except ImportError:
                raise ImportError(
                    "HDF5 output requires PyTables: pip install tables"
                )
        self.output_format = output_format
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
        common_dtypes = {
            col: dtype
            for col, dtype in self._current_soutdf.dtypes.items()
            if col in self.phreeqcbmi.soutdf.columns
        }
        updf = pd.concat(
            [
                self.phreeqcbmi.soutdf.astype(common_dtypes),
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
        t = self.mf6rtm.ctime
        headers = list(self.phreeqcbmi.sout_headers)
        time_row = next((i for i, h in enumerate(headers) if "time" in h.lower()), None)
        if time_row is not None:
            sout[time_row] = np.ones_like(sout[time_row]) * t
        df = pd.DataFrame(columns=headers)
        for col, arr in zip(headers, sout):
            df[col] = arr
        if time_row is None:
            df.insert(0, "time_d", t)
        df = self._add_spatial_columns(df)
        self._current_soutdf = df

    def _update_soutdf(self, df: pd.DataFrame) -> None:
        """Update the selected output dataframe to phreeqcrm object"""
        self.phreeqcbmi.soutdf = df

    def _check_sout_exist(self) -> bool:
        """Check if selected output file exists"""
        return os.path.exists(os.path.join(self.mf6rtm.wd, self.sout_fname))

    def _add_spatial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert 1-indexed MODFLOW spatial columns after the first (time) column."""
        dims = grid_dimensions(self.mf6api)
        cell_idx = np.arange(len(df))

        if "cell" in df.columns:
            df = df.drop(columns=["cell"])

        df.insert(1, "cell", cell_idx + 1)

        if len(dims) == 3:
            nlay, nrow, ncol = dims
            layers, rows, cols = np.unravel_index(cell_idx, (nlay, nrow, ncol))
            df.insert(2, "layer", layers + 1)
            df.insert(3, "row", rows + 1)
            df.insert(4, "col", cols + 1)
        elif len(dims) == 2:
            nlay, ncpl = dims
            layers, cell2d = np.divmod(cell_idx, ncpl)
            df.insert(2, "layer", layers + 1)
            df.insert(3, "cell2d", cell2d + 1)
        return df

    def _write_sout_headers(self) -> None:
        """Write selected output headers to a file"""
        if self.output_format == "hdf5":
            return
        dims = grid_dimensions(self.mf6api)
        if len(dims) == 3:
            spatial = ["cell", "layer", "row", "col"]
        else:
            spatial = ["cell", "layer", "cell2d"]
        phreeqc_headers = [h for h in self.phreeqcbmi.sout_headers if h != "cell"]
        if not any("time" in h.lower() for h in phreeqc_headers):
            headers = ["time_d"] + spatial + phreeqc_headers
        else:
            headers = phreeqc_headers[:1] + spatial + phreeqc_headers[1:]
        with open(os.path.join(self.mf6rtm.wd, self.sout_fname), "w") as f:
            f.write(",".join(headers))
            f.write("\n")

    def _rm_sout_file(self) -> None:
        """Remove the selected output file"""
        try:
            os.remove(os.path.join(self.mf6rtm.wd, self.sout_fname))
        except FileNotFoundError:
            pass

    def _append_to_soutdf_file(self) -> None:
        """Append the current selected output to the selected output file"""
        assert not self._current_soutdf.empty, "current sout is empty"
        path = os.path.join(self.mf6rtm.wd, self.sout_fname)
        if self.output_format == "hdf5":
            self._current_soutdf.to_hdf(
                path, key="sout", mode="a", append=True, format="table",
                data_columns=True, complevel=5, complib="blosc"
            )
        else:
            self._current_soutdf.to_csv(path, mode="a", index=False, header=False)
