from pathlib import Path

import numpy as np
import pandas as pd

from mf6rtm.simulation.discretization import grid_dimensions
from mf6rtm.utils import utils


class DDMTOutputWriter:
    """Write mobile and immobile DDMT concentrations for all cells and all saved times."""

    def __init__(self, mf6rtm, fname=None, output_format="hdf5"):
        self.mf6rtm = mf6rtm
        self.output_format = "hdf5" if output_format in ("h5", "hdf", "hdf5") else "csv"

        if fname is None:
            fname = "ddmt.h5" if self.output_format == "hdf5" else "ddmt.csv"

        self.path = Path(self.mf6rtm.wd) / fname
        self._first_write = True

        if self.output_format == "hdf5":
            try:
                import tables  # noqa: F401
            except ImportError as err:
                raise ImportError("HDF5 DDMT output requires PyTables: pip install tables") from err

    def _spatial_frame(self):
        dims = grid_dimensions(self.mf6rtm.mf6api)
        cell_idx = np.arange(self.mf6rtm.nxyz)

        df = pd.DataFrame({"cell": cell_idx + 1})

        if len(dims) == 3:
            nlay, nrow, ncol = dims
            layers, rows, cols = np.unravel_index(cell_idx, (nlay, nrow, ncol))
            df["layer"] = layers + 1
            df["row"] = rows + 1
            df["col"] = cols + 1
        elif len(dims) == 2:
            nlay, ncpl = dims
            layers, cell2d = np.divmod(cell_idx, ncpl)
            df["layer"] = layers + 1
            df["cell2d"] = cell2d + 1

        return df

    def _domain_frame(self, domain, time_d, kper, kstp, stage, conc_m3):
        conc_m3 = np.asarray(conc_m3, dtype=float).reshape(
            self.mf6rtm.phreeqcbmi.ncomps,
            self.mf6rtm.nxyz,
        )
        conc_l = utils.concentration_m3_to_l(conc_m3)

        df = self._spatial_frame()
        df.insert(0, "domain", domain)
        df.insert(0, "stage", stage)
        df.insert(0, "kstp", int(kstp))
        df.insert(0, "kper", int(kper))
        df.insert(0, "time_d", float(time_d))

        for i, component in enumerate(self.mf6rtm.phreeqcbmi.components):
            df[component] = conc_l[i]

        return df

    def record(self, time_d, kper, kstp, stage, mobile_conc_m3, immobile_conc_m3):
        frames = [
            self._domain_frame("mobile", time_d, kper, kstp, stage, mobile_conc_m3),
            self._domain_frame("immobile", time_d, kper, kstp, stage, immobile_conc_m3),
        ]
        df = pd.concat(frames, ignore_index=True)

        if self.output_format == "hdf5":
            mode = "w" if self._first_write else "a"
            df.to_hdf(
                self.path,
                key="ddmt",
                mode=mode,
                append=not self._first_write,
                format="table",
                data_columns=True,
                complevel=5,
                complib="blosc",
            )
        else:
            df.to_csv(
                self.path,
                mode="w" if self._first_write else "a",
                index=False,
                header=self._first_write,
            )

        self._first_write = False

    def close(self):
        """No-op placeholder for future backends that keep open file handles."""
        return