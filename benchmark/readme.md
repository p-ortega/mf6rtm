# Benchmarks

## Quick Start

1. Install the environment (see [Install Development Environment](#install-development-environment) below).
2. Fetch the MODFLOW 6 binaries:
   ```shell
   pixi run install-benchmark-bins
   ```
   Or manually with `get-modflow` (provided by FloPy) depending on your OS:
   ```shell
   # macOS
   get-modflow benchmark/bin/mac --subset mf6,libmf6
   # Linux
   get-modflow benchmark/bin/linux --subset mf6,libmf6
   # Windows
   get-modflow benchmark/bin/win --subset mf6,libmf6
   ```
3. Launch Jupyter and open any `ex*.ipynb` notebook.

---

These Jupyter Notebooks run classic, well-know reference models using this MF6RTM package to compare model outputs with those produced by PHT3D and PHREEQC (for example 4):

1. **Example 1: Engesgaard and Kipp 1992. 1D Precipitation and Dissolution Fronts.** A one-dimensional model domain in which an aqueous water composition that is in equilibrium with two minerals, calcite and dolomite, is successively replaced, i.e., flushed by water of a different chemical composition, leading to multiple precipitation-dissolution fronts. 
2. **Example 2: Walter 1994. 1D migration of AMD precipitation & dissolution fronts.** A one-dimensional, purely inorganic redox problem that  demonstrates the evolution of some important geochemical processes that occur when acidic mine drainage (AMD) leaches into an anaerobic carbonate aquifer.
3. **Example 2: Walter 1994. 2D migration of AMD precipitation & dissolution fronts.** A two-dimensional version of Example 2.
4. **Example 4: Parkhurst and Appelo 2013 (PHREEQC-3 Example 11) 1D Cation Exchange.** Cation exchange column flushing of a sodium-potassium nitrate solution with calcium chloride.
5. **Example 5: Appelo 1998. Pyrite Oxidation**  Modelling of an oxidation experiment with marine pyrite-containing sediments.
6. **Example 6: Parkhurst and Appelo 2013 (PHREEQC-3 Example 11) Cation Exchange.** 3D variant of Cation Exchange with DISV grid

## Install Development Environment

Follow the recommendations for [Developing](README.md#developing) on the main README.md for this repo -- `pixi install` from the repo root gives a working environment with mf6rtm already importable, and `pixi run install-benchmark-bins` (shown in Quick Start above) fetches the executables these benchmarks need.
