# MF6RTM: Reactive Transport Model via the MODFLOW 6 and PHREEQCRM APIs

## Benchmarks
Benchmark comparing model results against PHT3d are in `benchmark/`. Each folder contains a Jupyter notebook to write and execute an MF6RTM model via the MUP3D class. Additionally, PHT3d files are provided in the corresponding _pht3d directory.

## Considerations

The current version is intended to work with structured grids (dis object in MF6) and one MF6 simulation that includes the flow and transport solutions. No support is currently provided for a 'flow then transport scheme,' meaning that advanced packages cannot be incorporated yet.

On the PHREEQC side, the following have been included:

- Solution
- Equilibrium phases
- Cation Exchange

Future updates will include kinetic phases and surface complexation, with their respective benchmarks.

## Software requirements
All dependencies and executables are included in this repo. To run this test, I recommend creating the environment by typing in conda

```commandline
conda env create -f env.yml
```
## Authors
Pablo Ortega (Portega)