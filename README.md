# MF6RTM: Reactive Transport Model via the MODFLOW 6 and PHREEQCRM APIs

## Benchmark
Benchmarck comparing model results against PHT3d are in `benchmark/`. Each folder contains a jupyter notebook to write and execute a mf6rtm model via the MUP3D class. Additinally, pht3d files are provided in the corresponding `_pht3d` dir.

## Considerations

The current version is intended to work with structured grids (dis object in mf6) and one mf6 simulation that includes the flow and transport solutions. No support is currently provided for a 'flow then transport scheme' meaning that advances packages cannot be incorporated yet.

In the phreeqc side, the following have been included:
-   solution
-   equilibrum phases

Future updates will include kinetic phases, ion exchange and surface complexation, with their respective bechmarks.

## Software requirements
All dependencies and executables are included in this repo. To run this test I recommend creating the environment by typing in conda

```commandline
conda env create -f env.yml
```
## Authors
Pablo Ortega (Portega)