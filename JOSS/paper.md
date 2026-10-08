---
title: 'MF6RTM: a Python package for predictive reactive transport modeling via the MODFLOW 6 and PHREEQC APIs'
tags:
  - Python
  - Reactive Transport Modeling
  - MODFLOW 6
  - PHREEQC
  - Hydrogeology
  - Geochemistry
authors:
  - name: Pablo Ortega-Tong
    orcid: 0000-0003-4091-4221
    affiliation: "1"
    email: portega@intera.com
  - name: Anthony Aufdenkampe
    orcid: 0000-0002-5811-6458
    affiliation: "2"
    email: aaufdenkampe@limno.com
  - name: Andres Prieto-Estrada
    orcid: 0000-0002-8984-1177
    affiliation: "4"
    email: aestrada@intera.com
  - name: Allan Foster
    orcid: 0000-0002-3746-4226
    affiliation: "3"
    email: afoster@intera.com
  - name: Paul Tomasula
    orcid: 0009-0004-8120-5936
    affiliation: "2"
    email: ptomasula@limno.com
  - name: Lauren Mancewicz
    orcid: 0009-0002-9622-5636 
    affiliation: "5"
    email: lauren.k.mancewicz@usace.army.mil
affiliations:
 - name: INTERA Geosciences, Perth, WA, Australia
   index: 1
 - name: LimnoTech, Oakdale, MN, USA
   index: 2
 - name: INTERA INC., Denver, CO, USA
   index: 3
 - name: INTERA INC., Houston, TX, USA
   index: 4
 - name: Coastal and Hydraulics Lab, Engineer Research and Development Center, Vicksburg, MS, USA
   index: 5
date: 28 January 2026
bibliography: paper.bib
---

# Summary

Reactive transport modeling (RTM) plays a central role in predicting the coupled behavior of groundwater flow, solute transport, and geochemical reactions in subsurface systems [@Prommer2019]. This papers present MF6RTM (MODFLOW 6 Reactive Transport Module), a Python package that couples MODFLOW 6 [@Langevin2024] with the geochemical code PHREEQC-3[@Parkhurst2013]. The coupling is achieved through the MODFLOW API [@Hughes2022] and PhreeqcRM [@Parkhurst2015], which implement the Basic Model Interface (BMI) version 2.0 [@Hutton2020], extended in MODFLOW 6 by the eXtended Model Interface (XMI), without modifying the source code of either program.

MF6RTM supports the core features of both codes, from contaminant migration to mineral dissolution and precipitation and redox reactions, and can add geochemical reactions to existing MODFLOW 6 models built with FloPy. Chemistry-related inputs are written as external array files, so that reactive transport models can be built, run, and history-matched within a single Python workflow, with the tools already used for flow models such as PEST++ [@White2018] and pyEMU [@White2016].

# State of the Field

Actively developed open-source reactive transport codes include implicitly coupled simulators such as CrunchFlow [@Steefel2014], PFLOTRAN [@Hammond2022], and OpenGeoSys [@Kolditz2012], and codes that couple separate transport and reaction models, such as PHAST [@Parkhurst2010], PHT3D [@Prommer2003], and eSTOMP [@Nieplocha2006] (see @Steefel2015 for a review).

Within the MODFLOW ecosystem, PHREEQC has been coupled to MODFLOW-2005 in PHT3D [@Prommer2003] and to MODFLOW-USG Transport in PHT-USG [@Panday2020]. PHT3D has been used extensively in academia and practice [@Appelo2010], and PHT-USG is increasingly used by practitioners. Both couplings, however, required modification of the source code, which imposes a heavy maintenance burden and has tied them to older software versions: both still rely on PHREEQC-2, and updating to PHREEQC-3 [@Parkhurst2013] through PhreeqcRM would require substantial refactoring.

# Statement of Need

Despite the comprehensive ecosystem for reactive transport simulators, to our knowledge, no open-source software comprehensively couples the current major versions of MODFLOW and PHREEQC. This gap is significant because MODFLOW remains the dominant platform for groundwater flow and transport modeling in regulatory, consulting, and applied research contexts. Existing integrated RTM codes generally require users to rebuild models in alternative frameworks, limiting their adoption for MODFLOW-based workflows. Moreover, as MODFLOW 6 and PHREEQC continue to expand in capability and adoption, keeping a coupled code current requires an approach that does not depend on changes to either source code, and that preserves transparency, extensibility, and computational efficiency. MF6RTM addresses this need by providing a fully open, API-based integration between MODFLOW 6 and PHREEQC. 

Groundwater models are also increasingly expected to represent uncertainty explicitly and to support automated history matching and optimization [@Langevin2012; @White2017]. Historically, most reactive transport workflows have relied on manual modification of input files to perform these tasks, creating a substantial burden for modelers and limiting reproducibility. Because MF6RTM exposes the geochemical inputs as array files that can be modified in the same way as MODFLOW 6 inputs, reactive parameters such as mineral amounts in solid phase can be included directly in uncertainty analysis and multi-objective optimization. MF6RTM therefore brings reactive processes into the history-matching and uncertainty quantification workflows already applied to groundwater flow models.

# Software Design

Four design requirements guided the architecture: (i) reproducing established reactive transport benchmarks and agreeing with PHT3D and PHREEQC; (ii) supporting programmatic model construction, as MODFLOW workflows increasingly rely on FloPy [@Bakker2016; @Hughes2024]; (iii) integrating with uncertainty analysis frameworks; and (iv) separating model construction from execution, so that simulation partners like PEST++ can run a model without the Python objects used to build it.

MF6RTM is therefore organized into two main modules. `mup3d` (Model Utility Preprocessor 3D) plays for MF6RTM a role similar to that of FloPy for MODFLOW 6: it constructs the geochemical inputs and the arrays needed to build the MODFLOW 6 files, and is used once, when the model is built. `simulation` reads the files written by `mup3d` and MODFLOW 6 and coordinates initialization, time stepping, and data exchange through the APIs, from Python or from the `mf6rtm` command line. Supporting modules handle array input/output (`io`), run configuration (`config`), and shared utilities (`utils`):

```
mf6rtm
├── mup3d
│   └── base.py
├── simulation
│   ├── solver.py
│   ├── mf6api.py
│   ├── phreeqcbmi.py
│   └── discretization.py
├── io
│   ├── externalio.py
│   └── yaml_reader.py
├── config.py
└── utils.py
```

## Geochemical inputs

In `mup3d`, a geochemical system is defined with one class per PHREEQC input block: `Solutions`, `EquilibriumPhases`, `ExchangePhases`, `KineticPhases` (including rate parameters), and `SurfacePhases`. Each class holds a dictionary of numbered PHREEQC definitions and an integer array, matching the model grid, that assigns a definition to each cell; `ChemStress` holds the solutions assigned to MODFLOW 6 boundary conditions. The `Mup3d` class assembles these blocks and runs the initial equilibration in PhreeqcRM, which determines the transported components and their initial concentrations in mol m$^{-3}$. These arrays are passed to FloPy to build one MODFLOW 6 groundwater transport (GWT) model per component; alternatively, `Mup3d.from_mf6` replicates the transport model of an existing FloPy simulation for each component. `Mup3d` finally writes the PHREEQC input, the PhreeqcRM settings, and the run configuration read by `simulation`.

## Coupling and data exchange

The `simulation` module wraps the MODFLOW 6 shared library through `modflowapi` and PhreeqcRM through its BMI implementation, and couples them by sequential non-iterative operator splitting. For each transport time step: (i) the transport models of all components are solved to convergence; (ii) the cell saturation is read from the flow model; (iii) the concentrations are read from MODFLOW 6, converted from mol m$^{-3}$ to mol L$^{-1}$, and passed to PhreeqcRM as a single component-by-cell array; (iv) PhreeqcRM integrates the reactions over the same time step; and (v) the reacted concentrations are converted back and written to MODFLOW 6 before the time step is finalized. Data are exchanged by copy through the BMI `get_value` and `set_value` functions rather than shared memory, as each transfer requires a unit conversion, and porosity is applied by MODFLOW 6, with the PhreeqcRM porosity set to one. Reactions can be restricted to user-defined time steps and, in models without kinetic reactions, to cells whose relative concentration change exceeds a threshold, as kinetic reactions proceed even where transport leaves concentrations unchanged.

## External input files

When external input is enabled, the initial amounts of the exchange, equilibrium, and kinetic phases are written as one array file per layer, from which the PHREEQC initialization file is regenerated at the start of each run. PEST++ can therefore modify these files like any MODFLOW 6 array, and the forward run reduces to a call to the `mf6rtm` command.

# Benchmarks

Nine benchmark cases are included in the codebase, covering from mineral dissolution and precipitation fronts and cation exchange to pyrite oxidation and the kinetic biodegradation of hydrocarbons with multiple electron acceptors, and are compared against PHT3D and, in a few cases, PHREEQC. A 1D case with first-order kinetic decay is also compared against the analytical solution of @vanGenuchten1982, isolating the integration of kinetic reactions over each transport time step. Four documentation tutorials (https://mf6rtm.readthedocs.io) cover both model-building workflows.

Example 5 is presented here to demonstrate usage and verify the implementation. It reproduces the column experiment of @Appelo1998, in which marine sediment containing pyrite was equilibrated with a 280 mmol L$^{-1}$ MgCl$_2$ solution, flushed with a more dilute solution, and oxidized for four pore volumes with H$_2$O$_2$. Pyrite oxidation drives the hydrochemical evolution, accompanied by competing organic matter oxidation, kinetic calcite dissolution, cation exchange, and CO$_2$ sorption on goethite. The geochemical model is set up with `mup3d` as follows (abbreviated from `benchmark/ex5.ipynb`):

```python
from mf6rtm import mup3d

solution = mup3d.Solutions(solutions)             # PHREEQC SOLUTION definitions
solution.set_ic(1)                                # solution number per cell
exchanger = mup3d.ExchangePhases(exchanger_dict)
exchanger_ic = np.repeat([1, 2, 3, 4], 4)         # four zones of four cells
exchanger.set_ic(exchanger_ic.reshape(nlay, nrow, ncol))

# rate laws come from the database RATES block
kin_phases = {1: {"Pyrite":   {"m0": 4e-2, "parms": [3.42, 0.0, 0.5, 0.0]},
                  "Orgc_sed": {"m0": 10.0, "parms": [9.5e-10],
                               "formula": "Orgc_sed -1.0 C 1.0"}}}
kinetics = mup3d.KineticPhases(kin_phases)
kinetics.set_ic(1)                                # all cells use KINETICS block 1

model = mup3d.Mup3d("ex5", solution, nlay, nrow, ncol)
model.set_database("ex5.dat")
model.set_exchange_phases(exchanger)
model.set_phases(kinetics)
model.initialize()                                # initial equilibration in PhreeqcRM

wellchem = mup3d.ChemStress("wel")
wellchem.set_spd([2, 3])                          # injected solution per stress period
model.set_chem_stress(wellchem)

# then build one mf6 GWT model per component with FloPy
model.run()
```

MF6RTM closely reproduces the PHT3D simulation for all major ions, pH, alkalinity, and the calcite saturation index, and both models capture the main features of the measured effluent composition (\autoref{fig:ex5}). Both models overestimate Ca concentrations after ~250 mL of outflow relative to the measurements. As this discrepancy is shared by both codes, it is attributed to the reaction network rather than to the coupling.

![Comparison of effluent concentrations simulated with MF6RTM and PHT3D for Example 5, with the measured data of @Appelo1998. \label{fig:ex5}](ex5.png){width=100%}

# Research Impact Statement

The main contribution of MF6RTM is to make reactive transport modeling with MODFLOW 6 and PHREEQC accessible from Python through their APIs. MF6RTM has been applied to a field-scale 3D model of a deep-well injection trial, including a PEST++ setup for history matching and uncertainty analysis [@Dizon36], and to a synthetic aquifer storage and recovery case on a 3D unstructured grid (https://github.com/LimnoTech/mf6rtm-asr-example).

# AI Usage Disclosure

Generative AI tools were used in a limited capacity during the development of MF6RTM to draft docstrings, explore causes of software bugs, suggest code optimizations, and improve the grammar of the manuscript. During the revision stage, Claude (Anthropic; Opus 5.5), accessed through Claude Code, was also used to (i) draft docstrings, documentation tutorials, and the contribution guidelines, (ii) restructure the package modules and write unit tests, (iii) investigate software bugs reported by the reviewers, (iv) rerun and post-process benchmark simulations, and (v) help edit the revised manuscript. The architecture of the code and the coupling scheme were designed by the authors. All AI-generated content was reviewed, edited, and tested by the authors, who take full responsibility for the software and the manuscript.

# Acknowledgements

The software MF6RTM was supported by INTERA INC., and its Research and Development initiative. We also thank Henning Prommer for his insights and discussions during the benchmarking of MF6RTM.

# References
