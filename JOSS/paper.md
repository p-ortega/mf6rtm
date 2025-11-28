---
title: 'mf6rtm: a python package for predictive reactive transport modeling via the MODFLOW 6 and PHREEQC APIs'
tags:
  - Python
  - Reactive Transport Modeling
  - MODFLOW 6
  - PHREEQC
  - Hydrogeology
  - Geochemistry
authors:
  - name: Pablo Ortega-Tong
    orcid: 
    affiliation: "1"
    email: portega@intera.com
affiliations:
 - name: Intera Geosciencies, 166/580 Hay St, Perth
   index: 1
date: 28 November 2025
bibliography: paper.bib
---

# Summary
Reactive transport modeling plays a central role in characterizing and predicting the coupled behavior of groundwater flow, solute transport, and geochemical reactions in subsurface systems. This paper presents mf6rtm, a Python package that links MODFLOW-6, the current generation of the MODFLOW groundwater flow and transport code family, with PHREEQC, a widely used geochemical modeling engine. The coupling is achieved through the MODFLOWAPI and PHREEQCRM APIs, which enable efficient and consistent data exchange between hydraulic, transport, and geochemical components during simulation.
The software provides a unified computational environment for simulating a wide range of reactive transport processes, including contaminant migration, mineral dissolution and precipitation, and redox transformations. It also supports the core features of both MODFLOW-6 and PHREEQC, allowing users to represent complex hydrogeological conditions and geochemical systems.
In addition, mf6rtm includes an improved input and output system that can externally write chemistry-related input files in array format. This behavior is similar to the external file workflow already present in MODFLOW-6 and provides flexibility for integration with uncertainty analysis tools such as PEST++ ([@WHITE2018191]) and its Python interface PyEMU. These capabilities support workflows such as uncertainty quantification, sensitivity analysis, and multi-objective optimization.
Together, these features make mf6rtm a versatile and robust framework for predictive reactive transport modeling in hydrogeological and environmental applications.

# Statement of need
Several tools already couple flow-and-transport simulators with geochemical engines (including PHREEQC), or implement fully integrated reactive transport solutions (e.g., CrunchFlow, PFLOTRAN, OpenGeoSys). However, the MODFLOW family of codes remains one of the most widely used platforms for simulating flow and transport in real-world hydrogeologic applications. Given the number of models and workflows built around MODFLOW, having a robust and modern reactive transport coupling is essential.
Previous couplings with PHREEQC have been developed for MODFLOW-2005/MT3DMS (PHT3D) and for MODFLOW-USG (PHT-USG). PHT3D has seen extensive use in both academia and practice, while PHT-USG has gained traction more recently, particularly in consulting. A key limitation of both approaches is that they require modification of the underlying source code to enable the coupling. This imposes a heavy maintenance burden and increases the risk of the code falling out of date. Indeed, both PHT3D and PHT-USG still rely on PHREEQC-2 because updating to the latest PHREEQCRM libraries would require substantial refactoring. As MODFLOW-6 and PHREEQC continue to expand in capability and adoption, there is a clear need for a modern, open-source coupling that preserves transparency, extensibility, and computational efficiency.
mf6rtm addresses this need by providing a fully open, API-based integration between MODFLOW-6 and PHREEQC. This design eliminates custom file-based workflows, reduces opportunities for error, and enables users to construct complex reactive transport simulations directly in Python. With built-in compatibility with PEST++ and PyEMU, the tool also supports rigorous uncertainty analysis and multi-objective optimization. mf6rtm fills an important gap in the hydrogeologic modeling ecosystem, offering researchers and practitioners an accessible, reliable, and high-performance framework for reactive transport modeling.

# State of the Field

# Codebase
The codebase is organized into five modules, of which two, simulation and mup3d, serve as the core components.
The simulation module manages everything related to initializing, solving, and coordinating the interaction between MODFLOW-6 and PHREEQCRM.
The mup3d module (Model Utility Preprocessor 3D) focuses on providing users with a Python interface to help generating model input files, particularly those required for the geochemical components. Its role is similar to that of FloPy for MODFLOW(ref here).
The remaining modules provide supporting functionality: the io submodule handles reading and writing model files, while the utils and config modules assist in generating configuration files and managing the overall modeling workflow.

```
mf6rtm 0.2.1
├── mup3d
│   └── base.py
│
├── simulation
│   ├── solver.py
│   ├── mf6api.py
│   ├── phreeqcbmi.py
│   └── discretization.py
│
└── io
    └── externalio.py

└── config/utils
    ├── config.py
    ├── utils.py
    └── yaml_reader.py
```


# Acknowledgements

# References