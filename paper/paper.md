---
title: 'mf6rtm: a python package for reactive transport modeling via the MODFLOW 6 and PHREEQC APIs'
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
Reactive transport modeling plays a central role in characterizing the coupled behavior of groundwater flow, solute transport, and geochemical reactions in subsurface systems. This paper presents mf6rtm, a Python package that links MODFLOW-6, the current generation of the MODFLOW groundwater flow and transport code family, with PHREEQC, a widely used geochemical modeling engine. The coupling is achieved through the MODFLOWAPI and PHREEQCRM APIs, which enable efficient and consistent data exchange between hydraulic, transport, and geochemical components during simulation.
The software provides a unified computational environment for simulating a wide range of reactive transport processes, including contaminant migration, mineral dissolution and precipitation, and redox transformations. It also supports the core features of both MODFLOW-6 and PHREEQC, allowing users to represent realistic hydrogeological conditions and complex geochemical systems.
In addition, mf6rtm includes an improved input and output system that can externally write chemistry-related files. This behavior is similar to the external file workflow already present in MODFLOW-6 and provides flexibility for integration with calibration and uncertainty analysis tools such as PEST++ ([@WHITE2018191]) and its Python interface PyEMU. These capabilities support workflows such as uncertainty quantification, sensitivity analysis, and multi-objective optimization.
Together, these features make mf6rtm a versatile and robust framework for reactive transport modeling in hydrogeological and environmental applications.

# Statement of need
Existing tools for reactive transport modeling often require researchers to navigate separate groundwater flow and geochemical modeling frameworks, manually exchange data, or rely on tightly coupled but inflexible legacy systems. As MODFLOW-6 continues to expand its capabilities and adoption, there is a growing need for a modern, open-source solution that couples it with a powerful geochemical engine while maintaining transparency, extensibility, and computational efficiency.
mf6rtm addresses this need by providing a fully open, API-based integration between MODFLOW-6 and PHREEQC. This approach eliminates the need for custom file-based workflows, reduces opportunities for error, and enables users to construct complex reactive transport simulations directly in Python. By supporting parallel computations and offering compatibility with PEST++ and PyEMU, the software also meets the needs of researchers performing large-scale simulations and rigorous uncertainty analysis. The tool fills an important gap in the hydrogeologic modeling ecosystem and provides practitioners with an accessible, reliable, and high-performance framework for reactive transport modeling.

# State of the Field


# Acknowledgements


# References