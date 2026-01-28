---
title: 'MF6RTM: a python package for predictive reactive transport modeling via the MODFLOW 6 and PHREEQC APIs'
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
    affiliation: "5"
    email: aestrada@intera.com
  - name: Allan Foster
    orcid: 0000-0002-3746-4226
    affiliation: "4"
    email: afoster@intera.com
  - name: Paul Tomasula
    orcid: 0009-0004-8120-5936
    affiliation: "2"
    email: ptomasula@limno.com
  - name: Lauren Mancewicz
    orcid: 0009-0002-9622-5636 
    affiliation: "6"
    email: lauren.k.mancewicz@usace.army.mil
affiliations:
 - name: Intera Geosciences, Perth, WA, Australia
   index: 1
 - name: Limnotech, Oakdale, MN, USA
   index: 2
 - name: Intera Incorporated, Denver, CO, USA
   index: 4
 - name: Intera Incorporated, Houston, TX, USA
   index: 5
 - name: Coastal and Hydraulics Lab, Engineer Research and Developement Center, Vicksburg, MS, USA
   index: 6
date: 28 January 2026
bibliography: paper.bib
---

# Summary

Reactive transport modeling (RTM) plays a central role in characterizing and predicting the coupled behavior of groundwater flow, solute transport, and geochemical reactions in subsurface systems [@Prommer2019]. This paper presents MF6RTM (MODFLOW 6 Reactive Transport Module), a Python package that tightly couples MODFLOW 6 [@Langevin2024], the current generation of the MODFLOW groundwater flow and transport code family, with PHREEQC [@Appelo2010], a widely used geochemical modeling engine. The coupling is achieved through the MODFLOWAPI [@Hughes2022] and PHREEQCRM [@Parkhurst2015] APIs, which use the Basic Model Interface (BMI) version 2.0 [@Hutton2020] to enable efficient and consistent data exchange between hydraulic, transport, and geochemical components during simulation.

The software provides a unified computational environment for simulating a wide range of reactive transport processes, including contaminant migration, mineral dissolution and precipitation, and redox reactions. It also supports the core features of both MODFLOW 6 and PHREEQC modeling software packages, allowing users to represent complex hydrogeological conditions and geochemical systems.

In addition, MF6RTM includes an improved input and output system that can externally write chemistry-related input files in array format. This behavior is similar to the external file workflow already present in MODFLOW 6 and provides flexibility for integration with uncertainty analysis tools such as PEST++ [@White2018] and its Python interface PyEMU [@White2016]. These capabilities support fully-scripted workflows such as uncertainty quantification, sensitivity analysis, and multi-objective optimization.

Together, these features make MF6RTM a versatile and robust framework for predictive reactive transport modeling in hydrogeological and environmental applications.

# State of the Field

Several tools for reactive transport modeling exist that have coupled groundwater flow and solute transport simulators with geochemical engines or implemented fully integrated reactive transport solutions. A few actively developed open-source and standalone (implicitly coupled) reactive transport software systems are noteworthy, including CrunchFlow [@Steefel2014], PFLOTRAN [@Hammond2022], and OpenGeoSys [@Kolditz2012]. Other open-source software has explicitly coupled transport and reaction models, including PHAST [@Parkhurst2010], PHT3D [@Prommer2003], and eSTOMP [@Nieplocha2006]. For a more comprehensive overview, see the review by Steefel [@Steefel2015].

Previous PHREEQC couplings within the MODFLOW ecosystem include PHT3D for MODFLOW-2005 [@Prommer2003] and PHT-USG for MODFLOW-USG [@Panday2013]. PHT3D has seen extensive use in both academia and practice [@Appelo2010], while PHT-USG has gained traction more recently, particularly between practioners working with MODFLOW-USG. A key limitation of both approaches is that they require modification of the underlying source code to enable the coupling. This imposes a heavy maintenance burden and has effectively frozen these coupled systems to older software versions. Indeed, both PHT3D and PHT-USG still rely on PHREEQC-2 and updating to the latest PHREEQC version 3 [@Parkhurst2013] through the PHREEQCRM library would require substantial refactoring. As MODFLOW 6 and PHREEQC continue to expand in capability and adoption, there is a clear need for a modern, open-source coupling that preserves transparency, extensibility, and computational efficiency. 

# Statement of Need

Despite the comprehensive ecosystem for reactive transport simulators, no open-source software has ever coupled the current major versions of MODFLOW (v6 released in 2017) and PHREEQC (v3 released 2013). This gap is significant because the MODFLOW family remains the dominant platform for groundwater flow and transport modeling in regulatory, consulting, and applied research contexts. Existing integrated RTM codes generally require users to rebuild models in alternative frameworks, limiting their adoption for MODFLOW-based workflows. MF6RTM addresses this need by providing a fully open, API-based integration between MODFLOW 6 and PHREEQC. This design eliminates custom file-based workflows, reduces opportunities for error, and enables users to construct complex reactive transport simulations directly in Python. 

Moreover, there is a growing expectation that groundwater models, both reactive and non-reactive, explicitly represent uncertainty and support automated history-matching and optimization [@Langevin2012;@White2017]. Historically, most reactive transport workflows have relied on manual or ad hoc modification of input files to perform sensitivity analyses or history-matching, creating a substantial burden for modelers and limiting reproducibility. By enabling array-style input and output files, MF6RTM streamlines compatibility with uncertainty and optimization tools, and supports rigorous uncertainty analysis and multi-objective optimization. MF6RTM therefore fills an important gap in the hydrogeologic modeling ecosystem, providing researchers and practitioners with an accessible  MODFLOW-based framework for fully scripted reactive transport modeling.


# Software Design

MF6RTM was developed iteratively, beginning with a proof-of-concept implementation to demonstrate that MODFLOW 6 and PHREEQC could be successfully coupled through their respective APIs. Early benchmark tests were used to confirm numerical consistency and establish confidence in the coupling approach. Following this initial phase, the design focus shifted toward usability, extensibility, and integration with modern uncertainty and optimization workflows.

Several key design requirements guided the architecture. First, MF6RTM needed to reproduce established reactive transport benchmarks and agree with results from existing MODFLOW-based tools such as PHT3D. Second, the code had to support programmatic model construction, recognizing that MODFLOW workflows increasingly rely on scripting tools, as in FloPY [@Bakker2023], rather than graphical interfaces. Third, seamless integration with model-independent calibration and uncertainty analysis frameworks such as PEST++ and its Python interface PyEMU was essential. Finally, the codebase needed to be modular, with clear separation of responsibilities to reduce fragility and simplify future development.

To meet these goals, MF6RTM was organized into a four focused modules. The `simulation` module coordinates initialization, time stepping, and data exchange between MODFLOW 6 and PHREEQC via their APIs. The `mup3d` module, for Model Utility Preprocesor 3D, provides a Python-based preprocessor for constructing geochemical inputs and facilitate the arrays needed to build the MODFLOW 6 files with FloPY. Supporting modules handle configuration management and array-based input/output (`config` and `io`), enabling flexible external file handling and efficient coupling to uncertainty-driven workflows. The code structure is graphicaly presented below:


```
MF6RTM
├── mup3d
│   └── base.py
│
├── simulation
│   ├── solver.py
│   ├── mf6api.py
│   ├── phreeqcbmi.py
│   └── discretization.py
│
├── io
│    └── externalio.py
│
└── config/utils
    ├── config.py
    ├── utils.py
    └── yaml_reader.py
```

# Benchmarks

Eight benchmark test cases are currently included in the codebase. Each represents a well-known reactive transport scenario to confirm the accuracy of results for different combinations of processes. Seven of them correspond to models that apply different hydraulic fields and geochemical reaction networks, with results compared against PHT3D and in a few cases against PHREEQC. The Example 6 correspond to the same as Example 4 but uses the MODFLOW 6 discretization-by-vertices (DISV) package to illustrate the use of an unstructured grid. Additionaly, a fully 3D benchmark can also be found in the following repository [Dizon36](https://github.com/p-ortega/dizon36).

Here we present the following benchmark (Example 5 in codebase) to demonstrate usage and verify that the implementation is correct.

This benchmark models a 1D column oxidation experiment in marine sediments containing pyrite, originally described by Appelo et al. [@Appelo1998]. The hydrochemical system includes multiple coupled processes:

- **Pyrite oxidation**, the primary driver of hydrochemical evolution  
- **Secondary reactions**, including calcite dissolution, CO₂ sorption, and cation exchange  
- **Oxidation of organic matter**, which competes for the available oxidising capacity  

The model simulation consists of three sequential phases:

1. **Equilibration phase:** The sediment was saturated with a 280 mmol MgCl<sub>2</sub>  solution, filling the pore space and loading the exchange sites with Mg.  
2. **Dilute flushing phase:** The column was flushed with a more dilute MgCl<sub>2</sub> solution, providing data used to characterise non-reactive transport.  
3. **Oxidation phase:** The column was flushed for four pore volumes with an oxidising H<sub>2</sub>O<sub>2</sub> solution at the same flow rate. 

\autoref{fig:ex5} compares the MF6RTM simulation results with those from PHT3D and with the experimental data. The good agreement with PHT3D and the experimental data shows that MF6RTM accurately reproduces the benchmark behavior.

![Comparison between simulated values from MF6RTM against PHT3D \label{fig:ex5}](ex5.png){width=100%}

# Research Impact Statement 
MF6RTM has demonstrated relevance for both academic and applied hydrogeologic modeling. The code includes eight benchmark test cases representing well-known reactive transport scenarios, covering a range of coupled flow, transport, and geochemical processes. Simulations from MF6RTM show excellent agreement with both experimental data and established MODFLOW-based reactive transport tools such as PHT3D, confirming the reliability of the implementation.

Thanks to its integration with uncertainty analysis, MF6RTM has been incorporated into the Groundwater Modeling Decision Support Initiative ([GMDSI](https://www.gmdsi.org)). A fully 3D tutorial using an unstructured grid is currently in preparation and is expected to be released soon ([rtm-gmdsi](https://github.com/p-ortega/rtm-tutorial)), providing researchers and practitioners with a practical, hands-on guide to applying MF6RTM in complex hydrogeologic settings. In addition, MF6RTM is being currently applied to real-world Aquifer Storage and Recovery (ASR), and example of this implementation can be found in [ASR - DISV Example](https://github.com/LimnoTech/mf6rtm-asr-example).


# AI Usage Disclosure
AI tools were used in a limited and supportive capacity during the development of MF6RTM and the preparation of this manuscript.  No AI was used for the design of the code. Specifically, AI assistance was used to draft and refine docstrings, explore potential causes of software bugs, and suggest optimizations for selected sections of the code. AI tools were use to improve grammar, clarity, and writing quality of the manuscript.

# Acknowledgements

The software MF6RTM was supported by INTERA INC., and its Research and Development initiative. We also thank Henning Prommer for his insights and discussions during the benchmarking of MF6RTM.


# References