Introduction
============

Overview
--------

**mf6rtm** (MODFLOW 6 Reactive Transport Modeling) is a Python package that provides seamless integration between MODFLOW-6 and PHREEQC for simulating reactive transport processes in subsurface environments.

Reactive transport modeling is essential for understanding the intricate interplay between hydrogeological processes and chemical reactions in subsurface environments. mf6rtm bridges MODFLOW-6, the current version of the MODFLOW family of groundwater flow and transport codes, with PHREEQC, a versatile software for geochemical modeling.


What is Reactive Transport Modeling?
-------------------------------------

Reactive transport modeling combines:

* **Groundwater flow** - Movement of water through porous media
* **Solute transport** - Migration of dissolved components
* **Chemical reactions** - including:
  
  * Mineral dissolution and precipitation
  * Redox reactions
  * Ion exchange and sorption

Key Features
------------

Seamless Integration
~~~~~~~~~~~~~~~~~~~~

Through the integration facilitated by the MODFLOWAPI and PHREEQCRM APIs, mf6rtm provides a unified platform for modeling groundwater flow, solute transport, 
and chemical reactions within a single computational environment.

Uncertainty Analysis
~~~~~~~~~~~~~~~~~~~~

mf6rtm model files are stored in a folder, and the ``mf6rtm`` command
runs it from that folder. PEST++ can therefore drive it the same way as a
MODFLOW 6 model, for parameter estimation and uncertainty analysis, for
example with pyEMU's ``PstFrom``. To expose the initial amounts of minerals and
exchangers as files PEST++ can change, turn on external input before writing
the model:

.. code-block:: python

   model.set_config(reactive_externalio=True)
   model.write_simulation()

mf6rtm then writes one file per phase, species, and layer, named
``{phase}.{species}.m0.layer{n}.txt`` (for example
``kinetic_phases.Pyrite.m0.layer1.txt``), and rebuilds the PHREEQC input from
them at the start of every run. The files hold one value per line; pyEMU
expects nrow x ncol arrays, so reshape them before adding parameters.

`Dizon36 <https://doi.org/10.5281/zenodo.23095990>`_ is a worked example: a
3D field model history-matched with PEST++ IES, with pilot points on hydraulic
conductivity, specific storage, and the initial amount of pyrite (see
``setup_pest`` in its ``workflow.py``).

Code Structure
------------

mf6rtm is organized into several subpackages, but there are two main components, that the user will interact with the most: solver and mup3d.

* **Solver**: This component manages the coupling between MODFLOW-6 and PHREEQC, handling data exchange, time-stepping, and overall simulation control. 
* **Mup3d**: This module acts as a pre- and post-processor for preparing input files for the reactive transport simulations, especially the chemistry inputs for PHREEQCRM (think FloPy for MODFLOW-6 + PHREEQC).

In addition to these, the user will require some knowledge and familiarity with Modflow 6 and FloPy to be able to set up and run mf6rtm simulations.

Why mf6rtm?
-----------

mf6rtm represents a significant advancement in reactive transport modeling by:

* **Integrating** state-of-the-art flow and geochemical codes
* **Providing** a Python-based, user-friendly interface
* **Enabling** uncertainty analysis and model calibration
* **Offering** flexibility for diverse hydrogeological scenarios

Getting Started
---------------

To get started with mf6rtm, see the :doc:`tutorials/index`
or explore the :doc:`api/modules` documentation.

Installation
------------

Install mf6rtm using pip:

.. code-block:: bash

   pip install mf6rtm

For development installation:

.. code-block:: bash

   git clone https://github.com/p-ortega/mf6rtm.git
   cd mf6rtm
   pip install -e .
