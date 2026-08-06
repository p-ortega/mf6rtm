.. mf6rtm documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: Open-source Python package for reactive transport modeling — MODFLOW 6 coupled with PHREEQC (PhreeqcRM). A scriptable, benchmarked alternative to PHT3D.
   :keywords: reactive transport, MODFLOW 6, PHREEQC, PhreeqcRM, Python, groundwater modeling, geochemistry, PHT3D

mf6rtm — Reactive Transport Modeling with MODFLOW 6 and PHREEQC
===============================================================

Reactive transport modeling in Python — **MODFLOW 6** coupled with **PHREEQC** through the PhreeqcRM API.

mf6rtm couples MODFLOW 6 (groundwater flow and transport) with PHREEQC
(geochemistry) through the ``modflowapi`` and ``phreeqcrm`` APIs, providing a
single Python interface for simulating reactive transport in the subsurface.

.. code-block:: bash

   pip install mf6rtm

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   introduction
   tutorials/index
   api/modules
   development

Getting started
---------------

* :doc:`introduction` — what mf6rtm is and how it is organized.
* :doc:`tutorials/index` — worked reactive-transport examples.
* :doc:`api/modules` — full API reference.
* :doc:`development` — set up a dev environment and contribute.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
