Supported features
==================

MODFLOW 6
---------

Supported
~~~~~~~~~

- **Grids:** DIS and DISV.
- **Simulation:** one MF6 simulation with flow and transport solved together.  
- **Workflows:** classic ``Mup3d`` and transport-first ``Mup3d.from_mf6``.  
- **Transport options:** advection, dispersion and MST settings (including sorption, via ``set_mst_override``) copied from the template GWT model.  
- **Boundary chemistry** (``ChemStress``): ``aux`` (via SSM), ``cnc``, ``src``.  

Use with caution
~~~~~~~~~~~~~~~~

- Convertible layers and dry/rewetting cells (untested).

Not supported
~~~~~~~~~~~~~

- DISU grids.
- "Flow then transport" runs (FMI).
- Advanced transport packages: LKT, SFT, UZT, MWT, MVT, IST.
- Multiple GWF models or model exchanges.
- MF6 parallel (MPI) runs.

PhreeqcRM
---------

Supported
~~~~~~~~~

- **PHREEQC blocks:** SOLUTION, EQUILIBRIUM_PHASES, EXCHANGE, SURFACE,
  KINETICS (rates from the database), plus a postfix file.
- **Threads:** multithreaded reactions (``nthread``).

Use with caution
~~~~~~~~~~~~~~~~

- Temperature is fixed per solution.
- Not all PHREEQC block options have been tested.

Not supported yet
~~~~~~~~~~~~~~~~~

- GAS_PHASE and SOLID_SOLUTIONS.
- Varying temperature or pressure during a run.

mf6rtm
------

Limitations
~~~~~~~~~~~

- Use ``threshold`` cell skipping only for equilibrium chemistry (not kinetic reactions!).
- Reaction timing ``user``: kinetics only advance on the listed steps.

Planned
~~~~~~~

- Reaction timing ``adaptive``.
- Restart / checkpointing.