Theory and numerical approach
=============================

Overview
--------

mf6rtm couples MODFLOW 6 with PHREEQC through the MODFLOW API and PhreeqcRM,
which both use the Basic Model Interface (BMI) version 2.0 to exchange data
during a simulation. Neither program's source code is modified. MODFLOW 6
solves groundwater flow and solute transport, and PhreeqcRM solves the
geochemical reactions.

The geochemical system is defined in :mod:`mf6rtm.mup3d` with one class per
PHREEQC input block: ``Solutions``, ``EquilibriumPhases``, ``ExchangePhases``,
``KineticPhases`` (including rate parameters), and ``SurfacePhases``. The
``Mup3d`` class assembles these blocks into a PHREEQC initialization file and
runs the initial equilibration in PhreeqcRM. This determines the transported
components: total H, total O, charge, and the elements defined in the
solutions. Their initial concentrations are returned as grid arrays in
mol m\ :sup:`-3`.

Each component is transported by its own MODFLOW 6 groundwater transport (GWT)
model, and all GWT models share the same groundwater flow (GWF) model. These
models are built either with FloPy from the ``Mup3d`` arrays, or with
``Mup3d.from_mf6``, which takes an existing FloPy simulation with a single
conservative tracer and replicates its transport model for each component.

Operator splitting
------------------

The :mod:`mf6rtm.simulation` module couples transport and reactions by
sequential non-iterative operator splitting. For each transport time step:

#. The transport models of all components are solved to convergence.
#. The cell saturation is read from the flow model.
#. The concentration of each component is read from MODFLOW 6, converted from
   mol m\ :sup:`-3` to mol L\ :sup:`-1`, and passed to PhreeqcRM as a single
   component-by-cell array.
#. PhreeqcRM integrates the reactions over the same time step.
#. The reacted concentrations are converted back and written into the
   MODFLOW 6 concentration arrays before the time step is finalized.

Data are exchanged by copy through the BMI ``get_value`` and ``set_value``
functions rather than shared memory, because each transfer requires a unit
conversion. Porosity is applied by MODFLOW 6, so the PhreeqcRM porosity is
set to one.

Two options reduce the cost of the reaction step:

- **Threshold cell skipping.** Reactions are solved only in cells whose
  relative concentration change since the previous step exceeds a threshold.
  Use this only for equilibrium chemistry, because kinetic reactions can
  progress in a cell even when transport has not changed its concentrations.
- **Minimum concentration.** Concentrations below a set value, other than
  charge, are raised to it before being passed to PhreeqcRM. This avoids
  numerical artifacts at near-zero concentrations.

Time discretization
-------------------

The time step comes from the MODFLOW 6 TDIS package (``perlen``, ``nstp``,
``tsmult``). Transport and reactions use the same time step. The TDIS
``time_units`` are converted to seconds for PhreeqcRM, and runs with no time
units set are treated as seconds.

Reaction timing controls on which steps the reactions are solved:

- ``all`` (default): reactions are solved at every time step.
- ``user``: reactions are solved only at the listed time steps. Kinetic
  reactions advance only on those steps.
