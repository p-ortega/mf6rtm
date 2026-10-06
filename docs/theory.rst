Theory and numerical approach (please rename as you see fit)
=============================

Overview
--------
.. What mf6rtm couples (MF6 GWF + one GWT per component <-> PhreeqcRM via BMI).
.. Diagram of one coupling step.                                      
.. Advection-dispersion-reaction equation per component.              
.. Equilibrium vs. kinetic reactions.
.. Definition of "component" (total H, total O, charge, elements).    

Operator splitting                                                    
------------------
.. Sequential non-iterative approach: transport solve, then reaction solve, per step.
.. Algorithm (pseudo-code of Mf6RTM.solve).                           
.. VERY IMPORTANT: Splitting error O(dt); pros/cons                      

Time discretization                                                   
-------------------
.. dt from TDIS (perlen, nstp, tsmult); same dt for transport and kinetics.  
.. Time-unit conversion to seconds.                                     
.. How to avoid numerical dispersion: Courant / Peclet guidance, advection scheme choice.                                                               
.. Reaction timing options (all / user).
               