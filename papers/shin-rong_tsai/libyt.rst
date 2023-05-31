:author: Shin-Rong Tsai
:email: srtsai@illinois.edu
:institution: National Taiwan University, Department of Physics
:institution: University of Illinois at Urbana-Champaign, School of Information Sciences

:author: Hsi-Yu Schive
:email: hyschive@phys.ntu.edu.tw
:institution: National Taiwan University, Department of Physics
:institution: National Taiwan University, Institute of Astrophysics
:institution: National Taiwan University, Center for Theoretical Physics
:institution: National Center for Theoretical Sciences, Physics Division

:author: Matthew J. Turk
:email: mjturk@illinois.edu
:institution: University of Illinois at Urbana-Champaign, School of Information Sciences

:bibliography: mybib


---------------------------------------------------
libyt: a Tool for Parallel In Situ Analysis with yt
---------------------------------------------------

.. class:: abstract

   In the era of exascale computing, storage and analysis of large scale data have become 
   more important and difficult. 
   We present ``libyt``, an open source C++ library, that allows researchers to analyze and 
   visualize data using yt or other Python packages in parallel during simulation runtime. 
   We describe the code method for reading adaptive mesh refinement grid data structure, 
   handling data transition between Python and simulation with minimal memory overhead, and 
   conducting analysis with no additional time penalty using Python C API and NumPy C API. 
   We demonstrate how it solves the problem in astrophysical simulations and increases disk 
   usage efficiency. Finally, we conclude it with discussions about ``libyt``.
   

.. class:: keywords

   Astronomy data analysis, Astronomy data visualization, Open source software

Introduction
------------

This is introduction.
Should include motivation and aims.


Code Method
-----------


Overview
++++++++


Connecting Python and Simulation
++++++++++++++++++++++++++++++++


Executing Python Codes and Handling Errors
++++++++++++++++++++++++++++++++++++++++++


In Situ Analysis Under Parallel Computing
+++++++++++++++++++++++++++++++++++++++++


Applications
------------

Analyzing Fuzzy Dark Matter Vortices Simulation
+++++++++++++++++++++++++++++++++++++++++++++++

Analyzing Core-Collapse Supernova Simulation
++++++++++++++++++++++++++++++++++++++++++++

Discussions
-----------

It is well known :cite:`yt`


