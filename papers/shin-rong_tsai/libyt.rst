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
   visualize data using ``yt`` or other Python packages in parallel during simulation runtime. 
   We describe the code method for reading adaptive mesh refinement grid data structure, 
   handling data transition between Python and simulation with minimal memory overhead, and 
   conducting analysis with no additional time penalty using Python C API and NumPy C API. 
   We demonstrate how it solves the problem in astrophysical simulations and increases disk 
   usage efficiency. Finally, we conclude it with discussions about ``libyt``.
   
   (TODO: needs review)
   

.. class:: keywords

   astronomy data analysis, astronomy data visualization, in situ analysis, open source software

Introduction
------------
.. 
   problem we are trying to solve, our motivation and our goals

In the era of exascale computing, storage and analysis of large-scale data has become a critical 
bottleneck. 
Simulations often use efficient programming language like C and C++, while many data analysis tools 
are written in Python, for example ``yt`` [#]_ :cite:`yt`. 
``yt`` is an open-source, permissively-licensed Python package for analyzing and visualizing 
volumetric data. 
It is a light weight tool for quantitative analysis for astrophysical data, and it has also been 
applied to other scientific domains.
Normally, we would have to dump simulation data to hard disk first before conducting analysis using 
existing Python tools. 
This takes up lots of disk space when the simulation has high temporal and spatial resolution. 
This also forces us to store full datasets, even though our region of interest might contain only 
a small portion of simulation domain. 
It makes large simulation hard to analyze and manage due to the limitation of disk space. 
Is there a way to probe those ongoing simulation data using robust Python ecosystem? 
So that we don't have to re-invent data analysis tools and solve the disk usage issue at the same 
time.

.. [#] `https://yt-project.org/ <https://yt-project.org/>`_ 

.. 
   in situ analysis and features of libyt

In situ analysis, which is to analyze simulation data on-site, without intermediate step of writing 
data to hard disk is a promising solution.
We introduce ``libyt`` [#]_ , an open source C++ library, that allows researchers to analyze and 
visualize data by directly calling ``yt`` or any other Python packages during simulations runtime 
under parallel computation. 
Through wrapping ongoing simulation data using NumPy C API :cite:`numpy`, constructing proper Python 
C-extension methods and Python objects using Python C API :cite:`python3`, we can reuse C++ runtime 
data and realize back-communication of simulation information, allowing user to define their own 
data generating function, and use it to conduct analysis inside Python ecosystem. 
This is like using a normal Python prompt, but with access to simulation data. 
Using ``libyt`` does not add a time penalty to the analysis, because using Python for in situ analysis 
and post-processing are exactly the same, except that the former one reads data from memory and the 
later one reads data from disks. 
And converting the post-processing script to inline script is a one-line change.
``libyt`` provides another way for us to interact with simulations.


.. [#] `https://github.com/calab-ntu/libyt <https://github.com/calab-ntu/libyt>`_

..
   outline of the proceeding

In this proceeding, we will describe the code method, demonstrate how ``libyt`` solve the problem, 
and conlude it with discussions.


Code Method
-----------


Overview
++++++++

``libyt`` serves as a bridge between simulation processes and Python instances as 
illustrated in Fig :ref:`parallelism`.
It is the middle layer that handles data IO between simulations and Python instances.
When launching *N* MPI processes, each process contains one piece of simulation and 
one Python interpreter. Each Python interpreter has access to simulation data.  
A total of *N* Python instances will work together to conduct in situ analysis in the 
process space of MPI task.

.. figure:: Parallelism.pdf
   :figclass: thb

   This is the overall structure of ``libyt``, and its relationship with simulation 
   and Python. It provides an interface for exchanging data between simulations and 
   Python instances, and between each process, thereby enabling in situ parallel 
   analysis using multiple MPI processes. ``libyt`` can run arbitrary Python scripts 
   and Python modules, though here we focus on using ``yt`` as its core analysis 
   platform. 
   :label:`parallelism`


Connecting Python and Simulation
++++++++++++++++++++++++++++++++


It stores them in a linear fashion according to global id, so that ``libyt`` can easily 
look up information later. 

.. figure:: PassInData.pdf
   :figclass: thb

   This diagram shows how ``libyt`` loads and organizes simulation information and 
   data that is based on adaptive mesh refinement data structure. 
   ``libyt`` collects local grid hierarchy information (e.g., levels, parent id, grid 
   edges) and combines them all, so that each Python instance contains full hierarchy.
   As for simulation data, ``libyt`` wraps them using NumPy C API, which tells Python 
   how to interpret block of memory (e.g., shape, type, stride), without duplicating 
   memory. 
   :label:`passindata`

.. figure:: PythonAskData.pdf
   :figclass: thb

   This diagram describes how ``libyt`` requests simulation to generate data using 
   user-defined function, thus enabling back-communication of simulation information. 
   Those generated data is freed once it is no longer used by Python.
   :label:`pythonaskdata`

Executing Python Codes and Handling Errors
++++++++++++++++++++++++++++++++++++++++++

.. figure:: REPL.pdf
   :figclass: thb

   The procedure shows how ``libyt`` supports interactive Python prompt. 
   It takes user inputs in root process and executes Python codes across whole MPI processes. 
   The root process handles syntax errors and distinguishes whether or not the error is caused 
   by user hasn't done inputing yet.
   :label:`pythonprompt`


In Situ Analysis Under Parallel Computing
+++++++++++++++++++++++++++++++++++++++++

.. 
   yt parallelism feature, data chunking

.. 
   RMA

During in situ Python analysis, workloads may be decomposed and rebalanced according 
to the algorithm in Python packages.

.. figure:: RMA.pdf
   :figclass: thb

   This is the workflow of how ``libyt`` redistributes data.
   It is done via one-sided communication (Remote Memory Access in MPI). 
   Each process prepares requested data by other processes, after this, every process 
   fetches data located on different processes.
   This is a collective operation, and data is redistributed during this window epoch. 
   Since the data fetched is only for analysis purpose, it gets freed once Python doesn't 
   need it at all. 
   :label:`rma`

Applications
------------

Analyzing Fuzzy Dark Matter Vortices Simulation
+++++++++++++++++++++++++++++++++++++++++++++++

Analyzing Core-Collapse Supernova Simulation
++++++++++++++++++++++++++++++++++++++++++++

Discussions
-----------



