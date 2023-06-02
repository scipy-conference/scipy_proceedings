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
data generating C function, and use it to conduct analysis inside Python ecosystem. 
This is like using a normal Python prompt, but with direct access to simulation data. 
``libyt`` provides another way for us to interact with simulations.

.. [#] `https://github.com/calab-ntu/libyt <https://github.com/calab-ntu/libyt>`_

..
   outline of the proceeding

In this proceeding, we will describe the methods in Section `Code Method`_, demonstrate how ``libyt`` 
solve the problem in Section `Applications`_, and conlude it with Section `Discussions`_.

.. _Code Method:

Code Method
-----------

.. _Overview of libyt:

Overview of libyt
+++++++++++++++++

``libyt`` serves as a bridge between simulation processes and Python instances as 
illustrated in Fig :ref:`parallelism`.
It is the middle layer that handles data IO between simulations and Python instances, 
and between MPI processes. 
When launching *N* MPI processes, each process contains one piece of simulation and 
one Python interpreter. Each Python interpreter has access to simulation data. 
When doing in situ analysis, every simulation process pauses, and a total of *N* Python 
instances will work together to conduct Python tasks in the process space of MPI. 

.. figure:: Parallelism.pdf
   :figclass: thb

   This is the overall structure of ``libyt``, and its relationship with simulation 
   and Python. It provides an interface for exchanging data between simulations and 
   Python instances, and between each process, thereby enabling in situ parallel 
   analysis using multiple MPI processes. ``libyt`` can run arbitrary Python scripts 
   and Python modules, though here we focus on using ``yt`` as its core analysis 
   platform. 
   :label:`parallelism`

Simulations use ``libyt`` API [#]_ to pass in data and run Python codes during runtime, 
and Python instances use ``libyt`` Python module to request data directly from simulations 
using C-extension method and access Python objects that contain simulation information. 
Using ``libyt`` for in situ analysis is very similar to running Python scripts in post-processing 
under MPI platform, except that data are stored in memory instead of hard drives. 
``libyt`` is for general-purpose and can launch arbitrary Python scripts and Python modules, 
though here, we focus on using yt as our core analysis tool.

.. [#] For more details, please refer to ``libyt`` documents. 
   (`https://calab-ntu.github.io/libyt/libytAPI <https://calab-ntu.github.io/libyt/libytAPI>`_)


.. _Connecting Python and Simulation:

Connecting Python and Simulation
++++++++++++++++++++++++++++++++

We can extend the functionality of Python by calling C/C++ functions, and, likewise, 
we can also embed Python in a C/C++ application to enhance its capability. 
Python and NumPy provides C API for users to connect objects in a main C/C++ program to Python. 

Currently, ``libyt`` supports only adaptive mesh refinement (AMR) grid data strucutre. [#]_
How ``libyt`` organizes simulation with AMR grid data strucutre is illustrated in Fig :ref:`passindata`. 
It first gathers and combines local adaptive mesh refinement grid information 
(e.g., levels, parent id, grid edges, etc) in each process, such that every Python instance contains 
full information.
Next, it allocates array using ``PyArray_SimpleNew`` and stores those information in a linear 
fashion according to global grid id.
The array can be easily looked up and retrieve information by ``libyt`` at C side using ``PyArray_GETPTR2``. 
The operation involves only reading elements in an array. It can also be accessed at Python side. 
For simulation data, ``libyt`` wraps those data pointers using NumPy C API ``PyArray_SimpleNewFromData``. 
This tells Python how to interpret block of memory (e.g., shape, type, stride) and does not make a copy. 
``libyt`` also marks the wrapped data as read-only [#]_ to avoid something accidentally alters it, 
since they are actual data used in simulation's iterative process. 

.. [#] We will support more data structures (e.g., octree, unstrucutred mesh grid, etc) in the future.

.. [#] This can be done by using ``PyArray_CLEARFLAGS`` to clear writable flag ``NPY_ARRAY_WRITEABLE``.

.. figure:: PassInData.pdf
   :figclass: htb

   This diagram shows how ``libyt`` loads and organizes simulation information and 
   data that is based on adaptive mesh refinement (AMR) grid data structure. 
   ``libyt`` collects local AMR grid information and combines them all, so that each 
   Python instance contains whole information.
   As for simulation data, ``libyt`` wraps them using NumPy C API, which tells Python 
   how to interpret block of memory without duplicating it.
   :label:`passindata`

``libyt`` also supports back-communication of simulation information. 
Fig :ref:`pythonaskdata` shows the mechanism behind it. 
The process is triggered by Python when it needs the data generated by a user-defined 
C function. This usually happens when the data is not part of the simulation iterative 
process and requires simulation to generate it, or the data isn't stored in a contiguous 
memory block and requires simulation to help collect it. 
When Python needs the data, it first calls C-extension method in ``libyt`` Python module. 
The C-extension method allocates a new data buffer and passes it to user-defined C function, 
the function writes data in it. 
Finally, ``libyt`` wraps the data buffer and returns it back to Python. 
``libyt`` makes the data buffer owned by Python [#]_, so that the data gets freed when it is no 
longer needed.

.. [#] This can be done by using ``PyArray_ENABLEFLAGS`` to enable own-data flag ``NPY_ARRAY_OWNDATA``.

.. figure:: PythonAskData.pdf
   :figclass: thb

   This diagram describes how ``libyt`` requests simulation to generate data using 
   user-defined C function, thus enabling back-communication of simulation information. 
   Those generated data is freed once it is no longer used by Python.
   :label:`pythonaskdata`


Grid information and simulation data are properly organized in dictionaries under ``libyt`` 
Python module. One can easily call it during simulation runtime:

.. code-block:: python

   import libyt  # Import libyt Python module


.. _In Situ Analysis Under Parallel Computing:

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

.. _Executing Python Codes:

Executing Python Codes and Handling Errors
++++++++++++++++++++++++++++++++++++++++++

``libyt`` imports user's Python script at the initialization stage.
Every Python statement is executed inside the imported script's namespace using ``PyRun_SimpleString``. 
The namespace holds Python functions and objects. Every change made will also be stored under this 
namespace and be brought to the following round.

Using ``libyt`` for in situ analysis is just like running Python scripts in post-processing.
Their only difference lies in how the data is loaded.
Post-processing has everything store on hard disk, while data in in situ analysis is distributed 
in different computing nodes.

.. code-block:: python
   :linenos:

   import yt
   yt.enable_parallelism()
   def yt_post(data):
       ds = yt.load(data) # Load data from hard disk
       prj = yt.ProjectionPlot(ds, "z", ("gamer", "Dens"))
       if yt.is_root():
           prj.save()
   if __name__ == "__main__":
       yt_post("Data000000")





.. code-block:: python
   :linenos:

   import yt_libyt
   import yt
   yt.enable_parallelism()
   def yt_inline():
       ds = yt_libyt.libytDataset() # Load data from libyt
       prj = yt.ProjectionPlot(ds, "z", ("gamer", "Dens"))
       if yt.is_root():
           prj.save()

Simulation can call Python function using ``libyt`` API ``yt_run_Function`` and ``yt_run_FunctionArguments``.
For example, this calls the Python function defined above:

.. code-block:: c

   yt_run_Function("yt_inline");


Beside calling Python function, ``libyt`` also provides interactive prompt for user to update Python 
function, enter statements, and get feedbacks instantly. [#]_
This is like running Python prompt inside the ongoing simulation with access to data.
``libyt`` checks input Python syntax through compiling it to code object. If error occurs, it parses 
the error to see if this is caused by input not done yet or a real error.

.. [#] Currently, ``libyt`` interactive prompt only works on local machine or submit the job to HPC 
   platforms using interactive queue like ``qsub -I`` on PBS scheduler.

.. figure:: REPL.pdf
   :figclass: thb

   The procedure shows how ``libyt`` supports interactive Python prompt. 
   It takes user inputs in root process and executes Python codes across whole MPI processes. 
   The root process handles syntax errors and distinguishes whether or not the error is caused 
   by user hasn't done inputing yet.
   :label:`pythonprompt`


.. _Applications:

Applications
------------

``libyt`` has already been implemented in ``GAMER`` [#]_ :cite:`gamer-2` and ``Enzo`` [#]_ :cite:`enzo`.
``GAMER`` is a GPU-accelerated adaptive mesh refinement code for astrophysics. 
It features extremely high performance and parallel scalability and supports a rich set of physics 
modules. ``Enzo`` is a community-developed adaptive mesh refinement simulation code, 
designed for rich, multi-physics hydrodynamic astrophysical calculations.

Here, we demonstrate the results from ``GAMER`` using ``libyt``, and we show how ``libyt`` solves the 
problem of limitation in disk space and improves disk usage efficiency.

.. [#] `https://github.com/gamer-project/gamer <https://github.com/gamer-project/gamer>`_

.. [#] `https://enzo-project.org/ <https://enzo-project.org/>`_

Analyzing Fuzzy Dark Matter Vortices Simulation
+++++++++++++++++++++++++++++++++++++++++++++++

Analyzing Core-Collapse Supernova Simulation
++++++++++++++++++++++++++++++++++++++++++++

.. _Discussions:

Discussions
-----------

Using ``libyt`` does not add a time penalty to the analysis, because using Python for in situ analysis 
and post-processing are exactly the same, except that the former one reads data from memory and the 
later one reads data from disks. 
And converting the post-processing script to inline script is a one-line change.

.. figure:: Time-Proc-Ideal.pdf
   :figclass: htb

   Strong scaling of ``libyt``. The test compares the performance between in situ analysis 
   with ``libyt`` and post-processing for computing 2D profiles on a ``GAMER`` dataset. 
   The dataset contains seven adaptive mesh refinement levels with a total of :math:`9.9 \times 10^8` 
   cells. ``libyt`` outperforms post-processing by :math:`\sim 10 \textrm{ -- } 30\%` since the former 
   avoids loading data from disk to memory. The dotted line is the ideal scaling. 
   ``libyt`` and post-processing show a similar deviation from the ideal scaling because it directly 
   borrows the algorithm in ``yt``. Improvements have been made and will be made in ``yt`` to 
   eliminate the scaling bottleneck.
   :label:`performance`
