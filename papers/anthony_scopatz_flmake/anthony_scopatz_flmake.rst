:author: Anthony Scopatz
:email: scopatz@flash.uchicago.edu
:institution: The FLASH Center for Computational Science, Astrophysics Department, The University of Chicago

----------------------------------------------------------
Total Recall: flmake and the Quest for Reproducibility
----------------------------------------------------------

.. class:: abstract

   A short version of the long version that is way too long to be written as a
   short version anyway.  Still, when considering the facts from first
   principles, we find that the outcomes of this introspective approach is
   compatible with the guidelines previously established.

   In such an experiment, it is then clearl that the potential for further
   development not only depends on previous relationships found but also on
   connections made during exploitation of this novel new experimental
   protocol.

.. class:: keywords

   FLASH, reproducibility, version control

Introduction
------------
FLASH is a high-performance computing (HPC) multi-physics code which is used to perform
astrophysical and high-energy density physics simulations [FLASH]_.  It runs on the full 
range of systems from laptops to workstations to 100,000 processor super computers - such 
as the Blue Gene/P at Argonne National Laboratory.

Historically, FLASH was born from a collection of unconnected legacy codes written
primarily in Fortran and merged into a single project.  Over the past 13 years major
sections have been rewritten in other languages.  For instance, I/O is now implemented
in C.  However building, testing, and documentation are all performed in Python.

FLASH has a unique architecture which compiles *simulation specific* executables for each
new type of run.  This is aided by an object-oriented-esque inheritance model that is
implemented by inspecting the file system directory hierarchy.  This allows FLASH to
compile to faster machine code than a compile-once strategy.  However it also
places a greater importance on the Python build system.

To run a FLASH simulation, the user must go through three basic steps: setup, build, and
execution.  Canonically, each of these tasks are independently handled by the user.
However with the recent advent of flmake - a Python workflow management utility for
FLASH - such tasks may now be performed in a repeatable way [Flmake].

Previous workflow management tools have been written for FLASH.  (For example, the
"Milad system" was implemented entirely in Makefiles.)  However, none of the prior
attempts have placed reproducibility as their primary concern.  This is in part because
fully capturing the setup metadata requires alterations to the build system.

The development of flmake started by rewriting the existing build system
to allow FLASH to be run outside of the main line subversion repository.  It separates out
a project (or simulation) directory independent of the FLASH source directory.  This
directory is typically under its own version control.

Moreover for each of the important tasks (setup, build, run, etc), a sidecar metadata
*description* file is either written or modified.  This is a simple
dictionary-of-dictionaries JSON file which stores the environment of the
system and the state of the code when each flmake command is run.  This metadata includes
the version information of both the FLASH main line and project repositories.
However, it also may include all local modifications since the last commit.
A patch is automatically generated using the standard utilities and stored directly 
in the description.

Along with universally unique identifiers, logging, and Python run control files, the
flmake utility may use the description files to fully reproduce a simulation by
re-executing each command in its original state.  While ``flmake reproduce``
makes a useful debugging tool, it fundamentally increases the scientific merit of
FLASH simulations.

The methods described above may be used whenever
source code itself is distributed.   While this is true for FLASH (uncommon amongst compiled
codes), most Python packages also distribute their source.  Therefore the same
reproducibility strategy is applicable and highly recommended for Python simulation codes.
Thus flmake shows that reproducibility - which is notably absent from most computational science
projects - is easily attainable using only version control and standard library modules.


New Workflow Features
----------------------
Managing FLASH simulations may be a tedious task for both new and experienced
users.  The flmake command line utility eases the simulation and development cycle
by providing a modular tool that implements many common elements of a FLASH
workflow.  Moreover, at each stage this tool captures necessary metadata about the
task which it is performing.  Thus, flmake encapsulates the following opperations:

* setup/configuration,
* building,
* execution,
* logging,
* analysis & post-processing,
* and others.

It is highly recommended that both novice and advanced users adopt flmake as it 
*enables* reproducible research while simultaneously making FLASH easier to use.  
This is accomplished by a few key abstractions of previous mechanisms of setting up,
building, and executing FLASH.  The implemntation of these abstractions are 
critical flmake features and are discussed below.  Namely they are the sepaartion 
of project directroies, a serachable source path, logging, dynamic run control, and 
persisted metadata descriptions.

abstraction 1: projects
=======================
Without flmake, FLASH must be setup and built from within the FLASH source directory
(``FLASH_SRC_DIR``) using the setup script and make.  While this is fine for single
runs, it fails to separate projects in a meaningful way and makes it difficult to 
track local modifications.  

On the other hand, **flmake is intended to be run external to the FLASH source directory**
in what is known as the project directory.  When starting a new set of FLASH runs, the 
first thing that the user should do is create a new project directory somewhere on their 
file system::

    ~ $ mkdir proj
    ~ $ cd proj/
    ~/proj $ git init .

Here we called the project ``proj/`` but any name would work.  Note that this allows
all of the project-specific files to be placed under version control in a repository
that is separate from the main FLASH source. 

abstraction 2: source paths
==============================
After creating a project directory, the simulation source files must be assembled using
setup.  To run the classic Sedov problem::

    ~/proj $ flmake setup Sedov -auto
    [snip]
    SUCCESS
    ~/proj $ ls
    flash_desc.json  setup/

This command creates symbolic links to the the FLASH source files in the ``setup/`` directory.
Using the normal FLASH setup script, all of these files must live within 
``${FLASH_SRC_DIR}/source/``.  However, flmake's setup command searches additional paths to 
find potential source files.

If there is a local ``source/`` directory in the projects directory, this directory is 
searched first for any potential FLASH units.  The structure of this directory mirrors 
the layout found in ``${FLASH_SRC_DIR}/source/``.  For example, if the user wanted to write or 
overwrite their own driver unit, they could place all of the relevant files in 
``~/proj/source/Driver/``.  **Units found in the project source directory take precedence over 
units with the same name in the FLASH source.**

The most commonly overridden units, however, are simulations. Furthermore specific simulations 
live somewhat deep in the file system hierarchy residing in 
``source/Simulation/SimulationMain/${SimulationName}/``.  To make accessing simulations 
easier, a local project ``simulations/`` directory is first searched for any possible 
simulations.  Thus ``simulations/`` effectively aliases ``source/Simulation/SimulationMain/``. 
Continuing with the previous Sedov example the following directories, if they exist, are 
searched  in order of precedence:

#. ``~/proj/simulations/Sedov/``
#. ``~/proj/source/Simulation/SimulationMain/Sedov/``
#. ``${FLASH_SRC_DIR}/source/Simulation/SimulationMain/Sedov/``

Therefore, it is reasonable for a project directory to have the following structure::

    ~/proj $ ls
    flash_desc.json  setup/  simulations/  source/

abstraction 3: descriptions
=============================
In the previous section, after performing setup, a curious ``flash_desc.json`` file
appeared in the project directory.  This is the description file for the FLASH 
simulation which is currently being worked with.  This description is a sidecar
file whose purpose it is to store:

* the environment at execution of each flmake command,
* the version of both project and FLASH source repository, 
* local source code modifications (diffs),
* the run control files (see below),
* run ids and history, 
* and FLASH binary modification times.

Thus the ``flash_desc.json`` is meant to be a full picture of the way FLASH
code was generated, compiled, and executed.  **Total reproducibility of a FLASH
simulation is based on having a well-formed description file.**

The contents of this file are essentially persisted dictionary which contains 
all of the above information.  The top level keys include setup, build, run, 
and merge.  Each of these keys gets added with the corresponding flmake command.
Note that restart alters the run value and does not generate a top-level key.

During setup and build, ``flash_desc.json`` is modified in the project directory.
However, each run receives a copy of this file in the run directory with the run
information added.  Restarts and merges inherit from the file in the previous run 
directory.

The reproduce command is thus able to recreate a FLASH simulation from only
the ``flash_desc.json`` file and the associated repositories.  This is useful 
for testing and verification of the same simulation across multiple different 
machines and platforms.

It is generally not recommended that you place this file under version control
as it may change often and significantly.

abstraction 4: logging
==========================
In many ways computational simulation is more akin to experimental science than
theoretical science.  Simulations are executed in the same way that experiments
are run.  Therefore, it is useful for computational scientists to adopt the idea
of a lab notebook.  

A lab notebook is a way of storing information about why something was done in a 
particular way in conjunction with the resultant data.  The corollary concept in
software development is known as logging.  

**Thus every flmake command has the ability to log a message.**  This follows 
the ``-m`` convention from version control systems.  These messages and associated 
metadata is stored in the ``flash.log`` file in the project directory.  

Not every command uses logging; for trivial commands which do not change state
(such as ls-runs) log entries are not needed.  However for more serious commands 
such as run logging is a critical component.  While sensible default messages
will be generated automatically, it is **highly** recommended that the user provide
more detailed messages::

    ~/proj $ flmake -m "Run with 600 J laser" run -n 10

The :ref:`flmake log <ug_flmake_log>` command may then be used to display past log 
messages::

    ~/proj $ flmake log -n 1
    Run id: b2907415
    Run dir: run-b2907415
    Command: run
    User: scopatz
    Date: Mon Mar 26 14:20:46 2012
    Log id: 6b9e1a0f-cfdc-418f-8c50-87f66a63ca82

        Run with 600 J laser

The ``flash.log`` file should be placed under the project's version control.  Entries
in this file are not typically deleted.

abstraction 5: run control
============================
Many aspects of FLASH are declared in a static way.  Such declarations happen mainly
at setup and runtime.  For certain build and run operations several parameters may 
need to be altered in a consistent way to actually have the desired effect.  Such 
repetition can become tedious and usually leads to less readable inputs.

**To make the user input more concise and expressive, flmake introduces a run control
flashrc.py file in the project directory.**  This is Python module which is 
executed, if it exists, in an empty namespace whenever flmake is run.  The 
flmake commands may then choose to access specific data in this file.  Please see 
the individual command documentation for an explanation on if/how the run control
file is used.

The most important example of using ``flashrc.py`` is that the run and restart
commands will update the ``flash.par`` file with values from a ``parameters``
dictionary (or function which returns a dictionary).

Initial ``flash.par``::

    order = 3
    slopeLimiter = "minmod"
    charLimiting = .true.
    RiemannSolver = "hll"

Run control ``flashrc.py``::

    parameters = {"slopeLimiter": "mc",
                  "use_flattening": False}

Final ``flash.par``::

    RiemannSolver = "hll"
    charLimiting = .true.
    order = 3
    slopeLimiter = "mc"
    use_flattening = .true.

example workflow
==================
The fundamental flmake abstractions which affect users have now been explained
above.  Bringing this all together, a typical flmake workflow which sets up, 
builds, runs, restarts, and merges a fork of a Sedov simulation is demonstrated.
First, construct the project repository::

    ~ $ mkdir my_sedov
    ~ $ cd my_sedov/
    ~/my_sedov $ mkdir simulations/
    ~/my_sedov $ cp -r ${FLASH_SRC_DIR}/source/Simulation/SimulationMain/Sedov simulations/
    ~/my_sedov $ nano simulations/Sedov/Simulation_init.F90  # edit the simulation
    ~/my_sedov $ git init .
    ~/my_sedov $ git add .
    ~/my_sedov $ git commit -m "Initialized my Sedov project"

Next, create and run the simulation::

    ~/my_sedov $ flmake setup -auto Sedov
    ~/my_sedov $ flmake build -j 20
    ~/my_sedov $ flmake -m "First run of my Sedov" run -n 10
    ~/my_sedov $ flmake -m "Oops, it died." restart run-5a4f619e/ -n 10
    ~/my_sedov $ flmake -m "Merging my first run." merge run-fc6c9029 first_run
    ~/my_sedov $ flmake clean 1



A Note on Repeatability
---------------------------------
de nada


Acknowledgements
----------------
Dr. Milad Fatenejad provided a superb sounding board in the conception of the flmake utility
and aided in outlining the constraints of reproducibility.

The software used in this work was in part developed by the DOE NNSA-ASC OASCR Flash Center
at the University of Chicago.


References
----------
.. [FLASH] FLASH Center for Computational Science, *FLASH User's Guide, Version 4.0-beta,*
            http://flash.uchicago.edu/site/flashcode/user_support/flash4b_ug.pdf, 
            University of Chicago, February 2012.
.. [Flmake] A. Scopatz, *flmake: the flash workflow utility,* 
            http://flash.uchicago.edu/site/flashcode/user_support/tools4b/usersguide/flmake/index.html,
            The University of Chicago, June 2012.
