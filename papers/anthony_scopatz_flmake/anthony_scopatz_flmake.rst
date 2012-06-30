:author: Anthony Scopatz
:email: scopatz@flash.uchicago.edu
:institution: The FLASH Center for Computational Science, The University of Chicago

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
range of systems from laptops to workstations to 100,000 processor super computers, such 
as the Blue Gene/P at Argonne National Laboratory.

Historically, FLASH was born from a collection of unconnected legacy codes written
primarily in Fortran and merged into a single project.  Over the past 13 years major
sections have been rewritten in other languages.  For instance, I/O is now implemented
in C.  However building, testing, and documentation are all performed in Python.

FLASH has a unique architecture which compiles *simulation specific* executables for each
new type of run.  This is aided by an object-oriented-esque inheritance model that is
implemented by inspecting the file system directory tree.  This allows FLASH to
compile to faster machine code than a compile-once strategy.  However it also
places a greater importance on the Python build system.

To run a FLASH simulation, the user must go through three basic steps: setup, build, and
execution.  Canonically, each of these tasks are independently handled by the user.
However with the recent advent of flmake - a Python workflow management utility for
FLASH - such tasks may now be performed in a repeatable way [FLMAKE]_.

Previous workflow management tools have been written for FLASH.  (For example, the
"Milad system" was implemented entirely in Makefiles.)  However, none of the prior
attempts have placed reproducibility as their primary concern.  This is in part because
fully capturing the setup metadata required alterations to the build system.

The development of flmake started by rewriting the existing build system
to allow FLASH to be run outside of the main line subversion repository.  It separates out
a project (or simulation) directory independent of the FLASH source directory.  This
directory is typically under its own version control.

For each of the important tasks (setup, build, run, etc), a sidecar metadata
*description* file is either initialized or modified.  This is a simple
dictionary-of-dictionaries JSON file which stores the environment of the
system and the state of the code when each flmake command is run.  This metadata includes
the version information of both the FLASH main line and project repositories.
However, it also may include all local modifications since the last commit.
A patch is automatically generated using standard posix utilities and stored directly 
in the description.

Along with universally unique identifiers, logging, and Python run control files, the
flmake utility may use the description files to fully reproduce a simulation by
re-executing each command in its original state.  While ``flmake reproduce``
makes a useful debugging tool, it fundamentally increases the scientific merit of
FLASH simulations.

The methods described herein may be used whenever
source code itself is distributed.   While this is true for FLASH (uncommon amongst compiled
codes), most Python packages also distribute their source.  Therefore the same
reproducibility strategy is applicable and highly recommended for Python simulation codes.
Thus flmake shows that reproducibility - which is notably absent from most computational science
projects - is easily attainable using only version control, Python standard library modules, 
and ever-present command line utilities.


New Workflow Features
----------------------
As with many predictive science codes, managing FLASH simulations may be a tedious 
task for both new and experienced users.  The flmake command line utility eases the 
simulation burden and shortens the development cycle by providing a modular tool 
which implements many common elements of a FLASH workflow.  At each stage 
this tool captures necessary metadata about the task which it is performing.  Thus
flmake encapsulates the following opperations:

* setup/configuration,
* building,
* execution,
* logging,
* analysis & post-processing,
* and others.

It is highly recommended that both novice and advanced users adopt flmake as it 
*enables* reproducible research while simultaneously making FLASH easier to use.  
This is accomplished by a few key abstractions from previous mechanisms used to set up,
build, and execute FLASH.  The implemntation of these abstractions are 
critical flmake features and are discussed below.  Namely they are the sepaartion 
of project directories, a serachable source path, logging, dynamic run control, and 
persisted metadata descriptions.

Independent Project Directories
=================================
Without flmake, FLASH must be setup and built from within the FLASH source directory
(``FLASH_SRC_DIR``) using the setup script and make [GMAKE]_.  While this is fine for single
runs, it fails to separate projects and simulation campaigns from the source code.
Moreover, keeping simulations next to the source makes it difficult to track local 
modifications independent of the mainline code development.

Because of these difficulties in running suites of simulations from within ``FLASH_SRC_DIR``, 
flmake is intended to be run external to the FLASH source directory.  This is known as the 
project directory.  This directory should be managed by a version control system.
Notably now all of the project-specific files are in a repository that is separate from 
the main FLASH source.   Here this directory is called ``proj/`` but in practice is the 
name of the simulation campaign. 

Source & Project Paths Searching
=====================================
After creating a project directory, the simulation source files must be assembled using
the flmake setup command.  This is analogous to executing the traditional setup script. 
For example, to run the classic Sedov problem:

.. code-block:: sh

    ~/proj $ flmake setup Sedov -auto
    [snip]
    SUCCESS
    ~/proj $ ls
    flash_desc.json  setup/

This command creates symbolic links to the the FLASH source files in the ``setup/`` directory.
Using the normal FLASH setup script, all of these files must live within 
``${FLASH_SRC_DIR}/source/``.  However, flmake's setup command searches additional paths to 
find potential source files.

By default, uf there is a local ``source/`` directory in the projects directory this directory 
is searched first for any potential FLASH units.  The structure of this directory mirrors 
the layout found in ``${FLASH_SRC_DIR}/source/``.  For example, if the user wanted to write or 
overwrite their own driver unit, they could place all of the relevant files in 
``~/proj/source/Driver/``.  Units found in the project source directory take precedence over 
units with the same name in the FLASH source.

The most commonly overridden units, however, are simulations. Furthermore specific simulations 
live somewhat deep in the file system hierarchy residing within 
``source/Simulation/SimulationMain/``.  To make accessing 
simulations easier, a local project ``simulations/`` directory is first searched for any possible 
simulations.  Thus ``simulations/`` effectively aliases ``source/Simulation/SimulationMain/``. 
Continuing with the previous Sedov example the following directories, if they exist, are 
searched  in order of precedence:

#. ``~/proj/simulations/Sedov/``
#. ``~/proj/source/Simulation/``
        ``SimulationMain/Sedov/``
#. ``${FLASH_SRC_DIR}/source/``
        ``Simulation/SimulationMain/Sedov/``

Therefore, it is common for a project directory to have the following structure if the 
project require many modifications to FLASH that are - at least in the short term - 
inappropriate for mainline inclusion:

.. code-block:: sh

    ~/proj $ ls
    flash_desc.json  setup/  simulations/  source/

Logging
======================
In many ways computational simulation is more akin to experimental science than
theoretical science.  Simulations are executed to test the system at hand in analogy 
to how physical experiments probe the netural world..  Therefore, it is useful for 
computational scientists to adopt the time-tested strategy of a keeping a lab notebook.

Various example of virtual lab notebooks exist [VLABNB]_ as a way of storing 
information about how some experiment was performed in a particular way in 
conjunction with the resultant data.  However, the corollary concept in
pure software development is arguablly logging.  Unfortunately, most simulation
science makes use of neither of these two solutions.  Rather, than using an 
external rich-client, flmake makes use of the built-in Python logger.

Every flmake command has the ability to log a message.  This follows 
the ``-m`` convention from version control systems.  These messages and associated 
metadata is stored in a ``flash.log`` file in the project directory. 

Not every command uses logging; for trivial commands which do not change state
(such as listing or diffing) log entries are not needed.  However for more serious commands 
(such as building) logging is a critical component.  Understanding that many users cannot 
be bothered to create meaningful log messages at each step, sensible and default messages
are automatically generated.  Still, it is highly recommended that the user provide
more detailed messages:

.. code-block:: sh

    ~/proj $ flmake -m "Run with 600 J laser" run -n 10

The ``flmake log <ug_flmake_log>`` command may then be used to display past log 
messages:

.. code-block:: sh

    ~/proj $ flmake log -n 1
    Run id: b2907415
    Run dir: run-b2907415
    Command: run
    User: scopatz
    Date: Mon Mar 26 14:20:46 2012
    Log id: 6b9e1a0f-cfdc-418f-8c50-87f66a63ca82

        Run with 600 J laser

The ``flash.log`` file should be added to the version control of the project.  Entries
in this file are not typically deleted.

Dynamic Run Control
============================
Many aspects of FLASH are declared in a static way.  Such declarations happen mainly
at setup and runtime.  For certain build and run operations several parameters may 
need to be altered in a consistent way to actually have the desired effect.  Such 
repetition can become tedious and usually leads to less readable inputs.

To make the user input more concise and expressive, flmake introduces a run control
``flashrc.py`` file in the project directory.  This is a Python module which is 
executed, if it exists, in an empty namespace whenever flmake is called.  The 
flmake commands may then choose to access specific data in this file.  Please refer 
to individual command documentation for an explanation on if/how the run control
file is used.

The most important example of using ``flashrc.py`` is that the run and restart
commands will update the ``flash.par`` file with values from a ``parameters``
dictionary (or function which returns a dictionary).

Initial ``flash.par``:

.. code-block:: sh

    order = 3
    slopeLimiter = "minmod"
    charLimiting = .true.
    RiemannSolver = "hll"

Run control ``flashrc.py``:

.. code-block:: python

    parameters = {"slopeLimiter": "mc",
                  "use_flattening": False}

Final ``flash.par``:

.. code-block:: sh

    RiemannSolver = "hll"
    charLimiting = .true.
    order = 3
    slopeLimiter = "mc"
    use_flattening = .true.

Description Sidecar Files
============================
After performing setup, a ``flash_desc.json`` file appears in the project directory, 
as seen in the above examples.  This is the description file for the FLASH 
simulation which is currently being worked on.  This description is a sidecar
file whose purpose it is to store the following metadata:

* the environment at execution of each flmake command,
* the version of both project and FLASH source repository, 
* local source code modifications (diffs),
* the run control files (see above),
* run ids and history, 
* and FLASH binary modification times.

Thus the description files is meant to be a full picture of the way FLASH
code was generated, compiled, and executed.  Total reproducibility of a FLASH
simulation is based on having a well-formed description file.

The contents of this file are essentially a persisted dictionary which contains 
all of the above information.  The top level keys include setup, build, run, 
and merge.  Each of these keys gets added when the corresponding flmake command is
called.  Note that restart alters the run value and does not generate its own 
top-level key.

During setup and build, ``flash_desc.json`` is modified in the project directory.
However, each run receives a copy of this file in the run directory with the run
information added.  Restarts and merges inherit from the file in the previous run 
directory.

These seidecar files enable the flmake reproduce command which is capable of 
recreating a FLASH simulation from only
the ``flash_desc.json`` file and the associated source and project repositories.  
This is useful for testing and verification of the same simulation across multiple 
different machines and platforms.
It is generally not recommended that users place this file under version control
as it may change often and significantly.

Example Workflow
=====================
The fundamental flmake abstractions have now been explained
above.  A  typical flmake workflow which sets up, 
builds, runs, restarts, and merges a fork of a Sedov simulation is demonstrated.
First, construct the project repository:

.. code-block:: sh

    ~ $ mkdir my_sedov
    ~ $ cd my_sedov/
    ~/my_sedov $ mkdir simulations/
    ~/my_sedov $ cp -r ${FLASH_SRC_DIR}/source/Simulation/\
                 SimulationMain/Sedov simulations/
    ~/my_sedov $ # edit the simulation
    ~/my_sedov $ nano simulations/Sedov/Simulation_init.F90  
    ~/my_sedov $ git init .
    ~/my_sedov $ git add .
    ~/my_sedov $ git commit -m "Initialized my Sedov project"

Next, create and run the simulation:

.. code-block:: sh

    ~/my_sedov $ flmake setup -auto Sedov
    ~/my_sedov $ flmake build -j 20
    ~/my_sedov $ flmake -m "First run of my Sedov" run -n 10
    ~/my_sedov $ flmake -m "Oops, it died." restart \
                 run-5a4f619e/ -n 10
    ~/my_sedov $ flmake -m "Merging my first run." merge \
                 run-fc6c9029 first_run
    ~/my_sedov $ flmake clean 1


Why Reproducibility is Important
----------------------------------
True to its part of speech, much of 'scientific computing' has the trappings of 
science in that it is code produced to solve problems in (big-'S') Science.  
However, the process by which said programs are written is not itself typically 
itself subject to the rigors of the scientific method.  The vaulted method contains 
components of prediction, experimentation, duplication, analysis, and openess 
[GODFREY-SMITH]_.  While software engineerers often engage in such activites when 
programming, scientific developers usually forego these methods, typically to their 
detriment [WILSON]_.

Whatever the reason for this may be - ignorance, sloth, or other deadly sins - 
the impetus for adopting modern software development practices only increases 
every year.  The evolution of tools such as version control and envrionemnt 
reproducing mechanisms (via virtual machines/hypervisors) enable researchers to 
more easily capture information about the software during and after production.  
Furthermore, the appearent end of Silicon-based Moore's Law has nececitated a move
to more exotic arichteture and increased parallelism to see further speed 
increases [MIMS]_. This implies that code that runs on machines now may not
be able to run on future processors without significant refactoring.  

Therefore the scientific computing landscape is such that there are presently the
tools and the need to have fully reproducible simulations.  However, most scientists
choose to not utilize these technologies.  This is akin to a chemist not keeping a
lab notebook.  Thus lack of reproducuibility means that many solutions to science
problems garnered through computational means are relegated to the realm of technical 
achievements.  Irreproducible reults may be novel and interesting but they are not 
science.  Instead of the current paradigm of periscientific computing 
(computing-about-science), the community should redouble our efforts around 
diacomputiational science (computing-throughout-science).

The above being generally true, there are a couple of important caveats.  The first
is that there are researchers who are congnizant and repectful of these reproducibility
issues.  The efforts of these scientists help paint a less dire picture than the 
one framed above.  

The second exception is that while reproducibility is a key feature of fundemental science 
it is not the only one.  For example, openness is another point whereby the statement
"If a result is not produced openly then it is not science" holds.  Open access to 
reults - itself is a hotly contested issue [VRIEZE]_ - is certainly a component of 
computational science.  Additionally though, having open source and available code 
is likely critical and often outside of normal research practice.  This is for a 
vareity of reasons, including the fear that releasing code too early or at all will 
negatively impact personal publication records.

Therefore reproducibility is imporant because without it any results generated are 
periscientific.  For computational science there exist computational tools to aid 
in this endeavour, as in analouge science there are physical solutions.  Though it
is not the only critism to be levied against modern research practices, irreproducibility
is one that affects computation acutely and uniquely compared to other spheres.


The Reproduce Command
----------------------------

The ``flmake reproduce`` command is the key feature enabling the total reproducibility
of a FLASH simulation.  This takes a description file (eg ``flash_desc.json``) and implicitly 
the FLASH source and project repositories and replays the setup, build, and run commands 
originally executed.  Thus it has the following usage string:

.. code-block:: sh

    flmake reproduce [options] <flash_descr>

For each command, reproduction works by cloning both source and project repositories at a 
the point in history when they were run into temporary directories.  Then any local 
modifications which were present (and not under version control) are loaded from the 
description file and applied to the cloned repos.  It then copies out the run control 
file to the cloned repos and performs and command-specific modifications needed.  Finally,
it executes the appropriate command *from the cloned repository* using the original 
arguments provided on the command line.  Therefore ``flmake reproduce`` recreates the 
original simulation using the original commands (and not the versions currently present).

The reproduce command has the following limitations:

#. FLASH source directory must be under version control,
#. Project directory must be under version control,
#. The FLASH run must depend on only the parfile, the FLASH executable and 
   FLASH DATAFILES: This just means that you can’t reproduce the run if FL
   FLASH depends on random files that are not tracked by version control 
   since at a future date, those files might not be available,
#. and the user cannot modify the FLASH executable between building and run


A Note on Repeatability
---------------------------------
de nada


Conclusions
------------------------------


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
.. [FLMAKE] A. Scopatz, *flmake: the flash workflow utility,* 
            http://flash.uchicago.edu/site/flashcode/user_support/tools4b/usersguide/flmake/index.html,
            The University of Chicago, June 2012.
.. [GMAKE] Free Software Foundation, The GNU Make Manual for version 3.82, 
            http://www.gnu.org/software/make/, 2010.
.. [VLABNB] Rubacha, M.; Rattan, A. K.; Hosselet, S. C. (2011). *A Review of Electronic 
            Laboratory Notebooks Available in the Market Today*. Journal of Laboratory 
            Automation 16 (1): 90–98. DOI:10.1016/j.jala.2009.01.002. PMID 21609689. 
.. [GODFREY-SMITH] Godfrey-Smith, Peter (2003), *Theory and Reality: An introduction to 
            the philosophy of science*, University of Chicago Press, ISBN 0-226-30063-3.
.. [WILSON] G.V. Wilson, *Where's the real bottleneck in scientific computing?* Am Sci. 
            2005;94:5.
.. [MIMS] C. Mims, *Moore's Law Over, Supercomputing "In Triage," Says Expert,*
            http://www.technologyreview.com/view/427891/moores-law-over-supercomputing-in-triage-says/
            May 2012, Technology Review, MIT.
.. [VRIEZE] Jop de Vrieze, *Thousands of Scientists Vow to Boycott Elsevier to Protest Journal 
            Prices,* Science Insider, February 2012.
