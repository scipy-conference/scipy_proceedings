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
