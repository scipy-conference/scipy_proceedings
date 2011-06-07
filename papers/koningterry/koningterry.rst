:author: Matt Terry
:email: terry10@llnl.gov
:institution: Lawrence Livermore National Laboratory

:author: Joseph Koning
:email: koning1@llnl.gov
:institution: Lawrence Livermore National Laboratory

-------------------------------------------------------
Automation of Inertial Fusion Target Design with Python
-------------------------------------------------------

.. class:: abstract

    Inertial confinement fusion (ICF) is a means to achieve controlled thermonuclear fusion by way of compressing hydrogen to extremely large pressures, temperatures and densities.  ICF uses a high intensity driver to compress a spherical shell of cryogenically frozen fuel to more than 100 times solid density and imploding the shell at sufficient velocity that it stagnates with pressures of more than 100 GBar.  At stagnation, a fusion burn wave propagates from a central, low-density hot spot to a colder high-density fuel region.  The inertia of the fuel keeps it intact long enough for a significant fraction of the fuel to burn.

    Getting to these extreme conditions requires the driver to have a carefully designed, time dependent intensity profile.  The shape of which depends on many different physical processes in the target. The most important processes are hydrodynamic flow, radiative energy transfer, electron thermal conduction, equation of state and the energy deposition of the driver.  Performing experiments is complicated and expensive, so the ICF community relies on sophisticated multi-physics codes, such as HYDRA, to design experiments and simulate experimental measurements prior to fielding the experiment.

    The process of tuning a driver pulse shape to get the desired performance characteristics is a high latency, highly iterative, interactive process.  However, we have developed several techniques that can be used to automate much of the pulse tuning process.  These methods make use of Python in automating parameter scans, templated input file instantiation, and post-processing of simulations.  In addition to its physics capabilities, HYDRA provides an embedded Python interpreter, which facilitates the development of powerful, user defined, in code diagnostics and simulation controllers.  In this paper we will discuss the embedding of they Python interpreter in HYDRA and the automation techniques that it helps to facilitate.


.. class:: keywords

   inertial confinement fusion, python, automation

Introduction
------------

ICF is fun.  It uses comptuers and now python
 
Parallel python interpeters for pre-existing programs
-----------------------------------------------------

We added python to Hydra.  It was hard.

Automation of target design with python
---------------------------------------

Look at the cool stuff we can do with our new bindings.

This work performed under the auspices of the U.S. DOE by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.


References
----------


