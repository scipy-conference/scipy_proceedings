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

ICF is fun.  It uses computers and now python
 
Parallel python interpreters for pre-existing programs
------------------------------------------------------

We added python to Hydra.  It was hard.

Automation of target design with Python
---------------------------------------

One of the main applications of HYDRA is in inertial confinement fusion (ICF) capsule design.  ICF targets are typically small (~1 mm radius) spheres composed of several layers of cryogenic hydrogen, plastic, metal or other materials.  The intention is to produce significant thermonuclear yield by spherically compressing the hydrogen in the capsule to very large temperature and density.  The implosion is driven by a high intensity driver which illuminates, heats, and ablatesg the outer surface of the capsule.  This ablation pressure drives the implosion.

Consider the shock ignition approach to inertial fusion using lasers.  A spherical shell of frozen deuterium-tritium (DT) is compressed to high density (~x g/cc) by a sequence of moderate strength shocks.  It is then acclerated to moderate implosion velocity (~300 km/s).  When the imploding shell stagnates, it forms a central, low density, high temperature hot spot and a surrounding high density, low temperature shell.  A strong shock wave (the igniter shock) arrives at stagnation and further heats the hot spot as well as preventing the low pressure shell from coming into pressure equilibrium with the high pressure hot spot.  The combination of the stagnation of the shell and the timely arrival of a the igniter shock lifts the temperature of the central hot spot above the 12 keV threshold needed to initiate s fusion burn wave.  This burn wave propagates into the cold shell, producing most of the fusion yield.

Producing the needed shocks requires precise control of the time dependent driver power.  Driver powers ran

Compression shocks must be timed to breakout into the DT gas at the same time ("shock timing").  Main pulse should produce peak :math:`\rho R`.  Igniter pulse should produce maximum yield.

.. figure:: rt_materials.pdf

    Change me to include the laser profile.

This is a subsection on shock syncing
.....................................

Here I talk about how to syncronize shocks

.. figure:: auto_timing.pdf

    Change me to all guide lines for early and late.


Tuning the Main Pulse and Igniter Pulse
...........................................

Finding optimal main and igniter pulse timings are simple optimization problems.  Since the igniter pulse is responsible for actually igniter the target, the main pulse should maximize the potential burn.  The burn fraction scales with the peak areal density (:math:`\rho R`) of the assembled target 
(:math:`f \approx \frac{\rho R}{\rho R + 7}`) where 
(:math:`\rho R = \int \rho(r) dr`).  We use a modified bisection optimization method described in the following section for actual optimization.  For the particular target we under consideration, peak areal density is about 1.5, corresponding to a theoretical burn fraction of 20% and a yield of 40 MJ.  Note that this estimate does not take into account the ablation of the DT during the main pulse.  We require our optimization to converge within xx ps.  In Figure :ref:`figrhor`, we see that :math:`\rho R` peaks and is approximately flat over a xxps interval.

.. figure:: rhor_tune.pdf

    Tuning peak areal density :label:`figrhor`

Having fixed the main main pulse timing, we add the igniter pulse.  We tune the start of the igniter pulse to maximize fusion yield.


Optimization Techniques
-----------------------

Typical calculations take 5-20 minutes on a single core of an 2.8 GHz Intel Xeon processor.  Typical single variable optimization methods are designed for serial evaluation.  A "quick" convergence might take 12 function evaluations, translating to approximately four hours of run time.  Instead, we use a simple parallel bounded minimum optimization with 8 simultaneous evaluations.  We routinely achieve acceptable convergence within 3 iterations.
.. TODO: fix this sentence
Better sampling techniques and smaller intervals could likely reduce the number of iterations or reduce the number of parallel function evaulations, but for the initial scope of this paper things are fine.


References
----------

This work performed under the auspices of the U.S. DOE by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

