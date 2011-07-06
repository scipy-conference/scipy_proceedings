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

    The process of tuning an inertial confinement fusion pulse shape to a specific target design is highly iterative process.  When done manually, each iteration has large latency and is consequently time consuming.  We have developed several techniques that can be used to automate much of the pulse tuning process and significantly accelerate the tuning process by removing the human induced latency.  The automated data analysis techniques require specialized diagnostics to run within the simulation.  To facilitate these techniques, we have embedded a looselly coupled Python interpreter within a pre-existing radiation-hydrodynamics code, Hydra.  To automate the tuning process we use numerical optimization techniques and construct objective functions to identify tuned parameters.
    
.. class:: keywords

   inertial confinement fusion, python, automation

Inertial Confinement Fusion
---------------------------

Inertial confinement fusion (ICF) is a means to achieve controlled thermonuclear fusion by way of compressing hydrogen to extremely large pressures, temperatures and densities.  ICF capsule are typically small (~1 mm radius) spheres composed of several layers of cryogenic hydrogen, plastic, metal or other materials.  The intention is to produce significant fusion yield by spherically compressing the hydrogen in the capsule to very large temperature, density and pressure.  These extreme conditions are reached by illuminating the capsule with a very high intensity (100's TW) driver.  This compresses a shell of cryogenically frozen fuel to more than 100 times solid density and accelerates the radially converging shell to very high velocity (300 km/s).  As the shell stagnates, a fusion burn wave propagates from a central, low-density, high temperature region to a surrounding high-density, low temperature fuel region.  The inertia of the fuel keeps it intact long enough for a significant fraction of the fuel to burn.

There are several approaches to achieving a sustained fusion burn, but for this paper consider the shock ignition approach with the capsule directly driven by by lasers.  The capsule is a spherical shell of frozen deuterium-tritium (DT), coated with plastic or another ablator material.  Laser beams directly illuminate the target and deposit energy in the outer most layer called the ablator.  The ablation of the ablator supplies the pressure to drive the implosion.  We assume a spherically symmetric illumination of the capsule with total laser power incident changing with time, referred to here as the "pulse shape."

We divide the pulse shape into three logical sections, which correspond to three phases of the capsule implosion dynamics. The first section as the "pre-pulse" and is responsible for shock compression the DT shell to high density.  The pre-pulse consists of a short, high intensity spike in the laser power (the "picket") and thre pedestals, each with increasing laser power.  The pre-pulse is followed by the main pulse, which accelerates the shell to moderate implosion velocity (~300 km/s).  When the imploding shell stagnates, it forms a central, low density, high temperature hot spot and a surrounding high density, low temperature shell.

The final section of the pulse shape is the igniter pulse.  The ignite pulse consists of another pedestal of very high intensity.  This section launches a strong shock that arrives just as the shell is stagnating and further heats the hot spot as well as prevents the low pressure shell from coming into pressure equilibrium with the high pressure hot spot.  The combination of the stagnation of the shell and the timely arrival of the igniter shock lifts the temperature of the central hot spot above the 12 keV threshold needed to initiate a fusion burn wave.  This burn wave propagates into the cold shell which produces most of the fusion yield.

.. figure:: rt_materials.pdf

    A Radius-Time plot of the capsule implosion with the incident laser power overlay.  Lines plot the trajectory of fluid particle boundaries.  Lines are color coded by material.  TODO add the laser power.

The extreme conditions required to initiate a propagating fusion burn are difficult to attain and require balancing the constraints of driver performance, target construction, and the many underlying physical processes.

Futhermore, once a general target and pulse design is chosen, the pulse must be tuned to the specific details of the target.

Reaching these extreme conditions requires the driver to have a carefully designed, time dependent intensity profile.  The shape of which depends on many different physical processes in the target. The most important processes are hydrodynamic flow, radiative energy transfer, electron thermal conduction, equation of state and the energy deposition of the driver.  Performing experiments is complicated and expensive, so the ICF community relies on sophisticated multi-physics codes, such as Hydra, to design experiments and simulate experimental measurements prior to fielding the experiment.

Hydra as design tool.


Structure of Automatic Tuning
-----------------------------

We adopt the general strategy that a tuned pulse can be constructed by serially adding tuned pulse segments.  Additionally we require that we can "tune" each property of a pulse segment by numerically optimizing an appropriately chosen objective function.  Therefore, given a sequence of pulse properties and objective functions we can construct a program that will automatically tune the pulse.  It is important to realize that the choice and sequence of properties and objective functions embodies a strategy to achieve the desired target behavior.  The automation of this strategy does not guarantee the the tuned pulse/target will have the desired performance characteristics, just that the design strategy was faithfully executed.

In our auto-tuner we must systematically generate simulations and process those simulations.

Auto-tuning will therefore require generating many simulations that are only slight variations on nominal template simulation.   Additionally, we must automate the gathering of data from these simulations.

Mention uncertainly quantification community and tools like Dakota.

We generate simulations by means of a Python proxy for the Hydra input files.  The proxy has simple pre-processor like capabilities for modifying simple input file statements and for injecting more complicate structures into the input file by overwriting specially formatted directives.  For more complicated input file structures, we delegate responsibility to special purpose objects.  We adopt the convention that the string representation of an object (``str(obj)``) is appropriately formatted for insertion into a Hydra input file.  Furthermore, string conversion happens when an input file is generated.  This makes it easy to evolve the simulation parameters as various parameters are tuned.

Data gathering is more complicated than post-processing output files.  We do not know a priori when a watched for even will occur.  To have sufficient time resolution must either make very frequent data dumps or modify Hydra to be more introspective.  The following section discusses the addition of a parallel Python interpreter to Hydra.  Without this, the data retention requirements for auto-tuning would have been prohibitive.


Parallel Python interpreters for pre-existing programs
------------------------------------------------------

Hydra is a massively parallel multi-physics code in use since 1993. 
The code 
combines hydrodynamics with radiation diffusion, laser ray trace, 
and several more packages necessary for ICF design
and has over 40 users at national laboratories and universities. 

Hydra users set up their simulations using a built-in interpreter. The 
existing interpreter provides access to the program parameters
and provides functions to access and manipulate the data in parallel. Users
can access and alter the state while the simulation is running through
a message interface that runs at a specific cycle, time or if a specific
condition is met. 

To improve functionality the Python interpreter was added to Hydra.
Python was chosen 
due to the mature set of embedding API and extending tools
and the large number of third party libraries.  

The Hydra interpreter was augmented by embedding the
Python interpreter instead of extending Python itself.
The legacy Hydra interpreter was kept due to the large number of
existing input files or decks that could not be easily ported to a new
syntax.  The SWIG interface generator is used to wrap the Hydra C++ classes
and C functions.


The users can send commands to the Python interpreter using three separate 
methods a custom interactive interpreter based on the CPython interpreter;
a generic code module based interactive interpreter; and a file-based Python code block interpreter.

The Hydra code base is based on the message passing interface 
(MPI) library. This MPI library allows for efficient communication of data 
between processors in a simulation. The interactive and file based methods
need to have access to the Python source on all of the processors used in the simulation. The MPI library is used to read a line from stdin or an entire file on the root processor and broadcast this data to all of the other processors in the simulation. The simplest method to provide an interactive parallel Python interpreter would be to override the PyOs_Readline function in the Python code base.  Unfortunately, this function cannot be overridden so an alternative Python interpreter was developed to handle the parallel stdin access.  The parallel file access reads the entire file in as a string and broadcasts this string to all of the other processors. The string is then sent through the embedded Python interpreter function PyRun_SimpleString. This C function will take a char pointer as the input and run the string through the same parsing and interpreter calls as a file using the Python program. 


.. code-block:: c

   void runpycode(char* pystr) {
     PyRun_SimpleString(pystr);
   }

One limitation of the PyRun_SimpleString call is the lack of exception 
information. To alleviate this issue a second method was implemented that 
uses a file name or input deck information to give a better location for 
the exception. 

.. code-block:: c

   void runpycode(char* pystr) {
     pysrc = Py_CompileString(str, pyinput , start);
     v = PyEval_EvalCode((PyCodeObject*) pysrc, 
                          pmainDict, pmainDict);
   }


 description of interactive parser

With the above embedded Python support users can run arbitrary Python code 
through the Python interpreter. One of the mandates of the effort to embed 
the Python interpreter was to provide an enhanced version of the existing Hydra 
interpreter.  In order to provide this functionality Python must be able to 
access the information in the running Hydra simulation. This is accomplished
by wrapping the Hydra data structures, functions, and parameters using the 
Simplified Wrapper and  Interface Generator (SWIG). The embedded Python is 
extended by a module called hydra.  The code created by SWIG includes a C++ 
file compiled into Hydra as a Python extension library and a Python interface
file that is seralized and compiled into the Hydra code.

The main reason for the hydra module is to allow users to access the Hydra 
state. Hydra has several types of integer and floating point arrays ranging 
from one to three dimensional.  The multi dimensional arrays
have an additonal index to indicate the block.  The block defines a 
portion of the mesh on which the zonal, nodal, edge, and face base information
is defined.  Meshes can consist of several blocks.  These blocks are then 
decomposed into sub-blocks or domains depending on how many processors will 
be used in the simulation. Access to the multi-block parallel data structures
is provided by structures wrapped by C++ interface objects and then wrapped in 
SWIG using numpy as the array object in Python.

Objects in the top level, __main__, state are saved to a restart file.
This restart file is a portable file object written through 
the silo library interface. The restart state is a binary string
created through the pickle interface. The Python module used for the state 
saving functionality is the save state module by Oren Tirosh located at the activestate website [OT08]_. This module 
has been augmented with the addition of numpy support and None and Ellipsis Singleton object support.

Multiple versions of the Hydra code are available to users at any given time.
In order to add additional functionality and maintain version integrity, the hydra Python module is embedded in the Hydra code as a frozen module. The Python file resulting from the SWIG generator is marshaled using a script based on the freeze module in the Python distribution. This guarantees the modules
are always available even if the sys path is altered.

Message and callback information.


Automation of target design with Python
---------------------------------------

Producing the needed shocks requires precise control of the time dependent driver power.  Driver powers range three orders of magnitude.  Compression shocks must be timed to breakout into the DT gas at the same time ("shock timing").  Main pulse should produce peak :math:`\rho R`.  Igniter pulse should produce maximum yield.

Simply writing the tuning algorithm in paragraph form suggests that tuning could be performed purely in software.  Furthermore, if we can construct an appropriate objective function for each tuning, we can make use of powerful, mature optimization methods.

Parameterization of the pulse shape

We need an appropriately parameterized pulse shape and the ability to construct that pulse shape within Hydra.  


Embedded processing
-------------------

Our simulations must be appropriate parameterized so that they can be called as if they were simply expensive functions calls.  Additionally, we must gather the appropriate information from the running simulations.

Characteristic trackers.  The Euler equations.  Characteristic 
:math:`\dot{r} = v(r) - c_s(r)`   Hydra's Python interface exposes the needed variables and provides a means for registering callback functions.  Conveniently add arbitrary 

Dynamic steering of problem.  Characteristic trackers for locating breakout.  Advantage of operating independent of mesh and robust to motion of grid from pre-heat or spurious grid motion.  Makes measurement of "breakout time" and its associated objective function much far less noisy and thus more tractable for algorithmic optimization.

Embedding makes execution faster by easily ending the calculation when the desired has been extracted.  Especially important since it is hard to predict the time when important events will happen and to apply the appropriate resolution.

Use of the same language simplifies

Proxy classes and code generators.  Input file templates, ``str()`` for the Hydra representation and ``repr()`` for the .  Pickling was an option, but does allow for easy modification.


Synchronizing Shock Arrival
...........................

One of the key properties of shocks in ICF is that shocks launched later propagate faster and will eventually overtake the one launched before it.  We make the design decision that shocks should be timed such that the coalesce at the gas/ice interface.  This prevents strong shocks from forming by shock coalescence.  By timing them to coalesce at the gas/ice interface, we minimize the intensification of shocks due to radial convergence.

.. figure:: auto_timing.pdf

    Change me to all guide lines for early and late.  :label:`figtiming`

Consider the case of radially converging shocks launched at two different times from comparable radii.  The second shock is faster and will eventually overtake the first.  If we define a "shock breakout time" as when the first shock enters the gas region, we can plot the shock breakout time as a function of the launch time of the second shock (black line in :ref:`figtiming`).  The appropriate objective function should maximize the breakout time (recognizing that it saturates for large launch times) while also minimizing the launch time of the second shock.  We construct an aggregate objective function as a linear combination of the two constraints (:math:`f(t) = \omega t - b(t)`).  We find an tuned value of :math:`0.01 m`.  Where :math:`m` is the slope between two points chosen to be clearly early and later than ideal tuning.
.. Comments on error


Tuning the Main Pulse and Igniter Pulse
.......................................

Finding optimal main and igniter pulse timings are simple optimization problems.  Since the igniter pulse is responsible for actually igniter the target, the main pulse should maximize the potential burn.  The burn fraction scales with the peak areal density (:math:`\rho R`) of the assembled target 
(:math:`f \approx \frac{\rho R}{\rho R + 7}`) where 
(:math:`\rho R = \int \rho(r) dr`).  We use a modified bisection optimization method described in the following section for actual optimization.  For the particular target we under consideration, peak areal density is about 1.5, corresponding to a theoretical burn fraction of 20% and a yield of 40 MJ.  Note that this estimate does not take into account the ablation of the DT during the main pulse.  We require our optimization to converge within xx ps.  In Figure :ref:`figrhor`, we see that :math:`\rho R` peaks and is approximately flat over a xxps interval.

.. figure:: rhor_tune.pdf

    Tuning peak areal density :label:`figrhor`

Having fixed the main main pulse timing, we add the igniter pulse.  We tune the start of the igniter pulse to maximize fusion yield.


Optimization Techniques
-----------------------

Typical calculations take 5-20 minutes on a single core of an 2.8 GHz Intel Xeon processor.  Typical single variable optimization methods are designed for serial evaluation.  A "quick" convergence might take 12 function evaluations, translating to approximately four hours of run time.  Instead, we use a simple parallel bounded minimum optimization with 8 simultaneous evaluations.  We routinely achieve acceptable convergence within 4 iterations (3x speedup).  The use of more sophisticated sampling techniques would likely reduce the number of iterations or the number of parallel function evaluations.


Conclusions
-----------

Python is awesome!

This work performed under the auspices of the U.S. DOE by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

References
----------
.. [OT08] O. Tirosh, *Pickle the interactive interpreter state (Python recipe)*,
           http://code.activestate.com/recipes/572213-pickle-the-interactive-interpreter-state/ , 2008.



