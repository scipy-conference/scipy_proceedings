:author: João Lemes Gribel Soares
:email: jgribel@usp.br
:institution: Escola Politécnica of the University of Sao Paulo

:author: Mateus Stano Junqueira
:email: mateusstano@usp.br
:institution: Escola Politécnica of the University of Sao Paulo.

:author: Oscar Mauricio Prada Ramirez
:email: oscarmprada@usp.br
:institution: Escola Politécnica of the University of Sao Paulo.

:author: Patrick Sampaio dos Santos Brandão
:email: patricksampaio@usp.br
:institution: Escola Politécnica of the University of Sao Paulo.
:institution: École Centrale de Nantes

:author: Adriano Augusto Antongiovanni
:email: adrianoaugusto98@usp.br
:institution: Escola Politécnica of the University of Sao Paulo.

:author: Giovani Hidalgo Ceotto
:email: giovani.ceotto@alumni.usp.br
:institution: Escola Politécnica of the University of Sao Paulo.

------------------------------------------------------------------------------------------------------------
RocketPy: Combining Open-Source and Scientific Libraries to Make the Space Sector More Modern and Accessible
------------------------------------------------------------------------------------------------------------

.. class:: abstract

   In recent years we are seeing exponential growth in the space sectors and with that more people are fascinated and
   eager to participate in the field. However, rocketry is a very inaccessible field, yet extremely exciting. To make it
   more accessible, people need an active community and easy to use tools. Currently, there isn't a simulator that
   combines accuracy with the flexibility of creating complex simulations, while being easy-to-use and intuitive, with an
   active community maintaining it.
   
   RocketPy is here to solve this! RocketPy is a next-generation trajectory simulation solution for high-power rocketry
   built using SciPy and the Python Scientific Environment. The code allows for a sophisticated 6 degrees of freedom
   simulation of a rocket's flight trajectory, including high fidelity variable mass effects as well as descent under
   parachutes. All of this is packaged into an architecture that facilitates complex simulations, such as multi-stage
   rockets, design and trajectory optimization and dispersion analysis. Weather conditions, such as wind profile, can
   also be imported from detailed forecasts and reanalysis models, allowing for realistic simulations.
   
   While RocketPy has a sophisticated feature set, it is also accessible for anyone interested, as the code is well
   documented, and the repository`s GitHub page is filled with example Jupyter Notebooks that people can adapt for
   their specific use case. At the same time, RocketPy is also a well validated software, as detailed in the paper
   published at the Journal of Aerospace Engineering, comparing RocketPy`s result with rocketry flight data.

   .. TODO: Rewrite the following two paragraphs, since this is not a "talk" but rather an "article".

   This talk is intended for anyone who wants to get involved in the Space Sector and will show how RocketPy is a
   great entrance door to a welcoming community. It will be an opportunity to learn basic concepts about rocketry and
   to get to know this tool used by various rocketry groups around the world.
   
   By the end of the talk, the audience will have learned about the RocketPy`s architecture, how to simulate simple
   rockets, and hopefully will be motivated to join the RocketPy community, as members from diverse backgrounds can
   help us enhance and grow this project!

.. class:: keywords

   space, rocketry, rocket, simulation, trajectory, dynamics, ode, rocket trajectory simulation, sounding rockets, 
   high-powered rockets, flight dynamics, six degrees of freedom, Monte Carlo analysis, stochastic simulation

Introduction
============
.. First author: Ciclope

When it comes to rockets, there is a wide field ranging from orbital rockets to toy rockets. 
Between them, two types will be noted: sounding rockets and high-powered rockets (HPRs). 
Sounding rockets are mainly used by government agencies for scientific experiments in suborbital 
flights and HPR are mainly used for educational purposes, with increasing popularity among student competitions, 
such as SpacePort America Rocket Competition, with more than 100 teams from all over the world and happening annually. 
Also, with the reaching of the Kármán line by the university-built rocket TRAVELER IV (Aitoumeziane et al. 2019), 
both types of rockets are converging to a similar flight trajectory.

With this convergence, HPR are becoming bigger and more robust, increasing their potential hazard, along with their capacity, 
making safety an important issue.  Moreover, performance is always a requirement both for saving costs, 
and time and to accurately reach competition and scientific experiment goals.

In that scene, many parameters should be determined before launching an HPR for both safety and performance, 
such as the landing site coordinates, increasing safety and possibility of recovering the rocket (Wilde 2018); 
apogee altitude, avoiding collision with aircraft and maintaining the ideal altitude for the rocket to function.

To better attend to those issues, RocketPy was created as a computational tool that can accurately predict all dynamic parameters 
involved in the flight of sounding, model, and HPR, given parameters such as the rocket geometry, motor characteristics, 
and environmental conditions. It is also open source, well structured, and well documented, 
allowing minor changes to bring new features (ref: RocketPy)


Background 
==========

In this section, we clarify some terms specific to the rocketry field that may help to better understand the text.

Rocketry terminology
--------------------
.. First author: Ciclope

Apogee: the point at which a corpus is furthest from earth
Degrees of freedom: maximum number of independent values in an equation
Flight Trajectory: the 3-dimensional path, over time, of the rocket during its flight
Launch Rail: Guidance for the rocket to accelerate to a stable flight speed
Powered Flight: phase of the flight where the motor is active
Free Flight: phase of the flight where the motor is inactive and no other component 
but its inertia is influencing the rocket's trajectory
Standard Atmosphere: Average pressure, temperature, and air density for various altitudes
Nozzle: part of the rocket’s engine that accelerate the exhaust gases
Static hot-fire test: Test to measure the integrity of the motor and determine its thrust curve
Thrust Curve: Thrust overtime of a motor
Static Margin: Is a non-dimensional distance to analyze the stability
Nosecone: The forwardmost section of a rocket-shaped conically for aerodynamics
Fin: flattened append of the rocket providing stability during flight, keeping it in the flight trajectory


Flight Model
------------
.. First authors: Oscar/Gui

The flight model of a high-powered rocket takes into account at least three different phases:

1. The first phase consists of a linear movement along the launch rail:
The motion of the rocket is restricted to one dimension, which means that only the translation along the rail needs to be modelled. 
During this phase, four forces can act on the rocket: weight, engine thrust, rail reactions, and aerodynamic forces.

2. After completely leaving the rail, a phase of 6 degrees of freedom (DOF) is established, 
which includes powered flight and free flight:
The rocket is free to move in three-dimensional space. 
In this phase the weight, engine thrust, normal and axial aerodynamic forces are still important.

3. Once apogee is reached, a parachute is usually deployed, characterizing the third phase of flight:
the parachute descent.
In the last phase, the parachute is launched from the rocket, which is usually divided into two
or more parts joined by ropes. 

.. multibody dynamics is taken into account during descent.

Design: RocketPy Architecture
=============================
.. First authors: Oscar/Gui
   Length: 4/15 columns

There are four main classes that organize the dataflow during the simulations: motor, rocket, environment and flight. (cite:`ceotto2021rocketpy`).
Acctually there is also a helper class named `function`, which will be described further.
In the motor class, the main physical and geometric parameters of the motor are configured, 
such as: nozzle geometry, grain parameters, mass, inertia and thrust curve.
This first class acts as an input to the Rocket class where the user is also asked to define certain parameters of the rocket 
such as the inertial mass tensor, geometry, drag coefficients and parachute description. 
Finally, the Flight class joins the rocket and motor parameters with information from the Environment class, 
such as wind, atmospheric and earth models, to generate a simulation of the rocket's trajectory.
This modular architecture, along with its well-structured and documented code, facilitates complex simulations, 
starting with the use of Jupyter Notebooks that people can adapt for their specific use case.
The figure :ref:`fig1` illustrates RocketPy architecture. 

.. figure:: Fluxogram.png
   :align: center
   :scale: 24%
   :figclass: bht

   RocketPy classes interaction :label:`fig1`

Function
--------
.. First authors: Gribel
   (Talk a bit about the motivations behind Function class and what it is trying to solve.
   Go over its main features such as naturally doing algebra, interpolation/extrapolation, evaluating, differentiation/integration and plotting.
   Explain how rocketpy interpolations are faster than usual numpy/scipy methods due to utilizing values from previous interpolations - )
   Discuss ease-of-use vs. efficiency. Show an example.

Variable interpolation meshes/grids from different sources can lead to problems regarding coupling different data types. 
In order to solve this, RocketPy employs a dedicated *Function* class which allows for more natural and dynamic handling 
of these objects, structuring them like :math:`\mathbb{R}^n \to \mathbb{R}^n` mathematical functions. 

Through the use of magic methods, this approach allows for quick and easy arithmetic operations
between lambda expressions and list-defined interpolated functions, as well as scalars. 
Different interpolation methods are available to be chosen from, among them simple polynomial, spline and Akima (ref. paper original). 
Extrapolation of *Function* objects outside the domain constrained by a given dataset is also allowed.

Furthermore, evaluation of definite integrals of these *Function* objects is among their feature set. By cleverly exploiting
the choosed interpolation option, RocketPy calculates the values fast and precisely through the use of different 
analytical methods.  If numerical integration is required, the class makes use of SciPy's implementation of the QUADPACK Fortran library (citar referencia).
For 1-dimensional Functions, evaluation of derivatives at a point is made possible through the employment of a simple finite difference method.

.. melhorar parágrafo acima

Finally, in order to increase usability and readibility, all *Function* objects instances are callable and can be presented 
in multiple ways depending on the given arguments. If no argument is given, a matplotlib figure opens and a plot
of the function is shown inside it's domain. This is especially useful for [the post-processing methods where various
information on the multiple classes is presented, providing for more concise code]. If a n-sized array is passed
instead, RocketPy will try and evaluate the value of the Function at this given point [using different methods], returning
it's value. 

Additionally, if another *Function* object is passed, the class will try to match their respective domain and
codomain in order to return a third instance, representing a composition of functions, in the likes of: :math:`h(x) = (g \circ f)(x) = g(f(x))`. 
By imitating in syntax commonly used mathematical notation, RocketPy allows for more understandable and human-readable code, 
especially in the implementation of the more extense and cluttered rocket equations of motion.

.. The paragraph above should probably be broken into two...

.. Might be worth to add an example here, or maybe not... If anyone has any good ideas on concise examples of Function class,
   feel free to add it here!

Environment
-----------
.. First authors: Gui/Oscar

The Environment class reads, processes and stores all the information regarding wind and atmosphere model data, 
it receives as input the launch point coordinates, as well as the length of the launch rail, and then provides 
the flight class with six profiles as a function of altitude: wind speed in east and north directions, 
atmospheric pressure, air density, dynamic viscosity, and speed of sound.
For instance, it can be set an Environment object representing New Mexico, United States:

.. code-block:: python

   from rocketpy import Environment
   Env = Environment(
      railLength=5.2,
      latitude=32.990254,
      longitude=-106.974998,
      elevation=1400) 

RocketPy requires `datetime` library information specifying year, month, 
day and hour to compute the whether conditions to the specific day of the launch. 
As optional argument, a timezone may also be specified, 
but if the user prefers to omit the timezone RocketPy will assume 
the datetime is given in standard UTC time, just as following:

.. code-block:: python
   
   import datetime
   tomorrow = (
      datetime.date.today() + 
      datetime.timedelta(days=1)
   )
      
   date_info = (
      tomorrow.year,
      tomorrow.month, 
      tomorrow.day,
      12
   )  # Hour given in UTC time

By default the Standard Atmosphere (cite:`ISO Central Secretary. 1975`) is loaded as the atmospheric model, 
however, it is easy to set other model by importing data from different 
meteorological agencies datasets, such as Wyoming Upper Air Soundings and ECMWF, 
or to set a Custom Atmosphere based on user defined functions. 
As RocketPy supports integration with different meteorological agencies datasets, it allows for a 
sophisticated definition of wheater coditions including forecasts and historical reanalysis scenarios.

In this case the NOAA's Ruc Soundings data model is used, an wide-word open-source meteorological model made available online.
The file name is set as `GFS`, indicating the use of a global model with 0.25deg resolution that receives updates every 6 hours 
and create forecasts spaced by 3 hours. 

.. code-block:: python

   Env.setAtmosphericModel(
      type='Forecast', 
      file='GFS')
   Env.info()

What is accutually happennig behind the above code snippet is that RocketPy is using 
the OPeNDAP protocol to retrieve data from NOAA's server. 
It parses by using netCDF4 data management system, allowing for the definition of 
pressure, temperature, wind velocity, and surface elevation as a function of altitude. 
The Environment class then compute the following parameters: wind speed, wind heading, speed of sound, air density, 
and dynamic viscosity. 
Finally, plots of the evaluated parameters with respect to the altitude are all given to the mission 
analyst by using the `Env.info()`.

.. TODO: acrescentar imagem do environment?

Motor
-----
.. First author: Gribel

RocketPy is flexible enough to work with most types of motors used in sound rockets. 

.. Currently, a robust Motor class has been fully implemented and tested. 

The main function of thrus informations to provide the thrust curve, the propulsive mass, the inertia tensor, 
and the position of its center of mass as a function of time. 
Geometric parameters regarding propellant grains and the motor's nozzle must be provided, 
as well as a thrust curve as a function of time. The latter is preferably obtained empirically from a static hot-fire test, 
however, many of the curves for commercial motors are freely available online (citacao-1: thrustcurve.org). 
Alternatively, for homemade motors, there is a wide range of [Python-based - ?], open-source
internal ballistics simulators [packages], such as OpenMotor (citacao 2), which can predict the produced thrust 
with high accuracy for a given sizing and propellant combination.
There are different types of rocket motors: solid motors, liquid motors, and hybrid motors. 
Currently, a robust Solid Motor class has been fully implemented and tested.
For example, a typical solid motor can be created as an object in the following way:

.. code-block:: python
   
   MotorName = SolidMotor(
      thrustSource='Motor_file.eng',
      burnOut=2,
      reshapeThrustCurve= False,
      grainNumber=5,
      grainSeparation=3/1000,
      grainOuterRadius=33/1000,
      grainInitialInnerRadius=15/1000,
      grainInitialHeight=120/1000,
      grainDensity= 1782.51,
      nozzleRadius=49.5/2000,
      throatRadius=21.5/2000,
      interpolationMethod='linear')

Rocket
------

.. First author: Stano
.. 1Revisor: Adriano

The Rocket Class is responsible for creating and defining the rocket's core characteristics. Mostly composed of physical
attributes, such as mass and moments of inertia, the rocket object will be responsible to storage and calculate mechanical parameters.

A rocket object can be defined with the following code:

.. code-block:: python

   RocketName = Rocket(
      motor=MotorName,
      radius=127 / 2000,
      mass=19.197 - 2.956,
      inertiaI=6.60,
      inertiaZ=0.0351,
      distanceRocketNozzle=-1.255,
      distanceRocketPropellant=-0.85704,
      powerOffDrag="data/rocket/powerOffDragCurve.csv",
      powerOnDrag="data/rocket/powerOnDragCurve.csv",
   )

As stated in [RocketPy architecture], a fundamental input of the rocket is its motor, an object of the Motor class
that must be previously defined. Some inputs are fairly simple inputs that can be easily obtained with a CAD model
of the rocket such as radius, mass, and moment of inertia in two different directions. 
The 'distace' inputs are relative to center of mass, and define the position of the motor nozzle and the center of mass of the motor 
propellant. The *powerOffDrag* and *powerOnDrag* receive .csv data that represents the drag coefficient as a function of rocket 
speed for the case where the motor off and other for motor still burning, respectvely.

.. Revisor1: Nao colocaria a parte abaixo, me parece algo mais apr aum manual d RocketPy
.. The calculations made in the class consider, as the geometrical reference, the center of mass of the rocket.
.. Thus, all parts of the rocket must be defined considering its distace to the rockets CM

At this point, the simulation would run a rocket with a tube of a certain diameter, with its center of mass specified and a motor at its end. 
For an better simulation, a few more important aspects should then be defined, called *Aerodynamic surfaces*. Three of then are accepted 
in the code, these being the nosecone, fins, and tail. They can be simply added to the code via the following methods:

.. TODO: example image of a nosecone, fin and tail???
.. Rvisor1: Por mim nao coloca nenhum


.. code-block:: python
   
   Nosecone = Rocket.addNose(
      length=0.55829, kind="vonKarman", 
      distanceToCM=0.71971
   )
   FinSet = Rocket.addFins(
      4, span=0.100, rootChord=0.120, tipChord=0.040, 
      distanceToCM=-1.04956
   )
   Tail = Rocket.addTail(
      topRadius=0.0635, bottomRadius=0.0435, length=0.060, 
      distanceToCM=-1.194656
   )

All these methods receive defining geometrical parameters and its distance to the rockets center of mass (distanceToCM) as inputs.
Each of these surfaces generates, during the flight, a lift force that can be calculated via a lift coefficients, which is
calculated with geometrical properties, as shown in (cite: Barrowman). Further on, these coefficients are used to calculate 
the center of pressure and subseuquently the static margin. Inside each of these methods the static margin is reevaluated.

With the rocket fully defined, the `info()` and `allInfo()` methods can be called giving us information and plots of the calculations performed
in the class. 
One of the most relevant outputs of the Rocket class is the static margin, as it is iportant for the rocket stability and makes posible
several different analysis.
It is visualized thorught the time plot :ref:`figSM`, which shows the variation of the static margin as the motor burns its propellant.
.. Revisor1: Reduzi um pouco o texto e agrupei todas as infos de static margin antes de mostrar o exmeplo dela.
..One of the most relevant outputs of the Rocket class is the static margin, thorught the time plot :ref:`figSM`, which shows
..the variation of the static margin as the motor burns its propellant.

.. figure:: SMoutput.png
   :align: center
   :figclass: bht
   
   Static Margin :label:`figSM`

..Since the static margin is essential to understand the rocket stability, this plot is very useful for several different analysis.

Flight
------
.. First author: Giovani/Stano
.. Suggested topics:
..  (0) Basic intro describing what class does
..  (1) Use of LSODA and why (taking advantage of explicit and implitcit solvers) and how (if interesting)
..  (2) FlightPhases as a container datatype, which holds FlightPhase instances
..      (a) How is the FlightPases container initialized (rail phase and max time)
..      (b) The fact that it is dynamic, new phases can be added and removed
..      (c) The fact that it is iterable, and that it can be used in a for loop
..      (d) How flight phases are created during the simulation and when
..  (3) TimeNodes as a container datatype, which holds TimeNode instances
..      (a) TimeNodes as a basic discretization of the flight phase
..      (b) Why use TimeNodes: parachute release, control events, etc.
..  (4) Time overshoot - why? faster when events are rarely triggered
..  (5) Post processing and results (allInfo)

.. (0)

The Flight class is responsible for the integration of the rocket's equation of motion overtime (cite: RocketPaper).
Data from instances of the Rocket class and the Environment class are used as input to initialize it,
along with parameters such as launch heading and inclination relative to the Earth's surface:

.. code-block:: python
   
   TestFlight = Flight(
      rocket=Rocket,
      environment=Env,
      inclination=85,
      heading=0)

It is in this object of the Flight class that all information of the rocket's flight trajectory simulation is stored.

.. (1) TODO: Cite Scipy and LSODA (citations can be found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.html)

For the integration, the Flight class uses the LSODA solver implemented by Scipy's `scipy.integrate` module.
Usually, well designed rockets result in non-stiff equations of motion.
However, during flight, rockets may become unstable due to variations in its inertial and aerodynamic properties, which can result in a stiff system.
LSODA switches automatically between the nonstiff Adams method and the stiff BDF method, depending on the dected stiffness, perfectly handling both cases.

.. (2) FlightPhases as a container datatype, which holds FlightPhase instances

Since a rocket's flight trajectory is usally composed of multiple phases,
each with its own set of governing equations,
the Flight class uses a ``FlightPhases`` container to hold each ``FlightPhase``.
A ``FlightPhase`` object is initialized with a simulation time,
a function which calculates the equatios of motion,
a list of callbacks and a boolean parameter.
The parameter ``t`` indicates the initial time for that ``FlightPhases``, the ``derivative`` would be the derivative function of the motion of the body, the ``callback`` is for .... and the ``clear``
is used for .... .

.. (b) The fact that it is dynamic, new phases can be added and removed

The FlightPhases container will orchestrate the different FlightPhase elements, and compose them during the flight. This is important because there are events
unkown a priori, or that it's unknown when they will happen, therefore it's important for the computations done on the class Flight to have a mecanism of 
creating events that are not know a priori, but that can be detected during flight, that will change the derivative of equation of motion. 

like the ejection of N parachutes, that changes the derivative of the equation of motion for the rocket.

There are some events that are know a priori that are the out of rail event, apogee and impact. The out-of-rail event is important because they will change
completely the equation of motion of the rocket, as explained in [rocketpaper] the system will become a 6-DOF instead of 1-DOF. The apogee event is important
because .... And finally the Impact Event is important because it will mark the end of the flight. 

.. (c) The fact that it is iterable, and that it can be used in a for loop

.. (d) How flight phases are created during the simulation and when

The container is intialized with a *rail phase*, which is the start of the flight, and also a *max time* which is the maximum time of the flight.
Throught the simulation, more flight phases can be added and removed, but only after the current phase in order to preserve the order of the flight trajectory.
As an example, once the rocket leaves the rail, a new phase is added.

..  (3) TimeNodes as a container datatype, which holds TimeNode instances
..      (a) TimeNodes as a basic discretization of the flight phase
..      (b) Why use TimeNodes: parachute release, control events, etc.

The second data-type that is important to understand are the TimeNodes. An instance of the TimeNode class will contain the information important for
a given time of the Flight, it is a discretization of the continuous time. The TimeNode have similar parameters to the FlightPhase, it receives
the current time respective to the TimeNode, the parachutes that will be ejected on that specific TimeNode and callbacks functions that will be executed. 
Therefore the basic functioning is that the Flight is partioned on different FlightPhases, that have it's own equations and characteristics, and each
FlightPhase will have TimeNodes, which is a point in the time where the integration step will be executed, and this class is important to control the
parachute release and other discrete events.


.. Which phase is added, why and most importantly, how exactly?

.. TODO: Come up with a better section title, one which is shorter and clearer

Design: Adaptability of the Code and Accessibility 
==================================================
.. First author: Patrick
   Suggestions:
      It's easy and possible to implement new classes over rocketpy framework
      also it's an open-source project, 
      object-oriented programming makes everythin easir ad more accessible


RocketPy started to be build in 2017 with some requirements in mind: the code must run fast, this is important because we are interested in
running multiple simulations to compare different parameters, and also the possibility of implementing optimisation methods for the rocket parameters, the
code must be flexible, this is important because each team have their necessity, therefore we structured the code in a fashion that each major component of
the problem separated in classes, using concepts of Single Responsability Principle (SRP), and finnaly the code must be accessible, that's why the code
was published on the Github (citar rocketpy.org) and why we started the RocketPy Team to improve this tool and to create a community around it, facilitating the access of 
high quality simulation without a great level of specialization.

Through examples it will be clear how RocketPy is an usefull tool during the design, operation of the Rocket, enabling functionallities not available by
other rocket simulation softwares.

Examples
========
.. Length: 5/15 columns



Using RocketPy for Rocket Design 
--------------------------------

In this section we describe 
Using RocketPy for such thing is such kind special...

1.  Apogee by Mass using function helper class

   .. First author: Patrick
      For inspiration, you can see the following content:https://colab.research.google.com/github/giovaniceotto/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb#scrollTo=qsXBVgGANVGD
   ..Revisor1: Adriano

Because of performance and safety reasons, apogee is one of the most important results in rocketry competitions, and it's highly valuable for 
teams to understand how different Rocket parameters can change it. Since a direct relation is not available for this kind of computation, the 
caracteristic of running simulation quickly are utilized for evaluatin how the Apogee is affected by the mass of the Rocket. This function is 
highly used during the early phases of the design of a Rocket.

An example of code of how this could be achieved:

.. code-block:: python

   def apogee(mass):
      # Prepare Environment
      Env = Environment(....)

      Env.setAtmosphericModel(type="CustomAtmosphere", wind_v=-5)

      # Prepare Motor
      Pro75M1670 = SolidMotor(.....)

      # Prepare Rocket
      Calisto = Rocket(.....
         mass=mass,
         ......)

      Calisto.setRailButtons([0.2, -0.5])
      Nose = Calisto.addNose(.....)
      FinSet = Calisto.addFins(....)
      Tail = Calisto.addTail(....)

      # Simulate Flight until Apogee
      TestFlight = Flight(.....)
      return TestFlight.apogee


   apogeebymass = Function(apogee, inputs="Mass (kg)", outputs="Estimated Apogee (m)")
   apogeebymass.plot(8, 20, 20)

The possibility of generating this relation between mass and apogee in a graph shows the flexibility of Rocketpy and also the importance of the simulation being
designed to run fast.

1. Dynamic Stability Analysis
   
   .. First author: Guilherme

In this analysis the integration of three different RocketPy classes will be explored: Function, Rocket, and Flight.
The motivation is to investigate how static stability translates into dynamic stability, 
i.e. different static margins result relies on different dynamic behaviour, 
which also depends on the rocket's rotational inertia.

We can assume the objects stated on [motor] and [rocket] sections and just add couple variations on some input data in order to visualize the output effects. 
More specifically, 
the idea will be to explore how the dynamic stability of Calisto varies by changing the position of the set of fins by a certain factor.

In order to do that, we have to simulate multiple flights with different static margins, which is achieved by varying the rocket's fin positions. This can be done through a simple python loop, as described below:


.. code-block:: python
   
   simulation_results = []
   for factor in [0.5, 0.7, 0.9, 1.1, 1.3]:
      # remove previous fin set
      RocketName.aerodynamicSurfaces.remove(FinSet)
      FinSet = RocketName.addFins(
         4, span=0.1, rootChord=0.120, tipChord=0.040,
         distanceToCM=-1.04956 * factor
      )
      FlightName = Flight(
         rocket=RocketName,
         environment=Env,
         inclination=90,
         heading=0,
         maxTimeStep=0.01,
         maxTime=5,
         terminateOnApogee=True,
         verbose=True,
      )
      FlightName.postProcess()
      simulation_results += [
         (
         FlightName.attitudeAngle,
         RocketName.staticMargin(0),
         RocketName.staticMargin(FlightName.outOfRailTime),
         RocketName.staticMargin(FlightName.tFinal)
         )
         ]
   Function.comparePlots(
      simulation_results,
      xlabel="Time (s)",
      ylabel="Attitude Angle (deg)",
   )

The next step is to start the simulations themselves, which can be done through a loop where we call Flight class, perform the simulation, 
save the desired parameters into a list and then follow through the next iteration.
We'll also be using the *post-process* flight data method to make RocketPy evaluate additional result parameter after the simulation.

Finally, the `Function.comparePlots()` method is used to plot the final result.

[Precisa incluir imagem aqui e refinar o texto acima!]

Monte Carlo Simulation
----------------------
.. First author: Stano

The Monte Carlo simulations are trully special...

Validation of the results 
=========================

Validation is a big problem for libraries like RocketPy, where the true values for some results like Apogee, Maximum Velocity are not available. Therefore, in order
to make RocketPy a software more flexible, easier to modify while being rigorous on the results, some testing strategies have been implemented. First of all, Unit Test were implemented for the classes,
this ensures that each function is working properly, given the set of different inputs that each function can receive, the output is inside what is expected, and there are no unexpected
errors.

After, there is a second layer of testing which will avaliate if the equations are dimensionally correct, as some equations can get very convoluted, implementation errors are very common,
hence tests to verify if the computation is dimensionally correct are very useful. These tests implemented using the numericalunits library, which generates a random number that will
be associated to a given unit. For example, given one initialization of this library the meter will be equal to the numerical value of 4.08. Using this ideia, the classes Rocket, SolidMotor
are initilized with parameters with his respectives units.

Initilization without using numericalunits

.. code-block:: python

   @pytest.fixture
   def solid_motor():
      example_motor = SolidMotor(
         thrustSource="data/motors/Cesaroni_M1670.eng",
         burnOut=3.9,
         grainNumber=5,
         grainSeparation=5 / 1000,
         grainDensity=1815,
         grainOuterRadius=33 / 1000,
         grainInitialInnerRadius=15 / 1000,
         grainInitialHeight=120 / 1000,
         nozzleRadius=33 / 1000,
         throatRadius=11 / 1000,
         interpolationMethod="linear",
      )
      return example_motor


   @pytest.fixture
   def rocket(solid_motor):
      example_rocket = Rocket(
         motor=solid_motor,
         radius=127 / 2000,
         mass=19.197 - 2.956,
         inertiaI=6.60,
         inertiaZ=0.0351,
         distanceRocketNozzle=-1.255,
         distanceRocketPropellant=-0.85704,
         powerOffDrag="data/calisto/powerOffDragCurve.csv",
         powerOnDrag="data/calisto/powerOnDragCurve.csv",
      )
      return example_rocket

Initilization using numericalunits

.. code-block:: python

   import numericalunits

   @pytest.fixture
   def m():
      return numericalunits.m


   @pytest.fixture
   def kg():
      return numericalunits.kg

   @pytest.fixture
   def dimensionless_rocket(kg, m, dimensionless_solid_motor):
      example_rocket = Rocket(
         motor=dimensionless_solid_motor,
         radius=127 / 2000 * m,
         mass=(19.197 - 2.956) * kg,
         inertiaI=6.60 * (kg * m**2),
         inertiaZ=0.0351 * (kg * m**2),
         distanceRocketNozzle=-1.255 * m,
         distanceRocketPropellant=-0.85704 * m,
         powerOffDrag="data/calisto/powerOffDragCurve.csv",
         powerOnDrag="data/calisto/powerOnDragCurve.csv",
      )
      return example_rocket

   @pytest.fixture
   def dimensionless_solid_motor(kg, m):
      example_motor = SolidMotor(
         thrustSource="data/motors/Cesaroni_M1670.eng",
         burnOut=3.9,
         grainNumber=5,
         grainSeparation=5 / 1000 * m,
         grainDensity=1815 * (kg / m**3),
         grainOuterRadius=33 / 1000 * m,
         grainInitialInnerRadius=15 / 1000 * m,
         grainInitialHeight=120 / 1000 * m,
         nozzleRadius=33 / 1000 * m,
         throatRadius=11 / 1000 * m,
         interpolationMethod="linear",
      )
      return example_motor

Finally, to ensure that the equations implemented are dimensionally correct, we compare the value calculated by the class initilized with and without the numericalunits units. For example,
on the Rocket class it's calculated the staticMargin of the rocket, which is an adimensional value, so the class initilized with and without the units should have the same value,
so to make sure that the computation is correct it's possible to simply execute the following test

.. code-block:: python

   def test_static_margin_dimension(..., rocket, dimensionless_rocket, ...):
      #add aerodynamic surfaces to rocket and dimensioneless_rocket
      assert pytest.approx(dimensionless_rocket.staticMargin(0), 1e-12) == pytest.approx(
         rocket.staticMargin(0), 1e-12
      )
      assert pytest.approx(dimensionless_rocket.staticMargin(-1), 1e-12) == pytest.approx(
         rocket.staticMargin(-1), 1e-12
      )

And if the computation have a unit, the center of pressure, which is given in meters, the following test is implemented

.. code-block:: python

   def test_cpz_dimension(..., rocket, dimensionless_rocket, ...):
      #add aerodynamic surfaces to rocket and dimensioneless_rocket
      assert pytest.approx(dimensionless_rocket.cpPosition / m, 1e-12) == pytest.approx(
        rocket.cpPosition, 1e-12
    )

If the result given by dimensionless_rocket divided by the value of meter is not equal to the value given by the rocket, we can conclude that the formula responsible for calculating the
cpPosition was implemented incorrectly. 


Finally, it was implemented some tests at a more macroscopic scale, which are the Acceptance tests, that validates results like apogee, maximum velocity, apogee time, maximum aceleration.
These results depend on several functions and their interactions, after the publication of the [rocketpaper] we have defined a precision for these results for the flights for which we have
recorded experimental data. These tests will simply run a simulation of these flights and compare the experimental data with the data generated by RocketPy and evaluate if the results
are within the interval of tolerance defined. They are very important to ensure that with new changes the code will not lose precision. In conclusion those 3 layers of testing makes the
software reliable, where the team is confident that new changes will only improves the perfomance of the Software.

Conclusions 
===========
.. Length: 0.75/15 columns

By the end of this article we can conclude...

Acknowledgements
================
.. Length: 0.25/15 columns
.. Authors: ? / Giovani / ...
.. TODO: Who else should be mentioned?

The authors would like to thank the *University of São Paulo*, for the support during the development the current publication,
all members of Projeto Jupiter and the RocketPy Team who contributed in the making of the RocketPy library.

References
==========

