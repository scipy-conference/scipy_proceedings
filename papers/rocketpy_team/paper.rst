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
   Length: 1/15 columns
   Suggested Topics:
      (0) sounding rockets and high-powered rockets
      (1) high-powered rocketry importance is under high increase trajectory  
      (2) trajectory simulations are important for both safety and performance, 
      (3) RocketPy as a solution for trajectory simulations (the next generation!)

As described in :cite:`ceotto2021rocketpy` ...

Background 
==========
.. Length: 2/15 columns

Rocketry vocabulary
-------------------
.. First author: Ciclope
   Suggest topics:
      (1) explain different basic concepts regarding rockets!
      (2) explain expected ouputs (don't forget to include Monte Carlo basic concepts)

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

Design 
======
.. Length: 5/15 columns

RocketPy Architecture
---------------------
.. First authors: Oscar/Gui

There are four main classes that organize the dataflow during the simulations: motor, rocket, environment and flight. :cite:`ceotto2021rocketpy`.
Acctually there is also a helper class named `function`, which will be described further.
In the motor class, the main physical and geometric parameters of the motor are configured, such as: 
nozzle geometry, grain parameters, mass, inertia and thrust curve. This first class acts as an input 
to the Rocket class where the user is also asked to define certain parameters of the rocket 
such as the inertial mass tensor, geometry, drag coefficients and parachute description. 
Finally, the Flight class joins the rocket and motor parameters with information from the Environment class, such as wind, atmospheric and earth models, to generate a simulation of the rocket's trajectory.
This modular architecture, along with its well-structured and documented code, facilitates complex simulations, 
starting with the use of Jupyter Notebooks that people can adapt for their specific use case.
The figure :ref:`fig1` illustrates RocketPy architecture. 

.. figure:: Fluxogram.png
   :align: center
   :scale: 23%
   :figclass: bht

   RocketPy classes interaction :label:`fig1`

Function
++++++++
.. First authors: Gribel

(Talk a bit about the motivations behind Function class and what it is trying to solve.
Go over its main features such as naturally doing algebra, interpolation/extrapolation, evaluating, differentiation/integration and plotting.
Explain how rocketpy interpolations are faster than usual numpy/scipy methods due to utilizing values from previous interpolations - )
Discuss ease-of-use vs. efficiency. Show an example.)

[Variable interpolation meshes/grids from different sources can lead to problems regarding coupling different data types]. In order
to solve this, RocketPy employs a dedicated *Function* class which allows for more natural and dynamic handling of these objects, structuring
them in a way similar to `R^n \to R^n` mathematical functions. 

Through the use of magic methods, this approach allows for quick and easy arithmetic operations
between lambda expressions and list-defined interpolated functions, as well as scalars. Different interpolation methods are available to be chosen
from, among them simple polynomial, spline and Akima (ref. paper original). Extrapolation of *Function* objects outside the domain constrained
by a given dataset is also allowed.



.. Suggestion 1: different sources could contains different discretization due to, for instance, different time steps, this could lead to
.. Suggestion 2: Variable interpolation meshes/grids from different sources can lead to problems regarding coupling different data types. 

Environment
+++++++++++
.. First authors: Gui/Oscar

The Environment class reads, processes and stores all the information regardingall the information regarding wind and atmosphere model data, 
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

By default the atmospheric the Standard Atmosphere model (:cite:`ISO Central Secretary. 1975`) is loaded, 
however, it is easy to set other model by importing data from different 
meteorological agencies datasets, such as Wyoming Upper Air Soundings and ECMWF, 
or to set a Custom Atmosphere based on user defined functions. 
As RocketPy supports integration with different meteorological agencies datasets, it allows for a 
sophisticated definition of wheater coditions including forecasts and historical reanalysis scenarios.

In this case is used NOAA's Ruc Soundings data model, an wide-word open-source meteorological model made available online.
The file name is set as `GFS`, indicating the use of a global model with 0.25deg resolution that receives updates every 6 hours 
and create forecasts for 81 points spaced by 3 hours. 

.. code-block:: python

   Env.setAtmosphericModel(
      type='Forecast', 
      file='GFS'
   )
      
   Env.info()

What is accutually happennig behind the above code snippet is that RocketPy is using 
the OPeNDAP protocol to retrieve data from NOAA's server. 
It parses by using netCDF4 data management system, allowing for the definition of 
pressure, temperature, wind velocity, and surface elevation as a function of altitude. 
The Environment class then compute the following parameters: wind speed, wind heading, speed of sound, air density, 
and dynamic viscosity. Finally, plots of the evaluated parameters with respect to the altitude are all given to the mission 
analyst by using the `Env.info()`.

Motor
+++++
.. First author: Gribel

RocketPy is flexible enough to work with most types of motors used in sound rockets. 
Currently, a robust Motor class has been fully implemented and tested. The main function of 
thrus informations to provide the thrust curve, the propulsive mass, the inertia tensor, 
and the position of its center of mass as a function of time. Geometric parameters regarding propellant grains
and the engine's nozzle must be provided, as well as a thrust curve as a function of time. The latter is preferably
obtained empirically from a static hot-fire test, however, many of the curves for commercial motors are freely available
online (citacao-1: thrustcurve.org). Alternatively, for homemade motors, there is a wide range of [Python-based - ?], open-source
internal ballistics simulators [packages], such as OpenMotor (citacao 2), which can predict the produced thrust 
with high accuracy for a given sizing and propellant combination.
There are various types of rocket motors such as solid motors, liquid motors, and hybrid motors. RocketPy is flexible enough to work with most of them. 
Currently, a robust Solid Motor class has been fully implemented and tested.g anFor example, a typical solid motor can be created as an object in the following way:l

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
      interpolationMethod='linear'
   )

Rocket
++++++

.. First author: Stano

The Rocket Class is responsible  Rocket Class is responsible for creating and defining the rocket with its core characteristics. Mostly composed of physical
attributes, such as mass, radius and moments of inertia, the rocket object will be responsible for the storage and calculation 
of mechanical parameters.

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
   

As stated in (RocketPy architecture), a fundamental input of the rocket is its motor, an object of the Motor class
that must be previously defined. Some inputs are fairly simple inputs that can be easily obtained with a CAD model
of the rocket such as radius, mass, inertiaI and interiaZ. The 'distace' inputs are relative to center of mass, and define
the position of the motor nozzle and the center of mass of the motor propellant. The *powerOffDrag* and *powerOnDrag* 
receive .csv curves that represents the drag coefficient of the rocket with motor off and on, respectvely.
.. The calculations made in the class consider, as the geometrical reference, the center of mass of the rocket.
.. Thus, all parts of the rocket must be defined considering its distace to the rockets CM

Using only the class constructor for the definition of the rocket object leads to an unfineshed rocket. One large tube of a
certain diameter, with its center of mass positioned at a specific point along the axis of this tube, and a motor at the end would be
used in the simulation if left like that (too informal?). A few more important aspects should then be defined, these are
called *Aerodynamic surfaces*. Three of then are accepted in the code, these being the nosecone, fins, and tail. They can be
simply added to the code via the following methods:

.. TODO: example image of a nosecone, fin and tail???


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
the center of pressure and subseuquently the static margin. Inside each of these methods the static margin is reevaluated

With the rocket fully defined, the info() and allInfo() methods can be called giving us information and plots of the calculations performed
in the class. One of the most relevant outputs of the Rocket class is the sta(figure tic margin t)horugh time plot :ref:`figSM`, which shows
the variation of the static margin as the motor burns its propellant.

.. figure:: SMoutput.png
   :align: center
   :figclass: bht
   
   Static Margin :label:`figSM`

Since the static margin is essencial to understand the rocket stability, this plot is very useful for several different analysis

Flight
++++++
.. First author: Giovani/Stano
   Suggested topics:
    (1) Integration (LSODA)
    (2) Time nodes

The Flight class is awesome.

.. TODO: Come up with a better section title, one which is shorter and clearer

Adaptability of the Code and Accessibility 
------------------------------------------
.. First author: Patrick
   Suggestions:
      It's easy and possible to implement new classes over rocketpy framework
      also it's an open-source project, 
      object-oriented programming makes everythin easir ad more accessible

Examples
========
.. Length: 5/15 columns

In this section we present some examples...

Using RocketPy for Rocket Design 
--------------------------------

In this section we describe 
Using RocketPy for such thing is such kind special...

1.  Apogee by Mass using function helper class
.. First author: Patrick
.. For inspiration, you can see the following content:https://colab.research.google.com/github/giovaniceotto/rocketpy/blob/master/docs/notebooks/getting_started_colab.ipynb#scrollTo=qsXBVgGANVGD

Loren Ipsum...

1. Dynamic Stability Analysis
.. First author: Guilherme

In this analysis the integration of three different RocketPy classes will be explored: Function, Rocket, and Flight.
The motivation is to investigate how static stability translates into dynamic stability, 
i.e. different static margins result relies on different dynamic behaviour, 
which also depends on the rocket's rotational inertial.

We can assume the objects stated on [motor] and [rocket] sections and just add coulpe variations on some input data in order to visualize the output effects. 
Therefore, 
the idea here is to explore how the dynamic stability of Calisto varies if we change the position of the set of fins by a certain factor.

In order to do that, we have to Simulate flights with Different Static Margins by Varying Fin Position, this can be done through a simple loop from python, as described below:


.. code-block:: python
   
   simulation_results = []
   for factor in [0.5, 0.7, 0.9, 1.1, 1.3]:
      # remove previous fin set
      Calisto.aerodynamicSurfaces.remove(FinSet)
      FinSet = Calisto.addFins(
         4, span=0.1, rootChord=0.120, tipChord=0.040,
         distanceToCM=-1.04956 * factor
      )
      TestFlight = Flight(
         rocket=Calisto,
         environment=Env,
         inclination=90,
         heading=0,
         maxTimeStep=0.01,
         maxTime=5,
         terminateOnApogee=True,
         verbose=True,
      )
      TestFlight.postProcess()
      simulation_results += [
         (
         TestFlight.attitudeAngle,
         Calisto.staticMargin(0),
         Calisto.staticMargin(TestFlight.outOfRailTime),
         Calisto.staticMargin(TestFlight.tFinal)
         )
         ]
   Function.comparePlots(
      simulation_results,
      xlabel="Time (s)",
      ylabel="Attitude Angle (deg)",
   )

The next step is to start the simulations itself.
We are using the Post process flight data method.

Finally, the `Function.comparePlots()` method is used to plot final result.

[Precisa incluir imagem aqui e refinar o texto acima!]

Monte Carlo Simulation
----------------------
.. First author: Stano

The Monte Carlo simulations are trully special...

Conclusions 
===========
.. Length: 0.75/15 columns

By the end of this article we can conclude...

Acknowledgements
================
.. Length: 0.25/15 columns
.. Authors: ? / Giovani / ...
.. TODO: Who else should be mentioned?

The authors would like to thank the *University de São Paulo*, for the support during the development the current publication,
and also thank all members of Projeto Jupiter and the RocketPy Team who contributed in the making of the RocketPy library.

References
==========

