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
..:bibliography: mybib

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
   
   This talk is intended for anyone who wants to get involved in the Space Sector and will show how RocketPy is a
   great entrance door to a welcoming community. It will be an opportunity to learn basic concepts about rocketry and
   to get to know this tool used by various rocketry groups around the world.
   
   By the end of the talk, the audience will have learned about the RocketPy`s architecture, how to simulate simple
   rockets, and hopefully will be motivated to join the RocketPy community, as members from diverse backgrounds can
   help us enhance and grow this project!

.. class:: keywords

   space, rocketry, rocket, simulation, trajectory, dynamics, ode, rocket trajectory simulation, sounding rockets, 
   high-powered rockets, flight dynamics, six degrees of freedom, Monte Carlo analysis, stochastic simulation

Introduction - Ciclope
======================
.. (1/15 colunas)

   (0) sounding rockets and high-powered rockets
   (1) high-powered rocketry importance is under high increase trajectory  
   (2) trajectory simulations are important for both safety and performance, 
   (3) RocketPy as a solution for trajectory simulations (the next generation!)

As described in :cite:`ceotto2021rocketpy` ...

Background 
======================
.. (2/15 colunas)

Rocketry vocabulary - Ciclope
-----------------------------
.. 
   (1) explain different basic concepts regarding rockets!
   (2) explain expected ouputs (don't forget to include Monte Carlo basic concepts)

Flight Model - Oscar
-----------------------
..
   The flight model of a sounding rocket takes into account at least three different phases:
   (1) The first phase consists of a linear movement along the launch rail.
   (2) After completely leaving the rail, a phase of 6 DOF is established, which includes powered flight and free flight.
   (3) Once apogee is reached, a parachute is usually deployed, characterizing the third phase of flight: the parachute descent.

Design 
======
.. (5/15 colunas)

RocketPy architecture
---------------------

   Oscar  - There are four main classes that organize the dataflow during the simulations (and there's a function class), 
      
   
   Function: - Gribel

   Environment: - Guilherme

   Motor: - Gribel

   RocketPy is flexible enough to work with most types of motors used in sound rockets. 
   Currently, a robust engine class has been fully implemented and tested. The main function of 
   the engine class is to provide the thrust produced, the propulsive mass, the inertia tensor, 
   and the position of its center of mass as a function of time.

   Rocket: Stano
   Creating a rocket with rocketpy

   Flight: Giovani starts and Stano finishes
   Integration (LSODA) and time nodes methods
         
Patrick: Adaptability (flexibility) of the code and Accessibility 
-----------------------------------------------------------------

It's easy and possible to implement new classes over rocketpy framework
also it's an open-source project, object-oriented programming


Examples
========
.. (5/15 colunas)

In this section we present some examples...

Using RocketPy for Design and differet analysis - 1 example for Gui, other for patricksampaio
------------------------------------------------------------------------------------------------
..   
   optmization and comparinson features?
   e.g. apogee by mass (use as function example) 
   e.g. stability

Using RocketPy for such thing is such kind special...

Monte Carlo Simulation - Stano
------------------------------

The Monte Carlo simulation are trully special...

Conclusions 
===========
.. (0.75/15 colunas)

By the end of this article we can conclude...

Acknowledgements
================
.. (0.25/15 colunas)
We would like to thank...

References
==========

