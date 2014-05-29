:author: Scott Collis
:email: scollis@anl.gov
:institution: Environmental Sciences Division, Argonne National Laboratory.

:author: Scott Giangrande
:email: sgrande@bnl.gov
:institution: Atmospheric Sciences, Brookhaven National Laboratory.

:author: Jonathan Helmus
:email: jhelmus@anl.gov
:institution: Environmental Sciences Division, Argonne National Laboratory.

:author: Di Wu
:email: di.wu@nasa.gov
:institution: Environmental Sciences Division, Argonne National Laboratory.

:author: Jonathan Helmus
:email: ann.fridlind@nasa.gov
:institution: Environmental Sciences Division, Argonne National Laboratory.

:author: Jonathan Helmus
:email: marcus.vanlier-walqui@nasa.gov
:institution: Environmental Sciences Division, Argonne National Laboratory.



------------------------------------------------
Measuring rainshafts: Bringing python to bear on remote sensing data.
------------------------------------------------

.. class:: abstract
Remote sensing data is complicated, very complicated! It is not only
geospatially tricky but also indirect as the sensor measures the interaction
of the media with the probing radiation, not the geophysics. However the
problem is made tractable by the large number of algorithms available in the
Scientific Python community, what is needed is a common data model for active
remote sensing data that can act as a layer between highly specialized file
formats and the cloud of scientific software in Python. This presentation
motivates this work by asking: How big is a rainshaft? What is the natural
dimensionality of rainfall patterns and how well is this represented in fine
scale atmospheric models. Rather than being specific to the domain of
meteorology we will break down how we approach this problem in terms what tools
across numerous packages we used to read, correct, map and reduce the data to
forms able to answer our science questions. This is a "how" presentation,
covering signal processing using linear programming methods, mapping using KD
Trees, and image analysis using ndimage and, of course graphics using
Matplotlib.

.. class:: keywords

   Remote sensing, radar, meteorology, hydrology

Introduction
------------
Introduce the idea of radars and remote sensing
Meets computer science, quote BAMS [Maik2014]_
Data models
Particular problem, how do you go from complex moments to gaining an idea as to
the scale of precipitation and does a model achieve it?

The data source: Scanning centimeter wavelength radar
------------
What is a radar

The Python ARM Radar Toolkit: Py-ART
------------
The idea behind Py-ART

Pre-mapping corrections and calculations
~~~~~~~~~~~~~~~~~~~~~~

Mapping to a cartesian grid
~~~~~~~~~~~~~~~~~~~~~~

Calculations on the grid data
~~~~~~~~~~~~~~~~~~~~~~

Spatial distribution of rainfall: a objective test of fine scale models
------------

References
----------
.. [Maik2014] Heistermann, M., S. Collis, M. J. Dixon, S. E. Giangrande,
              J. J. Helmus, B. Kelley, J. Koistinen, D. B. Michelson, M. Peura,
              T. Pfaff and D. B. Wolff,
              2014: The Promise of Open Source Software for the Weather Radar
              Community. *Bulletin of the American Meteorological Society*,
              **In Press.**
