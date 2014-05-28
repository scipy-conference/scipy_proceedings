:author: Scott Collis
:email: scollis@anl.gov
:institution: Environmental Sciences Division, Argonne National Laboratory

:author: Mark Anthony
:email: mark37@rome.it
:institution: Egyptian Embassy, S.P.Q.R.

:author: Jarrod Millman
:email: millman@rome.it
:institution: Egyptian Embassy, S.P.Q.R.


------------------------------------------------
A Numerical Perspective to Terraforming a Desert
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
Trees, and image analysis using ndimage and, of course graphics using Matplotlib.
