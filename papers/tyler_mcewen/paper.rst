:author: Tyler McEwen
:email: tyler.mcewen@twdb.state.tx.us
:institution: Texas Water Development Board

:author: Dharhas Pothina
:email: dharhas.pothina@twdb.state.tx.us
:institution: Texas Water Development Board

:author: Solomon Negusse
:email: solomon.negusse@twdb.state.tx.us
:institution: Texas Water Development Board

----------------------------------------------------------------------------
Improving efficiency and repeatability of lake volume estimates using Python
----------------------------------------------------------------------------

.. class:: abstract

   With increasing population and water use demands in Texas, accurate estimates
   of lake volumes is a critical part of planning for future water supply needs.
   Lakes are large and surveying them is expensive in terms of labor, time and
   cost. High spatial resolution surveys are prohibitive to conduct, hence lakes
   are usually surveyed along widely spaced survey lines. While this choice
   reduces the time spent in field data collection, it increases the time
   required for post processing significantly. Standard spatial interpolation
   techniques available in commercial software are not well suited to this
   problem and a custom procedure was developed using in-house Fortran software.
   This procedure involved difficult to repeat manual manipulation of data in
   graphical user interfaces, visual interpretation of data and a laborious
   manually guided interpolation process. Repeatibility is important since
   volume differences derived from multiple surveys of individual reservoirs
   provides estimates of capacity loss over time due to sedimentation.
   Through python scripts that make use of spatial algorithms and GIS routines
   available within various Python scientific modules, we first streamlined our
   original procedure and then replaced it completely with a new pure python
   implementation. In this paper, we compare the original procedure, the
   streamlined procedure and our new pure python implementation with regard to
   automation, efficiency and repeatability of our lake volumetric estimates.
   Applying these techniques to Lake Texana in XX county, Texas, we show that
   the new pure python implementation reduces data post processing time from 90
   man hours to 8 man hours while improving repeatability and accuracy.

.. class:: keywords

   gis, spatial interpolation, hydrographic surveying, bathymetry, lake volume,
   reservoir volume, anisotropic, inverse distance wieghted, sedimentation

Introduction
------------

With increasing population and water use demands in Texas, accurate estimates of
lake volumes is a critical part of planning for future water supply needs. In
order to correctly manage surface water supplies for the State of Texas, it is
vital that managers and state water planners have accurate estimates of
reservoir volumes and capacity loss rates due to sedimentation.
To address these issues, in 1991 the Texas Legislature authorized the Texas
Water Development Board (TWDB) to develop a cost-recovery hydrographic surveying
program. The program is charged with determining reservoir storage capacities,
sedimentation levels, sedimentation rates, and available water supply
projections to benefit Texas. Since its inception, staff in the hydrographic
survey program have completed more than 125 lake surveys. Included in each
survey report are updated elevation-area-capacity tables and bathymetric contour
maps.

Lakes are large and surveying them is expensive in terms of labor, time and cost.
Over the years, the Texas Water Development Board (TWDB) has settled on a 500 ft
spacing of survey lines oriented perpendicular to an assumed relic stream
channel for hydrographic data collection as a good balance between survey effort
and level of data coverage. While this choice reduces the time spent in data
collection, it significantly increases the time needed for post-survey
processing. Currently, a typical lake survey can consume XX-XX days of field
data collection and XX-XX weeks of data post processing before a volumetric
estimate is available. 

Volumetric estimate algorithms available in commercial software are usually
based on a delauney triangulation of actual survey points and a digitized lake
boundary. When applied to data collected with widely spaced survey lines, these
techniques tend to underestimate the true volume of the lake.REF To overcome
this issue, TWDB preconditions the survey point dataset by inserting additonal
points in between survey lines and using directional linear interpolation to
estimate the bathymetry at the inserted points. Delauney triangulation of the
resulting dataset gives a more accurate estimate of lake volume. This technique
makes use of the assumption that the profile of the lake between each set of
survey lines is similar to that of the survey lines. Figure REF shows the
improvement in the representation of the bathymetry of the lake that can be
obtained by such preconditioning.

While effective in improving volume estimates, this technique as currently
implemented has a number of flaws. Notably, it depends on exact positions of
survey points and hence is difficult to apply repeatibly. requires
manual visual interpretation and manipulation of data in GUI's as well as a
laborious manually guided interpolation process.

Standard TWDB Surveying Technique
---------------------------------
REWRITE WITH SOME DETAILS Hydrographic Surveys are conducted using a boat
mounted single beam echosounder with differential GPS running along preplanned
survey lines. Survey planning,
operationally defined here as the spacing and orientation of pre-planned survey
lines, is likely to affect volumetric calculations if there are notable
bathymetric changes between surveyed lines. In many cases, however, reservoir
bathymetry will not be known before the survey, and survey lines must be planned
based on an interpretation of the reservoir shape in map-view and the presumed
location and orientation of the submerged stream channel. Previous
TWDB surveys have been conducted using lines spaced at 250 ft intervals
(TWDB, 2009b; TWDB, 2006), and at 500 ft intervals with selected areas of 100-ft
spaced survey lines (TWDB, 2009c). Analyses of data collected on Lake Kemp
indicate that greater volumes are obtained from surveys conducted with higher
density line spacing, yet the volume increase is a result of the surface
generation methodology used within ArcGIS (Furnans, 2006). reference LBJ REPORT

The TWDB standard bathymetric survey consists of data collection along survey
lines spaced 500-ft apart and oriented perpendicular to the assumed location of
the submerged river channel (usually taken to be along the centerline of the
lake). Radial lines are utilized when the shape of the lake and presumed shape
of the submerged river channel curve. Data post processing is then used to
improve the interpolation.

Data processing with HydroEdit
------------------------------

Over the years, the TWDB has developed several post processing routines that
have been packaged together in an in-house Fortran program, HydroEdit. HydroEdit
contains modules to integrate boat GPS and sonar bathymetric data,
calculate sediment thicknesses, extrapolate into regions with no survey data,
merge data files and generate the preconditioned dataset for volumetric
estimates. REFS. 

The main function of the Hydroedit software is to perform bathymetric data
interpolations. Using GPS software, areas of desired interpolation from one lake
bathymetric transect an adjacent transect are visually located and their point
identification numbers are manually recorded into a text file. Lake Texana had
approximately 3050 manually entered interpolations requiring approximately 90
hours to complete. Specialized interpolations are also available with the
appropriate text input format, allowing creativity within the lake bathymetry
interpolation.

IMAGE OF SELECTION OF SELF SIMILAR LINES BETWEEN TWO SURVEY LINES.

Streamlining HydroEdit Using Python
-----------------------------------

Seeking to improve upon the lengthy and tedious process required to manually
create a HydroEdit input text file, Python programming was utilized to
automatically generate the self-similar input text file using GIS line
shapefiles. Due to directionality requirements, data input procedures and
interpolations required between transects (lake cross-sections), multiple loops,
sorting, multiple KDtrees and attributing numerous variables to lines and points
was necessary to accommodate the existing requirements of the HydroEdit software.

Next, through the use of built in spatial algorithms, GIS routines available in
Python, and in-house Python scripts; automation, efficiency and repeatability
were introduced to lake volumetric estimates. The line automated interpolation
program improved efficiencies and speeded overall interpolations significantly,
however the program was limited and structured based on the necessary inputs to
the sequential HydroEdit software. The resulting interpolation point spatial
structure provided inconsistent point density.

Anisotropic Stretched Inverse Distance Weighting (ASIDW)
--------------------------------------------------------

Description of ASIDW algorithm for a channel. Include image of s-n coordinate
conversion. Image of lake with ellipse oriented along direction of interpolation

Description of applying ASIDW to a lake using multiple polygons and channel
lines. Image of polygons & lines.

Write algorithm.


Lake Texana
-----------

The Palmetto Bend Dam was completed in 1979, impounding the Navidad River and
creating Lake Texana [TWDB, 1974]_. At approximately 9,727 acres (3,936 ha),
Lake Texana is a small to medium major reservoir in Texas; the minimum acreage
of major reservoirs in Texas is 5,000 acres (2,023 ha).

TWDB collected bathymetric data for Lake Texana between January 12 and March 4,
2010. The daily average water surface elevations during that time ranged between
43.89 and 44.06 feet above mean sea level (NGVD29). For data collection, TWDB
used a Specialty Devices, Inc., single-beam, multi-frequency (200 kHz, 50 kHz,
and 24 kHz) sub-bottom profiling depth sounder integrated with differential
global positioning system (DGPS) equipment. Data collection occurred while
navigating along pre-planned range lines oriented perpendicular to the assumed
location of the original river channels and spaced approximately 500 feet apart.
The depth sounder was calibrated daily using a velocity profiler to measure the
speed of sound in the water column and a weighted tape or stadia rod for depth
reading verification. During the 2010 survey, team members collected nearly
244,000 data points over cross-sections totaling approximately 160 miles in
length. Figure 2 shows where data collection occurred during the 2010 TWDB
survey.


Results
-------

Conclusions
-----------

References
----------

.. [TWDB1974] TWDB (Texas Water Development Board), 1974, *Iron Bridge Dam and Lake Texana*, 
           Report 126, Engineering Data on Dams and Reservoirs in Texas, Part 1.
