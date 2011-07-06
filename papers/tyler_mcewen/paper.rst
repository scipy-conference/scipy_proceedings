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
lake volumes is a critical part of planning for future water supply needs. Lakes
are large and surveying them is expensive in terms of labor, time and cost.
Over the years, the Texas Water Development Board has settled on a 500 ft
spacing of survey lines oriented perpendicular to an assumed relic stream
channel (stream channel center line?) for hydrographic data collection as a good
balance between survey effort and level of data coverage. While this choice
reduces the time spent in data collection, it significantly increases the time
needed for post-survey processing.

.. figure:: interp_compare_close_4.png

   This is the caption. :label:`egfig`

.. figure:: interp_compare_close_4.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

Basic TWDB Surveying Technique
------------------------------

Data Processing with HydroEdit
------------------------------

Streamlining HydroEdit Using Python
-----------------------------------

Anisotropic Stretched Inverse Distance Weighting (ASIDW)
--------------------------------------------------------

Lake Texana
-----------

The Palmetto Bend Dam was completed in 1979, impounding the Navidad River and
creating Lake Texana. At approximately 9,727 acres (3,936 ha), Lake Texana is a
small to medium major reservoir in Texas; the minimum acreage of major
reservoirs in Texas is 5,000 acres (2,023 ha). 

Results
-------

Conclusions
-----------

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Fur08] Furnans, J. and Austin, B., *Hydrographic survey methods for determining reservoir volume*,
           Environmental Modelling & Software, Volume 23, Issue 2, February 2008, Pages 139-146, ISSN 1364-8152, DOI: 10.1016/j.envsoft.2007.05.011.



