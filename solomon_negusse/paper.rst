:author: Solomon Negusse
:email: solomon.negusse@twdb.state.tx.us
:institution: Texas Water Development Board

:author: Dharhas Pothina
:email: dharhas.pothina@twdb.state.tx.us
:institution: Texas Water Development Board

:author: Andrew Wilson
:email: andrew.wilson@twdb.state.tx.us
:institution: Texas Water Development Board

------------------------------------------------------------------
The Open Estuary: Using open source tools to model Texas estuaries
------------------------------------------------------------------

.. class:: abstract

	Simulation models of estuaries require a large effort, broadly divided into the
	various stages of data assembly, model setup, multiple simulation runs, post 
	processing and analysis of results. Traditionally, many of these stages have 
	required expensive commercial software. Frustration with the limitations of 
	black box software and expensive licenses led to an exploration of 
	alternatives. Now, two years later, Python is the glue that holds the whole
	together. Data assembly is aided by the python packages developed in-house for
	retrieval and QA/QC of data. Model setup and simulation runs use a combination 
	of commercial software, open source Fortran code and python scripts. 
	Numpy/Scipy have replaced Matlab for post processing and analysis and 2D and 3D
	visualization is being done with Matplotlib, Mayavi and Tecplot.

	We discuss how these open source tools and techniques have improved efficiency,
	reproducability and acted as enablers for new capabilities. We map our 
	transition from stand alone scripts to Python packages that are now being 
	released on Github under the swtools organization.

.. class:: keywords

   simulation models

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



