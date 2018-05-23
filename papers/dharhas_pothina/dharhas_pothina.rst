:author: Dharhas Pothina
:email: Dharhas.Pothina@erdc.dren.mil
:institution: US Army Engineer Research and Development Center
:corresponding:

:author: James A Bednar
:email: jbednar@anaconda.com
:institution: Anaconda, Inc.
:institution: University of Edinburgh

:author: Scott Christensen
:email: Scott.Christensen@erdc.dren.mil
:institution: US Army Engineer Research and Development Center
:equal-contributor:

:author: Kevin Winters
:email: Kevin.Winters@erdc.dren.mil
:institution: US Army Engineer Research and Development Center
:equal-contributor:

:author: Christopher Ball
:email: cball@anaconda.com
:institution: Anaconda, Inc.
:equal-contributor:

:author: Gregory Brener
:email: gregshipssoftware@gmail.com
:institution: Unaffiliated
:equal-contributor:

---------------------------------------------------------------------------------------
EarthSim: Flexible Environmental Simulation Workflows Entirely Within Jupyter Notebooks
---------------------------------------------------------------------------------------

.. class:: abstract

   Building environmental simulation workflows is typically a slow process involving multiple 
   proprietary desktop tools that do not interoperate well. In this work, we demonstrate building
   flexible, lightweight workflows entirely in Jupyter notebooks. We demonstrate these capabilities
   through examples in hydrology and hydrodynamics using the AdH and GSSHA simulators. The goal is 
   to provide a set of tools that can easily be reconfigured and repurposed as needed to rapidly 
   solve specific emerging issues.

   As part of this work, extensive improvements were made to several general-purpose open source 
   packages, including support for annotating and editing plots and maps in Bokeh and HoloViews, 
   rendering large triangular meshes and regridding large raster data in HoloViews, GeoViews, and 
   Datashader, and widget libraries for Param. In addition, two new open source projects are being 
   released for triangular mesh generation and environmental data access. 

.. class:: keywords

   python, visualization, workflows, environmental simulation, hydrology, hydrodynamics, grid generation

Introduction
------------

Environmental Simulation consists of using historical, current and forecasted environmental data in conjunction
with physics-based numerical models to simulate conditions at locations across the globe. The simulations of 
primary interest are weather, hydrology, hydrodynamics, soil moisture and groundwater transport. These simulations
combine various material properties such as soil porosity and vegetation types with topology such as land surface 
elevation and bathymetry, along with forcing functions such as rainfall, tide, and wind, to predict quantities of
interest such as water depth, soil moisture, and various fluxes. Currently, the primary methodology to conduct 
these simulations requires a combination of heavy proprietary desktop tools [cite SMS & CMB] that do not interoperate
well with each other. 

The process of building and running environmental simulations using these tools is a time-consuming process that 
requires a large amount of manual effort and a fair amount of expertise. Typically, the time required to build a 
reasonable model is measured in months. These workflows support some use cases well, especially multi-year projects 
where there is often the need for highly accurate, high-resolution physics. But the existing tools and workflows 
are too heavyweight for other potential applications, such as making short-term operational decisions in novel 
locations. 

In this work, we demonstrate building flexible, lightweight workflows entirely in Jupyter notebooks with the aim of
timely support for operational decisions, providing basic predictions of environmental conditions quickly and flexibly
for any region of the globe.  We demonstrate these capabilities through examples in hydrology and hydrodynamics using 
the AdH and GSSHA simulators [cite adh & gssha. The goal is to provide a set of tools that can easily be reconfigured and repurposed 
as needed to rapidly solve specific emerging issues. 

An explicit decision was made to avoid creation of new libraries as much as possible and to instead enhance existing
tools with the capabilities required. Hence, as part of this work, extensive improvements were made to several 
general-purpose open source packages, including support for annotating and editing plots and maps in Bokeh and 
HoloViews, rendering large triangular meshes and regridding large raster data in HoloViews, GeoViews, and Datashader, 
and widget libraries for Param [cite all software]. In addition, two new open source projects are being released for 
triangular mesh generation and environmental data access [cite filigree & quest].

Components
----------

