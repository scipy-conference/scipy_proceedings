:author: Mellissa Cross
:email: cros0324@umn.edu, mellissa.cross@gmail.com
:institution: Department of Earth Sciences, University of Minnesota

:video: http://www.youtube.com/watch?v=dhRUe-gz690

-----------------------------------------------------------------------------------------------------
PySPLIT: a Package for the Generation, Analysis, and Visualization of HYSPLIT Air Parcel Trajectories
-----------------------------------------------------------------------------------------------------

.. class:: abstract

   The HYSPLIT model outputs air parcel paths projected forwards or backwards in time (trajectories) and is used in a variety of scientific contexts.  Here we present the first package in the mainstream scientific Python ecosystem designed to facilitate HYSPLIT trajectory analysis workflow by providing an intuitive API for generating, inspecting, and plotting trajectory paths and data.

.. class:: keywords

   HYSPLIT, trajectory analysis, matplotlib Basemap

Introduction
------------
The NOAA Air Resources Laboratory's HYSPLIT (HYbrid Single Particle Lagrangian Transport) model is publicly available via the web READY interface- and has been since the late 1990s- or downloadable versions compatible with PC or Mac.  NOAA uses the HYSPLIT system, particularly the particle dispersion simulations, for research and emergency response.  In the scientific community, the trajectory simulations are applied to a variety of tasks, including visualizing regional atmospheric circulation patterns, investigating meteorological controls on the isotopic composition of precipitation, and calculating moisture uptake and transport.  A key component of these research problems is the along-trajectory data that HYSPLIT outputs.  Although the PC and Mac versions allow for greater batch processing than is available via the online READY interface, neither interfaces provide users with a means to inspect, sort or analyze trajectories on the basis of along-trajectory data.  Users are left with limited options: write their own scripts for performing the desired data analysis, or manage trajectory data by hand via spreadsheet and GIS programs.  Both options are time consuming, the latter limits the number of trajectories that can be inspected and is prone to error, and the former is typically not distributed for use to other labs.  Additionally, HYSPLIT ships with limited inbuilt options for trajectory visualization, though it does provide a shapefile/KML output tool.  PySPLIT's key aims are to provide a free, open source, replicable system inside the scientific Python ecosystem for a completely python-based workflow: bulk trajectory generation and for trajectory path and data analysis and visualization.
PySPLIT depends on python's core scientific packages, such as numpy and matplotlib, and comprises five classes and a trajectory generation toolkit.  PySPLIT's scope is currently bulk trajectory generation, trajectory data analysis and management, and path and data visualizations.  Due to the research interests of the author, PySPLIT currently has a particular focus on rainfall, moisture flux, and moisture uptake using trajectories run backwards in time.

The API
-------
The current PySPLIT API comprises five classes.  An early version of PySPLIT was procedurally based.  The helper functions and the constant re-reading of trajectory data files, however, was cumbersome and non-intuitive.  The fundamental class of Pysplit is the Trajectory type.  Each Trajectory represents one air parcel trajectory output by HYSPLIT.  Three of the other classes, TrajectoryGroup, Cluster, and ClusterGroup, are essentially variations on a Trajectory container.  The fifth data type is the MapDesign type, which is not a Trajectory-related class, but holds map construction information and creates a basemap from the stored parameters on command.  This class was included to enable the user to quickly create attractive basemaps without detracting much attention and effort from the trajectory analysis workflow.

Trajectory Generation
Of course, the first step in a HYSPLIT workflow is trajectory generation.  This can be accomplished via the online READY interface or the HYSPLIT GUI or commandline, but bulk generation is annoying and time-consuming.  In response, PySPLIT includes a script for generating large numbers of trajectories in a single call, allowing the user to set up a batch to run overnight without constant user monitoring or action.

This call, ``pysplit.generate_trajectories()``, currently just supports gdas1 data, but brings two additional features not available in the READY interface or directly through HYSPLIT.  The first is triggered by the ``get_foward`` keyword argument in the call:





The PySPLIT API is based on four object classes:
Trajectory: The basic unit of all subsequent classes.  Each object represents one air parcel trajectory output by HYSPLIT and contains its X, Y, Z coordinates, along-path data, path information, start information, and summary data.  Most calculations live in Trajectory methods.  Along-path data are stored as 1-D NumPy arrays of floats

TrajectoryGroup: This class is a container for Trajectory instances, initialized simply by giving pysplit.TrajectoryGroup() a list of Trajectory objects.  This object type is iterable, returning either a single Trajectory or a new TrajectoryGroup containing a slice of the list of Trajectory instances.  A typical workflow includes cycling through the Trajectory instances in a TrajectoryGroup for a particular attribute, placing them in a list, and creating a new TrajectoryGroup from that list of all Trajectory instances that share the desired attribute.  Additional attributes of the TrajectoryGroup, like self.trajcount, are included, and TrajectoryGroup instances also contain methods to summarize (self.grid_data()) and plot the paths and/or the along-path data of all member Trajectory instances.  TrajectoryGroup instances can also be created from the addition of other TrajectoryGroup instances.  During this process, two identical Trajectory instances representing the same trajectory are treated such that only one is included in the new TrajectoryGroup.

Cluster: This is a specialized subclass of TrajectoryGroup

ClusterGroup:  The ClusterGroup class is to Cluster what TrajectoryGroup is to Trajectory.  An object of this type is also fully iterable, able to cycle through Clusters.

MapDesign

Working with Trajectory Data
----------------------------
This is where shp file conversion would be nice
Step through trajectory group, sort into new trajectory groups
(probably merge section in with above)
Error Analysis

Map Making
----------
MapDesign class
Any basemap is accepted
plotting trajectories
plotting scatter data
plotting gridded data
Norms available
GribReader
