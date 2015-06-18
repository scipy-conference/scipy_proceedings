:author: Mellissa Cross
:email: cros0324@umn.edu, mellissa.cross@gmail.com
:institution: Department of Earth Sciences, University of Minnesota

:video: http://www.youtube.com/watch?v=dhRUe-gz690

-----------------------------------------------------------------------------------------------------
PySPLIT: a Package for the Generation, Analysis, and Visualization of HYSPLIT Air Parcel Trajectories
-----------------------------------------------------------------------------------------------------

.. class:: abstract

   The HYSPLIT model outputs air parcel paths projected forwards or backwards in time (trajectories) and is used in a variety of scientific contexts.  Here we present the first package in the mainstream scientific Python ecosystem designed to facilitate HYSPLIT trajectory analysis workflow by providing an elegant, intuitive API for generating, inspecting, and plotting trajectory paths and data.

.. class:: keywords

   HYSPLIT, trajectory analysis, matplotlib Basemap

Introduction
------------
The NOAA Air Resources Laboratory's HYSPLIT (HYbrid Single Particle Lagrangian Transport) model is publicly available via the web READY interface- and has been since the late 1990s- or downloadable versions compatible with PC or Mac.  NOAA uses the HYSPLIT system, particularly the particle dispersion simulations, for research and emergency response.  In the scientific community, the trajectory simulations are applied to a variety of tasks, including visualizing regional atmospheric circulation patterns, investigating meteorological controls on the isotopic composition of precipitation, and calculating moisture uptake and transport.

Trajectory Generation
---------------------
Bulk, need to add daily
need to add on condensation levels and pressure surfaces
Should add tool that changes parameters like T_RATIO
Also should add tool that allows you to change the meteorological variables without having to go into the text file yourself.

The API
-------
The PySPLIT API comprises five classes
PySPLIT

The PySPLIT API is based on four object classes:
Trajectory: The basic unit of all subsequent classes.  Each object represents one air parcel trajectory output by HYSPLIT and contains its X, Y, Z coordinates, along-path data, path information, start information, and summary data.  Most calculations live in Trajectory methods.  Along-path data are stored as 1-D NumPy arrays of floats

TrajectoryGroup: This class is a container for Trajectory instances, initialized simply by giving pysplit.TrajectoryGroup() a list of Trajectory objects.  This object type is iterable, returning either a single Trajectory or a new TrajectoryGroup containing a slice of the list of Trajectory instances.  A typical workflow includes cycling through the Trajectory instances in a TrajectoryGroup for a particular attribute, placing them in a list, and creating a new TrajectoryGroup from that list of all Trajectory instances that share the desired attribute.  Additional attributes of the TrajectoryGroup, like self.trajcount, are included, and TrajectoryGroup instances also contain methods to summarize (self.grid_data()) and plot the paths and/or the along-path data of all member Trajectory instances.  TrajectoryGroup instances can also be created from the addition of other TrajectoryGroup instances.  During this process, two identical Trajectory instances representing the same trajectory are treated such that only one is included in the new TrajectoryGroup.

Cluster: This is a specialized subclass of TrajectoryGroup

ClusterGroup:  The ClusterGroup class is to Cluster what TrajectoryGroup is to Trajectory.  An object of this type is also fully iterable, able to cycle through Clusters.

MapDesign
Object-oriented
Trajectory objects
TrajectoryGroup objects
Cluster objects as a subset of TrajectoryGroups
ClusterGroup objects
Manipulating trajectories:  TrajectoryGroups, Clusters, and Clustergroups fully iterable
Should I make trajectories iterable (step through time points)?

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
