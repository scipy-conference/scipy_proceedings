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
Of course, the first step in a HYSPLIT workflow is trajectory generation.  This can be accomplished via the online READY interface or the HYSPLIT GUI or commandline, but bulk generation is annoying and time-consuming.  In response, PySPLIT includes a script for generating large numbers of trajectories at various times of day and at several different altitudes in a single call, allowing the user to set up a comprehesive batch to run overnight without constant user monitoring or action.

This call, ``pysplit.generate_trajectories()``, currently just supports gdas1 data, but brings two additional features not available in the READY interface or directly through HYSPLIT.  The first is triggered by the ``get_foward`` keyword argument in the call:

.. code-block:: python

    generate_trajectories(basename, hysplit_working, output_dir, meteo_path,
                          years, months, hours, altitudes, coordinates, run,
                          isbackward, meteo_type='gdas1', get_forward=True,
                          get_clippedtraj=True)

If back trajectories are run, then PySPLIT can automatically open the new back trajectory file, read in the altitude, longitude, and latitude of the last time point, and initialize a forward-moving parcel at that location (or at the given coordinates below the 10000 m ceiling, if necessary).  In the Trajectory class, a method is included for inspecting the distance between the endpoint of the forward trajectory and the starting point of the back trajectory, allowing the user to make a judgement about the trajectory integration error.

The second utility is triggered by ``get_clippedtraj``.  The along-trajectory data output by HYSPLIT may span multiple lines if more than seven of nine possible available output variables are selected.  This is not a problem for PySPLIT or HYSPLIT except during clustering, which fails given multi-line output.  ``pysplit.clip_traj()`` opens a trajectory file, copies the trajectory header and path data, and outputs the header and path to a new file that HYSPLIT will readily use to perform clustering.

On Windows 8.1 system with 8GB of RAM, can generate about 350 trajectories in five minutes, depending on .  The output files are extensionless and live in  ``output_dir``, which also contains a subdirectory for the corresponding clipped trajectories and forward trajectories.

Trajectory Class
The Trajectory class is the fundamental unit in PySPLIT, designed to manage and promote the analysis of air-parcel trajectory data in an intuitive way.  Each object represents one air parcel trajectory calculated by HYSPLIT and contains its latitude, longitude, and altitude (m above ground level or m above sea level), along-path data, path start information, and summary data.  Along-path data are parsed into separate attributes as 1-D NumPy arrays of floats.  This means was chosen for the degree of exposure it provides, The original 2D array of data loaded from HYSPLIT and the header are also stored, and can be reloaded into attributes at any time, wiping out changes.

Most Trajectory analysis methods live in the Trajectory class.  These include calculations of along-trajectory and overall great-circle distance, humidity data conversions, along-trajectory moisture flux, and a flexible implementation of Sodeberg's moisture uptake calculation.  The results of most of these calculations are stored as new attributes, in 1D ndarrays of floats of identicial size.  The moisture uptake results are stored in a 2D array with a separate header, similar to how the original trajectory data is managed.

TrajectoryGroup
The TrajectoryGroup is the basic container for PySPLIT Trajectory objects, and is initialized simply by providing a list of Trajectory objects.  This class is also fully iterable, yielding a Trajectory when indexed and a new TrajectoryGroup when sliced.  A typical workflow begins by loading trjectory data into Trajectory objects and creating a new Trajectory group:

..code-block:: python

  #Load trajectories as Trajectory objects from file
  # Create TrajectoryGroup umn
  umn, _ = pysplit.make_trajectorygroup(r'C:/traj/minn*')

Then if necessary sort out the trajectories with desirable characteristics into a new TrajectoryGroup:

..code-block:: python

  # Sort out rain-bearing trajectories starting at 1700 UTC and 1500 m
  # 1700 UTC is noon at parcel launch site, the University of Minnesota, Twin # Cities
  umn_trajlist=[]

  for traj in umn:
    traj.set_rainstatus()
    if traj.rainstatus and traj.hour[0] == 17 and traj.altitude[0] == 1500:
      umn_trajlist.append(traj)

  # Create new TrajectoryGroup containing trajectories meeting above criteria:
  umn_rainy1500noon = pysplit.TrajectoryGroup(umn_trajlist)

And perform analyses:

..code-block:: python

  # Set attributes of Trajectory objects in new TrajectoryGroup
  for traj in umn_rainy1500noon:
    traj.set_vector()
    traj.set_specifichumidity()
    traj.set_distance()
    traj.dq_dw_dh()
    traj.calculate_moistureflux()

Repeating sorting and analysis as necessary.

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
