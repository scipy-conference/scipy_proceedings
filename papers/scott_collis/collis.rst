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
:institution: NASA Goddard Space Flight Center.

:author: Anne Fridlind
:email: ann.fridlind@nasa.gov
:institution: NASA Goddard Institute of Space Sciences.

:author: Marcis Vanlier-Walqui
:email: marcus.vanlier-walqui@nasa.gov
:institution: NASA Goddard Institute of Space Sciences.



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
forms able to answer our science questions. This is a "how" paper,
covering signal processing using linear programming methods, mapping using KD
Trees, and image analysis using ndimage and, of course graphics using
Matplotlib.

.. class:: keywords

   Remote sensing, radar, meteorology, hydrology

Introduction
------------
RADARs (RAdio Detecion And Ranging, henceforth radars) specialized to weather
applications do not measure the atmosphere, rather, the instument measures the
interaction of the probing radiation with the scattering medium (nominally cloud
or precipitation droplets or ice particulate matter). Therefore, in order to
extract geophysical insight, such as the relationshop between large scale
environmental forcing and heterogeneity of surface precipitation patterns, a
complex application chain of algorithms needs to be set up.

This paper briefly outlines a framework, using a common data model approach, for
assembling such processing chains: the Python-ARM Radar Toolkit, Py-ART
[Heistermann2014]_. The paper also provides an example
application: using rainfall maps to objectively metric the skill of fine scale
models in representing precipitation morphology.

The data source: Scanning centimeter wavelength radar
------------
In order to understand the spatial complexity of precipitating cloud systems a
sensor is required that can collect spatially diverse data. Radars emit a
spatailly descrete pulse of radiation with a particular beam with and pulse length.
A gated reciever that detects the backscattered signal and calculates a number
of measurements based on the radar spectrum (the power as a function of phase delay
which is due). These moments include radar reflectivity factor :math:`Z_e`, radial velocity
of the scattering medium :math:`v_r` and spectrum width :math:`w`. Polarimetric radars transmit
pulses with the electric field vector horizontal to the earth's surface and also
vertical to the earth's surface. These radars can give a measure of the anistropy
of the scattering medium and collect measurements including the differential
reflectivity :math:`Z_{DR}`, differential phase difference :math:`\phi_{dp}` and correlation
cooefficent :math:`\rho_{HV}`. The data is laid out on a time/range grid and each ray
(time step) has an associated azimuth and elevation. Data presented in this paper
are from 4 ARM [Mather2013]_ radar systems: One C-Band (5cm wavelenth) and three X-Band (3cm wavelength)
radars as outlined in table :ref:`radars`.

.. table:: ARM radar systems used in this paper. :label:`radars`

  +-------------+------------------+-----------------+
  |             | X-SAPR           |  C-SAPR         |
  +-------------+------------------+-----------------+
  |Frequency    | 9.4 GHZ          |6.25GHz          |
  +-------------+------------------+-----------------+
  |Transmitter  | Magnetron        |Magnetron        |
  +-------------+------------------+-----------------+
  |Power        | 200kW            | 350kW           |
  +-------------+------------------+-----------------+
  |Gate spacing | 50m              |120m             |
  +-------------+------------------+-----------------+
  |Maximum Range| 40km             |120km            |
  +-------------+------------------+-----------------+
  |Beam width   |1   :math:`^\circ`|1  :math:`^\circ`|
  +-------------+------------------+-----------------+
  |Polar. mode  |Simul. H/V        |Simul. H/V       |
  +-------------+------------------+-----------------+
  |Manufacturer | Radtec           |Adv. Radar Corp. |
  +-------------+------------------+-----------------+
  |Native format| Iris Sigmet      | NCAR MDV        |
  +-------------+------------------+-----------------+

These are arranged as show in :ref:`sgp`.

.. figure:: SGPlayout.png
   :scale: 20%

   Arrangement of radars around the ARM Southern Great Plains Facility from
   [Giangrande2014]_. :label:`sgp`


The Python ARM Radar Toolkit: Py-ART
------------
Radar data comes in a variety of binary formats but the data shape is
essentially the same: A time-range array with data describing the pointing and
geolocating the platform and (for mobile radar) the platform's motion. Py-ART
takes a common data model aproach: Carefully design the data containers and
mandate that functions/methods accept the container as an arguement and return
the same data structure. The common data model for radar data in Py-ART is the
radar object which stores data and metadata in python dictionaries in the fields.
object in the radar structure. Data is stored
in a numpy array and is always in the 'data' key. For example:

.. code-block:: python

  print xnw_radar.fields.keys()
  ['radar_echo_classification', 'corrected_reflectivity', 'differential_phase',
  'cross_correlation_ratio', 'normalized_coherent_power', 'spectrum_width',
  'total_power', 'reflectivity', 'differential_reflectivity', 'specific_differential_phase',
  'velocity', 'corrected_differential_reflectivity']
  print xnw_radar.fields['reflectivity'].keys()
  ['_FillValue', 'coordinates', 'long_name', 'standard_name', 'units', 'data']
  print xnw_radar.fields['reflectivity']['long_name']
  print xnw_radar.fields['reflectivity']['data'].shape
  Reflectivity
  (8800, 801)

So the xnw_radar has a variety of data fields, including 'reflectivity' with the
actual moment data stored in the 'data' key with 8800 time steps and 801 range
gates. Data on instrument pointing is stored in x_nw.azimuth and x_nw.elevation
while the centerpoint of each range gate is stored in x_nw.range. Again these
are dictionaries with data stored in the 'data' key. Methods in Py-ART can append
fields or modify data in existing fields (rare).

The vital key is a 'Babelfish' layer which ingests a variery of formats into the
common data model. As of writing table :ref:`formats` outlines compatibility.
Wrapping NASA's Radar Software Library opened a large number of formats.

.. table:: Py-ART formats. :label:`formats`

  +------------+-------------------------------+--------------+
  |Format name |Example radar system(s)        | Note         |
  +------------+-------------------------------+--------------+
  |CF-Radial   | NCAR SPOL, ARM Cloud Radars   | Output format|
  +------------+-------------------------------+--------------+
  |UF          | Lots of legacy data           | Via RSL      |
  +------------+-------------------------------+--------------+
  |Lassen      | BoM CPOL in Darwin, Australia | Via RSL      |
  +------------+-------------------------------+--------------+
  |IRIS Sigmet | ARM X-SAPR                    | Native       |
  +------------+-------------------------------+--------------+
  |NCAR MDV    | ARM C-SAPR                    | Native       |
  +------------+-------------------------------+--------------+
  |ODIN        |European radar network         | Native       |
  +------------+-------------------------------+--------------+
  |WSR-88D     |USA operational network        | Native       |
  +------------+-------------------------------+--------------+

We also have Pull Requests on GitHub for the NSF funded Colorado State University
CHILL radar and active development on NOAA NOX-P and NASA D3R radars. There is a
single output format, CF-Radial, a NetCDF based community format on which the
common data model is modeleded.

Pre-mapping corrections and calculations
~~~~~~~~~~~~~~~~~~~~~~
Once raw data is collected there is often a number of processing steps that need
to be performed. In our case this includes:

- Correcting false Azimuth readings in the Northwest X-Band system.
- Clean data of undesirable components such as multiple trips, clutter and
  non-meteorological returns.
- Processing the raw :math:`\phi_{DP}` and extracting the component due to
  rain water content by using a Linear Programming technique to fit a profile
  which mandates positive gradient, see [Giangrande2013]_.
- Using reflectivity and :math:`\phi_{DP}` to retrieve attenuation (in dBZ/km)
  due to rainwater path.
- Using the techniques outlined in [Ryzhkov2014]_ to retrieve rainfall rate (in
  mm/hr) from attenuation.

These are all outlined in the first of the three notebooks which accompany this
manuscript: http://nbviewer.ipython.org/github/scollis/notebooks/tree/master/scipy2014/.
Each process either appends a new field to the radar object or returns a field
dictionary. Py-ART also comes with visualization methods allowing for the conical
(or Plan Position Indicator, PPI) scan to be plotted up and geolocated using
Matplotlib and Basemap. An example of raw :math:`\phi_{DP}` and reflectivity
is shown in :ref:`raw_ppi`.

.. figure:: nw_ppi.png

   Raw Reflectivity factor and polarimetric phase difference from the lowest (0.5 degree)
   tilt. :label:`raw_ppi`

The code to plot is simply:

.. code-block:: python

  fields_to_plot = ['differential_phase', 'reflectivity']
  ranges = [(180, 240), (0, 52)]
  display = pyart.graph.RadarMapDisplay(xnw_radar)

  nplots = len(fields_to_plot)
  plt.figure(figsize=[7 * nplots, 4])
  for plot_num in xrange(nplots):
      field = fields_to_plot[plot_num]
      vmin, vmax = ranges[plot_num]
      plt.subplot(1, nplots, plot_num + 1)
      display.plot_ppi_map(field, 0, vmin=vmin, vmax=vmax, lat_lines=np.arange(20,60,.2),
                           lon_lines =  np.arange(-99,-80,.4), resolution = 'l')
      display.basemap.drawrivers()
      display.basemap.drawcountries()
      display.plot_range_rings([20,40])

Again, RadarMapDisplay class __init__ method expects a radar object but is
insensitive to the data source. The sample plotting routines can be used for
any source Py-ART has an ingest for.


Mapping to a cartesian grid
~~~~~~~~~~~~~~~~~~~~~~
Radars sample in radial coordinates of elevation azimuth and range. Mathematics
for atmospheric phenomina are greatly simplified on Cartesian and Cartesian-like
(eg pressure surfaces) grids. Therefore the raw and processed data in the radar
object needs to be mapped onto a regular grid. This is known as "Objective analysis"
(see, for example [Trapp2000]_). In this paper we use a technique known as Barnes
analysis [Barnes1964]_ which is an inverse distance weighting sphere of influence
based technique. For each grid point in the target Cartesian grid a set of radar
gates within a radius of influence are interpolated using a weighting function such as:

.. math::

   W(r) = e^\frac{-r_{infl}^2}{2.0*r^2}

where :math:`r` is the distance from the grid point and :math:`r_{infl}` is the
search radius of influence. The brute force way of doing the calculation would
be for each cartesian point linearly search the radar gates for those within
the radius of influence, an Order :math:`n^2` problem. With a typical grid being
200 by 200 by 37 grid points and a modern radar having on the order of 8000 time
samples and 800 range gates this quickly becomes untractable. A better way is to
store the radar gates in a KD-Tree ordered by distance. This reduces the search
to an order :math:`log(n)` problem. This is implimented in Py-ART. In addition a
variable radius of influence algorithm is implimented which analyzes the radar
volume coverage pattern and deduces an optimized :math:`r_{infl}(x,y,z)`. Unlike
many other objective analysis codes Py-ART accepts a tuple of radar objects and
treats the radar gates as a cloud of points. This allows very simple merging of
multiple radar data sets. The method is simple to invoke, for example:

.. code-block:: python

  mesh_mapped_x = pyart.map.grid_from_radars((xnw_radar,xsw_radar,xse_radar),
                                        grid_shape=(35, 401, 401),
                                        grid_limits=((0, 17000), (-50000, 40000), (-60000, 40000)),
                                        grid_origin = (36.57861, -97.363611),
                                        fields=['corrected_reflectivity', 'rain_rate_A', 'reflectivity'])

will map the three radar objects (in this case the three ARM X-Band systems
in figure :ref:`sgp`) to a grid that is (z,y,x) = (35,401,401) points with a domain
of 0 to 17km in altitude, -50 to 40km in meridional extend and -60 to 40km in
zonal extent. The method returns a grid object which follows a very similar shape
to a radar object: fields are in .fields, geolocation data is in .axes and data
is always in the 'data' key.

Again, as with the radar object Py-ART has a menu of available methods to visualize
grid data as well as an io layer that can inject CF-compliant netCDF grids and write
the grid object out to a CF-complaint file for future analysis and distribution.

For example figure :ref:`C-Band only` shows a slice thought mapped reflectivity
from the ARM C-SAPR at 500m and cross sections at 36.5N degrees latitude and
-97.65E longitude.

.. figure:: c_only_z.png

   Single C-Band reflectivity factor field. :label:`C-Band only`

In the vertical cross sections clear artifacts can be seen due to the poor sampling.
Figure :ref:`X-Band only` shows the same scene but a three radar meshgrid from the
X-Band network.


.. figure:: x_only_z.png

   Reflectivity factor mapped from a network of X-Band radars. :label:`X-Band only`

It is clear more fine scale detail is resolved due to the rain systems being closer
to any given radar. Both radars are mapped onto a grid with 225m spacing.
In addition, due to the density of high elevation beams being
increased (essentially a "web" of radar beams sampling the convective anvil) sampling
artifacts are greatly reduced and finer details aloft are able to be studied.

Of course mesh mapping only works for "specific" mesurements, ie not integrated
measurements like :math:`\phi_{DP}` or directionally dependent moments
like :math:`v_r`. One measurement that can be mapped is our retrieved rain rate.

Figures :ref:`C-Band rain` and :ref:`X-Band rain` show mappings for rain rate
using just the C-Band measurement and X-Band network respectively. Again the
mesh map of the X-Band retrieval shows very fine detail resolving (in a volumetric
dataset) fall streak patterns. The maxima near 4km (just below the freezing
level) is due to melting particles. The rainfall retrieval has a cut off at
the sounding determined freezing level but the "bright band" can extend some depth
below this. Future work will entail using polrimetric measurements to deterimine
where there is only pure liquid returns and conditionally apply the rainfall
retrieval to those positions.

.. figure:: c_only_rain.png

   Single C-Band rainfall field. :label:`C-Band rain`



.. figure:: x_only_rain.png

   Rainfall from a network of X-Band systems. :label:`X-Band rain`


Spatial distribution of rainfall: a objective test of fine scale models
------------

Previous sections have detailed the correction, retrieval from and mapping of radar
data to a Cartesian grid. The last section showed enhanced detail can be retrieved
by using a network of radars. The question remains: how can objectively compare the
detail in rain fields? Both radar derived and forecast model calculated. The meshes
generated using the mapping techniques previously discussed can be treated just like
image data and there are a variety of packages for analyzing images.

Measuring rainshafts using NDimage
~~~~~~~~~~~~~~~~~~~~~~
A simple technique for documenting the detail in an image is to segment it into
"blobs" which are above a certain threshold and calculate the number of blobs,
thier accumilated area and the mean rainfall across the blobs. The ndimage module
of Scipy is the perfect package for achieving this. Figure :ref:`seg` shows the
of ndimage.label to break up regions above

.. figure:: segmentation.png

   An example of figure segmentation using ndimage.label. :label:`seg`




Radar results
~~~~~~~~~~~~~~~~~~~~~~
coolness

Conclusions
------------
stuff

Acknowledgements
------------
DoE Standard


References
----------
.. [Heistermann2014] Heistermann, M., S. Collis, M. J. Dixon, S. E. Giangrande,
              J. J. Helmus, B. Kelley, J. Koistinen, D. B. Michelson, M. Peura,
              T. Pfaff and D. B. Wolff,
              2014: The Promise of Open Source Software for the Weather Radar
              Community. *Bull. Amer. Meteor. Soc.*,
              **In Press.**
.. [Mather2013] Mather, J. H., and J. W. Voyles, 2012:
                The Arm Climate Research Facility: A Review of Structure and
                Capabilities. *Bull. Amer. Meteor. Soc.*, **94**, 377–392,
                doi:10.1175/BAMS-D-11-00218.1.
.. [Giangrande2014] TBD
.. [Giangrande2013] Giangrande, S. E., R. McGraw, and L. Lei,
                     2013: An Application of Linear Programming to Polarimetric
                     Radar Differential Phase Processing.
                     *Journal of Atmospheric and Oceanic Technology*, **30**,
                     1716–1729, doi:10.1175/JTECH-D-12-00147.1.
.. [Ryzhkov2014] Ryzhkov, A. V., M. Diederich, P. Zhang, C. Simmer, 2014:
                 Potential utilization of specific attenuation for rainfall
                 estimation, mitigation of partial beam blockage, and radar
                 networking. Submitted, *J. Atmos. Oceanic Technol.*, **in press.**
.. [Trapp2000] Trapp, R. J., and C. A. Doswell, 2000: Radar Data Objective
               Analysis. *Journal of Atmospheric and Oceanic Technology*,
               **17**, 105–120, doi:10.1175/1520-0426(2000)017<0105:RDOA>2.0.CO;2.
.. [Barnes1964] Barnes, S. L., 1964: A Technique for Maximizing Details in
                Numerical Weather Map Analysis. *Journal of Applied Meteorology*,
                **3**, 396–409, doi:10.1175/1520-0450(1964)003<0396:ATFMDI>2.0.CO;2.
