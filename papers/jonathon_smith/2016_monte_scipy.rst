:author: Jonathon Smith, William Taber, Theodore Drain, Scott Evans, James Evans, Michelle Guevara, William Schulze, Richard Sunseri, Hsi-Cheng Wu
:email: jonathon.j.smith@jpl.nasa.gov
:institution: Jet Propulsion Laboratory, California Intitute of Technology / NASA
:corresponding:

--------------------------------------
MONTE Python for Deep Space Navigation
--------------------------------------

.. class:: abstract

The Mission Analysis, Operations, and Navigation Toolkit Environment
(MONTE) is the Jet Propulsion Laboratory's (JPL) signature astrodynamic
computing platform. It was built to support JPL's deep space exploration
program, and has been used to fly robotic spacecraft to Mars, Jupiter,
Saturn, Ceres, and many solar system small bodies. At its core, MONTE
consists of low-level astrodynamic libraries that are written in C++
and presented to the end user as an importable Python language module.
These libraries form the basis on which Python-language applications
are built for specific astrodynamic applications, like trajectory
design and optimization, orbit determination, flight path control, and
more. This paper gives a brief history of the project, shows some
examples of MONTE in action, and relates the stories of its greatest
successes.

.. class:: keywords

   astrodynamics, JPL, NASA

History
-------

MONTE is the latest in a long-line of Deep Space Navigation software
sets developed at JPL spanning back to the dawn of the space age. The
first artificial satellite sent to orbit by the United States was
Explorer I, and when it launched in 1958, NASA had yet to be
established as an official government agency. CalTech and its
Jet Propulsion Laboratory led the development and operations effort
for Explorer I.

In the 1960's, JPL organized its growing expertise in trajectory
design and spacecraft orbit determination into the Single Precision
Orbit Determination Program (SPODP), which supported the early robotic
reconnaissance of the Moon and inner solar system by navigating the
Ranger, Surveyor, Lunar Orbiter, and early Mariner and Pioneer
missions.

Starting in 1964, a group of engineers, led by Ted Moyer, began
developing the algorithms and software which would eventually become
the Double Precision Trajectory and Orbit Determination Program
(DPTRAJ/ODP). The DPTRAJ/ODP was used by JPL to navigate the "Golden Age"
of deep space exploration, including the later Mariner and Pioneer
Missions, Viking, Voyager, Magellan, Galileo and Cassini. This work
was codified in Moyer's foundational orbit determination papers
[Moy71]_ and [Moy03]_.

The story of MONTE begins in 1998, when JPL's navigation section
commissioned an update to the aging DPTRAJ/ODP library. The primary
idea was translate the Fortran-based legacy software into a more
maintainable, extensible and better tested C++ / Python application.
Throughout the first half of the 2000s, the reorganization of the
navigation algorithms into a rigorously tested, object-oriented
software package continued. In 2007, MONTE got its first operational
assignment navigating the Mars Phoenix mission. As MONTE grew in
capability over the next five years, all the remaining DPTRAJ/ODP
missions transitioned to MONTE.

Deep Space Navigation
---------------------

The practice of navigating spacecraft in deep space encompasses three
interelated disciplines: (1) Designing a reference trajectory which describes
the planned flight path of the spacecraft (mission design), (2) keeping track
of the actual spacecraft position while the mission is in flight (orbit
determination), and (3) designing maneuvers to bring the spacecraft back to
the reference trajectory when it has strayed (flight path control, Figure
:ref:`tour`).

.. figure:: figures/cassinitour.png

    Illustration of Cassini's reference trajectory at Saturn. The mission
    designers built this trajectory, and the orbit determination and maneuver
    design teams keep the spacecraft flying on these orbits during the
    mission. :label:`tour`

The process of designing a spacecraft reference trajectory begins at the very
earliest stages of mission planning. The mission design navigators work very
closely with the science teams to put together a reference orbit that allows
the spacecraft to take all the desired science measurements. They also work
with mission planners and spacecraft system engineers to make sure that the
spacecraft is able to withstand the rigors of it's planned trajectory.
After iterating through design after design, a process which often takes years
(and may still be revised later while the spacecraft is in flight), the result
is the mission reference trajectory. It will be up to the orbit determination
and flight path control teams to make sure the spacecraft actually follows this
flight plan when the spacecraft finally launches.

The job of the orbit determination team is to keep track of where the
spacecraft has been (orbit reconstruction), is currently (orbit determination),
and where it will go in the future (orbit prediction). The spacecraft is always
drifting away from it's planned flight path because of disturbances it
encounters in space. Even small disturbances, like the pressure of sunlight on
the spacecraft, can add up over time and push the spacecraft off course. The
mission designers do their best to account for these disturbances while creating
the reference orbit, but there is no accounting for randomness and
unpredictability of the real world. To further complicate matters, once the
spacecraft leaves the launch-pad, it can no longer be directly observed anymore.
Orbit determination analysts must process various forms of tracking data that
are tied mathematically to the evolution of the spacecraft orbit to figure out
it's position at any given time.

Once the orbit determination team has a good estimate for the current location
of the spacecraft, the flight path control team is responsible for evaluating
how far the spacecraft has drifted from the reference trajectory and designing
a maneuver to get the spacecraft back on course. The result of this maneuver
design is a delta-V vector, which stands for delta-Velocity or change in
velocity. This delta-V vector represents the direction and magnitude of the
required change in the spacecraft velocity which must be accomplished to get
the spacecraft back on course. Once in hand, this delta-V vector will be sent
to the spacecraft propulsion team, who will decompose it into actual thruster
firings on the spacecraft. These will be uplinked to the spacecraft, which will
then perform the maneuver.

After a maneuver has been performed, the cycle repeats. Perhaps the thrusters
were slightly misaligned, or the engine cutoff was a second too late. The orbit
determination team must take more tracking data to find out. This iterative
relationship between orbit determination and flight path control continues
without pause through the lifetime of a flight mission. The spacecraft is
constantly wandering off, and must be patiently nudged back on course.

Library Overview
----------------

The Mission Analysis, Operations, and Navigation Toolkit Environment
(MONTE) is JPL's signature astrodynamic computing platform. It
supports all phases of space mission development, from early stage
mission design and analysis through flight navigation services.

Most of the functionality of MONTE is encapsulated in the ``Monte`` and
``mpy`` libraries. ``Monte`` is written in C++ and wrapped in Python.
It is presented to the end user as a normal, importable Python-language
module. The ``mpy`` module is written entirely in Python, and contains
higher level applications built using ``Monte`` and other open-source
Python libraries.

Convention is to import the main ``Monte`` library as ``M``. Throughout
this paper, if a class is referred to with the prefix ``M.``, it means
this class belongs to the main MONTE library (e.g. ``M.TrajLeg``,
``M.Gm``, etc). The following example shows a simple script using the
``Monte`` and mpy libraries to get the state of the Cassini spacecraft
with respect to Saturn at the time of its Saturn Orbit Insertion (SOI)
burn. [#]_

.. [#] All MONTE code in this paper is current as of the v116 delivery.

.. code-block:: python

    import Monte as M
    import mpy.io.data as defaultData

    # Set up a project BOA database, and populate it with base
    # astrodynamic data from the default data depot
    boa = defaultData.load([ "time", "body", "frame",
      "ephem/planet/de405"])

    # Load the Saturn satellite ephemeris and Cassini
    # trajectory into # our BOA database
    boa.load("saturn_satellites.boa")
    boa.load("cassini_trajectory.boa")

    # Define time of SOI
    soiTime = M.Epoch("01-JUL-2004 02:48:00 UTC")

    # Get the trajectory manager from the BOA database
    tset = M.TrajSetBoa.read(boa)

    # Request the state of Cassini at SOI from the trajectory
    # manager in a Saturn-centered Earth Mean Orbit of 2000
    # coordinate frame.
    casAtSoi = tset.state(soiTime, "Cassini", "Saturn",
      "EMO2000")

Lets take time now to walk through some of MONTE's core systems,
several of which were used in the above example.

Core Library Systems
--------------------

MONTE's core systems are the scaffolding that support its more advanced
functionality. Time, units, trajectories, coordinate frames, event
finding, and tying them all together, MONTE's data broker, the Binary
Object Archive or BOA.

BOA
^^^

The Binary Object Archive (BOA) is MONTE's primary data management
system. Most MONTE classes that define concrete objects (for instance,
``M.Gm`` which defines a natural body GM, or ``M.FiniteBurn`` which
defines a spacecraft burn) are stored in BOA, and accessed by MONTE's
astrodynamic functions from BOA.

BOA is based on the binary XDR data format, which allows data to be
written-to and read-from binary on different operating systems and
using different transport layers (e.g. you can read and write locally
to your hard disk, or over a network connection).

The role that BOA plays in MONTE can perhaps be best understood as
"defining the universe" that MONTE's astrodynamic tools operate on.
In our example, we populated our "model universe" (e.g. our BOA
database) with time systems, natural body data, a planetary ephemeris,
the Cassini spacecraft trajectory, etc. We then asked MONTE's trajectory
manager (an astrodynamic tool) to examine this particular universe and
return the state of Cassini with respect to Saturn.

Default Data
^^^^^^^^^^^^

A standard MONTE installation comes with a collection of predefined,
publicly available astrodynamic datasets (the "default data depot").
These can be accessed and loaded into a BOA database via MONTE's
default data loader (``mpy.io.data``) and serve to help an analyst get a
"model universe" up and running quickly.

Time and Units
^^^^^^^^^^^^^^
MONTE has support for the Te, TDT, TAI, GPS, UTC, and UT1 time systems.
The primary class used for dealing with time is ``M.Epoch`` which
stores specific times and also allows a user to convert between
different time frames.

MONTE's unit system supports the notions of time, length, mass, and
angle. It has implemented operator overloading to allow unit
arithmetic, e.g. dividing a unit length by a unit time results in unit
velocity. Most functions that accept unit-quantities also check their
inputs for correctness, so supplying a unit length to a function that
expects unit time will raise an exception.

Trajectories
^^^^^^^^^^^^

MONTE models spacecraft and natural body trajectories in a number of
underlying formats; most of the differences involve how many data
points along the trajectory are actually stored, and how to
interpolate between these points. In addition, MONTE provides
conversion routines which allow some external trajectory formats to
be read and written (including NAIF "bsp" files, international "oem"
files).

The ``M.TrajSet`` class is MONTE's trajectory manager, and is
responsible for coordinating state requests between all of the
trajectories loaded into a given BOA database. It has access to
the coordinate frame system (described in the next section) allowing
it to make coordinate frame rotations when doing state queries. In fact,
most coordinate frame rotations in MONTE are accomplished by simply
requesting a state from ``M.TrajSet`` in the desired frame.

The general steps for building and using trajectories in MONTE are
illustrated in Figure :ref:`trajfig`.

.. figure:: figures/traj.png

   Dataflow through MONTE's trajectory system :label:`trajfig`

Coordinate Frames
^^^^^^^^^^^^^^^^^

The MONTE trajectory and coordinate frame systems are very analogous,
and have a tight integration that enables powerful state requests.
Figure :ref:`trajcoordfig` illustrates these similarities and how the
two systems are integrated.

MONTE models coordinate frames in a number of underlying formats and
provides conversion routines which allow some external coordinate
frame formats to be read and written (including
NAIF "ck"files).

.. figure:: figures/traj_coord.png

   Cooperation between MONTE's trajectory and coordinate frame systems :label:`trajcoordfig`

Event Finding
^^^^^^^^^^^^^

MONTE allows a user to search through astrodynamic relationships in a
given BOA database in pursuit of particular events. For instance, the
``M.AltitudeEvent`` class allows a user to search for when a spacecraft
is within a certain altitude range from another body.

Exploring bodies in motion
--------------------------

We now take a step further and show how we can use the systems described
above to explore astrodynamic relationships. For the following
examples we will be using the Voyager 2 spacecraft ephemeris, which can
be downloaded at http://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/. The
file name at the time of this writing is "voyager_2.ST+1992_m05208u.merged.bsp",
which we will shorten to just "voyager2.bsp" for ease of use.

*JPL hosts two excellent websites for accessing trajectory data for
natural solar system bodies and deep-space probes. The Horizons website (
http://ssd.jpl.nasa.gov/horizons.cgi) is maintained by JPL's Solar System
Dynamics group, and has a very expansive and powerful webapp for getting
ephemerides in a variety of formats. The Navigation and Ancillary Data
Facility (NAIF) at JPL hosts the navigation section of NASA's Planetary
Database System. At it's website (http://naif.jpl.nasa.gov/naif/data.html)
you will find a host of downloadable binary navigation files, which can be
used with the SPICE toolkit, and of course, with MONTE.*

Voyager 2 Trajectory
^^^^^^^^^^^^^^^^^^^^

Lets start off by creating a BOA database and loading the default data sets for
planetary ephemerides, coordinate frames, and body data. We will also load in
our Voyager 2 trajectory.

.. code-block:: python

   In [1]: import Monte as M
   In [2]: import mpy.io.data as defaultData
   In [3]: boa = M.BoaLoad()
   In [4]: defaultData.loadInto( boa,
      ...:   ["ephem/planet/de405", "frame", "body"] )
   In [5]: boa.load( "voyager2.bsp" )

Now lets retrieve the TrajSet manager from the BOA. As previously mentioned,
any BOA that has one or more trajectories will automatically be assigned a
TrajSet to manage them. We will use the BOA accessor TrajSetBoa to get
the TrajSet from the BOA. Once we have the TrajSet, we list all the
trajectories that are on the BOA (and being managed) using the
``.getAll`` method.

.. code-block:: python

   In [6]: traj = M.TrajSetBoa.read( boa )
   In [7]: traj.getAll()
   Out[7]: ['Mercury', 'Mercury Barycenter',
            'Venus', 'Venus Barycenter',
            'Earth', 'Earth Barycenter', 'Moon',
            'Mars', 'Mars Barycenter',
            'Jupiter Barycenter', 'Saturn Barycenter',
            'Uranus Barycenter', 'Neptune Barycenter',
            'Pluto Barycenter', 'Sun'
            'Solar System Barycenter', 'Voyager 2']

Good, so we have our solar system and our spacecraft. Now lets see what we can
start doing. First, lets check the span of our Voyager 2 trajectory, e.g. the
interval for which there is data about it's trajectory, using the
``TrajSet.totalInterval`` method. Note that if the trajectory has been updated at
the NAIF PDS website, the exact span you get may be different than what is
listed below.

.. code-block:: python

   In [8]: traj.totalInterval( "Voyager 2" )
   Out[8]:
   TimeInterval(
      [ '20-AUG-1977 15:32:32.1830 ET',
        '05-JAN-2021 00:00:00.0000 ET' ],
   )


It looks like the trajectory starts just post-launch in 1977, extends through
the present and has predictions out into the future. As a first step, lets find
the distance of Voyager 2 from Earth right now. We can use the ``Epoch.now``
static method to get the current time, and then use our TrajSet to request
the state of Voyager 2 with respect to Earth at the current time.

.. code-block:: python

   In [11]: currentTime = M.Epoch.now()
   In [12]: vygrTwoNow = traj.state(currentTime,
       ...:   "Voyager 2", "Earth", "EME2000" )
   In [13]: vygrTwoNow
   Out[13]:
   State (km, km/sec)
   'Earth' -> 'Voyager 2' in 'EME2000'
   at '06-JUN-2014 19:58:35.1356 TAI'
   Pos:  4.358633010242671e+09 -7.411125552099214e+09
        -1.302731854689579e+10
   Vel: -2.415141211951430e+01  2.640692963340520e+00
        -1.128801136174438e+01

To get the current state of the Voyager 2 spacecraft we used the
``TrajSet.state`` method, passing in the time at which to retrieve the state,
the target body, the reference body, and the coordinate frame to return the
state in. Because TrajSet has a global view of all the different
trajectories in our BOA, we could request the state of Voyager 2 with respect
to any other body for which we have a trajectory.

.. code-block:: python

   In [14]: vygrTwoNowVenus = traj.state( currentTime,
       ...:   "Voyager 2", "Venus", "EME2000" )
   In [15]: vygrTwoNowVenus
   Out[15]:
   State (km, km/sec)
   'Venus' -> 'Voyager 2' in 'EME2000'
   at '06-JUN-2014 19:58:35.1356 TAI'
   Pos:  4.216416788778397e+09 -7.523453172910529e+09
        -1.306899257275581e+10
   Vel: -4.457126033807687e+00 -3.509301445530399e+01
        -2.760459587874612e+01

   In [17]: vygrTwoNowNeptune = traj.state(currentTime,
       ...:   "Voyager 2", "Neptune Barycenter", "EME2000" )
   In [18]: vygrTwoNowNeptune
   Out[18]:
   State (km, km/sec)
   'Neptune Barycenter' -> 'Voyager 2' in 'EME2000'
   at '06-JUN-2014 19:58:35.1356 TAI'
   Pos:  2.423407540346480e+08 -5.860459060720786e+09
        -1.229435420991246e+10
   Vel:  2.036299646730726e+00 -8.760646249684767e+00
        -1.606470435709401e+01

The object returned by the ``TrajSet.state`` method is a MONTE State class.
The State class captures the relative position, velocity and acceleration
(or some subset) of one body with respect to another at a given time. It has a
number of methods that help with extracting and transforming the information it
contains. For instance, we can find the magnitude of the distance from Earth to
Voyager 2 like this.

.. code-block:: python

   In [26]: vygrTwoPoskm = vygrTwoNow.posMag()
   In [27]: vygrTwoPoskm
   Out[27]:  1.560876331389678e+10 * km

   In [28]: vygrTwoPoskm.convert( 'AU' )
   Out[28]: 104.33813824888766

Likewise, State has methods to get the magnitude of the relative velocity
(``.velMag``) and acceleration (``.accMag``), and much more. Often, when you
are reading states from a trajectory, you are interested making repeated calls
for the same body with respect to the same center, but at a number of different
times. TrajSet works fine for this application, but if the target and
center bodies don't change on repeated calls, some optimizations can be made for
better performance. The TrajQuery class is provided for this use case, and
can be thought of as simply a special case of TrajSet - where the body and
center are fixed for every call.

.. code-block:: python

   In [29]: vygrTwoQuery =  M.TrajQuery( boa,
       ...:   "Voyager 2", "Earth", "EME2000" )
   In [31]: vygrTwoQuery.state( currentTime )
   Out[31]:
   State (km, km/sec)
   'Earth' -> 'Voyager 2' in 'EME2000'
   at '06-JUN-2014 19:58:35.1356 TAI'
   Pos:  4.358633010242671e+09 -7.411125552099214e+09
        -1.302731854689579e+10
   Vel: -2.415141211951430e+01  2.640692963340520e+00
        -1.128801136174438e+01

In addition to providing optimization and a simpler interface, TrajQuery also
lets you control how light-time corrections are applied (this is a more
advanced use case, so we will only mention it here).

Uranus Encounter
^^^^^^^^^^^^^^^^

We said earlier that TrajSet and CoordSet, in their role as manager
classes, have a global view of the trajectory and coordinate systems. This
high-level perspective allows them to work with the *relationships* between
different bodies and frames, a capability we have so far used in a general
sense, primarily to get states between bodies in a given coordinate
frame at a given time. However, there are certain specific relationships
between bodies and frames that can be of great interest to an analyst. For
instance, identifying the time at which two bodies achieve their closest
approach (periapse), and the magnitude of that minimum distance, can
be a very important astrodynamic metric. We could certainly estimate these
quantities using trajectory queries, perhaps by plotting the relative distance
between the two bodies, and looking for the local minima. However, MONTE
provides us with an infrastructure for searching through various
relationship-spaces and identifying some of these key events. This
infrastructure is composed of EventSpec classes, which allow us to define
the type of event we are looking for and search through the requisite
relationships to identify occurrences, and an Event class which is used to
report the relevant data associated with an occurrence. Lets see how this works
in practice.  We will use ApsisEvent (which is a specific type of
EventSpec) to find the precise time and distance of Voyager 2's closest
approach with Uranus. The first step is to define our ApsisEvent.


.. code-block:: python

   In [6]: vygrTwoUranusQuery = M.TrajQuery( boa,
      ...:   "Voyager 2", "Uranus Barycenter", "EME2000" )
   In [7]: apsisSearch = M.ApsisEvent( vygrTwoUranusQuery,
      ...:   "PERIAPSIS" )

ApsisEvent takes as it's first argument a TrajQuery instance that is
configured to return the state of our target body with respect to the desired
center (in this case, the state of Voyager 2 with respect to Uranus). The
second argument specifies what type of apsis we are looking for; this can
be "PERIAPSIS", "APOAPSIS", or "ANY", which returns occurrences of both periapse
and apoapse. Now that we have our event type defined, we can use the ``.search``
method to locate the apsis. To run the search, we need to provide a time
interval to search over, and a search step size.

.. code-block:: python

   In [14]: searchInterval = M.TimeInterval(
       ...:   "01-JAN-1986 ET", "01-JAN-1987 ET" )
   In [15]: stepSize = 60 * sec
   In [16]: foundEvents = apsisSearch.search(
       ...:   searchInterval, stepSize )

The result of the search, which we have saved in the variable ``foundEvents``,
is an EventSet container class. This container will have all Events
found matching the EventSpec in the search window. In our case, there
should be only one close encounter with Uranus, so there should be only a
single event inside our EventSpec. However, if we were searching for the
periapse for an Earth orbiter, for instance, this would contain every separate
periapse occurrence in the search window. EventSet has a number of useful
methods for sorting and filtering events, however, since we should have only
one event (which we can confirm using the ``.size`` method on the EventSet),
we can read it out directly.

.. code-block:: python

   In [17]: foundEvents.size()
   Out[17]: 1

   In [18]: uranusPeriapse = foundEvents[0]
   In [19]: uranusPeriapse
   Out[19]:
   Event:
   Spec : Periapsis Uranus Barycenter to Voyager 2
   Type : Periapsis
   Epoch: 24-JAN-1986 17:59:45.6473 ET
   Value:  1.071300446056250e+05 * km

Another relationship which can play a significant role in deep space missions
is the angular offset between the Earth-Sun line and Earth-Spacecraft line
(often referred to as the Sun-Earth-Probe (SEP) angle). At low SEP values, the
spacecraft appears very close to the Sun from the vantage of Earth, requiring
radio transmissions from Earth to pass through the near-solar environment before
reaching the spacecraft. The highly-charged solar atmosphere can interfere with
the radio signal, which is why flight projects try to avoid performing critical
mission operations during periods of low-SEP. Lets set up an event search to find
periods of low-SEP for Voyager 2, from mission start through the end of our
trajectory data. We will do this using the AngleEvent event specification
class.

.. code-block:: python

   In [20]: sepSearch = M.AngleEvent(boa, "Sun", "Earth"
       ...:   "Voyager 2", 12 *deg, "BELOW")
   In [23]: searchWindow = traj.totalInterval("Voyager 2")
   In [25]: foundEvents = sepSearch.search(searchWindow,
       ...:   1 *hour)

We constructed our AngleEvent by defining the Sun-Earth-Probe angle - we
input the Sun for body one, the Earth as the center, and Voyager 2 as body two.
We also set twelve degrees as the angle of interest, and instructed the
specification to record periods "BELOW" twelve degrees as an event. There are
several other ways to define an AngleEvent that may be more appropriate to
other applications, but this one suited our purposes perfectly. Once again, we
can call the ``.size`` method on the returned EventSet to find how many
low-SEP windows we found. We can also use the ``.maxInterval`` and
``.minInterval`` methods to search for the largest and smallest SEP windows,
respectively.

.. code-block:: python

   In [26]: foundEvents.size()
   Out[26]: 15

   In [52]: foundEvents.maxInterval()
   Out[52]:
   Event:
   . . .
   Type : Angle below  1.200000000000000e+01 * deg
   Begin: 28-JUN-1978 07:34:09.7021 ET
   End  : 03-AUG-1978 05:22:28.3997 ET
   Value:  1.199999999999977e+01 * deg

   In [53]: foundEvents.minInterval()
   Out[53]:
   Event:
   . . .
   Type : Angle below  1.200000000000000e+01 * deg
   Begin: 31-DEC-1992 09:35:21.3322 ET
   End  : 07-JAN-1993 21:30:07.6066 ET
   Value:  1.199999999999999e+01 * deg

We can loop through all the events found in our search using Python iterator
syntax, and print out the time periods of each found low-SEP region.

.. code-block:: python

   In [56]: for event in foundEvents:
       ...:    print event.interval()
       ...:
   TimeInterval(
      [ '28-JUN-1978 07:34:09.7021 ET',
        '03-AUG-1978 05:22:28.3997 ET' ],
   )
   TimeInterval(
      [ '29-JUL-1979 03:25:57.3664 ET',
        '31-AUG-1979 14:35:53.2033 ET' ],
   )

   . . .

   TimeInterval(
      [ '26-DEC-1991 13:45:23.6951 ET',
        '12-JAN-1992 23:46:40.4029 ET' ],
   )
   TimeInterval(
      [ '31-DEC-1992 09:35:21.3322 ET',
        '07-JAN-1993 21:30:07.6066 ET' ],
   )

It looks like low-SEP periods occur on a near-yearly basis, which makes sense;
as the Earth makes a complete rotation around the Sun, there is bound to be a
period of time when the Sun falls in the line-of-sight of Voyager 2. Curiously
though, the last found low-SEP region was in the winter of 1992. After this
time, the Sun no longer obscures the Earth's view of Voyager 2 at all! We
suspect that something must have happened to the orbit of Voyager 2 sometime
previous to 1992 to change the annual low-SEP viewing geometry dynamic. If
Voyager 2 were to somehow leave the plane of the solar-system, the Earth would
have a constant unobstructed view of the spacecraft permanently. Lets
investigate this theory by looking at the distance of Voyager 2 from the
solar-system ecliptic plane. We will do this by setting up a trajectory query
to return the state of Voyager 2 with respect to the Sun in EMO2000 coordinates.
The Z-component of the position vector will then yield the offset from the
ecliptic plane. We will plot this distance over the course of the Voyager 2
mission and see how this distance evolved.

.. code-block:: python

   In [63]: eclipticQuery = M.TrajQuery(boa,
       ...:   "Voyager 2", "Sun", "EMO2000")
   In [64]: searchWindow
   Out[64]:
   TimeInterval(
      [ '20-AUG-1977 15:32:32.1830 ET',
        '05-JAN-2021 00:00:00.0000 ET' ],
   )

   In [65]: sampleTimes = M.Epoch.range(
       ...:   '21-AUG-1977 ET', '04-JAN-2021 ET', 1 *day)
   In [66]: z = []
   In [67]: for time in sampleTimes:
       ...:   state = eclipticQuery.state( time )
       ...:   z.append( state.pos()[2] )
       ...:
   In [68]: import mpylab
   In [69]: fig, ax = mpylab.subplots()
   In [70]: ax.plot( sampleTimes, z )
   In [71]: ax.set_xlabel( "Date" )
   In [72]: ax.set_ylabel(
       ...:   "Distance from Ecliptic Plane (Km)" )

The resulting plot should look similar to Figure :ref:`v2aturanus`.

.. figure:: figures/v2aturanus.png

    Distance in kilometers of Voyager 2 from the solar system
    ecliptic plane. :label:`v2aturanus`

Sure enough, it appears something happened in 1989 that caused Voyager 2 to
depart from the ecliptic plane. A quick glance at the Wikipedia page
for Voyager 2 confirms this, and reveals the cause of this departure.

   *Voyager 2's closest approach to Neptune occurred on August 25, 1989 ...
   Since the plane of the orbit of Triton is tilted significantly with respect
   to the plane of the ecliptic, through mid-course corrections, Voyager 2 was
   directed into a path several thousand miles over the north pole of Neptune
   ... The net and final effect on the trajectory of Voyager 2 was to bend its
   trajectory south below the plane of the ecliptic by about 30 degrees.*


References
----------

.. [Ntr12] R. Sunseri, H.-C. Wu, S. Evans, J. Evans, T. Drain, and M. Guevara, *Mission Analysis, Operations, and
         Navigation Toolkit Environment (MONTE) Version 040*, NASA Tech Briefs , Vol. 36, No. 9, 2012.

.. [Moy71] T. Moyer, *Mathematical Formulation of the Doube-Precision Orbit Determination Program (DPODP)*,
           TR 32-1527 Jet Propulsion Laboaratory, Pasadena 1971.

.. [Moy03] T. Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation*,
         John-Wiley & Sons, Inc. Hoboken, Jew Jersey, 2003.

