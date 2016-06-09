:author: Mark Wickert
:email: mwickert@uccs.edu
:institution: University of Colorado Colorado Springs


-----------------------------------------------------------------------------------------------------------------------------
A Simulation Framework for Studying the Code Acquisition and Tracking Functions of a Global Positioning System (GPS) Receiver
-----------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   In this talk a Python-based simulation framework is described that implements the waveform
   level signal processing needed to acquire and track the ranging signal of a global positioning
   system (GPS) satellite. This framework was developed Fall 2015 as an end-of-semester project for
   a digital signal processing course taken by electrical engineers. By design, GPS signals lie on
   top of one another, but are separable by virtue of a unique code and nearly orthogonal code
   assigned to each satellite. The key to position determination is the time difference of arrival
   (TDOA) of each of the satellite signals at the user receiver. A high precision clock maintains
   timing accuracy among the satellites. One of the most important tasks of the user receiver is
   to acquire and track the ranging code of three or more satellites in view at a given time.
   The framework allows the user to first explore a receiver for a single satellite signal. Object
   oriented Python then makes it easy to extend the receiver to processing multiple satellite signals
   in parallel. The source of signals used in the framework is either simulation or a low-cost (~$20)
   software defined radio dongle known as the RTL-SDR. With the RTL-SDR signals are captured from a
   GPS patch antenna, fed to the RTL-SDR, and then via USB captured into Python as a complex ndarray.
   The computer simulation project that utilizes the framework has the students performing a variety
   of simulation tasks, start from a single channel receiver building up to a four channel receiver
   with signal impairments present. As developed Fall 2015 the project utilizing this framework is
   entirely computer simulation based, but the ability to use real signals captured from the RTL-SDR,
   opens additional capability options. Making use of these signals is non-trival, as additional
   signal processing is needed to estimate the Doppler frequency error and if the data bits are to
   be recovered, the L1 signal carrier phase needs to be tracked. These aspects of the framework
   as currently under development (mid Spring 2016) for a communications theory course.

.. class:: keywords

   digital signal processing, software defined radio, communications systems

Introduction
------------

In this paper a Python-based simulation framework is described that implements the waveform level signal
processing needed to acquire and track the ranging signal of a global positioning system (GPS) satellite.
This framework was developed Fall 2015 as an end-of-semester project for a digital signal processing course [#]_
taken by electrical engineers [Opp2010]_. By design, GPS signals lie on top of one another, but are separable by virtue
of a unique code and nearly orthogonal code assigned to each satellite. The key to position determination
is the time difference of arrival (TDOA) of each of the satellite signals at the user receiver. A high
precision clock maintains timing accuracy among the satellites. One of the most important tasks of the
user receiver is to acquire and track the ranging code of three or more satellites in view at a given time.
The framework allows the user to first explore a receiver for a single satellite signal. Object oriented
Python then makes it easy to extend the receiver to processing multiple satellite signals in parallel.
The source of signals used in the framework is either simulation or a low-cost (~$20) software defined
radio dongle known as the RTL-SDR [RTLSDR]_. With the RTL-SDR signals are captured from a GPS patch antenna, fed
to the RTL-SDR, and then via USB captured into Python as a complex ndarray. The computer simulation
project that utilizes the framework has the students performing a variety of simulation tasks, start
from a single channel receiver building up to a four channel receiver with signal impairments present.

.. [#] The course notes, including Python projects, can be found at ``http://www.eas.uccs.edu/wickert/ece5650/``.

GPS was started in 1973 with the first block of satellites launched over the 1978 to 1985 time interval.
At the present time there are 31 GPS satellites in orbit. The satellites orbit at an altitude of about
20,350 km (~12,600 mi). This altitude classifies the satellites as being in a medium earth orbit (MEO),
as opposed to low earth orbit (LEO), or geostationary above the equator (GEO), or high earth orbit (HEO).
The orbit period is 11 hours 58 minutes with six SVs in view at any time from the surface of the earth.
Clock accuracy is key to the operation of GPS and the satellite clocks are very accurate. Four satellites
are needed for a complete :math:`(x, y, z)` position determination since the user clock is an uncertainty that
must be resolved. The maximum SV velocity relative to an earth user is 800m/s (the satellite itself is
traveling at ~7000 mph), thus the induced Doppler is up to 4.2 kHz on the L1 carrier frequency of 1.57542 GHz.
This frequency uncertainty plus any motion of the user itself, creates additional challenges in processing
the received GPS signals.

GPS uses unique ranging codes from each SV to ascertain the distance, :math:`r`, between a particular SV and the user.
With three range measurements and perfect timing, you can arrive at the exact location to within an ambiguity.
The ambiguity is resolved by choosing the solution closest to earth’s surface. A fourth satellite allows
elevation to be determined, and is also used in resolving local clock errors.

For commercial GPS use the coarse acquisition (CA) codes of period 1023 bits or chips are used for ranging.
Each SV is assigned a unique CA code from the family of Gold Codes [Kaplan1996]_, [Grewal1996]_. The bit or
chip rate of the
ranging code is 1.023 Mcps (Mega chips per second). Note that the code period is exactly 1 ms. There are 37
Gold Codes available. The Gold codes are special since they form a family of nearly orthogonal sequences.
Orthogonal here means that when two SV signals are received by a user, correlation-based signal processing,
the two signals do not interfere with each other.

In cellular telephony this is known as code division multiple access (CDMA), as multiple users can share
the same radio frequency spectrum and nominally inflict minimal interference on each other. This is
perfect for GPS too, as the user needs to receive multiple satellite signals (4 or more) in order to
get a position fix. As with any ranging code system, the user needs to properly synchronize a local
replica CA code with the received SV signal of interest. In the project the focus is on local replica
synchronization . Two facets of the synchronization process are [Kaplan1996]_ [Grewal1996]_:

* Coarse alignment of the local code with the received signal, which is known as code acquisition

* Fine code tracking using a feedback control system, since the SV and likely the user are in motion, and the local clock is not perfectly synchronous with the transmit clock

The framework implements serial search for code acquisition and a noncoherent delay-locked loop (DLL) for
fine code phase tracking. The DLL is very much like a phase-locked loop (PLL) found in communications systems.
Not present in this project, but part of the CA code signal, is a 50 bit/s data stream which contains:

* Satellite almanac data

* Satellite ephemeris data

* Signal timing data

* Ionospheric delay data

* A satellite health message


This project is entirely computer simulation based, but the design of the simulation assumes that the receiver
front-end utilizes the RTL-SDR software defined radio dongle [RTLSDR]_. What this means is that real signals capture
from the RTL-SDR may also be used. Making use of these signals is non-trival, as additional signal processing
is needed to estimate the Doppler frequency error and if the data bits are to be recovered, the L1 carrier phase
needs to be tracked. These aspects of the framework as currently under development (mid Spring 2016) for a
communications theory course.

Background Theory
=================

Waveforms Fig. :ref:`fig1`.

.. figure:: scipy_2016_fig1.pdf
   :scale: 55%
   :align: center
   :figclass: htb

   User time delay measurement using cross correlation with the local replica code. :label:`fig1`


Some text following this figure and caption.


From System Block Diagram to Python Class Design
================================================

To be written using the project reader document.


Results
=======


Simulation Examples
-------------------

To be written using the project reader document and project solutions contained in a Jupyter notebook.


Student Feedback
----------------

Student feedback was very positive. Having the framework contained in a Jupyter notebook [Jupyter]_ made the project very
approachable for all students, in spite of only having a brief introduction to the inner workings of GPS code
synchronization.

More ...


Conclusions and Future Work
---------------------------




Acknowledgments
---------------

TBD


References
----------
.. [Opp2010] Alan V. Oppenheim and Ronald W. Schafer, *Discrete-Time Signal Processing* (3rd ed.), Prentice Hall, 2010.
.. [RTLSDR] http://sdr.osmocom.org/trac/wiki/rtl-sdr.
.. [Kaplan1996] Elliot Kaplan, editor, *Understanding GPS Principles and Applications*, Artech, Boston, 1996.
.. [Grewal1996] M. Grewal, L. Weill, and A. Andrews, *Global Positioning Systems, Inertial Navigation, and Integration*, Wiley, New York, 2001.
.. [Wic2013] M.A. Wickert. *Signals and Systems for Dummies*,
           Wiley, 2013.
.. [Jupyter] http://jupyter.org.

