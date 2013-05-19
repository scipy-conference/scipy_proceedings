:author: Stuart Mumford
:email: stuart@mumford.me.uk
:institution: The University of Sheffield

:author: David Pérez-Suárez
:email: dps.helio@gmail.com
:institution: Finnnish Meteorological Institute

:author: Steven Christe
:email: steven.d.christe@nasa.gov
:institution: NASA Goddard Space Flight Center

----------------------------------
SunPy: Python for Solar Physicists
----------------------------------

.. class:: abstract

	SunPy is a data analysis toolkit specializing in providing the software necessary to analyze solar and heliospheric datasets in Python. The goal of SunPy is to provide a free and open-source alternative to the standard IDL-based SolarSoft (SSW) solar data analysis environment. We present on the latest release of SunPy (0.3). Though still in active 
development SunPy already provides important functionality such as integration with the
the Virtual Solar Observatory (VSO) which provides access to most solar data sets as well as integration with the Heliophysics Event Knowledgebase (HEK), a database of transient
solar events such as solar flares or coronal mass ejections. One of the major goals of SunPy is to provide a user-friendly, common programming and data analysis environmnent. In order to achieve this goal SunPy provides data objects for most data types such as images, lightcurves, and spectra. Using PyFits, SunPy can open image fits files from major solar missions (SDO/AIA, SOHO/EIT, SOHO/LASCO, STEREO) into WCS-aware maps. Using pandas, SunPy provides advanced time-series tools for data from mission such as GOES, SDO/EVE, and Proba2/LYRA as well as support for radio spectra (e.g. e-Callisto). Future releases will
build upon current work in the AstroPy library to bring greater functionality to SunPy users.

.. class:: keywords

   Python, Solar Physics, Scientific Python

Introduction
------------

Modern solar physics, similar to astrophysics, requires increasingly complex tools and 

Modern solar physics uses a large amount of high resolution ground- and space-based telescopes
to observe the Sun at different wavelengths and spatial scales. This data results in solar physicists 
having to download and process gigabytes of data; for example NASA's SDO satellite downloads over 1TB 
of data per day. This influx of data comes in many forms; the most common solar data type is that of images. 
However, there are many other types of regularly collected data types, such as 1D lightcurves or spectra of 
the whole Sun or parts of it. It is clear therefore that solar physicists require access to the tools to 
process and analyse large quantities of data at different times, wavelengths and spatial scales.

Currently most solar physics is currently performed with a library of IDL routines called solarsoft,
while this library of software is freely avalible the IDL program itself is not. One of SunPy's key aims
is to provide a free and open source alternative to the solarsoft library.

The scope of a solar physics library can be divided up into two main parts, data analysis and data processing.
Data analysis is the scientific analysis of calibrated aligned data where as data processing is the process 
of calibrating and aliging the data. SunPy's current scope is data analysis with minimal data processing.

.. * Solar Data
.. * SunPy Data types
.. * IDL / SSW
.. * Data processing / analysis

Need to add a list of dependencies.

SunPy Data Types
----------------

SunPy's core is based around defining interoperable data types that cover the wide range of observational data 
available. These cover multi-dimesional data and provide basic manipulation and visualisation routines while having 
a consistent API. There are three core data types: Lightcurve, Map and Spectrum / Spectrogram.

Lightcurve is a 1D data type for analysis of lightcurves or more generally flux over time data products.
Map is a core 2D image data type with 3D extensions for CompositeMap, different 2D images at different wavelengths, and 
MapCube, for time series of 2D Maps. Spectrum is a 1D flux agaisnt wavelength data type while Spectrogram is a 2D flux 
with wavelength and time type.

While these different data types have clear applications to different types of observations, there is also clear inter-links 
between them, for example a 1 pixel slice of a MapCube should result in a Lightcurve and a 1 pixel slice of a composite map 
should be a Spectrum. While these types of interoperability are not yet implemented in SunPy it is a future goal.

To this end, the 0.3 release of SunPy will include a large scale refactoring of all the data types. This is primarily motivated 
by the desire to move away from a structure where we inherit a numpy ndarray to the data type containing the data as an attribute. 
This design is also mirrored by AstroPy's NDData object which has many similarities to function of our data types especially the Map.

Map
===

The Map object is designed for interpreting and processing the most common form of solar data, that of a two-dimensional image most often taken by a CCD camera. Map data consists
of a data array combined with meta-data. Most often these data are provided in the form
of FITS files but other file types also exists such as JPG2000. SunPy makes use of the AstroPy's PyFITS library to read in FITS files. The metadata in most solar FITS files
conform to a historic standard to describe the image such as observation time, wavelength of the observation, exposure time, etc. In addition, standard header tags are used to provide the information necessary to transform the pixel coordinates to physical coordinates such as sky coordinates. Newer missions such as STEREO or AIA on SDO make use of a more precise standard defined by Thompson [WCS]_. Thompson also defined standard coordinate transformations to convert from observer-based coordinates to coordinates on the Sun. Since the Sun is a gaseous body with no fixed points of reference and different parts of the Sun rotate at different rates, this is a particularly tricky problem. SunPy maps through its WCS (World Coordinate System) library has implemented most of these coordinates systems and provides the functions to transform between them. SunPy maps also provides other conveinience functions such as plotting using matplotlib.

The SunPy map object recognizes different types of map types and is based on a common super class called MapBase. This object will likely inherit from AstroPy's NDData object in the next release of SunPy. MapBase provides very limited functionality while 2D image types are all derived from a GenericMap class that provides mission-specific 2D specific calibration and coordinate methods. To instantiate the correct subclass of GenericMap a 
MapFactory is used which is accesible to the user through a class named Map. The 2D image data processed by Map comes from a variety of instruments with different header parameters and processing reqirements. The map factory defines "sources" for each instrument, which subclasses GenericMap, the base 2D class. These sources register with a MapFactory which then automatically determines the instrument of the data being read and returns the correct source subclass. Other derived classes are GenericMap, meant to contain a single map. Other map types are CompositeMap and MapCube. These map classes are meant to hold many maps of a similar spatial region and a time series of maps respectively. 



.. Function, Scope and Organisation of

.. * Map
.. * Spectra
.. * LightCurve

Downloaders and Data Retrevial
------------------------------

Most of solar data observed from space mission follows an open policy [#] which makes it available to everyone as soon the data is downloaded.
However, they are normally archived by the institution in charge of the instrument that made the observations.  
Making the browsing and data retrieval a very tedious task for the scientist.  
The `Virtual Solar Observatory <http://virtualsolar.org>`_ (VSO) [VSO]_ has considerably simplified such work by building a centralized database with access to multiple archives.  
VSO allows the user to query by few parameters as instrument name or type, time, physical obsevable and/or spectral range.   
VSO's main interface is web based, however, they have developed an API based on a WSDL webservice.
SunPy has includes the capability to get data from VSO by used of that webservice.
-- this is made -- Florian/Joe should write about this...
* VSO
* HEK / Helio

Community
---------

* Solar physics and open source
* Scientific Python
* GSOC / SOCIS

Future
------

* Goals and Scope
* Local Database?
* AstroPy
* Publicity

References
----------
.. [VSO] F. Hill, et al. *The Virtual Solar Observatory—A Resource for International Heliophysics Research*,
         Earth Moon and Planets, 104:315-330, April 2009. DOI: 10.1007/s11038-008-9274-7

.. [WCS] W. T. Thompson, *Coordinate systems for solar image data*, A&A 449, 791–803 (2006)

.. [SSW] S. L. Freeland, B. N. Handy, *Data Analysis with the SolarSoft System*, Solar Physics, v. 182, Issue 2, p. 497-500 (1998)

.. [SSW] Freeland, S. L.; Handy, B. N., *SolarSoft: Programming and data analysis environment for solar physics*, 2012, Astrophysics Source Code Library, record ascl:1208.013

.. [#] All use of data comming from NASA mission from the Heliophysics Division followes a explicit `copyright and Rules of the Road <http://sdo.gsfc.nasa.gov/data/rules.php>`_.
