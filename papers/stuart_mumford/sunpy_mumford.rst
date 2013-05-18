:author: Stuart Mumford
:email: stuart@mumford.me.uk
:institution: The University of Sheffield

:author: David Pérez-Suárez
:email: dps.helio@gmail.com
:institution: Finnnish Meteorological Institute

----------------------------------
SunPy: Python for Solar Physicists
----------------------------------

.. class:: abstract

	SunPy aims to become a comprehensive package for solar data analysis and 
	processing.

.. class:: keywords

   Python, Solar Physics, Scientific Python

Introduction
------------

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

SunPy Data Types
----------------

SunPy's core is based around defining interoperable data types that cover the wide range of observational data 
avalible. These cover muti-dimesional data and provide basic manipulation and visualisation routines while having 
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

The Map type is designed for processing the most common type of solar data, that of the 2D image. As of SunPy 0.3 all of the Map types, 
GenericMap, CompositeMap and MapCube have a common super class MapBase. This is designed with compatibility to AstroPy's NDData object 
in mind. Map base itself provides very limited functionality as it is catering to 2D or 3D data with different coordinates for each axis.

2D image types are all derived from a GenericMap class that provides 2D specific calibration and coordinate methods. This super class 
is designed to be subclassed by subclasses specific to instruments or detectors. To instantiate the correct subclass of GenericMap a 
MapFactory was developed which is accecible to the user through a class named Map.

The 2D image data processed by Map comes from a variety of instruments with different header parameters and processing reqirements. 
This is catered for in Map by defining "sources" for each instrument, which subclass GenericMap, the base 2D class. These sources 
register with a MapFactory which then automatically determines the instrument of the data being read and returns the correct source 
subclass.

The Map types provide methods for calibration to physical units as well as image manipulation routines such as rescale, rotate and 
visualisation routines. Calibration routines for different instruments are generally placed inside SunPy's "instr" module and take 
Maps as arguments.

LightCurve
==========

Spectra
=======


.. Function, Scope and Organisation of

.. * Map
.. * Spectra
.. * LightCurve

Downloaders and Data Retrevial
------------------------------

Most of solar data observed from space mission follows an open policy [#]_ which makes it available to everyone as soon the data is downloaded.
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

One of SunPy's major advantages over it's predecessors is the development of SunPy as an open source community inside the wide 
and diverse general scientific python community. While the SolarSoft library is "open source" in terms of the code being freely 
avalible most of the development takes place behind closed doors and there is no clear process for contribution. As well as 
transitioning the solar physics communtity to Python SunPy also aims to instill the prinicpals of open source development in 
the community.

The scientifc python community is much more establised in other diciplines than solar physics. SunPy is making use of existing 
scientific python projects with deeper integration with projects like AstroPy and scikit-image possible in the future. This
collaboration is another strength that sets the scientific python community apart from other similar solutions.

SunPy has been in existance since **[enter date here]** and in this time has already developed into a vaiable analysis platform 
for solar data. This has been achieved with very little in the way of external funding apart from two ESA Summer Of Code In Space 
funded summer students and two Google Summer Of Code students in 2013. These students have made a massive contribution to the 
SunPy code base and enabled development at a much higer rate than would have been possible otherwise.

Future
------

SunPy 0.3 provides a excellent, flexable base for future development expansion of the project. The immediate goal for SunPy is 
to develop the project into a flexible package for data analysis and scientific application. While in the long term SunPy aims 
to become the defacto package for all solar physics data processing and analysis. 
To achive this goal it is required that SunPy gains more traction within the solar physics community. This is both to increase 
the user base and to attract new missions and instruments to adopt Python/SunPy for their data processing pipeline.

**??????**

References
----------
.. [VSO] F. Hill, et al. *The Virtual Solar Observatory—A Resource for International Heliophysics Research*,
         Earth Moon and Planets, 104:315-330, April 2009. DOI: 10.1007/s11038-008-9274-7

.. [#] All use of data comming from NASA mission from the Heliophysics Division followes a explicit `copyright and Rules of the Road <http://sdo.gsfc.nasa.gov/data/rules.php>`_.
