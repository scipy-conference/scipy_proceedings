:author: Stuart Mumford
:email: stuart@mumford.me.uk
:institution: The University of Sheffield

:author: David Pérez-Suárez
:email: dps.helio@gmail.com
:institution: Finnnish Meteorological Institute

:author: Steven Christe
:email: steven.d.christe@nasa.gov
:institution: NASA Goddard Space Flight Center

:author: Florian Mayer
:email: florian.mayer@bitsrc.org
:institution: Vienna University of Technology

----------------------------------
SunPy: Python for Solar Physicists
----------------------------------

.. class:: abstract

SunPy is a data analysis toolkit specializing in providing the software necessary to analyze solar and heliospheric datasets in Python. 
The goal of SunPy is to provide a free and open-source alternative to the standard IDL-based SolarSoft (SSW) solar data analysis environment. 
We present on the latest release of SunPy (0.3). 
Though still in active development SunPy already provides important functionality such as integration with the the Virtual Solar Observatory (VSO) which provides access to most solar data sets as well as integration with the Heliophysics Event Knowledgebase (HEK), a database of transient solar events such as solar flares or coronal mass ejections. 
One of the major goals of SunPy is to provide a user-friendly, common programming and data analysis environmnent. 
In order to achieve this goal SunPy provides data objects for most data types such as images, lightcurves, and spectra. 
Using PyFits, SunPy can open image fits files from major solar missions (SDO/AIA, SOHO/EIT, SOHO/LASCO, STEREO) into WCS-aware maps. 
Using pandas, SunPy provides advanced time-series tools for data from mission such as GOES, SDO/EVE, and Proba2/LYRA as well as support for radio spectra (e.g. e-Callisto). 
Future releases will build upon current work in the AstroPy library to bring greater functionality to SunPy users.

.. class:: keywords

   Python, Solar Physics, Scientific Python

Introduction
------------

Modern solar physics, similar to astrophysics, requires increasingly complex software tools both for the retrieval as well as the analysis of data. 
The Sun is the most well-observed star. 
As such, solar physics is unique in its ability to access large amounts of high resolution ground- and space-based observations of the Sun at many different wavelengths and spatial scales with high time candence. 
For example, NASA's SDO satellite records over 1TB of data per day all of which is telemtered to the ground and available for analysis. 
This results in scientists having to process large volumes of complex data products. 
In order to make meaningful advances in solar physics, it is important for the software tools to be standardized, easy to use, and transparent, so that the community can build upon a common foundation.

Currently, most solar physics analysis is performed with a library of routines called SolarSoft [SSW]_. 
SolarSoft is a set of integrated software libraries, data bases, and system utilities which provide a common programming and data analysis environment for Solar Physics. 
It is primarily an IDL-based system, although some instrument teams integrate executables written in other languages. 
While this library is open-source and freely available, IDL is not. 
In addition, contributing to the library is not open to the public. 
One of SunPy's key aims is to provide a free and open source alternative to the solarsoft library.

The scope of a solar physics library can be divided up into two main parts, data analysis and data processing.
Data analysis is the scientific analysis of calibrated aligned data where as data processing is the process of calibrating and aliging the data. 
SunPy's current scope is data analysis with minimal data processing.

SunPy currently depends upon the core scientific packages like NumPy, SciPy and matplotlib. 
As well as Pandas, suds, PYFITS / Astropy.io.fits and beautifulsoup4.
The latest release of SunPy is avalible in PyPI and can be installed thus:: 
    
    pip install sunpy


SunPy Data Types
----------------

SunPy's core is based around defining interoperable data types that cover the wide range of observational data available. 
These cover multi-dimesional data and provide basic manipulation and visualisation routines while having a consistent API. 
There are three core data types: Lightcurve, Map and Spectrum / Spectrogram.

Lightcurve is a 1D data type for analysis of lightcurves or more generally flux over time data products.
Map is a core 2D image data type with 3D extensions for CompositeMap, different 2D images at different wavelengths, and MapCube, for time series of 2D Maps. 
Spectrum is a 1D flux agaisnt wavelength data type while Spectrogram is a 2D flux with wavelength and time type.

While these different data types have clear applications to different types of observations, there is also clear inter-links between them, for example a 1 pixel slice of a MapCube should result in a Lightcurve and a 1 pixel slice of a composite map should be a Spectrum. 
While these types of interoperability are not yet implemented in SunPy it is a future goal.

To this end, the 0.3 release of SunPy include a large scale refactoring of all the data types. 
This is primarily motivated by the desire to move away from a structure where we inherit a numpy ndarray to the data type containing the data as an attribute. 
This design is also mirrored by AstroPy's NDData object which has many similarities to function of our data types especially the Map.

Map
===

The Map object is designed for interpreting and processing the most common form of solar data, that of a two-dimensional image most often taken by a CCD camera. 
Map data consists of a data array combined with meta-data. 
Most often these data are provided in the form of FITS files but other file types also exists such as JPG2000. 
SunPy makes use of the AstroPy's PyFITS library to read in FITS files. 
The metadata in most solar FITS files conform to a historic standard to describe the image such as observation time, wavelength of the observation, exposure time, etc. 
In addition, standard header tags are used to provide the information necessary to transform the pixel coordinates to physical coordinates such as sky coordinates. 
Newer missions such as STEREO or AIA on SDO make use of a more precise standard defined by Thompson [WCS]_. 
Thompson also defined standard coordinate transformations to convert from observer-based coordinates to coordinates on the Sun. 
Since the Sun is a gaseous body with no fixed points of reference and different parts of the Sun rotate at different rates, this is a particularly tricky problem. 
SunPy maps through its WCS (World Coordinate System) library has implemented most of these coordinates systems and provides the functions to transform between them. 
SunPy maps also provides other conveinience functions such as plotting using matplotlib.

2D image types are all derived from a GenericMap class that provides 2D specific calibration and coordinate methods. 
This super class is designed to be subclassed by subclasses specific to instruments or detectors. 
To instantiate the correct subclass of GenericMap a MapFactory was developed which is accecible to the user through a class named Map.

The 2D image data processed by Map comes from a variety of instruments with different header parameters and processing reqirements. 
This is catered for in Map by defining "sources" for each instrument, which subclass GenericMap, the base 2D class. 
These sources register with a MapFactory which then automatically determines the instrument of the data being read and returns the correct source subclass.

The Map types provide methods for calibration to physical units as well as image manipulation routines such as rescale, rotate and visualisation routines. 
Calibration routines for different instruments are generally placed inside SunPy's "instr" module and take Maps as arguments.

The SunPy map object recognizes different types of map types and is based on a common super class called MapBase. 
This object will likely inherit from AstroPy's NDData object in the next release of SunPy. 
MapBase provides very limited functionality while 2D image types are all derived from a GenericMap class that provides mission-specific 2D specific calibration and coordinate methods. 
To instantiate the correct subclass of GenericMap a MapFactory is used which is accesible to the user through a class named Map. 
The 2D image data processed by Map comes from a variety of instruments with different header parameters and processing reqirements. 
The map factory defines "sources" for each instrument, which subclasses GenericMap, the base 2D class. 
These sources register with a MapFactory which then automatically determines the instrument of the data being read and returns the correct source subclass. 
Other derived classes are GenericMap, meant to contain a single map. 
Other map types are CompositeMap and MapCube. 
These map classes are meant to hold many maps of a similar spatial region and a time series of maps respectively. 

LightCurve
==========

Spectra
=======

SunPy offers a Spectrogram object, with currently a specialization for e-Callisto spectrograms. It allows the user to seamlessly join different observations,
download data through an interface that only requires to specify location and time-range, linearizes the frequency axis and automatically downsamples large
observations to allow them to be rendered on a normal computer screen and much more to help analyze spectrograms.

The data can currently be read from Callisto FITS files (using PyFITS), but the system is designed in way that makes it easy to include new data-sources
with potentially other data formats (such as LOFAR).

.. Function, Scope and Organisation of

.. * Map
.. * Spectra
.. * LightCurve
	
Solar Data Retrieval and Access
-------------------------------

Most solar observations provided by NASA or ESA follow an open data policy [#] which means that all data is available to everyone as soon the data is telemetered to the ground. 
However, these data are normally archived by the institution in charge of the instrument that made the observations. 
This fact makes browsing data and data retrieval a difficult and tedious task for the scientist. 
In recognition of this fact, the `Virtual Solar Observatory <http://virtualsolar.org>`_ (VSO) [VSO]_ was developed. 
The VSO strives to provides a one stop shop to solar data by building a centralized database with access to multiple archives. 
The VSO allows the user to search using parameters as instrument name or type, time, physical obsevable and/or spectral range.  
VSO's main interface is web-based, however, an API based on a WSDL webservice is also available. SunPy provides a python front-end to this API. 

SunPy has includes the capability to get data from VSO by the use of that webservice.
It includes both a legacy interface that tries to mimic the origin vso_search SSW function and a new interface that allows the user to specify boolean condition
that the data needs to match, such as (instrument aia or instrument eit) and (time between 2013-01-01 and 2013-01-12), leaving the tool to resolve these conditions
to queries to the VSO and getting their results.

A new problem arise with the SDO mission. 
The large size of the images (4 times larger than the previous mission), together with the fastest cadence of their cameras (~10 images per minute) makes challenging to use of the data as it used to be. 
The `Heliophysics Event Knowledgebase <http://www.lmsal.com/hek/>`_ [HEK]_ was created to solve this overload of data. 
The principle behind the HEK is to run a number of automated detection algorithms on the pipeline of the data that is downloaded from SDO in order to fill a database with information about the features and event observed in each image. 
Thus, allowing the solar physicist to search for an event type or property and download just the portion and slices of the images needed for its further analysis. 
In SunPy the implementation just covers the search and retrieve of the information related with the events and not the downloading of the observational data. 
This allows, for example, to plot the feature contours on an image, study their properties and their evolution, etc.
The implementation in SunPy of this tool was done based on the VSO tool but changing observatory and instruments by features and their properties

Is uses the same approach as the VSO, allowing the user to specify boolean conditions the events need to match.

-- Jack and Florian are the best to describe how this was done.

Solar physicist are also interested in the understanding of how solar events disturb the solar system. 
Very high energy radiation produced during solar flares has effects on our ionosphere almost instantaneously, high-energy particles arriving few minutes later can permantly damage spacecraft, similarly big blob of plasma travelling at high velocities (~1000 km/s) produced as an effect of a coronal mass ejection can have multiple of effects on our technological dependent society. 
This effects can be meassured everywhere in the solar system, and the `HELiophysics Integrated Observatory <http://helio-vo.eu/>`_ [HELIO]_ has built a set of tools that helps to find where these events have been measured having into account the speed of the different events and the movement of planets and spacecraft within that timerange. 
HELIO includes Features and Event catalogues similar to what is offered by HEK, it also offers access to solar observations - as VSO - enhanced with access meassurements of the environment at other planetes and a propagation model to link any event with its origin or its effects. 
Each of these tools counts with their independent webservice, therefore it could be easily implemented as a set of independent tools. 
However, SunPy offers the opportunity to create a better implementation where the data retrieved could interact with the rest of SunPy's ecosystem. 
HELIO implementation on SunPy is on early development stages. 

Community
---------

* Solar physics and open source
* Scientific Python

SunPy has benefitiated mainly from Summer of Code projects. 
During its two first years (2011, 2012), Sunpy has participated on the `ESA Summer of code in space <http://sophia.estec.esa.int/socis2012/>`_ (SOCIS). 
This programme is inspired by `Google summer of code <https://developers.google.com/open-source/soc/>`_ (GSOC) and it is aimed to raise the awareness of open source projects related to space, promote the `European Space Agency <http://www.esa.int/>`_ and to improve the excisting space-related open-source software.   VSO implementation, and the first graphical user interface (GUI) were developed during these two summer programmes. 

In 2013 SunPy is also taking part on GSOC under the umbrella of the `Python Software Fundation <http://www.python.org/psf/>`_ (PSF), looking forward to the advances this will bring to the capabilities and spread of the project. 

SunPy has also been benefitiated on investements made by solar physics group, as it was the case on 2012 when the `Astrophysics Research Group <http://physics.tcd.ie/Astrophysics/>`_ at `Trinity College Dublin <http://www.tcd.ie>`_ contracted for the summer our first year SOCIS student to work on the addition of `CALLISTO solar radio spectrometer <http://www.e-callisto.org/>`_ to SunPy. 
CALLISTO is a very economic radio spectrometer that has been set on more than 30 different locations worldwide.


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
.. [HEK] N. Hurlburt, et al. *Heliophysics Event Knowledgebase for the Solar Dynamics Observatory (SDO) and Beyond*,
         Solar Physics, 275:67-78, January 2012. DOI: 10.1007/s11207-010-9624-2 arXiv:1008.1291
.. [HELIO] D. Pérez-Suárez et al. *Studying Sun–Planet Connections Using the Heliophysics Integrated Observatory (HELIO)*
           Solar Physics, 280:603-621, October 2012. DOI: 10.1007/s11207-012-0110-x
	

.. [WCS] W. T. Thompson, *Coordinate systems for solar image data*, A&A 449, 791–803 (2006)

.. [SSW] S. L. Freeland, B. N. Handy, *Data Analysis with the SolarSoft System*, Solar Physics, v. 182, Issue 2, p. 497-500 (1998)

.. [#] All use of data comming from NASA mission from the Heliophysics Division followes a explicit `copyright and Rules of the Road <http://sdo.gsfc.nasa.gov/data/rules.php>`_.
