:author: StuartMufmord
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
Function, Scope and Organisation of

* Map
* Spectra
* LightCurve

Downloaders and Data Retrevial
------------------------------

Most of solar data observed from space mission follows an open policy[#]_ which makes it available to everyone as soon the data is downloaded.
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
	


.. [#] All use of data comming from NASA mission from the Heliophysics Division followes a explicit `copyright and Rules of the Road <http://sdo.gsfc.nasa.gov/data/rules.php>`_.
