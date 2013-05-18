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
having to download and process gigabytes of data; for example NASA's lastest solar spacecraft, the Solar Dynamic Observatory (SDO), downloads over 1TB 
of data per day. This influx of data comes in many forms; the most common solar data type is that of images. 
However, there are many other types of regularly collected data types, such as 1D lightcurves or spectra of 
the whole Sun or parts of it. It is clear therefore that solar physicists require access to the tools to 
process and analyse large quantities of data at different times, wavelengths and spatial scales.

Currently most solar physics is currently performed with a library of IDL routines called solarsoft [SSW]_,
while this library of software is freely avalible the IDL program itself is not. One of SunPy's key aims
is to provide a free and open source alternative to the solarsoft library.

The scope of a solar physics library can be divided up into two main parts, data analysis and data processing.
Data analysis is the scientific analysis of calibrated aligned data whereas data processing is the process 
of calibrating and aligning the data. SunPy's current scope is data analysis with minimal data processing.

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

Most of solar data observed from space missions follow an open policy[#]_ which makes it available to everyone as soon the data is downloaded.
However, they are normally archived by the institution in charge of the instrument that made the observations.  
Such sparsity of archives makes the browsing and data retrieval a very tedious task for the scientist.  
The `Virtual Solar Observatory <http://virtualsolar.org>`_ [VSO]_ has considerably simplified such work by building a centralized database with access to multiple archives.  
VSO allows the user to query by few parameters as instrument name or type, time, physical obsevable and/or spectral range.   
VSO's main interface is web based, however, it can also be accessed by their WSDL webservice.
SunPy uses such API to query and download data from VSO.

-- this is made -- Florian/Joe should write about this...

A new problem arise with the SDO mission.  The large size of the images (4 times larger than the previous mission), 
together with the fastest cadence of their cameras (~10 images per minute) makes challenging to use of the data as it used to be.
The `Heliophysics Event Knowledgebase <http://www.lmsal.com/hek/>`_ [HEK]_ was created to solve this overload of data.  
The principle behind the HEK is to run a number of automated detection algorithms on the pipeline of the data that is downloaded
from SDO in order to fill a database with information about the features and event observed in each image.  
Thus, allowing the solar physicist to search for an event type or property and download just the portion and slices of the images
needed for its further analysis.  In SunPy the implementation just covers the search and retrieve of the information related with 
the events and not the downloading of the observational data.  This allows, for example, to plot the feature contours on an image,
study their properties and their evolution, etc.
The implementation in SunPy of this tool was done based on the VSO tool but changing observatory and instruments by features and
their properties

-- Jack and Florian are the best to describe how this was done.

Solar physicist are also interested in the understanding of how solar events disturb the solar system.  
Very high energy radiation produced during solar flares has effects on our ionosphere almost instantaneously, 
high-energy particles arriving few minutes later can permantly damage spacecraft, similarly
big blob of plasma travelling at high velocities (~1000 km/s) produced as an effect of a coronal mass ejection
can have multiple of effects on our technological dependent society.  
This effects can be meassured everywhere in the solar system, and the `HELiophysics Integrated Observatory <http://helio-vo.eu/>`_ [HELIO]_ has built a set of tools that helps to find where these events have been measured having into account the speed of the different events and the movement of planets and spacecraft within that timerange.
HELIO includes Features and Event catalogues similar to what is offered by HEK, it also offers access to solar observations - as VSO - enhanced with access meassurements of the environment at other planetes and a propagation model to link any event with its origin or its effects.  
Each of these tools counts with their independent webservice, therefore it could be easily implemented as a set of independent tools. 
However, SunPy offers the opportunity to create a better implementation where the data retrieved could interact with the rest of SunPy's ecosystem.
HELIO implementation on SunPy is on early development stages.

Community
---------

* Solar physics and open source
* Scientific Python

SunPy has benefitiated mainly from Summer of Code projects.  During its two first years (2011, 2012), Sunpy has participated on the `ESA Summer of code in space <http://sophia.estec.esa.int/socis2012/>`_ (SOCIS).  This programme is inspired by `Google summer of code <https://developers.google.com/open-source/soc/>`_ (GSOC) and it is aimed to raise the awareness of open source projects related to space, promote the `European Space Agency <http://www.esa.int/>`_ and to improve the excisting space-related open-source software.  VSO implementation, and the first graphical user interface (GUI) were developed during these two summer programmes.

In 2013 SunPy is also taking part on GSOC under the umbrella of the `Python Software Fundation <http://www.python.org/psf/>`_ (PSF), looking forward to the advances this will bring to the capabilities and spread of the project.

SunPy has also been benefitiated on investements made by solar physics group, as it was the case on 2012 when the `Astrophysics Research Group <http://physics.tcd.ie/Astrophysics/>`_ at `Trinity College Dublin <http://www.tcd.ie>`_ contracted for the summer our first year SOCIS student to work on the addition of `CALLISTO solar radio spectrometer <http://www.e-callisto.org/>`_ to SunPy. CALLISTO is a very economic radio spectrometer that has been set on more than 30 different locations worldwide.


Future
------

* Goals and Scope
* Local Database?
* AstroPy
* Publicity

References
----------
.. [SSW] S. L. Freeland and B. N. Handy. *Data Analysis with the Solarsoft System*,
         Solar Physics, 182:497-500, October 1998. DOI: 10.1023/A:1005038224881
.. [VSO] F. Hill, et al. *The Virtual Solar Observatory—A Resource for International Heliophysics Research*,
         Earth Moon and Planets, 104:315-330, April 2009. DOI: 10.1007/s11038-008-9274-7
.. [HEK] N. Hurlburt, et al. *Heliophysics Event Knowledgebase for the Solar Dynamics Observatory (SDO) and Beyond*,
         Solar Physics, 275:67-78, January 2012. DOI: 10.1007/s11207-010-9624-2 arXiv:1008.1291
.. [HELIO] D. Pérez-Suárez et al. *Studying Sun–Planet Connections Using the Heliophysics Integrated Observatory (HELIO)*
           Solar Physics, 280:603-621, October 2012. DOI: 10.1007/s11207-012-0110-x
	


.. [#] All use of data comming from NASA mission from the Heliophysics Division followes a explicit `copyright and Rules of the Road <http://sdo.gsfc.nasa.gov/data/rules.php>`_.
