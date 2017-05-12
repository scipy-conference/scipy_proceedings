:author: Will T. Barnes
:email: will.t.barnes@rice.edu
:institution: Department of Physics and Astronomy, Rice University, Houston, TX, USA
:corresponding:

:author: Kenneth P. Dere
:email: kdere@gmu.edu
:institution: College of Science, George Mason University, Fairfax, VA, USA

:bibliography: references

:video: http://youtube.com/the-video-of-my-talk.html

----------------------------------------------------------
ChiantiPy: a Python package for Astrophysical Spectroscopy
----------------------------------------------------------

.. class:: abstract

   ChiantiPy is an interface to the CHIANTI atomic database for astrophysical spectroscopy. The highly-cited CHIANTI project, now in its 20th year, is an invaluable resource to the solar physics community. The ChiantiPy project brings the power of the scientific Python stack to the CHIANTI database, allowing solar physicists and astronomers to easily make use of this atomic data and calculate commonly used quantities from it such as radiative loss rates and emissivities for particular atomic transitions. In this talk, we will briefly discuss the history of the CHIANTI database and the ChiantiPy project as well as the current state of the project and its place in the solar physics community. We will demonstrate some of the capabilities of the ChiantiPy code and show examples of how ChiantiPy can be used in both modeling and observational studies. Finally, we'll discuss how ChiantiPy helps to bring the power of the CHIANTIC atomic database to the growing set of astronomy-related tools in Python.

.. class:: keywords

   solar physics, atomic physics, astrophysics, spectroscopy

Introduction
------------
Nearly all astrophysical observations are done through *remote sensing*. Light at various wavelengths is collected by instruments, either ground- or space-based, in an attempt to understand physical processes happening in distant astrophysical objects. However, in order to translate these detector measurements to meaningful physical insight, we need to understand what physical conditions give rise to different spectral lines and continuum emission. Started in 1996 by researchers at the Naval Research Laboratory, the University of Cambridge, and Arcetri Astrophysical Observatory in Florence for the purpose of analyzing solar spectra, the `CHIANTI atomic database <http://www.chiantidatabase.org/>`_ provides a set of up-to-date atomic data for ions of hydrogen through zinc as well as a suite of tools, written in the proprietary Interactive Data Language (IDL), for analyzing this data. Described in `a series of 14 papers from 1997 to 2015 <http://www.chiantidatabase.org/chianti_papers.html>`_, including :cite:`young_chianti_2016` and :cite:`dere_chianti_1997`, that have been `cited collectively over 3000 times <http://www.chiantidatabase.org/chianti_ADS.html>`_, the CHIANTI database is an invaluable resource to the solar physics community. 

The ChiantiPy project, started in 2009, provides a Python interface to the CHIANTI database and an alternative to the IDL tools. ChiantiPy is not a direct translation of its IDL counterpart, but instead provides an intuitive object oriented interface to the database (compared to the more functional approach in IDL). Though it predates many of the software tools used in open-source scientific computing today, ChiantiPy has embraced modern development practices and is hosted on `GitHub <https://github.com/chianti-atomic/ChiantiPy>`_, maintains up-to-date `documentation on Read the Docs <http://chiantipy.readthedocs.io/en/latest/>`_, and `runs several tests on Travis CI <https://travis-ci.org/chianti-atomic/ChiantiPy>`_ at each check-in.

Database
--------
Details about how the CHIANTI database is laid out, how it is read into ChiantiPy, what sort of information is contained in the database, its size, how the data is obtained, updated, and maintained

Common Calculations and API
---------------------------
Details about what sort of calculations are included in ChiantiPy as well as how these are implemented in the code itself.

Line Emission
#############

Continuum Emission
##################

Radiative Losses
########################

Documentation, Testing, and Infrastructure
------------------------------------------
Details about documentation hosting and testing as well as the status of tests, documentation quality, code coverage, etc.

Also some details about our developer community, how bugs are reported, how pull requests are merged, governance, interaction with other packages (e.g. SunPy, Astropy, OpenAstronomy)

Future Work: Towards ChiantiPy v1.0
-----------------------------------
Goals, new features, fixes, refactoring, big projects, etc

References
----------



