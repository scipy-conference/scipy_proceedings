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

   ChiantiPy is an interface to the CHIANTI atomic database for astrophysical spectroscopy. The highly-cited CHIANTI project, now in its 20th year, is an invaluable resource to the solar physics community. The ChiantiPy project brings the power of the scientific Python stack to the CHIANTI database, allowing solar physicists and astronomers to easily make use of this atomic data and calculate commonly used quantities from it such as radiative loss rates and emissivities for particular atomic transitions. This paper will discuss the capabilities of the CHIANTI database and the ChiantiPy project as well as the current state of the project and its place in the solar physics community. We will demonstrate how the core modules in ChiantiPy can be used to study emission from optically thin transitions and the continuum in the x ray and EUV wavelengths. Additionally, we will discuss some of the infrastructure around the ChiantiPy project and some of the goals for the near future.

.. class:: keywords

   solar physics, atomic physics, astrophysics, spectroscopy

Introduction
------------
Nearly all astrophysical observations are done through *remote sensing*. Light at various wavelengths is collected by instruments, either ground- or space-based, in an attempt to understand physical processes happening in distant astrophysical objects. However, in order to translate these detector measurements to meaningful physical insight, it is necessary to understand what physical conditions give rise to different spectral lines and continuum emission. Started in 1996 by researchers at the Naval Research Laboratory, the University of Cambridge, and Arcetri Astrophysical Observatory in Florence for the purpose of analyzing solar spectra, the CHIANTI atomic database provides a set of up-to-date atomic data for ions of hydrogen through zinc as well as a suite of tools, written in the proprietary Interactive Data Language (IDL), for analyzing this data. Described in a series of 15 papers from 1997 to 2016 (see Table :ref:`chiantipapers`) that have been cited collectively over 3000 times, the CHIANTI database is an invaluable resource to the solar physics community.

.. table:: All publications describing the data and capabilities of the CHIANTI atomic database, the associated version of the database, and the number of citations as reported by the NASA Astrophysics Data System. :label:`chiantipapers`

   +-------------------------------------+---------+-----------+
   |Paper                                | Version | Citations |
   +=====================================+=========+===========+
   | :cite:`dere_chianti_1997`           | 1       | 1167      |
   +-------------------------------------+---------+-----------+
   | :cite:`young_chianti:_1998`         | 1       | 105       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chianti_1999`          | 2       | 94        |
   +-------------------------------------+---------+-----------+
   | :cite:`dere_chianti-atomic_2001`    | 3       | 156       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chianti-atomic_2002`   | 3       | 86        |
   +-------------------------------------+---------+-----------+
   | :cite:`young_chianti-atomic_2003`   | 4       | 250       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chianti-atomic_2006`   | 5       | 373       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chianti-atomic_2006-1` | 5       | 25        |
   +-------------------------------------+---------+-----------+
   | :cite:`dere_chianti_2009`           | 6       | 301       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chiantiatomic_2009`    | 6       | 25        |
   +-------------------------------------+---------+-----------+
   | :cite:`young_chiantiatomic_2009`    | 6       | 22        |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chiantiatomic_2012-1`  | 7       | 174       |
   +-------------------------------------+---------+-----------+
   | :cite:`landi_chiantiatomic_2013`    | 7.1     | 227       |
   +-------------------------------------+---------+-----------+
   | :cite:`del_zanna_chianti_2015`      | 8       | 60        |
   +-------------------------------------+---------+-----------+
   | :cite:`young_chianti_2016`          | 8       | 1         |
   +-------------------------------------+---------+-----------+
   |                                     |         | **Total** |
   +-------------------------------------+---------+-----------+
   |                                     |         | 3066      |
   +-------------------------------------+---------+-----------+

The CHIANTI database provides 

Give history of CHIANTI/ChiantiPy, where the data comes from, who uses it, why.


The ChiantiPy project, started in 2009, provides a Python interface to the CHIANTI database and an alternative to the IDL tools. ChiantiPy is not a direct translation of its IDL counterpart, but instead provides an intuitive object oriented interface to the database (compared to the more functional approach in IDL). Need some more details here.... 


Database
--------
The CHIANTI database is collection of directories and ASCII files that can be downloaded as a tarball from the CHIANTI database website or as part of the SolarSoftware (or SolarSoft) IDL package :cite:`freeland_data_1998`. The solar physics community has typically relied on the latter as SolarSoft has served as the main hub for solar data analysis software for the last several decades.

The structure of the CHIANTI database is such that each top level directory represents an element and each subdirectory is an ion of that element. Files in each of the subdirectories contain pieces of information attached to each ion. The database generally follows the structure :code:`{el}/{el}_{ion}/{el}_{ion}.{filetype}`. A few of these filetypes are summarized in Table :ref:`dbstructure`. For a complete description of all of the different filetypes available, see Table 1 of :cite:`young_chianti_2016` and the `CHIANTI user guide <http://www.chiantidatabase.org/cug.pdf>`_. Fig. :ref:`linelist` shows all of the available ions in the CHIANTI database as well as the number of levels available for each ion.

.. figure:: figures/linelist.pdf
   :align: center
   :figclass: w
   :scale: 55%   

   All ions available in the latest version (v8.0.6) of the CHIANTI atomic database. The color and number in each square indicate the number of available levels in the database. Adapted from Fig. 1 of :cite:`young_chianti_2016`. :label:`linelist` 

.. table:: Some of the filetypes available for each ion in the CHIANTI database. Adapted from Table 1 of :cite:`young_chianti_2016`. :label:`dbstructure`

   +----------+------------------------------------------------------------------------------+
   | Filetype | Description                                                                  |
   +==========+==============================================================================+
   | ELVLC    | Index and energy for each level                                              |
   +----------+------------------------------------------------------------------------------+
   | WGFA     | Wavelength, Einstein "A" values, and oscillator strengths for each transiton |
   +----------+------------------------------------------------------------------------------+
   | SCUPS    | Scaled effective collision strengths for each transition                     |
   +----------+------------------------------------------------------------------------------+
   | FBLVL    | Energy levels for free-bound continuum calculation                           |
   +----------+------------------------------------------------------------------------------+

ChiantiPy provides several low-level functions for reading raw data directly from the CHIANTI database. For example, to find the energy of the emitted photon for each transition for Fe V (i.e. the fifth ionization state of iron), you would first read in level information for each transition for a given ion,

.. code-block:: python

   import ChiantiPy.tools.util as ch_util
   fe5_wgfa = ch_util.wgfaRead('fe_5')
   ilvl1 = np.array(fe5_wgfa['lvl1']) - 1
   ilvl2 = np.array(fe5_wgfa['lvl2']) - 1

and then use the indices of the level to find the associated level energies in the ELVLC data,

.. code-block:: python
    
   fe5_elvlc = ch_util.elvlcRead('fe_5')
   delta_energy = (np.array(fe5_elvlc['ecm'])[ilvl2] 
                   - np.array(fe5_elvlc['ecm'])[ilvl1])

where the associated energy levels are given in :math:`\mathrm{cm}^{-1}`. In general, these functions are only used internally by the core ChiantiPy objects. However, users who need access to the raw data may find them useful.

In addition to each of the files associated with each ion, CHIANTI also provides abundance and ionization equilibrium data for each *element* in the database. The elemental abundance, :math:`N(X)/N(H)` (i.e. the number of atoms of element :math:`X` relative to the number of hydrogen atoms), in the corona and photosphere has been measured by many workers and these various measurements have been collected in the CHIANTI atomic database. For example, to read the abundance of Fe as measured by :cite:`feldman_potential_1992`,

.. code-block:: python
   
   import ChiantiPy.tools.io as ch_io
   import ChiantiPy.tools.util as ch_util
   ab = ch_io.abundanceRead('sun_coronal_1992_feldman')
   fe_ab = abundance['abundance'][ch_util.el2z('Fe')-1]

As with the other CHIANTI data files, the abundance values are typically read internally and then exposed to the user through more abstract objects like the :code:`ion` class so reading them in this way is not necessary. Similarly, the ionization equilibrium of each ion of each element is available as a function of temperature and various sets of ionization equilibria data can be used. More details about the ionization equilibrium can be found in later sections. 

Default values for the abundance and ionization equilibrium files as well as the units for wavelength (cm, :math:`\mathrm{\mathring{A}}`, or eV) and energy (ergs or photons) can be set in the users :code:`chiantirc` file, located in :code:`~/.chianti/chiantirc`. These settings are stored in :code:`ChiantiPy.tools.data.Defaults` and can be changed at anytime. 

Unless otherwise noted, all quantities are expressed in the cgs unit system, with the exception of wavelengths which are recorded in angstroms (:math:`\mathrm{\mathring{A}}`). As discussed above, some energies in the CHIANTI atomic database, particularly those pertaining to levels in an atom, may be stored in :math:`\mathrm{cm}^{-1}` for convenience (i.e. with :math:`h=c=1`, a common convention in atomic physics). Results of any calculation in ChiantiPy will always be returned in cgs (unless explicitly stated in the :code:`chiantirc` file, e.g. photons instead of ergs).

Common Calculations and API
---------------------------
The majority of the ChiantiPy codebase is divided into two modules: :code:`tools` and :code:`core`. The former contains utility and helper functions that are mostly for internal use. The latter contains the primary objects for interacting with the data in the CHIANTI atomic database and performing many common calculations with these data. A summary of the objects in :code:`core` can be found in Table :ref:`chiantipyapi`. These objects can be roughly divided into two categories: those that deal with information and calculations about individual ions and those that aggregate information over a range of ions in order to perform some calculation. The :code:`ion` and :code:`Continuum` objects calculate emissivity information related to specific ions while the :code:`ioneq`, :code:`spectrum`, and :code:`radLoss` require information from multiple ions and/or elements. 

.. table:: The primary objects in the public API of ChiantiPy. :label:`chiantipyapi`

   +-------------------+-----------------------------------------------------------------------+
   | Object Name       | Description                                                           |
   +===================+=======================================================================+
   | :code:`ion`       | Holds ion properties and calculaties level populations and emissivity |
   +-------------------+-----------------------------------------------------------------------+
   | :code:`Continuum` | Free-free and free-bound continuum for individual ions                |
   +-------------------+-----------------------------------------------------------------------+
   | :code:`ioneq`     | Ionization equilibrium for individual elements                        |
   +-------------------+-----------------------------------------------------------------------+
   | :code:`spectrum`  | Calculate synthetic spectra for a range of ions                       |
   +-------------------+-----------------------------------------------------------------------+
   | :code:`radLoss`   | Total radiative losses from multiple ions, including continuum        |
   +-------------------+-----------------------------------------------------------------------+

Line Emission
#############
The most essential and actively developed portion of the ChiantiPy package is the :code:`ion` object which provides an interface to the data and associated calculations for each ion in the database. The :code:`ion` object is initialized with an ion name, a temperature range, and a density [1]_,

.. code-block:: python

   import ChiantiPy.core as ch
   import numpy as np
   T = np.logspace(4,6,100)
   n = 1e9
   fe_5 = ch.ion('fe_5',temperature=T,eDensity=n)

In this example, we've initialized an :code:`ion` object for Fe V over a temperature range  of :math:`T=10^4-10^6` K at a constant electron density of :math:`n_e=10^9` :math:`\mathrm{cm}^{-3}`. All of the data discussed in the previous section are available as attributes of the :code:`ion` object (e.g. :code:`.Elvlc` and :code:`.Wgfa` are dictionaries holding the various fields available in the corresponding filetypes listed in Table :ref:`chiantipyapi`). In general, ChiantiPy objects follow the convention that methods are lowercase and return their value(s) to attributes with corresponding uppercase names [2]_. For example, the abundance value of Fe is stored in :code:`fe_5.Abundance` and the ionization equilibrium is calculated using the method :code:`fe_5.ioneqOne()` with the value being returned to the attribute :code:`fe_5.IoneqOne`.

One of the most often used calculations in CHIANTI and ChiantiPy is the energy level populations as a function of temperature. When calculating the energy level populations in a low density, high temperature optically-thin plasma,  collisional excitation and subsequent decay often occur much more quickly than ionization and recombination, allowing these two processes to be decoupled. Furthermore, it is assumed that all transitions occur between the excited state and the ground state. These two assumptions make up what is commonly known as the *coronal model approximation*. Thus, the level balance equation can be written as,

.. math::

   \sum_{k>j}N_kA_{kj} + n_e\sum_{i=j}N_jC_{ij} - \left(\sum_{i<j}N_jA{ji} + n_e\sum_{k=j}N_jC_{jk}\right) = 0,

where :math:`A_{kj}` is the radiative decay rate, :math:`C_{jk}` is the collisional excitation coefficient, and :math:`N_j` is the number of electrons in excited state :math:`j` :cite:`young_chianti_2016`. Since :math:`A` and :math:`C` are given by the CHIANTI database, this expression can be solved iteratively to find :math:`n_j=N_j/\sum_jN_j`, the fraction of electrons in excited state :math:`j` or the level population fraction.

To method :code:`fe_5.populate()` can then be used to calculate the level populations for Fe V. This method populates the :code:`fe_5.Population` attribute and a :math:`100\times34` array (i.e. number of temperatures by number of energy levels) is stored in :code:`fe_5.Population['population']`. ChiantiPy also provides the convenience method :code:`fe_5.popPlot()` which provides a quick visualization of level population as a function of temperature for several of the most populated levels. Note that this calculation can be quite expensive for large temperature/density arrays and for ions with many transitions. The left panel of Fig. :ref:`popplusspectrum` shows the level population as a function of temperature, :math:`n_j(T)`, for all of the energy levels of Fe V in the CHIANTI database.

.. figure:: figures/pop_and_spectrum.pdf
   :align: center
   :figclass: w
   :scale: 55%

   Level populations as a function of temperature (left) and intensity as a function of wavelength (right) for Fe V. The various curves in the left panel represent the multiple energy levels of the Fe V ion. The right panel shows the intensity at the discrete wavelength values (black) as well as the spectra folded through a Gaussian filter with :math:`\sigma=5\,\,\mathrm{\mathring{A}}` and a Lorentzian filter with :math:`\gamma=5\,\,\mathrm{\mathring{A}}`. :label:`popplusspectrum`

When dealing with spectral line emission, we are often most interested in the line *intensity*, that is, the power per unit volume as a function of temperature (and density). For a particular transition :math:`\lambda_{ij}`, the line intensity can be written as,

.. math::
   
   I_{ij} = \frac{1}{4\pi}\frac{hc}{\lambda}\mathrm{Ab}(X)X_kA_{ij}n_jn_e^{-1}

where :math:`\mathrm{Ab}(X)` is the abundance and :math:`X_k` is the ionization equilibrium. To calculate the intensity for each transition in CHIANTI for Fe V, we can use the method :code:`fe_5.intensity()` which returns a :math:`100\times219` array (i.e. dimension of temperature by the number of available transitions). The convenience methods :code:`fe_5.intensityPlot()` and :code:`fe_5.intensityList()` can also be used to quickly visualize and enumerate the most intense lines produced by the ion. 

Finally, to simulate an observed spectrum, the intensity can be convolved with a filter to calculate the intensity as a *continuous* function of wavelength to produce a *spectrum*. For a single ion this is done using the :code:`fe_5.spectrum()` method (see later sections for creating multi-ion spectra). To create a spectrum for Fe V between 2600 :math:`\mathrm{\mathring{A}}` and 2900 :math:`\mathrm{\mathring{A}}`,

.. code-block:: python

   wavelength = np.arange(2.6e3,2.9e3,0.1)
   fe_5.spectrum(wavelength)

This method also accepts an optional keyword argument for specifying a filter with which to convolve the intensity. The default filter is a Gaussian though :code:`ChiantiPy.tools.filters` includes several different filters including Lorentzian and Boxcar filters. The right panel of Fig. :ref:`popplusspectrum` shows the Fe V intensity (black) and spectrum folded through a Gaussian (blue) and Lorentzian (green) filter at the temperature at which the ionization fraction is maximized, :math:`T\approx8.5\times10^4` K. Similar to the :code:`fe_5.populate()` and :code:`fe_5.intensity()`, ChiantiPy also provides the convenience method :code:`fe_5.spectrumPlot()` for quickly visualizing a spectrum.

.. [1] A single temperature and an array of densities is also valid. The only requirement is that if one or the other is not of length 1, both arrays must have the same length. The ion object can also be initialized without any temperature or density information if only the ion data is needed.

.. [2] This convention is likely to change in the near future as the ChiantiPy codebase is brought into compliance with the `PEP 8 Style Guide for Python code <https://www.python.org/dev/peps/pep-0008/>`_.

Continuum Emission
##################
In addition to calculating emissivities for individual spectral lines, ChiantiPy also calculates the free-free and free-bound continuua as a function of wavelength and temperature for each ion through the :code:`Continuum` object. Free-free emission (or *bremsstrahlung*) is produced by collisions between free electrons and positively charged ions. The free-free emissivity (in units of erg :math:`\mathrm{cm}^3\,\mathrm{s}^{-1}\,\mathrm{\mathring{A}}^{-1}\,\mathrm{str}^{-1}`) is given by,

.. math::
   :type: align

   \frac{dW}{dtdVd\lambda} =& \frac{c}{3m_e}\left(\frac{\alpha h}{\pi}\right)^3\left(\frac{2\pi}{3m_ek_B}\right)^{1/2}\frac{Z^2}{\lambda^2T^{1/2}}\bar{g}_{ff} \\
   &\times\exp{\left(-\frac{hc}{\lambda k_BT}\right)},

where :math:`\alpha` is the fine structure constant, :math:`Z` is the nuclear charge, :math:`T` is the electron temperature, and :math:`\bar{g}_{ff}` is the velocity-averaged Gaunt factor :cite:`rybicki_radiative_1979`. :math:`\bar{g}_{ff}` is calculated using the methods of :cite:`itoh_relativistic_2000` (:code:`Continuum.itoh_gaunt_factor()`) and :cite:`sutherland_accurate_1998` (:code:`Continuum.sutherland_gaunt_factor()`), depending on the temperature range. 

Similarly, free-bound emission is produced when a free electron collides with a positively-charged ion and the previously-free electron is captured into an excited state of the ion. Because this process (unlike free-free emission) involves the details of the energy level structure of the ion, its formulation is necessarily quantum mechanical though a semi-classical treatment is possible (see Section 4.7.2 of :cite:`phillips_ultraviolet_2008` and Section 10.5 of :cite:`rybicki_radiative_1979`). From :cite:`young_chianti-atomic_2003`, the free-bound emission can be calculated as,

.. math::

   \frac{dW}{dtdVd\lambda} = \frac{1}{4\pi}\frac{2}{hk_Bc^3m_e\sqrt{2\pi k_Bm_e}}\frac{E^5}{T^{3/2}}\sum_i\frac{\omega_i}{\omega_0}\sigma_i^{bf}\exp\left(-\frac{E - I_i}{k_BT}\right),

where :math:`E=hc/\lambda` is the photon energy, :math:`\omega_i` and :math:`\omega_0` are the statistical weights of the :math:`i^{\mathrm{th}}` level of the recombined ion and the ground level of the recombing ion, respectively, :math:`\sigma_i^{bf}` is the photoionization cross-section, and :math:`I_i` is the ionization potential of level :math:`i`. The cross-sections are calculated using the methods of :cite:`verner_analytic_1995` (for the ground state, i.e. :math:`i=0`) and :cite:`karzas_electron_1961` (for :math:`i\neq0`). An optional :code:`use_verner` keyword argument (:code:`True` by default) is included in the :code:`Continuum.calclulate_free_bound_emission()` so that users can choose to only use the method of :cite:`karzas_electron_1961` in the photoionization cross-section calculation.

.. figure:: figures/continuum.pdf
   :align: center
   :figclass: w
   :scale: 55%

   Continuum emission for Fe XVIII. The left (middle) panel shows the free-free, free-bound, and total emission as a function of temperature (wavelength) for 
   :math:`\lambda\approx7.5\,\mathrm{\mathring{A}}` (:math:`T\approx10^7` K). The contours in the rightmost panel shows the total emissivity as a function of
   both temperature and wavelength on a log scale. The dashed lines indicate the cuts shown in the left and middle panels.

To calculate the free-free and free-bound emission with ChiantiPy,

.. code-block:: python

   import ChiantiPy.core as ch
   import numpy as np
   temperature = np.logspace(6,8.5,100)
   cont_fe18 = ch.Continuum('fe_18',temperature)
   wavelength = np.logspace(0,3,100)
   cont_fe18.calculate_free_free_emission(wavelength)
   cont_fe18.calculate_free_bound_emission(wavelength)

The :code:`Continuum.calculate_free_free_emission()` (:code:`Continuum.calculate_free_bound_emission()`) method stores the :math:`N_T` by :math:`N_{\lambda}` array (e.g. in the above example, :math:`100\times100`) in the :code:`Continuum.free_free_emission` (:code:`Continuum.free_bound_emission`) attribute. The :code:`Continuum` object also provides methods for calculating the free-free and free-bound radiative losses (i.e. the wavelength-integrated emission). These methods are primarily used by the :code:`radiativeLoss` module. The :code:`Continuum` module has recently been completely refactored and validated against the corresponding IDL results.

A contribution from the two-photon continuum can also be calculated with ChiantiPy though this is included in the :code:`ion` object through the method :code:`ion.twoPhoton()`. The two-photon continuum calculation is included in the :code:`ion` object and not the :code:`Continuum` object because the level populations are required when calculating the two-photon emissivity. See Eq. 11 of :cite:`young_chianti-atomic_2003`.

Ionization Equilibrium
######################
The ionization equilibrium of a particular ion describes what fraction of the ions of an element are in a particular ionization state at a given temperature. Specifically, the ionization equilibrium is determined by the balance ionization and recombination rates. For an element :math:`X` and an ionization state :math:`i`, assuming ionization equilibrium, the ionization state :math:`X_i=N(X^{+i})/N(X)` is given by,

.. math::

   I_{i-1}X_{i-1} + R_iX_{i+1} = I_iX_i + R_{i-1}X_i

where :math:`I_i` and :math:`R_i` are the total ionization and recombination rates for ionization state :math:`i`, respectively. In CHIANTI, these rates are assumed to be density-independent and only a function of temperature. 

In ChiantiPy, the ionization equilibrium for a particular element can be calculated using the :code:`ioneq` module,

.. code-block:: python

   import ChiantiPy.core as ch
   import numpy as np
   fe_ioneq = ch.ioneq('Fe')
   temperature = np.logspace(3.5,9.5,500)
   fe_ioneq.calculate(temperature)

The :code:`ioneq.calculate()` method sets the :code:`Ioneq` attribute, an array with :math:`Z+1` columns and :math:`N_T` rows, where :math:`N_T` is the length of the temperature array. In the example above, :code:`fe_ioneq.Ioneq` has 27 rows (i.e. :math:`Z=26` for Fe) and 500 columns. Fig. :ref:`ioneq` shows the ion population fractions for four different elements as a function of temperature, assuming ionization equilibrium.

.. figure:: figures/ioneq.pdf

   Population fractions as a function of temperature for (clockwise from upper left) H, Na, Fe, and S calculated using ionization and recombination data
   from CHIANTI and assuming ionization equilibrium. :label:`ioneq`

The :code:`ioneq` module also allows the user to load a predefined set of ionization equilibria via the :code:`ioneq.load()` method. Though CHIANTI includes several ionization equilibrium datasets from other workers, it is recommended to use the most up to data as supplied by CHIANTI (see :cite:`dere_chianti_2009` for more details). To load the ionization equilibrium data for Fe,

.. code-block:: python
   
   fe_ioneq = ch.ioneq('Fe')
   fe_ioneq.load()

This will populate the :code:`fe_ioneq.Temperature` and :code:`fe_ioneq.Ioneq` attributes with data from the appropriate ionization equilibrium file. By default, this will be :code:`ioneq/chianti.ioneq` unless otherwise specified in the :code:`chiantirc` file or the :code:`ioneqName` keyword argument.

Spectra
##################
Examples of how to calculate spectra for a single ion and for all ions over a range of temperature and density

Radiative Losses
#################
The radiative loss rate 

Documentation, Testing, and Infrastructure
------------------------------------------
The ChiantiPy project has made an effort to embrace modern development practices when it comes to developing, documenting and releasing the ChiantiPy codebase. Like many open source projects started in the late 2000s, ChiantiPy was originally hosted on SourceForge, but has now moved its development entirely to `GitHub <https://github.com/chianti-atomic/ChiantiPy>`_. The SVN commit history is in the process of being migrated to GitHub as well. The move to GitHub has provided increased development transparency, ease of contribution, and better integration with third-party services.

An integral part of producing quality scientific code, particularly that meant for a large user base, is continually testing said code and as improvements are made and features are added. For each merge into master as well as each pull request, a series of tests is run on `Travis CI <https://travis-ci.org/chianti-atomic/ChiantiPy>`_, a continuous integration service and that provides free and automated builds configured through GitHub webhooks. This allows each contribution to the codebase to be tested to ensure that these changes do not break the codebase in unexpected ways. Currently, ChiantiPy is tested on Python 2.7, 3.4, and 3.5, with full 3.6 support expected soon. Currently, the ChiantiPy package is installed in each of these environments and minimal set of tests of each core module is run along with documentation builds to ensure that Sphinx can generate the documentation. The actual module tests are currently quite sparse though one of the more pressing goals of the project is to increase test coverage of the core modules.

One of the most important parts of any codebase is the documentation. The ChiantiPy documentation is built using Sphinx and is `hosted on Read the Docs <http://chiantipy.readthedocs.io/en/latest/>`_. At each merge into the master branch, a new Read the Docs build is kicked off, ensuring that the ChiantiPy API documentation is never out of date with the most recent check in. In addition to the standard API documentation, the ChiantiPy Read the Docs page also provides a tutorial for using the various modules in ChiantiPy as well as a guide for those switching from the IDL version. 

ChiantiPy has benefited greatly from the `astropy-helpers package template <https://github.com/astropy/astropy-helpers>`_ provided by the Astropy collaboration :cite:`astropy_collaboration_astropy:_2013`. asropy-helpers provides boilerplate code for setting up documentation and testing frameworks which has allowed the package to adopt modern testing and documentation practices with little effort. 

Future Work: Towards ChiantiPy v1.0
-----------------------------------
Goals, new features, fixes, refactoring, big projects, etc

References
----------



