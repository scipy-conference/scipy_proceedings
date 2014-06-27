:author: John A. ZuHone
:email: jzuhone@milkyway.gsfc.nasa.gov
:institution: Astrophysics Science Division, Laboratory for High Energy Astrophysics, Code 662, NASA/Goddard Space Flight Center, Greenbelt, MD 20771

:author: Veronica Biffi
:email: biffi@sissa.it
:institution: SISSA - Scuola Internazionale Superiore di Studi Avanzati, Via Bonomea 265, 34136 Trieste, Italy

:author: Eric Hallman
:email: hallman13@gmail.com
:institution: Center for Astrophysics and Space Astronomy, Department of Astrophysical & Planetary Science, University of Colorado, Boulder, CO 80309

:author: Scott W. Randall
:email: srandall@cfa.harvard.edu
:institution: Harvard-Smithsonian Center for Astrophysics, 60 Garden Street, Cambridge, MA 02138

:author: Adam Foster
:email: afoster@cfa.harvard.edu
:institution: Harvard-Smithsonian Center for Astrophysics, 60 Garden Street, Cambridge, MA 02138

:author: Christian Schmid
:email: christian.schmid@sternwarte.uni-erlangen.de
:institution: Dr. Karl Remeis-Sternwarte & ECAP, Sternwartstr. 7, 96049 Bamberg, Germany

-----------------------------------------
Simulating X-ray Observations with Python
-----------------------------------------

.. class:: abstract

  X-ray astronomy is an important tool in the astrophysicist's toolkit to investigate
  high-energy astrophysical phenomena. Numerical simulations of astrophysical sources are 
  fully three-dimensional representations of all of the relevant physical quantities,
  whereas astronomical observations are two-dimensional projections of the emission
  generated via a number of mechanisms from these sources. To bridge the gap between 
  simulations and observations, packages for generating synthetic observations of 
  simulated data have been developed. We present an implementation of such an algorithm 
  in the ``yt`` analysis software package. We describe the underlying model for 
  generating the X-ray photons, the important role that ``yt`` and other Python packages
  play in its implementation, and present a detailed workable example of the creation of 
  simulated X-ray observations.
  
.. class:: keywords

  astronomical observations, astrophysics simulations, visualization

.. |einstein| replace:: *Einstein*
.. _einstein: http://heasarc.gsfc.nasa.gov/docs/einstein/heao2.html

.. |rosat| replace:: *ROSAT*
.. _rosat: http://science.nasa.gov/missions/rosat/

.. |chandra| replace:: *Chandra*
.. _chandra: http://chandra.harvard.edu

.. |xmm| replace:: *XMM-Newton*
.. _xmm: http://xmm.esac.esa.int/

.. |suzaku| replace:: *Suzaku*
.. _suzaku: http://www.isas.jaxa.jp/e/enterp/missions/suzaku/

.. |nustar| replace:: *NuSTAR*
.. _nustar: http://www.nustar.caltech.edu/

.. |astroh| replace:: *Astro-H*
.. _astroh: http://astro-h.isas.jaxa.jp/en/

.. |athena_plus| replace:: *Athena+*
.. _athena_plus: http://www.the-athena-x-ray-observatory.eu/

.. |phox| replace:: ``PHOX``
.. _phox: http://www.mpa-garching.mpg.de/~kdolag/Phox/

.. |yt| replace:: ``yt``
.. _yt: http://yt-project.org

.. |xspec| replace:: ``XSPEC``
.. _xspec: http://heasarc.gsfc.nasa.gov/xanadu/xspec

.. |pyxspec| replace:: ``PyXspec``
.. _pyxspec: http://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/

.. |marx| replace:: ``MARX``
.. _marx: http://space.mit.edu/ASC/MARX/

.. |simx| replace:: ``SIMX``
.. _simx: http://hea-www.harvard.edu/simx/

.. |sixte| replace:: ``Sixte``
.. _sixte: http://www.sternwarte.uni-erlangen.de/research/sixte/

.. |scipy| replace:: ``SciPy``
.. _scipy: http://www.scipy.org

.. |astropy| replace:: ``AstroPy``
.. _astropy: http://www.astropy.org

.. |ciao| replace:: ``CIAO``
.. _ciao: http://cxc.harvard.edu/ciao/

.. |sherpa| replace:: ``Sherpa``
.. _sherpa: http://cxc.harvard.edu/sherpa4.4/

.. |aplpy| replace:: ``APLpy``
.. _aplpy: http://aplpy.github.io/

.. |flash| replace:: ``FLASH``
.. _flash: http://flash.uchicago.edu

.. |enzo| replace:: ``Enzo``
.. _enzo: http://www.enzo-project.org

.. |athena| replace:: ``Athena``
.. _athena: http://www.astro.princeton.edu/~jstone/athena.html

.. |gadget| replace:: ``Gadget``
.. _gadget: http://www.mpa-garching.mpg.de/gadget/

.. |simput| replace:: ``SIMPUT``
.. _simput: http://hea-www.harvard.edu/heasarc/formats/simput-1.0.0.pdf

Introduction
------------

In the early 21st century, astronomy is truly a multi-wavelength enterprise. Ground and space-based instruments across the electromagnetic spectrum, from radio waves to gamma rays, provide the most complete picture of the various physical processes governing the evolution of astrophysical sources. In particular, X-ray astronomy probes high-energy processes in astrophysics, including high-temperature thermal plasmas and relativistic cosmic rays. 

Since X-rays are prevented from reaching ground-based instruments by Earth's atmosphere, astronomers must employ space-based telescopes to observe them. Such instruments have a long and successful pedigree. These include |einstein|_, |rosat|_, |chandra|_, |xmm|_, |suzaku|_, and |nustar|_, as well as upcoming missions such as |astroh|_ and |athena_plus|_. 

An important distinguishing feature of X-ray astronomy from that of studies at longer wavelengths is that it is inherently `discrete`, e.g., the numbers of photons per second that reach the detectors are small enough that the continuum approximation, valid for longer-wavelength photons such as those in the visible light, infrared, microwave, and radio bands, is invalid. Instead of maps of intensity, the fundamental data products of X-ray astronomy are tables of individual photon detection positions, energies, and arrival times. The precise characterization of the statistical and systematic sources of error

Integrating the insights from observations into the design and implementation of numerical simulations of astrophysical objects can be very difficult. On the flip-side, predicting from simulations what will be seen in observations is not necessarily straightforward, due to contaminating backgrounds, projection effects, and other sources of statistical and systematic error. This is because in contrast to simulations, where all of the physical quantities in 3 dimensions are completely known to the precision of the simulation algorithm, astronomical observations are by definition 2-D projections of 3-D sources along a given sight line, and the observed spectrum of emission is a complicated function of the fundamental physical properties (e.g., density, temperature, composition) of the source. 

Such difficulties in bridging these two worlds have given rise to efforts to bridge the gap in the direction of the creation of synthetic observations from simulated data. This involves the determination of the spectrum of the emission from the properties of the source, the projection of this emission along the chosen line of sight, and, in the case of X-ray (and :math:`\gamma`-ray) astronomy, the generation of synthetic photon samples. These photons are then convolved with the instrumental responses and (if necessary) background effects are added. One implementation of such a procedure, |phox|_, was described in [Biffi12]_ and [Biffi13]_, and used for the analysis of simulated galaxy clusters from smoothed-particle hydrodynamics (SPH) cosmological simulations. 

In this work, we describe an implementation of ``PHOX`` within the Python-based |yt|_ simulation analysis package. We outline the design of the ``PHOX`` algorithm, the specific advantages to implementing it in Python and ``yt``, and provide a workable example of the generation of a synthetic X-ray observation from a simulation dataset. 

Model
-----

The overall model that underlies the ``PHOX`` algorithm may be split up into roughly three parts: a spectral emission model that determines how the photons are generated by the emitting plasma, a spatial model that determines the spatial distribution of the emitted photons and their reception in the observer's frame, and an instrumental model that describes the properties of the instrument which detects the incoming photons. We briefly describe each of these in turn. 

.. figure:: schematic.png
   :align: center
   :figclass: w
   :scale: 25 %
   
   Schematic representation of a roughly spherical X-ray emitting object, such as a 
   galaxy cluster. The volume element :math:`\Delta{V}_i` at position :math:`{\bf r}_i` 
   in the coordinate system :math:`{\cal O}` of the source has a velocity 
   :math:`{\bf v}_i`. Photons emitted along the direction given by :math:`\hat{\bf n}`
   will be received in the observer's frame in the coordinate system :math:`{\cal O}'`,
   and will be Doppler-shifted by the line-of-sight velocity component :math:`v_{i,z'}`.
   :math:`{\rm Chandra}` telescope image credit: NASA/CXC. :label:`schematic`

Spectrum of the Emission
========================

In order to generate the simulated photons, a spectral model for the photons must first be specified. In general, the normalization of the photon spectrum for a given volume element will be set by the number density of emitting particles, and the shape of the spectrum will be set by the energetics of the same particles. 

As a specific and highly relevant example, one of the most common sources of X-ray emission is that from a low-density, high-temperature, thermal plasma, such as that found in the solar corona, supernova remnants, "early-type" galaxies, galaxy groups, and galaxy clusters. The specific photon emissivity associated with a given density, temperature, and metallicity of such a plasma is given by 

.. math::
  :label: emissivity

  \epsilon_E^\gamma = n_en_H\Lambda_E(T,Z)~{\rm photons~s^{-1}~cm^{-3}~keV^{-1}}

where :math:`n_e` and :math:`n_H` are the electron and proton number densities in :math:`{\rm cm^{-3}}` and :math:`\Lambda_E(T,Z)` is the spectral emission model. In this case, the normalization of the spectrum for a volume element is set by the emission measure EM = :math:`\int{n_en_H}dV`, and the shape of the spectrum is determined by the temperature and metallicity. The dominant contributions to :math:`\Lambda_E` for an optically-thin, fully-ionized plasma are bremmstrahlung ("free-free") emission and collisional line excitation. A number of models for the emissivity of such a plasma have been developed, including Raymond-Smith [Raymond77]_, MeKaL [Mewe95]_, and APEC [Smith01]_. These models (and others) are all built into the X-ray spectral fitting package |xspec|_, which includes a Python interface, |pyxspec|_.

However, astrophysical X-ray emission arises from a variety of physical processes and sources, and in some cases multiple sources may be emitting from within the same volume. For example, cosmic-ray electrons in galaxy clusters produce a power-law spectrum of X-ray emission at high energies via inverse-Compton scattering of the cosmic microwave background. Recently, the detection of previously unidentified line emission, potentially from annihilating dark matter particles, was made in stacked spectra of galaxy clusters [Bulbul14]_. The flexibility of our approach allows us to implement one or several models for the X-ray emission arising from a variety of physical processes as the situation requires. 

The emitted spectrum is modified by a number of physical processes. The first, occurring at the source itself, is Doppler shifting and broadening of spectral lines, which arises from bulk motion of the gas and turbulence. Second, since many X-ray sources are at cosmological distances, the entire spectrum is cosmologically redshifted. Finally, gas within the Milky Way galaxy situated between the observer and the source absorbs a large number of the photons, particularly at low energies. All of these effects must be taken into account when modeling the observed spectrum. 

Generating the Photons
======================

The total emission from any extended object as a function of position on the sky is a projection of the total emission along the line of sight, minus the emission that has been either absorbed or scattered out of the sight-line along the way. For most X-ray emitting sources that we are interested in, the plasma is optically thin to the photons, so they pass essentially unimpeded from the source to the observer (with the caveat that some photons are absorbed by Galactic foreground gas, as mentioned above). Therefore, when constructing our synthetic observations we assume the entire source is optically thin and that the observed emission is a simple integration of the all emitting volume elements along that line of sight. For a typical astrophysical simulation, the relevant volume elements are grid cells or Lagrangian particles, the latter of which has a spatial extent defined by a spatial smoothing kernel. These elements provide the model for the spatial distribution of the photons. 

In the first step of the ``PHOX`` algorithm, we generate a large sample of photon samples in three dimensions. For a given volume element :math:`\Delta{V}_i`, a spectrum of photons may be generated using the given emission model :math:`\Lambda_E(T_i,Z_i)`. The normalization of the spectrum is determined by several factors. The bolometric flux of photons received by the observer from the volume element is

.. math::
  :label: flux
  
  F^{\gamma}_i = \frac{n_{e,i}n_{H,i}\Lambda(T_i,Z_i)\Delta{V}_i}{4\pi{D_A^2}(1+z)^2}~{\rm photons~s^{-1}~cm^{-2}}

where :math:`z` is the cosmological redshift and :math:`D_A` is the angular diameter distance to the source (if the source is nearby, :math:`z \approx 0` and :math:`D_A` is simply the distance to the source). The total number of photons we generate from this source is given by

.. math::
  :label: n_phot
  
  N_{\rm phot} = \displaystyle\sum_i{F^{\gamma}_i}t_{\rm exp,0}A_{\rm det,0}
  
where :math:`t_{\rm exp,0}` is the exposure time of the observation and :math:`A_{\rm det,0}` is the collecting area of the instrument. Following the approach of [Biffi12]_, for this step a large number of photons :math:`N_{\rm phot}` are generated with energies in the source frame, by setting :math:`t_{\rm exp,0}` and :math:`A_{\rm det,0}` to values that are much larger than those associated with typical exposure times and actual detector areas. This is so that they be used as a suitably large Monte-Carlo sample to draw subsets of photons for more realistic observational parameters. Figure :ref:`schematic` shows a schematic representation of this model for a roughly spherical source of X-ray photons, such as a galaxy cluster. The photons are first generated for each volume element at position :math:`{\bf r}_i` in the source's coordinate system :math:`{\cal O}`, with the spectrum determined by its density :math:`\rho_i`, temperature :math:`T_i`, and metallicity :math:`Z_i`. 

The second step in the ``PHOX`` algorithm involves using this large 3-D sample of photons to create 2-D projections. A line-of-sight vector :math:`\hat{\bf n}` is chosen to define the primed coordinate system from which the photon sky positions :math:`(x',y')` in the observer's coordinate system :math:`{\cal O}'` are determined. The volume element has a velocity :math:`{\bf v}_i` in :math:`{\cal O}`, and the component :math:`v_{i,z'}` of this velocity along the line of sight results in a Doppler shift of the photon's energy of 

.. math::
  :label: doppler
   
  E_1 = E_0\sqrt{\frac{c+v_{z'}}{c-v_{z'}}}

where :math:`E_1` and :math:`E_0` are the Doppler-shifted and rest-frame energies of the photon, respectively, and :math:`c` is the speed of light in vacuum. For :math:`v_{z'} > 0` (an approaching source), the photon will be blueshifted, and for :math:`v_{z'} < 0` (a receding source), the photon will be redshifted. The photon's energy will be further reduced/redshifted by a factor of :math:`1/(1+z)`, before being received in the observer's frame. 

.. figure:: sloshing.png
   :align: center
   :figclass: w
   :width: 100%
   
   Slices of density (left) and temperature (right) of an ``Athena`` dataset of a 
   galaxy cluster core. :label:`sloshing`

Since we are now simulating an actual observation, we now choose realistic values for the exposure time :math:`t_{\rm exp}` and detector area :math:`A_{\rm det}` to decide on the number of photons to use from the original Monte-Carlo sample. At this point, we may also specify alternative values for the angular diameter distance :math:`D_A` and the cosmological redshift :math:`z`, if desired. The fraction :math:`f` of the photons that will be used in the actual observation is given by 

.. math::
  :label: fraction
  
  f = \frac{t_{\rm exp}}{t_{\rm exp,0}}\frac{A_{\rm det}}{A_{\rm det,0}}\frac{D_{A,0}^2}{D_A^2}\frac{(1+z_0)^3}{(1+z)^3}

where :math:`f \leq 1`, and the values with the :math:`0` subscript represent the values used in generating the original sample. 

The advantage of the ``PHOX`` algorithm is that the two steps of generating the photons in the source frame and projecting them along a given line of sight are separated, so that the first step, which is the most computationally expensive, need only be done once for a given source, whereas the typically cheaper second step may be repeated many times for many different lines of sight, different instruments, and different exposure times.  

Modeling Instrumental Effects
=============================

Unfortunately, the data products of X-ray observations do not simply consist of the original sky positions and energies of the received photons. Spatially, the positions of the received photons on the detector are affected by a number of instrumental factors. These include vignetting, the layout of the CCD chips, and a typically spatially dependent point-spread function. Similarly, the photon energies are binned up by the detectors into a set of discrete energy channels, and there is typically not a simple one-to-one mapping between which channel a given photon ends up in and its original energy, but is instead represented by a non-diagonal response matrix. Finally, the "effective" collecting area of the telescope is also energy-dependent, and also varies with position on the detector. When performing analysis of X-ray data, the mapping between the detector channel and the photon energy is generally encapsulated in a redistribution matrix file (RMF), and the effective area curve as a function of energy is encapsulated in an ancillary response file (ARF). 

For accurate comparison of our mock observations to real observations, and for compatibility with existing analysis tools, these effects must be taken into account. In our framework, we provide two ways of convolving the detected photons with instrumental responses, depending on the level of sophistication required. The first is a "bare-bones" approach, where the user can specify a point-spread function to convolve the photon positions with, and energy response files to convolve the photon energies with. This will result in photon distributions that are similar enough to the final data products of real observations to be sufficient for most purposes. 

However, some users may require a full simulation of a given telescope or may wish to compare observations of the same simulated system by multiple instruments. Several software packages exist for this purpose. The venerable |marx|_ software performs detailed ray-trace simulations of how `Chandra` responds to a variety of astrophysical sources, and produces standard event data files in the same FITS formats as standard `Chandra` data products. |simx|_ and |sixte|_ are similar packages that simulate most of the effects of the instrumental responses for a variety of current and planned X-ray missions. We provide convenient output formats for the synthetic photons in order that they may be easily imported into these packages. 

Implementation
--------------

The model described here has been implemented in ``yt`` [Turk11]_, a Python-based visualization and analysis toolkit for volumetric data. ``yt`` has a number of strengths that make it an ideal package for implementing our algorithm.

The first is that ``yt`` has support for analyzing data from a large number of astrophysical simulation codes. These include |flash|_, |enzo|_, |gadget|_, |athena|_, etc. The simulation-specific code is contained within various "frontend" implementations, and the user-facing API to perform the analysis on the data is the same regardless of the type of simulation being analyzed. This makes it possible to use the same scripts or IPython notebooks to generate photons for a number of different dataset types. 

The second strength is related, in that by largely abstracting out the simulation-specific concepts of "cells", "grids", "particles", "smoothing lengths", etc., ``yt`` provides a window on to the data defined primarily in terms of physically motivated volumetric region objects. These include spheres, disks, rectangular regions, regions defined on particular cuts on fields, etc. Arbitrary combinations of these region types are also possible. These volumetric region objects serve as natural starting points for generating X-ray photons from not only physically relevant regions within a simulation but also from simple "toy" models which have been constructed from scratch. 

The third major strength is that implementing our model in ``yt`` makes it possible to easily make use of the wide variety of useful libraries available within the scientific Python ecosystem. Our implementation uses |scipy|_ for integration, |astropy|_ for handling celestial coordinate systems and FITS I/O, and ``PyXspec`` for generating X-ray spectra. Tools for analyzing astrophysical X-ray data are also implemented in Python (e.g., |ciao|_'s |sherpa|_ package) so possibilities exist for integration with these tools as well. 

Example
-------

Here we present a workable example of creating simulated X-ray events using ``yt``'s photon simulator. This code has been implemented in ``yt`` v. 3.0 and is available as a IPython notebook at . 

We will use an ``Athena`` dataset of a galaxy cluster core, which can be downloaded from the ``yt`` website at http://yt-project.org/data/MHDSloshing.tar.gz.
You will also need to download a version of ``APEC`` from http://www.atomdb.org. Finally, the absorption cross section table used here and the *Chandra* response files may be downloaded from http://yt-project.org/data/xray_data.tar.gz. 

First, we must import the necessary modules: 

.. code-block:: python      

  import yt
  from yt.analysis_modules.photon_simulator.api \
      import *
  from yt.utilities.cosmology import Cosmology

Next, we load the dataset, which comes from a set of simulations presented in [ZuHone14]_:

.. code-block:: python    

   parameters={"time_unit":(1.0,"Myr"),
               "length_unit":(1.0,"Mpc"),
               "mass_unit":(1.0e14,"Msun")}

   ds = yt.load("MHDSloshing/virgo_low_res.0054.vtk",
                parameters=parameters)
   
Slices through the density and temperature of the simulation dataset are shown in Figure :ref:`sloshing`. The photons will be created from a spherical region centered on the domain center, with a radius of 250 kpc:

.. code-block:: python

  sp = ds.sphere("c", (250., "kpc"))
  
This will serve as our ``data_source`` that we will use later. Next, we
need to create the ``SpectralModel`` instance that will determine how
the data in the grid cells will generate photons. A number of options are available, but we will use the ``TableApecModel``, which allows one to use the ``APEC`` data tables:

.. code-block:: python

  atomdb_path = "/Users/jzuhone/Data/atomdb"

  apec_model = TableApecModel(atomdb_path,
                              0.01, 10.0, 2000,
                              apec_vers="2.0.2",
                              thermal_broad=False)

where the first argument specifies the path to the ``APEC`` files, the next three specify the bounds in keV of the energy spectrum and the number of bins in the table, and the remaining arguments specify the ``APEC`` version to use and whether or not to apply thermal broadening to the spectral lines. 

.. figure:: aplpy_figure.png
   :scale: 33 %
   
   100 ks exposure of our simulated galaxy cluster, from a FITS image plotted with
   ``APLpy``. :label:`image`

Now that we have our ``SpectralModel``, we need to connect this model to a ``PhotonModel`` that will connect the field data in the ``data_source`` to the spectral model to actually generate the photons which will serve as the sample distribution for observations. For thermal spectra, we have a special ``PhotonModel`` called ``ThermalPhotonModel``:

.. code-block:: python

  thermal_model = ThermalPhotonModel(apec_model, 
                                     X_H=0.75, 
                                     Zmet=0.3)

Where we pass in the ``SpectralModel``, and can optionally set values for
the hydrogen mass fraction ``X_H`` and metallicity ``Z_met``, the latter of which may be a single floating-point value or the name of the ``yt`` field representing the spatially-dependent metallicity.

Next, we need to specify "fiducial" values for the telescope collecting area, exposure time, and cosmological redshift, choosing generous values so that there will be a large number of photons to sample from. We also construct a ``Cosmology`` object, which will be used to determine the source distance from its redshift:

.. code-block:: python

  A = 6000.
  exp_time = 4.0e5
  redshift = 0.05
  cosmo = Cosmology()

Now, we finally combine everything together and create a ``PhotonList``
instance, which contains the photon samples:

.. code-block:: python

  photons = PhotonList.from_scratch(sp, redshift, A, 
                                    exp_time,
                                    thermal_model, 
                                    center="c",
                                    cosmology=cosmo)

where we have used all of the parameters defined above, and ``center`` defines the reference coordinate which will become the origin of the photon coordinates.

.. figure:: spectrum.png
   :scale: 33 %
   
   Counts spectrum of the photons from our simulated observation. :label:`spectrum`
   
At this point, the ``photons`` are distributed in the three-dimensional
space of the ``data_source``, with energies in the rest frame of the
plasma. These ``photons`` can be saved to disk in an HDF5 file:

.. code-block:: python

  photons.write_h5_file("my_photons.h5")

which is most useful if it takes a long time to generate the photons,
because a ``PhotonList`` can be created in-memory from the dataset
stored on disk:

.. code-block:: python

  photons = PhotonList.from_file("my_photons.h5")

so that they may be used later to generate different samples.

.. figure:: comparison.png
   :align: center
   :figclass: w
   :scale: 50 %
   
   100 ks exposures of our simulated galaxy cluster, observed with several
   different existing and planned X-ray detectors. The `Chandra` image
   was made with ``MARX``, while the others were made with ``SIMX``. All images have the
   same angular scale. :label:`comparison`

At this point the photons can be projected along a line of sight to create a synthetic observation. First, it is necessary to set up a spectral model for the absorption coefficient, similar to the spectral model for the emitted photons set up previously. Here again, there are multiple options, but for the current example we use ``TableAbsorbModel``, which allows one to use an absorption cross section vs. energy table written in HDF5 format. The only other argument that is needed is the column density ``N_H`` in units of :math:`10^{20}~{\rm cm}^{-2}`.

.. code-block:: python

  N_H = 0.1 
  a_mod = TableAbsorbModel("tbabs_table.h5", N_H) 

Second, we choose a line-of-sight vector ``L``. Third, we may adjust the exposure time, telescope area, and the source redshift to more appropriate values for the particular observation we are trying to simulate. Fourth, we'll pass in the absorption ``SpectrumModel``. We'll specify a ``sky_center`` in RA, Dec on the sky in degrees. In this case, we'll also provide two instrumental responses to convolve the observed photons with.     
     
.. code-block:: python
      
  ARF = "chandra_ACIS-S3_onaxis_arf.fits"
  RMF = "chandra_ACIS-S3_onaxis_rmf.fits"
  resp = [ARF,RMF]
  L = [0.0,0.0,1.0]
  events = photons.project_photons(L, 
                                   exp_time_new=1.0e5, 
                                   redshift_new=0.07, 
                                   absorb_model=a_mod,
                                   responses=resp)
       
``project_photons`` draws events uniformly from the ``photons`` sample, the number of which is set by the (optional) parameters ``redshift_new``, ``exp_time_new``, and ``area_new``, orients their positions in the coordinate frame defined by ``L``, and applies the Doppler and cosmological energy shifts. Lastly, a number of the events are removed according to the supplied Galactic absorption model ``absorb_model`` before arriving in the observer's frame. 

In the case where instrumental ``responses`` are provided, there are two additional steps. If an ARF is provided, the maximum value of the effective area curve will serve as the ``area_new`` parameter, and after the absorption step a number of events are further removed using the effective area curve as the acceptance/rejection criterion. If an RMF is provided, the event energies will convolved with it to produce a new array with their resulting spectral channels. 

The ``events`` may be binned into an image and written to a FITS file:           
             
.. code-block:: python

  events.write_fits_image("my_image.fits", 
                          clobber=True, 
                          emin=0.5, emax=7.0)
             
where ``emin`` and ``emax`` specify the energy range for the image. Figure :ref:`image` shows the resulting FITS image plotted using |aplpy|_. 

We can also bin up the spectrum into energy bins, and write it to a FITS table file. This is an example where we’ve binned up the spectrum according to the unconvolved photon energy:

.. code-block:: python

  events.write_spectrum("my_spec.fits", 
                        energy_bins=True, 
                        emin=0.1, emax=10.0, 
                        nchan=2000, clobber=True)

here ``energy_bins`` specifies whether we want to bin the events in unconvolved photon energy or convolved photon channel. Figure :ref:`spectrum` shows the resulting spectrum.

For outputting the photons for use with other software packages to simulate specific X-ray instruments, there are a couple of options. For input to ``MARX``, we provide an implementation of a ``MARX`` "user source" at http://bitbucket.org/jzuhone/yt_marx_source, which takes as input an HDF5 file:

.. code-block:: python

  events.write_h5_file("my_events.h5")
  
Input to ``SIMX`` and ``Sixte`` is handled via |simput|_, a file format designed specifically for the output of simulated X-ray data:

.. code-block:: python

  events.write_simput_file("my_events", 
                           clobber=True, 
                           emin=0.1, emax=10.0)
  
where ``emin`` and ``emax`` are the energy range in keV of the outputted events. Figure :ref:`comparison` shows several examples of the generated photons passed through various instrument simulations. For this to work correctly, the ``events`` object must be generated by a call to ``project_photons`` which does not apply responses, since these will be applied by the instrument simulator. 

Summary
-------

We have developed an analysis module within the Python-based volumetric data analysis toolkit ``yt`` to construct synthetic X-ray observations of astrophysical sources from simulation datasets, based on the ``PHOX`` algorithm. This algorithm generates a large sample of X-ray photons in the rest frame of the source from the physical quantities of the simulation dataset, and uses these as a sample from which a smaller number of photons are drawn and projected onto the sky plane, the number and spatial and energy distributions of which correspond to a simulated observation with a real detector. The utility of this algorithm lies in the fact that the most expensive step, namely that of generating the photons from the source, need only be done once, and these may be used as a Monte Carlo sample from which to generate as many simulated observations along as many projections and with as many instrument models as desired. 

The primary strength of our implementation of ``PHOX`` is its use of ``yt`` as the interface to the simulation data that the synthetic X-ray photons are created from. This allows us to take advantage of the full range of capabilities of ``yt``, especially its focus on physically motivated representations of simulation data and its support for a wide variety of simulation codes as well as generic ``NumPy`` array data generated on-the-fly. We also benefit from the choice of Python as the language for our module, including its object-oriented capabilities as well as the astronomical and scientific Python packages that we take advantage of in its design. 

Software such as ours benefits the astronomical community in a number of ways. The first is that it provides a crucial link between observations of astronomical sources and the simulations designed to represent the objects that are detected via their electromagnetic radiation, enabling some of the most direct possible testing of these simulations. Second, it is potentially useful as a proposer's tool, allowing observers to generate simulated observations from simulated data and even simple "toy models" of astrophysical systems, which can be used to precisely quantify and motivate the needs of a proposal for observing time on a particular instrument. Our software also serves as a model for how similar modules for simulating observations in other wavebands may be designed, particularly in how it makes use of several important Python packages for astronomy. 

References
----------

.. [Biffi12] Biffi, V., Dolag, K., Böhringer, H., & Lemson, G. 2012, MNRAS, 420, 3545

.. [Biffi13] Biffi, V., Dolag, K., Böhringer, H. 2013, MNRAS, 428, 1395 

.. [Bulbul14] Bulbul, E., Markevitch, M., Foster, A., et al. 2014, ApJ, 789, 13

.. [Mewe95] Mewe, R., Kaastra, J. S., & Liedahl, D. A. 1995, Legacy, 6, 16

.. [Raymond77] Raymond, J. C., & Smith, B. W. 1977, ApJS, 35, 419

.. [Smith01] Smith, R. K., Brickhouse, N. S., Liedahl, D. A., & Raymond, J. C. 2001, ApJL, 556, L91

.. [Turk11] Turk, M. J., Smith, B. D., Oishi, J. S., Skory, S., Skillman, S. W., Abel, T., & Norman, M. L. 2011, ApJS, 192, 9

.. [ZuHone14] ZuHone, J. A., Kunz, M. W., Markevitch, M., Stone, J. M., & Biffi, V. 2014, arXiv:1406.4031 


