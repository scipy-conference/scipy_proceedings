:author: Matthew J. Shannon
:email: Matthew.J.Shannon@nasa.gov
:institution: Universities Space Research Association, Columbia, MD
:institution: NASA Ames Research Center, MS245-6, Moffett Field, CA 94035-1000

:author: Christiaan Boersma
:email: Christiaan.Boersma@nasa.gov
:institution: San José State University Research Foundation, 210 N 4th St Fl 4, San Jose, CA 95112
:institution: NASA Ames Research Center, MS245-6, Moffett Field, CA 94035-1000

:bibliography: bib

-----------------------------------------------------------------------------------------------------------------------
Organic Molecules in Space: Insights from the NASA Ames Molecular Database in the era of the James Webb Space Telescope
-----------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   We present the software tool pyPAHdb to the scientific astronomical
   community, which is used to characterize emission from one of the
   most prevalent types of organic molecules in space, namely
   polycyclic aromatic hydrocarbons (PAHs). It leverages the detailed
   studies of organic molecules done at the NASA Ames Research
   Center. pyPAHdb is a streamlined Python version of the NASA Ames
   PAH IR Spectroscopic Database (PAHdb; `www.astrochemistry.org/pahdb
   <http://www.astrochemistry.org/pahdb>`_) suite of IDL tools. PAHdb
   has been extensively used to analyze and interpret the PAH
   signature from a plethora of emission sources, ranging from
   solar-system objects to entire galaxies. pyPAHdb decomposes
   astronomical PAH emission spectra into contributing PAH sub-classes
   in terms of charge and size using a database-fitting technique. The
   inputs for the fit are spectra constructed using the spectroscopic
   libraries of PAHdb and take into account the detailed photo-physics
   of the PAH excitation/emission process.

.. class:: keywords

   astronomy, databases, fitting, data analysis

Science rationale
==================

Polycyclic aromatic hydrocarbons
--------------------------------

Polycyclic aromatic hydrocarbons (PAHs) are a class of
molecules found throughout the Universe that drive
many critical astrophysical processes. They dominate the
mid-infrared (IR) emission of many astronomical objects, as they
absorb ultraviolet (UV) photons and re-emit that energy through a
series of IR emission features between 3-20 µm. They are seen in
reflection nebulae, protoplanetary disks, the diffuse interstellar
medium (ISM), planetary nebulae, and entire galaxies (e.g., Figure
:ref:`fig:M82`), among other environments. Structurally, they are composed of a
hexagonal carbon lattice (see Figure :ref:`fig:PAHdb`); taken as an
entire family, they are by far the largest known molecules in space.
PAHs are exceptionally
stable, allowing them to survive the harsh conditions amongst a
remarkably wide variety of astronomical objects.

.. figure:: fig_M82.png
   :align: center
   :scale: 55%

   A combined visible light-IR image from the *Spitzer Space
   Telescope* of the galaxy Messier-82 (M82), also known as the Cigar
   galaxy because of its cigar-like shape in visible light. The red
   region streaming away from the galaxy into intergalactic space
   traces the IR emission from PAHs. Credits:
   NASA/JPL-Caltech/C. Engelbracht (Steward Observatory) and the SINGS
   team.

   :label:`fig:M82`

The role of astronomical PAHs
-----------------------------------

Thanks to their ubiquity, PAH IR emission signatures are routinely
used by astronomers as probes of object type and astrophysical
processes. For example, the PAH IR signature is used as an indicator
of star formation in high redshift galaxies
:cite:`2014ApJ...786...31R` and to differentiate between black hole
and starburst engines in galactic nuclei
:cite:`1998ApJ...498..579G`. Those astronomers who study star and
planet formation use the IR PAH signature as an indicator of the
geometry of circumstellar disks :cite:`2001A&A...365..476M`
:cite:`2009A&A...502..175B`.

PAHs are believed to form in the circumstellar ejecta of late-type
stars, after which they become part of the ISM as the material travels
away from the star. Over time, PAHs are incorporated into dense clouds, wherein
they participate in ongoing chemistry and are eventually brought into
newly-forming star and budding planetary systems.

They play important roles in circumstellar processes and the diffuse ISM by
modulating radiation fields and influencing charge balance. Once
incorporated into dense molecular clouds, they can dominate cloud
cooling and promote H\ :sub:`2`\ -formation. PAHs also control the
large-scale ionization balance and thereby the coupling of magnetic
fields to the gas. Through their influence on the forces supporting
clouds against gravity, PAHs also affect the process of star formation
itself. They are a major contributor to the heating of diffuse atomic
gas in the ISM and thereby the physical conditions in such
environments and its structure.

The unique properties of PAHs, coupled with their spectroscopic
response to changing astrophysical conditions and their ability to convert
UV photons to IR radiation, makes them powerful probes of astronomical
objects at all stages of the stellar life cycle. Notably,
they allow astronomers to probe
properties of diffuse media in regions not normally accessible.



NASA Ames PAH IR Spectroscopic Database (PAHdb)
------------------------------------------------

The Astrophysics & Astrochemistry Laboratory at NASA Ames
Research Center :cite:`astrochem` provides data and tools
for analyzing and interpreting astronomical PAH spectra.
The NASA Ames PAH IR Spectroscopic Database (PAHdb;
:cite:`2018ApJS..234...32B` :cite:`2014ApJSS..211....8B`) is the
culmination of more than 30 years of laboratory and computational
research carried out at the NASA Ames Research Center to test and refine
the astronomical PAH model. PAHdb consists of three components
(all under the moniker of "PAHdb"): the spectroscopic libraries,
the website (see Figure :ref:`fig:PAHdb`), and the suite of off-line IDL\ [#]_ tools.
PAHdb has the world's foremost collection of PAH spectra.

.. [#] IDL is a registered trademark of `Harris Geospatial
       <http://www.harrisgeospatial.com/ProductsandSolutions/GeospatialProducts/IDL.aspx>`_.

.. organic molecules in astronomical
.. environments through a combination of quantum chemical calculations,
.. direct laboratory measurements and different analysis techniques of
.. astronomical data.

PAHdb is highly cited and is used to characterize and understand
organic molecules in our own Galaxy and external galaxies. The
database includes a set of innovative astronomical models and tools
that enables astronomers to probe and quantitatively analyze the state
of the PAH population. For instance, one can derive PAH ionization balance, size, structure, and
composition and tie these to the prevailing local astrophysical
conditions (e.g., electron density, parameters of the radiation field,
etc.) :cite:`2016ApJ...832...51B` :cite:`2018ApJ...858...67B`.

.. figure:: fig_screenshot.png
   :align: center

   Screenshot of the NASA Ames PAH IR Spectroscopic Database website
   located at `www.astrochemistry.org/pahdb/
   <http://www.astrochemistry.org/pahdb/>`_. Shown here are the
   details and vibrational spectrum for the PAH molecule ovalene (C\
   :sub:`32`\ H\ :sub:`14`\ ). Additionally, each vibrational
   transition is animated and can be inspected for ease of
   interpretation (shown in the lower-right).
   :label:`fig:PAHdb`

.. Accessing the data and tools
.. -----------------------------

.. At `www.astrochemistry.org/pahdb/
.. <http://www.astrochemistry.org/pahdb/>`_ these libraries can be
.. perused and/or downloaded. Figure :ref:`fig:PAHdb` presents a
.. screenshot of the website's landing page. Downloads are offered
.. formatted as ASCII or XML. In addition, several software tools are
.. provided that allow users to interact with a downloaded database
.. XML-file and perform the necessary steps to analyze astronomical
.. data. Historically, the astronomical community has embraced the IDL\
.. [#]_ programming language. As such, the software tools have been
.. developed in IDL. However, Python is seeing increasingly widespread
.. usage among astronomers, in part due to its non-proprietary
.. nature. Python has significantly matured over the last two decades and
.. many astronomical utilities once only available through IDL and/or
.. IRAF have been ported to Python (e.g., PyFITS; `www.astropy.org
.. <http://www.astropy.org>`_). Notably, many of the astronomical
.. utilities offered by the Space Telescope Science Institute, including
.. the Data Analysis Toolbox for use with NASA's upcoming *James Webb
.. Space Telescope* (*JWST*; `www.jwst.nasa.gov
.. <https://www.jwst.nasa.gov>`_), are being developed in Python.

.. .. [#] IDL is a registered trademark of `Harris Geospatial
..        <http://www.harrisgeospatial.com/ProductsandSolutions/GeospatialProducts/IDL.aspx>`_.



NASA's next great observatory for PAH research: JWST
--------------------------------------------------------------------

The next great leap forward for IR astronomy is the
the James Webb Space Telescope (*JWST*). *JWST* is NASA's
next flagship observatory and the successor to the
exceptionally successful *Hubble Space Telescope*
(`www.nasa.gov/hubble <https://www.nasa.gov/hubble>`_) and *Spitzer
Space Telescope* (`www.nasa.gov/spitzer
<https://www.nasa.gov/spitzer>`_). *JWST* is being developed through a
collaboration between NASA, the European Space Agency (ESA) and the
Canadian Space Agency (CSA). The telescope features a primary mirror
with a diameter of 6.5 m
and carries four science instruments. These instruments will observe
the Universe with unprecedented resolution and sensitivity in the
near- and mid-IR. The observatory is expected to launch in 2020.

As part of an awarded *JWST* Early Release Science (ERS) program\ [#]_,
we are developing a Python-based toolkit for
quickly analyzing PAH emission in IR spectroscopic data

.. [#] The ERS program is titled
       "Radiative Feedback from Massive Stars as Traced by Multiband Imaging
       and Spectroscopic Mosaics" (`jwst-ism.org <http://jwst-ism.org/>`_;
       ID: 1288).


.. ******

.. Detailed spectroscopic PAH analysis is currently performed at the
.. NASA Ames Research Center under the umbrella of
.. the NASA Ames PAH IR Spectroscopic Database (PAHdb), which provides
.. IDL tools and libraries for the astronomical community. To best exploit
.. the extensive capabilities of *JWST*, we will provide new tools for
.. the astronomical community, as *JWST* will be the foremost platform for
.. astronomical PAH research for years to come.

.. A 3D rendering
.. of the spacecraft is shown in Figure :ref:`fig:JWST`.

.. figure
.. fig_JWST.png
   :align: center
   :scale: 10%

   3D-rendering of *JWST* using the Maya® 3D animation, modeling,
   simulation, and rendering software
   (`www.autodesk.com/products/maya/overview
   <https://www.autodesk.com/products/maya/overview>`_). *JWST*'s
   signature 6.5 m-diameter primary mirror, made up of 18 hexagonal
   segments (gold), dominates the picture together with the stacked
   sunshield. The 3D-model is available from `nasa3d.arc.nasa.gov
   <https://nasa3d.arc.nasa.gov/search/jwst/>`_. :label:`fig:JWST`

pyPAHdb: a tool designed for JWST
=================================

The purpose of pyPAHdb is to derive astronomical parameters directly
from *JWST* observations, but the tool is not limited to *JWST*
observations alone. pyPAHdb is the light version of a full suite of
Python software tools\ [#]_ that is currently being developed, which
is an analog of the off-line IDL tools\ [#]_.  pyPAHdb will enable PAH
experts and non-experts alike to analyze and interpret astronomical
PAH emission spectra.

.. [#] *AmesPAHdbPythonSuite*: `github.com/PAHdb/AmesPAHdbPythonSuite <https://github.com/PAHdb/AmesPAHdbPythonSuite>`_

.. [#] *AmesPAHdbIDLSuite*: `github.com/PAHdb/AmesPAHdbIDLSuite <https://github.com/PAHdb/AmesPAHdbIDLSuite>`_

pyPAHdb analyzes spectroscopic observations (including spectral maps)
and characterizes the PAH emission using a database-fitting approach,
providing the PAH ionization and size fractions.

The module is imported using the following statement:

.. code-block:: python

    import pypahdb

The general program methodology is encapsulated in the flowchart
presented in Figure :ref:`fig:flowchart` and is as follows:

(1) Read-in a file containing spectroscopic PAH observations of an
    astronomical object. This functionality is provided by the class
    ``observation``, which is implemented in observation.py. The class
    uses a fall-through try-except chain to attempt to read the given
    filename using the facilities provided by ``astropy.io``. The
    spectroscopic data is stored as a class-attribute using a
    ``spectrum``-object, which holds the data in terms of absissa and
    ordinate values using ``numpy``\ -arrays. The units associated
    with the absissa and ordinate value are, in the case of a
    FITS-file, determined from the accompanying header, which is also
    stored as a class-attribute. The spectral coordinate system is
    interpreted from FITS-header keywords following the specification
    by :cite:`2006A&A...446..747G`. The ``spectrum`` class is
    implemented in spectrum.py and provides functionality to convert
    between different coordinate representations. Below is some
    example Python code demonstrating the use of the class.

.. code-block:: python

    import pypahdb
    import matplotlib.pyplot as plt
    observation = pypahdb.observation('NGC7023-NW-BRIGHT.txt_pahs.txt')
    plt.plot(observation.spectrum.abscissa, observation.spectrum.ordinate[:,0,0])
    plt.show()


(2) Decompose the observed PAH emission into contributions from
    different PAH subclasses, here charge and size. This
    functionality is provided by the class ``decomposer``, which is
    implemented in decomposer.py. The class takes as input a
    ``spectrum`` object, creates a deep copy and calls the
    ``spectrum.convertunitsto`` -method to convert the abscissa units
    to wavenumber. Subsequently, a pre-computed ``numpy``\ - matrix of
    highly oversampled PAH emission spectra stored in a ``pickle`` is
    loaded from file. Utilizing ``numpy.interp``, each of the PAH
    emission spectra, represented by a single column in the
    pre-computed matrix, is interpolated onto the frequency grid (in
    wavenumber) of the input spectrum. This process is parallelized
    using the ``multiprocessing`` package. ``optimize.nnls`` is used
    to perform a non-negative least-squares fit of the pre-computed
    spectra to the input spectrum. The solution vector (weights) is
    stored as an attribute and considered private. Combining lazy
    instantiation and Python's @property, the charge and size
    breakdown can be retrieved. In case the input spectrum represents
    a spectral cube and where possible, the calculations have
    parallelized across each pixel using, again, the
    ``multiprocessing`` package. Below is some example code
    demonstrating the use of the class.

.. code-block:: python

    result = pyPAHdb.decomposer(observation.spectrum)
    print(result.ionized_fraction)


(3) Produce output to file given a ``decomposer``-object. This
    functionality is provided by the class ``writer``, which is
    implemented in writer.py, and serves to deliver consistent output
    so that a user may assess the quality of the fit and store the PAH
    characteristics of their astronomical observations. The class uses
    ``astropy.fits`` to write the PAH characteristics to a FITS-file
    and ``matplotlib`` to generate a PDF summarizing the results. The
    class will attempt to incorporate relevant information from any
    (FITS-)header provided. Below is some example code demonstrating
    the use of the class.

.. code-block:: python

   pypahdb.writer(result, header=observation.header)

.. figure:: fig_flowchart.png
   :align: center

   pyPAHdb flowchart. Astronomical spectroscopic data is loaded,
   whether represented in FITS or ASCII files. An over-sampled
   precomputed matrix of PAH spectra is loaded and interpolated onto
   the wavelength grid of the astronomical
   observations. Database-fitting is performed using non-negative
   least-squares (NNLS), which yields the contribution of an
   individual PAH molecule to the total fit. As a result, we obtain a
   breakdown of the model fit in terms of PAH charge and size. The
   results are written to disk as a single FITS file and a PDF
   summarizing the model fit (one
   page per pixel, if a spectral cube is given as
   input). :label:`fig:flowchart`

.. figure:: fig_fit.png
   :align: center

   pyPAHdb-fit to the spectrum of a position in the spectral cube of
   NGC 7023. The upper panel displays the total model fit to the data;
   the middle panel the residual; and the lower panel the breakdown of
   PAHs in descending order from large, containing 30 atoms or more, to
   small. The charge breakdown (cation, neutral,
   anion) has been suppressed for clarity.
   :label:`fig:fit`

The performance of pyPAHdb relative to the full IDL suite was tested
by fitting a spectral cube. Using pyPAHdb, the spectral cube required
less than 4 seconds, while more than 60 seconds were needed to fit
with the full IDL suite.
It should be noted that their were differences in the actual implementation of
the two tests, which were inherent to the differences in the languages
used.

The underlying PAH physics
--------------------------

In order to analyze astronomical PAH *emission* spectra with the
spectroscopic data contained in PAHdb's libraries, a PAH emission
model is needed. pyPAHdb hides the underlying photo-physics in a
precomputed matrix. The precomputed matrix is constructed using the
full Python suite and takes modeled, highly-over-sampled PAH
emission spectra from version 3.00 of the library of computed
spectra. This matrix uses the data on a collection of "astronomical"
PAHs, which include those PAHs that have more than 20 carbon atoms,
have no hetero-atom substitutions except for possibly nitrogen, have
no aliphatic side groups, and are not fully dehydrogenated. In
addition, the fullerenes C\ :sub:`60` and C\ :sub:`70` are added.

While several more sophisticated emission models are available in the
full Python suite, here a PAH's emission spectrum is calculated from
the vibrational temperature it reaches after absorbing a single 7 eV
photon and making use of the thermal approximation (e.g.,
:cite:`1993ApJ...415..397S` and :cite:`2001A&A...372..981V`).

The spectral intensity :math:`I_{j}(\nu)`, in erg s\ :sup:`-1` cm\
:sup:`-1` mol\ :sup:`-1`, from a mol of the :math:`j^{\rm th}` PAH is
thus calculated as:

.. math::
   :label: eq:model

   I_{j}(\nu) = \sum\limits_{i=1}^{n}\frac{2hc\nu_{i}^{3}\sigma_{i}}{e^{\frac{hc\nu_{i}}{kT}} - 1}\phi(\nu)\ ,

with :math:`\nu` the frequency in cm\ :sup:`-1`, :math:`h` Planck's
constant in erg s, :math:`c` the speed-of-light in cm s\ :sup:`-1`,
:math:`\nu_{i}` the frequency of mode :math:`i` in cm\ :sup:`-1`,
:math:`\sigma_{i}` the integrated absorption cross-section for mode \
:math:`i` in cm mol\ :sup:`-1`, :math:`k` Boltzmann's constant in erg
K\ :sup:`-1`, :math:`T` the vibrational temperature in K, and
:math:`\phi(\nu)` is the frequency dependent emission profile
in cm. The sum is taken over all :math:`n` modes and the emission
profile is assumed Gaussian with a full-width at half-maximum (FWHM)
of 15 cm\ :sup:`-1`. Note that before applying the emission profile, a
redshift of 15 cm\ :sup:`-1` is applied to each of the band positions
(:math:`\nu_{i}`) to mimic some anharmonic effects. This redshift value
is currently the best estimate we have for PAH emission, as determined by experimental mid-IR studies
(see, e.g., the discussion in :cite:`2013ApJ...769..117B`).

.. Here a 15 cm−1
.. redshift is taken, a value consistent with shifts measured in a
.. number of experimental mid-IR studies (
.. Cherchneff & Barker
.. 1989; Flickinger&Wdowiak 1990, 1991; Flickinger et al. 1991;
.. Colangeli et al. 1992; Brenner&Barker 1992; Joblin et al. 1995;
.. Williams & Leone 1995; Cook & Saykally 1998
.. )


The vibrational temperature attained after absorbing a single 7 eV
photon is calculated by the molecule's heat capacity. The heat
capacity, :math:`C_{\rm V}` in erg K, of a molecular system can be
described in terms of isolated harmonic oscillators by:

.. math::
   :label: eq:heatcapacity

   C_{\rm V} = k\int\limits_{0}^{\infty}e^{-\frac{h\nu}{kT}}\left[\frac{\frac{h\nu}{kT}}{1-e^{-\frac{h\nu}{kT}}}\right]^{2}g(\nu)\mathrm{d}\nu\ ,

where :math:`g(\nu)` is known as the density of states and describes
the distribution of vibrational modes. However due to the discrete
nature of the modes, the density of states is just a sum of \
:math:`\delta`\ -functions:

.. math::
   :label: eq:delta

   g(\nu) = \sum\limits_{i=1}^{n}\delta(\nu-\nu_{i})\ .

The vibrational temperature is ultimately calculated by solving:

.. math::
   :label: eq:solve

   \int\limits_{0}^{T_{\rm vibration}}C_{\rm V} \mathrm{d}T = E_{\rm in}\ ,

where :math:`E_{\rm in}` is the energy of the absorbed photon—here
this is 7 eV.

In Python, in the full suite, Equation :ref:`eq:solve` is solved using
root-finding with ``scipy.optimize.brentq``. The integral is
calculated with ``scipy.optimize.quad``.

Figure :ref:`fig:model` illustrates the process on the spectrum of the
coronene cation (C\ :sub:`24`\ H\ :sub:`12`\ :sup:`+`\ ), which
reaches a vibrational temperature of 1406 K after absorbing a single 7
eV photon.

.. figure:: fig_model.png
   :align: center

   Demonstration of applying the simple PAH emission model as outlined
   in Equations :ref:`eq:model`\ -:ref:`eq:solve` to the 0 K spectrum
   of coronene (in black; C\ :sub:`24`\ H\ :sub:`12`\ :sup:`+`) from
   version 3.00 of the library of computed spectra of PAHdb. After
   applying the PAH emission model, but before the convolution with
   the emission profile, the blue spectrum is obtained. The final
   spectrum is shown in orange. For display purposes the profiles have
   been given a FWHM of 45 cm\ :sup:`-1`. :label:`fig:model`

Demonstration
-------------

.. figure:: fig_demonstration.png
   :align: center
   :figclass: w
   :scale: 45

   **Left:** An image of the reflection nebula NGC 7023 as obtained by the
   *Hubble Space Telescope*. Overlaid is a pixel grid representing a
   spectral cube of observations taken with the *Spitzer Space
   Telescope*; each pixel contains an infrared spectrum. In this
   figure, the exciting star is just beyond the lower left corner. We
   are observing a photodissociation region boundary: the
   material in the lower half of the figure is diffuse and exposed to
   the star; the material in the upper (right) half is molecular and
   shielded from the star. The diagonal boundary separating the two
   zones is clearly visible. PAHs are common in these
   environments. Figure adapted from :cite:`2018ApJ...858...67B`.
   **Right:**  We display PAH ionization across the NGC 7023 (white grid
   in left panel), using pyPAHdb. Here, an ionization fraction of ``1`` means
   all PAHs are ionized, while ``0`` means all are neutral.
   Note that in the diffuse, exposed cavity (lower half) the
   PAHs are on average more ionized than in the denser molecular zone
   (upper half).
   :label:`fig:7023`

As a more sophisticated demonstration of pyPAHdb's utility,
we analyze a spectral cube dataset of the reflection nebula
NGC 7023, as constructed from *Spitzer Space Telescope* observations.
This data cube is overlaid on a visible-light image of NGC 7023 from the
*Hubble Space Telescope* in Figure :ref:`fig:7023`, left panel
:cite:`2018ApJ...858...67B`.

The spectral cube is aligned such that, in these observations, we observe
the transition from diffuse, ionized/atomic
species (e.g., HI) near the exciting star to dense, molecular material
(e.g., H\ :sub:`2`) more distant from the star. The transition zone
between the two is the PDR, where PAHs have a strong presence. The
properties of the PAH molecules are known to vary across these
boundaries, since they are exposed to harsh radiation in the exposed
cavity of the diffuse zone, and shielded in the molecular region.

We use pyPAHdb to derive the variability of PAH properties across this
boundary layer by analyzing the full spectrum at every pixel. The code-block
below, which is taken from ``example.py`` included in the pyPAHdb
distribution, demonstrates how this is done. Note that this is the same
general syntax as is used for analyzing a single spectrum, but here
`NGC7023.fits` is a spectral cube.

.. code-block:: python

    # ----------------------------------------- #
    # ------------ Running pyPAHdb ------------ #
    # ----------------------------------------- #

    import pyPAHdb
    observation = pyPAHdb.observation('NGC7023.fits')
    result = pyPAHdb.decomposer(observation.spectrum)

    # This will output the results file,
    # 'NGC7023_pypahdb.fits':
    pyPAHdb.writer(result, header=observation.header)

With the results from the entire spectral cube, maps of relevant
astrophysical quantities can be constructed. For example, Figure
:ref:`fig:7023` (right panel) presents a map of the varying PAH ionization fraction
across NGC 7023. As expected, the fraction is systematically
higher across the diffuse region, where PAHs are more exposed to the
star, than the dense region, where PAHs are partially shielded from
the star. This figure was constructed in the following manner:

.. code-block:: python

    # ----------------------------------------- #
    # ----- Plotting a map of ionization ------ #
    # ----------------------------------------- #

    # Import needed/useful modules.
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from mpl_toolkits.axes_grid1 import \
        make_axes_locatable

    # Read in the results from pyPAHdb.
    # The data is 3-dimensional, with the first axis
    # denoting the PAH properties, and the latter two
    # being spatial.
    hdulist = fits.open('NGC7023_pypahdb.fits')
    ionization_fraction, large_fraction, norm = \
        hdulist[0].data

    # Create a figure instance.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot our ionization map; we've flipped it left-right
    # to match the Hubble image's orientation.
    im = ax.imshow(np.fliplr(ionization_fraction),
                   origin='upper', cmap='viridis',
                   interpolation='nearest')

    # Add a nice colorbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%",
                              pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('ionization fraction [#]',
                   rotation=270, labelpad=18)

    # Set axes labels.
    ax.set_xlabel('pixel [#]')
    ax.set_ylabel('pixel [#]')

    # Save the figure.
    plt.savefig('ionization_fraction_map.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()

The type of analysis demonstrated here allows users to quickly
interpret the distribution of PAHs in their astronomical observations
and variations in PAH charge and size. Note that in addition to the
ionization fraction, the pyPAHdb results file `NGC7023.fits`
contains a data array for the ``large PAH fraction`` and ``norm``,
also defined in the code above.


Summary
===================

pyPAHdb is in active development, but a finalized Python software
analysis tool is anticipated to be complete well before *JWST*'s launch, which is
currently scheduled for 2020. The astronomical community can already
benefit from pyPAHdb by using it to quickly analyze and interpret
archival data from other observatories, e.g., *ISO*, *Spitzer*,
*SOFIA*, etc. Our current efforts are focused on extending pyPAHdb, including
having it transparently accept spectroscopic observations in a variety
of digital formats, and consolidating output parameters. Further testing of the program logic
will be performed to ensure all parts of pyPAHdb function as
expected. Lastly, API documentation and a guide with analysis
"recipes" will be provided to help users get started and/or extend
pyPAHdb.

The development of a PAHdb tool in Python has turned out to be largely
straightforward as Python is backed by a large active
community. Python offers great flexibility and in combination with
pyPAHdb's development on GitHub, allows constructive feedback from a
considerable audience.
