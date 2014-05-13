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

  We present an implementation of such an algorithm in the ``yt`` volumetric analysis 
  software package.

.. class:: keywords

  astronomical observations, visualization

Introduction
------------
 
Method
------

Emission Models
===============

One of the most common sources of X-ray emission is that from a low-density, high-temperature, thermal plasma, such as that found in the solar corona, supernova remnants, "early-type" galaxies, galaxy groups, and galaxy clusters. The specific photon emissivity associated with a given density, temperature, and metallicity of such a plasma is given by 

.. math::
  :label: emissivity

  \epsilon_E^\gamma = n_en_H\Lambda(E,T,Z)~{\rm photons~s^{-1}~cm^{-3}}

where :math:`n_e` and :math:`n_H` are the electron and proton number densities in :math:`{\rm cm^{-3}}` and :math:`\Lambda(E,T,Z)` is the spectral. The dominant contributions to A number of models for the emissivity of an optically-thin, fully-ionized plasma have been developed, including . 

However, X-ray emission may arise from a variety of physical processes and sources. For example, cosmic-ray electrons in galaxy clusters produce a power-law spectrum of X-ray emission at high energies via inverse-Compton scattering of the cosmic microwave background. The flexibility of our approach allows us to implement one or several different models for the X-ray emission as the situation requires. 


Regardless of the emission mechanism, we 

Implementation
--------------

The model described here has been implemented in ``yt`` [], a Python-based visualization and analysis toolkit for volumetric data. ``yt``'s strength is that it 

Example
-------

Here we present a workable example of creating simulated X-ray events using ``yt``'s photon simulator. The code implemented here is available as a Python script at :

.. code-block:: python

  import yt
  from yt.analysis_modules.photon_simulator.api import *
  from yt.utilities.cosmology import Cosmology

We're going to load up an Athena dataset of a galaxy cluster core, which can be downloaded from http://yt-project.org/data:

.. code-block:: python

  ds = yt.load("MHDSloshing/virgo_low_res.0054.vtk",
               parameters={"time_unit":(1.0,"Myr"),
                           "length_unit":(1.0,"Mpc"),
                           "mass_unit":(1.0e14,"Msun")}) 

Slices through the density and temperature of the simulation dataset are shown in Figure . We will create the photons from a spherical region centered on the domain center, with a radius of 250 kpc:

.. code-block:: python

  sp = ds.sphere("c", (250., "kpc"))
  
This will serve as our ``data_source`` that we will use later. Next, we
need to create the ``SpectralModel`` instance that will determine how
the data in the grid cells will generate photons. A number of options are available, but we will use the ``XSpecThermalModel``, which allows one to
use any thermal model that is known to `XSPEC <https://heasarc.gsfc.nasa.gov/xanadu/xspec/>`_, such as ``"mekal"`` or ``"apec"``:

.. code-block:: python

  mekal_model = XSpecThermalModel("mekal", 0.01, 
                                  10.0, 2000)

This requires XSPEC and
`PyXspec <http://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/>`_ to
be installed. 

Now that we have our ``SpectralModel`` that gives us a spectrum, we need
to connect this model to a ``PhotonModel`` that will connect the field
data in the ``data_source`` to the spectral model to actually generate
photons. For thermal spectra, we have a special ``PhotonModel`` called
``ThermalPhotonModel``:

.. code-block:: python

  thermal_model = ThermalPhotonModel(apec_model, X_H=0.75, 
                                     Zmet=0.3)

Where we pass in the ``SpectralModel``, and can optionally set values for
the hydrogen mass fraction ``X_H`` and metallicity ``Z_met``, the latter of which may be a single floating-point value or the name of the ``yt`` field representing the spatially-dependent metallicity. Next, we need to specify "fiducial" values for the telescope collecting area, exposure time, and cosmological redshift. Since the initial photon generation will act as a source for Monte-Carlo sampling for more realistic values of these parameters later, we choose generous values so
that there will be a large number of photons to sample from. We also construct a ``Cosmology`` object:

.. code-block:: python

  A = 6000.
  exp_time = 4.0e5
  redshift = 0.05
  cosmo = Cosmology()

Now, we finally combine everything together and create a ``PhotonList``
instance:

.. code-block:: python

  photons = PhotonList.from_scratch(sp, redshift, A, 
                                    exp_time,
                                    thermal_model, 
                                    center="c",
                                    cosmology=cosmo)

At this point, the ``photons`` are distributed in the three-dimensional
space of the ``data_source``, with energies in the rest frame of the
plasma. Doppler and/or cosmological shifting of the photons will be
applied in the next step.

The ``photons`` can be saved to disk in an HDF5 file:

.. code-block:: python

  photons.write_h5_file("my_photons.h5")

Which is most useful if it takes a long time to generate the photons,
because a ``PhotonList`` can be created in-memory from the dataset
stored on disk:

.. code-block:: python

  photons = PhotonList.from_file("my_photons.h5")

This enables the creation of many simulated event sets, along different
projections, at different redshifts, with different exposure times, and
different instruments, with the same ``data_source``, without having to
repeat the expensive step of generating the photons.

Once a set of photons is generated, they can be projected along a line of sight to create a synthetic observation. First, if simulating galactic foreground absorption is desired,  it is necessary to set up a spectral model for the absorption coefficient, similar to the spectral model for the emitted photons set up previously. Here again, there are multiple 
options. Here, we use ``XSpecAbsorbModel``, which allows one to use any absorption model that XSpec is aware of that takes only the column density :math:`N_H` as input:

.. code-block:: python

  N_H = 0.1 
  abs_model = XSpecAbsorbModel("wabs", N_H) 

Now the photons may be projected. First, we choose a line-of-sight vector ``L``. Second, we adjust the exposure time and the redshift to more realistic values. Third, we'll pass in the absorption ``SpectrumModel``. Fourth, we'll specify a ``sky_center`` in RA, Dec on the sky in degrees.

.. code-block:: python

  ARF = "chandra_ACIS-S3_onaxis_arf.fits"
  RMF = "chandra_ACIS-S3_onaxis_rmf.fits"
  L = [0.0,0.0,1.0]
  events = photons.project_photons(L, exp_time_new=2.0e5, 
                                   redshift_new=0.07, 
                                   absorb_model=abs_model,
                                   sky_center=(187.5,12.333), 
                                   responses=[ARF,RMF])

some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots


The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +------------+----------------+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+


Perhaps we want to end off with a quote by Lao Tse:

  *Muddy water, let stand, becomes clear.*


References
----------


