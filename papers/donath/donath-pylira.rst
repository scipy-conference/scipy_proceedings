:author: Axel Donath
:email: axel.donath@cfa.harvard.edu
:institution: Center for Astrophysics | Harvard & Smithsonian
:orcid: 0000-0003-4568-7005

:author: Aneta Siemiginowska
:email: asiemiginowska@cfa.harvard.edu
:institution: Center for Astrophysics | Harvard & Smithsonian
:orcid: 0000-0002-0905-7375

:author: Vinay Kashyap
:email: vkashyap@cfa.harvard.edu
:institution: Center for Astrophysics | Harvard & Smithsonian
:orcid: 0000-0002-3869-7996

:author: Douglas Burke
:email: dburke@cfa.harvard.edu
:institution: Center for Astrophysics | Harvard & Smithsonian

:author: Karthik Reddy Solipuram
:institution: University of Maryland Baltimore County

:author: David van Dyke
:institution: Imperial College London

:bibliography: mybib

:video: -

----------------------------------------------------------------
Pylira: deconvolution of images in the presence of Poisson noise
----------------------------------------------------------------

.. class:: abstract

    All physical and astronomical imaging observations are affected by the limited angular
    resolution of the camera and telescope systems, and the recovery of the true image is limited by
    both how well the instrument characteristics are known and  by the magnitude of measurement noise.
    In the case of a high signal to noise ratio data, the image can be sharpened or “deconvolved” robustly
    by using established standard methods such as the Wiener Filter or the Richardson-Lucy method. More recently,
    given the existence of when sufficient training data are available, convolutional neural networks have also been
    shown to be very effective at this task. The situation changes for rare sparse data and the low signal to noise regime,
    where deconvolution leads inevitably to an amplification of noise. However the results of classical
    methods can be improved considerably by making use of prior assumptions and / or modern machine learning inspired
    techniques, such as k-fold cross validation and stochastic gradient decent. In this contribution we give a brief
    overview and comparison of existing methods in the Python scientific ecosystem and apply them to low counts astronomical
    data. At the same time we present This improved method is available in a new python package called Pylira :
    a new Python package dedicated to solving the deconvolution problem in the presence of Poisson noise.



.. class:: keywords

   deconvolution, point spread function, poisson, low counts, x-ray, gamma-ray

Introduction
------------
For vey low fluxes one enters the Poisson regime.

Careful treatment of the statistics.

x-ray and gamma-ray astronomy, but also microscopy...



Deconvolution Methods
---------------------

Richardson-Lucy
+++++++++++++++

An alternative approach to this analysis challenge is the use of deconvolution methods. While in other branches of astronomy
similar methods are already part of the standard analysis, such as the CLEAN algorithm for radio data, this is not the case
for gamma-ray astronomy. As any deconvolution method aims to enhance small scale structures in an image it becomes increasingly
hard to solve for the regime of low signal to noise ratio. The standard maximum likelihood solution accounting for the Poisson
statistics is known as the *Richardson-Lucy* (RL) method. However the method considers each pixel of an image as an independent
parameter and therefor tends to yield non-smooth and fragmented results, depending on the maximum number of iterations allowed. A solution
to this problem was proposed in :cite:`Esch2004`, by using a fully Bayesian treatment and introducing a multi-scale prior assumption.
State of the art de-blurring algorithms based on deep convolutional neural networks show exceptional results on conventional image data,
but they cannot be straightforwardly applied to astronomical gamma-ray data, because of a lack of ground truth and training data in general.

The *Cash* statistics for the :math:`k`-th dataset:

.. math::
   :label: cash

   \mathcal{C}_k = \sum_i M_i - D_i \log{M_i}


Extended Richardson-Lucy
++++++++++++++++++++++++
Based on the maximum likelihood formulation the model function can be extended
by taking into account the exposure, a baseline and multiple measurementof the
same underlying true flux distribution.

The *Cash* statistics for the :math:`k`-th dataset:

.. math::
   :label: cash

   \mathcal{C}_k = \sum_i M_i - D_i \log{M_i}

Where the individual :math:`M_i` are given by:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots


Where :math:`E` is the exposure, :math:`P` the PSF and :math:`B` the optional baseline model.

The `Cash` statistics Eq. :ref:`cash`


LIRA Multiscale Prior
+++++++++++++++++++++


.. math::
   :label: cash

   \mathcal{C}_k = \sum_i M_i - D_i \log{M_i}




The Pylira Package
------------------

Dependencies & Development
++++++++++++++++++++++++++

The *Pylira* package is a thin Python wrapper around the original *LIRA* implementation provided by
the authors of :cite:`Connors2011`. The original algorithm was implemented in *C* and made available
as a package to the *R Language* :cite:`rmath`. Thus the implementation depends on the *RMath* library,
which is still a required dependency to *Pylira*.
The Python wrapper was built using the *Pybind11* :cite:`pybind11` package. For the data handling *Pylira*
relies on *Numpy* :cite:`numpy` and *Astropy* :cite:`Astropy2018` for the *FITS* serialisation. The (interactive)
plotting functionality is achieved via *Matplotlib* :cite:`matplotlib` and *Ipywidgets* :cite:`ipywidgets`,
which are both optional dependencies. *Pylira* is openly developed on Github  at `https://github.com/astrostat/pylira <https://github.com/astrostat/pylira>`__.
It relies on *GitHub Actions* as a continuous integration service and uses the *Read the Docs* service
to build and deploy the documentation. The online documentation can be found on `https://pylira.readthedocs.io <https://pylira.readthedocs.io>`__.
*Pylira* implements a set of unit tests to assure compatibility and reproducibility of the
results with different versions of the dependencies and across different platforms.
As *Pylira* relies on random sampling for the MCMC process an exact reproducibility
of results is hard on different platforms, however the agreement of results is at least
guaranteed in the statistical limit of drawing many samples.

Installation
++++++++++++
*Pylira* is avaliable via the Python package index (`pypi.org <https://pypi.org/project/pylira/>`__),
currently at version 0.1. As *Pylira* still depends on the *RMath* library, it si required to install
this first. So the recommended way to install Pylira is on *MacOS* is:

.. code-block:: bash
   :linenos:

    $ brew install r
    $ pip install pylira

On *Linux* the *RMath* dependency can be installed using standard package managers:

.. code-block:: bash
   :linenos:

    $ sudo apt-get install r-base-dev r-base r-mathlib
    $ pip install pylira

For more detailed instructions see `Pylira installation instructions <https://pylira.readthedocs.io/en/latest/pylira/index.html#installation>`__.


Analysis Examples
+++++++++++++++++

.. figure:: images/pylira-chandra-gc.pdf
   :scale: 70%
   :figclass: w

   Pylira applied to Chandra data from the Galactic center region, using the observation IDs
   *4684* and *4684*. The image on the left shows the raw observed counts between
   0.5 and 7 keV. The image on the right shows the deconvolved version. The LIRA hyperprior
   values where chosen as *ms\_al\_kap1=1, ms\_al\_kap2=0.02, ms\_al\_kap3=1*.
   No baseline background model was taken into account.

.. figure:: images/pylira-fermi-gc.pdf
   :scale: 70%
   :figclass: w

   Pylira applied to Chandra data from the Galactic center region, using the observation IDs
   *4684* and *4684*. The image on the left shows the raw observed counts between
   0.5 and 7 keV. The image on the right shows the deconvolved version. The LIRA hyperprior
   values where chosen as *ms\_al\_kap1=1, ms\_al\_kap2=0.02, ms\_al\_kap3=1*.
   No baseline background model was taken into account.


The main API is exposed via the :code:`LIRADeconvolver` class, which takes the configuration of
the algorithm. The data which represented by a simple Python :code:`dict` data structure contains
a :code:`"counts"`, :code:`"psf"` and optionally :code:`"exposure"` and :code:`"background"` array.
The datasetis then passed to the :code:`LIRADeconvolver.run()` method to execute the deconvolution.
The result is a :code:`LIRADeconvolverResult` object, which features the possibility to write the
result as a *FITS* file, as well as to inspect the result with diagnostic plots.

.. code-block:: python
   :linenos:

    import numpy as np
    from pylira import LIRADeconvolver
    from pylira.data import point_source_gauss_psf

    # create example dataset
    data = point_source_gauss_psf()

    # define initial flux image
    data["flux_init"] = data["flux"]
    deconvolve = LIRADeconvolver(alpha_init=np.ones(5))

    result = deconvolve.run(data=data)

    # plot pixel traces, result shown in Figure 3
    result.plot_parameter_traces()

    # plot pixel traces, result shown in Figure 4
    result.plot_pixel_traces_region(
        center_pix=(16, 16), radius_pix=3
    )

Diagnostic Plots
++++++++++++++++


.. figure:: images/pylira-diagnosis.pdf
   :scale: 70%
   :align: center
   :figclass: w

   The curves show the traces of the log posterior
   value as well as traces of the values of the prior parameter values. The *SmoothingparamN* parameters
   correspond to the smoothing parameters per multi-scale level. The solid horizontal orange lines show the mean
   value, the shaded orange area the :math:`1~\sigma` error region. The burn in phase is shown transparent and ignored
   while estimating the mean.


.. figure:: images/pylira-diagnosis-pixel.pdf
   :scale: 60%
   :align: center
   :figclass: w

   The curves show the traces of value the pixel of interest for a simulated point source and its neighboring
   pixels (see code example). The image on the left shows the posterior mean. The white circle in the image
   shows the circular region defining the neighboring pixels. The blue line on the right plot shows the trace
   of the pixel of interest. The solid horizontal orange lines show the mean value, the shaded orange area
   the :math:`1~\sigma` error region. The burn in phase is shown in transparent blue and ignored while computing
   the mean. The shaded gray lines show the traces of the neighboring pixels.

Test Datasets
+++++++++++++
Describe test datasets here...


Summary & Outlook
-----------------
The *Pylira* package provides Python wrappers for the LIRA algorithm. It allows to deconvolve low-counts data
following Poisson statistics using a Bayesian sampling approach and a multi-scale smoothing prior assumption.
The results can be easily written to FITS files and inspected by plotting the trace of the sampling process.
This allows to check for general convergence as well as pixel to pixel correlations for selected regions of interest.
In the future the package will be extended to support distributed computing, more flexible prior definitions and to
account for systematic errors on the PSF.


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


