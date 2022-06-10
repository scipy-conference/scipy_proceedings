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

:author: David van Dyk
:institution: Imperial College London

:bibliography: mybib

:video: -

----------------------------------------------------------------
Pylira: deconvolution of images in the presence of Poisson noise
----------------------------------------------------------------

.. class:: abstract

    All physical and astronomical imaging observations are degraded by the finite angular
    resolution of the camera and telescope systems. The recovery of the true image is limited by
    both how well the instrument characteristics are known and by the magnitude of measurement noise.
    In the case of a high signal to noise ratio data, the image can be sharpened or “deconvolved” robustly
    by using established standard methods such as the Richardson-Lucy method. However, the situation changes
    for sparse data and the low signal to noise regime, such as those frequently encountered in
    X-ray and gamma-ray astronomy, where deconvolution leads inevitably to an amplification
    of noise and poorly reconstructed images. However, the results in this regime can be improved
    by making use of physically meaningful prior assumptions and statistically principled
    modeling techniques. One proposed method is the LIRA algorithm, which
    requires smoothness of the reconstructed image at multiple scales. In this contribution we
    introduce a new python package called *Pylira*, which exposes the original C implementation
    of the LIRA algorithm to Python users. We briefly describe the package structure, development
    setup and show a Chandra as well as Fermi-LAT analysis example.



.. class:: keywords

   deconvolution, point spread function, poisson, low counts, X-ray, gamma-ray

Introduction
------------
Any physical and astronomical imaging process is affected by the limited
angular resolution of the instrument or telescope. In addition, the quality
of the resulting image is also degraded by background or instrumental
measurement noise and non-uniform exposure.
For short wavelengths and associated low intensities
of the signal, the imaging process consists of recording individual photons
(often called "events") originating from a source of interest.
This imaging process is typical for X-ray and gamma-ray telescopes, but is also found
in magnetic resonance imaging or fluorescence microscopy.
For each individual photon, the incident direction, energy
and arrival time is measured. Based on this information the
event can be binned into two dimensional data structures to
form an actual image.

As a consequence of the low intensities associated to the recording of individual events,
the measured signal follows Poisson statistics. This imposes
a non-linear relationship between the measured signal and true
underlying intensity as well as a coupling of the signal intensity
to the the signal variance. Any statistically correct post-processing
or reconstruction method of thus requires a careful treatment of
the Poisson nature of the measured image.

To maximise the scientific use of the data, it is often desired
to correct the degradation introduced by the imaging process.
Besides correction for non-uniform exposure and background
noise this also includes the correction for the "blurring"
introduced by the point spread function (PSF) of the
instrument. Where the latter process is often called "deconvolution".
Depending on whether the PSF of the instrument is known,
one distinguishes between the "blind deconvolution" and
"non blind deconvolution" process. For astronomical
observations the PSF can often either be
simulated, given a model of the telescope and detector, or inferred
directly from the data by observing far distant objects,
which appear as a point source to the instrument.

While in other branches of astronomy deconvolution methods are already part
of the standard analysis, such as the CLEAN algorithm for radio data, developed
by :cite:`Hogbom1974`, this
is not the case for X-ray and gamma-ray astronomy. As any deconvolution method
aims to enhance small scale structures in an image it becomes increasingly
hard to solve for the regime of low signal to noise ratio, where small
scale structures are stronger affected by noise.


The Deconvolution Problem
-------------------------

Basic Statistical Model
+++++++++++++++++++++++
Assuming the noise in each pixel :math:`d_i` in the recorded counts image
follows a Poisson distribution, the total likelihood of obtaining the
measured image from a model image of the expected counts :math:`\lambda_i` with
:math:`N` pixels is given by:

.. math::
   :label: poisson

   \mathcal{L}\left( \mathbf{d} | \mathbf{\lambda} \right) = \prod_i^N \frac{{e^{ - d_i } \lambda_i ^ {d_i}}}{{d_i!}}

By taking the logarithm and dropping the constant terms one can transform the
product into a sum over pixels, which is also often called the *Cash* :cite:`Cash1979`
fit statistics:

.. math::
   :label: cash

   \mathcal{C}\left( \mathbf{d} | \mathbf{\lambda} \right) = \sum_i^N \lambda_i - d_i \log{\lambda_i}

Where the expected counts :math:`\lambda_i` are given by the convolution of the true underlying
flux distribution :math:`x_i` with the PSF :math:`p_k`:

.. math::
   :label: simple_model

   \lambda_i = \sum_k x_i p_{i - k}

This operation is often called "forward modelling" or "forward folding" with the instrument response.

Richardson Lucy (RL)
++++++++++++++++++++
To obtain the most likely model given the data one searches a minimum of the total likelihood
function, or equivalently of :math:`\mathcal{C}`. This high dimensional optimization problem
can e.g., be solved by a classic gradient decent approach. Assuming the pixels values :math:`x_i`
of the true image as independent parameters, one can take the derivative of the Eq.  :ref:`cash`
with respect to the individual :math:`x_i`. This way one obtains a rule for how to update the
current set pixels :math:`\mathbf{x}_n` in each iteration of the optimization:

.. math::
   :label: rl

   \mathbf{x}_{n + 1}  = \mathbf{x}_{n} -\alpha \cdot \frac{\partial \mathcal{C}\left( \mathbf{d} | \mathbf{x} \right)}{\partial x_i}

Where :math:`\alpha` is a factor to define the step size. This method is in general
equivalent to the gradient decent and backpropagation methods used in modern machine
learning techniques. This basic principle of solving the deconvolution problem for
images with Poisson noise was proposed by :cite:`Richardson1972` and :cite:`Lucy1974`.
Their method, named after the original authors, is often known as the *Richardson & Lucy* (RL)
method. It was shown by :cite:`Richardson1972` that this converges to a maximum
likelihood solution of Eq.  :ref:`cash`. A Python implementation of the standard RL method
is available e.g. in the `Scikit-Image` package :cite:`skimage`.

Instead of the iterative, gradient decent based optimization it is also possible to sample from
the likelihood function using a simple Metropolis-Hastings approach. This is demonstrated
in one of the *Pylira* online tutorials (`Introduction to Deconvolution using MCMC Methods <https://pylira.readthedocs.io/en/latest/pylira/user/tutorials/notebooks/mcmc-deconvolution-intro.html>`__).


RL Reconstruction Quality
+++++++++++++++++++++++++
While technically the RL method converges to a maximum likelihood solution, it mostly
still results in poorly restored images, especially if extended emission regions are
present in the image. The problem is illustrated in Fig.~:ref:`rl` using
a simulated example image. While for a low number of iterations the RL method
still results in a smooth intensity distribution, the structure of the image
decomposes more and more into a set of point-like sources with growing number
of iterations.

Because of the PSF convolution an extended emission region
can decompose into multiple nearby point sources and still lead to good model prediction,
when compared with the data. Those almost equally good solutions correspond
to many narrow local minima or "spikes" in the global likelihood surface. Depending
on the start estimate for the reconstructed image :math:`\mathbf{x}` the RL method will follow
the steepest gradient and converge towards the nearest narrow local minimum.
This problem has been described by multiple authors, such as :cite:`Reeves1994`
and :cite:`Fish95`.


.. figure:: images/richardson-lucy.png
   :scale: 90%
   :figclass: bht

   The images show the result of the RL algorithm applied to a simulated example dataset
   with varying numbers of iterations. The image in the upper left shows the simulated
   counts. Those have been derived from the ground truth (upper mid) by convolving with
   a Gaussian PSF of width :math:`\sigma=3~\mathrm{pix}` and applying Poisson noise to
   it. The illustration uses the implementation of the RL algorithm from the `Scikit-Image`
   package :cite:`skimage`.  :label:`rl`


LIRA Multiscale Prior
+++++++++++++++++++++
One possible solution to this problem was described in :cite:`Esch2004`
and :cite:`Connors2011`. First the simple forward folded model described
in Eq. ~:ref:`simple_model` can be extended by taking into account the
non-uniform exposure :math:`e_i` and a background estimate :math:`b_i`:

.. math::
   :label: model

   \lambda_i = \sum_k \left( e_i \cdot (x_i + b_i) \right) p_{i - k}

The background :math:`b_i` can be more generally understood
as a "baseline" image and thus e.g. include known structures,
which are not of interested for the deconvolution process.

Second the authors proposed to extend the Poisson log-likelihood
function (Equation :ref:`cash`) by a log-prior term that controls the
smoothness of the reconstructed image on multiple spatial scales.
For this the image :math:`x_i` is transformed into a multi-scale representation:
starting from the full resolution the image is divided into groups
of 2x2 pixels :math:`Q_k`. Each of the groups of 2x2 pixels is
then divided by their total sum. This results in an image containing
the "split proportions" with respect to the image down-sized
by a factor of two. The process is continued to further reduce the
resolution of the image until only one pixel, containing the total
sum of the full-resolution image is left. This multiscale
decomposition is illustrated in Figure~:ref:`ms-levels`.

For each of the 2x2 groups of the re-normalized images
a Dirichlet distribution is introduced as a prior:

.. math::
   :label: dirichlet

    \phi_k \propto \mathrm{Dirichlet}(\alpha_k, \alpha_k, \alpha_k, \alpha_k)

And multiplied across all 2x2 groups and resolution levels :math:`k`.
For each resolutionlevel a parameter :math:`\alpha_k` is introduced,
which represents the number of "prior counts" added to this resolution level,
equally to each pixel, which effectively results in a smoothing
of the image at the given resolution level. The distribution
of `\alpha` values at each resolution level is described
by a hyperprior distribution:

.. math::
   :label: hyperprior

    p(\alpha_k) = \exp{-\delta \alpha^3 / 3}

Resulting in a fully hierarchical Bayesian model. A complete more
detailed description of the prior definition is given in :cite:`Esch2004`.


.. figure:: images/ms-levels.png
   :scale: 80%
   :figclass: bht

   The image illustrates the multiscale decomposition used in the LIRA prior for
   a 4x4 pixels example image. Each quadrant of 2x2 sub-images is labelled with
   :math:`Q_N`. The sub-pixels in each quadrant are labelled :math:`\Lambda_{ij}`.
   :label:`ms-levels`

The problem is then solved by using a "hyprid" Gibbs MCMC sampling approach. For
each iteration of sampling an image :math:`x_i`, the :math:`\alpha_k` parameters are optimized
using a Newton method and thus "marginalized". After a "burn-in" phase the sampling
process typically reaches convergence and starts sampling from the
posterior distribution. The reconstructed image is then computed from the mean of the
posterior samples. As for each pixels a full distribution of its values is available,
the information can also be used to compute and associated error of the reconstructed
value. This is another main advantages over e.g. the standard RL algorithms.


The Pylira Package
------------------

Dependencies & Development
++++++++++++++++++++++++++

The *Pylira* package is a thin Python wrapper around the original *LIRA* implementation provided by
the authors of :cite:`Connors2011`. The original algorithm was implemented in *C* and made available
as a package for the *R Language* :cite:`rmath`. Thus the implementation depends on the *RMath* library,
which is still a required dependency to *Pylira*.
The Python wrapper was built using the *Pybind11* :cite:`pybind11` package, which allows to reduce
the code overhead introduced by the wrapper to a minimum. For the data handling *Pylira*
relies on *Numpy* :cite:`numpy` arrays for the serialisation to the *FITS*  data format
on *Astropy* :cite:`Astropy2018`. The (interactive)
plotting functionality is achieved via *Matplotlib* :cite:`matplotlib` and *Ipywidgets* :cite:`ipywidgets`,
which are both optional dependencies. *Pylira* is openly developed on Github  at `https://github.com/astrostat/pylira <https://github.com/astrostat/pylira>`__.
It relies on *GitHub Actions* as a continuous integration service and uses the *Read the Docs* service
to build and deploy the documentation. The online documentation can be found on `https://pylira.readthedocs.io <https://pylira.readthedocs.io>`__.
*Pylira* implements a set of unit tests to assure compatibility and reproducibility of the
results with different versions of the dependencies and across different platforms.
As *Pylira* relies on random sampling for the MCMC process an exact reproducibility
of results is hard to achieve on different platforms, however the agreement of results
is at least guaranteed in the statistical limit of drawing many samples.


Installation
++++++++++++
*Pylira* is available via the Python package index (`pypi.org <https://pypi.org/project/pylira/>`__),
currently at version 0.1. As *Pylira* still depends on the *RMath* library, it is required to install
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

API & Subpackages
+++++++++++++++++
*Pylira* is structured in multiple sub-packages. The :code:`pylira.src` module contains the original
C implementation and the *Pybind11* wrapper code. The :code:`pylira.core` sub-package
contains the main Python API, :code:`pylira.utils` includes utility functions for
plotting and serialisation. And :code:`pylira.data` implements multiple pre-defined
datasets for testing and tutorials.


Analysis Examples
-----------------

Simple Point Source
+++++++++++++++++++
*Pylira* was designed to offer a simple Python class based user interface,
which allows for a short learning curve of using the package, given that
users are familiar with Python in general and more specifically *Numpy*.
A typical complete usage example of the *Pylira* package is shown in the
following:


.. code-block:: python
   :linenos:

    import numpy as np
    from pylira import LIRADeconvolver
    from pylira.data import point_source_gauss_psf

    # create example dataset
    data = point_source_gauss_psf()

    # define initial flux image
    data["flux_init"] = data["flux"]

    deconvolve = LIRADeconvolver(
        n_iter_max=3_000,
        n_burn_in=500,
        alpha_init=np.ones(5)
    )

    result = deconvolve.run(data=data)

    # plot pixel traces, result shown in Figure 3
    result.plot_pixel_traces_region(
        center_pix=(16, 16), radius_pix=3
    )

    # plot pixel traces, result shown in Figure 4
    result.plot_parameter_traces()

    # finally serialise the result
    result.write("result.fits")


The main interface is exposed via the :code:`LIRADeconvolver` class, which takes the configuration of
the algorithm on initialisation. Typical configuration parameters include the total number of
iterations :code:`n_iter_max` and the number of "burn-in" iterations, to be excluded from the
posterior mean computation. The data, which represented by a simple Python :code:`dict` data structure,
contains a :code:`"counts"`, :code:`"psf"` and optionally :code:`"exposure"` and :code:`"background"` array.
The dataset is then passed to the :code:`LIRADeconvolver.run()` method to execute the deconvolution.
The result is a :code:`LIRADeconvolverResult` object, which features the possibility to write the
result as a *FITS* file, as well as to inspect the result with diagnostic plots. The result of
the computation is shown in the left panel of Fig. :ref:`diagnosis1`.


Diagnostic Plots
++++++++++++++++

.. figure:: images/pylira-diagnosis-pixel.pdf
   :scale: 60%
   :align: center
   :figclass: w

   The curves show the traces of value the pixel of interest for a simulated point source and its neighboring
   pixels (see code example). The image on the left shows the posterior mean. The white circle in the image
   shows the circular region defining the neighboring pixels. The blue line on the right plot shows the trace
   of the pixel of interest. The solid horizontal orange lines show the mean value, the shaded orange area
   the :math:`1~\sigma` error region. The burn in phase is shown in transparent blue and ignored while computing
   the mean. The shaded gray lines show the traces of the neighboring pixels.  :label:`diagnosis1`

To validate the quality of the results *Pylira* provide many built-in diagnostic plots.
One of these diagnostic plot is shown in the right panel of Fig. :ref:`diagnosis1`. The plot shows the
image sampling trace for a single pixel of interest and its surrounding circular region of interest.
This visualisation allows user to asses the stability of a small region in the image
e.g. an astronomical point source during the MCMC sampling process. Due to the correlation with
neighbouring pixels the actual value of a pixel might vary in the sampling process, which appears
as "dips" in the trace of the pixel of interested and anti-correlated "peaks" in the one or mutiple
of the surrounding pixels. In the this example a stable state of the pixels of interest
is reached after approximately 1000 iterations. This suggests the number of burn-in iterations, which
were defined beforehand, should be increased.


.. figure:: images/pylira-diagnosis.pdf
   :scale: 70%
   :align: center
   :figclass: w

   The curves show the traces of the log posterior
   value as well as traces of the values of the prior parameter values. The *SmoothingparamN* parameters
   correspond to the smoothing parameters :math:`\alpha_N` per multi-scale level. The solid horizontal orange lines show the mean
   value, the shaded orange area the :math:`1~\sigma` error region. The burn in phase is shown transparent and ignored
   while estimating the mean.  :label:`diagnosis2`

*Pylira* relies on an MCMC sampling approach to sample a series of reconstructed images from the posterior
likelihood defined by Eq. :ref:`cash`. Along with the sampling it marginalises over the smoothing
hyper-parameters and optimizes them in the same process. To diagnose the validity of the results it is
important to visualise the sampling traces of both the sampled images as well as hyper-parameters.

Figure :ref:`diagnosis2` shows another typical diagnostic plot created by the code example above.
In a multi-panel figure user can inspect the traces of the total log-posteriror as well as the
traces of th smoothing parameters. Each panel corresponds smoothing hyper parameter
introduced for each level of the multi-scale representation of the reconstructed image.
The figure also shows the mean value along with the :math:`1~\sigma` error
region. In this case the algorithm show stable convergence after a burn-in phase of approximately 200
iterations for the log-posterior as well as all of the multi-scale smoothing parameters.


Astronomical Analysis Examples
++++++++++++++++++++++++++++++

Both in the X-ray as well as gamma-ray regime the The Galactic Center is a complex emission
region. It shows point sources, extended sources as well as underlying diffuse emission and
thus represents a challenge for any astronomical data analysis. *Chandra* is a spaced based
X-ray observatory, which is in operation since 1999. It consists of nested cylindrical paraboloid
and hyperboloid surfaces, which form an imaging optical system for X-rays. In the focal plane
it has multiple instruments for different scientific purposes. This includes a high resolution
camera (HRC) and an Advanced CCD Imaging Spectrometer (ACIS). The typical angular resolution
is 0.5 arcsecond and the covered energy ranges from 0.1 - 10 keV.


Figure :ref:`chandra-gc` shows the result of the *Pylira* algorithm applied to Chandra data
of the Galactic center region between 0.5 and 7 keV. The PSF was obtained from simulations
using the official Chandra science tools *ciao 4.14* and the *simulate_psf* tool.
The algorithm achieves both an improved spatial resolution as well as a reduced noise
level and higher contrast of the image in the right panel compared to the unprocessed
counts data shown in the left panel.

.. figure:: images/pylira-chandra-gc.pdf
   :scale: 70%
   :figclass: w

   Pylira applied to Chandra data from the Galactic center region, using the observation IDs
   *4684* and *4684*. The image on the left shows the raw observed counts between
   0.5 and 7 keV. The image on the right shows the deconvolved version. The LIRA hyperprior
   values were chosen as *ms\_al\_kap1=1, ms\_al\_kap2=0.02, ms\_al\_kap3=1*.
   No baseline background model was taken into account. :label:`chandra-gc`
   TODO: include pixel size...

As second example we use data from the Fermi Large Area Telescope (LAT). The Fermi-LAT
is a satellite-based an imaging gamma-ray detector, which covers and energy range
of 20 MeV to >300 GeV. The angular resolution varies strongly with energy and ranges
from 0.1 to >10 degree [#]_.

Figure :ref:`fermi-gc` shows the result of the *Pylira*
algorithm applied to Fermi-LAT data above 1~GeV to the region around the Galactic Center. The PSF
was obtained form the official Fermitools v2.0.19 and the *gtpsf* tool.
First one can see that the algorithm achieves again a considerable improvement of the spatial resolution
compared to the raw counts. It clearly resolves multiple point sources left to the
the bright Galactic center source.

.. [#] https://www.slac.stanford.edu/exp/glast/groups/canda/lat_Performance.htm


.. figure:: images/pylira-fermi-gc.pdf
   :scale: 70%
   :figclass: w

   Pylira applied to Fermi-LAT data from the Galactic center region. The image on
   the left shows the raw measured counts between 5 and 1000 GeV. The image on the right
   shows the deconvolved version. The LIRA hyperprior values were chosen as
   *ms\_al\_kap1=1, ms\_al\_kap2=0.02, ms\_al\_kap3=1*. A baseline background model
   was not included. :label:`fermi-gc`


Summary & Outlook
-----------------
The *Pylira* package provides Python wrappers for the LIRA algorithm. It allows to deconvolve low-counts data
following Poisson statistics using a Bayesian sampling approach and a multi-scale smoothing prior assumption.
The results can be easily written to FITS files and inspected by plotting the trace of the sampling process.
This allows to check for general convergence as well as pixel to pixel correlations for selected regions of
interest. The package is openly developed on GitHub and includes tests and documentation, such that it can be
maintained and improved in future, while ensuring consistency of the results. It comes with multiple built-in
test datasets and explanatory tutorials in form of Jupyter notebooks. Future plans include the support
for parallelisation or distributed computing, more flexible prior definitions and the
possibility to account for systematic errors on the PSF during the sampling process.

