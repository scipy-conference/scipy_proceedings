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

    All physical and astronomical imaging observations are degraded by the limited angular
    resolution of the camera and telescope systems. The recovery of the true image is limited by
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
Any physical and astronomical imaging process is affected by the limited
angular resolution of the instrument or telescope. In addition the quality
of the resulting image is also degraded by background or instrumental
measurement noise and non-uniform exposure.
For short wavelengths and associated low intensities
of the signal, the imaging process consists in recording individual photons arriving from
a source of interest, often called "events".
This imaging process is typically for x-ray and gamma-ray telescopes, but is also found
in magnetic resonance imaging or fluorescence microscopy.
For each individual photon the incident direction and energy is
measured. Based on this information the event can be histogramed
into two dimensional data structures to form an actual image.

Because of the low intensities associated recording of individual
events, the measured signal follows Poisson statistics. This imposes
a non-linear relationship between the measured signal and true
underlying intensity as well as a coupling of the signal with the
measurement noise. Any statistically correct post-processing
or reconstruction method of thus requires are
careful treatment of the Poisson nature of the measured image.

To maximise the scientific use of the data is is often desired
to correct the degradation introduced by the imaging process.
Besides correction for non-uniform exposure and background
noise this also includes the correction for the "blurring"
introduced by the point spread function (PSF) of the
instrument, often called "deconvolution". Depending on whether
the PSF of the instrument is known, one distinguishes between
the "blind deconvolution" and "non blind deconvolution" process.
For astronomical observations the PSF can often either be
simulated, given a model of the detector or inferred
directly from the data by observing far distant objects,
which appear as point source to the instrument.

While in other branches of astronomy deconvolution methods are already part
of the standard analysis, such as the CLEAN algorithm for radio data, this
is not the case for x-ray and gamma-ray astronomy. As any deconvolution method
aims to enhance small scale structures in an image it becomes increasingly
hard to solve for the regime of low signal to noise ratio.
State of the art de-blurring algorithms based on deep convolutional neural networks
show exceptional results on conventional image data,
but they cannot be straightforwardly applied to astronomical counts data,
because of a lack of ground truth and training data in general.


Deconvolution Methods
---------------------

Richardson-Lucy
+++++++++++++++
One of the first methods for deconvolution of images with Poisson noise was
proposed by :cite:`Richardson1972` and :cite:`Lucy1974`. This method, named
after the original authors, is often known as the *Richardson & Lucy* (RL)
method. The method takes the fundamental statistical properties of the image
into account and describes the measurement process as a "forward fold" model.
The true image is then inferred from a likelihood based optimization procedure.

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
   :label: convolution

    \lambda_i = \sum_k x_i p_{i - k}

This operation is often called "forward modelling" or "forward folding" with the instrument response.
To obtain the most likely model given the data one searches a minimum of the total likelihood
function, or equivalently of :math:`\mathcal{C}`. This high dimensional optimization problem
can be solved by a classic gradient decent approach. Assuming the pixels values :math:`x_i`
of the true image as independent parameters, one can take the derivative of the Eq. :ref:`cash`
with respect to the individual :math:`x_i`. This way one obtains a rule for how to update the
current set pixels :math:`\mathbf{x}_n` in each iteration of the optimization:

.. math::
   :label: rl

    \mathbf{x}_{n + 1}  = \mathbf{x}_{n} -\alpha \cdot \frac{\partial \mathcal{C}\left( \mathbf{d} | \mathbf{x} \right)}{\partial x_i}

Where :math:`\alpha` is a factor to define the step size. It was shown by :cite:`Richardson1972`
that this converges to a maximum likelihood solution of Eq. :ref:`cash`. This method
is in general equivalent to the gradient decent and backpropagation methods used in
modern machine learning techniques. A Python implementation of the standard RL method
is available e.g. in the `Scikit-Image` package :cite:`skimage`. Instead of the gradient
decent based optimization it is also possible to sample from the likelihood function using
a simple Metropolis-Hastings approach. This is demonstrated in one of the *Pylira* online
tutorials (`Introduction to Deconvolution using MCMC Methods <https://pylira.readthedocs.io/en/latest/pylira/user/tutorials/notebooks/mcmc-deconvolution-intro.html>`__).

While technically the RL method converges to a maximum likelihood solution, it mostly
still results in poorly restored images, especially if extended emission regions are
present in the image. Because of the PSF convolution an extended emission region
can decompose into multiple nearby point sources and still lead to good model prediction,
when compared with the data. Those almost equally good solutions correspond
to many narrow local minima or "spikes" in the global likelihood surface. Depending
on the start estimate for the reconstructed image :math:`\mathbf{x}` the RL method will follow
the steepest gradient and converge towards the nearest narrow local minimum.
This problem has been described by multiple authors such as :cite:`Reeves1994`
and :cite:`Fish95`.


The LIRA method
+++++++++++++++
A solution to this problem was proposed in :cite:`Esch2004`.
Based on the maximum likelihood formulation the model function
can fist be be extended by taking into account the non uniform
exposure :math:`e_i` and a background estimate :math:`b_i`:

.. math::
   :label: convolution

    \lambda_i = \sum_k (e_i \cdot x_i) p_{i - k} + b_i

And by introducing a multi-scale prior to the likelihood term
in Eq. :ref:`cash`.




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


Simple Analysis Example
+++++++++++++++++++++++
*Pylira* was designed to offer a simple Python class based user interface,
which allow for a short learning curve of using the package, given that
users are familiar with Python in general and optionally *Numpy* and *Astropy*.
A typical complete usage example of the *Pylira* package is shown in the following:


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

    # plot pixel traces, result shown in Figure 1
    result.plot_parameter_traces()

    # plot pixel traces, result shown in Figure 2
    result.plot_pixel_traces_region(
        center_pix=(16, 16), radius_pix=3
    )


The main interface is exposed via the :code:`LIRADeconvolver` class, which takes the configuration of
the algorithm on initialisation. The data, which represented by a simple Python :code:`dict` data structure,
contains a :code:`"counts"`, :code:`"psf"` and optionally :code:`"exposure"` and :code:`"background"` array.
The dataset is then passed to the :code:`LIRADeconvolver.run()` method to execute the deconvolution.
The result is a :code:`LIRADeconvolverResult` object, which features the possibility to write the
result as a *FITS* file, as well as to inspect the result with diagnostic plots.


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
   while estimating the mean.  :label:`diagnosis1`


*Pylira* relies on an MCMC sampling approach to sample a series of reconstructed images from the posterior
likelihood defined by Eq. :ref:`post`. Along with the sampling it marginalises over the smoothing
hyper-parameters and optimizes them in the same process. To diagnose the validity of the results it is
important to visualise the sampling traces of both the sampled images as well as hyper-parameters.

Fig. :ref:`diagnosis1` shows one typical diagnostics plot created by the code example above.
In a multi-panel figure user can inspect the traces of the total log-posteriror as well as the
traces of th smoothing parameters. Each panel corresponds smoothing hyper parameter
introduced for each level of the multi-scale representation of the reconstructed image.
The figure also shows the mean value along with the :math:`1~\sigma` error
region. In this case the algorithm show stable convergence after a burn-in phase of approximately 200
iterations for the log-posterior as well as all of the multi-scale smoothing parameters.


.. figure:: images/pylira-diagnosis-pixel.pdf
   :scale: 60%
   :align: center
   :figclass: w

   The curves show the traces of value the pixel of interest for a simulated point source and its neighboring
   pixels (see code example). The image on the left shows the posterior mean. The white circle in the image
   shows the circular region defining the neighboring pixels. The blue line on the right plot shows the trace
   of the pixel of interest. The solid horizontal orange lines show the mean value, the shaded orange area
   the :math:`1~\sigma` error region. The burn in phase is shown in transparent blue and ignored while computing
   the mean. The shaded gray lines show the traces of the neighboring pixels.  :label:`diagnosis2`


Another useful diagnostic plot is shown in Fig. :ref:`diagnosis2`. The plot shows the
image sampling trace for a single pixel of interest and its surrounding circular region of interest.
This visualisation allows user to asses the stability of a small region in the image
e.g. an astronomical point source during the MCMC sampling process. Due to the correlation with
neighbouring pixels the actual value of a pixel might vary in the sampling process, which appears
as "dips" in the trace of the pixel of interested and anti-correlated "peaks" in the one or mutiple
of the surrounding pixels. In the this example a stable state of the pixels of interest
is reached after approximately 1000 iterations.



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


