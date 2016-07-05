:author: Brett Naul
:email: bnaul@berkeley.edu
:institution: University of California, Berkeley
:corresponding:

:author: Stéfan van der Walt
:email: stefanv@berkeley.edu
:institution: University of California, Berkeley

:author: Arien Crellin-Quick
:email: arien@berkeley.edu
:institution: University of California, Berkeley

:author: Joshua S. Bloom
:email: joshbloom@berkeley.edu
:institution: Lawrence Berkeley National Laboratory
:institution: University of California, Berkeley

:author: Fernando Pérez
:email: fperez@lbl.gov
:institution: Lawrence Berkeley National Laboratory
:institution: University of California, Berkeley

:bibliography: mybib

-----------------------------------------------------------
``cesium``: Machine Learning for Time Series Data in Python
-----------------------------------------------------------

.. class:: abstract

   Inference on time series data is a common requirement in many scientific
   disciplines, yet there are few resources available to domain scientists to
   easily, robustly, and repeatably build inference work flows on time series
   data: traditional statistical models of time series are often too rigid to
   explain complex time domain behavior, while popular machine learning packages
   require already-featurized dataset inputs. ``cesium`` is and end-to-end time
   series analysis framework, consisting of a Python library as well as a web
   front-end interface, that allows researchers to featurize raw data and apply
   modern machine learning techniques in a simple, easily reproducible, and
   extensible way. Users can apply out-of-the-box feature engineering workflows
   as well as save and replay their own analyses. Any steps taken in the front
   end can also be exported to an IPython notebook, so users can quickly iterate
   within the front end and then fine-tune their analysis using the more
   flexible back-end library.

.. class:: keywords

   time series, machine learning, reproducibile science

Introduction
============
From the reading of electroencephalograms (EEGs) to earthquake seismograms to
light curves of astronomical variable stars, gleaning insight from time series
data has been central to a broad range of scientific and medical disciplines.
When simple analytical thresholds or models suffice, technicians and experts can
be easily removed from the process of inspection and discovery by employing
custom algorithms. But when dynamical systems are not easily modeled (e.g.,
through physics-based models or standard regression techniques), classification
and anomaly detection have traditionally been reserved for the domain expert:
digitally recorded data are visually scanned to ascertain the nature of the time
variability and find important (perhaps life-threatening) outliers. *Does this
person have an irregular heartbeat? What type of supernova occurred in that
galaxy?* Even in the presence of sensor noise and intrinsic diversity of the
samples, well-trained domain specialists show a remarkable ability to make
discerning statements about the complex data.

In an era when more time series data are being collected than can be visually
inspected by domain experts, however, computational frameworks must necessarily
act as human surrogates. Capturing the subtleties that domain experts intuit in
time series data (and perhaps even besting the experts) is a non-trivial task.
In this respect, machine learning has already been used to great success in
several disciplines, including text classification, image retrieval,
segmentation of remote sensing data, internet traffic classification, video
analysis, and classification of medical data. Even if the results are similar,
some obvious advantages over human involvement are that machine learning
algorithms are tunable, repeatable, and deterministic. A computational framework
built with elasticity can scale, whereas experts (and even crowdsourcing)
cannot.

Despite the importance of time series in scientific research, there are few
resources available that allow domain scientists to easily build robust
computational inference workflows for their own time series data, let alone
data gathered more broadly in their field. The difficulties involved in
constructing such a framework can often greatly outweigh those of analyzing the
data itself: per :cite:`scu2014`:,
 
        It may be surprising to the academic community to know that only a tiny
        fraction of the code in many machine learning systems is actually doing
        "machine learning". When we recognize that a mature system might end up
        being (at most) 5% machine learning code and (at least) 95% glue code,
        reimplementation rather than reuse of a clumsy API looks like a much
        better strategy.

Even if a domain scientists works closely with machine learning experts, the
software engineering requirements are daunting. Being a modern data-driven
scientist should not, we believe, require an army of software engineers, machine
learning experts, statisticians and production operators. ``cesium`` was created
to allow domain experts to focus on the questions at hand rather than needing to
architect a complete engineering project.

The analysis workflow of ``cesium`` can be used in two forms: a web front end
which allows researchers to upload their data, perform analyses, and visualize
their models all within the browser; and a Python library which exposes more
flexible interfaces to the same analysis tools. The web front end is designed to
handle many of the more cumbersome aspects of machine learning analysis,
including data uploading and management, scaling of computational resources, and
tracking of results from previous experiments. The Python library is used within
the web back end for the main steps of the analysis workflow: extracting
features from raw time series, building models from these features, and
generating predictions. The library also supplies data structures for storing
time series (including support for irregularly-sampled time series and
measurement errors), features, and other relevant metadata.

In the next section, we'll describe a few motivating examples of scientific time
series analysis problems. The subsequent sections describe in detail the
``cesium`` library and web front end, including the different pieces of
functionality provided and various design questions and decisions that arose
during the development process. Finally, we present an end-to-end analysis of an
EEG seizure dataset, first using the Python library and then via the web front
end.

Example time series machine learning problems
=============================================
``cesium`` was designed with several time series inference challenges across various
scientific disciplines in mind.
 
1. **Astronomical time series classification.** The Large Synoptic Survey
   Telescope (LSST), beginning in 2020, will survey the entire night’s sky every
   few days producing high-quality time series data on approximately 800 million
   variable sources and transient events :cite:`lsst2009`. Much of the best
   science in the time domain (e.g., the discovery of the accelerating universe
   and dark energy using Type Ia supernovae :cite:`perlmutter1999,riess1998`)
   consists of first identifying possible phenomena of interest using broad data
   mining approaches and following up by collecting more detailed data using
   other, more precise observational tools. For many transient events, the time
   scale during which observations can be collected can be on the order of days
   or hours. Not knowing which of the millions of variable sources to examine
   more closely with larger telescopes and specialized instruments is tantamount
   to not having discovered those sources in the first place. Discoveries must
   be identified quickly or in real time so that informed decisions can be made
   about how best to allocate additional observational resources.

.. figure:: cesium-astro

   Typical data for a classification task on variable stars from the All Sky
   Automated Survey; shown are flux measurements for three stars
   irregularly sampled in time :cite:`richards2012`. :label:`astro`

2. **Neuroscience time series classification.** The study of
   neural systems presents a wide variety of challenges in time series analysis,
   made more pressing by the growing volume of high-quality, heterogeneous
   sensor data that cannot be effectively inspected visually. Indeed,
   neuroscience experiments now produce vast amounts of time series data that
   can have entirely different structures, spatial resolution, and temporal
   resolution, depending on the recording technique. Ultimately, we wish to
   connect complex recorded output to high-level cognition patterns: *How did
   that subject formulate the image of what they were seeing? What motion were
   they trying to instigate on their body? What are they thinking?* Given the
   prevalence of these various recording methods in experimental neuroscience,
   much of our data will arrive as time series but in a multitude of different
   possible forms: from a few channels with high spatial localization to
   hundreds of channels with unknown spatial specificity; from low to very high
   sampling rates; and with various types of confounds and recording artifacts
   specific to each recording method. Furthermore, in every case the volumes of
   available data are rapidly increasing. The neuroscience community is turning
   to the use of large-scale machine learning tools to extract insight from
   these complex datasets :cite:`lotte2007`. However, the community lacks tools
   to validate and compare data analysis approaches in a robust, efficient and
   reproducible manner: even recent expert reviews on the matter leave many of
   these critical methodological questions open for the user to explore in an ad
   hoc way and with little principled guidance :cite:`perez2007`. In addition,
   the problems of feature selection vary across data modalities (EEG, fMRI,
   etc.), yet these different modalities offer complementary views on the same
   underlying phenomena.

.. figure:: cesium-eeg

   EEG signals from patients with epilepsy. :label:`eeg`

3. **Earthquake detection, characterization and warning.** Earthquake early
   warning (EEW) systems are currently in operation in Japan, Mexico, Turkey,
   Taiwan and Romania :cite:`allen2009` and are under development in the US
   :cite:`brown2011`. These first-generation systems, most notably in Japan,
   have employed sophisticated remote sensors, real-time connectivity to major
   broadcast outlets (such as TV and radio), and have a growing resumé of
   successful rapid assessment of threat levels to populations and industry.
   Traditionally these warning systems trigger from data obtained by
   high-quality seismic networks with sensors placed every \~10 km. The
   algorithms used to detect earthquakes are based on methodologies developed in
   the 1960s. However, today’s accelerometers are embedded in many consumer
   electronics including computers and smartphones. There is tremendous
   potential to improve earthquake detection methods using streaming
   classification analysis both using traditional network data and also
   harnessing massive data from consumer electronics. The Big Data challenges in
   the statistical modeling of such streams arise due to lower-quality detectors
   in higher noise environments and the requirement that events be triggered on
   in real-time (thus precluding more sophisticated, computationally demanding
   algorithms). The main tension in determining an optimal triggering procedure
   is the mitigation of false positives (spurious triggering) versus false
   negatives (failure to trigger on a real event). The importance of these
   questions for national emergency response is clear: faster and more robust
   classification of earthquakes from noisy sensor data can increase warning
   times, mitigate damage to resources, and even prevent loss of life.

.. figure:: cesium-seismo

   Seismograms from a small California earthquake on April 6, 2016. :label:`seismo`

Simple and reproducible workflows
=================================
In recent years, there has been rapid growth in the availability of open-source
tools that implement a wide variety of machine learning algorithms: packages
within the R :cite:`team2013` and Python programming languages
:cite:`pedregosa2011`, standalone Java-based packages such as Moa
:cite:`bifet2010` and Weka :cite:`hall2009`, and online webservices such as the
Google Prediction API, to name a few. To a domain scientist that does not have a
formal training in machine learning, however, the availability of such packages
is both a blessing and a curse. On one hand, everyone now has access to nearly
every conceivable machine learning algorithm. At the same time, these algorithms
tend to be black boxes with a few enigmatic knobs to turn. A domain scientist
may rightfully ask just which of the many algorithms to use, which parameters to
tune, and what the results actually mean.

The goal of ``cesium`` is to simplify the analysis pipeline so that scientists
can spend less time solving technical computing problems and more time answering
scientific questions. ``cesium`` includes a number of out-of-the-box feature
engineering workflows, such as periodogram analysis, that transform raw time
series data to extract signal from the noise. By recording the inputs,
parameters, and outputs of previous experiments, ``cesium`` allows researchers
to answer new questions that arise out of previous lines of inquiry.  Saved
``cesium`` workflows can be can be applied to new data as it arrives and shared
with collaborators or published so that others may apply the same
beginning-to-end analysis for their own data.

For advanced users or users who wish to delve into the source code corresponding
to a workflow producing through the ``cesium`` web front end, we have provided
the ability to produce an ``IPython`` notebook from a saved workflow with a
single click. While our goal is to have the front end to be as robust and
flexible as possible, ultimately there will always be special cases where an
analysis requires tools which have not been anticipated, or where the debugging
process requires a more detailed look at the intermediate stages of the
analysis. Exporting a workflow to a runnable notebook provides a more detailed,
lower-level look at how the analysis is being performed, and can also allow the
user to reuse certain steps from a given analysis within any other Python
program.

``cesium`` library
==================
The first half of the ``cesium`` framework is the back-end Python library, aimed
at addressing the following uses cases:

1. A domain scientist who is comfortable with programming but is **unfamiliar
   with time series analysis or machine learning**.

2. A scientist who is experienced with time series analysis but is looking for
   **new features** that can better capture patterns within their data.

3. A user of the ``cesium`` web front end who realizes she requires additional
   functionality and wishes to add additional stages to their workflow.

Our framework primarily implements "feature-based methods", wherein the raw
input time series data is used to compute "features" that compactly capture the
complexity of the signal space within a lower-dimensional feature space;
standard machine learning approaches (such as random forests :cite:`breiman2001`
and support vector machines :cite:`suykens1999`) may then be used for supervised
classification or regression. 

``cesium`` allows users to select from a large library of features,
including both general time series features and domain-specific features drawn from
various scientific disciplines. Some specific advantages of the ``cesium``
featurization process include:

- Support for both regularly and irregularly sampled time series (i.e., where
  the time lags between data points are not constant).

- Ability to incorporate measurement errors, which can be provided for each data
  point of each time series (if applicable).

- Support for multi-channel data, for which features are computed separately for
  each dimension of the input data.

Example features
----------------
Some ``cesium`` features are extremely simple and intuitive: summary statistics
such as maximum/minimum values, mean/median values, and standard deviation or median
absolute deviation are a few such examples. Other features involve
measurement errors if they are available: for example, a mean and standard
deviation that is weighted by measurement errors allows noisy data with 
large outliers to be modeled more precisely.

.. figure:: cesium-ls

   Fitted multi-harmonic Lomb-Scargle model for a light curve from a periodic
   Mira-class star. :label:`ls`

Other more involved features could be the estimated parameters for various fitted
statistical models: Figure :ref:`ls` shows a multi-frequency,
multi-harmonic Lomb-Scargle model that describes the rich periodic behavior in
an example time series :cite:`lomb1976,scargle1982`. In particular, a time
series is modeled as a periodic function

.. math::

   \tilde{y}(t) = \sum_{i=1}^m \sum_{j=1}^n A_{ij} \cos i \omega_j t + B_{ij} \sin i \omega_j t,

where the parameters :math:`A_{ij}, B_{ij},` and :math:`\omega_j` are selected
via non-convex optimization to minimize the residual sum of squares
(weighted by measurement errors if applicable). The estimated periods,
amplitudes, phases, and goodness-of-fits can then be used as features which
broadly characterize the periodicity of the input time series.

API details
-----------
Here we provide a few examples of the main ``cesium`` API components that would
be used in a typical analysis task. A workflow will typically consist of three
steps: featurization, model building, and prediction on new data. The majority of
``cesium`` functionality is contained within the ``cesium.featurize`` submodule;
the ``cesium.build_model`` and ``cesium.predict`` submodules primarily provide
interfaces between sets of feature data, which contain both feature data and a
variety of metadata about the input time series, and machine learning models
from ``scikit-learn`` [?], which require dense, rectangular input data. Note
that, as ``cesium`` is under active development, some of the following details
are subject to change.

The featurization step is performed using one of two main functions:

- ``featurize_time_series(times, values, errors, ...)``

  - Takes in data that is already present in memory and computes the requested
    features (passed in as string feature names) for each time series.

  - Features can be computed in parallel across workers (``use_celery=True``) or
    locally in serial (``False``).

  - Class labels/regression targets and metadata/features with known values are
    passed in and stored in the output dataset.

  - Additional feature functions can be passed in as ``custom_functions``.

- ``featurize_data_files(uris, ...)``,

  - Takes in a list of file paths or URIs and dispatches featurization tasks to
    Celery workers.

  - Data is loaded only remotely by the workers rather than being copied, so
    this approach should be preferred for very large input datasets.

  - Features, metadata, and custom feature functions are passed in the same way
    as ``featurize_data_files``.

The output of both functions is a ``Dataset`` object from the ``xarray`` library
[?], which will also be referred to here as a "feature set" (more about
``xarray`` is given in the next section). The feature set stores the computed
feature values for each function (indexed by channel, if the input data is
multi-channel), as well as time series filenames or labels, class labels or
regression targets, and other arbitrary metadata to be used in building a
statistical model.

In order to simplify building ``sckit-learn`` models from (non-rectangular)
feature set data, the following functions are provided in the ``build_model``
submodule:

- ``build_model_from_featureset(featureset, ...)``
  
  - Returns a fitted ``scikit-learn`` model based on the input feature data.

  - A pre-initialized (but untrained) model can be passed in, or the model type
    can be passed in as a string.

  - Model parameters can be passed in as fixed values, or as ranges of values
    from which to select via cross-validation.

- ``rectangularize_featureset(featureset)``

  - Called internally by ``build_model_from_featureset`` to reshape feature data
    in a way that is consumable by ``scikit-learn`` models.

Analogous helper functions for prediction are available in the ``predict`` module:

- ``model_predictions(featureset, model, ...)``

  - Generates predictions from a feature set outputted by
    ``featurize_time_series`` or ``featurize_data_files``.

- ``predict_data_files(file_paths, model, ...)``

  - Like ``featurize_data_files``, generate predictions for time series which
    have not yet been featurized by dispatching featurization tasks to Celery
    workers and then passing the resulting featureset to ``model_predictions``.

The main function ``model_predictions`` takes a set of already-computed features
and predicts the corresponding class labels or regression targets.
Alternatively, the ``predict_data_files`` function can be used to make
predictions from raw time series data that is stored on disk; the features that
were used to train the given model will be computed for the new input data and
then used to make predictions. Depending on the quality of the predictions, new
models can easily be trained with more or fewer features without recomputing all the
previous feature values until the analysis is complete.


Other technological components
------------------------------
In order to eliminate redundant computation,
the set of necessary computations is represented internally as a directed
acyclic graph (DAG) and evaluated efficiently via ``dask`` (see Figure
:ref:`dask` for an example).
In addition to the built-in features, custom feature computation functions can
be passed in directly by the user; such custom functions can similarly make use
of the internal ``dask`` representation so that built-in features can be reused
for the evaluation of user-specified functions. Finally, meta-features (whose
for each time series is specified in advance) can also be passed to
``featurize_time_series`` and stored in the same output dataset alongside
computed feature values.

.. figure:: dask

   Example of a directed feature computation graph using ``dask``. :label:`dask`

Web front end
=============
The ``cesium`` front end provides easy, web-based access to time series
analysis, addressing three common use cases:

1. A scientist needs to perform time series analysis, but is
   **unfamiliar with programming** and library usage.
2. A group of scientists want to **collaboratively explore** different
   methods for time-series analysis.
3. A scientist is unfamiliar with time-series analysis, and wants to **learn**
   how to apply various methods to their data, using **industry best
   practices**.

.. figure:: architecture

   Architetural diagram of ``cesium`` analysis platform *TODO: UPDATE*. :label:`architecture`

The front-end system (together with its deployed back end), offers the
following features:

 - Distributed, parallelized fitting of machine learning models.
 - Isolated [#isolation]_, cloud-based execution of user-uploaded code.
 - Visualization and analysis of results.
 - Tracking of an entire exploratory workflow from start-to-finish for
   reproducibility (in progress).
 - Downloads of Jupyter notebooks to replicate analyses (in progress).

.. [#isolation] Isolation is currently provided by limiting the user
                to non-privileged access inside a Docker container. This
                does not theoretically guarantee 100% isolation.


Back end to front end communication
-----------------------------------
Traditionally, web front ends communicate with back ends via API
requests. For example, to add a new user, the front end would make an
asynchronous JavaScript (AJAX) POST to `/create_user`. The request
returns with a status and some data, which the front end relays to the
user as appropriate.

These types of calls are designed for short-lived request-answer
sessions: the answer has to come back before the connection times out,
otherwise the front end is responsible for implementing logic for
recovery. When the back end has to deal with a longer running task,
the front end typically polls repeatedly to see when it is done. Other
solutions include long polling or server-side events.

In our situation, tasks execute on the order of several (sometimes
tens of) minutes. This situation can be handled gracefully using
WebSockets |---| the caveat being that these can be intimidating to set
up, especially in Python.

We have implemented a simple interface for doing so that we informally call *message
flow*. It adds WebSocket support to any Python WSGI server (Flask, Django, Pylons, etc.),
and allows scaling up as demand increases.

A detailed writup of *message flow* can be found on the Cesium blog at <INSERT URL>. It
allows us to implement trivially modern data flow models such as `Flux
<https://facebook.github.io/flux/>`_, where information always flows in one direction:
from front end to back end via API calls, and from back end to front end via WebSocket
communication.

.. [^channels] At PyCon2016, Andrew Godwin presented a similar
               solution for Django called "channels". The work
               described here happened before we became aware of
               Andrew's, and generalizes beyond Django to, e.g.,
               Flask, the web framework we use.

Deployment
----------
While the deployment details of the web front end are beyond the scope of this paper, it
should be noted that it was designed with scalability in mind.

An NGINX proxy exposes a pool of websocket and WSGI servers to the
user. This gives us the flexibility to choose the best implementation
of each. Communications between WSGI servers and WebSocket servers
happen through a `ZeroMq <http://zeromq.org/>`_ XPUB-XSUB pipeline
(but can be replaced with any other broker, e.g., `RabbitMQ
<https://blog.pivotal.io/pivotal/products/rabbitmq-hits-one-million-messages-per-second-on-google-compute-engine>`_).

The overarching design principle is to connect several, small component, each
performing only one, simple task |---| the one it was designed for.

Computational Scalability
-------------------------
In many fields, the volumes of time series data available can be immense.
``cesium`` makes the process of analyzing time series easily parallelizable and
scaleable; scaling an analysis from a single system to a large cluster should
be easy and accessible to non-technical experts.

Both the back-end library and web front end make use of Celery and RabbitMQ for
distributing featurization tasks to multiple workers; this could be used for
anything from automatically utilizing all the available cores of a single machine,
to assigning jobs across a large cluster. Similarly, both parts of the
``cesium`` framework include support for various distributed filesystems, so
that analyses can be performed without copying the entire dataset into a
centralized location.

While the ``cesium`` library is written in pure Python, the overhead of the
featurization tasks is minimal; the majority of the work is done by the feature
code itself. Most of the built-in features are based on high-performance
``numpy`` functions; others are written in pure C with interfaces in Cython.
The use of ``dask`` graphs to eliminate redundant computations also serves to
minimize memory footprint and reduce computation times.

Automated testing and documentation
-----------------------------------
While the back-end library and web front end are developed in separate GitHub
repositories, the connections between the two somewhat complicate the continuous
integration testing setup. Both repositories are integrated with
`Travis CI <https://travis-ci.com/>`_ for
automatic testing of all branches and pull requests; in addition, any new pushes
to ``cesium/master`` trigger a set of tests of the front end using the new
version of the back-end library, with any failures being reported but not
causing the ``cesium`` build to fail (the reasoning being that the back-end
library API should be the "ground truth", so any updates represent a required
change to the front end, not a bug *per se*).

Documentation for the back-end API is automatically generated in ReStructured
Text format via ``numpydoc``; the result is combined with the rest of our
documentation and rendered as HTML using ``sphinx``. Code examples (without
output) are stored in the repository in Markdown format as opposed to Jupyter
notebooks since this format is better suited to version control. During the
doc-build process, the Markdown is converted to notebook format using
``notedown``, then executed using ``nbconvert`` and converted back to Markdown
(with outputs included), to be finally rendered by ``sphinx``. Both the HTML and
notebook versions are available for every example workflow.

Example EEG dataset analysis
============================
In this example we'll compare various techniques for epilepsy detection using a
classic EEG time series dataset from Andrzejak et al. :cite:`andrzejak2001`.
The raw data are separated into five classes: Z, O, N, F, and S; we will
consider a three-class classification problem of distinguishing normal (Z, O),
interictal (N, F), and ictal (S) signals. We'll show how to perform the
same analysis using both the back-end Python library and the web front end.

.. Here we present an example analysis of a light curve dataset from astronomy
   performed using both the Python library and the equivalent front end workflow. 
   The problem involves classifying light curves (i.e., time series consisting
   of times, star brightness values (in magnitudes), and measurement errors) based
   on the type of star from which they were collected. We follow the approach
   of :cite:`` using the same 810 training examples but with a reduced set of features
   for simplicity.

Python library
--------------
First, we'll load the data and inspect a representative time series from each class:
Figure :ref:`eeg` shows one time series from each of the three classes, after the time
series are loaded from ``cesium.datasets.andrzejak``.

Once the data is loaded, we can generate features for each time series using the
``cesium.featurize`` module. The ``featurize`` module includes many built-in choices of
features which can be applied for any type of time series data; here we've chosen a few
generic features that do not have any special biological significance.

If Celery is running, the time series will automatically be split among the available workers
and featurized in parallel; setting ``use_celery=False`` will cause the time series to be
featurized serially.

.. code-block:: python
        
        from cesium import featurize

        features_to_use = ['amplitude', 'maximum', 'max_slope',
                           'median', 'median_absolute_deviation',
                           'percent_beyond_1_std',
                           'percent_close_to_median', 'minimum',
                           'skew', 'std', 'weighted_average']
        fset_cesium = featurize.featurize_time_series(
                          times=eeg["times"],
                          values=eeg["measurements"],
                          errors=None,
                          features_to_use=features_to_use,
                          targets=eeg["classes"])

.. code-block:: python

        <xarray.Dataset>
        Dimensions:   (channel: 1, name: 500)
        Coordinates:
        * channel   (channel) int64 0
        * name      (name) int64 0 1 ...
          target    (name) object 'Normal' 'Normal' ...
        Data variables:
          minimum   (name, channel) float64 -146.0 -254.0 ...
          amplitude (name, channel) float64 143.5 211.5 ...
          ...


The output of ``featurize_time_series`` is an ``xarray.Dataset`` which contains all the
feature information needed to train a machine learning model: feature values are stored as
data variables, and the time series index/class label are stored as coordinates (a
``channel`` coordinate will also be used later for multi-channel data).

Custom feature functions not built into ``cesium`` may be passed in using the
``custom_functions`` keyword, either as a dictionary ``{feature_name: function}``, or as a
``dask`` graph. Functions should take three arrays ``times, measurements, errors`` as
inputs; details can be found in the ``cesium.featurize`` documentation. Here we'll
compute five standard features for EEG analysis suggested by Guo et al. :cite:`guo2011`:

.. code-block:: python
                
        import numpy as np
        import scipy.stats
        
        def mean_signal(t, m, e):
            return np.mean(m)
        
        def std_signal(t, m, e):
            return np.std(m)
        
        def mean_square_signal(t, m, e):
            return np.mean(m ** 2)
        
        def abs_diffs_signal(t, m, e):
            return np.sum(np.abs(np.diff(m)))
        
        def skew_signal(t, m, e):
            return scipy.stats.skew(m)

Now we'll pass the desired feature functions as a dictionary via the ``custom_functions``
keyword argument.

.. code-block:: python
        
        guo_features = {
            'mean': mean_signal,
            'std': std_signal,
            'mean2': mean_square_signal,
            'abs_diffs': abs_diffs_signal,
            'skew': skew_signal
        }
        
        fset_guo = featurize.featurize_time_series(
                       times=eeg["times"],
                       values=eeg["measurements"],
                       errors=None, targets=eeg["classes"], 
                       features_to_use=guo_features.keys(),
                       custom_functions=guo_features)

.. code-block:: python

        <xarray.Dataset>
        Dimensions:    (channel: 1, name: 500)
        Coordinates:
        * channel    (channel) int64 0
        * name       (name) int64 0 1 ...
          target     (name) object 'Normal' 'Normal' ...
        Data variables:
          abs_diffs  (name, channel) float64 4695.2 6112.6 ...
          mean       (name, channel) float64 -4.132 -52.44 ...
          mean2      (name, channel) float64 1652.0 5133.3 ...
          skew       (name, channel) float64 0.0328 -0.09271 ...
          std        (name, channel) float64 40.41 48.81 ...

The EEG time series considered here consist of univariate signal measurements along a
uniform time grid. But ``featurize_time_series`` also accepts multi-channel data; to
demonstrate this, we will decompose each signal into five frequency bands using a discrete
wavelet transform as suggested by Subasi :cite:`subasi2007`, and then featurize each band
separately using the five functions from above.

.. code-block:: python

        import pywt
        
        n_channels = 5
        eeg["dwts"] = [pywt.wavedec(m, pywt.Wavelet('db1'),
                                    level=n_channels-1)
                       for m in eeg["measurements"]]
        fset_dwt = featurize.featurize_time_series(
                       times=None, values=eeg["dwts"], errors=None,
                       features_to_use=guo_features.keys(),
                       targets=eeg["classes"],
                       custom_functions=guo_features)
        
.. code-block:: python

        <xarray.Dataset>
        Dimensions:    (channel: 5, name: 500)
        Coordinates:
        * channel    (channel) int64 0 1 ...
        * name       (name) int64 0 1 ...
          target     (name) object 'Normal' 'Normal' ...
        Data variables:
          abs_diffs  (name, channel) float64 25131 18069 ...
          skew       (name, channel) float64 -0.0433 0.06578 ...
          mean2      (name, channel) float64 12944 5362.3 ...
          mean       (name, channel) float64 -17.08 -6.067 ...
          std        (name, channel) float64 112.5 72.97 ...


The output feature set has the same form as before, except now the ``channel`` coordinate is
used to index the features by the corresponding frequency band. The functions in
``cesium.build_model``
and ``cesium.predict``
all accept feature sets from single- or multi-channel data, so no additional steps are
required to train models or make predictions for multichannel feature sets using the
``cesium`` library.

Model building in ``cesium`` is handled by the
``build_model_from_featureset``
function in the ``cesium.build_model`` submodule. The feature set output by
``featurize_time_series``
contains both the feature and target information needed to train a
model; ``build_model_from_featureset`` is simply a wrapper that calls the ``fit`` method of a
given ``scikit-learn`` model with the appropriate inputs. In the case of multichannel
features, it also handles reshaping the feature set into a (rectangular) form that is
compatible with ``scikit-learn``.

For this example, we'll test a random forest classifier for the built-in ``cesium`` features,
and a 3-nearest neighbors classifier for the others, as in :cite:`guo2011`.

.. code-block:: python
        
        from cesium.build_model import build_model_from_featureset
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cross_validation import train_test_split
        
        train, test = train_test_split(np.arange(len(eeg["classes"])), random_state=0)
        
        rfc_param_grid = {'n_estimators': [8, 32, 128, 512]}
        model_cesium = build_model_from_featureset(
                           fset_cesium.isel(name=train),
                           RandomForestClassifier(),
                           params_to_optimize=rfc_param_grid)
        knn_param_grid = {'n_neighbors': [1, 2, 3, 4]}
        model_guo = build_model_from_featureset(
                        fset_guo.isel(name=train),
                        KNeighborsClassifier(),
                        params_to_optimize=knn_param_grid)
        model_dwt = build_model_from_featureset(
                        fset_dwt.isel(name=train),
                        KNeighborsClassifier(),
                        params_to_optimize=knn_param_grid)

Making predictions for new time series based on these models follows the same pattern:
first the time series are featurized using
``featurize_timeseries``
and then predictions are made based on these features using
``predict.model_predictions``,

.. code-block:: python
        
        from sklearn.metrics import accuracy_score
        from cesium.predict import model_predictions
        
        preds_cesium = model_predictions(
                           fset_cesium, model_cesium,
                           return_probs=False)
        preds_guo = model_predictions(fset_guo, model_guo,
                           return_probs=False)
        preds_dwt = model_predictions(fset_dwt, model_dwt,
                           return_probs=False)
        
        print("Builtin: train acc={:.2%}, test acc={:.2%}"\
              .format(accuracy_score(preds_cesium[train],
                                     eeg["classes"][train]),
                      accuracy_score(preds_cesium[test],
                                     eeg["classes"][test])))
        print("Guo et al.: train acc={:.2%}, test acc={:.2%}"\
              .format(accuracy_score(preds_guo[train],
                                     eeg["classes"][train]),
                      accuracy_score(preds_guo[test],
                                     eeg["classes"][test])))
        print("Wavelets: train acc={:.2%}, test acc={:.2%}"\
              .format(accuracy_score(preds_dwt[train],
                                     eeg["classes"][train]),
                      accuracy_score(preds_dwt[test],
                                     eeg["classes"][test])))

.. code-block:: python

        Builtin: train acc=100.00%, test acc=83.20%
        Guo et al.: train acc=90.93%, test acc=84.80%
        Wavelets: train acc=100.00%, test acc=95.20%

The workflow presented here is intentionally simplistic and omits many important steps
such as feature selection, model parameter selection, etc., which may all be
incorporated just as they would for any other ``scikit-learn`` analysis.
But with essentially three function calls (``featurize_time_series``,
``build_model_from_featureset``, and ``model_predictions``), we are able to build a
model from a set of time series and make predictions on new, unlabeled data. In
the next section we'll introduce the web front end for ``cesium`` and describe how
the same analysis can be performed in a browser with no setup or coding required.

Web front end
-------------
*TODO Add web clickthrough.*

*How much will the front end be changing? Just the styling or will the actual
flow be different? Could go ahead and write it up before we finish the new
version, or wait til after...*

.. figure:: web1

   "Projects" tab :label:`web1`

.. figure:: web2

   "Data" tab :label:`web2`

.. figure:: web3

   "Featurize" tab :label:`web3`

.. figure:: web4

   "Build Model" tab :label:`web4`

.. figure:: web5

   "Predict" tab :label:`web5`


Conclusion
==========
The ``cesium`` framework provides tools that allow anyone from machine learning
specialists to domain experts without any machine learning experience to rapidly
prototype explanatory models for their time series data and quickly generate
predictions for new, unlabeled data. Aside from the applications to time domain
informatics, our project has several aspects which are relevant to the broader
scientific Python community.

First, the dual nature of the project (Python back end vs. web front end) presents
both unique challenges and interesting opportunities in striking a balance
between accessibility and flexibility of the two components.
Second, the ``cesium`` project places a strong emphasis on reproducible
workflows: all actions performed within the web front end are logged and can be
easily exported to an IPython notebook that exactly reproduces the steps of the
analysis. Finally, the scope of our project is simultaneously both narrow (time
series analysis) and broad (numerous distinct scientific disciplines), so
determining how much domain-specific functionality to include is an ongoing
challenge.

*TODO roadmap?*

References
==========
