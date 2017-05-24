:author: Michael Beyeler
:email: mbeyeler@uw.edu
:institution: Department of Psychology, University of Washington
:institution: Institute for Neuroengineering, University of Washington
:institution: eScience Institute, University of Washington
:corresponding:

:author: Ariel Rokem
:email: arokem@gmail.com
:institution: eScience Institute, University of Washington
:institution: Institute for Neuroengineering, University of Washington

:author: Geoffrey M. Boynton
:email: gboynton@uw.edu
:institution: Department of Psychology, University of Washington

:author: Ione Fine
:email: ionefine@uw.edu
:institution: Department of Psychology, University of Washington
:bibliography: bibliography

:video: https://github.com/uwescience/pulse2percept


--------------------------------------------------------------------
pulse2percept: A Python-based simulation framework for bionic vision
--------------------------------------------------------------------

.. class:: abstract

   By 2020 roughly 200 million people worldwide will suffer from photoreceptor
   diseases such as retinitis pigmentosa and age-related macular degeneration,
   and a variety of retinal sight restoration technologies are being developed
   to target these diseases.
   Two brands of retinal prostheses are already being implanted in patients.
   Analogous to cochlear implants, these devices use a grid of electrodes to
   stimulate remaining retinal cells.
   However, clinical experience with these implants has made it apparent that
   the vision restored by these devices differs substantially
   from normal sight.
   Here we present *pulse2percept*, an open-source Python implementation
   of a computational model that can predict the perceptual experience
   of retinal prosthesis patients across a wide range of implant configurations.
   A modular and extensible user interface
   exposes the different building blocks of the software,
   making it easy for users to simulate
   novel implants, stimuli, and retinal models.


.. class:: keywords

   bionic vision, retinal implant, pulse2percept


Introduction
------------

Two of the most frequenct causes of blindness in the developed world
are retinitis pigmentosa (RP) and age-related macular degeneration (AMD)
:cite:`Bunker1984,EyeDiseases2004`.
Both of these diseases are hereditary,
and can be identified by a progressive degeneration of
photoreceptors in the retina,
resulting in severe visual impairment.
In severe end-stage RP, approximately 95% of photoreceptors,
20% of bipolar cells,
and 70% of ganglion cells degenerate :cite:`Santos1997`.
A significant fraction of bipolar and ganglion cells are spared,
but the absence of photoreceptors prevents useful vision.

Although decimated in number, the surviving retinal ganglion
cells -- which represent the output layer of the retina that
sends visual signals to the brain via the optic nerve -- seem to
maintain their morphological structure and connectivity
:cite:`Humayun1999,Mazzoni2008`.
This has raised the possibility for patients blinded by RP and AMD
to potentially have their vision restored via the implantation
of microelectronic retinal prostheses.

Analogous to cochlear implants, the goal of electronic retinal prostheses
is to electrically stimulate the surviving retinal neurons
in order to evoke neuronal responses that are transmitted
to the brain and interpreted by patients as visual percepts
(see Fig. :ref:`figimplant` A).

.. figure:: figimplant.png
   :align: center
   :scale: 25%

   Electronic retinal prosthesis.
   A) Light from the visual scene is captured by an external camera and
   transformed into electrical pulses delivered through electrodes
   to stimulate the retina.
   B) Prostheses can be placed in the epiretinal, subretinal, or
   suprachoroidal space.
   TODO Need to redo the figure in order to avoid copyright issues.
   :label:`figimplant`

Several types of retinal prostheses are currently in development,
varying in user interface, light-detection method, signal processing,
and microelectrode placement within the retina
(for a recent review see :cite:`Weiland2016`).
The three main locations for microelectrode array placement are the
`epiretinal` (i.e., on top of the retinal surface, above the optic fiber layer),
`subretinal` (i.e., between bipolar cells and retinal pigmented epithelium),
and `suprachoroidal` space (i.e., between the choroid and the sclera)
as shown in Fig. :ref:`figimplant` B).
Each of these approaches is similar in that light from the visual scene
is captured and transformed into electrical pulses delivered through electrodes
to stimulate the retina.

.. figure:: figmodel.eps
   :align: center
   :figclass: w
   :scale: 35%

   Full model cascade. TODO explain.
   :label:`figmodel`


Two of these systems are approved for commercial
use and have already been implanted in patients across the US and Europe:
the Argus II device
(epiretinal, Second Sight Medical Products Inc., :cite:`daCruz2016`)
and the Alpha-IMS system (subretinal, Retina Implant AG, :cite:`Stingl2015`).
At the same time, a number of other devices have either started
or are planning to start clinical trials in the near future,
potentially offering a wide range of sight restoration options
for blinded individuals within a decade :cite:`Fine2015`.

However, clinical experience with existing retinal prostheses makes it
apparent that the vision provided by these devices differs substantially
from normal sight.
Evidence suggests that the interactions between implant electronics and
the underlying neurophysiology cause nontrivial perceptual distortions
in both space and time :cite:`FineBoynton2015,Beyeler2017`
that severely limit the quality of the generated visual experience.
For example, stimulating even a single electrode leads to percepts
that vary dramatically in shape
(e.g., varying in description from "blobs", to "streaks" and "half-moons")
and duration (e.g., fading over time).
Rather than seeing clear and sharp contours of objects,
patients report their visual experience to be more like
:cite:`PioneerPress2015`:
*"... looking at the night sky where you have millions of twinkly lights
that almost look like chaos"*.

We have previously developed a computational model of bionic vision
that can explain these perceptual distortions
across a wide range of implant configurations and stimulation protocols
:cite:`Horsager2009,Nanduri2012`.
Here we present an open-source implementation of these models as part of
*pulse2percept*, a Python-based simulation framework that relies solely on
the NumPy and SciPy stacks, as well as contributions
from the broader Python community.
Based on the detailed specification of a patient's implant configuration,
and given a desired electrical stimulation protocol,
the model then predicts the perceptual distortions experienced
by this "virtual patient" over both space and time.
We hope that this library will contribute substantially
to the field of medicine
by providing a tool to accelerate the development of visual prostheses
suitable for human trials.

.. Here we present *pulse2percept*, an open-source Python implementation
.. of a computational model that can predict the perceptual experience
.. of retinal prosthesis patients across a wide range of
.. implant configurations.


The remainder of this paper is organized as follows:
We start by detailing the computational model that underlies *pulse2percept*,
before we give a simple usage example and go into implementation details.
We then review our solutions to various technical challenges,
and conclude by discussing the broader impact for this work
for the computational neuroscience and neural engineering communities.


Computational Model of Bionic Vision
------------------------------------

Analogous to models of cochlear implants,
the here presented computational model closely mimics sensory information
processing in the human retina
in response to electrical stimulation.
The model consists of a number of linear and nonlinear filtering steps
that process an electrical pulse pattern in both space and time.
Model parameters were chosen to fit data from experiments in which patients
with prosthetic devices were asked to report about their threshold for
perceiving stimulation, and from experiments in which patients drew the shapes
of the percepts evoked by stimulation.
The model has been shown to generalize across individual
electrodes, patients, and devices, as well as across different experiments.
Detailed methods can be found in :cite:`Horsager2009,Nanduri2012,Beyeler2017`.
Here we provide a brief overview.

The full model cascade for an Argus I epiretinal prosthesis is illustrated in
Fig. :ref:`figmodel`, although this model generalizes to other epiretinal
and subretinal configurations.

The Argus I device consists of electrodes of 260 :math:`\mu m` and 520 :math:`\mu m`
diameter, arranged in a checkerboard pattern (Fig. :ref:`figmodel` A).
In this example, input to the model was a pair of simulated pulse
trains phase-shifted by :math:`\delta` ms,
which were delivered to two individual simulated electrodes.
The current spread for
each electrode decreased as a function of distance from the electrode center
(heat map in A).
We modeled the sensitivity of axon fibers (green lines in B;
location of the array with respect to the optic disc was inferred from
patients' fundus photographs) as decreasing exponentially as a
function of distance from the soma.

The resulting sensitivity profile (heat map in B) was then convolved
pixel-by-pixel with a number of linear (boxes C, D, and F)
and nonlinear (box E) steps to model the temporal senstivity
of the retinal tissue,
similar to models of auditory stimulation in cochlear implant users.

Linear responses were modeled as temporal low-pass filters,
or "leaky integrators",
modeled with gamma functions of order :math:`n`:

.. math::
   :label: eqgamma

   \delta(t, n, \tau) = \frac{\exp(-t / \tau)}{\tau (n - 1)!} \Big( \frac{t}{\tau} \Big)^{n-1}

where :math:`t` is time,
:math:`n` is the number of identical, cascading stages,
and :math:`\tau` is the time constant of the filter.

We began by convolving the electrical input stimulus :math:`f(s,t)`
with a one-stage gamma function (:math:`n=1`,
time constant :math:`\tau_1 = 0.42` ms)
to model the impulse response function of retinal ganglion cells
(Fig. :ref:`figmodel` C):

.. math::
   :label: eqfast

   r_1(s,t) = f(s,t) * \delta(t, 1, \tau_1),

where :math:`*` denotes convolution.

We assumed that the system became less sensitive as a function of
accumulated charge.
This was implemented by calculting the amount of accumulating charge
at each point of time in the stimulus, :math:`c(t)`,
and convolving this accumulation with a second one-stage gamma function
(:math:`n=1`, time constant :math:`\tau_2 = 45.3` ms;
Fig. :ref:`figmodel` D).
The output of this convolution was scaled by a factor
:math:`\epsilon_1 = 8.3` and subtracted from :math:`r_1` (Eq. :ref:`eqfast`):

.. math::
   :label: eqacc

   r_2(s,t) = r_1(s,t) - \epsilon_1\big( c(s,t) * \delta(t, 1, \tau_2) \big).

The response :math:`r_2(s,t)` was then passed through a stationary
nonlinearity (:ref:`figmodel` E) to model the nonlinear input-output
relationship of ganglion cell firing:

.. math::
   :label: eqnonlinear

   r_3(s,t) = r_2(s,t) \frac{\alpha}{1 + \exp{\frac{i - \max_t{r_2(s,t)}}{s}}}

where :math:`\alpha = 14` (asymptote),
:math:`s = 3` (slope),
and :math:`i = 16` (shift) were chosen
to match the observed psychophysical data.

Finally, the response :math:`r_3(s,t)` was convolved with another low-pass
filter described as a three-stage gamma function
(:math:`n = 3`, :math:`\tau_3 = 26.3` ms)
intended to model slower perceptual processes in the brain
(:ref:`figmodel` F):

.. math::
   :label: eqslow

   r_4(s,t) = \epsilon_2 r_3(s,t) * \delta(t, 3, \tau_3),

where :math:`\epsilon_2 = 1000` was a scaling factor used to
fit the output to subjective brightness values in [0, 100]
reported by patients on single-electrode stimulation tasks.
Thus the output of the model was a map of subjective brightness values
that change over time.
An example percept generated by the model is shown on the right-hand
side of Fig. :ref:`figmodel`, along with the perceived percept as
reported by one of the subjects.

.. The output of the model was a map of brightness values (arbitrary units) over time.
.. Subjective brightness was defined as the highest brightness value in the map.

All parameter values are given in Table :ref:`tableparams`.

.. raw:: latex

   \begin{table}[h]
     \begin{tabular}{|r|r|r|}
     \hline
     Name & Parameter & Value \\
     \hline
     Time constant: ganglion cell impulse response & $\tau_1$ & 0.42 ms \\
     Time constant: charge accumulation & $\tau_2$ & 45.3 ms \\
     Time constant: cortical response & $\tau_3$ & 26.3 ms \\
     TODO & & \\
     \hline
     \end{tabular}
     \caption{Parameter values}
     \label{tableparams}
   \end{table}



Implementation and Results
--------------------------

Code Organization
~~~~~~~~~~~~~~~~~

The project seeks a trade-off between object oriented programming
and ease of use. To facilitate ease of use, the simulations in *pulse2percept*
are organized as a standard Python package, consisting of the following primary
modules:

- :code:`api`: Provides a top-level Application Programming Interface.
- :code:`retina`: Includes implementations of the temporal cascade of events
  described in equations 1-5, as well as implementation of a model of the retinal
  distribution of nerve fibers, based on :cite:`JAN09`.
- :code:`implants`: Provides implementations of the details of different retinal
  prosthetic implants. This includes Second Sight's Argus I and Argus II arrays,
  but can easily be extended to custom implants (see Section Extensibility).
- :code:`stimuli`: Includes implementations of commonly used electrical stimulation
  protocols, including means to translate images and movies into simulated
  electrical pulse trains.
- :code:`files`: Includes a simple means to load and store data as images
  and videos.
- :code:`utils`: Utility and helper functions used in various parts of the code.


Basic Usage
~~~~~~~~~~~

Here we give a minimal usage example to produce the percept shown on the right-hand
side of Fig. :ref:`figmodel`.

Convention is to import the main :code:`pulse2percept` module
as :code:`p2p`. Throughout this paper, if a class is referred
to with the prefix :code:`p2p`, it means this class belongs to
the main pulse2percept library (e.g., :code:`p2p.retina`):

.. code-block:: python
   :linenos:

   import pulse2percept as p2p

Then an array can be placed at a particular location on the retina,
with respect to the fovea (microns). It can be placed at some height
above the tissue and rotated as you see fit:

.. code-block:: python
   :linenos:
   :linenostart: 2

   import numpy as np
   implant = p2p.implants.ArgusI(x_center=-800,
                                 y_center=0,
                                 h=80,
                                 rot=np.deg2rad(35))

An electrode array is a wrapper around a list of
:code:`p2p.implants.Electrode` objects, which are accessible
via indexing or iteration (e.g., via
:code:`[for i in implant]`).
In addition, every electrode in the array has its own name
(in the Argus I array, they are A1 - A16;
corresponding to the names that are commonly
used by Second Sight Medical Products Inc.).
The first electrode in the array can be accessed both via its
index (:code:`implant[0]`) and its name (:code:`implant['A1']`).

Once the array is created, it can be passed to the simulation framework.
This is also where you specify the backend.

.. code-block:: python
   :linenos:
   :linenostart: 7

   sim = p2p.Simulation(implant, engine='joblib',
                        num_jobs=8)

The simulation framework provides a number of setter functions
for the different layers of the retina.
These allow for flexible specificaton of optional settings,
while abstracting the underlying functionality.
Things that can be set include the spatial sampling rate of the
retina in the optic fiber layer (where the ganglion cell axons are):

.. code-block:: python
   :linenos:
   :linenostart: 9

   ssample = 100  # microns
   sim.set_optic_fiber_layer(sampling=ssample)


Similarly, for the ganglion cell layer we can choose one of the
pre-existing cascade models and specify a temporal sampling rate.
It's also possible to specify your own (custom) model, see
the section on extensibility below.

.. code-block:: python
   :linenos:
   :linenostart: 11

   tsample = 0.005 / 1000  # seconds
   sim.set_ganglion_cell_layer('Nanduri2012',
                               tsample=tsample)


.. figure:: figinputoutput.png
   :align: center
   :scale: 25%

   Input/output. TODO
   :label:`figinputoutput`


Finally, a stimulation protocol can be specified by assigning
stimuli from the :code:`p2p.stimuli` module to specific
electrodes.
An example is to set up a pulse train of particular stimulation
frequency and current amplitude. Because of safety considerations,
all real-world stimuli must be balanced biphasic pulse trains
(meaning they must have a positive and negative phase of equal area,
so that the net current delivered to the tissue sums to zero).
One way is to specify a pulse train for each electrode in the array.
However, for large array this becomes cumbersome.
Therefore, an easier way is to assign pulse trains to electrodes
via a dictionary:

.. code-block:: python
   :linenos:
   :linenostart: 14

   # Stimulate two specific electrodes
   stim = {
       'C1': p2p.stimuli.PulseTrain(tsample, freq=50,
                                    amp=20, dur=0.5)
       'B3': p2p.stimuli.PulseTrain(tsample, freq=50,
                                    amp=20, dur=0.5)
   }

At this point, we can visualize the array's location on the retina
with the :code:`sim.plot_fundus` method. If we pass it the
:code:`stim` dictionary, it will highlight the stimulated electrodes
in the array:

.. code-block:: python
   :linenos:
   :linenostart: 21

   sim.plot_fundus(stim)

The output can be seen in Fig. :ref:`figinputoutput` A.

Finally, the created stimulus serves as input to
:code:`sim.pulse2percept`, which is used to convert the
pulse trains into a percept.
Here we can choose to ignore pixels whose intensity values
are smaller than 25% of the largest value
(in order to save time),
and which retinal layers to consider for processing
(e.g., 'OFL': optic fiber layer, 'GCL': ganglion cell layer):

.. code-block:: python
   :linenos:
   :linenostart: 22

   # From pulse train to percept
   percept = sim.pulse2percept(stim, tol=0.25,
                               layers=['GCL', 'OFL'])

Here, the output :code:`percept` is a :code:`p2p.utils.TimeSeries`
object that contains the timeseries data in its :code:`data`
container.
We get a timeseries of brightness values (arbitrary units)
for every pixel in the percept image.
*pulse2percept* offers a bunch of functions to save the output
as a movie file (via Scikit-Video and ffmpeg).
Alternatively, we retrieve the brightest frame of the timeseries:

.. code-block:: python
   :linenos:
   :linenostart: 25

   frame = p2p.get_brightest_frame(percept)

Then we can plot it with the help of Matplotlib:

.. code-block:: python
   :linenos:
   :linenostart: 26

   import matplotlib.pyplot as plt
   %matplotlib inline
   plt.imshow(frame, cmap='gray')

The output is shown in Fig. :ref:`figinputoutput` B.



Extensibility
~~~~~~~~~~~~~

Extensibility is provided through class inheritance.
This allows users to create their own:

- Ganglion cell models: Inherit from :code:`p2p.retina.TemporalModel`
- Retinal implants: Inherit from :code:`p2p.implants.ElectrodeArray`
- Stimuli: Inherit from :code:`p2p.stimuli.PulseTrain`


A new ganglion cell model works on a single pixel.
It must provide a property called :code:`tsample`,
which is the temporal sampling rate,
and a method called :code:`model_cascade`,
which translates a single-pixel pulse train into
a single-pixel percept:

.. code-block:: python

   class MyGanglionCellModel(TemporalModel):
       def model_cascade(self, in_arr, pt_list, layers):
           return in_array


It can then be passed to the simulation framework:

.. code-block:: python

   mymodel = MyGanglionCellModel()
   sim.set_ganglion_cell_layer(mymodel)


Creating a new array involves inheriting from
:code:`pulse2percept.implants.ElectrodeArray`
and providing a property :code:`etype`,
which is the electrode type
(e.g., epiretinal, subretinal).

Creating a new array is as simple as:

.. code-block:: python

   import pulse2percept as p2p

   class MyArray(p2p.implants.ElectrodeArray):
       def __init__(self, etype):
           self.etype = etype


Creating new stimuli works the same way, either by inheriting
from :code:`pulse2percept.utils.TimeSeries`.
But, you can also inherit
from :code:`pulse2percept.stimuli.MonophasicPulse`,
:code:`pulse2percept.stimuli.BiphasicPulse`,
or :code:`pulse2percept.stimuli.PulseTrain`:



Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Some implementation details and some results.

The main challenge during *pulse2percept*'s development
was computational cost:
the simulations require a fine subsampling of space,
and span several orders of magnitude in time,
ranging from electrical activation of individual retinal ganglion cells
on the sub-millisecond time scale to visual perception occurring
over several seconds.

Like the brain, we solved this problem through parallelization. Computations
were parallelized across small patches of the retina using two back ends (Joblib
:cite:`JOB16` and Dask :cite:`DASK16`), with both multithreading and
multiprocessing options.
The user can select from available back ends when setting up the simulation
framework:

sim = p2p.Simulation(engine=engine, num_jobs=num_jobs)

where engine can be either "serial", "joblib", or "dask",
and num_jobs optionally specifies how many cores should be used.



A major computational bottleneck in computing the
temporal response in each patch of retina are convolutions of arrays describing
the responses of parts of the model at high temoral resolution (e.g., equations
2 and 3). These math-heavy sections of the code were additionally sped up using
the two following strategies: wherever possible, a direct convlution with the
entire time-series was avoided, by preprocessing sparse pulse input arrays, and
only convolving with those parts of the time-series that included non-zero
amplitudes. Furthermore, the calculation was sped up wih LLVM-base compilation
implemented using Numba :cite:`LAM15`.



Computational Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

We measured computational performance and scalability.
It doesn't run in real time but is pretty good,
I mean look at the pretty figure!

.. figure:: figure2.png
   :align: center
   :scale: 50%

   Computational performance. TODO
   :label:`figperformance`

Software availability and development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All code can be found at https://github.com/uwescience/pulse2percept,
with up-to-date documentation
available at https://uwescience.github.io/pulse2percept.
In addition, the latest stable release is available on the Python Package Index
and can be installed using pip:

.. code-block:: bash

  $ pip install pulse2percept

All code presented in this paper is current as of the v0.2 release.



Discussion
----------

We have presented an open-source, Python-based framework for modeling
the visual processing in retinal prosthesis patients.

*pulse2percept* has a number of potential uses.

For device developers, creating "virtual patients" with this software
can facilitate the development of improved pulse stimulation protocols
for existing devices, including generating datasets
for machine learning approaches.
"Virtual patients" are also a useful tool for device development,
making it possible to rapidly predict vision across
different implant configurations.
We are currently collaborating with two leading manufacturers
to use the software for this purpose.

For patients, their families, doctors, and regulatory agencies
(e.g., FDA and Medicare), these simulations can determine
at what stage of vision loss a prosthetic device would be helpful,
and can differentiate the vision quality provided by different devices.

Finally, device manufacturers currently develop their own behavioral tests
and some only publish a selective subset of data.
This makes it extremely difficult to compare patient visual performance
across different devices.
Any simulations that currently exist are proprietary and not available
to the scientific community, and manufacturer-published 'simulations'
of prosthetic vision are sometimes misleading,
if they do not take account of substantial neurophysiological distortions
in space and time.

A major goal of *pulse2percept* is to provide open-source simulations
that can allow any user to directly compare the perceptual experiences
likely to be produced across different devices.


Acknowledgments
---------------
This work was supported by the Washington Research Foundation Funds
for Innovation in Neuroengineering and Data-Intensive Discovery (MB),
as well as a grant by the Gordon & Betty Moore Foundation and
the Alfred P. Sloan Foundation to the University of Washington
eScience Institute Data Science Environment (MB and AR).
The GeForce TITAN X used for this research was donated
by the NVIDIA Corporation, and research credits for cloud computing
were provided by Amazon Web Services.
