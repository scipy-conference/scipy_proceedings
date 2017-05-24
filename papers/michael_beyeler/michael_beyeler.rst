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
   Two brands of retinal prostheses are currently approved for implantation in patients.
   Analogous to cochlear implants, these devices use a grid of electrodes to
   stimulate remaining retinal cells.
   Clinical experience with these implants has made it apparent that
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
Both of these diseases have a hereditary component,
and begin with a progressive degeneration of
photoreceptors in the retina.
In severe end-stage RP, approximately 95% of photoreceptors,
20% of bipolar cells,
and 70% of ganglion cells degenerate,  :cite:`Santos1997`,
resulting in severe visual impairment. 
Despite a significant fraction of bipolar and ganglion cells being spared,
the absence of photoreceptors means that little or no useful vision is retained in the later stages of these diseases.

Microelectronic retinal prostheses target surviving retinal bipolar and ganglion cells. Ganglion cells represent the output layer of the retina. Each ganglion cell sends an electrical signals to the brain via a long axon fiber that passes from the ganglion cell body to the optic nerve. Bipolar cells form an intermediate layer between photoreceptors and ganglion cells. As well as being reduced in number, these remaining cells also undergo corruptive re-modeling in late stages of the disease [MARC]. However at intermediate disease stages, the basic wiring of the non-photoreceptor layers can still be observed 
:cite:`Humayun1999,Mazzoni2008`.


Analogous to cochlear implants, the goal of electronic retinal prostheses is to electrically stimulate the surviving retinal neurons in order to evoke neuronal responses that are transmitted
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

Several types of retinal prostheses are currently in development. These vary in their user interface, light-detection method, signal processing,
and microelectrode placement within the retina
(for a recent review see :cite:`Weiland2016`).
As far as our model is concerned, the critical factor is the placement of the microelectrodes. The three main locations for microelectrode implant placement are 
`epiretinal` (i.e., on top of the retinal surface, above the ganglion cells),
`subretinal` (i.e., next to the bipolar cells in the space of the missing photoreceptors),
and `suprachoroidal` (i.e., between the choroid and the sclera)
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
use and are being implanted in patients across the US and Europe:
the Argus II device
(epiretinal, Second Sight Medical Products Inc., :cite:`daCruz2016`)
and the Alpha-IMS system (subretinal, Retina Implant AG, :cite:`Stingl2015`).
At the same time, a number of other devices have either started
or are planning to start clinical trials in the near future,
potentially offering a wide range of sight restoration options
for blinded individuals within a decade :cite:`Fine2015`.

However, clinical experience with existing retinal prostheses makes it
apparent that the vision provided by these devices differs very substantially
from normal sight.
Interactions between implant electronics and
the underlying neurophysiology cause nontrivial perceptual distortions
in both space and time :cite:`FineBoynton2015,Beyeler2017`
that severely limit the quality of the generated visual experience.
For example, stimulating a single electrode does not always (or even usually) result in the experience of a 'dot' of light. Instead, stimulating a single electrode leads to percepts
that vary dramatically in shape, varying in description from "blobs", to "streaks" and "half-moons". Percepts also do not remain constant over time. The percept produced by stimulating a single electrode with a continuous pulse train fades over time: usually over a course of seconds the percept will completely disappear.
As a result, when using their cameras to experience the visual world, patients do not report seeing an interpretable world. One patient describe it as like:cite:`PioneerPress2015`:
*"... looking at the night sky where you have millions of twinkly lights
that almost look like chaos"*.

Our goal was to develop a simulation framework that could describe patient percepts over space and time -- a 'virtual patient' analogous to the virtual prototyping that has proved so useful in other complex engineering applications. We hope that this library will contribute substantially to the field of medicine by providing a tool to accelerate the development of visual prostheses suitable for human trials.For researchers this tool can be used to improve stimulation protocols for existing devices, and provide a design-tool for future devices. For government agencies such as the FDA and Medicare this tool can help guide reimbursement decisions. For patients and doctors it can help guide patients and doctors in their decision as to when or whether to be implanted, and which device to select. 

Our simulation tool integrates and generalizes two computational models of bionic vision that separately explained spatial : cite:`Nanduri2012` and temporal :cite:`Horsager2009` perceptual distortions for the Second Sight Argus 1 and Argus 2 implants.

Here we present an open-source implementation of these models as part of
*pulse2percept*, a Python-based simulation framework that relies solely on
the NumPy and SciPy stacks, as well as contributions
from the broader Python community.
Based on the detailed specification of a patient's implant configuration,
and given a desired electrical stimulation protocol,
the model then predicts the perceptual distortions experienced
by this "virtual patient" over both space and time.

This simulation had the goal of meeting four significant computational challenges. First, ease of use. The intended users of this simulation include researchers or government officials who collect or assess perceptual data on prosthetic implants (MDs rather than computer scientists). Second, we our implementation is highly flexible as far as the engineering specifications of implants is concerned. It is impossible to predict the engineering specifications (e.g. implant hardware design and stimulation protocols) of future implants. Indeed, within most companies the specifications of implants currently in design is closely guarded intellectual property. Third, modularity in terms of the computational model. As research continues in this field, it is likely that the underlying computational models converting electrical stimulation to patient percept will improve. We used a modular design that makes it easy to update individual components of the model. Finally, our simulation requires computations that were both intensive in terms of both spatial and temporal resolution. Like the retina, this was solved by using a fully parallelized architecture, calculations were carried out independently across each small patch of the retina. 


.. Here we present *pulse2percept*, an open-source Python implementation
.. of a computational model that can predict the perceptual experience
.. of retinal prosthesis patients across a wide range of
.. implant configurations.


The remainder of this paper is organized as follows:
We start by detailing the computational model that underlies *pulse2percept*,
before we give a simple usage example and go into implementation details.
We then review our solutions to various technical challenges,
and conclude by discussing the broader impact for this work
for the computational neuroscience and neural engineering communities in more detail.


Computational Model of Bionic Vision
------------------------------------

Analogous to models of cochlear implants,[REF] the goal of our
computational model is to approximate, via a number of linear and nonlinear filtering steps, the neural computations that convert an electrical pulse pattern in both space and time into a perceptual experience. 

Model parameters were chosen to fit data from a variety of experiments in patients with prosthetic devices. For example, in some experiments patients were asked to report whether or not they detected a percept. Across many trials, the minimum stimulation current amplitude needed to reliably detect the presence of a percept on 80% of trials was found. This threshold was compared across a variety of pulse trains. In other experiments patients reported the apparent brightness or size of percepts on a rating scale. In others  patients drew the shapes
of the percepts evoked by stimulation.
The model has been shown to generalize across individual
electrodes, patients, and devices, as well as across different experiments.
Detailed methods of how the model was validated can be found in :cite:`Horsager2009,Nanduri2012,Beyeler2017`.
Here we provide a brief overview.

The full model cascade for an Argus I epiretinal prosthesis is illustrated in
Fig. :ref:`figmodel`. However, as described above, this model was designed to generalize to other epiretinal
and subretinal configurations.

The Argus I device consists of electrodes of 260 :math:`\mu m` and 520 :math:`\mu m`
diameter, arranged in a checkerboard pattern (Fig. :ref:`figmodel` A).
In this example, input to the model was a pair of simulated pulse
trains phase-shifted by :math:`\delta` ms,
which were delivered to two individual simulated electrodes.

The first stages of the model are only in the space domain, and describe the spatial distortions resulting from interactions between the electronics and the neuroanatomy of the retina. The current spread for
each electrode decreases as a function of distance from the electrode center, both in the x, y plane and as a function of electrode height (z). Thus, the heat maps in A describes the electrical current field across the retinal surface for each individual electrode.


EQUATION.

As described above, each ganglion cell has an axon fiber that travels from that ganglion cell body to the optic nerve. Stimulated electrodes can induce action potentials in axon fibers as well as cell bodies. Thus if an axon fiber passes under a stimulated electrode it will produce a percept in the perceived location of that axon's cell body. We modeled the sensitivity of the ganglion cell axon fibers (green lines in B;
location of the implant with respect to the optic disc inferred from patients' fundus photographs) as decreasing exponentially as a function of distance from the ganglion cell body.

Thus, for each electrode, the heat maps in B describe a 'effective stimulation map' across the retinal surface for each electrode; the expected percept produced by stimulating that electrode.


The remaining stages of the model carry out temporal computations that are fully parallelized in the space domain. For each point in the retina a series of linear (boxes C, D, and F) and nonlinear (box E) computations in the time domain were used to approximate temporal processing within retina and cortex.

As can be seen in the figure above, any given electrode generally only stimulates a small subregion of the retina. As a consequence, when only a few electrodes are active significant speed savings can often be obtained by skipping pixels which will not be significantly stimulated by that electrode, for example pixels whose intensity values in this heat map are less than a certain percent (e.g. 25%) of the largest value. 

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
fit the output to subjective brightness values in a range of [0, 100].
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
- :code:`implants`: Provides implementations of the details of different retinal
  prosthetic implants. This includes Second Sight's Argus I and Argus II implants,
  but can easily be extended to custom implants (see Section on extensibility).
- :code:`retina`: Includes implementation of a model of the retinal distribution of nerve fibers, based on :cite:`JAN09` and an implementation of the temporal cascade of events
  described in equations 1-5. Again this can be easily modified.
- :code:`stimuli`: Includes implementations of commonly used electrical stimulation
  protocols, including means to translate images and movies into simulated
  electrical pulse trains. Again, this can easily be extended to custom stimulation protocols (see Section Extensibility).
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

`implants`
Our goal was to create electrode implant objects that could be configured in a highly flexible manner.  
As far as placement is concerned, an implant can be placed at a particular location on the retina,
with respect to the fovea (microns) and rotated as you see fit. The height of the implant with respect to the tissue (including subretinal vs. epiretinal configuration) can also specified (Are tilted implants specified at the electrode level??):

.. code-block:: python
   :linenos:
   :linenostart: 2

   import numpy as np
   implant = p2p.implants.ArgusI(x_center=-800,
                                 y_center=0,
                                 h=80,
                                 rot=np.deg2rad(35))

The electrodes within the implant can also be specified. An implant is a wrapper around a list of
:code:`p2p.implants.Electrode` objects, which are accessible
via indexing or iteration (e.g., via
:code:`[for i in implant]`). The size and location of each individual electrode within the implant can be specified. Once configured, every Electrode object in the implant can be assigned a name
(in the Argus I implant, they are A1 - A16;
corresponding to the names that are commonly
used by Second Sight Medical Products Inc.).
The first electrode in the implant can be accessed both via its
index (:code:`implant[0]`) and its name (:code:`implant['A1']`).

Once the implant is created, it can be passed to the simulation framework.
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

'retina',

This includes the implementation of a model of the retinal distribution of nerve fibers, based on :cite:`JAN09` and implementations of the temporal cascade of events described in equations 1-5. 

Things that can be set include the spatial sampling rate of the
retina in the optic fiber layer (where the ganglion cell axons are):

.. code-block:: python
   :linenos:
   :linenostart: 9

   ssample = 100  # microns
   sim.set_optic_fiber_layer(sampling=ssample)


Similarly, for the ganglion cell layer we can choose one of the
pre-existing cascade models and specify a temporal sampling rate.

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

It's also possible to specify your own (custom) model, see the section on extensibility below.

At this point, we can visualize the implant's location on the retina with the :code:`sim.plot_fundus` method. 

.. code-block:: python
   :linenos:
   :linenostart: 21

   sim.plot_fundus


'stimuli`
Finally, a stimulation protocol can be specified by assigning
stimuli from the :code:`p2p.stimuli` module to specific
electrodes.
An example is to set up a pulse train of particular stimulation
frequency, current amplitude and duration. Because of safety considerations, all real-world stimuli must be balanced biphasic pulse trains (meaning they must have a positive and negative phase of equal area, so that the net current delivered to the tissue sums to zero).

It is possible to specify a pulse train for each electrode in the implant as follows: 

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

However, since implants are likely to have electrodes numbering in the hundreds or thousands, when assigning pulse trains across multiple electrodes this method will obviously rapidly become cumbersome.

Therefore, an easier way is to assign pulse trains to electrodes
via a dictionary:
??? Code here???

At this point, we can highlight the stimulated electrodes in the array:

.. code-block:: python
   :linenos:
   :linenostart: 21

   sim.plot_fundus(stim)

The output can be seen in Fig. :ref:`figinputoutput` A.

Finally, the created stimulus serves as input to
:code:`sim.pulse2percept`, which is used to convert the
pulse trains into a percept.

Using this model it is possible to generate simulations of the predicted percepts for simple input stimuli, such as a pair of electrodes. It is also possible to generate simulations of what a patient with a prosthetic implant might experience with more complex stimulation patterns, such as stimulation of a grid of electrodes in the shape of the letter E.

At this stage in the model it is possible to consider which retinal layers are included in the temporal model
(e.g., 'OFL': optic fiber layer, 'GCL': ganglion cell layer):
THIS UNCLEAR TOO

.. code-block:: python
   :linenos:
   :linenostart: 22

   # From pulse train to percept
   percept = sim.pulse2percept(stim, tol=0.25,
                               layers=['GCL', 'OFL'])

Here, the output :code:`percept` is a :code:`p2p.utils.TimeSeries`
object that contains the timeseries data in its :code:`data`
container.
This timeseries consists of brightness values (arbitrary units)
for every pixel in the percept image.

`files`

*pulse2percept* offers a collection of functions to convert the :code:`p2p.utils.TimeSeries` output into a movie file (via Scikit-Video and ffmpeg).

CODE HERE

Alternatively, it is possible to retrieve the brightest (mean over all pixels) frame of the timeseries:

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

As described above, this simulation was designed to allow users to generate their own implants,retinal models, and pulse trains. 

Extensibility is provided through class inheritance.

- Retinal implants: Inherit from :code:`p2p.implants.ElectrodeArray`

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

HOW DO YOU DEFINE ELECTRODE SIZE, LOCATION ETC.

- Retinal cell models: Inherit from :code:`p2p.retina.TemporalModel`

Any new ganglion cell model is descriped as a series of temporal operations that are 
carried out on a single pixel of the image.
It must provide a property called :code:`tsample`,
which is the temporal sampling rate,
and a method called :code:`model_cascade`,
which translates a single-pixel pulse train into
a single-pixel percept over time:

.. code-block:: python

   class MyGanglionCellModel(TemporalModel):
       def model_cascade(self, in_arr, pt_list, layers):
           return in_array


This method can then be passed to the simulation framework:

.. code-block:: python

   mymodel = MyGanglionCellModel()
   sim.set_ganglion_cell_layer(mymodel)


- Stimuli: Inherit from :code:`p2p.stimuli.PulseTrain`

THIS SECTION UNCLEAR
Creating new stimuli works the same way. One way of generating novel stimuli is via inheritance
from :code:`pulse2percept.utils.TimeSeries`.
But, you can also inherit
from :code:`pulse2percept.stimuli.MonophasicPulse`,
:code:`pulse2percept.stimuli.BiphasicPulse`,
or :code:`pulse2percept.stimuli.PulseTrain`:

EXAMPLE

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

As described above, the main challenge during *pulse2percept*'s development
was computational cost:
the simulations require a fine subsampling of space,
and span several orders of magnitude in time. In the space domain we wanted the model to be capable of simulating
electrical activation of individual retinal ganglion cells. In the temporal domain the model needed to be capable of 
dealing with pulse trains containing indvidual pulses on the sub-millisecond time 
scale that last over several seconds.
 
Like the brain, we solved this problem through parallelization in the spatial domain. 
After an initial stage that implemented spatial interactions within the retina, computations
were parallelized across small patches of the retina using two back ends (Joblib
:cite:`JOB16` and Dask :cite:`DASK16`), with both multithreading and
multiprocessing options. 

A second major computational bottleneck was computing the
temporal response for each patch of retina. Initial stages of the model require convolutions of arrays (e.g., equations
2 and 3).that describe responses of the model  at high temoral resolution (sampling rates on the order of 25 um) for 
pulse trains lasting for several seconds. These numerically-heavy sections of the code were sped up using a conjunction of
three strategies. First, as described above, any given electrode generally only stimulates a subregion of the retina. As a consequence, when only a few electrodes are active significant speed savings were often be obtained by ignoring pixels which will not be significantly stimulated by that electrode. Second, electrical stimulation is often carried out at relatively low pulse train frequencies of less than 30 Hz. Since the individual pulses within the pulse train are usually very short (~75-450 microseconds), input pulse trains are generally extremely sparse.
We exploited this to speed up computation time by avoiding direct convolution with the
entire time-series whenever possible. Preprocessing of sparse pulse train input arrays allowed us to only carry out temporal convolution for those parts 
of the time-series that included non-zero current amplitudes. 
Finally, these convolutions were sped up wih LLVM-base compilation
implemented using Numba :cite:`LAM15`.


Computational Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

We measured computational performance and scalability. Performance shown here was based on a XX computer, with XXX. 

The inital stage of the model calculates distortions across the retina.  This stage of the model scales as a function of both the number of spatial sampling points in the retina and the spatial sampling of axonal pathways, as shown in Figure 2a. However it should be noted that this stage only needs to be carried out once for a given implant/retina combination. When comparing the effects of different pulse trains a stored map of spatial distortions can be used. 

The remainder of the model is carried out in parallel, so computational time increases linearly with the number of spatial sampling points. Because computations are calculated across patches of the retina, the speed of the model is unaffected by the number of electrodes. The time taken per spatial sampling point depends on a number of factors: the duration of the pulse train, the temporal sampling of the pulse train, and the sparsity of the pulse train input. Figure 2b shows performance as a function of pulse train duration for 10 (very sparse), 60 and 200 Hz pulse trains containing 0.45 ms biphasic pulses (see inset). For each pulse train high and low temporal sampling rates are shown: 0.05 and 0.15 ms. 


.. figure:: figure2.png
   :align: center
   :scale: 50%

   Computational performance. (A) Computational performance for computing spatial distortions. Compute time to generate a 'effective stimulation map' is shown as a function of the number of spatial sampling points used to characterize the retina. The three curves represents three different samplings of ganglion cell axon pathways. (B) Computational performance in the time domain. Compute time for a 1000 patches of retina is shown as a function of pulse train duration for 3 pulse train frequencies (10, 60, 200Hz) at high and low temporal sampling rates (0.05 and 0.15 ms). 
 
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

We present here an open-source, Python-based framework for modeling
the visual processing in retinal prosthesis patients. This software generates a simulation of the perceptual experience of individual prosthetic users - a 'virtual patient'. 

The goal of *pulse2percept* is to provide open-source simulations
that can allow any user to evaluate the perceptual experiences
likely to be produced across both current and future devices devices. 

*pulse2percept* has a number of potential uses.

For device developers, creating "virtual patients" with this software
can facilitate the development of improved pulse stimulation protocols
for existing devices, including generating datasets
for machine learning approaches for finding improved stimulation protocols that minimize spatial and temporal distortions.

"Virtual patients" also provide a useful tool for implant development,
making it possible to rapidly predict vision across
different implant configurations.
We are currently collaborating with two leading manufacturers
to validate the use this software for both of these purposes.

Virtual patients such as these can also play an important role in the wider community. Manufacturer-published 'simulations'
of prosthetic vision do not take account of the substantial neurophysiological distortions
in space and time that are observed in actual patients. As such their predictions of visual outcomes might be misleading to a naive viewer. Any more sophisticated simulations that currently exist are proprietary and not available
to the public or the scientific community.

DO WE WANT THIS PARAGRAPH?Device manufacturers currently develop their own behavioral tests, only test a limited number of patients (who vary widely in age and cognitive abilities),
and some only publish a selective subset of data. Even small differences in task protocols can have significant effects on how patients perform. As a result it has been extremely difficult to evaluate the relative effectiveness of different implants. Simulations such as ours can integrate help differentiate the vision quality provided by different devices.

Prosthetic implants are expensive technology - costing roughly $100k per patient. Currently these implants are reimbursed on a trial basis across many countries in Europe, and are only reimbursed in a subset of states in the USA.    Simulations such as these can help guide government agencies such as the FDA and Medicare in reimbursement decisions.

Most importantly,these simulations can help patients, their families and doctors make an informed choice when deciding at what stage of vision loss a prosthetic device would be helpful. 



Acknowledgments
---------------
This work was supported by the Washington Research Foundation Funds
for Innovation in Neuroengineering and Data-Intensive Discovery (MB),
as well as a grant by the Gordon & Betty Moore Foundation and
the Alfred P. Sloan Foundation to the University of Washington
eScience Institute Data Science Environment (MB and AR).National 
Institute of Health EY-014645 (IF) and EY-12925 (GMB)
The GeForce TITAN X used for this research was donated
by the NVIDIA Corporation, and research credits for cloud computing
were provided by Amazon Web Services.