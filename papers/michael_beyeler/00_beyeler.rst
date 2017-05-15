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
Both diseases begin with the degeneration of photoreceptors in the retina,
though in later stages other retinal cells
(such as bipolar, amacrine, and ganglion cells) are affected as well.
Despite a significant loss of retinal ganglion cells in advanced stages
of RP and AMD,
their morphological structure and connections to the optic nerve
seem to be relatively well maintained :cite:`Humayun1999,Mazzoni2008`.
This has led researchers to develop implanatable microelectronic
visual prostheses that, analogous to cochlear implants,
directly stimulate remaining retinal neurons with electrical current.
The ultimate goal of most implants is to generate useful vision
in blind patients by transforming visual information into a spatial
and temporal sequence of electrical pulses
(see Fig. :ref:`retinalimplants`).

.. figure:: figure1.jpg
   :align: center
   :scale: 70%

   A schematic overview of an epiretinal prosthesis.
   TODO Should probably redo the figure to avoid copyright issue with
   Boston Retinal Implant group.
   :label:`retinalimplants`

To date, two different retinal prosthesis systems are already approved
for commercial use in patients across the US and Europe.
The Argus II device (Second Sight Medical Products Inc., :cite:`daCruz2016`)
is an `epiretinal` prosthesis,
which is placed on top of the retinal surface,
above the optic fiber layer;
thus directly stimulating retinal ganglion cells
while bypassing other retinal layers.
The Alpha-IMS system (Retina Implant AG, :cite:`Stingl2015`),
on the other hand, is a `subretinal` device,
which is placed on the outer surface of the retina,
between the photoreceptor layer and the retinal pigment epithelium;
thus directly stimulating retinal bipolar cells.
At the same time, a number of other devices have either started
or are planning to start clinical trials in the near future,
potentially offering a wide range of sight restoration options
for blinded individuals within a decade :cite:`Fine2015`.

.. figure:: figure1.eps
   :align: center
   :figclass: w
   :scale: 35%

   Full model cascade. TODO explain.
   :label:`figmodel`

However, clinical experience with existing retinal prostheses make it
apparent that the vision provided by these devices differs substantially
from normal sight.
Patients report the experience of prosthetic vision as being like
`"looking at the night sky where you have millions of twinkly lights
that almost look like chaos"`
:cite:`PioneerPress2015`.
Patients report perceptual distortions of the visual imagery created
by these devices in both space and time:
For example, stimulating even a single electrode leads to percepts
that vary dramatically in shape
(e.g., varying in description from "blobs", to "streaks" and "half-moons")
and duration (e.g., fading over time).
These perceptual distortions are thought to result from interactions
between implant electronics and the underlying neurophysiology
:cite:`FineBoynton2015,Beyeler2017`,
but the exact mechanisms remain poorly understood.
Therefore, in order to create perceptually meaningful vision,
it is necessary to predictly generate a range of brightness levels
over both space and time.

.. Clinical experience with these implants shows that these are still early days,
.. with current technologies resulting in nontrivial distortions of the
.. perceptual experience :cite:`FineBoynton2015`.

.. Here we present *pulse2percept*, an open-source Python implementation
.. of a computational model that can predict the perceptual experience
.. of retinal prosthesis patients across a wide range of
.. implant configurations.

We have developed a computational model of bionic vision that simulates
the perceptual experience of retinal prosthesis patients
across a wide range of implant configurations.
Here we present an open-source implementation of these models as part of
*pulse2percept*, a Python-based simulation framework that relies solely on
open-source contributions of the NumPy/SciPy stacks and the broader
Python community.
The model has been validated against human pyschophysical data,
and generalizes across individual electrodes, patients, and devices.

The remainder of this paper is organized as follows:
explain the computational model,
talk about implementation details,
show some results,
discuss and conclude.


Methods
-------

We developed a model that uses similar math as cochlear implants
:cite:`Horsager2009,Nanduri2012`.
Model parameters were fit to psychophysical data such as
threshold data and patient drawings.
Detailed methods can be found in the above two papers,
here we give a brief overview.

The full model cascade for an Argus I epiretinal prosthesis is illustrated in
Fig. :ref:`figmodel`, although this model generalizes to other epiretinal
and subretinal configurations.

The device consists of electrodes of 260 um or 520 um
diameter arranged in a checkerboard pattern (Fig. :ref:`figmodel` A).
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
and colvolving this accumulation with a second one-stage gamma function
(:math:`n=1`, time constant :math:`tau_2 = 45.3` ms;
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
(:math:`n = 3`, :math:`tau_3 = 26.3` ms)
intended to model slower perceptual processes in the brain
(:ref:`figmodel` F):

.. math::
   :label: eqslow

   r_4(s,t) = \epsilon_2 r_3(s,t) * \delta(t, 3, \tau_3),

where :math:`epsilon_2 = 1000` was a scaling factor used to
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

The code is organized into different submodules:

- :code:`api`: The API
- :code:`retina`: All retinal stuff
- :code:`implants`: All implants
- :code:`stimuli`: All stimuli
- :code:`files`: All I/O
- :code:`utils`: All utility functions

A minimal usage example is given in the listing below:

.. code-block:: python

   import pulse2percept as p2p

   # Place an Argus II array centered over the fovea
   implant = p2p.implants.ArgusII(x_center=0,
                                  y_center=0)

   # Start the stimulation framework, select joblib
   # backend
   sim = p2p.Simulation(implant, engine='joblib',
                        num_jobs=8)

   # Set optional parameters of the different retinal
   # layers; e.g, spatial sampling (`ssample`) and
   # temporal sampling rate (`tsample`)
   ssample = 100  # microns
   tsample = 0.005 / 1000  # seconds
   sim.set_optic_fiber_layer(sampling=ssample)
   sim.set_ganglion_cell_layer(model='Nanduri2012',
                               tsample=tsample)

   # Generate a stimulus: Biphasic pulse, 20 uA, 50 Hz,
   # 0.5 second duration
   pt = p2p.stimuli.PulseTrain(tsample, freq=50, amp=20,
                               dur=0.5)
   stim = {'E1': pt}

   # From pulse train to percept
   percept = sim.pulse2percept(stim, tol=0.25,
                               layers=['GCL', 'OFL'])



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
       def model_cascade(self, ecv):
           pass


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




We can create new stimuli:

Some implementation details and some results.

The main challenge during *pulse2percept*'s development
was computational cost:
the simulations require a fine subsampling of space,
and span several orders of magnitude in time,
ranging from electrical activation of individual retinal ganglion cells
on the sub-millisecond time scale to visual perception occurring
over several seconds.

Like the brain, we solved this problem through parallelization.
Computations were parallelized across small patches of the retina
using two back ends (Joblib and Dask),
with both multithreading and multiprocessing options.
Math-heavy sections of the code were additionally sped up using
just-in-time compilation (Numba).




Discussion
----------

*pulse2percept* has a number of potential uses.

For device developers, creating “virtual patients” with this software
can facilitate the development of improved pulse stimulation protocols
for existing devices, including generating datasets
for machine learning approaches.
“Virtual patients” are also a useful tool for device development,
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
to the scientific community, and manufacturer-published ‘simulations’
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
by the NVIDIA Corporation.




