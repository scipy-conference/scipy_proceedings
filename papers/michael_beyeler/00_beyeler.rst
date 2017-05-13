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

   terraforming, desert, numerical perspective


Introduction
------------

Retinal prostheses aim to recover functional vision in patients
blinded by degenerative retinal diseases,
such as retinitis pigmentosa and age-related macular degeneration,
by electrically stimulating remaining retinal cells.
Clinical experience with these implants shows that these are still early days,
with current technologies resulting in nontrivial distortions of the
perceptual experience :cite:`FineBoynton2015`.
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

.. figure:: figure1.eps
   :align: center
   :figclass: w
   :scale: 35%

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).
   :label:`figmodel`

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
The resulting sensitivity profile (heat map in B) was then convolved with a 
gamma function to model the impulse response function of retinal ganglion cells 
(:math:`\tau = 0.42` ms; C). 
We modeled loss of sensitivity as a function of previous stimulation by 
subtracting accumulated cathodic charge after convolution with a gamma function 
(:math:`\tau = 45.3` ms)
from the output of the retinal ganglion cells (D).
The signal was then half-rectified, passed through a static
nonlinearity (E), and convolved with a slow gamma function intended to model 
slower perceptual processes
(:math:`\tau = 26.3` ms; F). 
The output of the model was a map of brightness values (arbitrary units) over time. 
Subjective brightness was defined as the highest brightness value in the map.




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
   sim.set_ganglion_cell_layer(tsample=tsample)

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
       def __init__(self, tsample, **kwargs):
           self._tsample = tsample

       @property
       def tsample(self):
           return self._tsample

       def model_cascade(self, ecv):
           pass


It can then be passed to the simulation framework:

.. code-block:: python

   mymodel = MyGanglionCellModel(tsample=0.005 / 1000)
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


Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }

Important Part
--------------

It is well known that Spice grows on the planet Dune.  Test
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
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure2.png

   This is the caption. :label:`egfig`



.. figure:: figure2.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
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

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.


Acknowledgments
---------------
This work was supported by the Washington Research Foundation Funds
for Innovation in Neuroengineering and Data-Intensive Discovery (MB),
as well as a grant by the Gordon & Betty Moore Foundation and
the Alfred P. Sloan Foundation to the University of Washington
eScience Institute Data Science Environment (MB and AR).
The GeForce TITAN X used for this research was donated
by the NVIDIA Corporation.




