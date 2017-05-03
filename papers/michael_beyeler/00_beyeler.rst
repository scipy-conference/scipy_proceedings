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
perceptual experience [Fin15]_.
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

Here's how the model works.




Implementation and Results
--------------------------

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

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
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


References
----------
.. [Fin15] I. Fine and G. M. Boynton. *Pulse trains to percepts: the challenge of creating a perceptually intelligible world with sight recovery technologies*, Philos Trans R Soc Lond B Biol Sci 370(1677): 20140208, doi:`10.1098/rstb.2014.0208 <http://dx.doi.org/10.1098/rstb.2014.0208>`_.

.. [Hor09] A. Horsager, S. H. Greenwald, J. D. Weiland, M. S. Humayun, R. J. Greenberg, M. J. McMahon, G. M. Boynton, and I. Fine. *Predicting visual sensitivity in retinal prosthesis patients*, Invest Ophthalmol Vis Sci 50(4): 1483-1491, doi:`10.1167/iovs.08-2595 <http://dx.doi.org/10.1167/iovs.08-2595>`_.

.. [Hor11] A. Horsager, G. M. Boynton, R. J. Greenberg, and I. Fine. *Temporal interactions during pairedelectrode stimulation in two retinal prosthesis subjects*, Invest Ophthalmol Vis Sci 52(1): 549-557, doi:`10.1167/iovs.10-5282 <http://dx.doi.org/10.1167/iovs.10-5282>`_.


