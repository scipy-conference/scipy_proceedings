:author: Kristen M. Thyng
:email: kthyng@tamu.edu
:institution: Texas A&M University

:author: Robert D. Hetland
:email: hetland@tamu.edu
:institution: Texas A&M University

.. :author: Jarrod Millman
.. :email: millman@rome.it
.. :institution: Egyptian Embassy, S.P.Q.R.

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

-----------------------------------------------------------------
TracPy: Wrapping the FORTRAN Lagrangian trajectory model TRACMASS
-----------------------------------------------------------------

.. class:: abstract

   abstract

   .. A short version of the long version that is way too long to be written as a
   .. short version anyway.  Still, when considering the facts from first
   .. principles, we find that the outcomes of this introspective approach is
   .. compatible with the guidelines previously established.

   .. In such an experiment it is then clear that the potential for further
   .. development not only depends on previous relationships found but also on
   .. connections made during exploitation of this novel new experimental
   .. protocol.

.. class:: keywords

   keywords

   .. terraforming, desert, numerical perspective

Introduction
------------

.. introduce and motivate Lagrangian tracking

Drifters are used in oceanography and atmospherics *in situ* in order to demonstrate flow patterns in the fluid they are in. For example, in the ocean, drifters will often be deposited on the surface and allowed to passively be transported with the flow, reporting their location via GPS at regular intervals. In this way, drifters are gathering data in a Lagrangian perspective. GIVE EXAMPLES OF IN SITU DRIFTER DATA AND WHAT CAN BE LEARNED

Lagrangian trajectory modeling is a method of moving parcels through a fluid based on numerically modeled circulation fields. GIVE EXAMPLES OF NUMERICAL DRIFTERS AND WHAT CAN BE LEARNED

Numerical drifters may be calculated online, while a circulation model is running, in order to use the highest resolution model-predicted velocity fields available in time (on the order of seconds to minutes). Often, however, Lagrangian trajectories are calculated offline, using the velocity fields at the stored temporal resolution (on the order of minutes to hours). 

A given drifter's trajectory is calculated using velocity fields that are not perfectly resolved in space, either, given any numerical model grid's spatial resolution. To move the drifter, the velocity fields must be extended to the drifter's location, which in general will not be colocated with all necessary velocity information. Many Lagrangian trajectory models use interpolation in space to accomplish this, and may use low or high orders of interpolation. The algorithm discussed in this work has a somewhat different approach.


.. introduce TRACMASS with links to places it has been used

TRACMASS is a Lagrangian trajectory model that runs natively on velocity fields that have been calculated on a staggered Arakawa C grid. Originally written about 2 decades ago, it has been used in many applications. APPLICATIONS OF TRACMASS

.. introduce TracPy

The core algorithm for TRACMASS is written in Fortran for speed, and has been wrapped in Python for increased usability. This code package together is called TracPy. LINK TO GITHUB AND DOI


TRACMASS
--------

.. Explain algorithm

The TRACMASS algorithm for stepping numerical drifters in space is distinct from many algorithms because it runs natively on a staggered Arakawa C grid. This grid is used in ocean modeling codes, including the Regional Ocean Modeling System (ROMS) CITE, MITGCM CITE, and HyCOM CITE. In the staggered Arakawa C grid, the west-east or zonal velocity, :math:`u`, is located at the west and east walls of a grid cell; the north-south or meridional velocity, :math:`v`, is located at the north and south walls; and the vertical velocity, :math:`w`, is located at the vertically top and bottom cell walls (Figure :ref:`tracmass1`). The drifter is stepped as follows:

1. something

2. something else

.. figure:: tracmass1.pdf
   :scale: 40%

   A single rectangular grid cell is shown in :math:`x-y`. Zonal :math:`u(v)` velocities are calculated at the east/west (north/south) cell walls. In the vertical direction, :math:`w` velocities are calculated at the top and bottom cell walls. :label:`tracmass1`

.. Explain options like subgrid diffusion, time interpolation, and time-dependent algorithm


TracPy
------

.. Explain approach


.. Explain existing level of usage


.. Future work



Conclusions
-----------



.. Twelve hundred years ago  |---| in a galaxy just across the hill...

.. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
.. tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
.. sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
.. tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
.. eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
.. adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
.. ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
.. purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
.. quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
.. nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
.. Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
.. justo vitae odio tincidunt tempus at vitae tortor.

.. Of course, no paper would be complete without some source code.  Without
.. highlighting, it would look like this::

..    def sum(a, b):
..        """Sum two numbers."""

..        return a + b

.. With code-highlighting:

.. .. code-block:: python

..    def sum(a, b):
..        """Sum two numbers."""

..        return a + b

.. Maybe also in another language, and with line numbers:

.. .. code-block:: c
..    :linenos:

..    int main() {
..        for (int i = 0; i < 10; i++) {
..            /* do something */
..        }
..        return 0;
..    }

.. Or a snippet from the above code, starting at the correct line number:

.. .. code-block:: c
..    :linenos:
..    :linenostart: 2

..    for (int i = 0; i < 10; i++) {
..        /* do something */
..    }
 
.. Important Part
.. --------------

.. It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
.. some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
.. equation on a separate line:

.. .. math::

..    g(x) = \int_0^\infty f(x) dx

.. or on multiple, aligned lines:

.. .. math::
..    :type: eqnarray

..    g(x) &=& \int_0^\infty f(x) dx \\
..         &=& \ldots


.. The area of a circle and volume of a sphere are given as

.. .. math::
..    :label: circarea

..    A(r) = \pi r^2.

.. .. math::
..    :label: spherevol

..    V(r) = \frac{4}{3} \pi r^3

.. We can then refer back to Equation (:ref:`circarea`) or
.. (:ref:`spherevol`) later.

.. Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
.. consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
.. lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
.. turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
.. adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
.. arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
.. faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
.. consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
.. orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
.. sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
.. lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
.. volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
.. pulvinar id metus.

.. .. figure:: figure1.png

..    This is the caption. :label:`egfig`

.. .. figure:: figure1.png
..    :align: center
..    :figclass: w

..    This is a wide figure, specified by adding "w" to the figclass.  It is also
..    center aligned, by setting the align keyword (can be left, right or center).

.. .. figure:: figure1.png
..    :scale: 20%
..    :figclass: bht

..    This is the caption on a smaller figure that will be placed by default at the
..    bottom of the page, and failing that it will be placed inline or at the top.
..    Note that for now, scale is relative to a completely arbitrary original
..    reference size which might be the original size of your image - you probably
..    have to play with it. :label:`egfig2`

.. As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
.. figures.

.. .. table:: This is the caption for the materials table. :label:`mtable`

..    +------------+----------------+
..    | Material   | Units          |
..    +------------+----------------+
..    | Stone      | 3              |
..    +------------+----------------+
..    | Water      | 12             |
..    +------------+----------------+
..    | Cement     | :math:`\alpha` |
..    +------------+----------------+


.. We show the different quantities of materials required in Table
.. :ref:`mtable`.


.. .. The statement below shows how to adjust the width of a table.

.. .. raw:: latex

..    \setlength{\tablewidth}{0.8\linewidth}


.. .. table:: This is the caption for the wide table.
..    :class: w

..    +--------+----+------+------+------+------+--------+
..    | This   | is |  a   | very | very | wide | table  |
..    +--------+----+------+------+------+------+--------+


.. Perhaps we want to end off with a quote by Lao Tse:

..   *Muddy water, let stand, becomes clear.*


.. .. Customised LaTeX packages
.. .. -------------------------

.. .. Please avoid using this feature, unless agreed upon with the
.. .. proceedings editors.

.. .. ::

.. ..   .. latex::
.. ..      :usepackage: somepackage

.. ..      Some custom LaTeX source here.

.. References
.. ----------
.. .. [Atr03] P. Atreides. *How to catch a sandworm*,
..            Transactions on Terraforming, 21(3):261-300, August 2003.


