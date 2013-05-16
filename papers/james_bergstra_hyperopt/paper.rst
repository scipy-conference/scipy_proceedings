:author: James Bergstra
:email: james.bergstra@uwaterloo.ca
:institution: University of Waterloo

:author: Dan Yamins
:email: yamins@mit.edu
:institution: Massachusetts Institute of Technology

:author: David D. Cox
:email: davidcox@fas.harvard.edu
:institution: Harvard University


-------------------------------------------------------------------------------------------
Hyperopt: A Python Library for Optimizing the Hyperparamters of Machine Learning Algorithms
-------------------------------------------------------------------------------------------

.. class:: abstract

    The difficulty of manual algorithm configuration causes three problems in empirical machine learning:
    (1) research claims are not easily reproduced by fellow researchers and practitioners,
    (2) good performance does not transfer readily to data sets beyond those actually used to motivate new algorithms, and
    (3) algorithm researchers {\em themselves} may fail to discover how good
    their algorithms can be because they overlook part of the configuration
    space.
    Model-based optimization strategies (i.e. based on Bayesian optimization) have emerged as viable and effective
    alternatives to standard practice of grid-assisted manual search.
    The Hyperopt library (Hyperopt) makes model-based optimization available
    in Python, and provides an implementation of the TPE algorithm, which has
    proven to be an effective strategy for optimizing neural networks, deep
    networks, and computer vision systems.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
------------

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

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
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
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

.. figure:: figure1.png

   This is the caption. :label:`egfig`


.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+-------+
   | Material   | Units |
   +------------+-------+
   | Stone      | 3     |
   +------------+-------+
   | Water      | 12    |
   +------------+-------+

We show the different quantities of materials required in Table
:ref:`mtable`.

Perhaps we want to end off with a quote by Lao Tse:

  *Muddy water, let stand, becomes clear.*


.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


