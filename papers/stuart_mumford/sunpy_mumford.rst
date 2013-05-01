:author: StuartMufmord
:email: stuart@mumford.me.uk
:institution: The University of Sheffield


----------------------------------
SunPy: Python for Solar Physicists
----------------------------------

.. class:: abstract

	SunPy aims to become a comprehensive package for solar data analysis and 
	processing.

.. class:: keywords

   Python, Solar Physics, Scientific Python

Introduction
------------


 
Example rst code:
-----------------

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b
       
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

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

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

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


