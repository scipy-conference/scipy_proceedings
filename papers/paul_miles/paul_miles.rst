:author: Paul R. Miles
:email: prmiles@ncsu.edu
:institution: Department of Mathematics
:institution: North Carolina State University

:author: Ralph C. Smith
:email: rsmith@ncsu.edu
:institution: Department of Mathematics
:institution: North Carolina State University
:corresponding:

:bibliography: mybib

--------------------------------------------------------
Parameter Estimation Using the Python Package pymcmcstat
--------------------------------------------------------

.. class:: abstract

   The Python package pymcmcstat \cite{pymcmcstat2018v1.6.0} provides a robust
   platform for a variety of engineering inverse problems.  Bayesian
   statistical analysis is a powerful tool for parameter estimation,
   and many algorithms exist for numerical approaches that utilize
   Markov Chain Monte Carlo (MCMC) methods \cite{smith2014uncertainty}.
   
   In pymcmcstat, the user is provided with a suite of Metropolis based
   algorithms, with the primary approach being Delayed Rejection Adaptive
   Metropolis (DRAM) \cite{haario2006dram, haario2001adaptive}.  A simple
   procedure of adding data, defining model parameters and settings, and
   setting up simulation options provides the user with a wide variety of
   computational tools for considering inverse problem.  This approach to
   inverse problems utilizes data to provide insight into model limitations
   and provide accurate estimation of the underlying model and observation
   uncertainty. 

   As many Python packages currently exist for performing MCMC simulations,
   we had several goals in developing this code.  To our knowledge, no
   current package contains the $n$-stage delayed rejection algorithm,
   so pymcmcstat was intended to fill this gap.  Furthermore, many
   researchers in our community have extensive experience using the MATLAB
   toolbox mcmcstat.  Our implementation provides a similar user environment,
   while exploiting Python structures.  We hope to decrease dependence on
   MATLAB in academic communities by advertising comparable tools in Python.

   This package has been applied to a wide variety of engineering problems,
   including radiation source localization as well as constitutive model
   development of smart material systems.  This is not an exhaustive listing
   of scientific problems that could be analyzed using pymcmcstat, and more
   details regarding the program methodology can be found via the project
   homepage: https://github.com/prmiles/pymcmcstat/wiki.

   Localization of special nuclear material in urban environments poses a
   very important task with many challenges.  Accurate representation of
   radiation transport in a three-dimensional domain that includes various
   forms of construction materials presents many computational challenges.
   For a representative domain in Ann Arbor, Michigan we can construct
   surrogate models using machine learning algorithms based on Monte Carlo
   N-Particle (MCNP) simulations.  The surrogate models provide a
   computationally efficient approach for subsequent inverse model
   calibration, where we consider the source location ($x, y, z$) as our
   model parameters.  We will demonstrate the viability of using pymcmcstat
   for localization problems of this nature.

   Many smart material systems depend on robust constitutive relations for
   applications in robotics, flow control, and energy harvesting.  To fully
   characterize the material or system behavior, uncertainty in the model must
   be accurately represented.  By using experimental data in conjunction with
   pymcmcstat, we can estimate the model parameter distributions and visualize
   how that uncertainty propagates through the system.  We will consider
   specific examples in viscoelastic modeling of dielectric elastomers and
   also continuum approximations of ferroelectric monodomain crystal
   structures.

.. class:: keywords

   Markov Chain Monte Carlo (MCMC), Delayed Rejection Adaptive Metropolis (DRAM),
   Parameter Estimation, Bayesian Inference

Introduction
------------

Twelve hundred years ago  |---| in a galaxy just across the hill...

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
justo vitae odio tincidunt tempus at vitae tortor.

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
   
Inline code looks like this: :code:`chunk of code`.

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

   This is the caption.:code:`chunk of code` inside of it. :label:`egfig` 

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).
   This caption also has :code:`chunk of code`.

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it.  :label:`egfig2`

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

References
----------
