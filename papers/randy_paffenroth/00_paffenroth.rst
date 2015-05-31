:author: Randy Paffenroth
:email: rcpaffenroth@wpi.edu
:institution: Worecester Polytechnic Institute, Mathematical Sciences Department and Data Science Program

:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------
Python in Data Science Research and Education
------------------------------------------------

.. class:: abstract

  Here we demonstrate how Python can be used throughout an entire
  lifecycle of a graduate program in Data Science.  In
  interdisciplinary fields, such as Data Science, the students often
  come from a variety of different backgrounds where, for example,
  some students may have strong mathematical training but less
  experience in programming.  Python’s ease of use, open source
  license, and access to a vast array of libraries make it
  particularly suited for such students.  In particular, we will
  discuss how Python, IPython notebooks, scikit-learn, NumPy, SciPy,
  and pandas can be used in several phases of graduate Data Science
  education, starting from introductory classes (covering topics such
  as data gathering, data cleaning, statistics, regression,
  classification, machine learning, etc.) and culminating in degree
  capstone research projects using more advanced ideas such as convex
  optimization, non-linear dimension reduction, and compressed
  sensing.  One particular item of note is the scikit-learn library,
  which provides numerous routines for machine learning.  Having
  access to such a library allows interesting problems to be addressed
  early in the educational process and the experience gained with such
  “black box” routines provides a firm foundation for the students own
  software development, analysis, and research later in their academic
  experience.  This talk will be based upon the authors experience
  teaching in the Worcester Polytechnic Institute Data Science
  Program.

.. class:: keywords

   data science, education, statistics

Introduction
------------

Talk about DS program at WPI Talk about research Base on abstract
Discussion focus on 2 classes with quite different flavor.  DS501
Introduction to Data Science |--| a core data science course that gets
students from a wide variety of backgrounds a baseline set of skills
for the Data Science program.  MA542 Regression Analysis |--| a core
graduate statistics course.

So many libraries that any homework question is probably trivially
answerable if they look hard enough.  Need to be careful that the
ground rules are set correctly.  For example, need to say that they
need to solve the regression problem using the *normal equations*.  It
is ok to debug their code using the black box routine, but they still
need to write their own code.  For exmaple, I insist that they hand in
code.  *Not for grading* but to see how they did it.

A nice side effect is that you can carefully control the difficulty
and focus be saying which parts they do and which parts are ok to be a
black box.  Nice segue into DS501, where we wanted to focus on specific
ideas, but have the problem be interesting.

DS501 Introduction to Data Science
----------------------------------

Basic graduate course, big mix of students!  Strong students, but with
varied backgrounds.  Some experts in mathematics, with more limited
background in programming.  Some experts in programming, with more
limited background in mathematics.

Students needs to gether their hands dirty!

Discuss case studies with code examples.
Look at comments from students!

MA542 Regression Analysis
-------------------------

More advanced class, but perhaps with a greater concentration of
students who are mathematically focused.  Also, may students were
first time Python users, with the majority of the exceptions having 
taken DS501.

Numpy, matplotlib, and Pandas provided almost all of the functionality
they needed for the bulk of the class.   Even though book was more focused
on things like SAS and SPSS (double check book to make sure).

Were able to focus on the mathematics and not have the language, get
in the way.
Look at comments from students!

More advanced research
----------------------

Convex optimization, deep learning, large scale robust PCA (be careful to 
describe just the right amount), graphical models, communitie analysis,
supervised learning in BGP data.
Yes, they are all related at a deep mathematical level, but I won't bore you 
with the details.

Libraries available for them all!

Also discuss Turing with pycuda and mpi2py.

Finally, discuss manifold learning, and show 3D visualization using mayavi
of the WPI logo embedded in a non-linear manifold.  Make it colorful.
Brings all the pieces together.  Just looking for good Ph.D. student to
work on.

Conclusion
----------
Python rocks!
It can be used at all levels, and each level builds on the previous one.
There is such a broad array of libraries available in Data Science (or 
whatever you want to call it) that students can focus on what is important
to them.

Sample Stuff
------------
Twelve hundred years ago  |---| in a galaxy just across the hill...

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
 
Sample Stuff 2
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

In tellus metus, elementum vitae tincidunt ac, volutpat sit amet
mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
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

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


Stajano, Frank. "Python in education: Raising a generation of native speakers." Proceedings of 8 th International Python Conference. 2000.

Myers, Christopher R., and James P. Sethna. "Python for education: Computational methods for nonlinear systems." Computing in Science & Engineering 9.3 (2007): 75-79.
