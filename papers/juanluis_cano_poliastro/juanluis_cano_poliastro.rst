:author: Juan Luis Cano Rodríguez
:email: hello@juanlu.space
:orcid: 0000-0002-2187-161X
:corresponding:

:author: Mark Anthony
:email: mark37@rome.it
:institution: Egyptian Embassy, S.P.Q.R.

:author: Jarrod Millman
:email: millman@rome.it
:institution: Egyptian Embassy, S.P.Q.R.
:institution: Yet another place, S.P.Q.R.

:author: Brutus
:email: brutus@rome.it
:institution: Unaffiliated
:bibliography: mybib


:video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------------------
poliastro: a Python library for interactive Astrodynamics
---------------------------------------------------------

.. todo::
   Consider everything to be confirmed, from section titles to actual content.

.. class:: abstract

   A short version of the long version that is way too long to be written as a
   short version anyway.  Still, when considering the facts from first
   principles, we find that the outcomes of this introspective approach is
   compatible with the guidelines previously established.

   In such an experiment it is then clear that the potential for further
   development not only depends on previous relationships found but also on
   connections made during exploitation of this novel new experimental
   protocol.

.. class:: keywords

   astrodynamics, orbital mechanics, orbit propagation, orbit visualization, two-body problem

Introduction
------------

Astrodynamics
+++++++++++++

Astrodynamics as the branch of Mechanics that studies practical problems
concerning the motion of rockets and other artificial objects through space.

.. todo::
   Citation needed

Since in 1609 German mathematician and astronomer Johannes Kepler published his book *Astronomia nova*,
containing the most famous of all transcendental equations,
the motion of the celestial bodies has attracted the attention of the greatest minds in human history,
even sparking entire new fields in mathematics [Bat99Int]_.
It is easy to imagine that if even Kepler's equation,
the one that captures the essence of the two-body problem in its most restricted form,
already has this mathematical intricacy,
any further development will carry away similar or greater complexity.

.. todo::
   Use less evocative language?

.. math::

   M = E - e \sin{E}

.. todo::
   Make this a figure, rather than an equation?

Almost three centuries later, in 1903, Russian rocket scientist Konstantin E. Tsiolkovsky
first explained in his article *Exploration of Outer Space by Means of Rocket Devices*
precise conditions for artificial objects to reach the orbit of the Earth,
making a huge leap from the mere observation of the celestial bodies
and the science fiction stories that had inspired him
to the real possibility of going to space.

.. Regarding Saxon genitive and equation names, see http://english.stackexchange.com/a/301270/20057

.. math::

   \Delta v = v_e \ln \frac{m_0}{m_f}

.. todo::
   Make this a figure, rather than an equation?

Tsiolkovsky's contribution could be considered the starting point of Astrodynamics,
and many others ensued before they could be tested in practice during the second half of the 20th century.
In 1919 Yuri V. Kondratyuk conceived the gravitational slingshot or flyby
to accelerate a spacecraft through interplanetary flight
and suggested a mission profile for a Lunar landing \cite{siddiqi2000challenge},
in 1925 Walter Hohmann conjectured
that the minimum-fuel transfer between two coplanar circular orbits
consists of two tangent impulses along the line of apses
(although this result was not proved until almost forty years later in \cite{lawden1963optimal}),
and in 1926 Hermann J. Oberth observed
that the velocity gain of an impulsive maneuver
is higher when the kinetic energy is maximum
(nowadays known as the Oberth effect).
The severe limitations in weight and available energy for such kind of travels
were already apparent for these pioneers,
who were, in some way, anticipating the need to optimize on board fuel consumption.

.. todo::
   This whole paragraph is nice but it was used to justify the importance of low-thrust,
   we should reword it.

.. todo::
   Add more background on
   (1) the initial value two-body problem (propagation),
   (2) the boundary value two-body problem (initial orbit determination), and
   (3) analytical continuous thrust guidance laws,
   including modern references to research about these topics.
   Leave software references for later.

.. todo::
   Discuss the differences between real-world Earth satellite propagation with SGP4
   from more generic Astrodynamics work.

.. todo::
   Discuss software related to Astrodynamics,
   including classical, well-stablished open-source toolboxes like SPICE
   (does SPICE have propagation?),
   GUI-based software like GMAT and gpredict,
   and more modern initiatives like Skyfield.

State of the art
++++++++++++++++

Three main problems with Astrodynamics software:

1. Lack of reproducibility/"code available upon request"
2. Existing software requires deep expertise and has some implicit assumptions
   (like coordinate frame names etc)
3. There is no "scripting" alternative for Astrodynamics

Three main motives for poliastro existence:

1. Set an example on reproducibility and good coding practices in Astrodynamics
2. Become an approachable software even for novices
3. Offer an scripting interface

Other ideas:

- Common misconceptions (reference frames! TLE propagation! Mean anomaly!)

Methods
-------

Background
++++++++++

.. todo::
   Describe separately propagation, IOD, and continuous thrust.

Software Architecture
+++++++++++++++++++++

.. todo::
   Two-layered architecture, `poliastro.twobody` vs `poliastro.ephem`

poliastro usage
---------------

.. todo::
   Pick a few examples from the user guide
   and relevant notebooks from the gallery.

Future work
-----------

.. todo::
   Limitations and shortcomings of poliastro
   Technical: bad APIs, inconsistencies.
   Non-technical: Lack of development time/sustainability model beyond GSOC money and NumFOCUS grants,
   licensing concerns, reusability in the wider ecosystem.

On reusability:

- So-so: IBM/spacetech-ssa, AnalyticalGraphicsInc/STKCodeExamples
- Did not reuse: sbpy, beyond, mubody

On sustainability:

Several companies seem to use it, but there is no two-way communication.

Conclusions
-----------

poliastro is cool and nice,
it has some unique features,
and is decently fast
(and hopefully getting faster).
It does have some limitations
(both technical and non-technical)
that can be addressed with more development time.

---

Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension. 

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work. 

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references. 

So what would previously have registered as a in text reference ``[Atr03]_`` for 

:: 

     [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

what you actually see will be an empty reference rendered as **[?]**.

E.g., [Atr03]_.


If you wish to have a block quote, you can just indent the text, as in 

    When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication. :cite:`hume48`

Dois in bibliographies
++++++++++++++++++++++

In order to include a doi in your bibliography, add the doi to your bibliography
entry as a string. For example:

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = "10.1017/CBO9780511808432",
   }


If there are errors when adding it due to non-alphanumeric characters, see if
wrapping the doi in ``\detokenize`` works to solve the issue.

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = \detokenize{10.1017/CBO9780511808432},
   }

Source code examples
--------------------

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

References
----------
.. [Bat99Int] Battin, Richrd H.. *An Introduction to the Mathematics and Methods of Astrodynamics,
              Revised Edition*. AIAA Education Series. 1999.
.. [Val07Fun] Vallado, David A.. *Fundamentals of Astrodynamics and Applications (3rd Edition)*.
              Microcosm Press/Springer. May 2007.
