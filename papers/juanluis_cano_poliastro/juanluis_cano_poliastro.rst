:author: Juan Luis Cano Rodríguez
:email: hello@juanlu.space
:orcid: 0000-0002-2187-161X
:institution: Unaffiliated
:corresponding:

:author: Jorge Martínez Garrido
:email: contact@jorgemartinez.space
:institution: Unaffiliated

:bibliography: refs


:video: https://www.youtube.com/watch?v=VCpTgU1pb5k

=========================================================
poliastro: a Python library for interactive Astrodynamics
=========================================================

.. class:: abstract

   Space is more popular than ever, with the growing public awareness of interplanetary scientific missions,
   as well as the increasingly large number of satellite companies planning to deploy satellite constellations.
   Python has become a fundamental technology in the astronomical sciences,
   and it has also caught the attention of the Space Engineering community.

   One of the requirements for designing a space mission is
   studying the trajectories of satellites, probes, and other artificial objects,
   usually ignoring non-gravitational forces or treating them as perturbations:
   the so-called n-body problem.
   However, for preliminary design studies and most practical purposes,
   it is sufficient to consider only two bodies: the object under study and its attractor.

   Even though the two-body problem has many analytical solutions,
   orbit propagation (the initial value problem) and targeting (the boundary value problem)
   remain computationally intensive because of long propagation times, tight tolerances, and vast solution spaces.
   On the other hand, Astrodynamics researchers often do not share
   the source code they used to run analyses and simulations,
   which makes it challenging to try out new solutions.

   This paper presents poliastro, an open-source Python library for interactive Astrodynamics
   that features an easy-to-use API and tools for quick visualization.
   poliastro implements core Astrodynamics algorithms
   (such as the resolution of the Kepler and Lambert problems)
   and leverages numba, a Just-in-Time compiler for scientific Python,
   to optimize the running time.
   Thanks to Astropy, poliastro can perform seamless coordinate frame conversions
   and use proper physical units and timescales.
   At the moment, poliastro is the longest-lived Python library for Astrodynamics,
   has contributors from all around the world,
   and several New Space companies and people in academia use it. 

.. class:: keywords

   astrodynamics, orbital mechanics, orbit propagation, orbit visualization, two-body problem

Introduction
============

Astrodynamics
-------------

Astrodynamics is the branch of Mechanics that studies practical problems
concerning the motion of rockets and other artificial objects through space.

.. note::
   Citation needed

Since in 1609 German mathematician and astronomer Johannes Kepler published his book *Astronomia nova*,
containing the most famous of all transcendental equations,
the motion of the celestial bodies has attracted the attention of the greatest minds in human history,
even sparking entire new fields in mathematics :cite:`battin_introduction_1999`.
It is easy to imagine that if even Kepler's equation (:ref:`eq:kepler`),
the one that captures the essence of the two-body problem in its most restricted form,
already has this mathematical intricacy,
any further development will carry away similar or greater complexity.

.. note::
   Use less evocative language?

.. raw:: latex

   \begin{figure}
   \[ M = E - e \sin{E} \]
   \caption{The Kepler equation}
   \label{eq:kepler}
   \end{figure}

Almost three centuries later, in 1903, Russian rocket scientist Konstantin E. Tsiolkovsky
first explained in his article *Exploration of Outer Space by Means of Rocket Devices*
precise conditions for artificial objects to reach the orbit of the Earth,
making a huge leap from the mere observation of the celestial bodies
and the science fiction stories that had inspired him
to the real possibility of going to space.

.. Regarding Saxon genitive and equation names, see http://english.stackexchange.com/a/301270/20057

.. raw:: latex

   \begin{figure}
   \[ \Delta v = v_e \ln \frac{m_0}{m_f} \]
   \caption{The Tsiolkovsky equation}
   \label{eq:tsiolkovsky}
   \end{figure}

Tsiolkovsky's contribution could be considered the starting point of Astrodynamics,
and many others ensued before they could be tested in practice during the second half of the 20th century.
In 1919 Yuri V. Kondratyuk conceived the gravitational slingshot or flyby
to accelerate a spacecraft through interplanetary flight
and suggested a mission profile for a Lunar landing :cite:`siddiqi_challenge_2000`,
in 1925 Walter Hohmann conjectured
that the minimum-fuel transfer between two coplanar circular orbits
consists of two tangent impulses along the line of apses
(although this result was not proved until almost forty years later in :cite:`lawden_optimal_1963`),
and in 1926 Hermann J. Oberth observed
that the velocity gain of an impulsive maneuver
is higher when the kinetic energy is maximum
(nowadays known as the Oberth effect).
The severe limitations in weight and available energy for such kind of travels
were already apparent for these pioneers,
who were, in some way, anticipating the need to optimize on board fuel consumption.

.. note::
   This whole paragraph is nice but it was used to justify the importance of low-thrust,
   we should reword it.

.. note::
   Add more background on
   (1) the initial value two-body problem (propagation),
   (2) the boundary value two-body problem (initial orbit determination), and
   (3) analytical continuous thrust guidance laws,
   including modern references to research about these topics.
   Leave software references for later.

.. note::
   Discuss the differences between real-world Earth satellite propagation with SGP4
   from more generic Astrodynamics work.

.. note::
   Discuss software related to Astrodynamics,
   including classical, well-stablished open-source toolboxes like SPICE
   (does SPICE have propagation?),
   GUI-based software like GMAT and gpredict,
   and more modern initiatives like Skyfield.

State of the art
----------------

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
=======

Background
----------

.. note::
   Describe separately propagation, IOD, and continuous thrust.

In the context of the two-body motion,
there are six parameters that uniquely determine an orbit,
plus the gravitational parameter of the corresponding attractor (:math:`k` or :math:`\mu`).
Optionally, an epoch that contextualizes the orbit can be included as well.
This set of six parameters is not unique,
and several of them have been developed over the years to serve different purposes.
The most widely used ones are:

- **Cartesian elements**: Three components for the position :math:`(x, y, z)`
  and three components for the velocity :math:`(v_x, v_y, v_z)`.
  This set has no singularities.
- **Classical Keplerian elements**: Two components for the shape of the conic
  (usually the semimajor axis :math:`a` or semiparameter :math:`p` and the eccentricity :math:`e`),
  three Euler angles for the orientation of the orbital plane in space
  (inclination :math:`i`, right ascension of the ascending node :math:`\Omega`, and argument of periapsis :math:`\omega`),
  and one polar angle for the position of the body along the conic
  (usually true anomaly :math:`f` or :math:`\nu`).
  This set of elements has an easy geometrical interpretation
  and the advantage that, in pure two-body motion,
  five of them are fixed :math:`(a, e, i, \Omega, \omega)`
  and only one is time-dependent (:math:`\nu`),
  which greatly simplifies the analytical treatment of orbital perturbations.
  However, they suffer from singularities steming from the Euler angles ("gimbal lock")
  and equations expressed in them are ill-conditioned near such singularities.
- **Walker modified equinoctial elements**: Six parameters :math:`(p, f, g, h, k, L)`.
  Only :math:`L` is time-dependent and this set has no singularities,
  however the geometrical interpretation of the rest of the elements is lost.

Software Architecture
---------------------

The architecture of poliastro emerges from the following set of conflicting requirements:

1. There should be a high-level API that enables users to perform orbital calculations
   in a straightforward way and prevent typical mistakes.
2. The running time of the algorithms should be within the same order of magnitude
   of existing compiled implementations.
3. The library should be written in a popular open-source language
   to maximize adoption and lower the barrier to external contributors.

One of the most typical mistakes we set ourselves to prevent with the high-level API
is dimensional errors. Addition and substraction operations of physical quantities
are defined only for quantities with the same units :cite:`drobot_foundations_1953`:
for example, the operation :math:`1~\text{km} + 100~\text{m}`
requires a scale transformation of at least one of the operands,
since they have different units (kilometers and meters) but the same dimension (length),
whereas the operation :math:`1~\text{km} + 1~\text{kg}` is directly not allowed
because dimensions are incompatible (length and mass).
As such, software systems operating with physical quantities
should raise exceptions when adding different dimensions,
and transparently perform the required scale transformations
when adding different units of the same dimension.

With this in mind, we evaluated several Python packages for unit handling
(see :cite:`j_goldbaum_unyt_2018` for a recent survey) and chose ``astropy.units``
:cite:`the_astropy_collaboration_astropy_2018`.

.. code-block:: python

   radius = 6000  # km
   altitude = 500  # m

   # Wrong!
   distance = radius + altitude  

   from astropy import units as u

   # Correct
   distance = (radius << u.km) + (altitude << u.m)

This notion of providing a "safe" API extends to other parts of the library
by leveraging other capabilities of the Astropy project.
For example, timestamps use ``astropy.time`` objects,
which take care of the appropriate handling of time scales (such as TDB or UTC),
reference frame conversions leverage ``astropy.coordinates``, and so forth.

One of the drawbacks of existing unit packages is that
they impose a significant performance penalty.
Even though ``astropy.units`` is integrated with NumPy,
hence allowing the creation of array quantities,
all the unit compatibility checks are implemented in Python
and require lots of introspection,
and this can slow down mathematical operations by several orders of magnitude.
As such, to fulfill our desired performance requirement for poliastro,
we envisioned a two-layer architecture:

- The **Core API** follows a procedural style, and all the functions
  receive Python numerical types and NumPy arrays for maximum performance.
- The **High level API** is object-oriented, all the methods
  receive Astropy ``Quantity`` objects with physical units,
  and computations are deferred to the Core API.

Most of the methods of the High level API consist only of
the necessary unit compatibility checks,
plus a wrapper over the corresponding Core API function
that performs the actual computation.

.. code-block:: python

   @u.quantity_input(E=u.rad, ecc=u.one)
   def E_to_nu(E, ecc):
       """True anomaly from eccentric anomaly."""
       return (
           E_to_nu_fast(
               E.to_value(u.rad),
               ecc.value
           ) << u.rad
       ).to(E.unit)

As a result, poliastro offers a unit-safe API
that performs the least amount of computation possible
to minimize the performance penalty of unit checks,
and also a unit-unsafe API tha offers maximum performance
at the cost of not performing any unit validation checks.

.. figure:: architecture.pdf
   :scale: 75%
   :align: center

   poliastro two-layer architecture :label:`architecture`

Finally, there are several options to write performant code
that can be used from Python,
and one of them is using a fast, compiled language for the CPU intensive parts.
Successful examples of this include NumPy,
written in C :cite:`harris_array_2020`, SciPy, featuring a mix of
FORTRAN, C, and C++ code :cite:`virtanen_scipy_2020`, and pandas,
making heavy use of Cython :cite:`behnel_cython_2011`.
However, having to write code in two different languages
hinders the development speed, makes debugging more difficult,
and narrows the potential contributor base
(what Julia creators called "The Two Language Problem" :cite:`bezanson_julia_2017`).
As authors of poliastro we wanted to use Python
as the sole programming language of the implementation,
and the best solution we found to improve its performance
was to use Numba, a LLVM-based Python JIT compiler :cite:`lam_numba_2015`. 

Usage
=====

Basic ``Orbit`` and ``Ephem`` creation
--------------------------------------

The two central objects of the poliastro high level API are ``Orbit`` and ``Ephem``:

- ``Orbit`` objects represent an osculating (hence Keplerian) orbit of a dimensionless object
  around an attractor at a given point in time and a certain reference frame.
- ``Ephem`` objects represent an ephemerides, hence a sequence of spatial coordinates
  over a period of time in a certain reference frame.

Future work
===========

.. note::
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
===========

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

Dois in bibliographies
++++++++++++++++++++++

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

In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

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

References
==========
