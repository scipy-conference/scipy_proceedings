

:author: David P. Sanders
:email: dpsanders@ciencias.unam.mx
:institution: Department of Physics, Faculty of Sciences,
  National Autonomous University of Mexico (UNAM), Ciudad Universitaria,
  México D.F. 04510, Mexico

:author: Luis Benet
:email: benet@fis.unam.mx
:institution: Institute of Physical Sciences,
  National Autonomous University of Mexico (UNAM),
  Apartado postal 48-3, Cuernavaca 62551, Morelos,
  Mexico

------------------------------------------------------
Validated numerics with Python: the `ValidiPy` package
------------------------------------------------------

.. class:: abstract

    We introduce the ValidiPy package for *validated numerics* in
    Python. This suite of tools, which includes interval arithmetic and automatic
    differentiation, enables *rigorous* and *guaranteed* results using floating-point
    arithmetic. We apply the ValidiPy package to two classic problems in dynamical systems,
    calculating periodic points of the logistic map, and simulating the
    dynamics of a chaotic billiard model.

.. class:: keywords

    validated numerics, Newton method, floating point, interval arithmetic


Floating-point arithmetic
=========================

Scientific computation usually requires the manipulation of real
numbers. The standard method to represent real numbers internally in a
computer is floating-point arithmetic, in which a real number :math:`a`
is represented as

.. math:: a = \pm 2^e \times m.

The usual double-precision (64-bit) representation is that of the
`IEEE 754 standard <http://en.wikipedia.org/wiki/IEEE_floating_point>`__
[IEEE754]_: one bit is used for the sign, 11 bits for the exponent
:math:`e`, which ranges from :math:`-1022` to :math:`+1023`, and the
remaining 52 bits are used for the "mantissa" :math:`m`, a binary string
of 53 bits, starting with a :math:`1` which is not explicitly stored.

However, most real numbers are *not explicitly representable* in this
form, for example 0.1, which in binary has the infinite periodic
expansion

.. math:: 0.0\ 0011\ 0011\ 0011\ 0011\ldots,

in which the pattern :math:`0011` repeats forever. Representing this in
a computer with a finite number of digits, via truncation or rounding,
gives a number that differs slightly from the true 0.1, and leads to the
following kinds of problems. Summing :math:`0.1` many times -- a common
operation in, for example, a time-stepping code, gives the following
*unexpected* behaviour.

.. code:: python

    a = 0.1

    total = 0.0

    print("%20s %25s" % ("total", "error"))
    for i in xrange(1000):
        if i%100 == 0 and i>0:
            error = total - i/10
            print("%20.16g %25.16g" % (total, error))
        total += a


.. parsed-literal::

                   total                     error
        9.99999999999998    -1.953992523340276e-14
       20.00000000000001       1.4210854715202e-14
       30.00000000000016      1.56319401867222e-13
        40.0000000000003     2.984279490192421e-13
       50.00000000000044     4.405364961712621e-13
       60.00000000000058     5.826450433232822e-13
        70.0000000000003     2.984279490192421e-13
       79.99999999999973    -2.700062395888381e-13
       89.99999999999916    -8.384404281969182e-13


Here, the result oscillates in an apparently "random" fashion around the
expected value.

This is already familiar to new users of any programming language when
they see the following kinds of outputs of elementary calculations
[Gold91]_:

.. code:: python

    3.2 * 4.6




.. parsed-literal::

    14.719999999999999



Suppose that we now apply an algorithm starting with
an initial condition :math:`x_0 = 0.1`.
The result will be erroneous, since the initial condition used
differs slightly from the desired value. In chaotic systems, for
example, such a tiny initial deviation may be quickly magnified and
destroy all precision in the computation. Although there are methods to
estimate the resulting errors [High96]_, there is no guarantee that the
true result is captured. Another example is certain ill-conditioned
matrix computations, where small changes to the matrix lead to
unexpectedly large changes in the result.

Interval arithmetic
===================

Interval arithmetic is one solution for these difficulties.
In this method, developed over the last 50 years but
still relatively unknown in the wider scientfic community,
all quantities in a computation are treated as closed
intervals of the form :math:`[a, b]`. If the initial data are
contained within the initial intervals, then the result of the
calculation is *guaranteed* to contain the true result. To accomplish
this, the intervals are propagated throughout the calculation, based on
the following ideas:

1. All intervals must be *correctly rounded*: the lower limit :math:`a`
   of each interval is rounded downwards (towards :math:`-\infty`) and
   the upper limit :math:`b` is rounded upwards (towards
   :math:`+\infty`). [The availability of these rounding operations is
   standard on modern computing hardware.] In this way, the interval is
   guaranteed to contain the true result. If we do not apply rounding,
   then this might not be the case; for example, the interval given by
   :math:`I=Interval(0.1, 0.2)` does not actually contain the true :math:`0.1`
   if the standard floating-point representation for the lower
   end-point is used; instead, this lower bound corresponds to
   :math:`0.10000000000000000555111\ldots`.

2. Arithmetic operations are defined on intervals, such that the result
   of an operation on a pair of intervals is the interval that is `the
   result of performing the operation on any pair of numbers, one
   from each interval`.

3. Elementary functions are defined on intervals, such that the result
   of an elementary function :math:`f` applied to an interval :math:`I`
   is the *image* of the function over that interval,
   :math:`f(I) := \{f(x): x \in I\}`.

For example, addition of two intervals is defined as

.. math:: [a, b] + [c, d] := \{ x + y: x \in [a, b], y \in [c,d] \},

which turns out to be equivalent to

.. math:: [a, b] + [c, d] := [a+c, b+d].

The exponential function applied to an interval is defined as

.. math:: \exp([a,b]) := [\exp(a), \exp(b)],

giving the exact image of the monotone function :math:`\exp` evaluated
over the interval.

Once all required operations and elementary functions (such as
:math:`\sin`, :math:`\exp` etc.) are correctly defined, and given a
technical condition called "inclusion monotonicity", for any function
:math:`f: \mathbb{R} \to \mathbb{R}` made out of a combination of
arithmetic operations and elementary functions, we may obtain the
*interval extension* :math:`\tilde{f}`. This is a "version" of the
function which applies to intervals, such that when we apply
:math:`\tilde{f}` to an interval :math:`I`, we obtain a new interval
:math:`\tilde{f}(I)` that is *guaranteed to contain* the true,
mathematical image :math:`f(I) := \{f(x): x \in I \}`.

Unfortunately, :math:`\tilde{f}(I)` may be strictly larger than the true
image :math:`f(I)`, due to the so-called *dependency problem*. For
example, let :math:`I := [-1, 1]`. Suppose that :math:`f(x) := x*x`,
i.e. that we wish to square all elements of the interval. The true image
of the interval :math:`I` is then :math:`f(I) = [0, 1]`.

However, thinking of the squaring operation as repeated
multiplication, we may try to calculate

.. math:: I * I := \{xy: x \in I, y \in I \}.

Doing so, we find the *larger* interval :math:`[-1,1]`, since we "do not
notice" that the :math:`x`'s are "the same" in each copy of the
interval; this, in a nutshell, is the dependency problem.

In this particular case, there is a simple solution: we calculate
instead :math:`I^2 := \{x^2: x \in I\}`, so that there is only a single
copy of :math:`I` and the true image is obtained. However, if we
consider a more complicated function like :math:`f(x) = x + \sin(x)`,
there does not seem to be a generic way to solve the dependency problem
and hence find the exact range.

This problem may, however, be solved to an arbitrarily good approximation
by splitting up the initial interval into a union of subintervals.
When the interval extension is instead evaluated over those
subintervals, the union of the resulting intervals gives
an enclosure of the exact range that is increasingly better as the
size of the subintervals decreases [Tuck11]_.

Validated numerics: the ``ValidiPy`` package
============================================

The name "validated numerics" has been applied to the combination of
interval arithmetic, automatic differentiation, Taylor methods and other
techniques that allow the rigorous solution of problems using
finite-precision floating point arithmetic [Tuck11]_.

The ``ValidiPy`` package, a Python package for validated numerics, was
initiated during a Masters' course on validated numerics that the authors
taught in the Postgraduate Programmes in Mathematics and Physics at the
National Autonomous University of Mexico (UNAM) during the second half
of 2013. It is based on the excellent textbook *Validated Numerics* by
Warwick Tucker [Tuck11]_, one of the foremost proponents of interval
arithmetic today. He is best known for [Tuck99]_, in which he
gave a rigorous proof of the existence of the Lorenz attractor,
a strange (fractal, chaotic) attractor of a set of
three ordinary differential equations modelling convection in the atmosphere
that were computationally observed to be chaotic in 1963 [Lorenz]_.

Naturally, there has been previous work on implementing the different
components of Validated Numerics in Python, such as
`pyinterval <https://code.google.com/p/pyinterval/>`__ and
`mpmath <http://mpmath.org/>`__ for interval arithmetic, and
`AlgoPy <https://pythonhosted.org/algopy/>`__ for automatic
differentiation. Our project is designed to provide an understandable
and modifiable code base, with a focus on ease of use,
rather than speed.

An incomplete sequence of IPython notebooks from
the course, currently in Spanish, provide an
introduction to the theory and practice of interval arithmetic; they are
available on `GitHub <https://github.com/computo-fc/metodos_rigurosos/tree/master/clases>`__
and for online viewing at `NbViewer <http://nbviewer.ipython.org/github/computo-fc/metodos_rigurosos/tree/master/clases/>`__.

Code in Julia is also available, in our package
``ValidatedNumerics.jl`` [ValidatedNumerics]_.


Implementation of interval arithmetic
=====================================

As with many other programming languages, Python allows us to define new
types, as ``class`` es, and to define operations on those types. The
following working sketch of an ``Interval`` class may be extended to a
full-blown implementation (which, in particular, must include directed
rounding; see below), available in the [ValidiPy]_ repository.

.. code:: python

    class Interval(object):
        def __init__(self, a, b=None):
            # constructor

            if b is None:
                b = a

            self.lo = a
            self.hi = b

        def __add__(self, other):
            if not isinstance(other, Interval):
                other = Interval(other)
            return Interval(self.lo+other.lo,
                            self.hi+other.hi)

        def __mul__(self, other):
            if not isinstance(other, Interval):
                other = Interval(other)

            S = [self.lo*other.lo, self.lo*other.hi,
                 self.hi*other.lo, self.hi*other.hi]
            return Interval(min(S), max(S))

        def __repr__(self):
            return "[{}, {}]".format(self.lo, self.hi)

Examples of creation and manipulation of intervals:

.. code:: python

    i = Interval(3)
    i




.. parsed-literal::

    [3, 3]



.. code:: python

    i = Interval(-3, 4)
    i




.. parsed-literal::

    [-3, 4]



.. code:: python

    i * i




.. parsed-literal::

    [-12, 16]



.. code:: python

    def f(x):
        return x*x + x + 2

.. code:: python

    f(i)




.. parsed-literal::

    [-13, 22]



To attain multiple-precision arithmetic and directed rounding, we use
the ``gmpy2`` package [gmpy2]_.
This provides a wrapper around the ``MPFR``
[MPFR]_ C package for correctly-rounded multiple-precision arithmetic
[Fous07]_. For example, a simplified version of the ``Interval``
constructor may be written as follows, showing how the precision and
rounding modes are manipulated using the ``gmpy2`` package:

.. code:: python

    import gmpy2
    from gmpy2 import RoundDown, RoundUp

    ctx = gmpy2.get_context()

    def set_interval_precision(precision):
        gmpy2.get_context().precision = precision

    def __init__(self, a, b=None):
        ctx.round = RoundDown
        a = mpfr(str(a))

        ctx.round = RoundUp
        b = mpfr(str(b))

        self.lo, self.hi = a, b

Each arithmetic and elementary operation must apply directed rounding in
this way at each step; for example, the implementations of
multiplication and exponentiation of intervals are as follows:

.. code:: python

    def __mult__(self,other):

        ctx.round = RoundDown
        S_lower = [ self.lo*other.lo, self.lo*other.hi,
                    self.hi*other.lo, self.hi*other.hi ]
        S1 = min(S_lower)

        ctx.round = RoundUp
        S_upper = [ self.lo*other.lo, self.lo*other.hi,
                    self.hi*other.lo, self.hi*other.hi ]
        S2 = max(S_upper)

        return Interval(S1, S2)

    def exp(self):
        ctx.round = RoundDown
        lower = exp(self.lo)

        ctx.round = RoundUp
        upper = exp(self.hi)

        return Interval(lower, upper)

The Interval Newton method
==========================

As applications of interval arithmetic and of ``ValidiPy``, we will
discuss two classical problems in the area of dynamical systems. The
first is the problem of locating all periodic orbits of the dynamics,
with a certain period, of the well-known logistic map. To do so, we will
apply the *Interval Newton method*.

The Newton (or Newton--Raphson) method is a standard algorithm for
finding zeros, or roots, of a nonlinear equation, i.e. :math:`x^*` such
that :math:`f(x^*) = 0`, where
:math:`f \colon \mathbb{R} \to \mathbb{R}` is a nonlinear function.

The Newton method starts from an initial guess :math:`x_0` for the root
:math:`x^*`, and iterates

.. math::

    x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)},

where :math:`f' \colon \mathbb{R} \to \mathbb{R}` is the derivative of
:math:`f`. This formula calculates the intersection of the tangent line
to the function :math:`f` at the point :math:`x_n` with the
:math:`x`-axis, and thus gives a new estimate of the root.

If the initial guess is sufficiently close to a root, then this
algorithm converges very quickly ("quadratically") to the root: the number
of correct digits doubles at each step.

However, the standard Newton method suffers from problems: it may not
converge, or may converge to a different root than the intended one.
Furthermore, there is no way to guarantee that all roots in a certain
region have been found.

An important, but too little-known, contribution of interval analysis is
a version of the Newton method that is modified to work with intervals,
and is able to locate *all* roots of the equation within a specified
interval :math:`I`, by isolating each one in a small sub-interval, and
to either guarantee that there is a unique root in each of those
sub-intervals, or to explicitly report that it is unable to determine
existence and uniqueness.

To understand how this is possible, consider applying the interval
extension :math:`\tilde{f}` of :math:`f` to an interval :math:`I`.
Suppose that the image :math:`\tilde{f}(I)` does *not* contain
:math:`0`. Since :math:`f(I) \subset \tilde{f}(I)`, we know that
:math:`f(I)` is *guaranteed* not to contain :math:`0`, and thus we
guarantee that there *cannot be a root* :math:`x^*` of :math:`f` inside
the interval :math:`I`. On the other hand, if we evaluate :math:`f` at
the endpoints :math:`a` and :math:`b` of the interval :math:`I=[a,b]`
and find that :math:`f(a) < 0 < f(b)` (or vice versa), then we can
guarantee that there is *at least one root within the interval*.

The Interval Newton method does not just naively extend the standard
Newton method. Rather, a new operator, the Newton operator, is defined,
which takes an interval as input and returns as output either one or two
intervals. The Newton operator for the function :math:`f` is defined as

.. math:: N_f(I) := m -  \frac{f(m)}{\tilde{f}'(I)},

where :math:`m := m(I)` is the midpoint of the interval :math:`I`, which
may be treated as a (multi-precision) floating-point number, and
:math:`\tilde{f}'(I)` is an interval extension of the derivative
:math:`f'` of :math:`f`. This interval extension may easily be
calculated using *automatic differentiation* (see below). The division
is now a division by an interval, which is defined as for the other
arithmetic operations. In the case when the interval
:math:`\tilde{f}'(I)` contains :math:`0`, this definition leads to the
result being the union of *two disjoint intervals*: if
:math:`I = [-a, b]` with :math:`a>0` and :math:`b>0`, then we define
:math:`1/I = (1/[-a, -0]) \cup (1/[0, b]) = [-\infty, -1/a] \cup [1/b, \infty]`.

The idea of this definition is that the result of applying the operator
:math:`N_f` to an interval :math:`I` will necessarily contain the result
of applying the standard Newton operator at all points of the interval,
and hence will contain *all* possible roots of the function in that
interval.

Indeed, the following strong results may be rigorously proved [Tuck11]_:
1. If :math:`N_f(I) \cap I = \emptyset`, then :math:`I` contains no
zeros of :math:`f`; 2. If :math:`N_f(I) \subset I`, then :math:`I`
contains exactly one zero of :math:`f`.

If neither of these options holds, then the interval :math:`I` is split
into two equal subintervals and the method proceeds on each. Thus the
Newton operator is sufficient to determine the presence (and uniqueness)
or absence of roots in each subinterval.

Starting from an initial interval :math:`I_0`, and iterating
:math:`I_{n+1} := I_n \cap N_f(I_n)`, gives a sequence of lists of
intervals that is guaranteed to contain the roots of the function, as
well as a guarantee of uniqueness in many cases.

The code to implement the Interval Newton method completely is slightly
involved, and may be found in an IPython notebook in the
`examples` directory at
<https://github.com/computo-fc/ValidiPy/tree/master/examples>.

An example of the Interval Newton method in action is shown in
figure :ref:`roots-of-two`, where it was used to find the roots of
:math:`f(x) = x^2 - 2` within the initial interval :math:`[-5, 5]`.
Time proceeds vertically from bottom to top.


.. figure:: roots-of-two.pdf

    Convergence of the Interval Newton method to the roots of 2.
    :label:`roots-of-two`


Periodic points of the logistic map
===================================

An interesting application of the Interval Newton method is to dynamical
systems. These may be given, for example, as the solution of systems of
ordinary differential equations, as in the Lorenz equations [Lor63]_, or by
iterating maps. The *logistic map* is a much-studied dynamical system,
given by the map

.. math:: f(x) := f_r(x) := rx(1-x).

The dynamics is given by iterating the map:

.. math:: x_{n+1} = f(x_n),

so that

.. math:: x_n = f(f(f(\cdots (x_0) \cdots))) = f^n(x_0),

where :math:`f^n` denotes :math:`f \circ f \circ \cdots \circ f`, i.e.
:math:`f` composed with itself :math:`n` times.

*Periodic points* play a key role in dynamical system:
these are points :math:`x` such that
:math:`f^p(x) = x`; the minimal :math:`p>0` for which this is satisfied
is the *period* of :math:`x`. Thus, starting from such a point, the
dynamics returns to the point after :math:`p` steps, and then eternally
repeats the same trajectory. In chaotic systems, periodic points are dense
in phase space [Deva03]_, and properties of the dynamics may be calculated in
terms of the periodic points and their stability properties [ChaosBook]_.
The numerical enumeration of all periodic points is thus a necessary
part of studying almost any such system. However, standard methods
usually do not guarantee that all periodic points of a given period have
been found.

On the contrary, the Interval Newton method, applied to the function
:math:`g_p(x) := f^p(x) - x`, guarantees to find all zeros of the
function :math:`g_p`, i.e. all points with period at most :math:`p` (or
to explicitly report where it has failed). Note that this will include
points of lower period too; thus, the periodic points should be
enumerated in order of increasing period, starting from period
:math:`1`, i.e. fixed points :math:`x` such that :math:`f(x)=x`.

To verify the application of the Interval Newton method to calculate
periodic orbits, we use the fact that the particular case of :math:`f_4`
the logistic map with :math:`r=4` is *conjugate* (related by an
invertible nonlinear change of coordinates) to a simpler map, the tent
map, which is a piecewise linear map from :math:`[0,1]` onto itself,
given by

.. math::

   T(x) :=
   \begin{cases}
   2x, & \text{if } x < \frac{1}{2}; \\
   2 - 2x, & \text{if } x > \frac{1}{2}.
   \end{cases}


The :math:`n`\ th iterate of the tent map has :math:`2^n` "pieces" (or
"laps") with slopes of modulus :math:`2^n`, and hence exactly
:math:`2^n` points that satisfy :math:`T^n(x)=x`.

The :math:`i`\ th "piece" of the :math:`n`\ th iterate (with
:math:`i=0, \ldots, 2^n-1`) has equation

.. math::

   T^n_i(x) =
   \begin{cases}
   2^n x-i, & \text{if $i$ is even and $\frac{i}{2^n} \le x < \frac{i+1}{2^n}$} \\
   i+1 - 2^n x, & \text{if $i$ is odd and $\frac{i}{2^n} \le x < \frac{i+1}{2^n}$} \\
   \end{cases}

Thus the solution of :math:`T^n_i(x) = x` satisfies

.. math::

   x^n_i =
   \begin{cases}
   \frac{i}{2^n - 1}, & \text{if $i$ is even}; \\
   \frac{i+1}{1 + 2^n}, & \text{if $i$ is odd},
   \end{cases}

giving the :math:`2^n` points which are candidates for periodic points
of period :math:`n`. (Some are actually periodic points with period
:math:`p` that is a proper divisor of :math:`n`, satisfying also
:math:`T^p(x) = x`.)  These points are shown in figure
:ref:`tent-map-period-4`.


.. figure:: tent-map-period-4.pdf

    Periodic points of the tent map with period dividing 4.
    :label:`tent-map-period-4`


.. figure:: logistic-period-4.pdf

    Periodic points of the logistic map with period dividing 4.
    :label:`logistic-map-period-4`



It turns out [Ott]_ that the invertible change of variables

.. math:: x = h(y) = \sin^2(\textstyle \frac{\pi y} {2})

converts the sequence :math:`(y_n)`, given by iterating the tent map,

.. math:: y_{n+1} = T(y_n),

into the sequence :math:`(x_n)` given by iterating the logistic map
:math:`f_4`,

.. math:: x_{n+1} = f_4(x_n) = 4 x_n (1-x_n).

Thus periodic points of the tent map, satisfying :math:`T^m(y) = y`, are
mapped by :math:`h` into periodic points :math:`x` of the logistic map,
satisfying :math:`T^m(x) = x`, shown in figure :ref:`logistic-map-period-4`.

The following table (figure :ref:`period-4-data`) gives the midpoint of the intervals containing the
fixed points :math:`x` such that :math:`f_4^4(x)=x` of the logistic map,
using the Interval Newton method with standard double precision, and the
corresponding exact values using the correspondence with the tent map,
together with the difference. We see that the method indeed works very
well. However, to find periodic points of higher period, higher precision
must be used.


.. figure::  period-4.pdf

    Period 4 points: calculated, exact, and the difference.
    :label:`period-4-data`


Automatic differentiation
=========================

A difficulty in implementing the Newton method (even for the standard
version), is the calculation of the derivative :math:`f'` at a given
point :math:`a`. This may be accomplished for any function :math:`f` by
*automatic (or algorithmic) differentiation*, also easily implemented in
Python.

The basic idea is that to calculate :math:`f'(a)`, we may split a
complicated function :math:`f` up into its constituent parts and
propagate the values of the functions and their derivatives through the
calculations. For example, :math:`f` may be the product and/or sum of
simpler functions. To combine information on functions :math:`u` and
:math:`v`, we use

.. math::


   \begin{aligned}
   (u+v)'(a) &= u'(a) + v'(a) ,\\
   (uv)'(a) &= u'(a) v(a) + u(a) v'(a) ,\\
   (g(u))'(a) &= g'(u(a)) \, u'(a) .
   \end{aligned}

Thus, for each function :math:`u`, it is sufficient to represent it as
an ordered pair :math:`(u(a), u'(a))` in order to calculate the value
and derivative of a complicated function made out of combinations of
such functions.

Constants :math:`C` satisfy :math:`C'(a) = 0` for all :math:`a`, so that
they are represented as the pair :math:`(C, 0)`. Finally, the identity
function :math:`\mathrm{id}: x \mapsto x` has derivative
:math:`\mathrm{id}'(a) = 1` at all :math:`a`.

The mechanism of operator overloading in Python allows us to define an
``AutoDiff`` class. Calculating the derivative of a function ``f(x)`` at
the point ``a`` is then accomplished by calling ``f(AutoDiff(a, 1))``
and extracting the derivative part.

.. code:: python

    class AutoDiff(object):
        def __init__(self, value, deriv=None):

            if deriv is None:
                deriv = 0.0

            self.value = value
            self.deriv = deriv


        def __add__(self, other):
            if not isinstance(other, AutoDiff):
                other = AutoDiff(other)

            return AutoDiff(self.value+other.value,
                            self.deriv+other.deriv)

        def __mul__(self, other):
            if not isinstance(other, AutoDiff):
                other = AutoDiff(other)

            return AutoDiff(self.value*other.value,
                            self.value*other.deriv +
                              self.deriv*other.value)

        def __repr__(self):
            return "({}, {})".format(
                  self.value, self.deriv)


As a simple example, let us differentiate the function
:math:`f(x) = x^2 + x + 2` at :math:`x=3`. We define the function
in the standard way:

.. code:: python

    def f(x):
        return x*x + x + 2

We now define a variable ``a`` where we wish to calculate the derivative
and an object ``x`` representing the object that we will use in the automatic
differentiation. Since it represents the function :math:`x \to x` evaluated
at :math:`a`, it has derivative 1:

.. code:: python

    a = 3
    x = AutoDiff(a, 1)

Finally, we simply apply the standard Python function to this new object,
and the automatic differentiation takes care of the rest:

.. code:: python

    result = f(x)
    print("a={}; f(a)={}; f'(a)={}".format(
                a, result.value, result.deriv))

giving the result

.. parsed-literal::

    a=3; f(a)=14; f'(a)=7.0


The derivative :math:`f'(x) = 2x + 1`, so that :math:`f(a=3) = 14` and
:math:`f'(a=3) = 7`. Thus both the value of the function and its
derivative have been calculated in a completely *automatic* way, by
applying the rules encoded by the overloaded operators.

Simulating a chaotic billiard model
===================================

A dynamical system is said to be *chaotic* if it satisfies certain
conditions [Deva03]_, of which a key one is *sensitive dependence on
initial conditions*: two nearby initial conditions separate
*exponentially* fast.

This leads to difficulties if we want precise answers on the long-term
behaviour of such systems, for example simulating the solar system over
millions of years [Lask13]_. For certain types of systems, there are
*shadowing theorems*, which say that an approximate trajectory
calculated with floating point arithmetic, in which a small error is
committed at each step, is close to a true trajectory [Palm09]_; however,
these results tend to be applicable only for rather restricted classes
of systems which do not include those of physical interest.

Interval arithmetic provides a partial solution to this problem, since
it automatically reports the number of significant figures in the result
which are guaranteed correct. As an example, we show how to solve one of
the well-known "Hundred-digit challenge problems" [Born04]_, which
consists of calculating the position from the origin in a certain billiard problem.

Billiard problems are a class of mathematical models in which pointlike
particles (i.e. particles with radius :math:`0`) collide with fixed
obstacles. They can be used to study systems of hard discs or hard
spheres with elastic collisions, and are also paradigmatic examples of
systems which can be proved to be chaotic, since the seminal work of
Sinaï [Chern06]_.

Intuitively, when two nearby rays of light hit a circular mirror,
the curvature of the surface leads to the rays separating after they
reflect from the mirror. At each such collision, the distance
in phase space between the rays is, on average, multiplied by a
factor at each collision, leading to exponential separation and hence
chaos, or *hyperbolicity*.

The trajectory of a single particle in such a system will hit a sequence
of discs. However, a nearby initial condition may, after a few
collisions, miss one of the discs hit by the first particle, and will
then follow a completely different future trajectory. With standard
floating-point arithmetic, there is no information about when this
occurs; interval arithmetic can guarantee that this has *not* occurred,
and thus that the sequence of discs hit is correct.

The second of the Hundred-digit challenge
problems [Born04]_ is as follows:

A point particle bounces off fixed discs of radius :math:`\frac{1}{3}`,
placed at the points of a square lattice with unit distance between
neighbouring points. The particle starts at :math:`(x,y) = (0.5, 0.1)`,
heading due east with unit speed, i.e. with initial velocity
:math:`(1, 0)`. Calculate the distance from the origin of the particle
at time :math:`t=10`, with 10 correct significant figures.

To solve this, we use a standard implementation of the billiard by
treating it as a single copy of a unit cell, centred at the origin and
with side length :math:`1`, and periodic boundary conditions. We keep
track of the cell that is reached in the corresponding "unfolded"
version in the complete lattice.

The code used is a standard billiard code, that may be written in an
*identical* way to use either standard floating-point method or interval
arithmetic using ``ValidiPy``, changing only the initial conditions to
use intervals instead of floating-point variables. Since :math:`0.1` and
:math:`1/3` are not exactly representable, they are replaced by the
smallest possible intervals containing the true values, using directed
rounding as discussed above.

It turns out indeed to be necessary to use multiple precision in the
calculation, due to the chaotic nature of the system. In fact, our
algorithm requires a precision of at least 96 binary digits (compared to
standard double precision of 53 binary digits) in order to guarantee
that the correct trajectory is calculated up to time :math:`t=10`. With
fewer digits than this, a moment is always reached at which the
intervals have grown so large that it is not guaranteed whether a given
disc is hit or not.
The trajectory is shown in figure :ref:`billiard-traj`.


.. figure:: billiard_trajectory.pdf

    Trajectory of the billiard model up to time 10; the black dot shows the initial position.
    :label:`billiard-traj`


With 96 digits, the uncertainty on the final distance, i.e. the diameter
of the corresponding interval, is :math:`0.0788`. As the number of
digits is increased, the corresponding uncertainty decreases
exponentially fast, reaching :math:`4.7 \times 10^{-18}` with 150
digits, i.e. at least 16 decimal digits are guaranteed correct.

.. With
.. 1000 binary digits of precision, for example, the uncertainty reduces to
.. :math:`5.9 \times 10^{-274}`, guaranteeing 272 correct decimal digits.

Extensions
==========

Intervals in higher dimensions
------------------------------

The ideas and methods of interval arithmetic may also be applied in
higher dimensions. There are several ways of defining intervals in 2 or
more dimensions [Moo09]_. Conceptually, the simplest is perhaps to take
the Cartesian product of one-dimensional intervals:

.. math:: I = [a,b] \times [c,d]

We can immediately define, for example, functions like
:math:`f(x,y) := x^2 + y^2` and apply them to obtain the corresponding
interval extension :math:`\tilde{f}([a,b], [c,d]) := [a,b]^2 + [c,d]^2`,
which will automatically contain the true image :math:`f(I)`. Similarly,
functions :math:`f: \mathbb{R}^2 \to \mathbb{R}^2`
will give an interval extension producing a two-dimensional
rectangular interval. However, the result is
often much larger than the true image, so that the subdivision technique
must be applied.

Taylor series
-------------

An extension of automatic differentiation is to manipulate Taylor series
of functions around a point, so that the function :math:`u` is
represented in a neighbourhood of the point :math:`a` by the tuple
:math:`(a, u'(a), u''(a), \ldots, u^{(n)}(a))`. Recurrence formulas
allow these to be manipulated relatively efficiently. These may be used,
in particular, to implement arbitrary-precision solution of ordinary
differential equations.

An implementation in Python is available in ValidiPy, while an
implementation in the Julia is
available separately, including Taylor series in multiple variables
[TaylorSeries]_.

Conclusions
===========

Interval arithmetic is a powerful tool which has been, perhaps,
under-appreciated in the wider scientific community. Our contribution is
aimed at making these techniques more widely known, in particular at
including them in courses at masters', or even undergraduate, level,
with working, freely available code in Python and Julia.

Acknowledgements
================

The authors thank Matthew Rocklin for helpful comments during the open
refereeing process, which improved the exposition.
Financial support is acknowledged from
DGAPA-UNAM PAPIME grants PE-105911 and PE-107114,
and DGAPA-UNAM PAPIIT grants IG-101113 and IN-117214.
LB acknowledges support through a Cátedra Moshinsky (2013).

References
==========

.. [IEEE754] *IEEE Standard for Floating-Point Arithmetic*, 2008, IEEE Std
    754-2008.

.. [Gold91] D. Goldberg (1991), What Every Computer Scientist Should Know
    About Floating-Point Arithmetic, *ACM Computing Surveys* **23** (1), 5-48.

.. [High96] N.J. Higham (1996), *Accuracy and Stability of Numerical
    Algorithms*, SIAM.

.. [Tuck11] W. Tucker (2011), *Validated Numerics: A Short Introduction to
    Rigorous Computations*, Princeton University Press.

.. [Tuck99] W. Tucker, 1999, The Lorenz attractor exists, *C. R. Acad. Sci.
    Paris Sér. I Math.* **328** (12), 1197-1202.

.. [ValidiPy] D.P. Sanders and L. Benet, ``ValidiPy`` package for Python,
    <https://github.com/computo-fc/ValidiPy>

.. [ValidatedNumerics] D.P. Sanders and L. Benet, ``ValidatedNumerics.jl``
    package for Julia, <https://github.com/dpsanders/ValidatedNumerics.jl>

.. [gmpy2] ``GMPY2`` package, <https://code.google.com/p/gmpy>

.. [MPFR] ``MPFR`` package, <http://www.mpfr.org>

.. [Fous07] L. Fousse et al. (2007), MPFR: A multiple-precision binary
    floating-point library with correct rounding, *ACM Transactions on
    Mathematical Software* **33** (2), Art. 13.

.. [Lor63] E.N. Lorenz (1963), Deterministic nonperiodic flow, *J. Atmos.
    Sci.* **20** (2), 130-141.

.. [ChaosBook] P. Cvitanović et al. (2012), *Chaos: Classical and Quantum*,
    Niels Bohr Institute. <http://ChaosBook.org>

.. [Ott] E. Ott (2002), *Chaos in Dynamical Systems*, 2nd edition, Cambridge
    University Press.

.. [Deva03] R.L. Devaney (2003), *An Introduction to Chaotic Dynamical
    Systems*, Westview Press.

.. [Lask13] J. Laskar (2013), Is the Solar System Stable?,
  in *Chaos: Poincaré Seminar 2010* (chapter 7), B. Duplantier,
  S. Nonnenmacher and V. Rivasseau (eds).

.. [Palm09] K.J. Palmer (2009), Shadowing lemma for flows,
  *Scholarpedia* **4** (4). http://www.scholarpedia.org/article/Shadowing\_lemma\_for\_flows

.. [Born04] F. Bornemann, D. Laurie, S. Wagon and J. Waldvogel (2004),
    *The SIAM 100-Digit Challenge: A Study in High-Accuracy Numerical Computing*,
    SIAM.

.. [Chern06] N. Chernov and R. Markarian (2006), *Chaotic Billiards*,
    AMS.

.. [TaylorSeries] L. Benet and D.P. Sanders, ``TaylorSeries`` package,
    <https://github.com/lbenet/TaylorSeries.jl>

.. [Moo09] R.E. Moore, R.B. Kearfott and M.J. Cloud (2009), *Introduction to
    Interval Analysis*, SIAM.

.. [Lorenz] E.N. Lorenz (1963), Deterministic nonperiodic flow, *J. Atmos. Sci*
    **20** (2), 130-148.
