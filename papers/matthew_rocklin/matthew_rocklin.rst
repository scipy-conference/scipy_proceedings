:author: Matthew Rocklin 
:email: mrocklin@cs.uchicago.edu
:institution: University of Chicago, Computer Science

------------------------------------------------
Symbolic Statistics with SymPy
------------------------------------------------

.. class:: abstract

   We add a random variable type to a mathematical modeling language.


.. class:: keywords

   Symbolics, mathematical modeling, uncertainty, SymPy

Introduction
------------

Mathematical modeling is important. 

Symbolic computer algebra systems are a nice way to simplify the modeling process. They allow us to clearly describe a problem at a high level without simultaneously specifying the method of solution. This allows us to develop general solutions and solve specific problems independently. This enables a community to act with far greater efficiency.

Uncertainty is important. Mathematical models are often flawed. The model
itself may be overly simplified or the inputs may not be completely known. It
is important to understand the extent to which the results of a model can be
believed. To address these concerns it is important that we characterize the
uncertainty in our inputs and understand how this causes uncertainty in our 
results. 

In this paper we present one solution to this problem. We can add uncertainty to symbolic systems by adding a random variable type. This enables us to describe stochastic systems while adding only minimal complexity.

Motivating Example - Mathematical Modeling
------------------------------------------

Consider an artilleryman firing a cannon down into a valley. He knows the
initial position :math:`(x_0, y_0)` and orientation, :math:`\theta`, of the cannon as well as the muzzle velocity, :math:`v`, and the altitude of the target, :math:`y_f`.

.. code-block:: python

    # Inputs
    >>> x0 = 0
    >>> y0 = 0
    >>> yf = -30 # target is 30 meters below
    >>> g = -10 # gravitational constant
    >>> v = 30 # m/s
    >>> theta = pi/4

If this artilleryman has a computer nearby he may write some code to evolve
forward the state of the cannonball. If he also has a computer algebra system
he may choose to solve this system analytically. 

.. code-block:: python

    >>> t = Symbol('t') # SymPy variable for time
    >>> x = x0 + v * cos(theta) * t
    >>> y = y0 + v * sin(theta) * t
    >>> impact_time = solve(y - yf, t)
    >>> xf = x0 + v * cos(theta) * impact_time
    >>> xf.evalf() # evaluate xf numerically
    65.5842

.. code-block:: python

    # Plot x vs. y for t in (0, impact_time)
    >>> plot(x, y, (t, 0, impact_time))

.. figure:: cannon-deterministic.png

    The trajectory of a cannon shot :label:`cannon-1`

If he wishes to use the full power of SymPy he may choose to solve this problem
generally. He can do this simply by changing the numeric inputs to be symbolic 

.. code-block:: python
    
    >>> x0 = Symbol('x_0')
    >>> y0 = Symbol('y_0') 
    >>> yf = Symbol('y_f')
    >>> g = Symbol('g')
    >>> v = Symbol('v')
    >>> theta = Symbol('theta')

He can then run the same modeling code found in (missing code block label) to
obtain full solutions for impact_time and the final x position.

.. code-block:: python
    
    >>> impact_time

.. math:: 

    \frac{- v \sin{\left (\theta \right )} + \sqrt{- 4 g y_{0} + 4 g y_f + v^{2}
    \sin^{2}{\left (\theta \right )}}}{2 g}

.. code-block:: python
    
    >>> xf

.. math:: 

    x_{0} + \frac{v \left(- v \sin{\left (\theta \right )} + \sqrt{- 4 g y_{0}
    + 4 g y_f + v^{2} \sin^{2}{\left (\theta \right )}}\right) \cos{\left
      (\theta \right )}}{2 g}

Motivating Example - Uncertainty
--------------------------------

To control the velocity of the cannon ball the artilleryman introduces a
certain quantity of gunpowder to the cannon. While he takes care he is aware that his estimate of the velocity is uncertain. 

He models this uncertain quantity as a *random variable* that can take on a
range of values, each with a certain probability. In this case he believes that
the velocity is normally distributed with mean 30 and standard deviation 1.

.. code-block:: python

    >>> from sympy.stats import *
    >>> z = Symbol('a')
    >>> v = Normal('v', 30, 1)
    >>> pdf = density(v)
    >>> plot(pdf(z), (z, 27, 33))

.. math::

    \frac{\sqrt{2} e^{- \frac{1}{2} \left(z -30\right)^{2}}}{2 \sqrt{\pi}}

.. figure:: cannon-deterministic.png

    The distribution of possible velocity values :label:`velocity-distribution`

The artilleryman can now rerun the mathematical model (reference to code
above) without modification. The expressions x, y, impact_time, xf are now
stochastic expressions and we can use operators like P, E, variance, density to
convert stochasitc expressions into computational ones. 

For example we can ask the probability that the muzzle velocity is greater than
31. 

.. code-block:: python

    >>> P(v > 31)

.. math::

    - \frac{1}{2} \operatorname{erf}{\left (\frac{1}{2} \sqrt{2} \right )} +
      \frac{1}{2}


This converts a random/stochastic expression ``v > 31`` into a deterministic
computation. The expression ``P(v > 31)`` actually produces an integral
expression.

.. code-block:: python

    >>> P(v > 31, evaluate=False)

.. math::

    \int_{31}^{\infty} \frac{\sqrt{2} e^{- \frac{1}{2} \left(
    z -30\right)^{2}}}{2 \sqrt{\pi}}\, dz

We can ask similar questions about the other expressions. For example we can
compute the probability density of the position of the ball as a function of
time.

.. code-block:: python

    >>> density(x).expr

.. math::
    
    \frac{\sqrt{2} e^{- \frac{z^{2}}{t^{2}}} e^{30 \frac{\sqrt{2} z}{t}}}{2
    \sqrt{\pi} e^{450}}

.. code-block:: python

    >>> density(y).expr

.. math::
    
    \frac{\sqrt{2} e^{- \frac{\left(z + 10 t^{2}\right)^{2}}{t^{2}}} e^{30
    \frac{\sqrt{2} \left(z + 10 t^{2}\right)}{t}}}{2 \sqrt{\pi} e^{450}}

Note that to obtain these expressions the only novel work the modeler needed to
do was to describe the uncertainty of the inputs. The modeling code (cite code)
was not touched. 

We can attempt to compute more complex quantities such as the expectation and
variance of ``impact_time`` the total time of flight

.. code-block:: python

    >>> E(impact_time)

.. math::
    
    \int_{-\infty}^{\infty} \frac{\left(v + \sqrt{v^{2} + 2400}\right) e^{-
    \frac{1}{2} \left(v -30\right)^{2}}}{40 \sqrt{\pi}}\, dv

In this case the necessary integral proved too challenging for the SymPy
integration algorithms and we are left with a correct though unresolved result.

Sampling
````````

While this case is unfortunate it is also quite common. Many mathematical models
are too complex for analytic solutions. There are many approaches to these
problems, the most common of which is standard monte carlo sampling. 

SymPy.stats contains a basic monte carlo backend which can be easily accessed
with an additional keyword argument

.. code-block:: python

    >>> E(impact_time, numsamples=10000)
    3.09058769095056
    >>> variance(impact_time, numsamples=30000)
    0.00145642451022709

    >>> E(xf, numsamples=1000)
    65.4488501921592

Implementation
--------------



Multi-Compilation
-----------------

We shouldn't make monolithic solutions that encompass several
computational and mathematical disciplines. We should create thin compilers and
clear interface layers. 

This makes current individual solutions slow. 
This enables future growth.

Conclusion
----------

References
----------
