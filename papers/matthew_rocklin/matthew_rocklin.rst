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

Motivating Example
------------------

Consider an artilleryman firing a cannon down into a valley. He knows the
initial position :math:`(x_0, y_0)` and orientation, :math:`theta` of the cannon as well as the muzzle velocity, :math:`v`.

```python
# Inputs
>>> x0 = 0
>>> y0 = 0
>>> g = -10 # gravitational constant
>>> v = 30 # m/s
>>> theta = pi/4```

If this artilleryman has a computer nearby he may write some code to evolve
forward the state of the cannonball. If he also has a computer algebra system
he may choose to solve this system analytically. 

```python
>>> yf = Symbol('y_f') # SymPy variable to represent the altitude of the target
>>> t = Symbol('t') # SymPy variable to represent time
>>> x = x0 + v * cos(theta) * t
>>> y = y0 + v * sin(theta) * t
>>> impact_time = solve(y - yf, t)```




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
