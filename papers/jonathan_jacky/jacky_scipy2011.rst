PyModel: Model-based testing in Python
======================================

:author: Jonathan Jacky
:email: jon@uw.edu
:institution: University of Washington

--------------------------------------
PyModel: Model-based testing in Python
--------------------------------------

.. class:: abstract

In unit testing, the programmer codes the test cases, and also codes
assertions that check whether each test case passed.  In model-based
testing, the programmer codes a "model" that generates as many test
cases as desired and also acts as the oracle that checks the cases.
Model-based testing is recommended where so many test cases are needed
that it is not feasible to code them all by hand.  This need arises
when testing behaviors that exhibit history-dependence and
nondeterminism, so that many variations (data values, interleavings,
etc.) should be tested for each scenario (or use case).  Examples
include communication protocols, web applications, control systems,
and user interfaces.  PyModel is a model-based testing framework in
Python.  PyModel supports on-the-fly testing, which can generate
indefinitely long nonrepeating tests as the test run executes.
PyModel can focus test cases on scenarios of interest by composition,
a versatile technique that combines models by synchronizing shared
actions and interleaving unshared actions.  PyModel can guide test
coverage according to programmable strategies coded by the programmer.

.. class:: keywords

   testing, model-based testing, automated testing, executable
   specification, finite state machine, nondeterminism, exploration,
   offline testing, on-the-fly testing, scenario, composition

Introduction
------------

All is explained in [Jacky08]_.  See Figure :ref:`abp`.

.. figure:: abp.pdf
   :figclass: bht

   Alternating Bit Protocol represented by a Finite State Machine (FSM) :label:`abp`

References
----------

.. [Jacky08] Jonathan Jacky, Margus Veanes, Colin Campbell, and Wolfram Schulte.
             *Model-Based Software Testing and Analysis with C#*,
	     Cambridge University Press, 2008.


