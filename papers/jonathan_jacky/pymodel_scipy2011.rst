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

Model-based testing automatically generates, executes, and checks any
desired number of test cases, of any desired length or complexity,
given only a fixed amount of programming effort.  This contrasts with
unit testing, where additional programming effort is needed to code
each test case.

Model-based testing is intended to check *behavior*: ongoing
activities that may exhibit history-dependence and nondeterminism.
The correctness of a behavior may depend on its entire history, not
just its most recent action.  This contrasts with typical unit
testing, which checks particular *results*, such as the return value
of a function, given some arguments.    

It is advisable to check entire behaviors, not just particular
results, when testing applications such as communication protocols,
web services, embedded control systems, and user interfaces.  Many
different variations (data values, interleavings etc.)  should be
tested for each scenario (or use case).  This is only feasible with
some kind of automated test generation and checking.

Model-based testing is an automated testing technology that uses an
executable specification called a *model program* as both the test
case generator and the oracle that checks the results of each test
case.  The developer or test engineer must write a model program for
each implementation program or system they wish to test.  They must
also write a *test harness* to connect the model program to the
(generic) test runner.  

With model program and test harness in hand, developers or testers can
use the tools of the model-based testing framework in various
activities: Before generating tests from a model, it is helpful to use
an *analyzer* to validate the model program, visualize its behaviors,
and (optionally) perform safety and liveness analyses.  An *offline
test generator* generates test cases and expected test results from
the model program, which can later be executed and checked by a *test
runner* connected to the implementation through the test harness.
This is a similar workflow to unit testing, except the test cases and
expected results are generated automatically.  In contrast,
*on-the-fly testing* is quite different: the test runner generates the
test case from the model as the test run is executing. On-the-fly
testing can execute indefinitely long nonrepeating test runs, and can
accommodate nondeterminism in the implementation or its environment.

To focus automated test generation on scenarios of interest,
it is possible to code an optional *scenario machine*, a lightweight
model that describes a particular scenario.  The tools can combine
this with the comprehensive *contract model program* using an
operation called *composition*.  It is also possible to code an
optional *strategy* in order to improve test coverage according to
some chosen measure.

PyModel is an open-source model-based testing framework for Python.
It provides the PyModel Analyzer ``pma``, the PyModel Graphics program
``pmg`` for visualizing the analyzer output, and the PyModel Tester
``pmt`` for generating, executing, and checking tests, both offline
and on-the-fly.  It also includes several demonstration samples
including contract model programs, scenario machines, and test
harnesses.

Related work
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


