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
The correctness of behavior may depend on its entire history, not
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
and on-the-fly.  It also includes several demonstration samples, each
including a contract model program, scenario machines, and a test
harness.

Traces and Actions
------------------

We need to describe behavior.  To show how, we discuss the Alternating
Bit Protocol [ABP11]_, a simple example that exhibits
history-dependence and nondeterminism.  The protocol is designed to
send messages over an unreliable network.  The sender keeps sending
the same message, labeled with the same bit (1 or 0), until the
receiver acknowledges successful receipt by sending back the same bit.
The sender then complements the bit and sends a new message labeled
with the new bit until it receives an acknowledgement with that new
bit, and so on.  When the connection starts up, both ends send bit 1.
The sender labels the first real message with 0.

A sample of behavior is called a *trace*.  A trace is a sequence of
*actions*, where each action has a name and may have arguments (so
actions resemble function calls).  The alternating bit protocol has
only two actions, named ``Send`` and ``Ack``.  Each action has one
argument that can take on only two values, ``0`` or ``1``.  (We
abstract away the message contents, which do not affect the protocol
behavior). Here are some traces that are allowed by the protocol, and
others that are forbidden::

  Allowed    Allowed   Allowed   Forbidden  Forbidden 
  -------    -------   -------   ---------  ---------
  Send(0)    Send(1)   Send(1)   Send(0)    Send(0)
  Ack(0)     Send(1)   Send(1)   Ack(0)     Ack(1)
  Send(1)    Ack(1)    Ack(1)    Send(0)    Send(1)
  Ack(1)     Send(0)   Send(1)   Ack(0)     Ack(1)
             Ack(1)    Ack(1)
             Ack(1)    Send(1)
             Send(0)
             Ack(0)

Traces like these might be collected by a test harness connected to
the sender.  The ``Send`` are *controllable actions* invoked by the
sender while the ``Ack`` are *observable actions* that are observed by
monitoring the network.  (If the test harness were connected to the
receiver instead, the ``Send`` would be the observable actions and the
``Ack`` would be controllable.)


Finite Models
-------------

A model is an executable specification that can generate traces (to
use as test cases) or check traces (to act as an oracle).  To act as a
specification, the model must be able to generate (or accept) any
allowed trace and must not be able to generate any forbidden trace (it
must reject any forbidden trace).

The alternating bit protocol is *finite* because there are only a
finite number of actions (only a finite number of possible values for
each action argument).  Therefore this protocol can be modeled by a
*finite state machine* (FSM), which can be represented by a graph
where the edges represent actions and the nodes represent states
(Figure :ref:`abp`).  Every allowed trace can be obtained by tracing
paths around this graph.  In the figure, some of the nodes have
doubled borders.  These are the *accepting states* where traces are
allowed to stop.  A trace that stops in a non-accepting state is
forbidden.  If not accepting states are specified, all states are
considered accepting states.

.. figure:: abp.pdf
   :figclass: bht

   Alternating bit protocol represented by a finite state machine (FSM) :label:`abp`

In PyModel, a finite state machine is represented by its graph: a
tuple of tuples, where each tuple represents a state transition, the
current state (a node), the action (an edge), and the next state
(another node)::

  graph = ((0, (Send, (1,),), 0),
           (0, (Ack, (1,),), 0),
           (0, (Send, (0,),), 1),
           (1, (Ack, (0,),), 2),
           ... etc. ...
           (4, (Send, (0,),), 1))


The PyModel Graphics program ``pmg`` generated Figure :ref:`abp` from
this code.

Most interesting systems are infinite and cannot be described by
finite state machines. In PyModel, finite state machines are most
often used to describe *scenario machines* that are composed with
infinite *contract model programs* to focus test case generation on
scenarios of interest.

Infinite Models
---------------

Most interesting systems require infinite models.  A system requires
an infinite model when it has an infinite number of actions.  This
occurs whenever any of its action arguments are drawn from types
that have an infinite number of values: numbers, strings, or compound
types such as tuples, lists, dictionaries, or classes.

Simple systems can be infinite.  Consider a stack, a last-in
first-out queue which provides a ``Push`` action that puts a value on
top of the stack and a ``Pop`` action that removes the value from the top of
the stack and returns it.  Here are some allowed traces:::

  Push(1,)      Push(1,)      Push(1,)     
  Push(2,)      Pop(), 1      Push(2,)
  Push(2,)      Push(2,)      Push(2,)
  Push(1,)      Pop(), 2      Push(1,)
  Pop(), 1      Push(1,)      Push(1,)
  Pop(), 2      Pop(), 1      Push(1,)
  Pop(), 2      Push(2,)      Push(2,)
  Push(2,)      Pop(), 2      Push(2,)
  Push(1.)      Push(1,)      Push(1,)
  Push(1,)      Pop(), 1      Push(1,)

In PyModel, an infinite model is expressed by a Python module with a
an *action function* for each action and variables to represent the
*state*, the information stored in the system.  In this example, the
state is a list that stores the stack contents in order.  Constraints
on the ordering of actions are expressed by providing each action with
an optional *guard* or *enabling condition*: a Boolean function that
is true for all combinations of arguments and state variables where
the action is allowed to occur.  In this example, ``Push`` is always
enabled so no enabling function is needed; ``Pop`` is only enabled in
states where the stack is not empty.  Here is the model, as coded in
the module ``Stack``::

  stack = list()      # State

  def Push(x):        # Push is always enabled
    global stack
    stack.insert(0,x)

  def Pop():          # Pop requires an enabling condition
    global stack
    result = stack[0]
    del stack[0]
    return result

  def PopEnabled():   # Pop enabled when stack not empty
    return stack      

Analysis
--------

It can be helpful to visualize the behavior of a model program.  The
PyModel Graphics program ``pmg`` can generate a graph from finite
state machine, as in Figure :ref:`abp`.  The PyModel Analyzer ``pma``
generates a finite state machine from an infinite model program, by a
process called *exploration* which is a kind of concrete state
model-checking.  In order to finitize the model program, it is
necessary to limit the action arguments to finite *domains* and it is
also necessary to limit the state by *state filters*, Boolean
functions which the state must satisfy.  This domain limits the
arguments to ``Push`` to the domain ``0, 1`` and this state filter
limits the stack to fewer than four elements:::

  domains = { Push: {’x’:[0,1]} }

  def StateFilter():
    return len(stack) < 4

.. figure:: stack-finitized.pdf
   :figclass: bht   

   FSM for finitized Stack model program, generated by exploration. :label:`stack`

Subject to these limitions, ``pma`` generates a finite state machine
that is rendered this way by ``pmg`` (Figure :ref:`stack`).



Related work
------------

All is explained in [Jacky08]_.  


References
----------

.. [ABP11] Alternating Bit Protocol, Wikpedia, viewed June 2011.
           http://en.wikipedia.org/wiki/Alternating_bit_protocol

.. [Jacky08] Jonathan Jacky, Margus Veanes, Colin Campbell, and Wolfram Schulte.
             *Model-Based Software Testing and Analysis with C#*,
	     Cambridge University Press, 2008.


