:author: Dorota Jarecka
:email: djarecka@gmail.com
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA

:author: Mathias Goncalves
:email: todo
:institution: Stanford University, Stanford, CA, USA
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA

:author: Christopher J. Markiewicz
:email:  todo
:institution: Stanford University, Stanford, CA, USA

:author: Oscar Esteban
:email: todo
:institution: Stanford University, Stanford, CA, USA

:author: Nicole Lo
:email: todo
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA


:author: Jakub Kaczmarzyk
:email: todo
:institution: Stony Brook University School of Medicine, Stony Brook, NY, USA
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA

:author: Satrajit Ghosh
:email: satra@mit.edu
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA


--------------------------------------------------------------------------
Pydra - a flexible and lightweight dataflow engine for scientific analyses
--------------------------------------------------------------------------

.. class:: abstract

This paper presents a new lightweight dataflow engine written
in Python: Pydra. TODO


.. class:: keywords

   dataflow engine, scientific workflows, reproducibility

Introduction
------------

Scientific workflows often require sophisticated analyses that encompass
a large collection of algorithms.
Some may be written in Python, while others might require calling external programs.
It is a common practice to create semi-manual workflows that require scientist
interaction to handle files and partial results between algorithms and external tools.
This approach is conceptually simple and easy to implement, but the resulting workflow
is often time consuming, error-prone and difficult to share with others.
Consistency, reproducibility and scalability demand scientific workflows be organized
into fully automated pipelines.
This was the motivation behind writting a new dataflow engine written in Python, *Pydra*,
that is presented in this paper.

The package is a part of the second generation of the *Nipype* ecosystem [TODO: ref]
--- an open-source framework that provides a uniform interface to existing neuroimaging
software and facilitates interaction between them.
Nipype project was born in the neuroimaging community, and has been helping scientists build
workflows for a decade, providing a uniform interface to such neuroimaging packages
as FSL [ref], ANTs [ref], AFNI [ref], FreeSurfer [ref] and SPM [ref].
This flexibility has made it an ideal basis for popular preprocessing tools,
such as fMRIPrep [ref] and C-PAC[ref].
The second generation of *Nipype* ecosystem is meant to provides additional flexibility
and is being developed with reproducibility, ease of use, and scalability in mind.
Pydra is a standalone project and is designed as a general-purpose dataflow engine
to support any scientific domain.

The goal of Pydra is to provide a lightweight dataflow engine for computational graph construction,
manipulation, and distributed execution, as well as ensuring reproducibility of scientific pipelines.
In Pydra, a dataflow is represented as a directed acyclic graph, where each node represents a 
Python function, execution of an external tool, or another reusable dataflow.
The combination of several key features makes Pydra a customizable and powerful dataflow engine:

* Composable dataflows: any node of a dataflow graph can be another dataflow,
  allowing for nested dataflows to arbitrary depths and encourages creating reusable dataflows.

* Flexible semantics for creating nested loops over input.
  Any Task or dataflow can be run over input parameter sets and the outputs can be recombined
  (similar concept to Map-Reduce model, but Pydra extends this to graphs with nested dataflows).

* A content-addressable global cache: hash values are computed for each graph and each Task.
  Supports reuse of previously computed and stored dataflows and Tasks.

* Support for Python functions and external (shell) commands.

* Native container execution support. Any dataflow or Task can be executed in an associated
  container (via Docker or Singularity) enabling greater consistency for reproducibility.


Pydra is a pure Python package with a limited set of dependencies, which are themselves only dependent on
the Python Standard library. Pydra uses the *attr* package for type annotation and validation of inputs and 
outputs of tasks, the *cloudpickle* package to pickle interactive task definitions, and the *pytest* testing 
framework. Pydra is intended to help scientific workflows which rely on significant file-based operations and 
which evaluate outcomes of complex dataflows over a hyper-space of parameters. It is important to note, that
Pydra is not a framework for writing efficient scientific algorithms or for use in applications where caching and 
distributed execution are not necessary. Since Pydra relies on a filesystem cache at present it is also not
designed for dataflows that need to operate purely in memory. 


The next section will describe the Pydra architecture --- main package classes
and interactions between them. The *Key Features* section focuses on a set of features whose
combination distinguishes Pydra from other dataflow engines. The paper concludes with a set
of applied examples demonstrating the power and utility of Pydra.


TODO: provenance

Architecture
------------
*Pydra* architecture has three main components |---| runnable objects, i.e. *Tasks*,
*Submitter* and *Worker*.
There is one type of *Submitter*, but several types of *Tasks*
and *Workers* have been developed, see schematic presentation in Fig. :ref:`classes`.



.. figure:: classes.pdf
   :figclass: h!
   :scale: 30%

   A schematic presentation of principal classes in Pydra. :label:`classes`



Dataflows Components: Task and Workflow
=======================================
A *Task* is the basic runnable component of Pydra and is descibed by the class ``TaskBase``.
There are several classes that inherit from ``TaskBase`` and each has a different application:

* ``FunctionTask`` is a *Task* that is design as a wrapper for Python function.
  Every Python defined function could be tranformed to the ``FunctionTask`` by using Pydra
  decorator - ``mark.task``.
  In addition, the user can use Python's function annotation or another Pydra decorator
  |---| ``mark.annotate`` to specify the output, see an example below:

  .. code-block:: python

     @mark.task
     @mark.annotate(
         {"return": {"mean": float, "std": float}}
     )
     def mean_dev(my_data):
         import statistics as st
         return st.mean(my_data), st.stdev(my_data)

     task = mean_dev(my_data=[...])

* ``ShellCommandTask`` is a *Task* that is built around shell commands.
  It can be used with a simple command without any arguments, or with specific set of arguments, e.g.:

  .. code-block:: python

     ShellCommandTask(executable="pwd")

     ShellCommandTask(executable="ls", args="my_dir")



  The *Task* can accomodate  much more complicated commands by allowing to customize input and output
  to specify position name of the input, position in the command, flag, type, etc. FSL's BET command
  (Brain Extraction Tool) could be used as an example (note, this is only a short version
  of specification and not fully working example):

  .. code-block:: python

    bet_input_spec = SpecInfo(
        name="Input",
        fields=[
        (
            "in_file",
            File,
            {
             "help_string": "input file ...",
             "position": 1,
             "mandatory": True,
            }
        ),
        (
            "out_file",
            str,
            {
             "help_string": "name of output ...",
             "position": 2,
             "output_file_template": "{in_file}_br",
            }
        ),
        (
            "mask",
            bool,
            {
             "help_string": "create binary mask",
             "argstr": "-m",
             }
        )
        ],
        bases=(ShellSpec,),
    )

    ShellCommandTask(executable="bet",
                     input_spec=bet_input_spec)

* ``ContainerTask`` class is child class of ``ShellCommandTask`` an a parent class
  for ``DockerClass`` and ``SingularityTask``.
  Both *Container Tasks* run shell commands within containers with specific user defined environments
  using *Docker* [ref] and *Singularity* [ref] software respectively.
  These classes can be defined directly, or can be created automatically,
  when an optional argument ``container_info`` is used when creating a *Shell Task*.
  These two syntax are equivalent:

  .. code-block:: python

     DockerTask(executable="pwd", image="busybox")

     ShellCommandTask(executable="ls",
                      container_info=("docker", "busybox"))


* ``Workflow`` - is a special *Task* that has an additional attribute - an executable graph.
  Each node of the graph contains a *Task* of any type, and can be add simply by calling ``add`` method,
  and the connections are defined by using so called *Lazy Input* or *Lazy Output*, e.g.:

  .. code-block:: python

    wf = Workflow(input_spec=["x", "y"])
    # adding a task and connecting task's input
    # to the workflow input
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    # adding anoter task and connecting task's input
    # to the "mult" task's output
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    # setting worflow output
    wf.set_output([("out", wf.add2.lzout.out)])


State
=====

All *Tasks*, including *Workflows*, could have an optional ``State`` attribute,
that is used when *Task* should be run multiple times for various sets of input.
In order to specify how the input should be split, and optionally combined after
the *Task* execution, the user could set so called *splitter* and *combiner*,
by calling ``split`` and ``combine`` methods, e.g.:

.. code-block:: python

  task_state = add2(x=[1, 5]).split("x").combine("x")

Implemented types of *splietters* will be discussed in details in the next section.
If *Task* has to be split, ``State`` class is responsible for creating list of proper
set of inputs indices and values, that should be run for each run.


Submitter
=========

In order to execute *Workflows* and single *Task* with multiple set of inputs,
``Submitter`` class was created.
The goal of this class is to manage properly the *Tasks*,
that is needed when *Tasks* has *state*, or is a *Workflow*.
This class is responsible for checking if particular *Tasks* are ready
to run, i.e. if all the inputs that are connected to outputs from different *Tasks*
are available.


Workers
=======

*Workers* in *Pydra* are responsible for execution the *Tasks*.
At this moment *Pydra* supports three types of software: *ConcurrentFutures* [ref],
*Slurm* [ref] and *Dask* [ref].
Currently ``ConcurrentFuturesWorker`` has the biggest support, but ``SlurmWorker``
and ``DaskWorker`` are planned to have a full support.



Key Features
------------

In this section, chosen features of *Pydra* will be presented.
Some of the features are present in other packages, but the combination
of the following features makes *Pydra* a powerful tool in scientific computation.

Nested Workflows
================

*Workflows* in *Pydra* can contain multiple *Tasks*, but they are still *Tasks*,
and have all of the *Tasks* attributes and methods.
As a consequence, a *Workflow* can be also used as a node in the executable graph.
This provides an easy way of creating nested workflows of arbitrary depth,
as shown in Fig. :ref:`nested`.

.. figure:: nested_workflow-crop.pdf
   :figclass: h!
   :scale: 40%

   A nested Pydra workflow, black circles represent single Task,
   and Workflows are represented by red rectangles. :label:`nested`




State and Nested Loops over Input
=================================



One of the main goal of *Pydra* is to support flexible creation
of loops over inputs, i.e. flexible mapping of the values of the
user provided inputs to the specific *Task*'s execution,
similarly to the concept of the *Map-Reduce*.
In order to set input splitting (or mapping), *Pydra* requires to set
so called *splitter*, it can be done by using method ``split``.
The simplest example if for *Task* that have one field in the input,
and therefore there is only one way of splitting the input:


.. math::

   \textcolor{red}{\mathnormal{S} = x}: x=[x_1, x_2, ..., x_n] \longmapsto x=x_1, x=x_2, ..., x=x_n

(TODO: mathcall font doesn't work, using mathnormal for now)

That is also represented in Fig. :ref:`ndspl1`

.. figure:: nd_spl_1-crop.pdf
   :figclass: h!
   :scale: 100%

   Diagram representing a Task with one input and a simple splitter. :label:`ndspl1`


Whenever *Task* has more complicated input, i.e. multiple fields, there are
two ways of creating the mapping and in *Pydra* API there are called *scalar splitter*,
and *outer splitter*.

The first one, the *scalar splitter* requires that the lists of values for two fields
have the same length, since "element wise" mapping is made.
It is represented by parenthesis, ``()``:

.. math::

   \textcolor{red}{\mathnormal{S} = (x, y)}: x=[x_1, .., x_n], y=[y_1, .., y_n], \longmapsto (x, y)=(x_1, y_1), ..., (x, y)=(x_n, y_n)


This is also represented as a diagram in Fig. :ref:`ndspl4`

.. figure:: nd_spl_4-crop.pdf
   :figclass: h!
   :scale: 90%

   Diagram representing a Task with two input fields and a scalar splitter. :label:`ndspl4`

The second option of mapping the input when there are multiple fields is supported by the *outer splitter*
and representhed by square brackets, ``[]``.
When *outer splitter* is used all combination are created:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{S} = [x, y]} &:& x=[x_1, .., x_n], y=[y_1, .., y_m], \\
   &\longmapsto& (x, y)=(x_1, y_1), (x, y)=(x_1, y_2)..., (x, y)=(x_n, y_m)

(todo: perhaps I can remove repetition of ``(x,y)=``??)

This is schematically represented in Fig. :ref:`ndspl3`

.. figure:: nd_spl_3-crop.pdf
   :figclass: h!
   :scale: 75%

   Diagram representing a Task with two input fields and an outer splitter. :label:`ndspl3`


In addition to the splitting the input, *Pydra* supports grouping or combining the output together.
Taking as an example the simple *Task* represented in Fig. :ref:`ndspl1`, in some application
it could be useful to combine all the outputs at the end.
In order to do it *Task* has to have so called *combiner* that could be set by calling ``combine`` method.
This could be written as follow:


.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{S} = x} &:& x=[x_1, x_2, ..., x_n] \longmapsto x=x_1, x=x_2, ..., x=x_n \\
   \textcolor{red}{\mathnormal{C} = x} &:& out(x_1), out(x_2), ...out(x_n) \longmapsto out=[out(x_1), out(x_2), ...out(x_n)]


Again, in the situation where input has multiple fields, there are various way of combining the output.
Taking as an example *Task* represented in Fig. :ref:`ndspl4`, it could be useful to combine all the outputs
for one specific values of *x* and all the values of *y*.
The combining operation could be written as follow:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{C} = y} &:& out(x_1, y1), out(x_1, y2), ...out(x_n, y_m) \\
    &\longmapsto& [[out(x_1, y_1), ..., out(x_1, y_m)], \\
    && ..., \\
    && [out(x_n, y_1), ..., out(x_n, y_m)]]

And is represented in Fig. :ref:`ndspl3comb1` (todo: should probably change a,b to x,y)


.. figure:: nd_spl_3_comb1-crop.pdf
   :figclass: h!
   :scale: 75%

   Diagram representing a Task with two input fields, an outer splitter and a combiner. :label:`ndspl3comb1`

It is also possible to combine all values of *x* for a specifc values of *y*, but it is also possible
to combine all the values by providing a list of fields to the combiner.
When a full combiner is set, i.e. all fields are in the combiner, the output is a one dimensional list:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{C} = [x, y]} : out(x_1, y1), ...out(x_n, y_m)
    \longmapsto [out(x_1, y_1), ..., out(x_n, y_m)]


And is represented in Fig. :ref:`ndspl3comb3` (todo: should probably change a,b to x,y)


.. figure:: nd_spl_3_comb3-crop.pdf
   :figclass: h!
   :scale: 75%

   Diagram representing a Task with two input fields, an outer splitter and a full combiner. :label:`ndspl3comb3`


These are the basic examples of *splitters* and *combiners*, but *Pydra* allows for mixing
*splitters* and *combiners* on various level.
They could be set on a single *Task* level, or on *Workflow* level.
They could be also passed from one *Task* to the followings within a *Workflow*.


Global Cache
============



Applications and Examples
-------------------------

Machine Learning: Model Comparison
==================================


.. code-block:: python

  ml example TODO


Summary and Future Directions
-----------------------------



Acknowledgement
---------------
This was supported by NIH grants P41EB019936, R01EB020740.
We thank the neuroimaging community for feedback during development.
