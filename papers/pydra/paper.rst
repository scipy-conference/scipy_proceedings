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
The algorithms, that were originally not necessarily designed to work together,
often written by different authors in different times.
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
The second generation of *Nipype* ecosystem is meant to provide additional flexibility
and is being developed with reproducibility, ease of use, and scalability in mind.
*Pydra* itself is a standalone project and is designed as a general-purpose dataflow engine
to support any scientific domain.

The goal of Pydra is to provide a lightweight dataflow engine for computational graph construction,
manipulation, and distributed execution, as well as ensuring reproducibility of scientific pipelines.
In Pydra, a dataflow is represented as a directed acyclic graph, where each node represents a 
Python function, execution of an external tool, or another reusable dataflow.
The combination of several key features makes Pydra a customizable and powerful dataflow engine:

* Composable dataflows: any node of a dataflow graph can be another dataflow,
  allowing for nested dataflows to arbitrary depths and encourages creating reusable dataflows.

* Flexible semantics for creating nested loops over input.
  Any *Task* or dataflow can be run over input parameter sets and the outputs can be recombined
  (similar concept to Map-Reduce model, but Pydra extends this to graphs with nested dataflows).

* A content-addressable global cache: hash values are computed for each graph and each Task.
  Supports reuse of previously computed and stored dataflows and Tasks.

* Support for Python functions and external (shell) commands.

* Native container execution support. Any dataflow or Task can be executed in an associated
  container (via Docker or Singularity) enabling greater consistency for reproducibility.


Pydra is a pure Python package with a limited set of dependencies, which are themselves only dependent on
the Python Standard library.
Pydra uses the *attr* package for type annotation and validation of inputs and
outputs of tasks, the *cloudpickle* package to pickle interactive task definitions,
and the *pytest* testing framework.
Pydra is intended to help scientific workflows which rely on significant file-based operations and
which evaluate outcomes of complex dataflows over a hyper-space of parameters.
It is important to note, that Pydra is not a framework for writing efficient scientific algorithms
or for use in applications where caching and distributed execution are not necessary.
Since Pydra relies on a filesystem cache at present it is also not
designed for dataflows that need to operate purely in memory. 


The next section will describe the Pydra architecture --- main package classes
and interactions between them. The *Key Features* section focuses on a set of features whose
combination distinguishes Pydra from other dataflow engines. The paper concludes with a set
of applied examples demonstrating the power and utility of Pydra.


TODO: provenance??

Architecture
------------
*Pydra* architecture has three main components |---| runnable objects, i.e. *Tasks*,
*Submitter* and *Worker*.
There is one type of *Submitter*, but several types of *Tasks*
and *Workers* have been developed, see schematic presentation in Fig. :ref:`classes`.
In the following subsection all of these components will be briefly described.



.. figure:: classes.pdf
   :figclass: h!
   :scale: 30%

   A schematic presentation of principal classes in Pydra. :label:`classes`


Dataflows Components: Task and Workflow
=======================================
A *Task* is the basic runnable component of *Pydra* and is descibed by the class ``TaskBase``.
There are several classes that inherit from ``TaskBase`` and each has a different application:

* ``FunctionTask`` is a *Task* that is design as a wrapper for Python function.
  Every Python defined function could be transformed to the ``FunctionTask`` by using Pydra
  decorator - ``mark.task``.
  In addition, the user can use Python's function annotation or another Pydra decorator
  |---| ``mark.annotate`` in order to specify the output, see an example below:

  .. code-block:: python

     @mark.task
     @mark.annotate(
         {"return": {"mean": float, "std": float}}
     )
     def mean_dev(my_data):
         import statistics as st
         return st.mean(my_data), st.stdev(my_data)

     task = mean_dev(my_data=[...])

* ``ShellCommandTask`` is a *Task* that is built around shell commands and executables.
  It can be used with a simple command without any arguments, or with specific set of arguments and flags, e.g.:

  .. code-block:: python

     ShellCommandTask(executable="pwd")

     ShellCommandTask(executable="ls", args="my_dir")



  The *Task* can accommodate  much more complicated commands by allowing to customize input and output
  to specify name of the input, position in the command, flag, type, etc.
  FSL's BET command (Brain Extraction Tool) could be used as an example (note, this is only a short version
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

* ``ContainerTask`` class is a child class of ``ShellCommandTask`` an a parent class
  for ``DockerTask`` and ``SingularityTask``.
  Both *Container Tasks* run shell commands or executables within containers with specific user defined
  environments using *Docker* [ref] and *Singularity* [ref] software respectively.
  This might be extremely useful for users and projects that require environment encapsulation and sharing.
  Using container technologies allows to ensure scientific workflows reproducibility.
  These *Container Tasks* can be defined by using ``DockerTask`` and ``SingularityTask`` directly,
  or can be created automatically from ``ShellCommandTask``,
  when an optional argument ``container_info`` is used when creating a *Shell Task*.
  These two syntax are equivalent:

  .. code-block:: python

     DockerTask(executable="pwd", image="busybox")

     ShellCommandTask(executable="ls",
                      container_info=("docker", "busybox"))


* ``Workflow`` - is a special *Task* that has an additional attribute - an executable graph.
  Each node of the graph contains a *Task* of any type, and can be add to the *Workflow* simply by calling ``add`` method.
  The connections between *Tasks* are defined by using so called *Lazy Input* or *Lazy Output*,
  as it is presented by this example:

  .. code-block:: python

    # creating workflow with two input fields - x and y
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
that is used when *Task* should be run multiple times for various sets of input fields.
In order to specify how the input should be split, and optionally combined after
the *Task* execution, the user could set so called *splitter* and *combiner*.
These attributes can be set by calling ``split`` and ``combine`` methods respectively, e.g.:

.. code-block:: python

  task_state = add2(x=[1, 5]).split("x").combine("x")

If *Task* has to be split, ``State`` class is responsible for creating list of proper
set of inputs indices and values, that should be passed to the *Task* for each run.
The way how this *Task* is executed and the types of implemented *splietters*
will be discussed in details in the next section.


Submitter
=========

In order to execute *Workflows* and single *Task* with multiple set of inputs,
``Submitter`` class was created.
The goal of this class is to  start proper *Worker* depending on the user defined plugin name
and manage properly the *Tasks* execution.
The execution depends whether the *runnable* is a single *Task*, or in fact is a *Workflow*.
It does also depend whether the *Task has a *State* or not.
When the *runnable* is a *Workflow*, the *Submitter* is responsible for checking if
the *Tasks* from the graph are ready to run, i.e. if all the inputs are available,
including the inputs that are set to the *Lazy Outputs* from previous *Tasks*.
Once the *Task* is ready to run, the *Submitter* sends it to the *Worker*.
When the runnable has a *State*, than the input has to be properly split, and multiple
copy of the *Task* are sent to the *Worker*.
In order to avoid big memory consumption, the *Tasks* are sent as a pointer to a pickle file,
together with information about its state, so the proper input could be retrieved just before
running the *Task*


Workers
=======

*Workers* in *Pydra* are responsible for execution the *Tasks* and are connected
directly to the *Submitter*
At this moment *Pydra* supports three types of software: *ConcurrentFutures* [ref],
*Slurm* [ref] and *Dask* [ref].
Currently ``ConcurrentFuturesWorker`` has the best support, but ``SlurmWorker``
and ``DaskWorker`` are planned to have a full support.
When  ``ConcurrentFuturesWorker`` is created, ``ProcessPoolExecutor`` is used
to create a "pool" for adding the runnables.
``SlurmWorker`` creates a proper bash script in order to execute the runnable, using *sbatch* command,
and ``DaskWorker`` make use of ``Client`` class and the ``submit`` method.
All workers use *async functions* from *AsyncIO* in order to handle asynchronous processes.


Key Features
------------

In this section, chosen features of *Pydra* will be presented.
Some of the features are present in other packages, but the combination
of the following features makes *Pydra* a powerful tool in scientific computation.

Nested Workflows
================

*Pydra* was design to provide an easy way of creating very complex scientific workflows,
and flexible reusing already existing workflows in new applications.
This is the reason why ``Workflow`` class has been implemented as a child class of the ``TaskBase`` class,
and can be treated by users as any other *Task* and added to a new *Workflow*.
The *Submitter* is responsible for checking the type of each runnable and is able
to dynamically extend the execution graph.
This provides an easy way of creating nested workflows of arbitrary depth,
and reuse already existing *Workflows*.
This is schematically shown in Fig. :ref:`nested`.

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
similarly to the concept of the *Map-Reduce* [ref].
In order to set input splitting (or mapping), *Pydra* requires to set
so called *splitter*, it can be done by using *Task*'s ``split`` method.
The simplest example would be a *Task* that have one field *x* in the input,
and therefore there is only one way of splitting its input.
Assuming that the user provides a list as a value of *x*, *Pydra* slits
the list, so each copy of the *Task* will get one element of the list:


.. math::

   \textcolor{red}{\mathnormal{S} = x}: x=[x_1, x_2, ..., x_n] \longmapsto x=x_1, x=x_2, ..., x=x_n

That is also represented in Fig. :ref:`ndspl1`, where *x=[1, 2, 3]* as an example.

.. figure:: nd_spl_1-crop.pdf
   :figclass: h!
   :scale: 100%

   Diagram representing a Task with one input and a simple splitter. The white node represents
   an original Task with x=[1,2,3], as an input. The coloured nodes represent copies of
   the original Task after splitting the input, these are the runnables that are executed by Workers.
   :label:`ndspl1`


Whenever *Task* has more complicated input, i.e. multiple fields, there are
two ways of creating the mapping and they are called *scalar splitter*,
and *outer splitter*.

The first one, the *scalar splitter*, requires that the lists of values for two fields
have the same length, since "element wise" mapping is made.
The *scalar splitter* is represented by parenthesis, ``()``:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{S} = (x, y)} &:& x=[x_1, .., x_n], y=[y_1, .., y_n] \\
    &\mapsto& (x, y)=(x_1, y_1), ..., (x, y)=(x_n, y_n)


The situation is also represented as a diagram in Fig. :ref:`ndspl4`

.. figure:: nd_spl_4-crop.pdf
   :figclass: h!
   :scale: 90%

   Diagram representing a Task with two input fields and a scalar splitter.
   The symbol convention as described in :ref:`ndspl1`.
   :label:`ndspl4`

The second option of mapping the input, when there are multiple fields, is provided by
so called *outer splitter*.
The *outer splitter* creates all combination of the input values, and does not require
the lists to have the same lengths.
The *outer splitter* is represented by square brackets, ``[]``:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{S} = [x, y]} &:& x=[x_1, .., x_n], y=[y_1, .., y_m], \\
   &\mapsto& (x, y)=(x_1, y_1), (x, y)=(x_1, y_2)..., (x, y)=(x_n, y_m)

(todo: perhaps I can remove repetition of ``(x,y)=``??)

The *outer splitter* for a node with two input fields is schematically represented in Fig. :ref:`ndspl3`

.. figure:: nd_spl_3-crop.pdf
   :figclass: h!
   :scale: 75%

   Diagram representing a Task with two input fields and an outer splitter.
   The symbol convention as described in :ref:`ndspl1`.
   :label:`ndspl3`


In addition to the splitting the input, *Pydra* supports grouping or combining the output together.
Taking as an example the simple *Task* represented in Fig. :ref:`ndspl1`, in some application
it could be useful to combine at the end all the values of output.
In order to do it, *Task* has so called *combiner*, that could be set by calling ``combine`` method.
Note, that the *combiner* makes only sense when *splitter* is set first.
When *combiner=x*, all values are combined together within one list, and each element of the list
represents an output of the *Task* for the specific value of the input *x*.
Splitting and combining for this example can be written as follow:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{S} = x} &:& x=[x_1, x_2, ..., x_n] \mapsto x=x_1, x=x_2, ..., x=x_n \\
   \textcolor{red}{\mathnormal{C} = x} &:& out(x_1), ...,out(x_n) \mapsto out=[out(x_1), ...out(x_n)]


In the situation where input has multiple fields, there are various way of combining the output.
Taking as an example *Task* represented in Fig. :ref:`ndspl3`, it might be useful to combine all the outputs
for one specific values of *a* and all the values of *b*.
The combined output is a two dimensional list, each inner element for each value of *a*,
this could be written as follow:

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

   Diagram representing a Task with two input fields, an outer splitter and a combiner.
   The Tasks are run in exactly the same way as previously, but at the end the values of output
   for all values of *b* are combined together.
   The symbol convention as described in :ref:`ndspl1`.
   :label:`ndspl3comb1`

However, for the diagram from :ref:`ndspl3`, it might be also useful to combine all values of *a* for
specific values of *b*.
It can be also needed to combine all the values together.
This can be achieve by providing a list of fields, *[a, b]* to the combiner.
When a full combiner is set, i.e. all the fields from splitter are also in the combiner,
the output is a one dimensional list:

.. math::
   :type: eqnarray

   \textcolor{red}{\mathnormal{C} = [x, y]} : out(x_1, y1), ...out(x_n, y_m)
    \longmapsto [out(x_1, y_1), ..., out(x_n, y_m)]


And is represented in Fig. :ref:`ndspl3comb3` (todo: should probably change a,b to x,y)


.. figure:: nd_spl_3_comb3-crop.pdf
   :figclass: h!
   :scale: 75%

   Diagram representing a Task with two input fields, an outer splitter and a full combiner.
   The Tasks are run in exactly the same way as previously, but at the end all of the output values
   are combined together.
   The symbol convention as described in :ref:`ndspl1`.
   :label:`ndspl3comb3`


These are the basic examples of *Pydra*'s *splitter* and *combiners* concept.
It is important to note, that *Pydra* allows for mixing *splitters* and *combiners* on various level.
They could be set on a single *Task* level, or on *Workflow* level.
They could be also passed from one *Task* to the followings *Task* within a *Workflow*.
Some example of this flexible syntax will be presented in the next section.

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
