:author: Dorota Jarecka
:email: djarecka@gmail.com
:institution: Massachusetts Institute of Technology, Cambridge, MA, USA

:author: Mathias Goncalves
:email: mathiasg@stanford.edu
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
The *Nipype* project was born in the neuroimaging community, and has been helping scientists build
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

* **Composable dataflows:** Any node of a dataflow graph can be another dataflow,
  allowing for nested dataflows to arbitrary depths and encourages creating reusable dataflows.

* **Flexible semantics for creating nested loops over input sets:**
  Any *Task* or dataflow can be run over input parameter sets and the outputs can be recombined
  (similar concept to Map-Reduce model, but Pydra extends this to graphs with nested dataflows).

* **A content-addressable global cache:** Hash values are computed for each graph and each Task.
  Supports reuse of previously computed and stored dataflows and Tasks.

* **Can integrate Python functions and external (shell) commands:** Pydra can
  decorate and use existing functions in Python libraries alongside external
  command line tools, allowing easy integration of existing code and software.

* **Native container execution support:** Any dataflow or Task can be executed in an associated
  container (via Docker or Singularity) enabling greater consistency for reproducibility.

* **Auditing and provenance tracking:** Pydra provides a simple JSON-LD -based message
  passing mechanism to capture the dataflow execution activties as a provenance
  graph. These messages track inputs and outputs of each task in a dataflow, and
  the resources consumed by the task.


*Pydra* is a pure Python 3.7+ package with a limited set of dependencies, which are
themselves only dependent on the Python Standard library. It leverages *type annotation*
and *AsyncIO* in its core operations. Pydra uses the *attr* package for extended
annotation and validation of inputs and outputs of tasks, the *cloudpickle* package
to pickle interactive task definitions, and the *pytest* testing framework.
*Pydra* is intended to help scientific workflows which rely on significant file-based operations and
which evaluate outcomes of complex dataflows over a hyper-space of parameters.
It is important to note, that *Pydra* is not a framework for writing efficient scientific algorithms
or for use in applications where caching and distributed execution are not necessary.
Since *Pydra* relies on a filesystem cache at present it is also not
designed for dataflows that need to operate purely in memory. 

The next section will describe the *Pydra* architecture --- main package classes
and interactions between them. The *Key Features* section focuses on a set of features whose
combination distinguishes *Pydra* from other dataflow engines. The paper concludes with a set
of applied examples demonstrating the power and utility of *Pydra*.


Architecture
------------
*Pydra architecture has three core components: *Task*, *Submitter* and *Worker*.
*Tasks* form the basic building blocks of the dataflow, while *Submitter*
orchestrates the dataflow execution model. Different types of *Workers* allow

*Pydra* to execute the task on different compute architectures. Fig. :ref:`classes`
shows the Class hierarchy and links between them in the present Pydra
architecture. It was designed this way to decouple and allow *Workers* to
operate.  In order to describe *Pydra*'s most notable features in the next
section, we briefly describe the role and function of each of these classes.

.. figure:: classes.pdf
   :figclass: h!
   :scale: 30%

   A schematic presentation of principal classes in *Pydra*. :label:`classes`

Dataflows Components: Task and Workflow
=======================================
A *Task* is the basic runnable component of *Pydra* and is described by the
class ``TaskBase``. A *Task* has named inputs and outputs thus allowing
construction of dataflows. It can be hashed and executes in a specific working
directory. There are several classes that inherit from ``TaskBase`` and each has
a different application:

* ``FunctionTask`` is a *Task* that executes Python functions. Most Python functions
  declared in an existing library, package, or interactively in a terminal can
  be converted to a ``FunctionTask`` by using *Pydra* decorator - ``mark.task``.

  .. code-block:: python

     import numpy as np
     from pydra import mark
     fft = mark.annotate({'a': np.ndarray,
                      'return': float})(np.fft.fft)
     fft_task = mark.task(fft)()
     result = fft_task(a=np.random.rand(512))


  `fft_task` is now a Pydra task and result will contain a Pydra ``Result`` object.
  In addition, the user can use Python's function annotation or another Pydra
  decorator |---| ``mark.annotate`` in order to specify the output. In the
  following example, we decorate an arbitrary Python function to create named
  outputs.

  .. code-block:: python

     @mark.task
     @mark.annotate(
         {"return": {"mean": float, "std": float}}
     )
     def mean_dev(my_data):
         import statistics as st
         return st.mean(my_data), st.stdev(my_data)

     result = mean_dev(my_data=[...])()

  When the task is executed `result.output` will contain two attributes: `mean`
  and `std`. These named outputs allowing passing different outputs to
  different downstream nodes in a dataflow.

* ``ShellCommandTask`` is a *Task* used to run shell commands and executables.
  It can be used with a simple command without any arguments, or with specific
  set of arguments and flags, e.g.:

  .. code-block:: python

     ShellCommandTask(executable="pwd")

     ShellCommandTask(executable="ls", args="my_dir")

  The *Task* can accommodate more complex shell commands by allowing the user to
  customize inputs to and output of commands. Once can generate an input
  specification to specify names of inputs, positions in the command, types of
  the inputs, and other metadata. As a specific example, FSL's BET command (Brain
  Extraction Tool) can be called on the command line as:

  .. code-block:: python

    bet input_file output_file -m

  Each of these inputs can be augmented as a named argument to the
  ``ShellCommandTask``. As shown next, even an output is specified by specifying
  how to construct the out_file field using a template:

  .. code-block:: python

    bet_input_spec = SpecInfo(
        name="Input",
        fields=[
        ( "in_file", File,
          { "help_string": "input file ...",
            "position": 1,
            "mandatory": True } ),
        ( "out_file", str,
          { "help_string": "name of output ...",
            "position": 2,
            "output_file_template":
                              "{in_file}_br" } ),
        ( "mask", bool,
          { "help_string": "create binary mask",
            "argstr": "-m", } ) ],
        bases=(ShellSpec,) )

    ShellCommandTask(executable="bet",
                     input_spec=bet_input_spec)

  Outputs can also be specified separately using a similar output specification.

* ``ContainerTask`` class is a child class of ``ShellCommandTask`` and serves as
  a parent class for ``DockerTask`` and ``SingularityTask``. Both *Container Tasks*
  run shell commands or executables within containers with specific user defined
  environments using *Docker* [ref] and *Singularity* [ref] software respectively.
  This might be extremely useful for users and projects that require environment
  encapsulation and sharing. Using container technologies helps improve scientific
  workflows reproducibility. These *Container Tasks* can be defined by using
  ``DockerTask`` and ``SingularityTask`` classes directly, or can be created
  automatically from ``ShellCommandTask``, when an optional argument
  ``container_info`` is used when creating a *Shell Task*. The following two
  syntaxes are equivalent:

  .. code-block:: python

     DockerTask(executable="pwd", image="busybox")

     ShellCommandTask(executable="ls",
          container_info=("docker", "busybox"))


* ``Workflow`` - is a subclass of *Task* that provides support for creating *Pydra*
  dataflows. As a subclass, a *Workflow* acts like a *Task* and has inputs, outputs,
  is hashable, and is treated as a single unit. Unlike *Tasks*, workflows embed
  a directed acyclic graph. Each node of the graph contains a *Task* of any type,
  including another *Workflow*, and can be added to the *Workflow* simply by calling
  the ``add`` method. The connections between *Tasks* are defined by using so
  called *Lazy Inputs* or *Lazy Outputs*. These are special attributes that allow
  assignment of values when a *Workflow* is executed rather than at the point of
  assignment. The following example creates a *Workflow* from two *Pydra* *Tasks*.

  .. code-block:: python

    # creating workflow with two input fields
    wf = Workflow(input_spec=["x", "y"])
    # adding a task and connecting task's input
    # to the workflow input
    wf.add(mult(name="mlt",
                   x=wf.lzin.x, y=wf.lzin.y))
    # adding anoter task and connecting
    # task's input to the "mult" task's output
    wf.add(add2(name="add", x=wf.mlt.lzout.out))
    # setting worflow output
    wf.set_output([("out", wf.add.lzout.out)])


State
=====

All *Tasks*, including *Workflows*, can have an optional attribute representing
an instance of the ``State`` class. This attribute controls the execution of a
*Task* over different input parameter sets. This class is at the heart of *Pydra's*
powerful map-reduce over arbitrary inputs of nested dataflows feature. The ``State``
class formalizes how users can specify arbitrary combinations. Its functionality
is used to create and track different combinations of input parameters, and
optionally allow limited or complete recombinations. In order to specify how the
inputs should be split into parameter sets, and optionally combined after
the *Task* execution, the user can set *splitter* and *combiner* attributes of the
``State`` class. These attributes can be set by calling ``split`` and ``combine``
methods in the *Task* class. Here we provide a simple map-reduce example:

.. code-block:: python

  task_with_state =
        add2(x=[1, 5]).split("x").combine("x")

In this example, the ``State`` class is responsible for creating a list of two
separate inputs, which should be passed to the *Task* for each run, and grouped
back when returning the result from the *Task*. While this example
illustrates mapping and grouping of results over a single parameter, Pydra
extends this to arbitrary combinations of input fields and downstream grouping
over nested dataflows. Details of how splitters and combiners power Pydra's
scalable dataflows are described later.


Submitter
=========

The ``Submitter`` class is responsible for unpacking *Workflows* and single
*Tasks* with or without ``State`` into standalone stateless jobs that are then
executed on *Workers*. When the *runnable* is a *Workflow*, the *Submitter* is
responsible for checking if the *Tasks* from the graph are ready to run, i.e. if
all the inputs are available, including the inputs that are set to the
*Lazy Outputs* from previous *Tasks*. Once a *Task* is ready to run, the
*Submitter* sends it to a *Worker*. When the runnable has a *State*, then the
*Submitter* unpacks the *State* and sends multiple jobs to the *Worker* for the
same *Task*. In order to avoid memory consumption as a result of scaling of *Tasks*,
each job is sent as a pointer to a pickle file, together with information about
its state, so that proper input can be retrieved just before running the *Task*.
*Submitter* uses *AsyncIO* to manage all job executions to work in parallel,
allowing scaling of execution as *Worker* resources are made available.

Workers
=======

*Workers* in *Pydra* are responsible for the actual execution of the *Tasks* and
are initialized by the *Submitter*. *Pydra* supports three types of execution
managers: *ConcurrentFutures* [ref], *Slurm* [ref] and *Dask* [ref] (experimental).
When  ``ConcurrentFuturesWorker`` is created, ``ProcessPoolExecutor`` is used
to create a "pool" for adding the runnables. ``SlurmWorker`` creates an`sbatch`
submission script in order to execute the task, and ``DaskWorker`` make use of
Dask's ``Client`` class and its ``submit`` method. All workers use
*async functions* from *AsyncIO* in order to handle asynchronous processes. All
*Workers* rely on a `load_and_run` function to execute each job from its pickled
state.


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

In this section a few example of *Pydra* usage will be presented.
The first example will be a "toy example" to show the power of *Pydra*'s splitter and combiner.
The second example will cover machine learning model comparison.

Mathematical Toy Example: Sine Function Approximation
=====================================================

In this section a toy mathematical example will be used to present
the flexibility of *Pydra*'s splitters and combiners.
The exemplary workflow will calculate the approximated values of Sine function
for various values of `x`.
The *Workflow* uses a Taylor polynomial for Sine function:

.. math::

  \sum_{n=0}^{n_{max}} \frac{(-1)^n}{(2n+1)!} x^{2n+1} = x -\frac{x^3}{3!} + \frac{x^5}{5!} + ...

where `n_{max}` (TODO) is a degree of approximation.

Since the idea is to make the execution parallel as much as possible, each of the term
for each value of `x` should be calculated separately, and this will be done by functin `term (x, n)`.
In addition, `range_fun(n_max)` will be used to return a list of integers from `0` to `n_max`,
and `summing(terms)` will sum all the terms for the specific value of `x` and `n_max`.


.. code-block:: python

  from pydra import Workflow, Submitter, mark
  import math

  @mark.task
  def range_fun(n_max):
      return list(range(n_max+1))

  @mark.task
  def term(x, n):
      import math
      fract = math.factorial(2 * n + 1)
      polyn = x ** (2 * n + 1)
      return (-1)**n * polyn / fract

  @mark.task
  def summing(terms):
      return sum(terms)


The *Workflow* itself will take two inputs - list of values of `x`
and list of values of `n_max`.
In order to calculate various degrees of the approximation for each value of `x`,
the `outer splitter` has to be used `[x, n_max]`.
At the end all approximations for the specific values of `x` will be combined
together by using `n_max` as a combiner.


.. code-block:: python

  wf = Workflow(name="wf", input_spec=["x", "n_max"])
  wf.split(["x", "n_max"]).combine("n_max")
  wf.inputs.x = [0, 0.5 * math.pi, math.pi]
  wf.inputs.n_max = [2, 4, 10]

All three *Function Tasks* have to be added to the *Workflow* and connected together.
The second task, `term`, has to be additionally split over `n`,
and at the end combine all the terms together.

.. code-block:: python


  wf.add(range_fun(name="range", n_max=wf.lzin.n_max))
  wf.add(term(name="term", x=wf.lzin.x, n=wf.range.lzout.out).
         split("n").combine("n"))
  wf.add(summing(name="sum", terms=wf.term.lzout.out))


After setting the *Workflow* output by using ``set_output`` method,
the *Workflow* could be run.

.. code-block:: python

   wf.set_output([("sin", wf.sum.lzout.out)])
   res = wf(plugin="cf")


The result gives a two dimensional list of `Results`, for each value of `x` will be a list of
three approximations, as an example, for `x=\pi/2` there should be the following list:

.. code-block:: python

 [...[Result(output=Output(sin=1.0045248555348174),
             runtime=None, errored=False),
      Result(output=Output(sin=1.0000035425842861),
             runtime=None, errored=False),
      Result(output=Output(sin=1.0000000000000002),
             runtime=None, errored=False)],
 ...]


Each `Result` contains three elements: `output`, `runtime` and `errored`.
As expected, the values of the Sine function are getting closer to `1` with higher degree of the approximation.

The described *Workflow* is schematically presented in Fig. :ref:`wfsin`.

.. figure:: wf_10_paper-crop.pdf
   :figclass: ht
   :scale: 60%

   Diagram representing part of the Workflow for calculating Sine function approximations of various degrees
   for various values of x.
   The symbol convention as described in :ref:`ndspl1`.
   :label:`wfsin`



Machine Learning: Model Comparison 
==================================

The massive parameter space in machine learning makes it a perfect use case for *Pydra*. 

Here we show an example of a general-purpose machine learning *Pydra* *Workflow*, which perform model comparison 
across a given dictionary of classifiers and associated hyperparameters:

*pandas* and *Pydra* 

.. code-block:: python

  clfs = [
   ('sklearn.ensemble', 'ExtraTreesClassifier',
    dict(n_estimators=100)),
   ('sklearn.neural_network', 'MLPClassifier',
    dict(alpha=1, max_iter=1000)),
   ('sklearn.neighbors', 'KNeighborsClassifier', dict(),
   [{'n_neighbors': [3, 7, 15],
     'weights': ['uniform','distance']}]),
   ('sklearn.ensemble', 'AdaBoostClassifier', dict())]


It leverages *Pydra*'s powerful splitters and combiners to scale across a set of classifiers and metrics.  
It will also use *Pydra*'s caching to not redo model training and evaluation when new metrics 
are added, or when number of iterations is increased.  This is a shorten version of the *pydraml*
package implemented here TODO


Let use the iris dataset as an example.

.. code-block:: python

  from sklearn import datasets
  import pandas as pd
  X, y = datasets.load_iris(return_X_y=True)
  dat = pd.DataFrame(X,
          columns=['sepal_length', 'sepal_width',
                   'petal_length', 'petal_width'])
  dat['label'] = y


Have a look at the structure of the data and save it to a csv.  Goal is to write a workflow that will read 
and process any data in the same format.

.. code-block:: python

  print(dat.sample(5))
     sepal_length sepal_width petal_length petal_width label
  137     6.4         3.1         5.5         1.8        2
  55      5.7         2.8         4.5         1.3        1
  127     6.1         3.0         4.9         1.8        2
  4       5.0         3.6         1.4         0.2        0
  68      6.2         2.2         4.5         1.5        1
  dat.to_csv('iris.csv')




Our *Workflow* consist of 3 *Task*s, each *Task* approximately corresponds to:

  1. Load & split data
  2. Set up model selection method
  3. Preprocessed, tune & compare models 


*Task* 1 reads csv data as a *pandas* *DataFrame* from a path, with the option define name of target 
variables, row indices to train and data grouping.  It returns the training data, labels
and grouping, corresponding to the `X`, `Y` and `groups` inputs to *Task* 2.

.. code-block:: python

  @mark.task 
  @mark.annotate({"return": {
      "X": ty.Any, "Y": ty.Any, "groups": ty.Any}})
  def read_data(filename, x_indices=None,
                target_vars=None, group='groups'):
     import pandas as pd
     data = pd.read_csv(filename)
     X = data.iloc[:, x_indices]
     Y = data[target_vars]
     if group in data.keys():
         groups = data[:, [group]]
     else:
         groups = list(range(X.shape[0]))
     return X.values, Y.values, groups


*Task* 2 generates a set of train-test splits with `GroupShuffleSplit` in `scikit-learn` given `n_splits` 
and `test_size`, with the option to define `group` and `random_state`. It returns `train_test_splits`

.. code-block:: python

  @mark.task  
  @mark.annotate({"return":
      {"splits": ty.Any, "split_indices": ty.Any}})
  def gen_splits(n_splits, test_size, X, Y,
                 groups=None, random_state=0):
      """Generate a set of train-test splits"""
      from sklearn.model_selection import GroupShuffleSplit
      gss = GroupShuffleSplit(n_splits=n_splits,
                              test_size=test_size,
                              random_state=random_state)
      train_test_splits = list(gss.split(X, Y,
                                         groups=groups))
      split_indices = list(range(n_splits))
      return train_test_splits, split_indices


Now we need to train the classifiers. The most optimized model for a classifer can be easily found
using *scikit-learn*'s `GridSearchCV` given a parameter grid.   However, there isn't a easy way in 
*scikit-learn* to compare models across a variety of classifiers without using loops, especially
when some classifier don't requires tuning.  


*Task* 3 train and tests classifiers on actual or permuted labels given outputs of *Task* 2 and 
a dictionary in the same format as `clfs` shown earlier.  We can then compare f1 scores from
models fit on actual and permuted data to evaluate


.. code-block:: python

  @mark.task
  @mark.annotate({"return": {"f1": ty.Any}})
  def train_test_kernel(X, y, train_test_split,
                 split_index, clf_info, permute):
     
     from sklearn.preprocessing import StandardScaler
     from sklearn.pipeline import Pipeline
     from sklearn.metrics import f1_score
     from sklearn.model_selection import GridSearchCV
     import numpy as np
     mod = __import__(clf_info[0],
                      fromlist=[clf_info[1]])
     clf = getattr(mod, clf_info[1])(**clf_info[2])
     if len(clf_info) > 3:
         # Run a GridSearch when param_grid available
         clf = GridSearchCV(clf, param_grid=clf_info[3])
     train_index, test_index =
                  train_test_split[split_index]
     pipe = Pipeline([('std', StandardScaler()),
                      (clf_info[1], clf)])
     y = y.ravel()
     if permute:
         # Run a generic permut. to create a null model
         pipe.fit(X[train_index],
                  y[np.random.permutation(train_index)])
     else:
         pipe.fit(X[train_index], y[train_index])
     f1 = f1_score(y[test_index],
                   pipe.predict(X[test_index]),
                   average='weighted')
     return round(f1, 4)


Now we add everything together in a *Workflow*.  Here is where *Pydra*'s splitter really gets to shine. 
An outer split for `clf_info` and `permute` on the *Workflow*-level means every classifier and permutation
combination gets run through the pipeline.   TODO




.. code-block:: python

  # Encapsulate tasks in a Workflow,
  # reuse script output cache
  wf = Workflow(name="ml_wf", **inputs,
                input_spec=list(inputs.keys()),
                # workflow cache
                cache_dir=wf_cache_dir,
                # reuses script cache
                cache_locations=[cache_dir])

  # joint map over classifiers and permutation
  wf.split(['clf_info', 'permute'])
  wf.add(read_file(name="readcsv",
                   # connect workflow input
                  filename=wf.lzin.filename,
                  x_indices=wf.lzin.x_indices,
                  target_vars=wf.lzin.target_vars))

  wf.add(gen_splits(name="gensplit",
            # connect workflow input
            n_splits=wf.lzin.n_splits,
            test_size=wf.lzin.test_size,
            # connect lazy-eval output of previous task
            X=wf.readcsv.lzout.X, Y=wf.readcsv.lzout.Y,
            groups=wf.readcsv.lzout.groups))

  wf.add(train_test_kernel(name="fit_clf",
            # use outputs from both tasks
            X=wf.readcsv.lzout.X, y=wf.readcsv.lzout.Y,
            train_test_split=wf.gensplit.lzout.splits,
            split_index=wf.gensplit.lzout.split_indices,
            clf_info=wf.lzin.clf_info,
            permute=wf.lzin.permute))

  # Parallel spec
  wf.fit_clf.split('split_index').combine('split_index')
  # connect workflow output
  wf.set_output([("f1", wf.fit_clf.lzout.f1)])



TODO explain results and return inputs


.. code-block:: python

  inputs = {"filename": 'iris.csv',
           "x_indices": range(4), "target_vars": ("label"),
           "n_splits": 3, "test_size": 0.2,
           # same clf shown earlier
           "permute": [True, False], "clf_info": clfs}
  n_procs = 8 # for parallel processing
  cache_dir = os.path.join(os.getcwd(), 'cache')
  wf_cache_dir = os.path.join(os.getcwd(), 'cache-wf')

  # Execute the workflow in parallel using multiple processes
  with pydra.Submitter(plugin="cf", n_procs=n_procs) as sub:
      sub(runnable=wf)
  
  print(wf.result(return_inputs=True))

  [({'ml_wf.clf_info':
         ('sklearn.ensemble','ExtraTreesClassifier',
          {'n_estimators': 100}),
     'ml_wf.permute': True},
    Result(output=Output(f1=[0.2622, 0.1733, 0.2975]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
          ('sklearn.ensemble', 'ExtraTreesClassifier',
           {'n_estimators': 100}),
     'ml_wf.permute': False},
    Result(output=Output(f1=[1.0, 0.9333, 0.9333]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
          ('sklearn.neural_network', 'MLPClassifier',
           {'alpha': 1, 'max_iter': 1000}),
     'ml_wf.permute': True},
    Result(output=Output(f1=[0.2026, 0.1468, 0.2952]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
          ('sklearn.neural_network', 'MLPClassifier',
           {'alpha': 1, 'max_iter': 1000}),
     'ml_wf.permute': False},
    Result(output=Output(f1=[1.0, 0.9667, 0.9668]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
         ('sklearn.neighbors', 'KNeighborsClassifier', {},
          [{'n_neighbors': [3, 7, 15],
            'weights': ['uniform', 'distance']}]),
     'ml_wf.permute': True},
    Result(output=Output(f1=[0.1813, 0.1111, 0.4326]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
         ('sklearn.neighbors', 'KNeighborsClassifier', {},
          [{'n_neighbors': [3, 7, 15],
            'weights': ['uniform', 'distance']}]),
     'ml_wf.permute': False},
    Result(output=Output(f1=[0.9658, 0.9665, 0.9664]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
         ('sklearn.ensemble', 'AdaBoostClassifier', {}),
     'ml_wf.permute': True},
    Result(output=Output(f1=[0.3276, 0.1702, 0.2091]),
           runtime=None, errored=False)),
   ({'ml_wf.clf_info':
         ('sklearn.ensemble', 'AdaBoostClassifier', {}),
     'ml_wf.permute': False},
    Result(output=Output(f1=[0.9658, 0.9333, 0.8992]),
           runtime=None, errored=False))]





Summary and Future Directions
-----------------------------



Acknowledgement
---------------
This was supported by NIH grants P41EB019936, R01EB020740.
We thank the neuroimaging community for feedback during development.
