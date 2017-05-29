:author: Klaus Greff
:email: klaus@idsia.ch
:institution: IDSIA
:institution: USI
:institution: SUPSI

:author: Aaron Klein
:email: kleinaa@cs.uni-freiburg.de
:institution: University of Freiburg

:author: Martin Chovanec
:email: chovamar@fit.cvut.cz
:institution: Czech Technical University in Prague

:author: Jürgen Schmidhuber
:email: juergen@idsia.ch
:institution: IDSIA
:institution: SUPSI

:bibliography: sacred


:video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------------------
Sacred: How I Learned to Stop Worrying and Love the Research
------------------------------------------------------------

.. class:: abstract

In this talk we will present a toolchain for conducting and organizing computational experiments consisting of Sacred -- the core framework --  and two supporting tools: Labwatch and Sacredboard.
These tools are agnostic of the methods and libraries used, and instead focus on solving the universal everyday problems of running computational experiments like reproducing results, bookkeeping, tuning hyperparameters, and organizing and analyzing the runs and results.
Attendees will be introduced to the core features of these libraries, and learn how they can form the basis for an effective and efficient workflow for machine learning research.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
============

A major part of machine learning research has become empirical and typically includes a large number of computational experiments run with many different hyperparameter settings.
This process holds many practical challenges such as flexible exposition of hyperparameters, hyperparameter tuning, ensuring reproducibility, bookkeeping of the runs, and organizing and maintaining an overview over  the results.
To make matters worse, experiments are often run on diverse and heterogeneous environments ranging from laptops to cloud computing nodes.
Due to deadline pressure and the inherently unpredictable nature of research there is usually little incentive for researchers to build robust infrastructure.
As a result, research code often evolves quickly and bad trade-offs are made that sacrifice important aspects like bookkeeping and reproducibility.


Many tools exist for tackling different aspects of this process like databases, version control systems, command-line interface generators, tools for automated hyperparameter optimization, spreadsheets, and so on.
However, very few tools even attempt to integrate these aspects into a unified system, so each tool has to be learned and used separately, each incurring its own overhead.
Since there is no common basis to build a workflow, the tools people build will be tied to their particular setup.
This impedes sharing and collaboration on a toolchain for important problems like optimizing hyperparameters, summarizing and analysing results, rerunning experiments, distributing runs.

Sacred aims to fill this gap by providing the basic infrastructure for running computational experiments.
It is our hope that it will help researchers and foster the development of a rich collaborative ecosystem of shared tools.
In the following we briefly introduce Sacred and two supporting tools:
Labwatch integrates a convenient unified interface to many automated hyperparameter optimizers such as RoBO, SMAC, or random search.
Sacredboard offers a web-based interface to view runs and supports maintaining an overview and organizing results.



Sacred
======
Sacred is an open source Python framework that aims to bundle solutions for the most frequent challenges when conducting computational experiments.
It does not enforce any particular workflow, and is independent of the choice of machine learning libraries.
Sacred was designed to remain useful even under deadline pressure, and therefore tries to
offer maximum convenience while minimizing boilerplate code.


.. By combining these features into a unified but flexible workflow with minimum boilerplate, Sacred  enables its users to focus on research and still ensures that all the relevant information for each run are captured.
   The standardized configuration process allows streamlined integration with other tools such as Labwatch, for hyperparameter optimization.
   By storing the data in a central database comprehensive query and sorting functionality for bookeeping becomes available, thus enabling downstream analysis and allowing other tools such as Sacredboard to provide a powerful graphical user interface organizing runs and maintaining an overview.


Overview
--------
To adopt Sacred all that is required is to instantiate an ``Experiment`` and to decorate a main function that serves as entry-point:

.. code-block:: python

    from sacred import Experiment
    ex = Experiment()
    ...
    @ex.automain
    def main():
        ...

.. The Experiment class represent the core abstraction of Sacred

Hyperparameters can then be defined in native Python using special decorated functions, dictionaries or configuration files (see :ref:`configuration`).
The experiment can be run through an automatically generated command-line interface, or from Python by calling ``ex.run()``.
Both modes offer the same ways for passing options, setting parameters, and adding observers.
Sacred then 1) interprets the options 2) evaluates the parameter configuration 3) gathers information about dependencies and host, and 4) constructs and calls a ``Run`` object that is responsible for executing the main function.
The Run captures the stdout, custom information and fires events to the observers at regular intervals for bookkeeping (see :ref:`bookkeeping`).
For each run, relevant information like parameters, package dependencies, host information, source code, and results are automatically captured, and are saved regularly by optional observers.
Several observers are available for databases, disk storage, or sending out notifications.



Configuration
-------------
An important goal of Sacred is to make it convenient to define, expose and use hyperparameters, which we will call the configuration of the experiment.

Defining a Configuration
++++++++++++++++++++++++
The main way to set up the configuration is through so called ConfigScopes.
This means decorating a function with ``@ex.config`` which Sacred executes and adds to its local variables the configuration:

.. code-block:: python

    @ex.config
    def cfg():
        variant = 'simple'
        learning_rate = 0.1
        filename = 'stuff_{}.log'.format(a)

This is syntactically convenient and allows using the full expressiveness of Python, which includes calling functions and variables that depend on others.
For users that instead prefer plain dictionaries or external configuration files, those can also be used.
All the entries of the configuration are enforced to be JSON-serializable, such that they can easily be stored and queried.

Using Config Values
+++++++++++++++++++
To make all configuration entries easily accessible, Sacred employs the mechanism of *dependency injection*.
That means, any function decorated by ``@ex.capture`` simply accept any configuration entry as a parameter.
Whenever such a function is called Sacred will automatically fill in those parameters from the configuration.
This allows for flexible and convenient use of the hyperparameters everywhere:

.. code-block:: python

    @ex.capture
    def do_stuff(variant, learning_rate=1.0):
        ...

    ...

    do_stuff()  # parameters are automatically filled
    do_stuff('debug')  # manually set

Injection follows the priority: 1. explicitly passed arguments, 2. config values, 3. default values.

.. Main function and commands are automatically captured

Updating Parameters
+++++++++++++++++++
Configuration values can be set (overridden) externally when running an experiment.
This can happen both from the commandline

.. code-block:: bash

    >> python my_experiment.py with variant='complex'

or from Python calls:

.. code-block:: python

    from my_experiment import ex
    ex.run(config_updates={'variant': 'complex'})

Sacred treats these values as fixed and given when executing the ConfigScopes.
In this way they influence dependent values as you would expect (so here: ``filename='stuff_complex'``).

Sometimes a particular set of settings belongs together and should be saved.
To collect them sacred offers the concept of named configs.
They are defined similar to configurations using ``@ex.named_config``, dictionaries, or from config files.
They can be added en-block from the commandline and from Python, and are treated as a set of updates.

.. example ??



Bookkeeping
-----------

Bookkeeping in Sacred is accomplished by implementing the observer pattern :cite:`gamma1994`:
The experiment publishes all kinds of information in the form of events that zero or more observers can subscribe to.
Observers can be added dynamically from the commandline or directly in code:

.. code-block:: python

    from sacred.observers import MongoObserver
    ex.observers.append(MongoObserver.create("DBNAME"))

Collected Information
+++++++++++++++++++++
Events are fired when a run is started, every couple of seconds while it is running (heartbeat), and once it stops, (either successfully or by failing).
This way information is available already during runtime, and partial data is captured even in case of failures. 

Sacred collects a lot of information about the experiment and the run. 
Most importantly of course it will save the configuration and the result. 
But it will also among others save a snapshot of the source-code, a list of auto-detected package dependencies and the stdout of the experiment. 
Below is a summary of all the collected data:


Configuration
    configuration values used for this run
Source Code
    source code of all detected source files
Dependencies
    version numbers for all detected package dependencies
Host
    information about the host that is running the experiment including CPU, OS, and Python version. Optionally also other informatino like GPU or environment variables.
Metadata
    start and stop times, current status, result, and fail-trace (if needed)
Live Information
     Including captured stdout, extra files needed or created by the run that should be saved, custom information, and custom metrics about the experiment.


Observers
+++++++++

Sacred ships with observers that stores all the information from these events in a MongoDB, SQL database, or locally on disk.
Furthermore ther are two observers that can send notifications about runs via Telegram or Slack.
However, the observer interface is generic and supports easy addition of custom observers.

The recommended observer is the ``MongoObserver`` that writes to a MongoDB :cite:`mongo`.
MongoDB is a noSQL database, or more precisely a *Document Database*:
It allows the storage of arbitrary JSON documents without the need for a schema like in a SQL database.
These database entries can be queried based on their content and structure.
This flexibility makes it a good fit for Sacred, because it permits arbitrary configuration for each experiment that can still be queried and filtered later on.
In particular this feature has been very useful to perform large scale studies like the one in :cite:`greff2015`.


Reproducibility
---------------
An important goal of Sacred is to collect all the necessary information to make computational experiments reproducible.
The result of such an experiment depends on several factors including: the source code, versions of the used packages, the host system, resources, and (pseudo-)randomness.
To ensure reproducibility Sacred attempts to automatically collect as much data about these factors as possible.

Dependencies
++++++++++++
When an experiment is started Sacred uses Python inspection to detect imported packages and determines their version-numbers.
This detection will catch all dependencies that are imported from the main file before the experiment was started.
This might miss certain nested imports, but further dependencies can easily be added manually 

To ensure that it features a simple integrated version control system that guarantees that for each run all the required files are stored in the same database.
Sacred actually also saves the contents of that file in a separate collection.
The same mechanism can also be used to save additional resources or files created by the run (called artifacts).

There is one major obstacle of reproducibility left: randomness.
Randomization is an important part of many machine learning algorithms, but it inherently conflicts with the goal of reproducibility.
The solution of course is to use pseudo random number generators (PRNG) that take a seed and generate seemingly random numbers from that in a deterministic fashion.
But this is only effective if the seed of the PRNG is not manually set and kept track of.
Also if the seed is set to a fixed value as part of the code, then all runs will share the same randomness, which can be an undesired effect.

Sacred solves these problems by always generating a seed for each experiment that is stored as part of the configuration.
It can be accessed from the code in the same way as every other config entry, but Sacred can also automatically generate seeds and PRNGs that deterministically depend on that root seed for you.
Furthermore, Sacred automatically seeds the global PRNGs of the ``random`` and ``numpy`` modules, thus making most applications of randomization reproducible without any intervention of the user.


Labwatch
========

Finding the correct hyperparameter setting for machine learning algorithms is often done by trial and error even though it sometimes makes the difference between state-of-the-art performance or performance that is as good as random guessing.
A growing number of tools that can automate the optimization of hyperparameters have recently emerged, allowing users, instead of manual tuning, to define a searchspace and leave the search for good configurations to the optimizer.
However, in practice each optimizer often requires users to adapt their code to a certain interface.
Labwatch supports a unified interface through Sacred to a variety of hyperparameter optimizers that allows for an easy integration of hyperparameter optimization into the daily workflow.


LabAssistant
------------

At the heart of Labwatch is the so called LabAssistant, which connects the Sacred experiment with a hyperparameter configuration search space, simply dubbed searchspace and a hyperparameter optimizer through a database.

.. code-block:: python

    from sacred import Experiment
    from labwatch.assistant import LabAssistant
    from labwatch.optimizers import RandomSearch
    
    ex = Experiment()       
    a = LabAssistant(experiment=ex,
                     database_name="labwatch",
                     optimizer=RandomSearch)


.. Labwatch provides a simple way for defining searchspaces that is well integrated into the Sacred workflow, and integrates hyperparameter optimizers such as various Bayesian optimization methods (e.g `RoBO <https://github.com/automl/RoBO/>`_ , `SMAC <https://github.com/automl/SMAC3/>`_) random search, or bandit strategies  (Hyperband [4])

If the experiment is now called with a searchspace rather than a configuration, Labwatch will pass all entries of this experiment in the database to the hyperparameter optimizer and let it suggest a configuration. This configuration is then used to run the experiment.

 
For bookkeeping it leverages the database storage of evaluated hyperparameter configurations, which allows parallel distributed optimization and also enables the use of post hoc tools for assessing hyperparameter importance (e.g Fanova :cite:`hutter-icml14a`).



Search Spaces
-------------

In general Labwatch distinguishes between *categorical* hyperparameters that can have only discret choices, and *numerical* hyperparameters that can have either integer or float values.
For each hyperparameter the search space defines a prior distribution (e.g. uniform or Gaussian) as well as its type and its scale (e.g. log scale, linear scale) and a default value.

Search spaces follow the same interface as Sacred's named configs:

.. code-block:: python

    @ex.config
    def cfg():
        batch_size = 128
        learning_rate = 0.001

    @a.searchspace
    def search_space():
        learning_rate = UniformFloat(lower=10e-3,
                                     upper=10e-2,
                                     default=10e-2,
                                     log_scale=True)
        batch_size = UniformNumber(lower=32,
                                   upper=64,
                                   default=32,
                                   type=int,
                                   log_scale=True)

Now by executing the Experiment for instance through the command line:

.. code-block:: bash

    >> python my_experiment.py with search_space

Labwatch triggers the optimizer to suggest a new configuration based on all configurations that are stored in the database and have been drawn from the same search space.

Every hyperparameter optimization method, such as Bayesian optimization or random search, often needs to evaluate some configuration before it approaches a good region in the search space.
This means that Labwatch needs to run the same experiment multiple times.
Labwatch's Labassitant allows to easily do this from Python via:

.. code-block:: python

    a.run_suggestion(100)

This runs the same experiment 100 times with different hyperparameter configurations and saves all results to a database.



Multiple search spaces
++++++++++++++++++++++

Since search spaces are named configurations, Labwatch also allows to have multiple search spaces, which is very convenient if one wants to keep single hyperparameters fixed and only optimize a few other hyperparameters.
Assume that we now only want to optimize the learning rate and keep the batch size fixed, we can create a second smaller search space:

.. code-block:: python

    @a.searchspace
    def small_search_space():
        learning_rate = UniformFloat(lower=10e-3,
                                     upper=10e-2,
                                     default=10e-2,
                                     log_scale=True)

We can run our experiment now in the same way but calling it with this new search space: 

.. code-block:: bash

    >> python my_experiment.py with small_search_space


Labwatch passes only entries of the database from the same search space to the optimizer in order to avoid inconsistencies. The optimizer will now only suggest a value for the learning rate. 
All other hyperparameters, such as the batch size, are set to the values that are defined in the config.



Hyperparameter Optimizers
-------------------------


Labwatch offers a simple but also flexible interface to a variety of state-of-the-art hyperparameter optimization methods.
Even though the interface for all optimizer is the same, every optimizer has its own properties and might not work in all use cases.
The following list gives a brief overview of optimizers that can be used with Labwatch and in which
setting they work and which they do not. For more details we refer to the corresponding papers:

- **Random search** is probably the simplest hyperparameter optimization method :cite:`bergstra-jmlr12a`. It just samples hyperparameter
  configurations randomly from the corresponding prior distributions. Due to its simplicity, random search works in discrete as well as continuous search
  spaces and can be easily run in parallel.

- **Bayesian optimization**  fits a probabilistic model to capture the current believe of the objective function :cite:`shahriari-ieee16a, snoek-nips12a`.
  To select a new configuration, it uses a utility function that only depends on the
  probabilistic model to trade off exploration and exploitation. Here we use a Gaussian process to model our objective
  function, which works well in low (<10) dimensional continuous search spaces but does not work with categorical
  hyperparameters.

- **SMAC** is also a Bayesian optimization method but uses random forest instead of Gaussian processes to model
  the objective function :cite:`hutter-lion11a`. That allows it to work in high dimensional mixed continuous and discret input spaces but will
  be probably outperformed by Gaussian process based Bayesian optimization in low dimensional continuous search spaces :cite:`eggensperger-bayesopt13`.



Sacredboard
===========
Sacredboard provides a convenient way for browsing runs of experiments stored in a Sacred database.
It comes as a Python package that connects to the database and
In a web browser window, a list of both running and finished experiments can be viewed, together with their current state and results.
A detail view shows the hyperparameters used, information about the machine and environment where the experiment was run,
and the standard output produced by the experiment.
Sacredboard comes with a lightweight web server, such that it can be easily installed as a Python package.
It only requires Python and a recent web browser to run. Currently it only supports MongoDB,
but in future work we will provide an interface to the various other backends that are supported by Sacred.

.. figure:: sacredboard.png
   :scale: 35 %
   :alt: map to buried treasure




Example
=======

.. code-block:: python

    import tensorflow as tf
    import sacred
    from model import Model
    from tensorflow.examples.tutorials.mnist\
        import input_data


    ex = sacred.Experiment("MNIST")

    @ex.config
    def config():
        steps = 500
        learning_rate = 0.5
        minibatch_size = 100
        log_dir = "./log/default"


    @ex.automain
    @sacred.stflow.LogFileWriter(ex)
    def experiment(_run, steps, learning_rate,
                    minibatch_size, log_dir):
        mnist = input_data.read_data_sets("MNIST_data/",
                                          one_hot=True)
        sess = tf.InteractiveSession()
        nn_model = Model(learning_rate, mnist, sess)
        summary_writer = tf.summary.FileWriter(log_dir)
        test_summary = tf.summary.merge(
                        [nn_model.test_sum_cross_entropy,
                        nn_model.test_sum_acc])
        for _ in range(steps):
            nn_model.train(minibatch_size)
            # evaluate on test
            summary, val_crentr, val_acc = \
                sess.run((test_summary,
                          nn_model.cross_entropy,
                          nn_model.accuracy),
                         feed_dict=
                         {nn_model.x: mnist.test.images,
                          nn_model.y_: mnist.test.labels})
            summary_writer.add_summary(summary, steps)
            _run.log_scalar("test.cross_entropy",
                            float(val_crentr))
            # We can also specify the step number directly
            _run.log_scalar("test.accuracy",
                            float(val_acc), steps)

        return float(val_acc)

Related Work
============
There are only a few projects that we are aware of that have a focus similar to Sacred with the closest one being Sumatra :cite:`davison2012`.
It comes as a command-line tool that can operate also with non-Python experiments, and helps to do all the bookkeeping.
Under the hood it uses a SQL database to store all the runs and comes with a versatile web-interface to view and edit the stored information.
The main drawback of Sumatra, and indeed the main reason why we opted for our own library, is its workflow.
It requires initializing a project directory, the parameters need to be in a separate file and the experiment must be an executable that takes the name of a config-file as a command-line parameter.

The CDE project :cite:`guo2012` takes a completely different and much more general approach to facilitate reproducible research.
It uses the linux kernel to track *all* files, including data, programs and libraries that were used for an experiment.
These files are then bundled together and because it also includes system libraries, the resulting package can be run on virtually any other linux machine.
It does not help organization or bookkeeping, but, given that the user takes care of parameters and randomness, provides a very thorough solution to the problem of reproducibility.

:cite:`jobman` is a Python library that grew out of the need for scheduling lots of machine learning experiments.
It helps with organizing hyperparameter searches and as a side-effect it also keeps track of hyperparameters and results.
It requires the experiment to take the form a Python function with a certain signature.

Experiment databases :cite:`vanschoren2012, smith2014` make an effort to unify the storage of machine learning problems and experiments by expressing them in a common language.
By standardizing that language, they improve comparability and communicability of the results.
The most wellknown example of might be the OpenML project \cite{vanschoren2014}.
Expressing experiments in a common language implies certain restrictions on the performed experiments.
For this reason we chose not to build Sacred ontop of an experiment database, to keep it applicable to as many usecases as possible.
That being said, we believe there is a lot of value in adding (optional) interfaces to experiment databases to Sacred.


Conclusion
==========
fobor

Future Work
===========
Sacred is a framework that mainly integrates different solutions to data-science research problems.
Because of that, there are many useful ways in which it could be extended. Apart from the above mentioned interface to OpenML, the following points are high up our list:

Hyperparameter optimization has become a common and very important part of machine learning research, and with the powerful configuration system of Sacred in place, this an obvious next step.
So with the next release (0.7) of Sacred, we plan to ease integration of tools like ``spearmint`` :cite:`snoek2012` and ``hyperopt`` :cite:`bergstra2013` into the workflow.
In the same vein it is necessary to include tools for analysing the importance of hyperparameters like the FANOVA framework of :cite:`hutter2014`.

The next important step will also be to provide a graphical interface to help inspecting and editing past and current runs.
Ideally this will take the form of a web-interface that connects directly to the database.

Another popular request is to have a bookkeeping backend that supports local storage. That could be in the form of flat files in a directory or a SQLite database. These backends are particularly easy to add so we also hope for contributions from the users for more specialized usecases.



Acknowledgements
================
fobof




.. Customised LaTeX packages
.. -------------------------

.. latex::
   :usepackage: microtype
