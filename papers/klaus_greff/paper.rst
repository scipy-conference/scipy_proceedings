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

In this talk we’ll present a toolchain for conducting and organizing computational experiments consisting of Sacred -- the core framework --  and two supporting tools: Labwatch and Sacredboard.
These tools are agnostic of the methods and libraries used, and instead focus on solving the universal everyday problems of running computational experiments like reproducing results, bookkeeping, tuning hyperparameters, and organizing and analyising the runs and results.
Attendees will be introduced to the core features of these libraries, and learn how they can form the basis for an effective and efficient workflow for machine learning research.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
============

A major part of machine learning research has become empirical and typically includes a large number of  computational experiments run with many different hyperparameter settings.
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
Labwatch integrates a convenient unified interface to many automated hyperparameter optimizers like robo, pysmack, or hyperopt.
Sacredboard offers a web-based interface to view runs and supports maintaining an overview and organizing results.



Sacred
======
Sacred is an open source python framework that aims to bundle solutions for the most frequent challenges when conducting computational experiments.
It doesn't enforce any particular workflow, and is independent of the choice of machine learning libraries.
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

Hyperparameters can then be defined in native python using special decorated functions, dictionaries or configuration files (see :ref:`configuration`).
The experiment can be run through an automatically generated command-line interface, or from python by calling ``ex.run()``.
Both modes offer the same ways for passing options, setting parameters, and adding observers.
Sacred then 1) interprets the options 2) evaluates the parameter configuration 3) gathers information about dependencies and host, and 4) constructs and calls a ``Run`` object that is responsible for executing the main function.
The Run captures the stdout, custom information and fires events to the observers at regular intervals for bookkeeping (see :ref:`bookkeeping`).
For each run, relevant information like parameters, package dependencies, host information, source code, and results are automatically captured, and are saved regularly by optional observers.
Several observers are available for databases, disk storage, or sending out notifications.



Configuration
-------------
An important goal of Sacred is to make it convenient to define, expose and use hyperparameters, which we'll call the configuration of the experiment.

Defining a Configuration
++++++++++++++++++++++++
The main way to set up the configuration is through so called ConfigScopes.
This means decorating a function with ``@ex.config`` which sacred executes and adds its local variables the configuration:

.. code-block:: python

    @ex.config
    def cfg():
        variant = 'simple'
        learning_rate = 0.1
        filename = 'stuff_{}.log'.format(a)

This is syntactically convenient and allows using the full expressiveness of python, which includes calling functions and variables that depend on others.
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

Injection follows the priority 1. explicitly passed arguments 2. config values 3. default values.
.. Main function and commands are automatically captured

Updating Parameters
+++++++++++++++++++
Configuration values can be set (overridden) externally when running an experiment.
This can happen both from the commandline

.. code-block:: bash

    >> python my_experiment.py with variant='complex'

or from python calls:

.. code-block:: python

    from my_experiment import ex
    ex.run(config_updates={'variant': 'complex'})

Sacred treats these values as fixed and given when executing the ConfigScopes.
In this way they influence dependent values as you would expect (so here: ``filename='stuff_complex'``).

Sometimes a particular set of settings belongs together and should be saved.
To collect them sacred offers the concept of named configs.
They are defined similar to configurations using ``@ex.named_config``, dictionaries, or from config files.
They can be added en-block from the commandline and from python, and are treated as a set of updates.

.. example ??


Running
-------
Sacred interferes as little as possible with running an ``Experiment``, thus leaving the user free to incorporate them in whatever workflow they are used to.
Each ``Experiment`` automatically comes with a command-line interface, but they can just as easily be called directly from other Python code.

The command-line interface allows changing arbitrary configuration entries, using the standard python syntax like this:

.. code-block:: python

    > python example.py run with C=3.4e-2 gamma=0.5


Apart from running the main function (here: ``run``) it also provides commands to inspect the configuration (``print\_config``) and to display the dependencies (``print\_dependencies``).
It also provides flags to get help, control the log-level, add a MongoDB observer, and for debugging. The command-line interface also allows adding custom commands, by just decorating a function with ``@ex.command``.

All of the above can just as easily accomplished directly from python:

.. code-block:: python

    from example import ex
    # runs the default configuration
    r = ex.run()
    # run with updated configuration
    r = ex.run(config_updates={'C': 3.4e2, 'gamma': 0.5})
    # run the print_config command
    r = ex.run_command('print_config', config_updates={'gamma': 7})

After each of these calls ``r`` will contain a ``Run`` object with all kinds of details about the run including the result and the (modified) configuration.



Bookkeeping
-----------

``Experiment`` s implement the observer pattern :cite:`gamma1994` by publishing all kinds of information in the form of events and allowing observers to subscribe to them.
These events are fired when a run is started, every couple of seconds while it is running and once it stops (either successfully or by failing).
Sacred ships with an observer that stores all the information about the run in a MongoDB database, but the interface also supports adding custom observers.

Collected Information
+++++++++++++++++++++
The MongoObserver collects a lot of information about the experiment and the run. Most importantly of course it will save the configuration and the result. But it will also among others save a snapshot of the source-code, a list of auto-detected package dependencies and the stdout of the experiment. Below is a summary of all the collected data:


Configuration
    configuration values used for this run
Source Code
    source code of all used source files
Dependencies
    version of all detected package dependencies
Host
    information about the host that is running the experiment
Metadata
    start and stop times, status, result or fail-trace if needed
Custom info
    a dictionary of custom information
stdout
    captured console output of the run
Resources and Artifacts
     extra files needed or created by the run that should be saved


MongoDB
+++++++

:cite:`mongo` is a noSQL database, or more precisely a *Document Database*:
It allows the storage of arbitrary JSON documents without the need for a schema like in a SQL database.
These database entries can be queried based on their content and structure.
This flexibility makes it a good fit for Sacred, because it permits arbitrary configuration for each experiment that can still be queried and filtered later on.
In particular this feature has been very useful to perform large scale studies like the one in :cite:`greff2015`.


Reproducibility
---------------
Maybe the most important goal of Sacred is to collect all the necessary information to make all the runs reproducible.
To ensure that it features a simple integrated version control system that guarantees that for each run all the required files are stored in the same database.
Notice that the database entry in autoref{lst:mongo} contains the name and MD5 hash of the ``example.py`` file (line 12).
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
The correct hyperparameter setting for machine learning algorithms can often make the difference between state-of-the-art performance or random guessing.
A growing number of tools that can automate the optimization of hyperparameters have recently emerged that allow the users to, instead of manual tuning, define a searchspace and leave the search for good configurations to the optimizer.
Labwatch provides a simple way for defining searchspaces that is well integrated into the Sacred workflow, and integrates hyperparameter optimizers such as various Bayesian optimization methods (e.g RoBO[2], SMAC[3]) random search, or Bandit strategies  (Hyperband [4])
For bookkeeping it leverages the database storage of evaluated hyperparameter configurations, which allows parallel distributed optimization and also enables the use of post hoc tools for assessing hyperparameter importance (e.g Fanova [5]).


Sacredboard
===========
Sacredboard[6] provides a convenient way for browsing runs of experiments stored in a Sacred database. In a web browser window, a list of both running and finished experiments can be viewed, together with their current state and results.
A detail view shows the hyperparameters used, information about the machine and environment where the experiment was run, and the standard output produced by the experiment.
Sacredboard comes with a lightweight web server, such that it can be easily installed as a Python package. It only requires Python and a recent web browser to run. Currently it only supports MongoDB, but in future work we will provide an interface to the various other backends that are supported by Sacred.



Example
=======

fobor



Related Work
============
There are only a few projects that we are aware of that have a focus similar to Sacred with the closest one being Sumatra :cite:`davison2012`.
It comes as a command-line tool that can operate also with non-python experiments, and helps to do all the bookkeeping.
Under the hood it uses a SQL database to store all the runs and comes with a versatile web-interface to view and edit the stored information.
The main drawback of Sumatra, and indeed the main reason why we opted for our own library is its workflow.
It requires initializing a project directory, the parameters need to be in a separate file and the experiment must be an executable that takes the name of a config-file as a command-line parameter.

The CDE project :cite:`guo2012` takes a completely different and much more general approach to facilitate reproducible research.
It uses the linux kernel to track *all* files, including data, programs and libraries that were used for an experiment.
These files are then bundled together and because it also includes system libraries the resulting package can be run on virtually any other linux machine.
It doesn't help organization or bookkeeping, but, given that the user takes care of parameters and randomness, provides a very thorough solution to the problem of reproducibility.

:cite:`jobman` is a python library that grew out of the need for scheduling lots of machine learning experiments.
It helps with organizing hyperparameter searches and as a side-effect it also keeps track of hyperparameters and results.
It requires the experiment to take the form a python function with a certain signature.

Experiment databases :cite:`vanschoren2012, smith2014` make an effort to unify the storage of machine learning problems and experiments by expressing them in a common language.
By standardizing that language they improve comparability and communicability of the results.
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
Because of that, there are many useful ways in which it could be extended. Apart from the above mentioned interface to OpenML the following points are high up our list:

Hyperparameter optimization has become a common and very important part of machine learning research, and with the powerful configuration system of Sacred in place this an obvious next step.
So with the next release (0.7) of Sacred we plan to ease integration of tools like ``spearmint`` :cite:`snoek2012` and ``hyperopt`` :cite:`bergstra2013` into the workflow.
In the same vein it is necessary to include tools for analysing the importance of hyperparameters like the FANOVA framework of :cite:`hutter2014`.

The next important step will be to also provide a graphical interface to help inspecting and edit past and current runs.
Ideally this will take the form of a web-interface that connects directly to the database.

Another popular request is to have a bookkeeping backend that supports local storage. That could be in the form of flat files in a directory or a SQLite database. These backends are particularly easy to add so we also hope for contributions from the users for more specialized usecases.



Acknowledgements
================
fobof




.. Customised LaTeX packages
.. -------------------------

.. latex::
   :usepackage: microtype