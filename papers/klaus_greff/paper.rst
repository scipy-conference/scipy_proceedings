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


We introduce Sacred, an open-source Python framework to help set up, run, and organize machine learning experiments.
It focuses on convenience of use and features a powerful configuration system that can combine entries from within the code, from config-files, and from the command-line.
Furthermore it helps to organize and keep the results reproducible by storing all relevant information in a database.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
------------

Machine learning research has become largely empirical and typically includes a large number of  computational experiments run with many different hyperparameter settings.
This process holds many practical challenges that distract from the underlying research, such as hyperparameter tuning, ensuring reproducibility, bookkeeping of the runs, maintaining an overview, and organizing the runs.
There exists many tools to tackle different aspects of this process like databases, version control systems, command-line interface generators, tools for automated hyperparameter optimization, spreadsheets, and so on.
However, there exist few tools that integrate these aspects into a unified system, so each tool has to be learned and used separately, each incurring its own overhead.
To make matters worse, experiments are often run on diverse and heterogeneous environments ranging from laptops to cloud computing nodes.
Due to deadline pressure and the inherently unpredictable nature of research there is usually little incentive for researchers to build robust infrastructure, and as a result research code often evolves quickly and important aspects like bookkeeping and reproducibility tend to fall through the cracks.


The process of ML research still involves a lot of manual work apart from the actual innovation.
This includes making the implementation configurable, tuning the hyperparamters, storing and organizing the parameters and results, and making sure they are reproducible.
For many of these problems automatic solutions exist, but integrating them with the experiment still requires a lot of manual work.
This is why under the pressure of deadlines important pillars of research like reproducibility often get neglected.
What is missing is a framework that integrates solutions to all these parts together, such that dealing with these problems becomes effortless.
This is the gap that Sacred tries to fill.


Sacred
------
Sacred[1] is an open source python framework that aims provide a unified workflow for running machine learning experiments that addresses the aforementioned challenges.
It was designed for maximum convenience while requiring only minimal boilerplate, to ensure that it remains useful even under deadline pressure.
Experiments represent the core abstraction of Sacred, and can be executed from python or through the automatically generated command-line interface.
They feature a flexible and powerful system for managing and updating configurations of hyperparameters that can be defined in native python using decorated functions, dictionaries or configuration files.
For each run, relevant information like parameters, package dependencies, host information, source code, and results are automatically captured, and are saved regularly by optional observers.
The standard observer writes to a MongoDB, but several other observers are available  for other databases, disk storage, or sending out notifications.

By combining these features into a unified but flexible workflow with minimum boilerplate, Sacred  enables its users to focus on research and still ensures that all the relevant information for each run are captured.
The standardized configuration process allows streamlined integration with other tools such as Labwatch, for hyperparameter optimization.
By storing the data in a central database comprehensive query and sorting functionality for bookeeping becomes available, thus enabling downstream analysis and allowing other tools such as Sacredboard to provide a powerful graphical user interface organizing runs and maintaining an overview.



Sacred is an open-source Python library designed for easy of use and to be minimally intrusive to research code.
The sourcecode was released under the MIT licence on Github (2015).
Its main goal is to help with all the problems of day to day machine learning research, while requiring only minimal adjustments of the code.
The core abstraction in Sacred is the Experiment which has at least a name and a main method.
Converting a python script into an experiment is very easy and involves only adding a handful of lines as demonstrated in Listing 1.
Once that is done Sacred helps to making it configurable, run it from the command-line, and keep track of all runs by storing all related information like configuration, results, the source code, dependencies, random seeds, etc. in a database.



Configuration
-------------
Each Experiment has a configuration whose entries represent its (hyper-)paramaters.
An important goal of Sacred is to make it convenient to expose these parameters, such that they can be automatically optimized and kept track of.

Setting up
++++++++++
The preferred way to set up the configuration is by decorating a function with ``@ex.config`` which adds the variables from its local scope to the configuration.
This is syntactically convenient and allows using the full expressiveness of python.
For users that prefer plain dictionaries or external configuration files, those can also easily be added to the configuration.
All the entries of the configuration are enforced to be JSON-serializable, such that they can easily be stored and queried.

Accessing
+++++++++
To make all configuration entries easily accessible, Sacred employs the mechanism of *dependency injection*.
Every captured function can simply accept any configuration entry as a parameter.
Whenever such a function is called Sacred will automatically fill in those parameters from the configuration.
The main function is automatically considered a captured function, and so is any other function decorated with ``@ex.capture``.




Running
-------
Sacred interferes as little as possible with running an ``Experiment``, thus leaving the user free to incorporate them in whatever workflow they are used to.
Each ``Experiment`` automatically comes with a command-line interface, but they can just as easily be called directly from other Python code.

The command-line interface allows changing arbitrary configuration entries, using the standard python syntax like this:

.. code-block::

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

``Experiment``s implement the observer pattern :cite:`gamma1994` by publishing all kinds of information in the form of events and allowing observers to subscribe to them.
These events are fired when a run is started, every couple of seconds while it is running and once it stops (either successfully or by failing).
Sacred ships with an observer that stores all the information about the run in a MongoDB database, but the interface also supports adding custom observers.

Collected Information
+++++++++++++++++++++
The MongoObserver collects a lot of information about the experiment and the run. Most importantly of course it will save the configuration and the result. But it will also among others save a snapshot of the source-code, a list of auto-detected package dependencies and the stdout of the experiment. Below is a summary of all the collected data:


Configuration
    configuration values used for this run
    \item[Source Code] source code of all used source files
    \item[Dependencies] version of all detected package dependencies
    \item[Host] information about the host that is running the experiment
    \item[Metadata] start and stop times, status, result or fail-trace if needed
    \item[Custom Info] a dictionary of custom information
    \item[stdout] captured console output of the run
    \item[Resources and Artifacts] extra files needed or created by the run that should be saved
\end{description}

\subsection{MongoDB}

:cite:`mongo` is a noSQL database, or more precisely a *Document Database*:
It allows the storage of arbitrary JSON documents without the need for a schema like in a SQL database.
These database entries can be queried based on their content and structure.
This flexibility makes it a good fit for Sacred, because it permits arbitrary configuration for each experiment that can still be queried and filtered later on.
In particular this feature has been very useful to perform large scale studies like the one in :cite:`greff2015`.

% \begin{listing*}
% \begin{lstlisting}[numbers=left]
% {
%   "_id" : ObjectId("5575aa4d123967150bf934c7"),
%   "config" : {
%     "C" : 1,
%     "gamma" : 0.7,
%     "seed" : 757825173},
%   "experiment" : {
%     "name" : "iris_rbf_svm",
%     "dependencies" : [["numpy", "1.9.2"],
%                       ["sacred", "0.6.3"],
%                       ["sklearn", "0.15.2"]],
%     "sources" : [["[...]/example.py", "d00e3a8be6b35960744db3794a0f2462"]]},
%   "result" : 0.9833333333333333,
%   "info" : {},
%   "captured_out" : "",
%   "status" : "COMPLETED",
%   "start_time" : ISODate("2015-06-08T16:44:29.349Z"),
%   "heartbeat" : ISODate("2015-06-08T16:44:29.359Z"),
%   "stop_time" : ISODate("2015-06-08T16:44:29.360Z"),
%   "host" : {
%     "hostname" : "Pecorino",
%     "os_info" : "Linux-3.13.0-52",
%     "cpu" : "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",
%     "cpu_count" : 8,
%     "python_version" : "3.4.0"},
%   "artifacts" : [ ],
%   "resources" : [ ] }
% \end{lstlisting}
% \vspace{-2em}
% \caption{Example MongoDB entry (slightly shortened) as saved by Sacred}
% 
% \end{listing*}


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


Related Work
------------
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


Roadmap
-------
Sacred is a framework that mainly integrates different solutions to data-science research problems.
Because of that, there are many useful ways in which it could be extended. Apart from the above mentioned interface to OpenML the following points are high up our list:

Hyperparameter optimization has become a common and very important part of machine learning research, and with the powerful configuration system of Sacred in place this an obvious next step.
So with the next release (0.7) of Sacred we plan to ease integration of tools like ``spearmint`` :cite:`snoek2012` and ``hyperopt`` :cite:`bergstra2013` into the workflow.
In the same vein it is necessary to include tools for analysing the importance of hyperparameters like the FANOVA framework of :cite:`hutter2014`.

The next important step will be to also provide a graphical interface to help inspecting and edit past and current runs.
Ideally this will take the form of a web-interface that connects directly to the database.

Another popular request is to have a bookkeeping backend that supports local storage. That could be in the form of flat files in a directory or a SQLite database. These backends are particularly easy to add so we also hope for contributions from the users for more specialized usecases.


Labwatch
--------
The correct hyperparameter setting for machine learning algorithms can often make the difference between state-of-the-art performance or random guessing.
A growing number of tools that can automate the optimization of hyperparameters have recently emerged that allow the users to, instead of manual tuning, define a searchspace and leave the search for good configurations to the optimizer.
Labwatch provides a simple way for defining searchspaces that is well integrated into the Sacred workflow, and integrates hyperparameter optimizers such as various Bayesian optimization methods (e.g RoBO[2], SMAC[3]) random search, or Bandit strategies  (Hyperband [4])
For bookkeeping it leverages the database storage of evaluated hyperparameter configurations, which allows parallel distributed optimization and also enables the use of post hoc tools for assessing hyperparameter importance (e.g Fanova [5]).

Sacredboard
-----------
Sacredboard[6] provides a convenient way for browsing runs of experiments stored in a Sacred database. In a web browser window, a list of both running and finished experiments can be viewed, together with their current state and results.
A detail view shows the hyperparameters used, information about the machine and environment where the experiment was run, and the standard output produced by the experiment.
Sacredboard comes with a lightweight web server, such that it can be easily installed as a Python package. It only requires Python and a recent web browser to run. Currently it only supports MongoDB, but in future work we will provide an interface to the various other backends that are supported by Sacred.


:cite:`hume48`


Bibliographies, citations and block quotes
------------------------------------------

If you wish to have a block quote, you can just indent the text, as in 

    When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication. :cite:`hume48`


Source code examples
--------------------

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }
 
Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


