:author: Vyas Ramasubramani
:email: vramasub@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Carl S. Adorf
:email: csadorf@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Paul M. Dodd
:email: pdodd@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Bradley D. Dice
:email: bdice@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor

:author: Sharon C. Glotzer
:email: sglotzer@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: Department of Materials Science and Engineering, University of Michigan, Ann Arbor
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: Biointerfaces Institute, University of Michigan, Ann Arbor

:bibliography: paper

-----------------------------------------------------------
signac: A Python framework for data and workflow management
-----------------------------------------------------------

.. class:: abstract

Computational research requires versatile data and workflow management tools that can easily adapt to the highly dynamic requirements of scientific investigations.
Many existing tools require strict adherence to a particular usage pattern, so researchers often use less robust ad hoc solutions that they find easier to adopt.
The resulting data fragmentation and methodological incompatibilities significantly impede research.
Our talk showcases ``signac``, an open-source Python framework that offers highly modular and scalable solutions for this problem.
The framework's powerful workflow management tools enable users to construct and automate workflows that transition seamlessly from laptops to HPC clusters.
Crucially, the underlying data model is completely independent of the workflow.
The flexible, serverless, and schema-free ``signac`` database can be introduced into other workflows with essentially no overhead and no recourse to the ``signac`` workflow model.
Additionally, the data model's simplicity makes it easy to parse the underlying data without using ``signac`` at all.
This modularity and simplicity eliminates significant barriers for consistent data management across projects, facilitating improved provenance management and data sharing with minimal overhead.

.. class:: keywords

	data management, database, data sharing, provenance, computational workflow

Introduction
------------

.. figure:: summary_figure.pdf
   :align: center
   :scale: 120 %
   :figclass: w

   The data in a ``signac`` project (A) is contained in its workspace (dark grey outline), which in turn is composed of individual data points (grey points) that exist within some multidimensional parameter space (light grey background)
   Each data point, or job, is associated with a unique hash value (e.g., 3d5) computed from its state point, the unique key identifying the job
   Using ``signac``, the data can be easily searched, filtered, grouped, and indexed
   To generate and act on this data space, ``signac`` can be used to define workflows (B), which are generically represented as a set of operations composing a directed graph
   Using a series of pre- and post-conditions defined on these operations, ``signac`` tracks the progress of this workflow on a per-job basis (C) to determine whether a particular job is complete (greyed text, green check), eligible (bold text, arrow), or blocked (normal text, universal no).
   :label:`fig:summary`

Streamlining data generation and analysis is a critical challenge for science in the age of big data and high performance computing (HPC).
Modern computational resources can generate and consume enormous quantities of data, but process automation and data management tools have lagged behind.
The highly file-based workflows characteristic of computational science are not amenable to traditional relational databases, and HPC applications require that data is available on-demand, enforcing strict performance requirements for any data storage mechanism.
Building processes acting on this data requires transparent interaction with HPC clusters without sacrificing testability on personal computers, and these processes must be sufficiently malleable to adapt to changes in scientific inquiries.

To illustrate the obstacles that must be overcome, we consider a simple example in which we study the motion of an object through a fluid medium.
If we initially model the motion only as a function of one parameter, an ad hoc solution for data storage would be to store the trajectories in paths named for the values of this parameter.
If we then introduce some post-processing step, we could run it on each of these files.
However, a problem arises if we realize that some additional parameter is also relevant.
A simple solution might be to just rename the files to account for this parameter as well, but this approach would quickly become intractable if the parameter space increased further
A more flexible traditional solution involving the use of, e.g., a relational MySQL :cite:`mysql` database might introduce undesirable setup costs and performance bottlenecks for file-based workflows on HPC.
Even if we do employ such a solution, we also have to account for our workflow process: we need a way to run analysis and post-processing on just the new data points without performing unnecessary work on the old ones.

This paper showcases the ``signac`` framework, a data and workflow management tool that aims to address these issues in a simple, powerful, and flexible manner (fig. :ref:`fig:summary`).
By storing JSON-encoded metadata and the associated data together directly on the file system, ``signac`` provides database functionality such as searching and grouping data without the overhead of maintaining a server or interfacing with external systems, and it takes advantage of the high performance file systems common to HPC.
With ``signac``, data space modifications like the one above are trivially achievable with just a few lines of Python code.
Additionally, ``signac``'s workflow component makes it just as easy to modify the process of data generation, since we simply define the post-processing as a Python function.
The workflow component of the framework, ``signac-flow``, will immediately enable the use of this calculation on the existing data space through a single command, and it tracks which tasks are completed to avoid redundancy.
The resulting data can be accessed without reference to the workflow, ensuring that it is immediately available to anyone irrespective of the tools they are using.


Overview and Examples
---------------------

.. figure:: make_data_space.pdf
   :align: center
   :scale: 100 %
   :figclass: tw

   A very simple example using ``signac`` to create the basics of a data space.
   In this example, all work is conducted inside a Jupyter notebook to indicate how easily this can be done.
   Note how fewer than ten lines of code are required to initialize a database and add data.
   :label:`fig:data`

To demonstrate how ``signac`` works, we take a simple, concrete example of the scenario described above.
Consider an experiment in which we want to find the optimal launch angle to maximize the distance traveled by a projectile through air.
Fig. :ref:`fig:data` shows how we might organize the data associated with this investigation using ``signac``.
The central object in the ``signac`` data model is the *project*, which represents all the data associated with a particular instance of a ``signac`` data space.
All of the project's data is contained within the *workspace* directory.
The workspace holds subdirectories corresponding to *jobs*, which are the individual data points in the data space.
Each job is uniquely identified by its *state point*, which is an arbitrary key-value mapping.
Although we see that these objects are stored in files and folders, we will show that these objects are structured in a way that provides layers of abstraction, making them far more useful than simple file system storage.

One could easily imagine interfacing existing scripts with this data model.
The only requirement is some concept of a unique key for all data so that it can be inserted into the database.
The unique key is what enables the creation of the 32 character hash, or *job id*, used to identify the job and its workspace folder (shown in fig. :ref:`fig:data`).
The uniqueness of this hash value is what enables ``signac``'s efficient indexing and searching functionality.
Additionally, this hash value is automatically updated to reflect any changes to individual jobs, making them highly mutable.
For example, if we instead wanted to consider how changing initial velocity affects the distance traveled for a particular angle, we can add the velocity to the existing job state points by taking advantage of the fact that the project object is an iterable:

.. code-block:: python

    for job in project:
        job.sp.v = 1

In this case, we wanted to modify the entire workspace; more generally, however, we might want to modify only some subset of jobs.
One way to accomplish this would be to apply a filter within the loop using conditionals based on the job state point, e.g. ``if job.sp.theta < 5: job.sp.v = 1``.
A more elegant solution, however, is to take advantage of ``signac``'s query API, which allows the user to find only the jobs of interest using a dictionary as a filter.
For example, in the above snippet we could replace ``for job in project`` with ``for job in project.find_jobs()``, using an arbitrary dictionary as the argument to ``find_jobs`` to filter on the state point keys.
The job finding functionality of ``signac`` is the entry point for its database functionality, enabling advanced indexing, selection, and grouping operations.

Having made the above change to our data space, we could now easily add new data points to test:

.. code-block:: python

    from numpy import linspace
    for v in [1, 2, 3]:
        for theta in np.round(linspace(0, 1.57, 5), 2):
            sp = {"v": v, "theta": theta}
            project.open_job(sp).init()

Jobs that already exist in the data space will not be overwritten by the ``init`` operation, so there is no harm in performing a loop like this multiple times.

So far, we have shown examples of working with ``signac`` both in scripts and inside Jupyter notebooks.
In fact, all of ``signac``'s core functionality is also available on the command line, making it easy to interface ``signac`` with almost any pre-existing workflow.
While these features are critical for interfacing with non-Python code bases, they are also very useful for more ad hoc analyses of ``signac`` data spaces.
For example, the searching the database on the command line can be very useful for quick inspection of data:

.. code-block:: bash

    $ # Many simple queries are automatically
    $ # translated into JSON
    $ signac find theta 0.39
    Interpreted filter arguments as '{"theta": 0.39}'.
    d3012d490304c3c1171a273a50b653ad
    1524633c646adce7579abdd9c0154d0f
    22fa30ddf3cc90b1b79d19fa7385bc95

    $ # Operators (e.g. less than) are available
    $ # using a ".-operator" syntax
    $ signac find v.\$lt": 2}}'
    d61ac71a00bf73a38434c884c0aa82c9
    00e5f0c36294f0eee4a30cabb7c6046c
    585599fe9149eed3e2dced76ef246903
    22fa30ddf3cc90b1b79d19fa7385bc95
    9fa1900a378aa05b9fd3d89f11ef0e5b

    $ # More complex queries can be constructed 
    $ # using JSON directly
    $ signac find '{"theta": {"$in": [0, 0.78]}}'
    2faf0f76bde3af984a91b5e42e0d6a0b
    585599fe9149eed3e2dced76ef246903
    03d50a048c0423bda80c9a56e939f05b
    3201fd381819dde4329d1754233f7b76
    d61ac71a00bf73a38434c884c0aa82c9
    13d54ee5821a739d50fc824214ae9a60

The query syntax is based on the MongoDB :cite:`mongodb` syntax, enabling, *e.g.*, logical or arithmetic operators.
In fact, ``signac`` databases can be easily exported to external database programs such as MongoDB, which in conjunction with the common query syntax makes switching back and forth between the two systems quite easy.

Additionally, at any point we can get an overview of what the implicit data space schema looks like:

.. code-block:: bash

    $ signac schema
    {
     'theta': 'int([3], 1), float([0.0, ..., 1.57], 5)',
     'v': 'int([1, 2, 3], 3)',
    }


Workflows
=========

While the ``signac`` database is designed to be a drop-in solution for data management issues, the ``signac`` framework was designed to simplify the entire process of data generation, which involves clearly defining the processes that generate and operate on this data cleanly and concisely.
To manage workflows, the ``signac-flow`` component of the framework provides the ``FlowProject`` class (not to be confused with the ``signac`` *Project* class that interfaces with the data in a ``signac`` project).
The FlowProject encodes operations acting on ``signac`` data spaces as well as the sequence information required to string these operations together into a complete workflow. 
In fig. :ref:`fig:ops`, we demonstrate how ``signac-flow`` can be used to automate our projectile investigation.

.. figure:: run_ops.pdf
   :align: center
   :scale: 100 %
   :figclass: w

   The ``signac-flow`` module enables the easy automation of workflows operating on ``signac`` workspaces.
   In this case, the workspace consists only of one job; the real power of the FlowProject arises from its ability to automatically handle an arbitrary sequence of operations on a large number of jobs.
   Note that in this figure we are still assuming ``v=1`` for simplicity.
   :label:`fig:ops`

In this script, we register a simple function ``calculate`` as an operation with the ``FlowProject.operation`` decorator.
We store our output in the *job document*, a lightweight JSON storage mechanism that ``signac`` provides, and we check the document to determine when the operation has been completed using the ``@FlowProject.post`` decorator.
Note that any function of a job can be used as a pre- and post-condition, but in this case our use of the job document makes the check quite simple.
Although this particular example is quite simple, ``signac-flow`` scales to arbitarily complex workflows that use pre- and post-conditions on individual operations to construct a directed graph.

By default, the ``project.py run`` interface demonstrated in fig. :ref:`fig:ops` will automatically run the entire workflow for every job in the workspace.
When conditions are defined the manner shown above, ``signac-flow`` will ensure that only incomplete tasks are run, i.e., once ``tmax`` has been calculated for a particular job, the ``calculate`` operation will not run again for that job.
Rather than running everything at once, it is also possible to exercise more fine-grained control over what operations to run using ``signac-flow``:

.. code-block:: bash

    $ # Runs all outstanding operations for all jobs
    $ python project.py run
    $ # `exec` ignores the  workflow and just runs a
    $ # specific job-operation
    $ python project.py exec -o ${OP} -j ${JOB_ID}
    $ # Run up to two operations for a specific job
    $ python project.py run -j ${JOB_ID} -n 2

A critical feature of the ``signac`` framework is its scalability to HPC.
The file-based data model is designed to leverage the high performance file systems common on such systems, and workflows designed locally are immediately executable on HPC clusters.
In particular, any operation that can be successfully executed in the manner shown in fig. :ref:`fig:ops` can also be immediately submitted to cluster schedulers.
The ``signac-flow`` package achieves this by creating cluster job scripts that perform the above operations:

.. code-block:: bash

    $ # Print the script for one 12-hour job
    $ # Additional scheduler directives are customizable
    $ python project.py submit -n 1 -w 12 --pretend
    Query scheduler...
    Submitting cluster job 'Projectiles/d61...':
     - Operation: calculate(d61...)
    #PBS -N Projectiles/d61...
    #PBS -l walltime=12:00:00
    #PBS -l nodes=1
    #PBS -V

    set -e
    set -u

    cd /path/to/project

    # Operation 'calculate' for job 'd61...':
    python project.py exec calculate d61


The workflow tracking functionality of ``signac-flow`` extends to compute clusters.
Users can always check the status of particular jobs to see how far they have progressed in the workflow, and when working on a system with a scheduler, ``signac-flow`` will also provide information about the status of jobs submitted to the scheduler.

.. code-block:: bash

    $ # Submit 3 random jobs for 12 hours
    $ python project.py submit -n 3 -w 12
    $ # Status output has options to control detail
    $ python project.py status -de
    # Overview:
    Total # of jobs: 15

    label    ratio
    -------  -------
    [no labels to show]

    # Detailed View:

    ## Labels:
    job_id                            labels
    --------------------------------  --------
    d61ac71a00bf73a38434c884c0aa82c9
    ...

    ## Operations:
    job_id   operation    eligible    cluster_status
    ------  -----------  ----------  ----------------
    d61ac7  calculate    Y           Q
    41dea8  calculate    Y           A
    585599  calculate    Y           Q
    2fc415  calculate    Y           I
    ...

Underneath each job, information is provided on a per-operation basis.
The symbols indicate the status of a particular job relative to the workflow, and the character in brackets indicates status on the cluster.
In this instance, all jobs in the projects are currently eligible for the ``calculate`` operation, three of them have been submitted to the scheduler (and are therefore marked as active).
Of these three, one has actually begun running (and is marked as ``[A]``), while the other two indicate that they are queued (the final job shown is inactive on the cluster).
Users also have the option of creating and registering arbitrary functions as *labels* to provide additional information on job status using the ``@FlowProject.label`` decorator.
When defined, these labels will populate the empty section and columns above.
They also typically provide natural conditions for operation execution
For example, we could have a simple label defined by ``def calculated(job): tmax in job.document`` to indicate that the *calculate* operation had been performed, and then we could register it as a post-condition using ``@FlowProject.post(FlowProject.calculated)``.

The quick overview of this section highlights the core features of the ``signac`` framework.
Although the example demonstrated here is quite simple, the data model scales easily to thousands of data points and far more complex and nonlinear workflows.
Demonstrations can be seen on the documentation on ReadTheDocs (`signac.readthedocs.io`), the ``signac`` website (`signac.io`), or the original paper in the Journal of Computational Materials Science :cite:`ADORF2018220`.


Design and Implementation
-------------------------

Having provided an overview of ``signac``'s functionality, we now provide a few more specifics on its implementation.
The framework prioritizes modularity and interoperability over monolithic functionality, so it is implemented in pure Python with no hard dependencies to ensure that it can be easily used with other programs.
The software runs equally well on Python 2.7 and 3.4+, and the full-featured command line interface enables its use even with non-Python tools.
To ensure that the data representation is completely independent of the workflow, the data component is developed as a standalone database solution.
This database is the primary dependency for ``signac-flow``.

The ``signac`` package is designed to be as lightweight and flexible as possible, simultaneously offering the benefits of filesystem usage and more traditional DBMS.
From the Python implementation standpoint, the central component to the ``signac`` framework is the Project class, which provides the interface to ``signac``'s data model and features.
The ``signac`` Project encapsulates much of the functionality for searching, selecting, and grouping individual data points from the data space.
Central to this process is ``signac``'s efficient on-the-fly indexing.
This process, which leverages the fact that the state point keys map uniquely to a workspace directory, uses filesystem crawlers to efficiently traverse the data space.
All functions that require indexes construct them automatically, which generally frees the user from explicit index construction.
Accessing individual data points from this index leads to the instantiation of Job objects, which are Python handles that represent individual data points.
Since these data points effectively correspond to filesystem locations, they can be mapped directly and operated on in this fashion.

The central object in the ``signac-flow`` package is the Python FlowProject class, which encapsulates a set of operations acting on a ``signac`` data space.
There is a tight relationship between the FlowProject and the underlying data space, because operations are assumed to act on a per-job basis.
Using the sequence of conditions associated with each operation, a FlowProject also tracks workflow progress on per-job basis to determine which operations to run next for a given job.
Different HPC environments and cluster schedulers are represented by separate Python classes that provide the means for querying schedulers for cluster job statuses, writing out the job scripts, and constructing the submission commands.
Job scripts are created using templates written in ``jinja2`` :cite:`jinja2`, making them easily customizable for the requirements of specific compute clusters or users.
This means that workflows designed on one cluster can be easily ported to another, and that users can easily contribute new environment configurations that can be used by others.

The extensibility of the ``signac`` framework makes it easy to build other tools on top of ``signac``.
One such tool is ``signac-dashboard``, a web interface to ``signac`` data spaces that provides an easy way to visualize ``signac`` data spaces.
The tool has been released open source and is currently under active development.


Comparisons
-----------

In recent years, many Python tools have emerged to address issues with data provenance and reproducibility in computational science.
While they are very similar to the ``signac`` framework in their goals, a major distinction between ``signac`` and other tools is that the ``signac`` data management component is independent of ``signac-flow``, making it much easier to interact with the data outside the context of the workflow.
As a result, our initial comparisons will focus on existing packages that solve the same problem as ``signac``, but generally take different and less modular approaches to doing so.

Of these tools, some of the best known are Fireworks :cite:`Fireworks`, AiiDA :cite:`Pizzi2016`, Sacred :cite:`sacred`, and Sumatra :cite:`sumatra`.
Fireworks and AiiDA are full-featured workflow managers that, like ``signac-flow``, interface with high performance compute clusters to execute complex, potentially nonlinear workflows.
These tools in fact currently offer more powerful features than ``signac-flow`` for monitoring the progress of jobs, features that are supported by the use of databases on the back end.
However, accessing data generated by these tools outside the context of the workflow definition is more challenging than it would be with ``signac`` because the data representation is closely tied to the workflows.
Additionally, the need to maintain a server for workflow management can be cumbersome and introduces additional unnecessary complexities.

Sacred and Sumatra are slightly different tools in that their primary focus is maintaining data provenance, not orchestrating complex workflows.
Superficially, the ``signac`` framework appears quite similar to Sacred.
Both use decorators to convert functions into command line executable operations, and configurations can be injected into these functions (in ``signac``'s case, using the job object).
However, the similarities largely stop there.
The focus of Sacred is to track provenance by recording information such as when an operation is executed, the configuration used, and whether any data was saved.
Therefore, in principle ``signac`` and Sacred are complementary pieces of software that could be used in concert to achieve different benefits.

However, Sacred is currently incompatible with ``signac-flow``.
Sacred and ``signac-flow`` both depend on the registration of particular functions with some internal API: in ``signac-flow``, functions are stored as operations within the FlowProject, whereas Sacred tracks functions through the *Experiment* class.
Since the actual script can only be run through one of these interfaces (whether on the command line or directly in Python), while it is possible to use ``signac``'s database facilities in concert with Sacred, running operations using ``signac-flow`` prevents tracking them using Sacred.
Conversely, the Sumatra provenance tracking tool *can* be integrated with ``signac``.
Sumatra provides a command line utility for simple usage, but it also allows direct integration into Python scripts via a Python API, and it tracks similar information to Sacred.
While the command line API is not flexible enough to allow passing arguments through to ``signac-flow``, the Python API can be easily integrated into ``signac-flow`` operations for tracking workflows managed by ``signac-flow``.


Data Management
===============

We have found fewer alternatives to direct usage of the ``signac`` data model; as mentioned previously, most currently existing software packages tightly couple their data representation with the workflow model.
The closest comparison that we have found is datreant :cite:`datreant`, which provides the means for interacting with files on the file system along with some features for finding, filtering, and grouping.
There are two primary distinctions between datreant and ``signac``: ``signac`` requires a unique key for each data point, and ``signac`` offers a tightly integrated workflow management tool.
The datreant data model is even simpler than ``signac``'s, which provides additional flexibility at the cost of ``signac``'s database functionality.
This difference is indicative of datreant's focus on more general file management problems than the issues ``signac`` is designed to solve.
The generality of the datreant data model makes integrating it into existing workflows just as easy as integrating ``signac``, and the MDSynthesis package is one example of a workflow tools built around a datreant-managed data space.
However, this tool is highly domain-specific, unlike ``signac-flow``, and it cannot be used for other types of computational investigations.
In the field of molecular simulation, the combination of MDSynthesis :cite:`mdsynthesis` and datreant is the closest analog to the ``signac`` framework, but that software does not generalize to other use-cases.


Conclusions
-----------

The ``signac`` framework provides all the tools required for thorough data, workflow, and provenance management in scientific computing investigations.
Motivated by the need for managing the dynamic, heterogeneous data spaces characteristic of computational science investigations, the tools are tailored for the use-cases most commonly faced in this field.
The framework has strived to achieve high ease of use and interoperability by emphasizing simple interfaces, minimizing external requirements, and employing open data formats like JSON.
By doing so, the framework aims to minimize the initial barriers for new users, making it easy for researchers to begin using ``signac`` with little effort.
The framework frees computational scientists from repeatedly solving common data and workflow problems throughout their research, and at a higher level, reduces the burden of sharing data and provenance tracking, both of which are critical to accelerating the production of reproducible and reusable scientific results.
