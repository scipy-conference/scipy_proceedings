..   .. latex::
..      :usepackage: footmisc

:author: Bradley D. Dice
:email: bdice@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: These authors have an equal contribution to the work presented.

:author: Brandon L. Butler
:email: butlerbr@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: These authors have an equal contribution to the work presented.

:author: Vyas Ramasubramani
:email: vramasub@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Alyssa Travitz
:email: atravitz@umich.edu
:institution: Macromolecular Science and Engineering Program, University of Michigan, Ann Arbor

:author: Michael M. Henry
:email: mikehenry@u.boisestate.edu
:institution: Micron School of Materials Science and Engineering, Boise State University

:author: Hardik Ojha
:email: hojha@ch.iitr.ac.in
:institution: Department of Chemical Engineering, Indian Institute of Technology Roorkee

:author: Kelly Wang
:email: kelwang@umich.edu
:institution: Macromolecular Science and Engineering Program, University of Michigan, Ann Arbor

:author: Carl S. Adorf
:email: csadorf@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Eric Jankowski
:email: ericjankowski@boisestate.edu
:institution: Micron School of Materials Science and Engineering, Boise State University

:author: Sharon C. Glotzer
:email: sglotzer@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: Biointerfaces Institute, University of Michigan, Ann Arbor

:bibliography: paper

-------------------------------------------------------------------
signac: Data Management and Workflows for Computational Researchers
-------------------------------------------------------------------

.. class:: abstract

The **signac** data management framework (https://signac.io) helps researchers execute reproducible computational studies, scales workflows from laptops to supercomputers, and emphasizes portability and fast prototyping.
With **signac**, users can track, search, and archive data and metadata for file-based workflows and automate workflow submission on high performance computing (HPC) clusters.
We will discuss recent improvements to the software's feature set, scalability, scientific applications, usability, and community.
Newly implemented synced data structures, features for generalized workflow execution, and performance optimizations will be covered, as well as recent research using the framework and changes to the project's outreach and governance as a response to its growth.

.. class:: keywords

   data management, data science, database, simulation, collaboration, workflow, HPC, reproducibility


Introduction
------------

.. figure:: signac_overview.pdf
    :align: center
    :scale: 40 %
    :figclass: w

    Overview of the **signac** framework.
    Users first create a project, which initializes a workspace directory on disk.
    Users define state points which are dictionaries that uniquely identify a job.
    The workspace holds a directory for each job, containing JSON files that store the state point and job document.
    The job directory name is a hash of the state point's contents.
    Here, the ``init.py`` file initializes an empty project and adds one job with state point ``{"a": 1}``.
    Next, users define a workflow using a subclass of **signac-flow**'s ``FlowProject``.
    The workflow shown has three operations (simulate, analyze, visualize) that, when executed, produce two new files ``results.txt`` and ``plot.png`` in the job directory. :label:`overview`

Working examples of features presented in this paper can be found online at [#]_.

.. [#] https://github.com/glotzerlab/signac-examples

Research projects often address problems where questions change rapidly, data models are always in flux, and compute infrastructure varies widely from project to project.
The **signac** data management framework is a tool designed by researchers, for researchers, to simplify the process of prototyping and performing reproducible scientific computations.
It forgoes serializing complex data files into a database in favor of working with these files directly, providing fast indexing utilities for a set of directories.
Using **signac**, a data space on the file system can be initialized, searched, and modified using either a Python or command-line interface.
The companion package **signac-flow** interacts with the data space to generate and analyze data through reproducible workflows that scale from laptops to supercomputers.
Arbitrary shell commands can be run by **signac-flow** as part of a workflow, making it as flexible as a script in any language of choice.

With **signac**, file-based data and metadata are organized in folders and JSON files, respectively (see Figure :ref:`overview`).
A **signac** data space, or "workspace," is composed of jobs, which are individual directories associated with a single primary key known as a "state point" stored in a file ``signac_statepoint.json`` in that directory.
These files allow **signac** to index the data space, providing a database-like interface to a collection of directories.
Arbitrary user data may be stored in user-created files in these jobs, although **signac** also provides convenient facilities for storing simple lightweight data or array-like data via JSON and HDF5 utilities.
Readers seeking more details about **signac** may refer to past **signac** papers: :cite:`signac_commat, signac_scipy_2018` as well as the **signac** website [#]_ and documentation [#]_.

.. [#] https://signac.io
.. [#] https://docs.signac.io

This filesystem-based approach has both advantages and disadvantages.
Its key advantages lie in flexibility and portability.
The serverless design removes the need for any external running server process, making it easy to operate on any filesystem.
The design is also intrinsically distributed, making it well suited for highly parallel workflows where multiple processes concurrently read or write file-based data stored in job directories.
Conversely, this distributed approach precludes the performance advantages of centralized data stores with persistent indexes in memory.
Typically, the **signac** approach works very well for projects up to 100,000 jobs, while significantly larger projects may have wait times that constrain interactive usage.
These limits are inherent to **signac**'s use of small files for each job's state point, but the framework has been aggressively optimized and uses extensive caching/buffering to maximize the achievable throughput within this model.

The framework is a strong choice for applications meeting one or more of the following criteria:

- input/output data is primarily file-based
- prototype research code where data schemas may change or evolve
- computations will use an HPC cluster
- the amount of computation per job is large
- parameter sweeps over a range of values (with values on a grid or dynamically determined by e.g. active learning)
- heterogeneous data (not all jobs have the same keys present in their state points)

For example, M. W. Thompson *et al.* in :cite:`thompson.etal2019a` used 396 jobs/state points to execute computer simulations of room-temperature ionic liquids with GROMACS :cite:`pronk.etal2013,lindahl.etal2001,hess.etal2008,abraham.etal2015a` simulations.
The study investigated 18 compositions (by mass fraction) and 22 unique solvents from five chemical families (nitriles, alcohols, halocarbons, carbonyls, and glymes), with a state point for each pairing of mass fraction and solvent type.

Users working with large tabular data (e.g. flat files on disk or data from a SQL database) may prefer to use libraries like pandas, dask, or RAPIDS that are specifically designed for those use cases.
However, it is possible to create a **signac** project with state points corresponding to each row, which may be a good use of **signac** if there is file-based data affiliated with each row's parameters.

This paper will focus on developments to the **signac** framework over the last 3 years, during which features, flexibility, usability, and performance have been greatly improved.
The core data structures in **signac** have been overhauled to provide a powerful and generic implementation of "synced collections," that we will leverage in future versions of **signac** to enable more performant data indexing and flexible data layouts.
In **signac-flow**, we have added support for submitting groups of operations with conditional dependencies, allowing for more efficient utilization of large HPC resources.
Further developments allow for operations to act on arbitrary subsets of the data space, or "aggregates," rather than single jobs alone.
Moving beyond code development, this paper will also discuss the scientific work these features have enabled and the organizational developments from key partnerships and affiliations with scientific software initiatives and organizations such as MoSDeF and NumFOCUS.
We will share our project's experience in progressively revising project governance to catalyze sustained contributions of many kinds, while adding more points of entry for learning about the project (Slack support, weekly public office hours), and participating in Google Summer of Code in 2020 as a NumFOCUS Affiliated Project.

Applications of signac
----------------------

The **signac** framework has been cited 51 times, according to Google Scholar, and has been used in a range of scientific fields with various types of computational workflows.
Some of these studies include quantum calculations of small molecules :cite:`govoni.etal2018`,
4,480 simulations of epoxy curing (each containing millions of particles) :cite:`thomas.etal2018`,
inverse design of pair potentials :cite:`adorf.etal2018`,
identifying photonic band gaps in 151,593 crystal structures :cite:`cersonsky.etal2021`,
benchmarking atom-density representations for use in machine learning :cite:`musil.etal2021`,
simulating fluid flow in polymer solutions :cite:`howard.etal2019`,
design of optical metamaterials :cite:`harper.etal2020`,
and economic analysis of drought risk in agriculture :cite:`rodziewicz.etal2020`.
Much of the published research using **signac** comes from chemical engineering, materials science, or physics, the home fields of many of **signac**'s core developers and thus fields where the project has had greater exposure.
In addition to social factors such as the "home field advantage," materials research commonly requires large HPC resources with shared file systems, a use case where **signac** excels.
However, there are many other fields with similar hardware needs where **signac** can be applied.
These include simulation-heavy HPC workloads such as fluid dynamics, atomic/nuclear physics, or genomics, data-intensive fields such as economics or machine learning, and applications needing fast, flexible prototypes for optimization and data analysis.

..
    TODO: Categorize papers by field, show counts? e.g. The most common scientific fields citing signac are materials science (10), molecular simulation (8), optical materials (5), ...

    Brandon - this is cool, but may be more work than is worth it.

While there is no "typical" **signac** project, factors such as computational complexity and data sizes offer some rough guidelines for when **signac**'s database-on-the-filesystem is appropriate.
For instance, the time to check the status of a workflow depends on the number of jobs, number of operations, and number of conditions to evaluate for those jobs.
To give a rough idea of the limits of scalability, it can be difficult to scale **signac** projects beyond around 100,000 jobs while keeping tasks like checking workflow status in an "interactive" time scale of 1-2 minutes.
Many **signac** projects have 100 to 10,000 jobs, with each job workspace containing arbitrarily large data sizes (the total file size of the job workspace has little effect on the speed of the **signac** framework).
Some users that primarily wish to leverage **signac-flow**'s workflows for execution and submission may have a very small number of jobs (< 10).
One example of this would be executing a small number of expensive biomolecular simulations using different random seeds in each job's state point.
Importantly, projects with a small number of jobs can be expanded at a later time, and make use of the same workflow defined for the initial set of jobs.
The abilities to grow a project and change its schema on-the-fly catalyze the kind of exploration that is crucial to answering research questions.

The workflow submission features of **signac-flow** interoperates with popular HPC schedulers including SLURM, PBS/TORQUE, and LSF automating the generation and submission of scheduler batch scripts.
Directives are set through Python function decorators and define resource and execution requests for operations.
Examples of directives include number of CPUs or GPUs, the walltime, and memory.
The use of directives allows **signac-flow** workflows to be portable across HPC systems by generating resource requests that are specific to each machine's scheduler.

Overview of New Features
------------------------

The last three years of development of the **signac** framework have expanded its usability, feature set, user and developer documentation, and potential applications.
Some of the largest architectural changes in the framework will be discussed in their own sections, namely extensions of the workflow model (support for executing groups of operations and aggregators that allow operations to act on multiple jobs) and a much more performant and flexible re-implementation of the core "data structure" classes that synchronize **signac**'s Python representation of state points and job documents with JSON-encoded dictionaries on disk.

Data Archival
~~~~~~~~~~~~~

The primary purpose of the core **signac** package is to simplify and accelerate data management.
The **signac** command line interface is a common entry point for users, and provides subcommands for searching, reading, and modifying the data space.
New commands for import and export simplify the process of archiving **signac** projects into a structure that is both human-readable and machine-readable for future access (with or without **signac**).
Archival is an integral part of research data operations that is frequently overlooked.
By using highly compatible and long-lived formats such as JSON for core data storage with simple name schemes, **signac** aims to preserve projects and make it easier for studies to be independently reproduced.
This is aligned with the principles of TRUE (Transparent, Reproducible, Usable by others, and Extensible) simulations put forth by the MoSDeF collaboration :cite:`thompson.etal2020`.

..
    TODO: mention MIDAS Reproducibility Challenge? signac won an award.
    https://signac.io/talks/2020/08/05/midas-reproducibility.html

    Brandon - I don't know if it fits in this section, except maybe as proof of our claims. I think
    we are fine without mentioning it though especially since it is an internal to UM organization.

Improved data storage, retrieval, and integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data access via the shell:**
The ``signac shell`` command allows the user to quickly enter a Python interpreter that is pre-populated with variables for the current project or job (when in a project or job directory).
This means that manipulating a job document or reading data can be done through a hybrid of bash/shell commands and Python commands that are fast to type.

.. code-block:: shell

    ~/project $ ls
    signac.rc workspace
    ~/project $ cd workspace/42b7b4f2921788e.../
    ~/project/workspace/42b7b4f2921788e... $ signac shell
    Python 3.8.3
    signac 1.6.0

    Project:        test
    Job:            42b7b4f2921788ea14dac5566e6f06d0
    Root:           ~/project
    Workspace:      ~/project/workspace
    Size:           1

    Interact with the project interface using the
    "project" or "pr" variable. Type "help(project)"
    or "help(signac)" for more information.

    >>> job.sp
    {'a': 1}

**HDF5 support for storing numerical data:**
Many applications used in research generate or consume large numerical arrays.
For applications in Python, NumPy arrays are a de facto standard for in-memory representation and manipulation.
However, saving these arrays to disk and handling data structures that mix dictionaries and numerical arrays can be cumbersome.
The **signac** H5Store feature offers users a convenient wrapper around the ``h5py`` library :cite:`collette2013` for loading and saving both hierarchical/key-value data and numerical array data in the widely-used HDF5 format :cite:`hdf5`.
The ``job.data`` attribute is an instance of the ``H5Store`` class, and is a key-value store saved on disk as ``signac_data.h5`` in the job workspace.
Users who prefer to split data across multiple files can use the ``job.stores`` API to save in multiple HDF5 files.
Corresponding ``project.data`` and ``project.stores`` attributes exist, which save data files in the project root directory.
Using an instance of ``H5Store`` as a context manager allows users to keep the HDF5 file open while reading large chunks of the data:

.. code-block:: python

    with job.data:
        # Copy array data from the file to memory
        # (which will persist after the HDF5 file is
        # closed) by indexing with an empty tuple:
        my_array = job.data["my_array"][()]

**Advanced searching and filtering of the workspace:**
The ``signac diff`` command, available on both the command line and Python interfaces, returns the difference between two or more state points and allows for easily assessing subsets of the data space.
By unifying state point and document queries, filtering, and searching workspaces can be more fine-grained and intuitive.

Data Visualization and integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Integrating with the PyData Ecosystem:**
Users can now summarize data from a **signac** project into a pandas DataFrame for analysis.
The ``project.to_dataframe()`` feature exports state point and job document information to a pandas DataFrame in a consistent way that allows for quick analysis of all jobs' data.
Support for Jupyter notebooks has also been added, enabling rich HTML representations of **signac** objects.

**Dashboards:**
The companion package **signac-dashboard** allows users to quickly visualize data stored in a **signac** data space.
The dashboard runs in a browser and allows users to display job state points, edit job documents, render images and videos, download any file from a job workspace, and search or browse through state points in their project.
Dashboards can be hosted on remote servers and accessed via port forwarding, which makes it possible to review data generated on a remote HPC system without needing to copy it back to a local system for inspection.
Users can quickly save notes into the job document and then search those notes, which is useful for high throughput studies that require some manual investigation (e.g. reviewing plots).

Performance Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~

In early 2021, a significant portion of the codebase was profiled and refactored to improve performance and these improvements were released in **signac** 1.6.0 and **signac-flow** 0.12.0.
As a result of these changes, large **signac** projects saw 4-7x speedups for operations such as iterating over the jobs in a project compared to the 1.5.0 release of **signac**.
Similarly, performance of a sample workflow that checks status, runs, and submits a FlowProject with 1,000 jobs, 3 operations, and 2 label functions improved roughly 4x compared to **signac-flow** 0.11.0.
These improvements allow **signac** to scale to ~100,000 jobs.

In **signac**, the core of the ``Project`` and ``Job`` classes were refactored to support lazy attribute access and delayed initialization, which greatly reduces the total amount of disk I/O by waiting until data is actually requested by the user.
Other improvements include early exits in functions, reducing the number of required system calls with smarter usage of the ``os`` library, and switching to algorithms that operate in constant time (:math:`O(1)`) instead of linear time (:math:`O(N_{jobs})`).
Optimizations were identified by profiling the performance of common operations on small and large real-world projects with cProfile and visualized with snakeviz :cite:`snakeviz`.

..
    TODO: include a graph of performance from 1.0 to now

Similarly, performance enhancements were also made in the **signac-flow** package.
Some of the optimizations identified include lazy evaluation of run commands and directives, and caching of job status information.
In addition, the improvements in **signac** such as faster iteration over large **signac** projects used in **signac-flow** made **signac-flow**'s primary functions — checking project status, executing operations, and submitting operations to a cluster — significantly faster.

Improved User Output
~~~~~~~~~~~~~~~~~~~~

**Workflow graph detection:**
The preconditions and postconditions of operations in a **signac-flow** ``FlowProject`` implicitly define a graph. For example, if the operation "analyze" depends on the operation "simulate" via the precondition ``@FlowProject.pre.after(simulate)``, then there is a directed edge from "simulate" to "analyze."
This graph can now be detected from the workflow conditions and returned in a NetworkX compatible format for display or inspection.

**Templated status output:**
Querying the status of a **signac-flow** project now has many options controlling the information displayed and has been templated to allow for raw, Markdown, or HTML output. In doing so, the output has also become cleaner and compatible with external tools.

Enhanced Workflows
~~~~~~~~~~~~~~~~~~

**Directives:**
Directives provide a way to specify required resources on HPC schedulers such as number of CPUs/GPUs, MPI ranks, OpenMP threads, walltime, memory, and others. Directives can be a function of the job as well as the operation, allowing for great flexibility. In addition, directives work seamlessly with operation groups, job aggregation, and submission bundling (all of which are described in a later section).

**Dynamic Workspaces:**
The **signac-flow** package can now handle workspaces where jobs are created as the result of operations on other jobs. This is crucial for optimization workflows and iteratively sampling parameter spaces, and allows projects to become more automated with some data points only run if a prior condition on another data point is reached.

Executing complex workflows via groups and aggregation
------------------------------------------------------

.. figure:: signac-flow_aggregation-groups-bundling.pdf
    :align: center
    :scale: 90 %
    :figclass: w

    Aggregation, groups, and bundling allow users to build complex workflows.
    The features are orthogonal, and can be used in any combination.
    Aggregation enables one operation or group to act on multiple jobs.
    Groups allow users to combine multiple operations into one, with dependencies among operations resolved at run time.
    Bundling helps users efficiently leverage HPC schedulers by submitting multiple commands in the same script, to be executed in serial or parallel. :label:`workflow`

Two new concepts in **signac-flow** provide users with significantly more power to implement complex workflows: groups and aggregation.
A third concept bundling is also involved in providing flexibility to users in their workflows, but
on the execution not definition side.
Figure :ref:`workflow` show a graphical illustration of the three concepts.

As the names of both groups and aggregation imply, the features enable the "grouping" or "aggregating" of existing concepts: operations in the case of groups and jobs in the case of aggregates.
The conceptual model of **signac-flow** builds on **signac**'s notions of the ``Project`` and ``Job`` (the unit of the data space) through a ``FlowProject`` class that adds the ability to define and execute operations (the unit of a workflow) that act on jobs.
Operations are Python functions or shell commands that act on a job within the data space, and are defined using Python decorator syntax.

.. code-block:: python

    # project.py
    from flow import FlowProject

    @FlowProject.operation
    @Flowproject.post.true("initialized")
    def initialize(job):
        # perform necessary initialize steps
        # for simulation
        job.doc.initialized == True

    if __name__ == "__main__":
        FlowProject().main()

When this project is run using **signac-flow**'s command line API (``python project.py run``), the current state point is prepared for simulation.
Operations can have preconditions and postconditions that define their eligibility.
All preconditions must be met in order for a operation to be eligible for a given job.
If any postcondition is met, that indicates an operation is complete (and thus ineligible).
Examples of such conditions include the existence of an input file in a job's workspace or a key in the job document (as shown in the above snippet).
However, this type of conditional workflow can be inefficient when sequential workflows are coupled with an HPC scheduler interface, because the user must log on to the HPC and submit the next operation after the previous operation is complete.
The desire to submit large and long-running jobs to HPC schedulers encourages users to write large operation functions which are not modular and do not accurately represent the individual units of the workflow, thereby limiting **signac-flow**'s utility and reducing the readability of the workflow.

Groups, implemented by the ``FlowGroup`` class and ``FlowProject.make_group`` method, allows users to combine multiple operations into a single entity that can be run or submitted.
Submitting a group allows **signac-flow** to dynamically resolve preconditions and postconditions of operations as each operation is executed, making it possible to combine separate operations (e.g. for simulation and analysis and plotting) into a single submission script with the expectation that all will execute despite the dependencies between operations.
Furthermore, groups are aware of directives and can properly combine the directives of their constituent operations to specify resources and quantities like walltime whether executing in parallel or serial.

.. code-block:: python

    from flow import FlowProject

    new_group = FlowProject.make_group(name="new_group")

    @new_group.with_directives(
        {"ngpu": 2,
         "walltime": lambda job: job.doc.hours_to_run})
    @FlowProject.post.true("simulated")
    @FlowProject.operation
    def simulate(job):
        # run simulation
        job.doc.simulated = True

    @new_group
    @FlowProject.pre.after(simulate)
    @FlowProject.post.true("analyzed")
    @FlowProject.operation
    def analzye(job):
        # analyze simulation results
        job.doc.analzyed = True

Groups also allow for specifying multiple machine specific resources (CPU or GPU) with the same operation.
An operation can have unique directives for each distinct group to which it belongs.
By associating an operation's directives with respect to a specific group, groups can represent distinct compute environments, such as a local workstation or a remote supercomputing cluster.
The below snippet shows an ``expensive_simulate`` operation which can be executed with three
different directives depending on how it is written.
If executed through ``cpu_env`` the operation will request 48 cores, if ``gpu_env`` 4 GPUs, if
neither then it will request 4 cores.
This represents the real use case where an user may want to run an operation locally (in this case
without a group), or on a CPU or GPU focused HPC/workstation.

.. code-block:: python

    from flow import FlowProject

    cpu_env = FlowProject.make_group(name="cpu")
    gpu_env = FlowProject.make_group(name="gpu")

    @cpu_env.with_directives({"np": 48})
    @gpu_env.with_directives({"ngpu": 4})
    @FlowProject.operation.with_directives({"np": 4})
    def expensive_simulate(job):
        # expensive simulation for running on either
        # CPUs or GPUs
        pass

Users also frequently work with multiple jobs when performing tasks such as plotting data from all jobs in the same figure.
Though the **signac** package has methods like ``Project.groupby``, which can generate subsets of the project that are grouped by a state point key, there has been no similar feature in **signac-flow** to allow operations to act on multiple jobs.
The concept of aggregation provides a straightforward way for users to write and submit operations that act on arbitrary subsets of jobs in a **signac** data space.
Just as the groups feature acts as an abstraction over operations, aggregation can be viewed as an abstraction over jobs.
When decorated with an aggregator, operations can accept multiple job instances as positional arguments through Python's argument unpacking.
Decorators are used to define aggregation behavior, encompassed in the ``@aggregator`` decorator for single operations and in the argument ``aggregator_function`` to ``FlowProject.make_group`` for groups of operations.

.. code-block:: python

    from flow import FlowProject

    @aggregator
    @FlowProject.operation
    def plot_enzyme_activity(*jobs):
        import matplotlib.pyplot as plt
        import numpy as np

        x = [job.sp.temperature for job in jobs]
        y = [job.doc.activity for job in jobs]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title(
            "Enzymatic Activity Across Temperature")
        fig.savefig("enzyme-activity.png")

Like groups, there are many reasons why a user might wish to use aggregation.
For example, a **signac** data space that describes weather data for multiple cities in multiple years might want to plot or analyze data that uses ``@aggregator.groupby("city")`` to show changes over time for each city in the data space.
Similarly, aggregating over replicas (e.g. the same simulation with different random seeds) facilitates computing averaged quantities and error bars.
Another example is submitting aggregates with a fixed number of jobs in each aggregate to enable massive parallelization by breaking a large MPI communicator into a smaller communicator for each independent job, which is necessary for efficient utilization of leadership-class supercomputers like OLCF Summit.

Finally, bundling is another way to use workflows in conjunction with an HPC scheduling system.
Whereas aggregates are concerned with jobs and groups operations, bundling is concerned with combining what are effectively units of execution into a single submission script.
This distinction means that bundling is not part of the workflow definition, but a means of tailoring batch scripts for different HPC systems.
Bundles allow users to leverage scheduler resources effectively and minimize queue time, and can be run in serial (the default) or parallel.
Users enable bundling by passing the command line argument ``--bundle``, optionally with another argument ``--parallel`` to run each command in the bundle in parallel (the Python API has corresponding options as well).
The simplest case of a bundle is a submission script with the same operation being executed for multiple jobs.
Bundling is what allows the submission script to contain multiple jobs executing the same operation.
By storing information about the generated bundles during submission, **signac-flow** prevents accidental resubmission just as in the unbundled case.
While the example above does not use either groups or aggregation, bundles works seamlessly with both.

Synced Collections: Backend-agnostic, persistent, mutable data structures
-------------------------------------------------------------------------

Motivation
~~~~~~~~~~

All of **signac**'s principal functions are designed around efficiently indexing a collection of directories.
By organizing job directories by the hash of their state point, **signac** can perform many operations in constant time.
To present a Pythonic API, state points are exposed via a dictionary-like interface, making it very easy to modify a state point and have that change transparently reflected in both the JSON file and the name of the corresponding directory (which is the state point's hash).

The need to parse these JSON files for indexing and the complexity of modifying them represent the most significant barriers to scaling **signac**.
Even in the absence of file modification, simply reading a large number of files to produce a database index becomes prohibitively expensive for large data spaces.
Although various optimizations have incrementally improved **signac**'s scalability, an alternative means of storing the state point and associated metadata that circumvents the heavy I/O costs of our current approach has the potential to make a much larger impact.
However, replacing individual JSON files as the primary data source for **signac** without breaking **signac**'s API would require a generic means for providing the same interface to the underlying index and metadata irrespective of the underlying storage mechanism.
Once developed, however, such an API would abstract out enough of the internals of **signac** to enable other generalizations as well, such as making it relatively easy to support alternate (and nearly arbitrary) data space layouts.

The synced collections subpackage of **signac** represents the culmination of our efforts to expose this functionality, providing a generic framework within which interfaces corresponding to any of Python's built-in types can be easily constructed with arbitrary underlying synchronization protocols.
For instance, with synced collections it becomes easy to define a new list-like type that automatically saves all its data in a plain-text CSV format.
However, the flexibility of this new framework extends far beyond that, defining a generic protocol that can be used to provide a dictionary, list, or set-like API to any arbitrary underlying data structure, including other in-memory objects that do not present a similarly Pythonic API.

Summary of Features
~~~~~~~~~~~~~~~~~~~

We designed synced collections to be flexible, easily extensible, and independent of the rest of **signac**.
The central element is the ``SyncedCollection`` class, which defines a new abstract class extending the ``collections.abc.Collection`` from the Python standard library.
A ``SyncedCollection`` is a ``Collection`` that adds two additional groups of abstract methods that must be implemented by its subclasses.
One group includes methods for synchronizing with an underlying resource, while the other contains methods for synchronizing with a standard collection of the underlying base type.
For instance, a ``JSONDict`` would implement the first set of methods to define how to save a dictionary to a JSON file and reload it, while it would implement the second set of methods to define how to convert between a ``JSONDict`` instance and a standard Python dictionary.

Critically, these two sets of functions are orthogonal.
Therefore, it is possible to implement different backend types and different data structures independently, then combine them into concrete classes using multiple inheritance.
This solution is analogous to the way that language server protocols separate support for programming languages from support for editors, turning an :math:`M \times N` problem into a simpler :math:`M+N` problem.
In practice, our synced collections framework comes bundled with a set of backend classes, such as the ``JSONCollection``, and a set of data structure classes, such as the ``SyncedDict``.
Each of these inherits from ``SyncedCollection`` and implements a subset of its methods, but remains abstract until combined (via multiple inheritance) with a class implementing the remaining methods.
This design pattern makes defining the functional classes at the bottom of the hierarchy trivial.
For example, the ``JSONDict`` is implemented by inheriting from ``JSONCollection`` and ``SyncedDict``, but requires no additional code to function.
Similarly, the ``JSONList`` class inherits from ``JSONCollection`` and ``SyncedList``.

This infrastructure is also flexible enough to accommodate general modifications to the synchronization protocol.
A prominent example is the ``BufferedCollection``, a subclass of ``SyncedCollection`` that introduces additional synchronization primitives that enable toggling synchronization to and from the underlying resource with synchronization to and from an intermediate cache for improved performance.
Similarly to base ``SyncedCollection`` functions, different buffering behaviors' synchronization can be implemented independently of the specific backend (or even the data structure, for any buffer that supports generic objects).

Applications of Synced Collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new synced collections promise to substantially simplify both feature and performance enhancements to the **signac** framework.
Performance improvements in the form of Redis-based storage are already possible with synced collections, and as expected they show substantial speedups over the current JSON-based approach.
The use of the new buffering protocol has enabled us to prototype new buffering approaches that further improve performance in buffered mode.
At a larger scale, synced collections are a critical first step to enabling different data layouts on disk, such as the use of a single tabular index (e.g.
a SQLite database) for much faster work on homogeneous data spaces or the use of more deeply nested directory structures where a deeper hierarchy on disk offers organizational benefits.

The generality of synced collections makes them broadly useful even outside the **signac** framework.
The framework makes it easy for developers to create Pythonic APIs for data structures that might otherwise require significant additional implementation overhead.
Crucially, synced collections support nesting as a core feature, something that could be quite difficult to handle for developers of custom collection types.
Moreover, while the framework was originally conceived to support synchronization of an in-memory data structure with a resource on disk, it can also be used to synchronize with another in-memory resource.
One powerful example of this would be the use of a synced collection to provide a Pythonic API to a collection-like data structure implemented as a C or C++ extension module that could function like a Python dictionary with suitable plumbing but lacks the standard APIs expected of such a class.
With the synced collections framework, creating a new class providing such an API is reduced to simply requiring the implementation of two straightforward methods defining the synchronization protocol.

..
    TODO: discuss independence from the rest of signac, possibility of releasing as a separate package?


Project Evolution
-----------------

The **signac** project has evolved from being an open-source project mostly developed and managed by the Glotzer Group at the University of Michigan, to being supported by over 30 contributors and 8 committers/maintainers on 3 continents and with over 55 citations from academic and government research labs and 12 talks at large scientific, Python, and data science conferences.
The growth in involvement with **signac** is the result of our focus on developing features based on user needs, as well as our efforts to transition **signac** users to **signac** contributors, through many initiatives in the past few years.
Through encouraging users to become contributors, we ensure that **signac** addresses real users' needs.
We have expanded **signac**'s contributor base beyond the University of Michigan through research collaborations with other universities, sharing the framework at conferences, and through the Google Summer of Code (GSoC) program.
Working with and mentoring students through GSoC led to a new committer and significant work on the synced collections and aggregation projects presented above.
To encourage code contributions from existing users, we maintain active support and discussion through Slack.
In addition, we have started hosting weekly "office hours" for in-person (virtual) introduction and guided contributions to the code base.
By pairing new contributors with experienced **signac** developers, we significantly reduce the knowledge barrier to joining a new project.
Office hours creating space for users to make contributions has also led to more features and documentation born directly out of user need.
Contributing to documentation has been a productive starting point for new users-turned-contributors, both for the users and the project, since it improves the users' familiarity with the API as well as addresses weak spots in the documentation that are more obvious to new users.

In our growth with increasing contributors and users, we recognized a need to change our governance structure to make contributing easier and provide a clear organizational structure to the community.
We based our new model on the Meritocratic Governance Model and our manager roles on Numba Czars.
We decided on a four category system with maintainers, committers, contributors, and users.
Code review and pull request merge responsibilities are granted to maintainers and committers, who are (self-) nominated and accepted by a vote of the project maintainers.
Maintainers are additionally responsible for the strategic direction of the project and administrative duties.
Contributors consist of all members of the community who have contributed in some way to the framework, which includes adding or refactoring code as well as filing issues and improving documentation.
Finally, users refer to all those who use **signac** in any capacity.

In addition, to avoid overloading our committers and maintainers, we added three rotating manager roles to our governance model that ensure project management goes smoothly: triage, community, and release.
These managers have specific rotation policies based on time (or release cycles).
The triage manager role rotates weekly and looks at new issues or pull requests and handles cleanup of outdated issues.
The community manager role rotates monthly and is in charge of meeting planning and outreach.
Lastly, the release manager rotates with each release cycle and is the primary decision maker for the timeline and feature scope of package releases.
This prevents burnout among our senior developers and provides a sense of ownership to a greater number of people, instead of relying on a "benevolent dictator/oligarchy for life" mode of project leadership.


Conclusions
-----------

From the birth of the **signac** framework to now, **signac** has grown in usability, performance, and use.
In the last three years, we have added exciting new features, like groups, aggregation, and synced collections, while learning how to manage outreach and establish sustainable project governance in a burgeoning scientific open-source project.
As maintainers and committers, we hope to continue expanding the framework through user-oriented development, reach users in research fields beyond materials science that routinely have projects suited for **signac**, and welcome new contributors with diverse backgrounds and skills to the project.

Getting signac
--------------

The **signac** framework is tested for Python 3.6+ and is compatible with Linux, macOS, and Windows.
The software is available under the BSD-3 Clause license.

To install, execute

.. code-block:: bash

    conda install -c conda-forge signac \
    signac-flow signac-dashboard

or

.. code-block:: bash

    pip install signac signac-flow signac-dashboard

Source code is available on GitHub [#]_ [#]_ and documentation is hosted online by ReadTheDocs [#]_.

.. [#] https://github.com/glotzerlab/signac
.. [#] https://github.com/glotzerlab/signac-flow
.. [#] https://docs.signac.io/


Acknowledgments
---------------

We would like to thank Kelly Wang for contributing the concept and content of Figure :ref:`overview`.
We would also like to thank NumFOCUS, whose staff have provided the **signac** project with helpful advice on open-source governance, project sustainability, and community outreach.

(TODO: Acknowledgments are incomplete.)
B.D. is supported by a National Science Foundation Graduate Research Fellowship Grant DGE 1256260.
M.M.H is supported by the National Science Foundation under Grant No. 1835593.

Author contributions
--------------------

Conceptualization, B.D.D., B.L.B., V.R., A.T., M.M.H., H.O., and C.S.A.;
data curation, B.D.D., B.L.B., V.R., A.T., and M.M.H.;
funding acquisition, E.J. and S.C.G.;
investigation, B.D.D., B.L.B., V.R., and C.S.A.;
methodology, B.D.D., B.L.B., V.R., A.T., M.M.H., H.O., and C.S.A.;
project administration, B.D.D., B.L.B., V.R., A.T., M.M.H., H.O., and C.S.A.;
software, B.D.D., B.L.B., V.R., A.T., M.M.H., H.O., and C.S.A.;
supervision, S.C.G.;
visualization, B.D.D., B.L.B., A.T., and K.W.;
writing – original draft, B.D.D., B.L.B., V.R., A.T., and H.O.;
writing – review & editing, B.D.D., B.L.B., V.R., A.T., M.M.H., H.O., K.W., and S.C.G.
All authors have read and agreed to the published version of the manuscript.