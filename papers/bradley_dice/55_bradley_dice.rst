:author: Bradley D. Dice
:email: bdice@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor

:author: Brandon L. Butler
:email: butlerbr@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Vyas Ramasubramani
:email: vramasub@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Alyssa Travitz
:email: atravitz@umich.edu
:institution: TODO Department, University of Michigan, Ann Arbor

:author: Michael M. Henry
:email: mike.henry@choderalab.org
:institution: TODO

:author: Hardik Ojha
:email: hojha@ee.iitr.ac.in
:institution: TODO Department, Indian Institute of Technology Roorkee

:author: Carl S. Adorf
:email: TODO
:institution: TODO

:author: Sharon C. Glotzer
:email: sglotzer@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: Department of Materials Science and Engineering, University of Michigan, Ann Arbor
:institution: Biointerfaces Institute, University of Michigan, Ann Arbor

:bibliography: paper

-------------------------------------------------------------------
signac: Data Management and Workflows for Computational Researchers
-------------------------------------------------------------------

.. class:: abstract

The signac data management framework (https://signac.io) helps researchers execute reproducible computational studies, scaling from laptops to supercomputers and emphasizing portability and fast prototyping.
With signac, users can track, search, and archive data and metadata for file-based workflows and automate workflow submission on high performance computing (HPC) clusters.
We will discuss recent improvements to the software’s feature set, scalability, scientific applications, usability, and community.
Newly implemented synced data structures, workflow subgraph execution, and performance optimizations will be covered, as well as recent research using the framework and the project’s efforts on improving documentation, contributor onboarding, and governance.

.. class:: keywords

   data management, TODO


Introduction
------------

.. figure:: signac_overview.pdf
    :align: center
    :scale: 40 %
    :figclass: w

    Overview of the signac framework.
    Users first create a project, which initializes a workspace directory on disk.
    Users define state points which are dictionaries that uniquely identify a job.
    The workspace holds a directory for each job, containing JSON files that store the state point and job document.
    The job directory name is a hash of the state point's contents.
    Here, the ``init.py`` file initializes an empty project and adds one job with state point ``{"a": 1}``.
    Next, users define a workflow using a subclass of signac-flow's ``FlowProject``.
    The workflow shown has three operations (simulate, analyze, visualize) that, when executed, produce two new files ``results.txt`` and ``plot.png`` in the job directory.

The full source code of all examples in this paper can be found online [#]_.

.. [#] https://github.com/glotzerlab/signac-examples

Research projects often address problems where questions change rapidly, data models are always in flux, and compute infrastructure varies widely from project to project.
The signac data management framework is a tool designed by researchers, for researchers, to make prototyping quick and reproducibility easy.
It forgoes serializing complex data files into a database in favor of working with these files directly, providing fast indexing utilities for a set of directories.
Using signac, a data space on the file system is easily initialized, searched, and modified using either a Python or command-line interface.
The companion package signac-flow interacts with the data space to generate and analyze data through reproducible workflows that easily scale from laptops to supercomputers.
signac-flow can run arbitrary commands as part of a workflow, making it as flexible as a script in any language of choice.

With signac, file-based data and metadata are organized in folders and JSON files, respectively (see Figure 1).

..
    TODO: Add figure label and update figure references -- Bradley couldn't get the paper to build after adding a label.

A signac data space (or workspace) is composed of jobs, individual directories associated with a single primary key (or state point) stored in a file `signac_statepoint.json` in that directory.
Signac uses these files to index the data space, providing a database-like interface to a collection of directories.
Arbitrary user data may be stored in user-created-files in these jobs, although signac also provides convenient facilities for storing simple lightweight data or array-like data via JSON and HDF5 utilities.
Readers seeking more detail on signac may refer to past signac papers: :cite:`signac_commat, signac_scipy_2018` as well as the signac website [#]_ and documentation [#]_.

.. [#] https://signac.io
.. [#] https://docs.signac.io

This filesystem-based approach has both advantages and disadvantages.
Its key advantages lie in flexibility and portability.
The serverless design removes the need for any external running server process, making it easy to operate on any arbitrary filesystem.
The design is also intrinsically distributed, making it well suited for highly-parallel workflows where each instance generates more file-based data.
Conversely, this distributed approach precludes performance advantages of centralized data stores with persistent indexes in memory.
Typically, the signac approach works very well for projects up to 100,000 jobs, while significantly larger projects may have wait times that constrain interactive usage.
These limits are inherent to signac's use of small files for each job's state point, but the framework has been aggressively optimized and uses extensive caching/buffering to maximize the throughput that can be achieved within this model.

The framework is a strong choice for applications involving file-based workflows, especially those that are quickly evolving or will run on HPC clusters, especially where the amount of required computation per job is large.
For example, (...TODO: insert examples of real world projects with scientific applications [HOOMD, MPB, etc], project sizes [100,000 jobs], number of jobs / number of terabytes).
Users working with large tabular data (e.g. flat files on disk or data from a SQL database) may prefer to use libraries like pandas, dask, or RAPIDS that are specifically designed for those use cases.
However, it is possible to create a signac project with state points corresponding to each row, which may be a good use of signac if there is file-based data affiliated with each row's parameters.

This paper will focus on developments to the signac framework over the last 3 years, during which features, flexibility, usability, and performance have been greatly improved.
The core data structures in signac have been overhauled to provide a powerful, generic implementation of "synced collections," that we will leverage in future versions of signac to enable more performant data indexing and more flexible data layouts.
In signac-flow, we have added support for submitting groups of operations with conditional dependencies, allowing for more efficient utilization of large HPC resources, and further developments allow for operations to act on arbitrary subsets of the data space rather than single jobs alone.
Meanwhile, performance enhancements have enabled scaling up to much larger data spaces.
Moving beyond code development, this paper will also discuss the scientific work these features have enabled, and key partnerships and affiliations with scientific software initiatives and organizations such as MoSDeF and NumFOCUS.
We will share our project's experience in progressively revising project governance to catalyze sustained contributions of many kinds, while adding more points of entry for learning about the project (Slack support, office hours), and participating in Google Summer of Code in 2020 as a NumFOCUS Affiliated Project.

Applications of signac
----------------------

The signac framework has been cited 51 times, according to Google Scholar.
The framework has been used in a range of scientific fields and with many types of computational workflows.
Some of these studies include quantum calculations of small molecules, screenings of ionic liquids and organic solvents, inverse design of pair potentials, optimizing photonic band gaps in colloidal materials, analyzing colloidal self-assembly with machine learning, and economic analysis of drought risk [no citation].
Much of the published research using signac comes from chemical engineering, materials science, or physics, the home fields of many of signac's core developers and thus fields where the project has had greater exposure.
In addition to social factors such as the "home field advantage," materials research commonly requires large HPC resources with shared file systems, a use case where signac excels.
However, there are many other fields with similar hardware needs where signac can be applied.
These include simulation-heavy HPC workloads such as fluid dynamics, atomic/nuclear physics, or genomics; data-intensive fields such as economics or machine learning; and applications needing fast, flexible prototypes for optimization and data analysis.

..
    TODO: Categorize papers by field, show counts? e.g. The most common scientific fields citing signac are materials science (10), molecular simulation (8), optical materials (5), ...

While there is no "typical" signac project, factors such as computational complexity and data sizes offer some rough guidelines for when signac's database-on-the-filesystem is appropriate.
For instance, the time to check the status of a workflow depends on the number of jobs, number of operations, and number of conditions to evaluate for those jobs.
To give a rough idea of the limits of scalability, it can be difficult to scale signac projects beyond around 100,000 jobs while keeping tasks like checking workflow status in an "interactive" time scale of 1-2 minutes.
Many signac projects have 100 to 10,000 jobs, with each job workspace containing arbitrarily large data sizes (the file size of the job workspace has little effect on the speed of the signac framework).
Some users that primarily wish to leverage signac-flow's workflows for execution and submission may have a very small number of jobs (< 10).
One example of this would be executing a small number of expensive biomolecular simulations using different random seeds in each job's state point.

..
    TODO Try to find example of a project with small number of state points in literature citing signac.

The workflow submission features of signac-flow interoperate with popular HPC schedulers including SLURM, PBS/TORQUE, and LSF.
Operations in a FlowProject can define directives, which indicate hardware to request such as the number of processors or GPUs, the amount of memory, or the walltime needed to complete the operation.

..
    TODO Address redundancy with above content about processors/GPUs

These directives allow signac-flow to generate scripts for the currently present scheduler, enabling portability across HPC systems.
Moreover, signac-flow can combine operations and their directives in a number of ways, such as in serial or parallel bundles, or the new features for groups and aggregation discussed below.

..
    TODO Make sure to discuss bundling in the aggregation section. Avoid discussing serial/parallel bundles right here, because it hasn't been defined.

This allows users to leverage scheduler resources effectively and minimize queue time (or optimize for HPC policies that prefer large submissions) by bundling many operations into a small number of scheduler submissions.

..
    (TODO: Move this into the section above?) The framework emphasizes performance for common user workspaces and workflows.

In early 2021, a significant portion of the codebase was profiled and refactored to improve performance, many of these are changes listed above.
These improvements were released in signac v1.6.0 and signac-flow v0.12.0.
Large signac projects saw 4-7x for operations such as iterating over the jobs in a project compared to the v1.5.0 release of signac.
Similarly, performance of a sample workflow that checks status, runs, and submits a FlowProject with 1,000 jobs, 3 operations, and 2 label functions improved roughly 4x compared to the v0.11.0 release of signac-flow.

Some signac developers have begun conversations with experimental researchers about how the framework might be useful for a broader range of research tasks, such as workflows that combine computational steps such as optimization or post processing with steps that might be performed (or manually triggered) by a researcher, such as the collection of data files from a microscope or robot.

Overview of New Features
------------------------

The last three years of development on the signac framework have improved its usability, feature set, user and developer documentation, and potential applications.
Some of the largest architectural changes in the framework will be discussed in their own sections, namely extensions of the workflow model (support for executing operation groups and aggregators that allow operations to act on multiple jobs) and a much more performant and flexible re-implementation of the core "data structure" classes that synchronize signac's Python representation of state points and job documents with JSON-encoded dictionaries on disk.

Data Archival
~~~~~~~~~~~~~

The primary purpose of the core signac package is to simplify and accelerate data management.
The signac command line interface is a common entry point for users, and provides subcommands for searching, reading, and modifying the data space.
New commands for import and export simplify the process of archiving signac projects into a structure that is both human-readable and machine-readable for future access (with or without signac).
Archival is an integral part of research data operations that is frequently overlooked.
By using highly compatible and long-lived formats such as JSON for core data storage with simple name schemes, signac aims to preserve projects and make it easier for studies to be independently reproduced.
This is aligned with the principles of TRUE (Transparent, Reproducible, Usable by others, and Extensible) simulations put forth by the MoSDeF collaboration (https://doi.org/10.1080/00268976.2020.1742938).
(TODO: mention MIDAS Reproducibility Challenge? signac won an award.
https://signac.io/talks/2020/08/05/midas-reproducibility.html)

Simplifying and streamlining existing functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data access via the shell: The ``signac shell`` command allows the user to quickly enter a Python interpreter that is pre-populated with variables for the current project or job (when in a project or job directory).
This means that manipulating a job document or reading data can be done through a hybrid of bash/shell commands and Python commands that are fast to type.

.. code-block:: shell

    ~/project $ ls
    signac.rc workspace
    ~/project $ cd workspace/42b7b4f2921788ea14dac5566e6f06d0/
    ~/project/workspace/42b7b4f2921788ea14dac5566e6f06d0 $ signac shell
    Python 3.8.3
    signac 1.6.0

    Project:        test
    Job:            42b7b4f2921788ea14dac5566e6f06d0
    Root:           ~/project
    Workspace:      ~/project/workspace
    Size:           1

    Interact with the project interface using the "project" or "pr" variable.
    Type "help(project)" or "help(signac)" for more information.
    >>> job.sp
    {'a': 1}

Alternative short snippet using -c command flag:

.. code-block:: shell

    ~/project/workspace/42b7b4f2921788ea14dac5566e6f06d0 $ signac shell -c "print(job.sp)"
    {'a': 1}

HDF5 support for storing numerical data: Many applications used in research generate or consume large numerical arrays. For applications in Python, NumPy arrays are a de facto standard for in-memory representation and manipulation. However, saving these arrays to disk and handling data structures that mix dictionaries and numerical arrays can be cumbersome. The signac H5Store feature offers users a convenient wrapper around the h5py library for loading and saving both hierarchical/key-value data and numerical array data in the widely-used HDF5 format. The ``job.data`` attribute is an instance of the ``H5Store`` class, and is a key-value store saved on disk as ``signac_data.h5`` in the job workspace. Users who prefer to split data across multiple files can use the ``job.stores`` API to save in multiple HDF5 files. Corresponding ``project.data`` and ``project.stores`` attributes exist, which save data files in the project root directory. Using an instance of ``H5Store`` as a context manager allows users to keep the HDF5 file open while reading large chunks of the data.

.. code-block:: python

    job.stores[store_name][key_name] = np.random.rand(3, 3, 3)
    with job.data:
        # Copy array data from the file to memory (which will persist
        # after the HDF5 file is closed) by slicing with an empty tuple:
        my_array = job.data["my_array"][()]

Integrating with the PyData Ecosystem: Users can now summarize data from a signac project into a pandas DataFrame for analysis. The ``project.to_dataframe()`` feature exports state point and job document information to a pandas DataFrame in a consistent way that allows for quick analysis of all jobs' data. (TODO: Make note about heterogeneous schemas, interesting use cases?) Support for Jupyter notebooks has also been added, enabling rich HTML representations of signac objects.

Advanced searching and filtering of the workspace: The ``signac diff`` command, available on both the command line and Python interfaces, returns the difference between two or more state points and allows for easily assessing subsets of the dataspace. By unifying sp and doc querying, filtering, and searching workspaces can be more fine-grained and intuitive.

Core Performance Enhancements (overlaps with content in Applications section)
The scalability of the signac framework has been massively improved through performance enhancements that enable real-time interactive usage for workspaces with up to 100,000 jobs. The core of the signac Project and Job classes were refactored to support lazy attribute access and delayed initialization, which greatly reduces the total amount of disk I/O by waiting until data is actually requested by the user. Other improvements include early exits in functions, reducing the number of required system calls with smarter usage of the ``os`` library, and switching to algorithms that operate in constant time ($O(1)$) instead of linear time ($O(N_{jobs})$). Optimizations were identified by profiling the performance of common operations on small and large real-world projects with cProfile and visualized with snakeviz. (TODO: include a graph of performance from 1.0 to now)

Flow Performance Enhancements (overlaps with content in Applications section)
Performance enhancements were also made in the signac-flow package. Some of the optimizations identified include lazy evaluation of run commands and directives, caching of job status information, and faster iteration over large signac projects in shared code paths for signac-flow's primary functions: checking project status, executing operations, and submitting operations to a cluster.

Improved User Output
~~~~~~~~~~~~~~~~~~~~

Workflow graph detection: The preconditions and postconditions of operations in a signac-flow ``FlowProject`` implicitly define a graph. For example, if the operation "analyze" depends on the operation "simulate" via the precondition ``@FlowProject.pre.after(simulate)``, then there is a directed edge from "simulate" to "analyze."
This graph can now be detected from the workflow conditions and returned in a NetworkX compatible format for display or inspection.

Templated status output: Querying the status of a signac-flow project now has many more options and has been templated to allow for raw, Markdown, or HTML output. In doing so, the output has also become cleaner and compatible with external tools.

Enhanced Workflows
~~~~~~~~~~~~~~~~~~

Directives: Directives provide a way to specify required resources on HPC schedulers such as number of CPUs/GPUs, MPI ranks, OpenMP threads, walltime, memory, and others. Directives can be a function of the job as well as the operation, allowing for great flexibility. In addition, directives work seamlessly with operation groups, job aggregation, and submission bundling (all of which are described in a later section).

Dynamic Workspaces: The signac-flow package can now handle workspaces where jobs are created as the result of operations on other jobs. This is crucial for optimization workflows and iteratively sampling parameter spaces, and allows projects to become more automated with some data points only run if a prior condition on another data point is reached.

Executing complex workflows via groups and aggregation
------------------------------------------------------

Although already capable of implementing reproducible quality workflows, signac-flow has enhanced the usability through two new concepts: groups and aggregation.
As both names imply, the features enable the "grouping" or "aggregating" of existing concepts: operations in the case of groups and jobs in the case of aggregates.
In the conceptual model of signac-flow, flow builds on signac's notions of the project and job (the unit of the data space) through a FlowProject class that adds the ability to execute operations (the unit of a workflow) to a signac Project.
Operations are functions (Python functions or shell commands) that act on a job within the data space, and are created using Python decorator syntax (show snippet).
(Hardik added a snippet below -- Probably not the best example.
He thinks that for this portion, the snippets should be consistent so that readers can easily run these,)

.. code-block:: python

    # project.py
    from flow import FlowProject

    @FlowProject.operation
    @Flowproject.post.true("city")
    def store_current_city(job):
        job.doc.city == "Ann Arbor"

    if __name__ == '__main__':
        FlowProject().main()

When this project is run using signac-flow's command line API (``python project.py run``), the user's current city is written into the job document Ann Arbor in this case. (Hardik doesn't know if it's a good idea to display a high level of detail for the paper, but if we decide not to put this, we can delete later)
Operations can have preconditions and postconditions that define their eligibility, e.g. the existence of an input file in a job's workspace or a key in the job document (as shown in the above snippet) can be a precondition that must be met before an operation can be executed, or a postcondition that indicates an operation is complete. However, this type of conditional workflow can be inefficient when sequential workflows are coupled with an HPC scheduler interface, because the user must log on to the HPC and submit the next operation after the previous operation is complete. This encourages large operations which are not modular and do not accurately represent the individual units of the work-flow limiting signac-flow's utility.

The concept of a group, implemented by the ``FlowGroup`` class and ``FlowProject.make_group`` interface, allows users to combine multiple operations into a group. Submitting a group allows signac-flow to dynamically resolve preconditions and postconditions of operations as each operation is executed, making it possible to combine separate operations (e.g. for simulation and analysis and plotting) into a single submission script with the expectation that all will execute despite later operations depending on the former. Furthermore, groups are aware of directives and can properly combine the directives of their constituent operations to specify resources and quantities like walltime whether executing in parallel or serial.

.. code-block:: python

    from flow import FlowProject

    new_group = FlowProject.make_group(
        name="new_group")

    @new_group.with_directives(
        {"ngpu": 2.0,
         "walltime": lambda j: j.sp.size * 4})
    @FlowProject.post.true("foo")
    @FlowProject.operation
    def foo(job):
        job.doc.foo = True

    @new_group
    @FlowProject.pre.true("foo")
    @FlowProject.post.true("bar")
    @FlowProject.operation
    def bar(job):
        job.doc.bar = True

Groups also allow for specifying multiple machine specific resources (CPU v GPU) with the same operation. An operation can have unique directives for each group it is in. By associating an operation's directives with respect to a specific group, groups can represent distinct compute environments such as a local workstation or a remote supercomputing cluster.

.. code-block:: python

    from flow import FlowProject

    cpu_env = FlowProject.make_group(name="cpu")
    gpu_env = FlowProject.make_group(name="gpu")


    @cpu_env.with_directives({"np": 48})
    @gpu_env.with_directives({"ngpu": 4})
    @FlowProject.operation
    def expensive_operation(job):
        # expensive computation for either
        # CPU or GPU here
        pass

Users also frequently work with multiple jobs at once in a consistent manner.
Though the signac package has methods like ``Project.groupby``, which can generate subsets of the project that are grouped by a state point key, there has been no similar feature in signac-flow to allow operations to act on multiple jobs.
The concept of _aggregation_ provides a straightforward way for users to write and submit operations that act on arbitrary subsets of a signac data space.
Just as groups act as an abstraction over operations, aggregation can be viewed as an abstraction over jobs.
The operation syntax changes from `def my_operation(job):` to `def my_operation(*jobs):`, using Python's argument unpacking syntax to support user input of one or more job instances (keeping backwards compatibility).
Decorators are used to define aggregation behavior, encompassed in the ``aggregator`` decorator for single operations and in the argument ``aggregator_function`` to ``FlowProject.make_group`` for groups of operations.

.. code-block:: python

    from flow import FlowProject

    @aggregator
    @FlowProject.operation
    def operation_on_all_jobs(*jobs):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.array(
            [job.sp.temperature for job in jobs])
        y = np.array(
            [job.doc.activity for job in jobs])
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(
            "Enzymatic Activity Across Temperature")
        fig.savefig("enzyme-activity.png")

Like groups, there are many reasons why a user might wish to use aggregation.
For example, a signac data space that describes weather data for multiple cities in multiple years might want to plot or analyze data that uses ``aggregator.groupby("city")`` to show changes over time for each city in the data space.
Similarly, aggregating over replicas facilitates computing averaged quantities and errors.
Another example is submitting aggregates with a fixed number of jobs in each aggregate to enable massive parallelization by breaking a large MPI communicator into a smaller communicator for each independent job, which is necessary for efficient utilization of leadership-class supercomputers like OLCF Summit.

Synced Collections: Backend-agnostic, persistent, mutable data structures
-------------------------------------------------------------------------

Motivation
~~~~~~~~~~

All of signac's principal functions are designed around efficiently indexing a collection of directories.
By organizing job directories by the hash of their state point, signac can perform many operations in constant time.
To present a Pythonic API, state points are exposed via a dictionary-like interface, making it very easy to modify a state point and have that change transparently reflected in both the JSON file and the name of the corresponding directory.

The need to parse these JSON files for indexing and the complexity of modifying them represent the most significant barriers to scaling signac.
Even in the absence of file modification, reading a large number of files simply to produce a database index becomes prohibitively expensive for large data spaces.
Although various optimizations have incrementally improved signac's scalability, an alternative means of storing the state point and associated metadata that circumvents the heavy I/O costs of our current approach has the potential to make a much larger impact.
However, replacing individual JSON files as the primary data source for signac without breaking signac's API required a generic method for providing the same interface to the underlying index and metadata files irrespective of the underlying storage mechanism.
Once developed, however, such an API would abstract out enough of the internals of signac to enable other generalizations as well, such as making it relatively easy to support alternate (and nearly arbitrary) data space layouts.

The synced collections subpackage of signac represents the culmination of our efforts to expose this functionality, providing a generic framework within which interfaces corresponding to any of Python's built-in types can be easily constructed with arbitrary underlying synchronization protocols.
For instance, with synced collections it becomes easy to define a new list-like type that automatically saves all its data in a plain-text CSV format.
However, the flexibility of this new framework extends far beyond that, defining a generic protocol that can be used to provide a dictionary, list, or set-like API to any arbitrary underlying data structure, including other in-memory objects that do not present a similarly Pythonic API.

Summary of Features
~~~~~~~~~~~~~~~~~~~

We designed synced collections to be flexible, easily extensible, and independent of the rest of signac.
The central element is the ``SyncedCollection`` class, which defines a new abstract class extending the ``collections.abc.Collection`` from the Python standard library.
A ``SyncedCollection`` is a ``Collection`` that adds two additional groups of abstract methods that must be implemented by its subclasses.
One group includes methods for synchronizing with an underlying resource, while the other contains methods for synchronizing with a standard collection of the underlying base type.
For instance, a ``JSONDict`` would implement the first set of methods to define how to save a dictionary to a JSON file and reload it, while it would implement the second set of methods to define how to convert between a ``JSONDict`` instance and a standard Python dictionary.

Critically, these two sets of functions are orthogonal.
Therefore, it should be possible to implement different backend types and different data structures independently, then combine them after the fact.
This solution is analogous to the way that language server protocols separate support for programming languages from support for editors, turning a :math:`M*N` problem into a simple :math:`M+N` problem.
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

The new synced collections promise to substantially simplify both feature and performance enhancements to the signac framework.
Performance improvements in the form of Redis-based storage are already possible with synced collections, and as expected they show substantial speedups over the current JSON-based approach.
The use of the new buffering protocol has enabled us to prototype new buffering approaches that further improve performance in buffered mode.
At a larger scale, synced collections are a critical first step to enabling different data layouts on disk, such as the use of a single tabular index (e.g.
a SQLite database) for much faster work on homogeneous data spaces or the use of more deeply nested directory structures where a deeper hierarchy on disk offers organizational benefits.

The generality of synced collections makes them broadly useful even outside the signac framework.
The framework makes it easy for developers to create Pythonic APIs for data structures that might otherwise require significant additional implementation overhead.
Crucially, synced collections support nesting as a core feature, something that could be quite difficult to handle for developers of custom collection types.
Moreover, while the framework was originally conceived to support synchronization of an in-memory data structure with a resource on disk, it can just as easily be used to synchronize with another in-memory resource.
One powerful example of this would be the use of a synced collection to provide a Pythonic API to a collection-like data structure implemented as a C or C++ extension module that could function like a Python dictionary with suitable plumbing but lacks the standard APIs expected of such a class.
With the synced collections framework, creating a new class providing such an API is reduced to simply requiring the implementation of two straightforward methods defining the synchronization protocol.

..
    TODO: discuss independence from the rest of signac, possibility of releasing as a separate package?

Related Software
~~~~~~~~~~~~~~~~

Are there other packages with related purposes? Vyas is not aware of any, the closest thing is Zict, a project Bradley pointed out a while ago.
However, its scope is limited to composing mutable mappings.
However, one natural question I'd expect from people is how this package to add a collection-like interface to some object compares to those objects directly implementing the interface.
For example, I'd expect our closest comparison for a Redis-backed dict to be pyredis itself, which offers a dictionary-like API.
I'd expect us to pretty much always be slower, but also to be much easier to work with and to support a lot more out-of-the-box (e.g.
nested objects, buffering, and composition of data structures that may require more internal plumbing otherwise).


Project Evolution
-----------------

The signac project has evolved from being an open-source project mostly developed and managed by the Glotzer Group at the University of Michigan, to being supported by over 30 contributors and committers/maintainers on 3 continents and with over 55 citations from academic and government research labs and 12 talks at large scientific, Python, and data science conferences.
The growth in involvement with signac is the result of our focus on developing features based on user needs, as well as our efforts to transition signac users to signac contributors, through many initiatives in the past few years.
Through encouraging users to become contributors, we ensure that signac addresses real users' needs.

..
    TODO: mention GSoC

We have expanded signac's contributor involvement to outside of the University of Michigan through expanded use in diverse research groups (and through maintainers graduating and staying involved?), but more notably through the Google Summer of Code (GSoC) program.
Our experience from the GSoC led to a new committer (explained later in this section) and much work on some of the developments presented above, namely synced collections and aggregation.
To encourage code contributions from existing users, we maintain active support and discussion through Slack.
In addition, we have started hosting weekly "office hours" for in-person (virtual) introduction and contributions to the code base.
By pairing new contributors with experienced signac developers, we significantly reduce the knowledge barrier to joining a new project.
Office hours creating space for users to make contributions has also led to more features and documentation born directly out of user need.
Contributing to documentation has been a productive starting point for new users-turned-contributors, both for the users and the project, since it improves the users' familiarity with the API as well as addresses weak spots in the documentation more obvious to newer users.

We will share our project's experience in progressively revising project governance to catalyze sustained contributions of many kinds, adding more points of entry for learning about the project (Slack support, office hours), and participating in Google Summer of Code in 2020 as a NumFOCUS Affiliated Project.

In our growth with increasing contributors and users, we recognized a need to change our governance structure to make contributing easier and provide a clear organizational structure to the community.
We based our new model on the Meritocratic Governance Model and our manager roles on Numba Czars.
We decided on a four category system with maintainers, committers, contributors, and users.
Code review and PR merge responsibilities are granted to maintainers and committers, who are (self-)nominated and accepted by a vote of the project maintainers.
Contributors consist of all members of the community who have contributed in some way to the framework, which includes adding or refactoring code as well as filing issues and improving documentation.
Finally, users refer to all those who use signac in any capacity.

In addition, to avoid overloading our committers and maintainers, we added three rotating manager roles to our governance model that ensure project management goes smoothly: triage, community, and release.
These managers have specific rotation policies based on time (or release cycles).
The triage manager role rotates weekly and looks at new issues or pull requests and handles cleanup of outdated issues.
The community manager role rotates monthly and is in charge of meeting planning and outreach.
Lastly, the release manager rotates with each release cycle and is the primary decision maker for the timeline and feature scope of package releases.
This prevents burnout among our senior developers and provides a sense of ownership to a greater number of people, instead of relying on a "benevolent dictator/oligarchy for life" mode of project leadership.


Conclusions
-----------

From the birth of the signac framework to now, signac has grown in usability, performance, and use.
Since our last proceedings papers, we have added exciting new features, like groups, aggregates, and synced collections and learned how to better manage outreach and governance in a burgeoning scientific open-source project.
As maintainers and committers, we are looking to continue expanding the framework through user-oriented development and continued outreach to research fields that routinely have projects suited for signac.
For example, extensions into experimental research labs is currently being sought after with an aim to provide the strong data management and providence signac provides into experimentalist communities.

Getting signac
--------------

The signac framework is tested for Python 3.6+ and is compatible with Linux, macOS, and Windows.
The software is available under the BSD-3 Clause license.

To install, execute

.. code-block:: bash

    conda install -c conda-forge signac signac-flow signac-dashboard

or

.. code-block:: bash

    pip install signac signac-flow signac-dashboard

Source code is available on GitHub [#]_ [#]_ and documentation is hosted online by ReadTheDocs [#]_.

.. [#] https://github.com/glotzerlab/signac
.. [#] https://github.com/glotzerlab/signac-flow
.. [#] https://docs.signac.io/


Acknowledgments
---------------

All authors should check to be sure their acknowledgements are included! Karen will help with this for Glotzer peeps.

We would like to thank Kelly Wang for contributing the concept and content of Figure 1.
We would also like to thank NumFOCUS, whose staff have provided the signac project with helpful advice on open-source governance, project sustainability, and community outreach.
(Who/what else should we thank besides our respective funding sources / grants?)

B.D. is supported by a National Science Foundation Graduate Research Fellowship Grant DGE 1256260. (...)
