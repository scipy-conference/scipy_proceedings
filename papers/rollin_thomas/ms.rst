:author: Rollin Thomas
:email: rcthomas@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0002-2834-4257
:corresponding:

:author: Laurie Stephey
:email: lastephey@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0003-3868-6178

:author: Annette Greiner
:email: amgreiner@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720
:orcid: 0000-0001-6465-7456

:author: Brandon Cook
:email: bgcook@lbl.gov
:institution: National Energy Research Scientific Computing Center,
              Lawrence Berkeley National Laboratory,
              1 Cyclotron Road MS59-4010A,
              Berkeley, California, 94720

:video: http://www.youtube.com/watch?v=dhRUe-gz690

=====================================================
Monitoring Scientific Python Usage on a Supercomputer
=====================================================

.. class:: abstract

   **Kind of a placeholder**
   In 2020, more than 35% of users at the National Energy Research Scientific
   Computing Center (NERSC) used Python on the Cori supercomputer.
   How do we know this?
   We developed a simple, minimally invasive monitoring framework that leverages
   standard Python features to capture Python imports and other job data.
   The data are analyzed with GPU-enabled Python libraries (Dask + cuDF) in a
   Jupyter notebook, and results are summarized in a Voila dashboard.
   After detailing our methodology, we provide a high-level tour of some of the
   data we’ve gathered over the past year.
   We conclude by outlining future work and potential broader applications.

.. class:: keywords

   keywords, procrastination

Introduction
============

..
   Why is the work important?

The National Energy Research Scientific Computing Center (NERSC_) is the primary
scientific computing facility for the US Department of Energy's Office of
Science.
Some 8,000 scientists use NERSC to perform basic, non-classified research in
predicting novel materials, modeling the Earth's climate, understanding the
evolution of the Universe, analyzing experimental particle physics data,
investigating protein structure, and much more.
NERSC procures and operates supercomputers and large-scale storage systems under
a strategy of balanced, timely introduction of new hardware and software
technologies to benefit the broadest possible subset of this workload.
Since any research project aligned with the mission of the Office of Science may
apply for access, NERSC's workload is diverse and demanding.
While procuring new systems or supporting users of existing ones, NERSC relies
on detailed analysis of its workload to help inform its strategy.

*Workload analysis* is the process of collecting and marshaling data to build a
picture of how applications and users really interact with and utilize systems.
It is one part of a procurement strategy that also includes surveys of user and
application requirements, emerging computer science research, developer or
vendor roadmaps, and technology trends.
Understanding our workload helps us engage in an informed way with stakeholders
like funding agencies, vendors, developers, users, standards bodies, and other
high-performance computing (HPC) centers.
Actively monitoring the workload enables us to identify suboptimal or
potentially problematic user practices and address them through direct
intervention, improving our documentation, or adjusting software deployment to
make it easier for users to use software in better ways.
Measuring the relative frequency of use of different software components can
help us optimize delivery of software, retiring less-utilized packages,
and promoting timely migration to newer versions.
Understanding which software packages are most useful to our users helps us
focus support, explore opportunities for collaborating with key software
developers and vendors, or at least advocate on our users' behalf to the right
people.
Detecting and analyzing trends in user behavior with software over time also
helps us anticipate user needs and respond to those needs proactively.
Comprehensive, quantitative workload analysis is a critical tool in keeping
NERSC a productive supercomputer center for science.

With Python assuming a key role in scientific computing, it makes sense to apply
workload analysis to Python in production settings like NERSC.
Once viewed in HPC circles as merely a cleaner alternative to Perl or Shell
scripting, Python has evolved into a robust platform for orchestrating
simulations, running complex data processing pipelines, managing artificial
intelligence workflows, visualizing massive data sets, and more.
Adapting workload analysis practices to scientific Python gives its community
the same data-driven leverage that other language communities in HPC now enjoy.
This article documents the approach to Python workload analysis we have taken at
NERSC and what we have learned from taking it.

In the next section we provide an overview of related work including existing
tools for workload data collection, management, and analysis.
In Methods, we describe an approach to Python-centric workload analysis that
uses built-in Python features to capture usage data, and a Jupyter-notebook
based workflow for exploring the data set and communicating what we discover.
Our results include high-level statements about what Python packages are used
most often and at what scale on Cori, but also some interesting deeper dives
into use of certain specific packages along with a few surprises.
In the Discussion, we follow-up on the results from the previous section, share
the pluses and minuses of our workflow, the lessons we learned in setting it up,
and outline plans for expanding the analysis to better fill out the picture of
Python at NERSC.
The Conclusion suggests areas for future work and includes an invitation to
developers to contact us about having their packages added to our list of
monitored scientific Python packages.

Related Work
============

..
   What is the context for the work?

The simplest approach that is actually used to get a sense of what applications
run on a supercomputer is to scan submitted batch job scripts for executable
names.
In the case of Python applications, this is problematic since some potentially
huge fraction of users will invoke Python scripts directly instead of as an
argument to the ``python`` executable.
This method also provides only a crude count of actual ``python`` invocations,
and gives little insight into deeper questions about Python packages, libraries,
or frameworks in use.

Software environment modules are a very common way for HPC centers to deliver
software to users [Fur91]_ [Mcl11]_.
Modules operate primarily by setting, modifying, or deleting environment
variables upon invocation of a module command (such as load, swap, or unload).
This provides an entrypoint for software usage monitoring: Staff can inject
code into a module load operation to record the name of the module being
loaded, its version, and other information about the user's environment.
Lmod documentation includes a guide on how to configure Lmod to use syslog and
MySQL to collect module loads through a hook function [lmod]_.
Counting module loads as a way to track Python usage is simple but has issues.
Users often include module load commands in their shell initialization/resource
files (e.g., `.bashrc`), meaning that shell invocation or mere user login may
trigger a detection even if the user never actually uses it.
Capturing information at the package, library, or framework level using module
counts would also require that individual packages be installed as separate
modules.
Module counts also miss Python usage outside of the module environment, such as
user-installed Python environments or stacks.

Tools like ALTD [Fah10]_ and XALT [Agr14]_ are commonly used in HPC contexts to
track library usage in compiled HPC applications.
The approach is to introduce wrappers that intercept and introduce operations at
link time and when the job runs the application via batch job launcher (e.g.
``srun`` in the case of Slurm).
At link time, wrappers can inject metadata into the executable header, take a
census of libraries being linked in, and forward that information to a file or
database for subsequent analysis.
At job launch, information stored in the header at link time can be dumped and
forwarded also.
On systems where all user applications are linked and launched with instrumented
wrappers, this approach yields a great deal of actionable information to HPC
center staff.
However, popular Python distributions such as Anaconda Python arrive on systems
fully built, and often are installed by users without assistance from center
staff.
Later versions of XALT can address this through an ``LD_PRELOAD`` environment
variable setting.
This enables XALT to identify compiled extensions that are imported in Python
programs using a non-instrumented Python, but pure Python libraries currently
are not detected.
XALT is an active project so this may be addressed in a future release.

In [Mac17]_ the author describes an approach based on instrumenting Python on
Blue Waters capture information about Python package using only native Python
built-in features: ``sitecustomize`` and ``atexit``.
During normal Python interpreter start-up, an attempt is made to import a module
named ``sitecustomize`` that has the ability to perform any site-specific
customizations it contains.
In this case, the injected code registers an exit handler through the ``atexit``
standard library module.
This exit handler inspects ``sys.modules`` which in normal circumstances
includes a list of all packages imported in the course of execution.
On Blue Waters, ``sitecustomize`` was installed into the Python distribution
installed and maintained by staff.
Collected information was stored to plain text log files on Blue Waters.
An advantage of this approach is that ``sitecustomize`` failures are nonfatal,
and and placing the import reporting step into an exit hook (as opposed to
instrumenting the ``import`` mechanism) means that it minimizes interference
with normal operation of the host application.

**Need a paragraph telling why we like this last method**

Methods
=======

..
   How was the work done?

* Goals
* Short background on OMNI, MODS
* Strategy of exit hook; potential shortcomings, implications and mitigations

  * Injection, but we do it from /opt (local to the node) and users can deactivate
  * Libraries monitored is a subset of the whole
  * Slurm may kill the job before it fires the exit hook
  * Mpi4py also: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
  * What if monitoring downstream fails (canary jobs)

* Explain the code
* Path we take from exit hook execution through syslog/kafka(?), elastic
* Talk about the analysis flow: Papermill, Dask, Jupyter, Voila
* Talk about how/why we choose these various pieces

There are a number of differences between our Python deployment and that on Blue
Waters.
NERSC provides Python to users through software environment modules.
Users can type ``module load python`` at the command line and the module system
will provide access to a staff-maintained Anaconda distribution.
Users also have the option of creating their own Conda environments, or even
installing their own Python from source or using the Anaconda/Miniconda
installer scripts.
The latter option is a common approach followed by projects, experiments, or
collaborations of users to manage a collaboration-wide Python software stack.
NERSC also provides a container runtime called Shifter that allows users to run
Docker images they build, and it is another popular way for individual users and
collaborations to use Python stacks.
This variability means that we need a different point of injection than on Blue
Waters.
We also deemed the use of plain text log files on platform storage to be
infeasible given the rate of Python jobs we would be monitoring.
The amount of data being gathered is consequential enough that we turned to the
Python data ecosystem to help us manage it and discuss our experiences with a
Jupyter notebook-based workflow for exploring the data.
We also talk about what kinds of things we learned from the data.

Customs: Inspect and Report Packages
------------------------------------

We call our package "Customs" since it is for inspecting and reporting on Python
package imports of particular interest.
Customs can be understood in terms of three very simple concepts.
A **Check** is a simple object that represents a Python package by its name and
a callable that is used to verify that the package is present in a dictionary.
In production this dictionary should be ``sys.modules`` but during testing it is
allowed to be a mock ``sys.modules`` dictionary.
The **Inspector** is a container of Check objects, and is responsible for
applying each Check to ``sys.modules`` (or mock) and returning the names of
packages that are detected.
Finally, the **Reporter** is an abstract class that takes some action given a
list of detected package names.
Reporter implementations should record or transmit the list of detected
packages, but exactly how this is done is up to the implementor.
Customs includes a few reference Reporter reference implementations and an
example of a custom Customs Reporter.

Generally, system administrators only interact with Customs through its primary
entry point, the function ``register_exit_hook``.
This function takes two arguments.
The first argument is a list of strings or tuples that are converted into
Checks.
The second argument is the type of Reporter to be used.
The exit hook can be registered multiple times with different package
specification lists or Reporters.

The intended pattern is that a system administrator will create a list of
package specifications they want to check for, select or implement an
appropriate Reporter, and pass these to ``register_exit_hook`` within
``sitecustomize.py`` and install the latter module into ``sys.path``.
When a user invokes Python, the exit hook will be registered using the
``atexit`` standard library module, the application proceeds as normal, and then
at shutdown ``sys.modules`` is inspected and detected packages of interest are
reported.

There are a few ways that an administrator may choose to deploy
``sitecustomize.py`` to ``sys.path``.
One way is to simply install it to ``{prefix}/lib/python3.X/site-packages``
and maintain it as part of the installation of Python.
A system administrator may also set a default ``PYTHONPATH`` for all users on
the system to broaden coverage to user-installed environments.
This has the advantage (or from some perspectives, disadvantage) of allowing
users to opt-out of data collection.
It is advised that system administrators take care that extending ``sys.path``
via ``PYTHONPATH`` be done in such a way that it does not harm performance at
start up.
For instance, installing the monitoring software to compute node images instead
of serving it over a distributed file system.

Message Logging and Storage
---------------------------

We send our messages to Elastic via nerscjson.

* LDMS, ask Taylor/Eric for ref and refs

Talk about LDMS, [Age14]_.

The Story: Prototyping, Production and Publication with Jupyter
---------------------------------------------------------------

.. epigraph::

    Data scientists are involved with gathering data, massaging it into a
    tractable form, making it tell its story, and presenting that story to
    others.

    -- Mike Loukides, `What is Data Science?
    <https://www.oreilly.com/radar/what-is-data-science/>`_

OMNI includes Kibana, a visualization interface that enables NERSC staff to
visualize indexed Elasticsearch data collected from NERSC systems, including
data collected for MODS.
The MODS team uses Kibana for creating plots of usage data, organizing these
into attractive dashboard displays that communicate MODS metrics at a high
level, at a glance.
Kibana is very effective at easily providing a high-level picture of MODS, but
the MODS team wanted deeper insights from the data and obtaining these through
Kibana presented some difficulty.
Given that the MODS team is fairly fluent in Python, and that NERSC provides
users (including staff) with a good Python ecosystem for data analytics, using
Python tools for understanding the data was a natural choice.
**So we figured out the toolchain we needed and here it is.**

Our first requirement was the ability to explore MODS Python data interactively
to prototype new analyses, but we wanted to be able to record that process,
document it, share it, and enable others to re-run or re-create the results.
Jupyter Notebooks specifically target this problem, and NERSC already runs a
user-facing JupyterHub service that enables access to Cori.
Members of the MODS team manage notebooks in a Gitlab instance managed by NERSC,
but can also share them with one another (and from Gitlab) using an NBViewer
service running alongside NERSC's JupyterHub.

Iterative prototyping of big data analysis pipelines often starts with testing
hypotheses or algorithms against a small subset of the data and then scaling
that analysis up to the entire data set.
The initial subset of the data used should be large and well-enough sampled to
avoid prematurely presenting an overly biased impression of the entire data set.
HPC hardware and software tools enable the prototyping phase to proceed with as
much data as possible.
Hardware is NVIDIA V100, A100 and to use the GPUs we used RAPIDS libraries like
cuDF and CuPy.
To scale up the analysis we use Dask.
Software stack is in Docker container.
**FIXME**

**FIXME**
We want to also be able to convert this exploratory phase into something we
can use in production, and it would be best not to have to start with a
Jupyter notebook for an analysis and then have to convert it to a script.
Making it possible to execute notebooks programmatically, on a scheduler
(using our batch scheduler), means Papermill.

**FIXME**
Finally, we want to be able to share the results of our analysis using
Python-backed dashboards.  For this we use Voila to run the notebooks generated
by Papermill in our container-as-a-service system Spin.
To avoid version compatibility problems within the Python stack used for the
analysis we use Docker containers.  At runtime the Docker containers are run
using Shifter, and in Spin they are just Docker containers managed by Rancher
2, orchestrated with Kubernetes.
We use cell notebook metadata to execute the Spin-appropriate cells and not the
Cori-appropriate ones in Spin.

Results
=======

..
   What were the results of the work?  What did we learn, discover, etc?

* Other alternatives analyses
* How hard was it to set up, experiment with, maintain
* May need to follow up with users
* Most jobs are one node
* Plotting/viz libraries rank higher than expected
* Even on our GPU system, there are lots of CPU imports (unclear how high GPU utilization really is)
* For Dask, users may be/sometimes unaware they are actually using it
* Multiprocessing use is really heavy
* Quantitative statements like
   * Top 10 libraries
   * Mean job size
   * Job size as a function of library
   * Correlated libraries and dependency patterns

Discussion
==========

..
   What do the results mean?  What are the implications and directions for future work?

* "Typical" Python user on our systems does what?
* Qualitative statements about our process and its refinement
* How did we proceed and are there things others could learn from it?
* Revisit limitations, implications, and mitigations

* Why do we do it this way?

  * Test dog food
  * Able to interact with the data using Python which allows more sophisticated analysis
  * Lends itself to a very appealing prototype-to-production flow

    * We make something that works
    * Show it to stakeholder, get feedback,
    * Iterate on the actual notebook in a job
    * Productionize that notebook without rewriting to scripts etc

Putting all the steps in the analysis (extraction, aggregation, indexing,
selecting, plotting) into one narrative greatly improves communication,
reasoning, iteration, and reproducibility.
Therefore, one of our objectives was to manage as much of the data analysis as
we could using one notebook per topic and make the notebook functional both as a
Jupyter document and as dashboard.
Using cell metadata helped us to manage both the computationally-intensive
"upstream" part of the notebook and the less expensive "downstream" dashboard
within a single notebook.
One disadvantage of this approach is that it is very easy to remove or forget to
apply cell tags.
Another is that some code, particularly package imports in one part of the
notebook need to be repeated in another.
These shortcomings could be addressed by making cell metadata easier to apply
and manage **see if there's a tool we should use already out there?**.
Oh could install the Voila extension for JupyterLab that may help.

The analysis part of a notebook is performed on a supercomputer, while the
dashboard runs on a separate container-as-a-service platform, but we were able
to use the notebooks in both cases and use the same exact containers whether
using Jupyter or Voila.
The reason for this is that while the runtime on Cori for containers is Shifter,
and Spin uses Kubernetes to orchestrate container-based services, they both take
Docker as input.
Some of our images were created using Podman, and others using Docker, it didn't
matter.
The Jupyter kernel, the Dask runtime in both places, all the exact same stack.

Conclusion
==========

..
   Summarize what was done, learned, and where to go next.

We have described how we characterize, as comprehensively as possible, the
Python workload on Cori.
We leverage Python's built-in ``sitecustomize`` loader, ``atexit`` module, and
``PYTHONPATH`` environment variable to instrument Python applications to detect
key package imports and gather runtime environment data.
This is implemented in a very simple Python package we have created and released
called ``customs`` that provides interfaces for and reference implementations of
the separate concerns of inspecting and reporting package detections.
Deploying this as part of Cori's node images and container runtime **???**
enables us to gather information on Python applications no matter how they are
installed.
Unsetting the default ``PYTHONPATH`` allows users to opt-out.
Collected data is transmitted to a central data store via syslog.
Finally, to understand the collected data, we use a PyData-centered workflow
that enables exploration, interactivity, prototyping, and report generation:

* **Jupyter Notebooks,** to interactively explore the data, iteratively
  prototype data analysis and visualizations, and arrange the information for
  reporting, all within a single document.
* **CuPy, cuDF** and other GPU-enabled Python libraries to accelerate
  computation within a single node.
* **Dask** to scale data transformations and analytics to multiple GPUs.
* **Papermill,** to automate extraction and transformation of the data as well as
  production runs of Notebooks in batch jobs on Cori.
* **Voila** to create responsive, interactive dashboards for both internal use
  by NERSC staff and management, but also to external stakeholders.

**Rephrase**
Putting all the steps in the analysis (extraction, aggregation, indexing,
selecting, plotting) into one narrative greatly improves communication,
reasoning, iteration, and reproducibility.

**Rephrase**
The analysis part of a notebook is performed on a supercomputer, while the
dashboard runs on a separate container-as-a-service platform, but we were able
to use the notebooks in both cases and use the same exact containers whether
using Jupyter or Voila.

We invite developers to suggest their packages.

In the future we would like to capture more than just the list of packages that
match our filter, being able to easily filter out standard library packages by
default as will be possible in Python 3.10 would help with this.
Part of the problem is the message transport layer.

* Future work includes watching users transition to new GPU-based system

  * Do these users run the same kind of workflow?
  * Do they change in response to the system change?

* More sophisticated, AI-based analysis and responses for further insights

  * Anomaly/problem detection and alert to us/user?

Acknowledgments
===============

This research used resources of the National Energy Research Scientific
Computing Center (NERSC), a U.S. Department of Energy Office of Science User
Facility located at Lawrence Berkeley National Laboratory, operated under
Contract No. DE-AC02-05CH11231.

References
==========

.. _NERSC: https://www.nersc.gov/about/

.. [Age14] A. Agelastos, B. Allan, J. Brandt, P. Cassella, J. Enos, J. Fullop,
           A. Gentile, S. Monk, N. Naksinehaboon, J. Ogden, M. Rajan, M. Showerman,
           J. Stevenson, N. Taerat, and T. Tucker
           *Lightweight Distributed Metric Service: A Scalable Infrastructure for 
           Continuous Monitoring of Large Scale Computing Systems and Applications*
           Proc. IEEE/ACM International Conference for High Performance Storage,
           Networking, and Analysis, SC14, New Orleans, LA, 2014.

.. [Agr14] K. Agrawal, M. R. Fahey, R. McLay, and D. James.
           *User Environment Tracking and Problem Detection with XALT*
           Proceedings of the First International Workshop on HPC User Support
           Tools, Piscataway, NJ, 2014.
           <http://doi.org/10.1109/HUST.2014.6>

.. [Fah10] M. Fahey, N Jones, and B. Hadri, 
           *The Automatic Library Tracking Database*
           Proceedings of the Cray User Group, Edinburgh, United Kingdom, 2010

.. [Fur91] J. L. Furlani, *Modules: Providing a Flexible User Environment*
           Proceedings of the Fifth Large Installation Systems Administration
           Conference (LISA V), San Diego, CA, 1991.

.. [Mac17] C. MacLean. *Python Usage Metrics on Blue Waters*
           Proceedings of the Cray User Group, Redmond, WA, 2017.

.. [Mcl11] R. McLay, K. W. Schulz, W. L. Barth, and T. Minyard, 
           *Best practices for the deployment and management of production HPC clusters*
           In State of the Practice Reports, SC11, Seattle, WA, <https://doi.acm.org/10.1145/2063348.2063360>

.. [lmod]  https://lmod.readthedocs.io/en/latest/300_tracking_module_usage.html

