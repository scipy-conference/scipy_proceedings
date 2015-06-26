:author: Faical Yannick Palingwende Congo
:email: yannick.congo@gmail.com
:institution: Blaise Pascal University

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------------------------
Building a Cloud Service for Reproducible Simulation Management
---------------------------------------------------------------

.. class:: abstract

   The notion of capturing each execution of a script and workflow and its
   associated metadata is enormously appealing and should be at the heart of
   any attempt to make scientific simulations repeatable and reproducible.

   Most of the work in the literature focus in the terminology and the
   approaches to acquire those metadata. Those are critical but not enough.
   Since the purpose of capturing the execution is to be able to recreate the
   same execution environment as in the original run, there is great need to
   investigate ways to recreate a similar environment from those metadata and
   also to be able to make them accessible to the community for collaboration.
   The so popular social collaborative *pull request* mechanism in Github is a
   great example of how cloud infrastructures can bring another layer of public
   collaboration. We think reproducibility could benefit from a cloud social
   collaborative presence because capturing the metadata about a simulation
   is far from being the end game of making it reproducible, repeatable or of
   any use to another scientist that has difficulties to easily have access to
   them.

   In this paper we define a reproducibility assessable record and the cloud
   infrastructure to support it. We also provide a use case example with the event
   based simulation management tool *Sumatra* and the container system *Docker*.

.. class:: keywords

   metadata, simulations, repeatable, reproducible, Sumatra, cloud, Docker.

Introduction
------------

Reproducibility in general is important because it is the cornerstone of
scientific advancement. Either done manually or automatically; reusability,
refutability and discovery are the key proprieties that make research results
repeatable and reproducible.

One will find that in the literature many research have been done in defining
the terminology (repeatability, reproducibility and replicability)
[SlezakWaczulikova2010]_ and investigating approaches regarding the recording
of simulations metadata using workflows [Oinn2000]_, libraries
[StephenLanger2010]_ or event control systems [GuoCDE2011]_). These research
are critical because they focus on getting to the point where the metadata
about a simulation execution has been captured in a qualitative and reliable
way. Yet the use of this metadata to recreate the proper execution environment
is challenging and is not only extremely valuable to the scientist that ran
the simulation. It is more valuable to other scientists that share the same
interest and need access to this metadata and an easy way to at least get the
same results consistently. This is why we think that reproducibility can
benefit from a more active presence in the cloud through infrastructures that
bring an easy access and collaboration around those captured metadata. The
social collaborative *pull request* mechanism from Github [MacDonnell2012]_ is
a great example about the importance of cloud infrastructures in enhancing
collaboration. In fact many scientific projects like SciPy [Oliver2013]_ got
some interest and contribution because of their exposure on Github and its
ease for collaboration.

In this paper we discuss on a structure of a reproducible
assessable record. It is a record that we propose to ease the reconstruction
of the execution environment and allow an easy assessment of the
reproducibility of a record from another by comparing their records. Then we
propose a cloud platform to deliver an online collaborative access around
these records. And finally we present an integration use case with the data
driven simulation management tool Sumatra [DavidsonSumatra2010]_.

A reproducible assessable record
--------------------------------

Defining what are the requirements that have to be recorded to better enforce
the reproducibility of a simulation is of good interest in the community. From
more general approaches like defining rules that have to be fulfilled
[Sandve2013]_, to more specific approaches [Heroux2011]_, we can define a set
of metadata that are useful to determine the reproducibility of a simulation.
To do so, we have to go from the fact that the execution of a simulation
involves mostly five different components: the source code or executable, the
input files, the output files, the dependencies and the hosting system. The
source code or executable gives all the information about what the simulation
is, where it can be found (repository) and how it was run. The input files are
all the files being loaded by the simulation during its execution. The output
files are all the files that the simulation produced during its execution. The
dependencies are all the libraries and tools that are needed by the simulation
to run. The hosting system is the system in which the simulation is being ran.
These components can be classified into two groups regarding repeatability as
a goal. To repeat a simulation execution, the source code and the inputs are
part of a group of components that are kept as the same. The dependencies and
the host system on the other end are part of the component that will most
likely change from the original executing system to another that attempt a
repeat. We think of them as a cause of uncertainties that lead to variations
in the outputs when the source code and inputs are still the same. To assess a
reproducibility property on a simulation, we provide the Table
:ref:`assesstable`. It defines the reproducibility properties involved
(repeatable, reproducible, non-repeatable, non-reproducible or unknown) when
comparing the source code, inputs and outputs of two simulations. This table
is  used in conjunction with the models presented later to assess the
reproducibility property of any record in the system compared to another
through a requesting mechanism that will be detailed further.

.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|r|}
     \hline
     \multirow{2}{*}{Output Files} & \multicolumn{4}{c|}{Source Code and Input Files}\tabularnewline
     \cline{2-5}
      & Same and Same & Same and Different & Different and Same & Different and Different\tabularnewline
     \hline
     Same  & Repeatable & Reproducible & Reproducible & Reproducible\tabularnewline
     \hline
     Different & non-repeatable & Unknown & Unknown & Unknown\tabularnewline
     \hline
     \end{longtable*}

     \caption{Reproducibility assessment based on source code, inputs and outputs \DUrole{label}{assesstable}}

   \end{table*}

One thing is to be able to gather crucial information about a simulation yet
another challenging one is to be able to recreate the same execution context
as when the simulation was done the first time. It is impossible to
consistently reproduce a simulation across platforms and machines if we do
not have an uniform and portable way to bundle the whole simulation execution
environment.

We think that containers based systems [Bottomley2014]_ are a possible
solution to ensure the consistency of the operating system and dependencies on
which the simulation runs. Building and sharing a container that
will deliver a runnable image in which the simulation execution is well scoped
and controlled will ensure that across machines and platforms we get closer to
a consistent execution environment [Melia2014]_.

Thus we propose here a container based recording system along with some
metadata as a set of four models that combined together should be enough to
deliver a reproducible simulation record storage. We show here the project
model in Table :ref:`projecttable`.

.. table:: Simulation metadata Project Model. :label:`projecttable`

   +--------------+-------------------------------------------+
   | Fields       | Descriptions                              |
   +==============+===========================================+
   | created      | string: simulation creation timestamp.    |
   +--------------+-------------------------------------------+
   | private      | boolean: false if project is public.      |
   +--------------+-------------------------------------------+
   | name         | string: project name.                     |
   +--------------+-------------------------------------------+
   | description  | string: full description of the project.  |
   +--------------+-------------------------------------------+
   | goals        | string: project goals.                    |
   +--------------+-------------------------------------------+
   | owner        | user: the creator of the project.         |
   +--------------+-------------------------------------------+
   | history      | list: container images list.              |
   +--------------+-------------------------------------------+

It describes the simulation and its *history*
field is the list of container images that have been built each time that the
project source code changes. The container is setup directly from the source
code of the simulation. We also propose a container model that is as simple as
shown in the Table :ref:`containertable`.

.. table:: Simulation metadata Container Model. :label:`containertable`

   +--------------+-------------------------------------------+
   | Fields       | Descriptions                              |
   +==============+===========================================+
   | created      | string: simulation creation timestamp.    |
   +--------------+-------------------------------------------+
   | system       | string: docker, rocket, ...               |
   +--------------+-------------------------------------------+
   | version      | dict: version control source code's tag . |
   +--------------+-------------------------------------------+
   | image        | string: path to the image in the cloud.   |
   +--------------+-------------------------------------------+

Based on the project's model in Table :ref:`projecttable`, we came up with a
record model shown in Table :ref:`recordtable`. A record is related to a
project and a container in the history of the project containers. When a
record is created, its container is the last container in the the project's
history at that time. Thus, a record that will be done on a modified project source code has
to be performed after the new container for this modified version of the
project get pushed to the history field. This way we ensure that two records
with different containers are from two different sources codes and also two records
with the same containers are from the same source code.

.. table:: Simulation metadata Record Model. :label:`recordtable`

   +--------------+-------------------------------------------+
   | Fields       | Descriptions                              |
   +==============+===========================================+
   | created      | string: execution creation timestamp.     |
   +--------------+-------------------------------------------+
   | updated      | string: execution update timestamp.       |
   +--------------+-------------------------------------------+
   | program      | dictionary: command, version control,...  |
   +--------------+-------------------------------------------+
   | inputs       | list: input files.                        |
   +--------------+-------------------------------------------+
   | outputs      | list: output files.                       |
   +--------------+-------------------------------------------+
   | dependencies | list: dependencies.                       |
   +--------------+-------------------------------------------+
   | status       | string: unknown, started, paused, ...     |
   +--------------+-------------------------------------------+
   | system       | dictionary: machine and os information.   |
   +--------------+-------------------------------------------+
   | project      | project: the simulation project.          |
   +--------------+-------------------------------------------+
   | image        | container: reference to the container.    |
   +--------------+-------------------------------------------+

A record reproducibility property assessment is done through a differentiation
process. A differentiation process is a process that allows the resolution of
a record reproducibility property compared to another. In this situation, the
two records are considered being from simulations that try to achieve the same
goals. It is quite hard to know at a high level standpoint if two records are
the same because it will most likely be a domain related decision that proves
that both records support the same claims. We focus here in an approach that
provides some basic differentiation methods and allow the definition of new
ones. Thus, the differentiation will most likely be based on the targeted
record owner domain knowledge and understanding on the method used. Since the
record is the state of a simulation execution, the inputs, outputs,
dependencies and system fields have to be provided every time because from a
run to another any of those may be subject to a change. Sometimes an action as
simple as upgrading a library can have terrible and not easy to determine
consequences on the outputs of another execution of the same simulation in the
same system.


A differentiation request or shortly *diff request* is the *contract* on which
the mechanism described before runs. A requesting record owner asks a targeted
record owner to validate a record reproducibility proposal from him. In this
mechanism, the requesting party has to define what the assessment is based on:
repeated, reproduced, non-reproduced and non-repeated. This party also has to
define the base differentiation method on which the assessment has been made:
default, visual and custom. A default differentiation method is a Leveinstein
distance [#]_ based diff on the text data. A visual one is a observation based
knowledge assessment. And custom is left to the requester to define and
propose to the targeted. It is important to point that the table
:ref:`assesstable` is the core scheme of comparison that all differentiation
request have to go through upon submission. To be accepted in the platform,
the *diff request* assessment has to comply with the content of that table. As
such a * diff request* for two requests that have different inputs contents
cannot be assessed as repeat compared to one another because an input
variation should lead to a reproducible assessment as pointed in the Table
:ref:`assesstable`. It is the generic The targeted record owner has then to
answer to the request by setting after verification on his side, the status of
the request to agreed or denied. By default the status value is *proposed*.
The table :ref:`requesttable` represents the fields that a diff request
contains. In fact one may say that in a model level a solved diff request is a
relationship of reproducibility assessment between two records.

.. [#] Levenshtein distance is a string metric for measuring the difference between two sequences.

.. table:: Simulation Record Differentiation Request Model. :label:`requesttable`

   +--------------+-------------------------------------------+
   | Fields       | Descriptions                              |
   +==============+===========================================+
   | created      | string: request creation timestamp.       |
   +--------------+-------------------------------------------+
   | sender       | user: responsible of the request.         |
   +--------------+-------------------------------------------+
   | toward       | record: targeted record.                  |
   +--------------+-------------------------------------------+
   | from         | record: requesting record.                |
   +--------------+-------------------------------------------+
   | diff         | dictionary: method of differentiation.    |
   +--------------+-------------------------------------------+
   | proposition  | string: repeated,reproduced,...           |
   +--------------+-------------------------------------------+
   | status       | string: agreed,denied,proposed.           |
   +--------------+-------------------------------------------+

A project reproducibility properties can be assessed from the differentiation requests
on its records. All the requests that have a status to *agreed* represent an accepted
couple of records that have been resolved as: repeated, reproduced, non-repeated and
non-reproduced.


Data Driven Cloud Service Platform
----------------------------------

.. figure:: figure0.png
   :align: center
   :figclass: w
   :scale: 60%

   Platform Architecture. :label:`paltformfig`

To support simulation management tools metadata, we propose a cloud
platform that implements the reproducible assessable record described
previously. This platform has two sides. As shown in the Figure
:ref:`paltformfig`, an API [#]_ access and a Web Frontend [#]_ access. These two
services communicate are linked to a MongoDB database [#]_ that
contains: the user accounts, the projects, the records, the containers and the
differentiation requests. We implemented some restrictions depending on the type
of access.

The API service exposes endpoints that are accessible by the
Simulation management tool from the executing machine. It is a token based
credential access that can be activated and renewed only from the Web Frontend
access. The API allows the Simulation Management tools to push, pull and
search projects and records. The API documentation will be available
publicly and will present the endpoints, HTTP [#]_ methods and the mandatory fields
in a structured JSON [#]_ format request content.

The Web Frontend service on the other end is controlled by the Cloud service.
The Cloud service is accessible only from the Web Frontend. Thus when the user
interacts with the Web Frontend, he is actually securely communicating with the
Cloud service. This strongly coupled design allows a flexible deployment and 
upgrades but at the same time harden the security of the platform. This frontend access
allows the user to manage his account and handle his API credentials which are used
by the Simulation Management tool to communicate with the platform.
It also allows the user to visualize his projects, records and requests. It is
the only place where the user can update some content regarding a project, record
or interact with his differentiation requests.

On the platform, the API is the only place where projects and records
are automatically created. On the Web side this is still possible but it is 
a manual process.

A Simulation tool that needs to interact with our platform has to follow the 
endpoints descriptions in Tables :ref:`projendtable` and :ref:`recoendtable`.

.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|r|}
     \hline
     \multirow{2}{*}{Endpoint} & \multicolumn{2}{c|}{Content}\tabularnewline
     \cline{2-3}
      & Method & Envelope\tabularnewline
     \hline
     $/api/v1/<api-token>/project/pull/<project-name>$  & GET & null. Note: pull metadata about the project.\tabularnewline
     \hline
     $/api/v1/<api-token>/project/push/<project-name>$ & POST & name, description, goal... custom. Note: push project metadata.\tabularnewline
     \hline
     \end{longtable*}

     \caption{REST Project endpoints \DUrole{label}{projendtable}}

   \end{table*}


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|r|}
     \hline
     \multirow{2}{*}{Endpoint} & \multicolumn{2}{c|}{Content}\tabularnewline
     \cline{2-3}
      & Method & Envelope\tabularnewline
     \hline
     \hline
     $/api/v1/<api-token>/record/push/<project-name>$ & POST & program, inputs, outputs... Note: push metadata about the record.\tabularnewline
     \hline
     $/api/v1/<api-token>/record/pull/<project-name>$ & GET & null. Note: pull the container.\tabularnewline
     \hline
     $/api/v1/<api-token>/record/display/<project-name>$ & GET & null. Note: metadata of the record.\tabularnewline
     \hline
     \end{longtable*}

     \caption{REST Record endpoints \DUrole{label}{recoendtable}}

   \end{table*}


.. [#] Application Programming Interface.
.. [#] Client browser access.
.. [#] An Agile, Scalable NoSQL Database: https://www.mongodb.org/ 
.. [#] HyperText Transfert Protocol. 
.. [#] A Data-Interchange format: http://json.org/ 


Integration with Sumatra and Use Case
-------------------------------------

*Sumatra Integration*

Sumatra is an open source event based simulation management tool.
To integrate our cloud API into Sumatra we briefly investigate
how Sumatra stores the metadata about a simulation.

To store records about simulations, Sumatra implements record stores. It also
has data stores that allow the storage of the simulation results. As of today,
Sumatra provides three data storage options:

.. raw:: latex

    \begin{itemize}
      \item FileSystemDataStore: It provides methods for accessing files stored on a local file system, under a given root directory.
      \item ArchivingFileSystemDataStore: It provides methods for accessing files written to a local file system then archived as .tar.gz.
      \item MirroredFileSystemDataStore: It provides methods for accessing files written to a local file system then mirrored to a web server.
    \end{itemize}

Sumatra also provides three ways of recording the simulation metadata:

.. raw:: latex

    \begin{itemize}
      \item ShelveRecordStore: It provides the Shelve based record storage.
      \item DjangoRecordStore: It provides the Django based record storage (if Django is installed).
      \item HttpRecordStore: It provides the HTTP based record storage.
    \end{itemize}

Regarding the visualization of the metadata from a simulation, Sumatra
provides a Django [#]_ tool named *smtweb*. It is a local web app that provides a
web view to the project folder from where it has been ran.
For a simulation management tool like Sumatra there are many advantages in
integrating a cloud platform into its record storage options:

.. [#] Python Web Framework: https://www.djangoproject.com/

.. raw:: latex

    \begin{itemize}
      \item Cloud Storage capability: When pushed to the cloud, the data is accessible from anywhere.
      \item Complexity reduction: There is no need for a local record viewer. The scientist can have access to his records anytime and anywhere.
      \item Discoverability enhancement: Everything about a simulation execution is a click away to being publicly shared.
    \end{itemize}

As presented in the list of record store options, Sumatra already has an HTTP
based record store available. Yet it does not suite the requirements of our
cloud platform. Firstly because there is no automatic mechanism to push the
data in the cloud. The MirroredFileSystemDataStore has to be fully done by the
user. Secondly we think there is need for more atomicity. In fact, Sumatra
gather the metadata about the execution and store it at the end of the
execution, which can have many disadvantages generally when the simulation
process dies or the Sumatra instance dies.

To integrate the cloud API and fully comply to the requirement cited before,
we had to implement and update some parts of the Sumatra source code:

.. raw:: latex

    \begin{itemize}
      \item DataStore: Currently the collect of newly created data happens at the end of the execution. This creates many issues regarding concurrent runs of the same projects because the same files are going to be manipulated. We are investigating two alternatives. The first is about running the simulation in a labeled working directory. This way, many runs can be done at the same time while having a private labeled space to write to. The second alternative consists of writing directly into the cloud. This will most likely break the already implemented data and record store paradigm in Sumatra.
      \item RecordStore: We make the point that the simulation management tool is the one that should comply to as many API interfaces as possible to give the user as many interoperability as possible with cloud platforms that support reproducible records. Thus, we intend to provide a total new record store that will fully integrate our API into Sumatra.
      \item Recording Mechanism: In Sumatra the knowledge of the final result of the execution combined with atomic state monitoring of the process will allow us to have a dynamic state of the execution. We want to make Sumatra record
      creation a dynamic many
       points recorder. In addition to an active monitoring, this feature allows the scientist to have basic informations about its runs may they crash or not. 
    \end{itemize}

*Reproducibility instrumentation with Sumatra*

The Sumatra repository [#]_ provides three test example projects. Our
instrumentation demo is based on the python one. This is the demo skeleton
model that we propose as a base line to make your simulation comply with the
principles described here. We are working on adding new tools and examples.

.. [#] https://github.com/open-research/sumatra.git

The demo is the encapsulation of the execution of a python simulation code
main.py with some parameter files. The instrumented project is organized as
following:

.. raw:: latex

    \begin{itemize}
      \item Python main: It's the simulation main source code.
      \item Git ignore: It contains the files that will not be versioned by git.
      \item Requirements: It contains all the python requirements needed by the simulation.
      \item Dockerfile: It contains the simulation docker container setup.
      \item Manage files: It's a script that allows the researcher to manage the container builds and
      the simulation executions.
    \end{itemize}

To instrument a simulation, the researcher has to follow some few steps:

.. raw:: latex

    \begin{itemize}
      \item Source code: The scientist may remove the script main.py and include his source code.
      \item Requirements: The scientist may provide the python libraries used by the simulation there.
      \item Dockerfile: This file contains sections that needs to be updated by the scientists such as: the git global parameters and the simulation name at smt init.
      \item Management: In the manage scripts, the researcher has to update the mapping data folder with docker. For example in the default case we are mapping the default.param file that is needed by the simulation.
    \end{itemize}

In addition, it is important that the scientist build the container every time
that the source changes as explained before when presenting the record model.
In this case a newly exported image will be available to be ran with Sumatra.
After a build, a run will execute the simulation and create the associated
record that will be pushed to our cloud API. The interesting part of such a
design is that the record image can be ran by any other scientist with the
possibility to change the input data. This allow reproducibility at an input
data level. For source code level modification, the other scientist has to
recreate an instrumented project. In the manage script, an API token is
required to be able access our cloud API. The researcher will have to put his
own. A further detailed documentation will be provided as soon as Sumatra is
fully integrated to our cloud infrastructure. The source code of the demo can
be found in the Github SciPy proceeding repository [#]_ under the 2015 branch
named *demo-sumatra*.It has been tested on an Ubuntu 15.04 machine and will
work on any Linux or OsX machine that has docker installed.

.. [#] https://github.com/faical-yannick-congo/scipy_proceedings.git


Conclusion and Perspective
--------------------------

Scientific computational experiments through simulation is getting more
support to enhance the reproducibility of research results. Execution metadata
recording systems through event control, workflows and libraries are the
approaches that are investigated and quite a good number of softwares and
tools implement them. Yet the aspect of having of these results discoverable
in a reproducible manner is still an unfulfilled need. This paper proposes a
container based reproducible record and the cloud platform to support it. The
cloud platform provides an API that can easily be integrated to the existing
Data Driven Simulation Management tools and allow: reproducibility
assessments, world wide web discoverable and sharing. We described an
integration use case with Sumatra and explained how beneficial and useful it
is for Sumatra users to link our cloud API to their Sumatra tool. This
platform main focus is to provide standard and generic ways for scientists to
make some differentiation procedures that will allow them to assess if a
simulation is repeatable, reproducible, non-repeatable, non-reproducible  or
if its an ongoing research. A differentiation request description has been
provided and is a sort of an hand shake between researchers regarding the
result of simulation runs. One can request a reproducibility assessment
property validation from a record against another.

We are under integration investigation for other simulation management tools
used in the community. In the short term this platform will hopefully be where
researchers could clone the entire execution environment that another
researcher did. And from there be able to verify the claims of the project and
investigate other execution on different data. The container based record
described, we hope, will allow a better standard environment control across
repeats and reproductions, which is a very hard battle currently for all
simulation management tools. Operating systems, compilers and dependencies
variations are the nightmare of reproducibility tools because the information
is usually not fully accessible and recreating the appropriate environment is
not an easy straight forward task.
 

References
----------

.. [SlezakWaczulikova2010] P. Slezák and I. Waczulíková. *Reproducibility and Repeatability*,
        Comenius University, July 2010.

.. [Oinn2000] Tom Oinn et al. *Taverna: Lessons in creating a workflow environment for the life sciences*, 
       Concurrency Computation, p. 2, September 2002

.. [StephenLanger2010] Stephen Langer et al. *gtklogger: A Tool For Systematically Testing Graphical User Interfaces*,
        NIST Internal Publication, pp. 2-3, October 2014.

.. [GuoCDE2011] Philip Guo. *CDE: A Tool for Creating Portable Experimental Software Packages*,
       Reproducible Research For Scientific Computing, pp. 2-3, October 2012.

.. [MacDonnell2012] John MacDonnell. *Git for Scientists: A Tutorial*,
       July 2012.

.. [Oliver2013] Marc Oliver. *Introduction to the Scipy Stack – Scientific Computing Tools for Python*,
       Jacobs University, November 2013.

.. [DavidsonSumatra2010] Andrew Davidson. *Automated tracking of computational experiments using Sumatra*,
       EuroSciPy 2010, Paris.

.. .. [Goodman2013] Alyssa Goodman. *10 Simple Rules for the Care and Feeding of Scientific Data*,
..         Harvard University Seminar – What to Keep and How to Analyze It: Data Curation and Data Analysis with Multiple Phases, May 2013.

.. [Sandve2013] Sandve GK et al. *Ten Simple Rules for Reproducible Computational Research.*,
        PLoS Comput Biol, October 2013.

.. [Heroux2011] Michael A. Heroux. *Improving CSE Software through Reproducibility Requirements*,
       Sandia National Laboratories, revised May 2011.

.. [Bottomley2014] James Bottomley. *What is All the Container Hype?*,
        Linux Foundation, p. 2, April 2014.

.. [Melia2014] Ivan Melia et al. *Linux Containers: Why They are in Your Future and What Has to Happen First*,
       Cisco and RedHat, p.7, September 2014