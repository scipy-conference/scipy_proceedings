:author: Faical Yannick Palingwende Congo
:email: yannick.congo@gmail.com
:institution: Blaise Pascal University

:video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------------------------
Building a Cloud Service for Reproducible Simulation Management
---------------------------------------------------------------

.. class:: abstract

   The notion of capturing each execution of a script and workflow and its
   associated metadata is enormously appealing and should be at the heart of
   any attempt to make scientific simulations repeatable and reproducible.

   Most of the work in the literature focus on the reproducibility
   requirements and the tools to acquire those metadata. Those are critical
   but there is also a great need to support the discoverabiliy of the
   metadata produced and also to investigate on the content of what a
   reproducible simulation execution context is.

   In this paper we propose our investigation results into defining a
   reproducibility assessable record and the cloud infrastructure to support
   it. A use case example with Sumatra and docker is provided.

.. class:: keywords

   metadata, simulations, repeatable, reproducible, Sumatra, cloud.

Introduction
------------

Reproducibility in general is important because it is the cornerstone of
scientific advancement. Either done manually or automatically; reusability,
refutability and discovery are the key proprieties that make research results
repeatable and reproducible.

While most of the literature focus on the terms (repeatable, reproducible and
replicable) [SlezakWaczulikova2010]_ and the techniques of recording the
simulation metadata (workflow [Oinn2000]_, library [StephenLanger2010]_, event
control [GuoCDE2011]_), there are less contributions into cloud
infrastructures to support these data. We think that reproducibility at its
current state is lacking a data driven presence in the cloud. This is a
problem. And we think there must be a focus on defining what is reproducible
record and how to support it in the cloud.  Version control is the proper
example that we can refer to show what is the impact of the cloud in
collaboration based tools. Github [MacDonnell2012]_ and Bitbucket
[Delorenzo2015]_ are certainly the big names in the cloud arena for version
control and many scientific project like SciPy [Oliver2013]_ got some interest
and contribution because of their exposure on Github.

In this paper we will discuss on the structure of a reproducibility
assessable record structure. Then we will present the proposed a cloud
platform to support data driven simulation management tools. And finally we
present an integration use case with the Data driven simulation management
tool Sumatra [DavidsonSumatra2010]_.

A reproducible assessable record
--------------------------------

Defining what are the requirements that have to be recorded to better enforce
the reproducibility of a simulation is a well known interest in the community.
From more general approaches like defining rules that has to be
fulfilled [Sandve2013]_, to more specific approaches [Heroux2011]_, we can
define a set of data that are important to assess the reproducibility of a
simulation.

The execution of a simulation involves five different components: The source
code or executable, the input files, the output files, the dependencies and
the hosting system. The source code or executable gives all the information
about what the simulation is, where it can be found and how it was run. The
input files represent all the files being loaded by the simulation during its
execution. The output files represent all the files that the simulation
produced during its execution. The dependencies are all the libraries and
tools that are needed by the simulation to run. The hosting system is the
system in which the simulation is being ran. These components can be
classified in two groups regarding repeatability as a goal. When trying to
repeat a simulation, the obvious variable components are: the dependencies and
the hosting system. We think of them as a cause of uncertainty in altering the
output files when the source code and inputs are still the same. To asses a
reproducibility property on a simulation as being repeatable, reproducible,
non-repeatable, non-reproducible and unknown, we provide the Table
:ref:`assesstable`. It consists of comparing two simulations by taking The
source code, inputs and outputs values of records and assessing the
reproducibility property of one to another.

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

One thing is to be able to gather crutial information about a simulation yet
another challenging one is to be able to recreate the exact execution context
as when the simulation was done. Sometimes an action as simple as upgrading a
library can have terrible and not easy to determine consequences. It is
impossible to consistently reproduce a simulation accross platforms and
machines if we do not have a uniform and portable way of constraining the
whole simulation execution.

We think that containers based systems [Bottomley2014]_ are a possible
solution to ensure the consistency of the operating system and dependencies on
which the simulation runs. Building and sharing a container that
will deliver a runnable image in which the simulation execution is well scoped
and controlled will ensure that across machines and platforms we get closer to
a consistent execution environment [Melia2014]_.

Thus we propose here a container based record allong with some metadata as a
set of four models that combined together should be necessary to deliver a
reproducible simulation record storage. We show here a project model in Table
:ref:`projecttable`.

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

It's the structure that describes the simulation project and its *history*
field is the list of container images that have been built each time that the
project source code changed. The container is setup directly from the source
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

Based on the project's model in Table :ref:`assesstable`, we came up with a
record model shown in Table :ref:`recordtable`. A record is related to a
project and a container in the history of the project containers. When a
record is created its container is the last container in the the project's
history. Thus a record that will be done on a modified project source code has
to be performed after the new container for this modified version of the
project be pushed to the history field. This way we ensure that two records
with different containers are from different source codes and also two records
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

Compared to a project, a record reproducibility assessment is done through a
differentiation process. A differentiation process is a procedure that allows
the resolution of a record reproducibility property compared to another. In
this case, the two records are considered being from simulations that try to
achieve the same goals. It is quite hard to assess at a high level standpoint
if two records are the same because it will most likely be a domain related
decision that proves that both records support the same claims. We focus here
in an approach that provides basic differentiation methods and allow the
definition of new ones. Thus, the differentiation will most likely be based
on the targeted record owner domain knowledge and understanding on the method
used.


A differentiation request or shortly *diff request* is the *contract* on which
the mechanism described before runs. A requesting record owner asks a targeted
record owner to validate a record reproducibility proposal from him. In this
mechanism, the requesting party has to define what the assessment is based on:
repeated, reproduced, non-reproduced and non-repeated. This party also has to
define the base differentiation method on which the assessment has been made:
default, visual, custom. A default differentiation method is a Leveinstein
based diff on the text data. A visual one is a observation based knowledge
assessment. And custom is left to the requester to define and propose to the
targeted. The targeted record owner has then to answer to the request by
setting after verification on his side, the status of the request to agreed or denied. By
default the status value is *proposed*. The table :ref:`requesttable` represents
the fields that a diff request should contain. In fact one may say that in a
model level a solved diff request is a relationship of reproducibility
assessment between two records.

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

Differention requests on a project's records allow queries 


A project reproducibility properties can be assessed from the differentiation requests
on its records. All the request that have a status to agreed concern an accepted
couple of records that have been resolved as: repeated, reproduced, non-repeated and
non-reproduced.


Data Driven Cloud Service Platform
----------------------------------

.. figure:: figure0.png
   :align: center
   :figclass: w
   :scale: 60%

   Platform Architecture. :label:`paltformfig`

To support simulation management tools metadata, we are proposing a cloud
platform that implements the reproducible assessable record described
previously. This platform has two sides. As shown in the Figure
:ref:`paltformfig`, a API[#]_ access and a Web Frontend[#]_ access. All those two
accesses communicate at the most end with a MongoDB database [#]_ that
contains: the user accounts, the projects, the records, the containers and the
differentiation requests. We implemented some restrictions depending on the type
of access.

The API service exposes endpoints that are accessible by the
Simulation management tool from the executing machine. It is a token based
credential access that can be activated and renewed only from the Web Frontend
access. The API allows the Simulation Management tools to push, pull and
search for projects and records. The API documentation will be available
publicly and will present the endpoints, HTTP[#]_ methods and the mandatory fields
in the structured JSON[#]_ format request content.

The Web Frontend service on the other end is controlled by the Cloud service.
The Cloud service is accessible only from the Web Frontend. Thus when the user
interacts with the Web Frontend, he is actually securely interacting with the
Cloud service. This strongly coupled design allows a flexible deployment and 
upgrades but at the same time harden the security of the platform. This access
allows the user to manage his account, handle his API credentials that are used
by the Simulation Management tool to communicate with the API interfaces.
It also allows the user to visualize his projects, records and requests. It is
the only place where the user can update some content. 

On the platform, the API is the only place where projects and records
are automatically created. On the Web side this is still possible but it is 
a manual process. Differentiation requests on the other end can only be created
and resolved from the Web Frontend access.

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
     $/api/v1/<api-token>/project/pull/<project-name>$  & GET & null\tabularnewline
     \hline
     $/api/v1/<api-token>/project/push/<project-name>$ & POST & name, description, goal and custom\tabularnewline
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
     $/api/v1/<api-token>/record/push/<project-name>$ & POST & program, inputs, outputs, dependencies, system and custom\tabularnewline
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
To integrate our cloud API into Sumatra we have to briefly investigate
how Sumatra stores the metadata that it records.

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
provides a Django[#]_ tool namely smtweb. It is a local web app that provides a
web view to the project folder that it has been run from within.
For a simulation management tool like Sumatra there are many advantages in
integrating a cloud platform into its record storage options. We can cite:

.. [#] Python Web Framework: https://www.djangoproject.com/

.. raw:: latex

    \begin{itemize}
      \item Local Storage irrelevance: There is no need to store the data locally they can be pushed to the cloud.
      \item Complexity reduction: There is no need for a local record viewer. The scientist can have access to his records anytime and anywhere.
      \item Discoverability enhancement: Everything about a simulation execution is a click away to be publicly shared.
      \item Better scope: The team can fully focus on improving the event control based recording process.
    \end{itemize}

As presented in the list of record store options, Sumatra already has a HTTP
based record store available. Yet it does not suite the requirements of our
cloud platform. Firstly because there is no automatic mechanism to push the
data in the cloud. The MirroredFileSystemDataStore has to be fully done by
user. Secondly we think there is need for more atomicity. In fact, Sumatra
gather the metadata about the execution and store it at the end of the
execution, which can have many disadvantages generally when the simulation
process dies or the Sumatra instance dies.

To integrate the cloud API and fully comply to the requirement cited before,
we had to implement and update some parts of the Sumatra source code:

.. raw:: latex

    \begin{itemize}
      \item DataStore: Currently the collect of newly created data happens a the end of the execution. This creates many issues regarding concurrent runs of the same projects because the same files are going to be manipulated. We are investigating two alternatives. The first is about running the simulation in a labeled working directory. This way many runs can be done at the same time while having a private labeled space to write to. The second alternative consists of writing directly into the cloud. This will most likely break the already implemented data and record store paradigm in Sumatra.
      \item RecordStore: We make the point that the simulation management tool is the one that should comply to as many API interfaces as possible to give the user as many interoperability as possible with cloud platforms that support reproducible records. Thus, we intend to provide a total new record store that will fully integrate our API into Sumatra.
      \item Recording Mechanism: In Sumatra the knowledge of the final result of the execution combined with atomic state monitoring of the process will allow us to have a live state of the execution. We are modifying the source code so that this information along with any information that is available be pushed on the go. An update endpoint on a record will be available to allow this. We want to make Sumatra record
      creation a dynamic 'on the time available data' recorder. In addition to a live monitoring, this case allows the scientist to have a basic information about its runs may they crash or not. 
    \end{itemize}

*Reproducibility instrumentation with Sumatra*

The Sumatra repository[#]_ provides three test example projects. Our
instrumentation demo is based on the python one. This is the demo skeleton
model that we propose as a base line to make your simulation comply with the
principles described here. This one is for Sumatra users and we are working on
providing alternatives.

.. [#] https://github.com/open-research/sumatra.git

The demo is the encapsulation of the execution of a python simulation code
main.py with some parameter files. The instrumented project is organized as
following:

.. raw:: latex

    \begin{itemize}
      \item Python main: It's the simulation main source code.
      \item Git ignore: It contains the files that will not be versioned.
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
data level. For source level modification, the other scientist has to recreate
an instrumented project. In the manage script an API token is required to be
able access our cloud API. The researcher will have to put his own.A further
detailed documentation will be provided as soon as Sumatra is integrated to
our cloud infrastructure. The source code of the demo can be found in my
Github SciPy proceeding repository [#]_ under the 2015 branch named *demo-
sumatra*. It has been tested on an Ubuntu 15.04 machine and will work on any
Linux or OsX machine that has docker installed.

.. [#] https://github.com/faical-yannick-congo/scipy_proceedings.git


Conclusion and Perspective
--------------------------

Scientific computational experiments through simulation is getting more
support to enhance the reproducibility of the produced research results.
Execution metadata recording systems through event control, workflows and
libraries are the approaches that are investigated and quite a good number of
software and tools implement them. Yet the aspect of the discoverability of
these results in a reproducible manner is still an unfulfilled need. This
paper proposes a container based reproducible record and the cloud platform to
support it. The cloud platform provide an API that can easily be integrated to
the existing Data Driven Simulation Management tools and allow:
reproducibility assessments, world wide web discoverability and sharing. We
described an integration use case with Sumatra and explained how beneficial
and useful it is for a Sumatra user to link our cloud API account to the
Sumatra tool. This platform main focus is to provide standard and generic ways
for scientists to make some differentiation procedures that will allow them to
assess if a simulation is repeatable, reproducible, non-repeatable, non-
reproducible or if its an ongoing research. Some metrics have been provided to
determine the degree of those properties from the atomic records during the
executions of the simulation. A differentiation request description has been
provided and is a sort of hand shake between researchers regarding the result
of simulation runs. One can request a reproducibility assessment property
validation from a record against another one.

We are under integration investigation for other simulation management tools
used in the community. In the short term this platform will hopefully be where
researchers could clone the entire execution environment that another
researcher did. And from there be able to verify the claims of the project and
investigate other execution on different data. The container based record
described, we hope, will allow a better standard environment control across
repeats and reproductions, which is a very hard battle currently for all
simulation management tools. Operating Systems, Compilers and Dependencies
variations are the nightmare of reproducibility tools because the information
is not fully accessible and there is not usually an easy way to recreate the
appropriate environment.
 

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

.. [Delorenzo2015] Ike DeLorenzo. *Coding in the cloud with Bitbucket*,
       Frebruary 2015.

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