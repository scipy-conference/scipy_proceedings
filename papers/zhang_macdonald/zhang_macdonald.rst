:author: Bihan Zhang
:email: bihan.zh@gmail.com
:equal-contributor:
:bibliography: mybib


:author: Austin Macdonald
:email: austin@redhat.com
:institution: Red Hat
:equal-contributor:


--------------------------------------------------
Reproducible Environments for Reproducible Results
--------------------------------------------------

.. class:: abstract

   Trustworthy results require reproducibility. Publishing source code is
   necessary but not sufficient for complete reproducibility since complex
   programs often depend on external code. [Pulp](https://pulpproject.org/)
   is open source, written in python, and can be used to manage and maintain
   reproducible computational environments. Attendees of any skill level can
   come learn best practices for creating reproducible environments, and
   managing them with Pulp.


.. class:: keywords

   dependency-hell, reproducibility


Introduction
============

Trustworthy results require reproducibility. Publishing code is necessary
but not sufficient for complete reproducibility. Complex programs often depend
on external code. “An article […] in a scientific publication is not the
scholarship itself, it is merely advertising of the scholarship. The actual
scholarship is the complete software development environment and the complete
set of instructions which generated the figures”(Buckheit, Emphasis Added).

Fortunately, this is a common problem and there are a number of best practices
and tools that can make this easier. A common solution for high level
dependencies is to explicitly “pin” them (Wilson). In python this would be
running

.. code-block:: bash

   pip freeze > requirements.txt


But this only manages package dependencies and not system dependencies. And,
because these resources are user managed and exist on 3rd party platforms,
content can be modified or removed making it difficult or impossible to
guarantee reproducibility.

This paper seeks to explore several different methods at managing environmental
reproducibility, and introduces Pulp as a manager for various environments.


Measuring Reproducibility
=========================

Two factors have to be considered when we think about reproducibility.

Complete reproducibility is having the researcher and reviewer share identical
'bits' of the necessary system, program, and dependencies.

Vandewalle identifies several necessity for complete reproducibility [Vandewalle]:
the program's source code, package dependencies, system requirements and
configuration, data source used, and documentation on running the provided the source code.

On the other side one must determine if these programs and environments are flexible to change. Software moves fast, and even widely used programs become
legacy and eventually deprecated. Pinning dependencies might accelerate this process.

A good tool, and management system will balance complete reproducibility and code rot.

Tools
=====

Published Source Code
---------------------

Scholarly research containing descriptions of methodology is no longer sufficient.
For standalone scripts, publishing source code might be acceptable, But as computational systems grow more complex,
this method becomes more unreliable. Nontrivial research oftentimes depend on other external libraries for everything from left-padding
a line, to building scalable machine learning. This "has led to an ever larger and more complex
black box between what was actually done and what is described in literature." [Boettinger An intro to docker for reproducible research]

Publishing source code in papers makes them inflexible to change- bugs fixed after publication
cannot be communicated to the readers of the paper. Code is not versioned and even if the source code is updated and
made available it is hard to communicate what issues were fixed.

Git
----

Using an online git repository is a great way to keep track of source code [Good Enough Practices for Scientific Computing].
With git you can easily track changes you make to data and software. Git identifies commits by a unique hash, which can be used
to reference a specific point in the source code.

What git lacks is the ability to do environmental management.
Git is not a package manager. System dependencies in git can only be documented- and need the user to install them following instructions.
It is recommended that git be used to store the source code, and that some other package manager be used to manage the system environment.

Python
------

Ansible
-------

Ansible is an IT automation tool. It can configure systems, deploy software, and orchestrate more advanced tasks [ansible website]
With ansible it is possible to install python dependencies and system dependencies.

The approach is characterized by scripting, rather than documenting, a description of the necessary dependencies for software to run, usually from the Operating System [...] on up” [Clark berkley’s common scientific compute environments for research and education]


With ansible you write an ansible playbook that executes a set of tasks. Each task is idempotent.


.. code-block:: yaml

   - name: Install python3-virtualenvwrapper (Fedora)
     package:
     name:
       - which
       - python3-virtualenvwrapper
     when:
       - pulp_venv is defined
       - ansible_distribution == 'Fedora'

   - name: Create a virtualenv
     command: 'python3 -m venv my_venv'
     args:
       creates: 'my_venv'
     register: result

   - pip:
     name: scipy
     version: 1.1.0

   - dnf:


Ansible is only as good as your playbook. To make your environment reproducible, your playbook has to follow best practices like
pinning packages to a version. A default host OS also should be specified when the playbook is written: ansible uses separate plugins
to install system dependencies, and to be multiplatform the researcher needs to do some ansible host checking to use the right plugins.

Containers
----------

Containers* are a great way to publish and share a virtualized image of your system, source code, and data.

A Docker Image (a snapshot of a filesystem that is inert) can be passed to users through the centralized DockerHub. This image can contain
all system dependencies, a pre setup environment, and the source files and instructions.

It is recommended that a Dockerfile is used to create this image; while images can be created interatively through docker scripting tools, this process leaves little record
of what went into creating the image.

This Dockerfile can be kept in github, and linked to DockerHub so that the image is rebuilt with every change to the Dockerfile.

This is not a problem to immutable images- docker keeps track of each image with a hash, a publication should be referenced with the hash to make sure the correct version is obtained.

This example dockerfile creates an ubuntu image and installs tensorflow on it.

.. code-block:: text

   FROM ubuntu:16.04
   RUN pip --no-cache-dir install tensorflow


Note that while the Docker image is immutable, running `docker build` on the same Dockerfile does not gurantee an identical image. If tensorflow has been updated
since, the 2nd built image will have a newer version of tensorflow.

Once this image is built it can be pushed to DockerHub with

.. code-block:: bash

   docker build
   docker push


and shared with 3rd parties by providing them with the image id/hash and having them run:


.. code-block:: bash

   docker pull


Docker used to have a save to disk function, however that has been some issues with its deprecation in the past. Onereason
to use OCIcojntgainers instead of docker isaoaffuture proof

Docker is not concerned wiht breaking older specifciations; docker save

* Footnote: Most often people think of docker containers when the word container is mentioned. Docker is the most well known, however docker schema, and standards are not well documented.
Containers in this case can refer to Linux Container which is a superset of Docker Containers, Rkt, LXC, and other implementations. While most of the ideas discussed
 here will be generic across containers, the Docker Container, and DockerHub will be uesd as examples, due largely in part to their popularity.


Multi Environmental Management
==============================

Pulp
----

Artifactory
-----------

Summary
=======

Acknowledgements
================

References
==========
