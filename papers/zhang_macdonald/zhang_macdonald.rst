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

Python Packaging
----------------

Python has a strong community, and many libraries and tools are hosted on the Python Package Index.
Currently, the standard tool for installing packages is [pip](https://pip.pypa.io/en/stable/),
which installs Python packages and their Python dependencies. For development, it is strongly
recommended to use pip with virtual environments [footnote 0]. Doing so will allow the developed
projects to use the newest stable versions of their dependencies, and well maintained dependencies
should work correctly together.

.. code-block:: bash

   $ mkvirtualenv venv-demo (venv-demo) $ pip install scipy

After development is complete and analysis begins, the need for reproducibility overtakes the for
keeping dependencies up to date. Though many projects strive to maintain backwards compatibility, a
researcher would not want to use numpy-1.13.1 for part of their analysis and numpy-1.14.2 for
another, the stakes are simply too high. At this point, users can “pin” their versions.

.. code-block:: bash

   $ workon venv-demo (venv-demo) $ pip freeze > scipy-requirements.txt

Pip can use [requirements files](https://pip.readthedocs.io/en/1.1/requirements.html) to achieve
more stability. Creating a requirements file in this way specifies the exact version of each
dependency.

.. code-block:: bash

   numpy==1.14.3 scipy==1.1.0

The requirements file can now be used to recreate the same environment using the same versions.

.. code-block:: bash

   $ mkvirtualenv separate-env (separate-env) $ pip install -r scipy-requirements.txt

For Python users who need to guarantee deterministic builds, another step is suggested. Adding
hashes to a requirements.txt provides the guarantee that the exact bits are installed. PyPI now
supports sha256, which is strongly recommended over md5, which has known vulnerabilities. Pip can
be used to calculate the hashes, which are then added to the requirements file.

.. code-block:: bash

   $ pip download numpy==1.14.3 Collecting numpy==1.14.3 Saved
   ./numpy-1.14.3-cp27-cp27mu-manylinux1_x86_64.whl Successfully downloaded numpy $ pip hash
   ./numpy-1.14.3-cp27-cp27mu-manylinux1_x86_64.whl ./numpy-1.14.3-cp27-cp27mu-manylinux1_x86_64.whl:
   --hash=sha256:0db6301324d0568089663ef2701ad90ebac0e975742c97460e89366692bd0563

Add these hashes to your requirements file, and use the `--require-hashes` option. Note that these
files are specific to architecture and python type. For code that should run in more than one
environment, multiple hashes can be specified.

.. code-block:: bash

   numpy==1.14.3 \ --hash=sha256:0db6301324d0568089663ef2701ad90ebac0e975742c97460e89366692bd0563
   scipy==1.1.0 \ --hash=sha256:08237eda23fd8e4e54838258b124f1cd141379a5f281b0a234ca99b38918c07a

.. code-block:: bash

   $ mkvirtualenv deterministic-venv (deterministic-venv) $ pip install --require-hashes -r
   scipy_requirements.txt

Guarantees:
- All Python dependencies installed this way will contain exactly the same bits Hashes
safeguard against man in the middle attacks
- Hashes safeguard against malicious modification of
packages on PyPI

Limitations: Packages on PyPI can be removed at any time by their maintainer TODO(hellllllllp, this
seems pretty good to meeeeee)

Pip was selected because it is the standard tool, and it is most likely to maintain backward
compatibility. However, there are other tools with rich feature sets that simplify the process. In
particular, [pipenv](https://docs.pipenv.org/) uses hashing and virtual environments by default for
a smooth experience.

[footnote 0] A virtual environment, often abbreviated “virtualenv” or “venv”, is an isolated python
environments that is used to prevent projects and their dependencies from interfering with with
each other. Under the hood, virtual environments work by managing the PYTHON_PATH (TODO: is this
the right var name?) Another benefit of virtual environments is that they do not require root
privileges and are safer to use.


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
