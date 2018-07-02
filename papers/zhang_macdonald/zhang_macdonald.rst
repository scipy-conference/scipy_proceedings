:author: Bihan Zhang
:email: bihan.zh@gmail.com
:institution: Red Hat
:equal-contributor:

:author: Austin Macdonald
:email: austin@redhat.com
:institution: Red Hat
:equal-contributor:

:bibliography: mybib

--------------------------------------------------
Reproducible Environments for Reproducible Results
--------------------------------------------------

.. class:: abstract

   Trustworthy results require reproducibility, which must include the full computational
   environment used to produce and analyze the data. Creating perfectly reproducible
   environments is difficult because of fundamental challenges in the software packaging
   domain, and understanding these problems is necessary to mitigate them. Focusing on the Python
   ecosystem and a univeral package manager called Pulp_, this paper explores various approaches
   to software packaging and distribution from the angle of reproducibility.


.. class:: keywords

   dependency-hell, reproducibility, packaging


Introduction
============

Reliability of research, and therefore reproducibility, is the backbone of cummulative knowledge,
and it has been identified as a bottleneck for efficient scientific progress.[citation needed] Open
data, reproducibility incentives, and comprehensive experimental procedures are all key areas for
improvement[citation], but the subtle problem of environmental reproducibility is in need of more
attention. Because software is an integral part of modern science, software distrubution and its
fundamental problems of environmental differences and code entropy impact researchers directly.
Fortunately for scientists, software development as a whole has comparable needs. For each software
ecosystem, tools and best practice workflows have been developed to encourage rapid development,
simplify maintainance, and improve long term stability.  This paper will demonstrate how to apply
software industry tools and workflows to research code, focusing on the Pulp project and the Python
ecosystem.

Software Packaging Basics
=========================

Software packaging and distribution is an inherrently complex problem domain because software is
dependent on other software. Ensuring that code can be installed with compatible requirements is a
fundamental problem in the packaging domain. The needs of users, developers, and administrators
frequently interfere, leading to increasingly complex solutions. Various communities (often formed
around a single programming language) address the problem with their own tool set and workflows, each
optimizing for a different balance of needs. Rapid evolution within these communities can increase
the already significant learning curve, and it can be difficult to keep up, especially for those
with an unrelated primary interest.

Broadly, there have been two high level approaches to packaging, and comparing them is useful to
demonstrate the challenges to improving environmental reproducibility (TODO define?). The simplest
approach, (which will be refered to as "monolithic"), is to bundle code with all dependencies and
distribute it together as a single software package. This pattern is most common for user-centric
platforms like Windows and OSX. Monolithic packages are simple to install, but can sacrifice
transparency and flexibility for initial user convenience. The administrators and the developers
become responsible for maintanance of the whole environment. For cross-platform applications,
packages may need to be built for every combination of hardwares, operating systems, and language
versions. This process can be difficult and time consuming, leading to long development cycles and
infrequent updates. This pattern is leads to chronicly outdated software, and when security is an
important consideration, it can be costly to compensate.

A different approach commonly used by Linux communities is called the "modular" approach. Code is
packaged independently and any dependencies are resolved at installation time, usually with the
aid of a smart package manager. This pattern allows each component to be maintained and updated on
its own schedule, and the entire ecosystem tends to stay current. Project maintainers do not need
to be directly involved in every dependency update, but the responsibility of the user is
increased. Especially for software with complex dependency trees, this pattern can lead to
dependency conflicts which can be notoriously difficult to resolve. Some communities, (e.g. Fedora)
offer an additional component to address this problem, curated repositories. Curated repositories
can be thought of as a hybrid approach; the community as a whole maintains the interoperability of the
software in the repository, releaving some of the user respoonsibility. However, this
option also has problems; it can be less flexible because projects need to use dependency versions
that are available in specific repositories. This can be particularly troublesome for developers
when different repositories have different major versions of software. [seealso: semver] Because
code is run with different dependency versions in different environments, reproducibility is hard
to guarantee, particularly across platforms.

Python Packaging Ecosystem
==========================

Python has a strong community that facilitates packaging with many open source tools. The Python
Package Index (PyPI) is a community repository that allows package authors to contribute their
work and users to find and install whatever packages they need. Currently, the standard tool to
install Python software from PyPI is pip_, which also installs any dependencies. PyPI is not a
curated repository like Fedora, so dependency problems can be frequent. To address this, it is
generally recommended by the community to use virtual environments [0]_. A "virtualenv" is used to
to isolate installed Python projects (along with their dependencies), which allows the user to
install software which might otherwise have conflicts. Additionally, each virtualenv can use a
different Python version, which frees users from the restrictions of system-wide installations.

The current best practice for indicating dependencies is to list them in a `requirements.txt` [TODO
link] file, which is used by pip or comparable tools at installation time. This file offers some
powerful options to promote stability, and if used carefully, it can be used to ensure a degree
of reproducibility. Each requirement specifier can be "pinned" to an exact version (or even hash)
of a dependency. If more flexibility is needed, `requirements.txt` can specify minimum versions, or
even be configured to install the most recent version that maintains backwards compatibility.

During development, keeping dependencies flexible is ideal because the latest versions of a
dependency will have a longer life before they become obsolete. However, after development is
complete and final analysis begins, the need for reproducibility overtakes the benefits of
keeping dependencies up to date. Semantic versioning is considered a best practice in the Python
community, and uses X.Y.Z version numbers. Even for backwards compatible release (Y or Z releases),
a researcher would not want to use numpy-1.13.1 for part of their analysis and numpy-1.14.2 for
another. The risk that subtle changes could affect the results is too great, so it is recommended
that researchers pin their versions.


Published Source Code
---------------------

Scholarly research containing descriptions of methodology is no longer sufficient. For standalone
scripts, publishing source code might be acceptable, but as computational systems grow more
complex, this method becomes more unreliable. Nontrivial research often depends on other external
libraries for everything from left-padding a line to a framework for scalable machine learning.
This "has led to an ever larger and more complex black box between what was actually done and what
is described in literature." :cite:`Boettiger`

Using an online git repository is a great way to keep track of source code :cite:`Wilson`.  With
git you can easily track changes to data and software. Git identifies commits by a unique hash,
which can be used to reference a specific point in the source code, practically guaranteeing the
correct bits. This approach is incomplete however, because git lacks the ability to do
environmental management, it is not a package manager. System dependencies in git can only be
documented-- it is the responsibility of the user to determine if the entire environment is
identical, and the documentation may not contain enough information to verify. Instead, we
recommend using git to store source code in addition to a package manager.

Python requirements files can also specify urls to import packages from a variety of version
control systems, including git. When combined with virtual environments, developers can implement a
clever workflow that treats a git as a personally curated repository. This gives developers a
significant amount of control over their dependency pipeline, but can be difficult to manage. One
problem is that because all requirements are pinned in the project source, dependency updates have
similar difficulties as monolithic packages-- any dependency update requires a new version for the
whole project. This control also requires the maintainers to be actively engaged in each of the
dependencies to know when updates are necessary. Also like monolithic packages, security is a
concern because the maintainers may not be able to rerelease, or they may not be aware of important
patches.

A general concern with most packaging workflows is dependence on 3rd party services. These services
can go down or introduce backwards incompatible changes. Some services, like PyPI allow package
authors to remove content at any time. If reproducibility is critical, the entire dependency
pipeline should be under the control of the maintainers.

Introducing Pulp
================

Each of the approaches discussed offer a fundamental tradeoff when choosing a package management
strategy. Strategies that increase control can improve reliability, but put significantly more
responsibility on the maintainers. Even if a particular strategy well works for a specific project
in its ecosystem, another ecosystem with a different tool set may not fit the strategy the same
way, and will also come with a new learning curve.

An alternative to the eclectic strategies native to various ecosystems is a universal package
manager like Pulp. Pulp is a fully open source Python project that manages packages of any type by
leveraging a plugin architecture. With the python plugin, for example, Pulp is able to
fetch content from PyPI and publish content that can be consumed by pip_, allowing Pulp users to
implement reproducibility focused workflows that transfer across packaging ecosystems.

Pulp 3, which recently entered beta offers additional features that simplify reproducibility, such
as versioned repositories and immutable publications. When combining Pulp 3's promotion/rollback
workflows with the strategies discussed above, researchers can achieve the rigorous stability of
monolithic packages/curated repositories (via a hosted, immutable publication) and the flexibility
and short development cycle of a community repository like PyPI. Pulp users host their own servers,
and therefore own their entire dependency pipeline.

Example Workflow
================

With the rich feature set provided by the Python ecosystem and the powerful workflows enabled by
Pulp, it is necessary to demonstrate how they can be used together to achieve flexible development
while also ensuring reproducibility. This section discusses workflows at a very high level and
does not include all steps for brevity. The Pulp documentation should be referenced for
comprehensive workflows and specific commands.

When developing a new tool, it is ideal to work with the latest versions of dependencies. A Pulp
server can be set up and configured to fetch these dependencies from PyPI, and pip can be
configured to install from a hosted Pulp publication. Each time Pulp fetches new content, it
creates a new repository version. Development is never blocked because the administrator can
instantly (without downtime) roll back to a stable version whenever there is a problem.

When the project matures enough to be used in publishable research, a curated repository is created
containing only the desired versions of packages. Source code should be packaged with twine,
uploaded into the curated repository, and the repository should be published. When the publication
of a curated repository is shared, it can be used to create a Python environment with exactly the
same bits; the procedure documentation of the research should include instructions for configuring
pip to use this publication.

Even as research proceeds through peer review and publication, development can still continue
against recently updated dependencies. Each Pulp publication is isolated and immutable, allowing
legacy publications that ensure reproducibility to be served parallel to new publications used for
flexible, up-to-date development.

If a security flaw is discovered in a dependency that was used in published research, a new
"hotfix" publication can be created that bumps the version of a single dependency. This hotfix
publication can be hosted in parallel at a testing location, allowing researchers to carefully
verify identical results before seamlessly replacing the original publication at the advertised
location.

Beyond Python
=============

Computational environments created with Python tools cannot be 100% reproducible because many
aspects of the complete environment are not managed by Python packaging. Vandewalle identifies
several necessities for complete reproducibility :cite:`Vandewalle`: the program's source code,
package dependencies, system requirements and configuration, data source used, and documentation on
running the provided the source code. Together, Python tools, Pulp, and pulp_python can be used to
preserve source code and dependencies, but system requirements and configuration are outside of the
scope of Python packaging.

Ansible
-------

Ansible_ is an IT automation tool that can be used to configure systems, deploy software, and
orchestrate arbitrary advanced tasks. It has an active community, well established idioms, and a
large set of community extensions called Ansible modules. With Ansible it is possible to install
system dependencies in addition to Python dependencies.

"The approach is characterized by scripting, rather than documenting, a
description of the necessary dependencies for software to run, usually from the
Operating System [...] on up" :cite:`Clark`

Ansible tasks can be grouped into "Roles" and published to a community repository called "Ansible
Galaxy". Pulp and pulp_ansible can be used to manage these roles. Researchers can use pulp_ansible
to manage systems, and when used with pulp_python they are enabled to take another step toward
complete reproducibility.

Containers
----------

Containers_ [1]_ "are technologies that allow you to package and isolate applications with their
entire runtime environment—all of the files necessary to run." Containers are particularly well
suited for reproducibility, each container contains a system image, a copy of source code,
installed dependencies, and data to be used. These are stored in a static file called an Image.

This Image can be shared with reviewers, collaborators, and reproducers, preserving a computational
environment in its entirety. However the Image itself is opaque, and it is hard to tell what
dependencies have been installed on the image without substantial inspection.  It is recommended
that the Image is built from a Dockerfile or Ansible roles for full transparency.

Docker images can also be managed by Pulp and pulp_docker, following workflows that are nearly
identical to those of pulp_python.

Extending Pulp with a new Plugin
--------------------------------


Summary
=======

For researches who use code in their methods, it is crucial to consider the reproducibility of the
software environments they use. Excellent research can become nearly impossible to replicate
because of the difficulty of maintaining a reliable dependency chain. By using the tools and best
practices developed for software engineering, researchers can take steps to prevent code entropy
and preserve the efficacy of their work.

Acknowledgements
================

We appreciate Red Hat's continued support for open source technologies
(including Pulp), and to the PyPA for their continuous effort at making
Python packaging usable and stable. A special thank you to Michael Hrivnak,
who helped formulate and fact check the Containers terminology, and Dana
Walker for proof reading.


References
==========

.. [0] A virtual environment, often abbreviated “virtualenv” or “venv”,
    is an isolated python environments that is used to prevent projects and
    their dependencies from interfering with with each other. Under the hood,
    virtual environments work by managing the PYTHON_PATH Another benefit of
    Virtual environments is that they do not require root privileges and are
    safer to use.

.. [1] Most often people think of docker containers when the word container is
    mentioned. Docker is the most well known, however docker schema, and
    standards are not well documented.  Containers in this case can refer to
    Linux Container which is a superset of Docker Containers, Rkt, LXC, and
    other implementations. While most of the ideas discussed here will be
    generic across containers, the docker container, and DockerHub will be used
    as examples, due largely in part to their popularity.

.. [2] There are several closed sourced alternatives; Artifactory and Nexus are
    the two that are most commonly used.

.. [3] https://github.com/moby/moby/issues/20424

.. [4] https://github.com/moby/moby/issues/20380

.. [5] https://docs.pulpproject.org/en/3.0/nightly/plugins/plugin-writer/index.html

.. [#Pulp] Pulp Project, 2018, A Red Hat Community Project, https://pulpproject.org/

.. [#pip] pip, 2008-2017, PyPA, https://pip.pypa.io/en/stable/

.. [#requirements] requirements.txt, 2008-2017, PyPA, https://pip.readthedocs.io/en/1.1/requirements.html

.. [#pipenv] pipenv, Kenneth Reitz, https://docs.pipenv.org/

.. [#Ansible] Ansible, 2018, Red Hat, Inc, https://www.ansible.com/

.. [#Containers] containers, 2018 Red Hat, Inc, https://www.redhat.com/en/topics/containers

.. [#concepts] concepts, 2018, A Red Hat Community Project,
    https://docs.pulpproject.org/en/3.0/nightly/overview/concepts.html
