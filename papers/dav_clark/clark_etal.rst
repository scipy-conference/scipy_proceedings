:author: Dav Clark
:email: davclark@berkeley.edu
:institution: UC Berkeley

:author: Aaron Culich
:email: aculich@berkeley.edu
:institution: UC Berkeley

:author: Ryan Lovett
:email: rylo@berkeley.edu
:institution: UC Berkeley

:author: Chris Paciorek
:email: paciorek@stat.berkeley.edu
:institution: UC Berkeley


--------------------------------------------------------------------------------
BCE: Berkeley's Common Scientific Compute Environment for Research and Education
--------------------------------------------------------------------------------

.. class:: abstract

  * A common problem is running scientific codes across a range of machines.
  * We examine the success of [OSGeo-Live]_ in providing a standard environment for GIS projects, both for developer deployment (e.g. developers of QGIS) and for researcher evaluation of new tools.
  * We introduce available tools for Virtual machines and devops. These are somewhat widely used, but in an ad hoc manner.
  * We also provide a brief review of existing projects, and how they are using these technologies.
  * Members of the UC
    Berkeley D-Lab, Statistical Computing Facility (SCF), and Berkeley Research
    Computing (BRC) have evaluated a breadth of virtualization technologies and
    present a strategy for constructing the Berkeley Common Environment [BCE]_.
  * We examine a variety of concrete training and research use-cases in which
    this approach increases productivity, reuse, and reproducibility.
  * Recommendations.

.. class:: keywords

   education, reproducibility, virtualization

Introduction
------------

Quote: "btw- I have never setup for this LaTeX workflow, so the make_paper.sh
fails for me miserably. Therefore, no advanced formatting and very probably
more basic errors right now.. but, the prose is coming along" --One of the
Authors

Segue to general problem, perhaps mention nbconvert tooling (parallel to the above).

Here we frame the problem space and describe the general class of devops
solutions.

Available Tools
---------------

"Collaboratool" was conceived as a project for building, integrating, and
deploying tools that support portable, reproducible data science.  We started
thinking about how to deploy virtualized containers that provide things like
IPython notebooks through the web. We were very inspired by
[jiffylab](http://github.com/ptone/jiffylab). From there, we decided that it
made more sense to focus on a complete virtual environment, which is easy to
deploy in a variety of contexts, which is what you'll find here now.

*Virtual Machines (VMs)*

Full virtualization: VirtualBox, (VMware, mention encryption), others? (KVM, etc.)

Systems like EC2, only available as a VM.

Lightweight virturalization (/ containerization) includes Docker / LXC / VMWare
ESX.

Only with exotic hardware is GPGPU [unpack] available to fully virtualized environments. Check on containers? [XXX IT people?]

port-mapping, shared files, GUI vs. “remote-like” operation

*DevOps*

XXX - Particularly need help from Aaron here (Richard

Vagrant (heavily explored, set aside)
Packer (currently used)
Docker (potential future)

XXX - Is Hashdist here or in exsting projects?


Do not expect the readership to be familiar with VM technologies such as VMWare, VirtualBox, Vagrant, Docker. Most scientists do not clearly see what they are good for when looking at the original documentation (which is more written for system administrators or web site developers). However, it should be easy to explain the problem setting to scientists: it is always a big mess to get all software up and running on a platform, especially for a course. Very often scientific computing implies a lot of sophisticated software beyond Anaconda or Enthought ready-make setups. You present smooth solutions, and most scientists will benefit from them.
Explain or avoid terms in the VM community such as provisioning. Make sure you reach out to the average scientist (who knows very well that setting up a Python environment is often non-trivial).
The big difficulty (for me) is to choose the right VM tool. Any experience with VMWare, VirtualBox, Vagrant, Docker would be very useful at this stage.
So far VM tools are mostly used on individual laptops. The idea that a VM can be used on computer systems at a university and could services as well, and that one can simply move the laptop VM to these platforms, is something that will greatly increase productivity.[a]


OSGeo-Live: A Successful Common Environment
-------------------------------------------

The OSGeo-Live, a project the Open Source Geospatial Foundation, is a vivid example
of both a sophisticated compute environment, and synergistic community process;
quoting from the OSGeo-Live [1] website:


'''
The OSGeo-Live is a self-contained bootable DVD, USB thumb drive or Virtual
Machine based on Xubuntu, that allows you to try a wide variety of open source
geospatial software without installing anything. It is composed entirely of free
software, allowing it to be freely distributed, duplicated and passed around.

It provides pre-configured applications for a range of geospatial use cases,
including storage, publishing, viewing, analysis and manipulation of data. It
also contains sample datasets and documentation.
'''

The OSGeo-Live is formally a project of the Open Source Geospatial Foundation
(OSGeo), an international body modeled on the Apache Foundation [3]. Eight years
ago, there existed several very large and growing open-source geospatial
projects, whose founders and developers decided would benefit from a common
legal and technical infrastructure. Those projects included GRASS, Mapserver,
GDAL and QGis.  At the same time. the OSGeo-Live began with a smaller open
project based in Australia that sought to build an "easy to try and use"
software environment for these and other spatial data applications. After some
discussion and planning conducted between a handful of intrepid principals
across the globe on the Internet, the nascent OSGeo-Live project committed
itself to the larger OSGeo Foundation structure in its second year. More than
fifty (50) open-source projects now actively maintain and improve their own
install scripts, examples and documentation. After long years of "tepid" progress and iteration, a combination of techincal stability,
tool sets, community awareness and clearly-defined steps to contribute, provided the basis
for substantial growth. The OSGeo-Live is now very stable, easily incorporates advances in
components, and widely adopted.


OSGeo-Live is now very stable, easily incorporates advances in components,
and widely adopted. Let's look at each of these building blocks briefly:

Technical Stability
^^^^^^^^^^^^^^^^^^^

An original goal of the OSGeo-Live was to operate well on minimal hardware with
broad support for common peripherals, and a license structure compatible with
project goals. The XUbuntu version of Ubuntu Linux was chosen as a foundation,
and it was been very successful. To this day, almost all applications
operate easily in very modest RAM and disk space (with the notable exception of
Java-based software which requires substantially more RAM).

OSGeo-Live itself is not a "linux distribution" per se, primarily because the
project does not provide a seamless upgrade process from one version to another.
OSGeo-Live relies on the Ubuntu/Debian/GNU, apt-based ecosystem to handle
the heavy-lifting of system updates and upgrades. This is a win-win, as updates
are proven reliable over a very large Ubuntu community process, and frees
project participants to concentrate on adding value to its featured components.

As we shall see, due to a component architecture, individual software projects
can be installed as-needed on a generic base.

Tool Sets
^^^^^^^^^

It cannot be overstated that a
key component to the success of the overall project has been the availability of
widely-known and reliable tools, to developers from all parts of the world and
in all major spoken languages. It is also important to note that, rather than
require formal installation packages ".deb" for each project, the OSGeo-Live
chose to use a simple install script format, one per installed project. This
choice proved crucial in the earliest stages, as an outside open-source project
evaluating participation in the Live could get started with fewer barriers to
entry, and then add rigor and features later. Almost by definition, the
candidate open-source projects had install scripts already built for Linux which
could be readily adopted to the OSGeo-Live install conventions. By providing
ample examples on the OSGeo-Live of install scripts in major deployment
contexts, for both applications and server processes,  and clear guidelines for
installation conventions, a new open-source project could almost immediately
develop and iterate their own install scripts in a straightforward way.

**detailed build directions here**
Particular example: web, including apache, WSGI, etc. Standard layout of web
directory. Fully working examples available for each "kind" of project.

Subversion repo -- asset heirarchy -- individual install scripts -- Live build
 scripts trac-subversion   http://trac.osgeo.org/osgeo/report/10

see screenshots

Community Awareness
^^^^^^^^^^^^^^^^^^^

The processes of  adoption of new
technology - initial awareness, trialability, adoption and iteration -
are well-known [4].

In the case of the OSGeo-Live, an orginial design goal was to provide tools
to those doing geospatial fieldwork with limited resources, and who often lack
advanced programming and administration skills.


Several years into the project, funding was established via a grant from the Australian
government to build documentation on applications in the Overview and Quickstart formats
to professional graphic design standards. A single page for every application,
(Overview) and a second page with step-by-step instructions for a capable reader but no previous
exposure to the software (Quickstart). Each of these two pages is then translated into
various spoken languages, primarily by volunteers. Much later, a graph of "percentage complete"
for each human language group was added, which essentially makes translation into a sort of
competition. This has proved very successful. Note that the initial effort to build
standardized documentation required paid professionals. It seems unlikely that the
documentation would have been successful if only ad-hoc volunteer efforts were used.

The Open Source Geospatial Foundation (OSGeo) the hub for a variety of projects to interoperate, and
potentially share with each other / synergy. OSGeo raises awareness of other projects.

(see the transfer of tech, e.g., military technology to environmental applications)
(Maybe include story about Haiti response with open source mapping)


Steps to Contribute

A FAQ was written and published in an easily accessible location. Outreach was
conducted through formal and informal networks.

Major step in diffusion is folks knowing what the thing is at all. Value add /
branding - OSGeo has credibility from foundation status, participants,
consistent / good graphic design.

[1] http://live.osgeo.org
[2]  build stuff
[3]  repo stuff
[4] Diffusion of Innovation; Rogers et al 1962
http://en.wikipedia.org/wiki/Diffusion_of_Innovations

**misc cut text**

Eight
years ago, there existed several very large and growing open-source geospatial
projects, whose founders and developers decided would benefit from a common
legal and technical infrastructure. Those projects included GRASS, Mapserver,
GDAL and QGis.  At the same time. the OSGeo-Live began with a smaller open
project based in Australia that sought to build an "easy to try and use"
software environment for these and other spatial data applications. After some
discussion and planning conducted between a handful of intrepid principals
across the globe on the Internet, the nascent OSGeo-Live project committed
itself to the larger OSGeo Foundation structure in its second year.


missing title
-------------
What are the steps that established credibility to get projects contributing to
the distribution. Initially, just shell scripts to install five core / important
packages (XXX - what were they and why?). Reached out to 50 projects, more
outside of U.S. than in, including many non-english (as a primary language),
esp. from Europe. The social component of building the community was not
necessarily explicit or even shared or known to all contributors (provable?).

It consists of a choice of basic tools that are widely known to free software
developers: shell, Perl, or Python. Scripts may call package managers, few
constraints (e.g., keep recipes contained to a particular directory). Core,
customizable scripts designed to bootstrap new recipes.

Particular example: web, including apache, WSGI, etc. Standard layout of web
directory. Fully working examples available for each "kind" of project.

The result is that certain tools are becoming more and more visible. Projects
are encouraged that are implemented in standard frameworks (i.e., not Forth).

There's still not complete consensus, but the project is moving towards some
consensus infrastructure. Also see the transfer of, e.g., military technology to
environmental applications.

Maybe include story about Jamaica response with open source mapping.



Other virtual machines
----------------------

From [Mining the Social Web, a Chef+Vagrant solution](https://rawgit.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/master/ipynb/html/_Appendix%20A%20-%20Virtual%20Machine%20Experience.html)

From Matt Gee (of [DSSG](http://dssg.io): We've been trying a number of
different approaches to the standard development environment. For this year's
fellowship we went with a Chef cookbook + OpsWorks. This works for provisioning
our core resources. However, for weekend learn-a-thons and more portable VM.
We've tried our own VM using docker and well as some hosted boxes like yhat's
new Science Box. We should compare notes.

VM from Philip.

BCE: The Berkeley Common Environment
------------------------------------

The goal for the BCE is to provide both the ready-made environments, and also
the "recipes" or scripts setting up these environments. It should be easy for a
competent linux user to create recipes for custom tools that might not be
braodly useful (and thus, not already in BCE).

For classwork and research in the sciences at Berkeley, broadly defined to
include social science, life science, physical science, and engineering. Using
these tools, users can start up a virtual machine (VM) with a standardized Linux
operating environment containing a set of standard software for scientific
computing. The user can start the VM on their laptop, on a university server, or
in the cloud. Furthermore, users will be able to modify the instructions for
producing or modifying the virtual machine in a reproducible way for
communication with and distribution to others.

We envision the following core use cases:

  * creating a common computing environment for a course or workshop,
  * creating a common computational environment to be shared by a group of
    researchers or students, and
  * disseminating the computational environment so outsiders can reproduce the
    results of a group.

Other use cases/benefits:

 * Thin client / staff computing
 * Exam environments
 * Instructional labs
 * Sharing licensed software?
 * Make it easy to do the "right" thing (hard to do "wrong" thing)
 * Stable infrastructure
 * Managing complexity
 * Impacts beyond "the course"

What problems does BCE solve for you?

 * No more obscure installation issues - download and run a single virtual
   machine or get the same environment on a bare metal or virtual server.
 * I'm teaching a class - when you tell a student that a program behaves a
   certain way, it does!
 * I'm collaborating on some scientific research - now all of your collaborators
   can run your code without complex installation instructions.
 * Easy Deployment
 * Replication / Reproducible research
 * Easy transition across scales (laptop to cluster)
 * Tricky installs

To accomplish this, we envision that BCE will encompass the following:

 * a reproducible workflow that creates the standard VM/image
   with standard scientific computing software such as Python, R, git, etc.,
 * a standard binary image, produced by the workflow, that can be distributed as is and
   used on-the-fly with VirtualBox or VMWare Player with minimal dependencies, and
 * (possibly) an augmented workflow that represents multiple possible distributions tailored
   for different types of uses (e.g., different disciplines, different
   computational needs, class vs. research use, etc.). This might
   represent either a sequence or a tree of possible VMs.


*Tentative list of features*

 * VMs

   * A fixed, versioned VM provided each semester as a binary image for classes
     and workshops
   * Ideally, the same VM usable for research, with functionality for parallel
     computing and provisioned such that it can be used as the VM for virtual
     cluster nodes
   * The VM runnable on user laptops (Mac/Windows/Linux) and on cloud machines
   * The VM usable on user machines with minimal dependencies (e.g., either
     VirtualBox or VMware) and minimal setup, and with clear instructions for
     users on setup and on getting data/files into and out of the VM
   * Agreement on minimal hardware requirements on the host machine - do we
     support 32 bit, any minimum RAM required?
   * Shared folders (EBS on AWS), or other tech to make it possible to separate
     data from VM.

 * Provisioning

   * Provisioning is fully scripted - if the appropriate software is installed,
     the recipe should run reliably.
   * The provisioning details used to create a given VM available to users and
     with clear instructions on how to use and modify the provisioning; ideally
     the provisioning would be relatively simple for users to understand
   * The ability for a user to add software to a VM and then 'export' that
     information back into the provisioning workflow that can be used to
     recreate the modified VM

 * Logistics and training

   * A GitHub repository or the like plus a project website with all BCE
     materials available
   * Communication with users on bugs, desired features, and the like via the
     repository and a mailing list
   * Management / Versioning / Snapshotting

 * Problems

   * VMs reserve compute resources exclusively (less of a problem with LXC-like
     solutions).
   * Testing / Issue tracking

*Students ("horizontal" collaboration), Researchers ("vertical" collaboration)*

If you'd like to use the VM as a student, researcher, or instructor, our goal is
to make this easy for you.

If you're using VirtualBox, [follow these instructions](using-virtualbox.html).

If you'd like to use the VM on Amazon's EC2 cloud platform, [follow these
instructions](using-ec2.html).

Adding modules?

*Creating (and modifying) the BCE VM*

All the files for creating the VM are in the collaboratool repository on GitHub.

To clone the repository from the command line:

    git clone https://github.com/dlab-berkeley/collaboratool

Then go to the provisioning directory and see the information in HOWTO.md.

*VirtualBox*

  * Download and install VirtualBox from the [VirtualBox
    website](https://www.virtualbox.org/wiki/Downloads). This is the tool that
    runs the virtual machine for you.
  * Download the BCE VM in the form of an OVA file from [UNDER
    CONSTRUCTION](BCE-xubuntu-14.04-amd64.ova).
  * Open VirtualBox and import the BCE-xubuntu-14.04-amd64.ova file you just
    downloaded by going to "File->Import Appliance" and then selecting the .ova
    file from wherever you downloaded it to (possible 'Downloads' in your home
    directory on the machine).
  * Wait a few minutes...
  * Start the virtual machine by clicking on the tab for
    "BCE-xubuntu-14.04-amd64" on the left side and then clicking "Start" at the
    top. This will start a virtual Linux computer within your own machine. After
    a few seconds you should see black screen and then soon you'll see the
    desktop of the VM.

You now have a machine that has all the software installed as part of BCE,
including IPython and useful Python packages and R, RStudio and useful R
packages.

You can get a terminal window that allows you to type commands in a UNIX-style
shell by clicking on the icon of the black box with the $ symbo on the top
panel. Using this you can start IPython Notebook by simply typing "ipython
notebook" or  R by simply typing 'R' at the prompt in the terminal. This starts
a bare-bones R session. To start RStudio, either type 'rstudio' at the prompt on
go to "Applications->Programming->RStudio".

You can restart the VM at any time by opening VirtualBox and clicking on the tab
for the VM and clicking "Start" as you did above.

*Sharing folders and copying files between your computer and the VM*

One useful thing will be to share folders between the VM and the host machine so
that you can access the files on your computer from the VM. Do the following:

  * Got to "Devices->Shared Folder Settings" and click on the icon of a folder
    with a "+" on the right side.
  * Select a folder to share, e.g. your home directory on your computer by
    clicking on "Folder Path" and choosing "Other" and navigating to the folder
    of interest. For our purposes here, assume we click on "Documents".
  * Click "make permanent" and "auto-mount" and then click "Ok".
  * Reboot the machine by going to applications button on the left of the top
    toolbart, clicking on "Log Out", and choosing "Restart" in the window that
    pops up.
  * Once the VM is running again, click on the "Shared" folder on the desktop.
    You should see the folder "sf_Documents" (or whatever the folder name you
    selected was, in place of 'Documents'). You can drag and drop files to
    manipulate them.
  * Alternatively, from the Terminal, you can also see the directory by doing
    "cd ~/Desktop/shared/sf_Documents" and then "ls" will show you the files.

Be careful: unless you selected "read only" at the same time as "make
permanent", any changes to the shared folder on the VM affects the folder in the
'real world', namely your computer.

*EC2*

  * Go to [EC2 management console](http://console.aws.amazon.com) and choose the
    US-West-2 (Oregon) region, as that is where we have posted the BCE AMI.
    (You'll need to have an account set up.)
  * On the "AMIs" tab, search for the BCE AMI amongst public images.
  * Launch an instance 55. Follow the instructions given in the "Connect" button
    to SSH to the instance
  * If you want to connect as the "oski" user, you can deposit your public SSH
    key in the .ssh folder of the "oski" user.



Conclusion
----------

Keep in mind that *you* are now at the cutting edge. Extra care should be taken to make your tooling accessible to your collaborators. Where possible, use tools that your collaborators already know - shell, scripting, package management, etc.

That said, technologies that allow efficient usage of available hardware stand to provide substantial savings, and potential for re-use by researchers with less direct access to capital. [e.g., Docker, aggregation of cloud VM providers]

Let’s be intentional.
Be transparent/explicit about our choices/assumptions.
That *doesn’t* have to be technical - a simple text file or even a PDF can provide ample explanation that a human can understand.
Be willing to make strong recommendations based on what we are actually using (eat own dogfood)
Be willing to adopt/adapt/change/throw stuff out (have an exit strategy)

Recipe for setting up sicpy_proceedings build system on Ubuntu 14.04.

________________

Examples for proper rst formatting
----------------------------------

Code highlighting:

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
   +------------+----------------+
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


Perhaps we want to end off with a quote by Lao Tse:

  *Muddy water, let stand, becomes clear.*


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

.. [BCE] http://collaboratool.berkeley.edu
.. [OSGeo-Live] http://www.osgeo.org/
   # A more proper reference
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


[a]Copied from https://github.com/scipy-conference/scipy_proceedings/pull/98#issuecomment-46784086
