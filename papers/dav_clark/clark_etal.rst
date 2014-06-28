:author: Dav Clark
:email: davclark@berkeley.edu
:institution: UC Berkeley

:author: Aaron Culich
:email: aculich@berkeley.edu
:institution: UC Berkeley

:author: Brian Hamlin
:email: maplabs@light42.com
:institution: UC Berkeley

:author: Ryan Lovett
:email: rylo@berkeley.edu
:institution: UC Berkeley


--------------------------------------------------------------------------------
BCE: Berkeley's Common Scientific Compute Environment for Research and Education
--------------------------------------------------------------------------------

.. class:: abstract

  * A common problem is running scientific codes across a range of machines (e.g. laptop, workstation, hpc cluster, cloud platform).
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

  btw- I have never setup for this LaTeX workflow, so the make_paper.sh
  fails for me miserably. Therefore, no advanced formatting and very probably
  more basic errors right now.. but, the prose is coming along

  --One of the Authors

Every author of a paper in a SciPy proceedings has to deal with the tooling required to test that their paper is building and displays properly. This is not a profoundly difficult problem, but it serves to illustrate the point. In particular, documenting the “solution” to building SciPy papers is unnecessarily complex due to the variety of potential environments (e.g., Mac, Windows, \*nix), and even methods that authors might have used to set up Python on their machine (e.g., Anaconda, Canopy, python.org).

This problem is quite general. In a university setting, students in a class, collaborating researchers, and the provision of reproducible workflows for literally anyone to run the code used for a set of research results. Developers in industry similarly need to ensure code will work in a specified environment (like on production servers!), and a large number of tools have been developed and widely adopted to manage this piece of complexity.

A number of solutions have also been developed to allow for multiple *Python* environments, including environments that peacefully co-exist on the same computer (e.g., virtualenv, venv, conda, buildout), but our compute environment often pulls in non-trivial tooling outside of Python (though tools like conda *can* pull in non-python tooling). While these tools cannot solve all of the problems we describe below, there is no reason that any of them could not be used within the broader approach we’ll describe. Indeed, a tool like conda could ultimately perform *most* of the work – though as we’ll see, it should likely never be able to perform all of that work.

Prosaic examples include the toolchain required to use nbconvert (very much parallel to the above). Or, for example, ensuring that a properly configured database is available.

The afore-mentioned “tools from industry” -- generally referred to as *devops* tools -- are directed at solving this larger problem. Unfortunately, the variety and complexity of tools match the variety and complexity of the problem space, and the target space for most of them was *not* scientific computing. Thus, before discussing available tooling, we first lay out a set of issues relevant to supporting scientific computing.

Issues
------

Historically, the users of computational tools (and their collaborators) were equipped with a suite of largely self-taught or informally learned foundational skills (command line usage, basic software architecture, etc.). The tooling and technical skills employed by members of a discipline provide notable boundaries between those who do and do not (and perhaps cannot) participate in that discipline. However, we are entering an era where these boundaries are becoming barriers to the mission of our university (and presumably others).

For the purposes of BCE, there are three broad use-cases. Instructional use requires simultaneous support of *teaching* and *learning*. *Collaboration* introduces difficulties where individuals from different backgrounds have disjoint tooling and skill-sets. A further difficulty arises with the general problem of “collaborating” with scientists around the world that may be interested in *re-using* our code, or *reproducing* our research. Finally, it’s important to consider the demands that might be placed on the individuals actually *developing* and *supporting* the common environment.

Our primary concern at present for the BCE is educational, particularly introductory computational science and statistics. However, where possible, we wish to build an environment that supports the broader set of uses we outline here.

For instruction
^^^^^^^^^^^^^^^

We are entering an era where experimental philosophers want to take serious courses in computationally demanding statistics, sociologists have a need for best-of-breed text analysis, and psychologists wish to pick up scalable machine learning techniques. These students are often willing to work hard, and might sign up for the university courses meant to provide these skills. But while the group that the course was originally designed for (e.g., statistics or computer science students) have a set of *assumed* skills that are necessary to succeed in the class, these skills aren’t taught *anywhere* in the curriculum. After some struggle, students with divergent backgrounds often drop these classes with the sense that they simply can’t obtain these skills. This is not an equitable situation.

Given current standards for university courses, it’s difficult to write instructions that would work for any potential student. As mentioned above, students come to a course with many possible environments (i.e., on their laptop or a server). But if a standardized environment is provided, this task becomes much simpler. Written instructions need fewer special cases, and illustrations can be essentially pixel-identical to what students should be seeing on their screen.

Uniformity of the environment in which the user is clicking.

In our experience, some students will not be able to run the VM while others have difficulty getting regular access to a stable network connection (though fortunately, almost never both!). So, consistency across server and local versions of the environment is critical to effectively support students with either of these difficulties.

For scientific collaboration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Even among collaborators who are competent in their own domain, crossing disciplines can often demand the use of novel tools – for example, a neuroscientist may be well-versed in Matlab, and wish to collaborate with a policy researcher who’s skilled in SPSS. It’s easy to end up with a collaboration where neither party has any idea how to use the tools of the other.

In other words, I have an environment and you have an environment. I want to do things my way, you want to do them yours. If we wish to work together, either as student and teacher, or as collaborators, this doesn’t really work.

A place to ensure reproducability.


For administration
^^^^^^^^^^^^^^^^^^

The solution of using standardized virtual environments, or accounts on a shared server for instruction is not new. At UC Berkeley, the D-Lab has supported courses and short trainings with these technologies. Similarly, the Statistical Computing Facility supports an instructional lab and cloud-based VMs for some courses, and computer science courses often provide a VM (often provisioned by a graduate student assistant). In each and every case, multiple technical challenges are common. These technical glitches can delay or reduce the quality of instruction as compared to an environment that students are already familiar with. It is also a drag on the time of those supporting the course – time that could be better directed at course content!

However, if we had a standard environment, developed in an open-source fashion, many of these problems rapidly diminish, and likely reverse to net *savings* in time wasted. The more broadly an environment is adopted across campus, the more familiar it will be to all students. Technical glitches can be tracked or resolved by a community of competent contributors, allowing course instructors to simply use a well-polished end product, while reducing the complexity of instructions for students to set up course-specific software. These environments can also be tuned in ways that would be beyond the scope of what would be worth doing for an individual course - for example simple optimizations to increase the efficiency of numeric computations or network bandwidth for remote desktops.

It is at this point that our use case starts to sound like the case in which product developers are working together to deploy software on a production server, while maintaining a useful development environment on their own machines, testing servers, and so on. However, going forwards, we will suggest that these tools be largely the domain of administrator-contributors to a useful common environment. Students and even professors and researchers can continue to use the tools they are familiar with, such as the Ubuntu package manager, pip, shell scripts, and so on. However, before considering the tooling that might be used for this process, we provide a brief list of what a reasonable common environment should be like.

Features of a useful common environment
---------------------------------------

Simple things like gedit, nano with tab-stops set up properly. Setting up the background to be more efficient solid color.

There are also idiosyncratic things about individual VM software, like the way shared folders are handled (managing group membership, symlinks to the appropriate mount folder).

Python packages are installed from a basic pip requirements file.

Debian packages are similarly installed from a list.
Other packages are installed via bash, e.g., downloading and installing RStudio.


Available Tools
---------------

"Collaboratool" was conceived as a project for building, integrating, and
deploying tools that support portable, reproducible data science.  We started
thinking about how to deploy virtualized containers that provide things like
IPython notebooks through the web. We were very inspired by
[jiffylab](http://github.com/ptone/jiffylab). From there, we decided that it
made more sense to focus on a complete virtual environment, which is easy to
deploy in a variety of contexts, which is what you'll find here now.

Sage?

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

XXX - Is Hashdist here or in exsting projects? Conda goes here also. Why not conda? Still hard to just install a list of pip requirements




Do not expect the readership to be familiar with VM technologies such as VMWare, VirtualBox, Vagrant, Docker. Most scientists do not clearly see what they are good for when looking at the original documentation (which is more written for system administrators or web site developers). However, it should be easy to explain the problem setting to scientists: it is always a big mess to get all software up and running on a platform, especially for a course. Very often scientific computing implies a lot of sophisticated software beyond Anaconda or Enthought ready-make setups. You present smooth solutions, and most scientists will benefit from them.
Explain or avoid terms in the VM community such as provisioning. Make sure you reach out to the average scientist (who knows very well that setting up a Python environment is often non-trivial).
The big difficulty (for me) is to choose the right VM tool. Any experience with VMWare, VirtualBox, Vagrant, Docker would be very useful at this stage.
So far VM tools are mostly used on individual laptops. The idea that a VM can be used on computer systems at a university and could services as well, and that one can simply move the laptop VM to these platforms, is something that will greatly increase productivity.

Summarizing the pull-request feedback
- the most common case for VM tools is the laptop
- mobility of VMs between computing platforms will increase productivity
- Choosing between the multiplicity of VM tools is a big difficulty
- explain what is possible or what these tools enable: the utility of VM tools is not obvious to scientists by reading the tool’s documentation
- to reach a general (scientific) audience, avoid VM jargon
- scientific computing requires more than just the packaging basics

Problem 1: The quote at the beginning of this paper represents the first barrier to collaboration in which the full set of requirements are not explicitly stated and there is an assumption that all collaborators already have or can set up an environment to collaborate. The number of steps or the time required to satisfy these assumptions is unknown, and regularly exceeds the time available. For example, in the context of a 1.5 hour workshop or a class with only handful of participants, if all cannot be set up within a fixed amount of time (typically 20 minutes at most) it will jeopardize successfully completing the workshop or class materials and will discourage participation.

Solution 1: Eliminate dependency hell. Provide a method to ensure that all participants can successfully complete the installation with a fixed number of well-known steps across all platforms within a fixed amount of time. The BCE platform provides this in a scalable and quantifiable way.

Problem 2: We will consider a participant’s laptop the unit-of-compute since it is the primary platform widely used across the research and teaching space and is a reasonable assumption to require: specifically a 64-bit laptop with 4GB of RAM. These requirements are usually sufficient to get started, however the algorithms or size of in-memory data may exceed the available memory of this unit-of-compute and the participant may need to migrate to another compute resource such as a powerful workstation with 128GB of RAM, an amount of memory not yet available in even the most advanced laptops which typically max-out at 16GB at the time of this writing.

Solution 2: Enable computing beyond the laptop. Though a workstation with plentiful memory by virtue of exactly replicating the environment available in Solution 1, the participant is guaranteed to replicate the data processing, transformations, and analysis steps they ran on their laptop in these other environments with the benefit of more memory available on those systems. This also includes the ability to use the common GUI interface provided by BCE as a VDI (Virtual Desktop Integration).

Problem 3: Even though Solution 2 allows us to grow beyond the laptop, the time required

Solution 3: Enable pleasantly parallel scale-out. A cluster may be available in your department or at your institution or at national facilities that provides the equivalent of a hundred or a thousand of the workstations you may have in your lab, enabled by Solution 2. BCE works in these environments and allows you to install additional software components as you wish without relying on cluster administrators for help.

Problem 4: Assuming you have the grant money to buy a large workstation with lots of memory and many processors, you may only need that resource for a 1 to 2 week period of time, so spending your money on a resource that remains unused 95% of the time is a waste of your grant money.

Solution 4: Enable on-demand resizing of resources. The BCE solution works on cloud resources that may allow you to scale out

OSGeo-Live: A Successful Common Environment
-------------------------------------------

The OSGeo-Live virtual machine is an example of exactly the kind of environment described above. Despite the availability of sophisticated devops tools, and considerable technical competence amongst contributors to the project, the project is configured using a combination of Python, Perl and shell scripts, along with some straightforward conventions that support easy inter-operation of different projects.

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

It cannot be overstated that, a key component to the success of the 
overall project has been the availability of widely-known and reliable tools, 
to developers from all parts of the world and in all major spoken languages. 
It is also important to note that rather than require formal installation 
packages ".deb" for each project, the OSGeo-Live chose to use a simple install script format, one per installed project. This
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

Directory gisvm - a detailed layout

File Structure
==============

bin/
     /main.sh # Call all the other scripts
       /setup.sh # Download, and install all core files and set up config files
       /install_project1.sh # Download, and install all files for project1
       /install_project2.sh # Download, and install all files for project2
       /install_desktop.sh
       /install_main_docs.sh
       /setdown.sh

       /build_iso.sh
         /load_mac_installers.sh
         /load_win_installers.sh

     bootstrap.sh
     inchroot.sh
     package.sh
     sync_livedvd.sh

app-conf/
     /project1/   # config files used by install_package1.sh script
     /project2/   # config files used by install_package2.sh script


app-data/
     /project1/   # data & help files used by package1
     /project2/   # data & help files used by package2

desktop-conf/     # data files and images used for the main desktop background
     
doc/
     /index_pre.html            # header for summary help page
     /index_post.html           # footer for summary help page
     /arramagong.css
     /jquery.js
     /template_definition.html  # example of project_definition.html file
     /template_description.html # example of project_description.html file
     /template_licence.html     # incorportate into project_description.html???

     /descriptions/
       /package_definition.html    # short (1 sentence) summary of installed pkg 
       /package_description.html   # getting started instructions for the LiveDVD user

download/       # copy of the livedvd project's download server webpage

sources.list.d/ # Supplimentary package repositories for /etc/apt/sources.list



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

Recipe for setting up sicpy_proceedings build system on Ubuntu 14.04 (or BCE proper?).

Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.

References
----------

.. [BCE] http://collaboratool.berkeley.edu
.. [OSGeo-Live] http://www.osgeo.org/
   # A more proper reference
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


[a]Copied from https://github.com/scipy-conference/scipy_proceedings/pull/98#issuecomment-46784086
