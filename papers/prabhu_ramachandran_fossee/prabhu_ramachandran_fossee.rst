:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:


--------------------------
The FOSSEE Python Project
--------------------------

.. class:: abstract

    The FOSSEE (Free Open Source Software for Science and Engineering
    Education) project (http://fossee.in) is a project funded by the Ministry
    of Human Resources and Development, MHRD, (http://mhrd.gov.in) of the
    Government of India.  The FOSSEE project is based out of IIT Bombay and
    has for its goal to eliminate the use of proprietary tools in the college
    curriculum.  FOSSEE promotes various open source packages.  Python is one
    of them.

    In this paper, the Python-related activities and initiatives of FOSSEE are
    discussed.  The important activities include, the creation of
    spoken-tutorials on Python, the creation of over 400+ IPython-based
    textbook companions, an online testing tool for a variety of programming
    languages, a course akin to software carpentry at IIT Bombay, the
    organization of the SciPy India conference, and finally spreading the
    adoption of Python in schools and colleges.



Introduction
-------------

The FOSSEE project (http://fossee.in) started in 2009 with the goal to try
help minimize the use of proprietary software in the college curriculum in
India.  The project is funded by the Ministry of Human Resources and
Development, MHRD, (http://mhrd.gov.in) of the Government of India.  FOSSEE is
part of the MHRD's National Mission on Education through ICT (NMEICT).  The
NMEICT project started in 2009 as as initiative to improve the quality of
education in India.  As part of this project there have been several
initiatives.  One sterling example is the NPTEL project which provides content
for over 900 courses at the graduate and post-graduate level (400 web-based
and 500 video-based) online.  These are proving to be extremely useful all
over the country.  Other projects include the Spoken Tutorial project
(http://spoken-tutorial.org) which has also been previously presented at
SciPy 2014.  FOSSEE is one such project that is the outcome of the NMEICT
funding.


The FOSSEE project is based out of IIT Bombay and promotes the use of various
open source packages in order to help eliminate the use of proprietary
packages in the curriculum.  A large number of colleges tend to unecessarily
purchase commercial licenses when they really do not need it.  The difficulty
with using commercial packages to teach basic concepts and computational
techniques is well known:

- The packages are typically expensive, the money could be better spent on
  equipment.  This is especially relevant in India.

- Students cannot legally take the software with them home or after they
  complete their course.

- Academic licenses are not enough as the students end up becoming dependent
  on the packages after the leave the institution and this limits them.

In order to help reduce the dependence on commercial packages, the FOSSEE
project's efforts are focused towards training students and teachers to use
FOSS tools for their curricular activities.  This also requires development
efforts in order to either enhance existing projects or fill in any areas
where FOSS tools are lacking.  There are about 10+ PIs actively involved in
various sub-projects.  Some of the most active projects are Scilab, Python,
eSim (an EDA tool), OpenFOAM, Osdag (open source design of steel structures),
etc.

After the initial efforts in 2009 and 2010 we found that some of our
initiatives worked and scaled up well whereas others did not.  As a result,
all of the FOSSEE sub-projects follow a similar structure.  Typically
each sub-project produces the following output:

- Generates "spoken-tutorials" that new users can use to self-learn a
  particular software package.

- Organize a crowd-sourced development of "textbook companions" for popular
  textbooks used in the curriculum.  A textbook companion is created when
  every solved example in a text is solved using a particular open source
  software package like Scilab or Python.

- Support user questions on a forum for the packages that are promoted.

- Develop new software that is useful in a particular domain.

- Support hardware interfacing to encourage open experimentation.

- Migrate labs that use proprietary packages and help them switch to a FOSS
  equivalent.

- Conduct workshops and conferences to spread the word and teach students and
  teachers.

Some of these are project specific.  For example, the Scilab project is able
to help with lab migrations as Scilab is a close equivalent to Matlab and this
makes it easier for people to switch to it from Matlab.

In this paper we discuss some of the outcomes and efforts that we have
undertaken towards popularizing the use of Python.



Spoken-tutorials
----------------

When the project started in 2009, we tried to conduct many live workshops but
this proved to be too time consuming and did not scale.  There are more than
3000 colleges in the country and live workshops cannot reach all of these
institutions.  At this time it was felt that preparing self-learning material
that students can learn on their own was much more effective and scalable.  A
sister project, the spoken-tutorial project pioneered the generation and
dissemination of spoken-tutorials.  A spoken tutorial is basically a carefully
designed screencast for a roughly 10 minute duration or less.  Any screencast
cannot qualify as a spoken-tutorial.  Notably a spoken tutorial should be made
such that a newbie can understand it.  The spoken-tutorial project ensures
that all new tutorials undergo a novice check to make sure that this is indeed
the case.  A spoken tutorial also features a carefully written script.  This
allows a spoken tutorial to be dubbed into multiple languages.  A series of
spoken tutorials can thus be used to effectively teach a programming language
or software package.

The spoken tutorial project has trained over a million students and teachers
on a variety of software packages.  The project hosts over 700 individual
spoken-tutorials.  Over 20 different Indian languages are supported.  Some
languages have better representation than others.

As part of the Python initiative we have created about 40 initial spoken
tutorials to teach non-CSE undergraduate students how to use Python for their
curricular computational tasks.  A new set of around 50 tutorials is currently
being recorded.  The spoken tutorials include tutorials on starting with
IPython, plotting with matplotlib, etc.  Currently these are only available in
`English <http://spoken-tutorial.org/tutorial-search/?search_language=English&search_foss=Python&page=1>`_.

These spoken tutorials can be accessed by anyone and can also be downloaded
into a self-contained CD by users.  Around 40000 users have gone over this
material.

The FOSSEE team generates the spoken tutorials and the spoken tutorial team
coordinates the conduct of workshops where students use this material to teach
themselves Python.  FOSSEE staff also help support user questions that may
arise during these workshops.



Textbook companions
--------------------

One oft quoted problem with FOSS tools is the lack of documentation.  Good
quality documentation is not easy to write and requires both expertise as well
as the ability to explain things at the level of the user.  This is often
difficult for a developer who knows everything about the package.  On the
other hand it is not always easy for an inexperienced user to write
documentation.

Textbook companions offer an interesting approach to this task.

- Student programs solved examples using FOSS package.
- Submits IPython notebooks.
- Checked by reviewer.
- Hosted online.

This task can also be scaled up by crowd-sourcing.  Over 400 textbook
companions have been created in this fashion.  Around 200 more are currently
in progress.  The companions are hosted online.  Users submit these in the
form of chapter-wise IPython notebooks.  These are hosted at
http://tbc-python.fossee.in

The examples can be browsed online, we also host a tmpnb server that allows
these notebooks to be edited online.  The sources for this are all hosted on
github.

Readers can also leave comments on the notebooks.


XXX figure of tbc interface



Online test tool: Yaksh
------------------------

- History and motivation for Yaksh.

- Basic features

  - Open source
  - Django based
  - Support for MCQs and programming questions
  - Immediate feedback for user
  - Ability to monitor student's progress
  - Handles multiple programming languages
  - Sandboxes user code.
  - Docker friendly.

- Installation and usage of Yaksh

- Online Interface

- Web API for other uses

- Personal use cases

- Future plans



SDES course
------------

- Motivation for course

- Course structure

- Experience from teaching the course

- Impact


SciPy India
------------

- Conference stats and info

- Impact of conference



Future plans
------------

- Python in CBSE curriculum and issues
- Python in school initiative


Conclusions
------------



Acknowledgements
-----------------
