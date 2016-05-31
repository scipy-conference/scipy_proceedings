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
makes it easier for people to switch to it from Matlab.  Kannnan Moudgalya's
paper in 2014 [kmm14]_ discusses in detail the approach and design decisions
made by the FOSSEE and spoken-tutorials projects.  In particular the paper
discusses spoken tutorials, textbook companions, and lab migrations.  In this
paper we focus on the Python-specific activities that are of potential direct
interest to the SciPy community.



Spoken-tutorials
----------------

When the project started in 2009, we tried to conduct many live workshops but
this proved to be too time consuming and did not scale.  There are more than
3000 colleges in the country and live workshops cannot reach all of these
institutions.  At this time it was felt that preparing self-learning material
that students can learn on their own was much more effective and scalable.  A
sister project, the spoken-tutorial project (http://spoken-tutorial.org)
pioneered the generation and dissemination of spoken-tutorials.  A spoken
tutorial is basically a carefully designed screencast for a roughly 10 minute
duration or less.  Any screencast cannot qualify as a spoken-tutorial.
Notably, a spoken tutorial should be made such that a newbie can understand
it.  The spoken-tutorial project ensures that all new tutorials undergo a
novice check to make sure that this is indeed the case.  A spoken tutorial
also requires a carefully written script.  This allows a spoken tutorial to be
dubbed into multiple languages.  A series of spoken tutorials can thus be used
to effectively teach a programming language or software package.

The spoken tutorial project has trained over a million students and teachers
on a variety of software packages.  The project hosts over 700 individual
spoken-tutorials.  Over 20 different Indian languages are supported.  Some
languages have better representation than others.

As part of the Python initiative we have created about 40 spoken tutorials to
teach non-CSE undergraduate students how to use Python for their curricular
computational tasks.  A new set of around 50 tutorials is currently being
recorded.  The spoken tutorials include tutorials on starting with IPython,
plotting with matplotlib, etc.  Currently these are only available in `English
<http://spoken-tutorial.org/tutorial-search/?search_language=English&search_foss=Python&page=1>`_.

Fig. :ref:`fig:py-st` shows a typical Python spoken tutorial.  It shows the
main screencast video.  Below the video is an outline of the tutorial.
Information on installation and other instructions is also listed.  Users can
also easily navigate to prerequisite tutorials.  In addition, users can post
their questions on the forum.

.. figure:: python_spoken_tutorial.png
   :alt: Python spoken tutorials on the spoken-tutorial.org website.

   An example of a Python spoken tutorial.  The video can be viewed, an
   outline of the material is available below the video.  An instruction sheet
   and installation sheet is also available.  Prerequisite videos are listed
   and users can also post questions on a forum. :label:`fig:py-st`

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
difficult for a developer who knows almost everything about the package.  On
the other hand it is not always easy for an inexperienced user to write
documentation.

Textbook companions offer an interesting approach to this task.  As discussed
in detail in [kmm14]_, textbook companions are created by writing Python code
for every solved example in a textbook.  Students create these textbook
companions which are then reviewed by either teachers reviewers at FOSSEE.
This task scales very well as students are eager to take up the task. Students
are given an honorarium of Rs. 10000 after their textbooks are reviewed.
Currently, there are over 530 Scilab textbook companions [STC]_ created.  The
Python project has 395 completed books with over 225 textbooks in progress.
The Python companions are hosted online at http://tbc-python.fossee.in

The Python Textbook Companions (PTC's) are submitted in the form of Jupyter
notebooks.  This is important for several reasons:

- Jupyter  notebooks allow us to put together, formatted HTML, code, and the
  results in one self-contained file.
- IPython notebooks are easy to render and a HTML listing can be generated.
- The file can also be hosted online and interactively used.
- The huge popularity of the notebook makes this a very useful resource.

The FOSSEE group has also customized the generated HTML such that users can
leave comments on the IPython notebooks.  This is done by linking disqus
comments to each rendered notebook.  The disqus API is then queried for any
new comments each day and contributors are sent a consolidated email about any
potential comments for them to address.  This feature is relatively new and
needs more user testing.

The submission process and hosting of the IPython notebooks is done using a
Django web application that can be seen at http://tbc-python.fossee.in.  The
code for the interface is also available from github
(https://github.com/FOSSEE/Python-TBC-Interface).  Once a textbook is reviewed
it is also committed to a git repository on github:
https://github.com/FOSSEE/Python-Textbook-Companions.

The process works as follows:

 1. The student picks a few possible textbooks that has not been completed and
    informs the textbook companion coordinator.
 2. Once a particular book is assigned to the contributor, the student submits
    one sample chapter which is reviewed by the coordinator.
 3. The student then completes the entire book.  Each chapter is submitted as
    a separate IPython notebook.
 4. The student also uploads a few screenshots of their favorite notebooks
    that are displayed.
 5. The textbook is reviewed and any corrections are made by the contributor.
 6. The notebooks are then committed to the git repository.
 7. The committed notebooks are automatically picked up by the TBC web
    application.

After the textbook is reviewed and accepted the student is sent an honorarium
for their work.  Fig. :ref:`fig:tbc-main` shows the main Python TBC interface
with information about the project and the editor's picks.

.. figure:: python_tbc_main.png
   :alt: The main landing page for the Python TBC site.

   The Django application which hosts the Python textbook
   companions. :label:`fig:tbc-main`


Fig. :ref:`fig:tbc-text` shows a typical textbook.  The Jupyter notebooks for
each chapter can be viewed or downloaded.  More information on the book itself
can be seen including an ISBN search link for the student to learn more about
a book, a link to the actual IPython notebook on github and other details are
also available.  The entire book can be downloaded as a ZIP file.

.. figure:: tbc_textbook.png
   :alt: A typical textbook shown on the TBC interface.

   A typical textbook is shown.  The figure shows some screenshots to pique
   the interest of the casual reader.  The Jupyter notebook corresponding to
   each chapter is listed and can be viewed or
   downloaded. :label:`fig:tbc-text`

Upon clicking a chapter, a typical rendered HTML file is seen.  This is seen
in Fig. :ref:`fig:tbc-render`.  A button to edit the chapter is seen, this
will fire up a tmpnb_ instance which allows users to easily modify and run the
code.  This makes it extremely convenient to view, modify, copy, and learn the
created content.  In the figure, one can see an icon for entering comments.
This links a disqus comment field at the bottom of the page.  This lists all
current comments and allows users to submit new comments on the particular
chapter.

.. figure:: tbc_render.png
   :alt: A rendered textbook chapter.

   A typical textbook chapter being rendered.  The button to edit examples of
   the chapter fires up a tmpnb_ instance so users can edit the code and try
   their changes. :label:`fig:tbc-render`


A large number of solved examples are indeed quite simple but there are
several that are fairtly involved.  Some of the submitted textbook companions
are very well made.  These are highlighted in the editor's pick section.

The Python TBC's have not been advertised too much yet and we have not been
keeping careful track of the number of hits.  We are planning to popularize
these more in the future.  It is still unclear as to how different people are
using the notebooks.  We do have very good feedback from the contributors to
the project.  Many of them have enjoyed creating these notebooks and have
benefitted by this effort.  Some contributor comments are quoted in [kmm14]_.


.. _tmpnb:  https://github.com/jupyter/tmpnb


SDES course
------------

Initially the Python group focussed on teaching Python at various colleges.
It was soon felt that this was not enough.  Students needed to learn how to
use Unix shells effectively, use version control, a bit of LaTeX, good
software development practices in addition to Python.  In order to fill this
need a course was designed in late 2009.  The course is titled Software
Development techniques for Engineers and Scientists (SDES).  This course takes
inspiration from the Software Carpentry Course material [SWC]_.  However, the
course is tailored for undergraduate students.  Two courses at IIT Bombay were
also created so students could take this as part of their course-work.

The course starts with teaching students on how to use Unix command line tools
to carry out common (mostly text processing) tasks.  The course then goes on
to teach students how to automate typical tasks using basic shell-scripting.
The students are then taught version control.  The course originally used
mercurial_, however, this has changed to git_.  The students are then taught
basic and advanced Python.  The emphasis is on typical engineering/numerical
computations such as those that involve (basic) manipulation of large arrays
in an efficient manner.  Good programming style is discussed along with
debugging and test driven development.  They also learn LaTeX and document
creation with reStructuredText_.  The course material is available from
github, at http://github.com/FOSSEE/sees.

As part of the evaluation students pick a software project and attempt to
apply all that they have learned.  Students are also given many programming
assignments to test their ability to program.  We have built a very convenient
online testing tool called Yaksh that is discussed in a subsequent section for
this task.  This makes online tests fun and very helpful for instructors to
assess student's understanding.

.. _mercurial: https://www.mercurial-scm.org
.. _git: https://git-scm.com/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html

The course has been offered twice and will be offered in the fall of 2016.
The course has been well received by students and is quite popular.  We
restrict the number of students to about 60 each time.  During the last
delivery we felt that the student projects were not done well enough.  A more
aggressive and systematic approach is needed to push students to work
consistently over the duration of the course, rather than in the last minute.
We also find that it is difficult for students and instructors to pick
meaningful projects that are neither too trivial or too difficult.  We plan to
push students a bit more aggressively to work systematically on their
projects.  We plan to use git logs to assess team contribution and systematic
work.  Instead of always picking new projects, we are thinking of giving
students a pool of existing projects and ask them to improve them.

While teaching this course is fun and is very useful, it does take a lot of
effort and a good team of TAs is necessary.  Fortunately, the FOSSEE Python
team helps in this regard.


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



Acknowledgments
----------------

This work would not be possible without the work of many of the FOSSEE staff
members involved in this project.


References
-----------

.. [kmm14] Kannan Moudgalya, Campaign for IT literacy through FOSS and Spoken
    Tutorials, Proceedings of the 13th Python in Science Conference, SciPy,
    July 2014.

.. [STC] Scilab Team at FOSSEE, Scilab textbook companions,
    http://scilab.in/Textbook_Companion_Project, May 2016.

.. [SWC] Greg Wilson.  Software Carpentry, http://software-carpentry.org,
    Seen on May 2016.
