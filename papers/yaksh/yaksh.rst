:author: Prabhu Ramachandran
:email: prabhu@aero.iitb.ac.in
:institution: Department of Aerospace Engineering
:institution: IIT Bombay, Mumbai, India
:corresponding:

:author: Prathamesh Salunke
:email: pratham920@gmail.com

:author: Ankit Javalkar
:email: ankitrj.iitb@gmail.com

:author: Aditya Palaparthy
:email: aditya94palaparthy@gmail.com

:author: Mahesh Gudi
:email: mahesh.p.gudi@gmail.com

:author: Hardik Ghaghada
:email: hardy_the1@yahoo.com


--------------------------------------
Yaksh: Facilitating Learning by Doing
--------------------------------------

.. class:: abstract

    Yaksh is a free, and open-source online evaluation platform. At its core,
    Yaksh focuses on problem-based learning and lets teachers create practice
    exercises and quizzes which are evaluated in real-time. With a large array
    of question types like multiple choice, fill-in-the-blanks, assignment
    upload and assertion or standard I/O based programming questions
    available, Yaksh supports Python, C, C++, Java, Bash and Scilab
    programming languages. In addition, Yaksh allows teachers to create
    full-blown courses with video and/or markdown text-based lessons. Yaksh is
    designed to be secure, easily deployable, and can scale-up to 500+ users
    simultaneously.


Introduction
-------------

Yaksh_ is created by the `FOSSEE Python team`_. The `FOSSEE project`
(http://fossee.in) based at IIT Bombay, is funded by the Ministry of Human
Resources and Development, MHRD (http://mhrd.gov.in) of the Government of
India. The goal of the FOSSEE project is to increase the adoption of Free and
Open Source Software in Education in India. The project started in 2009 and
develops, and promotes a variety of open source projects. FOSSEE's Python
group attempts to promote the adoption of Python in India. More details on the
activities of the Python group of FOSSEE have been presented earlier at SciPy
2016 [PR2016]_. Yaksh was described briefly there. However, Yaksh has evolved
considerably in the last few years. It has been used for several courses at
IIT Bombay as well as online. Yaksh provides a simple interface to host a MOOC
and we discuss this feature as well.

As part of FOSSEE's efforts we have created learning material for Python and
have conducted hundreds of workshops on Python. We find that to effectively
train people to learn to program, it is imperative to make them solve
programming problems.  Yaksh_ has been created by FOSSEE for this purpose.


.. _`FOSSEE Python team`: https://python.fossee.in
.. _`FOSSEE project`: https://fossee.in
.. _Yaksh: https://github.com/FOSSEE/online_test

The FOSSEE project is based out of IIT Bombay and promotes the use of various
open source packages in order to help eliminate the use of proprietary
packages in the curriculum.


Overview of Yaksh
---------------------

Since the emergence of learning management system (LMS) and massive open
online course (MOOC) providers, e-learning has grown significantly. Despite
the ever increasing adopters, major platforms still use simple question types
like multiple-choice questions and uploading of assignments from students as a
means to evaluate students' performance. Yaksh seeks to improve on this.

It is well known that practice assignments and problem solving improve
understanding. In the case of programming languages, this is especially so.
Programming is a skill and to develop it, one must necessarily write programs.
By providing an interface where users can attempt a question and immediately
obtain feedback on the correctness of their program would be very useful both
to a student and also to a teacher. This same interface could also be used to
assess the performance of the student and assess how much the student has
understood. This is useful for the student to understand where they can
improve and to the teacher to find out which concepts are not properly
understood by the students. In the Indian context, a recent study [AM2017]_
that even though there are many graduates from a computer science background,
that only 5% of the students are able to write the correct logic for the
program. Indeed, our own experience is that many students learn computer
science theoretically without writing too many computer programs. It is
therefore important to provide a tool that facilitates practice programming
and programming assessment.

In 2011, the first version of Yaksh was developed to administer programming
quizzes for an online teacher training course that FOSSEE conducted. More than
600 teachers were trained and we wanted them to be able to write programs and
have those corrected. This work was presented in at SciPy India 2011 [PR11]_.
It would have been impossible to do this manually.

Yaksh is a free, and open-source online evaluation software that allows
teachers to create courses and students to watch lessons and attempt tests
which are evaluated immediately. Yaksh is designed to be used by a large
number of users concurrently thereby making it apt for using in schools,
colleges and other educational institutes for training a large number of
students.

Yaksh is implemented in Python and uses Django
(https://www.djangoproject.com/). It is also written as a pip-installable
Django app, thus allowing other Django based web projects to install the app
within their project. The sources are available from:
https://github.com/FOSSEE/online_test

To use Yaksh, one could sign-up on the official https://yaksh.fossee.in
website or host it on one's own servers. The most standard and secure way to
deploy Yaksh on a server is to build separate docker images using docker
compose. Instructions for this are available in the yaksh sources and are easy
to setup.

For teachers, Yaksh provides a wide array of question types which include the
basic question types like multiple choice, fill-in-the-blanks, assignment
upload, etc. One can also add standard I/O and assertion test cases based
questions for simple and basic programming questions. Also, for complex
programs, teachers can add a hook-based test case which would enable them to
take the student answer and evaluate it in whatever way they want. Once the
questions are created, they can create a question paper that can be added to a
practice exercise or a quiz. The question paper can have a mixed set of fixed
questions or a random set of questions selected from a pool of questions. In
conjunction with quizzes, teachers can also add video or markdown text-based
lessons. With that, teachers can also monitor students real time during a
test, as well as their overall progress for the course, there by gaining
insight on how students are performing.

Yaksh is designed to be easy-to-use by a student. All they have to do is
sign-up, enroll for a course and start. They could go through the lessons,
practice a few questions and then attempt the quiz, on which their performance
is rated. While doing so, they get easy to understand feedback for their
answers from the interface, thereby improving their answers.

Yaksh is being used extensively by the FOSSEE team to teach Python to many
students all across India. Over 3000 students have used the interface to learn
Python. It has been used in several courses taught at IIT Bombay and also for
conducting recruitment interviews internally.

In this talk we first demonstrate yaksh and then discuss its features and
implementation. We show an example of a complete Python course that we host at
FOSSEE using yaksh.

Yaksh is created and maintained by the Python team at FOSSEE
(http://fossee.in), based at IIT Bombay. Yaksh is completely free and
open-source, distributed under BSD license. The source code can be found
https://github.com/FOSSEE/online_test/

XXX old figures included here as examples.

.. figure:: yaksh_login.png
   :alt: Yaksh login screen.

   The Yaksh application login screen with a video on how one can use
   it. :label:`fig:yaksh-login`

.. figure:: yaksh-mcq.png
   :alt: Yaksh interface for an MCQ question.

   The interface for a multiple-choice question on
   yaksh. :label:`fig:yaksh-mcq`

.. figure:: yaksh-code.png
   :alt: Yaksh interface for a programming question.

   The interface for a programming question on yaksh. :label:`fig:yaksh-code`

.. figure:: yaksh_monitor.png
   :alt: Yaksh interface for monitoring student progress.

   The moderator interface for monitoring student progress during an exam on
   yaksh. :label:`fig:yaksh-monitor`

XXX Examples showing how to put in an image and how to refer it.  Redo this.

Fig. :ref:`fig:yaksh-login` shows the login screen for Yaksh.

Fig. :ref:`fig:yaksh-mcq` shows the interface for an MCQ question.
Fig. :ref:`fig:yaksh-code` shows the interface for a programming question.

Fig. :ref:`fig:yaksh-monitor` shows a typical moderator interface while
monitoring a running quiz.


Installation and setup
~~~~~~~~~~~~~~~~~~~~~~~~

XXX

The demo course/exams
~~~~~~~~~~~~~~~~~~~~~~~


Basic features
---------------

- For a student.
  - The generic interface and how quizzes etc. are taken.

- For an instructor

  - Different question types, their use.
  - Stdio
  - Assertion
  - Philosophy of allowing multiple submissions to make it easier.
  - Assignment upload.





Internal design
~~~~~~~~~~~~~~~~


The two essential pieces:

- Code server
- Django interface


Code server internal details and tornado interface.

- Sandboxing.
- Handling infinite loops.
- Docker.
- Logging of the answers.


Supporting a new language, the Grader etc.

Django models and overall approach.

Use of docker.

An example of interfacing yaksh on the ST website.

Additional plugin app support.

Import and export.

API?



Some experiences using yaksh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Usage in the internal courses at IIT, AE 102, SDES etc.

- Usage for hiring!

- Usage for practice

- Usage for full MOOC course




Plans
~~~~~~

New features planned.

Things already under way.

Other features we are thinking of.



Conclusions
------------



Acknowledgments
----------------

FOSSEE would not exist but for the continued support of MHRD and we are
grateful to them for this. This work would not be possible without the efforts
of the many FOSSEE staff members. The past and present members of the project
are listed here: http://python.fossee.in/about/ the author wishes to thank
them all.


References
-----------

.. [PR2016] Prabhu Ramachandran, Spreading the Adoption of Python in India: the
    FOSSEE Python Project", Proceedings of the 15th Python in Science
    Conference (SciPy 2016), July 6-12, 2016, Austin, Texas, USA.
    http://conference.scipy.org/proceedings/scipy2016/prabhu_ramachandran_fossee.html

.. [kmm14] Kannan Moudgalya, Campaign for IT literacy through FOSS and Spoken
    Tutorials, Proceedings of the 13th Python in Science Conference, SciPy,
    July 2014.

.. [FOSSEE-Python] FOSSEE Python group website.  http://python.fossee.in, last
    seen on May 7 2018.

.. [PR11] Prabhu Ramachandran.  FOSSEE: Python and Education, Python
    for science and education, Scipy India 2011, 4th-11th December 2011,
    Mumbai India.

.. [AM2017] 95% engineers in India unfit for software development jobs,
    claims report.  http://www.aspiringminds.com/automata-national-programming-report
