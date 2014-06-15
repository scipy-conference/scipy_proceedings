:Author: Kannan M. Moudgalya
:email: kannan@iitb.ac.in
:institution: Dept. of Chemical Engineering and Education Technology Group IIT Bombay, India

===============================================================
Campaign for IT literacy through FOSS and Spoken Tutorials
===============================================================

.. class:: abstract

  A Spoken Tutorial is a ten minute audio video tutorial on open source
  software, created to provide training on important IT topics. Spoken
  Tutorials are created for self learning, using the pedagogical methods
  developed at IIT Bombay. The spoken part of these tutorials is dubbed in all
  Indian languages, to help children who are weak in English. The main
  objective of the Spoken Tutorial project is to improve the employment
  potential. At present, there are about 500 spoken tutorials in English and
  2,000 dubbed tutorials in other Indian languages, with the Python statistics
  being 51 and 108, respectively. Although one can self-learn from spoken
  tutorials, we also organise effective workshops through students and
  volunteers. In the past three years, we have trained close to 400,000
  students. Details of this effort are at http://spoken-tutorial.org. We have
  so far trained about 20,000 college students and teachers in Python alone.

  The Spoken Tutorial activity is supported by the FOSSEE project, one of the
  main achievements of which is massive document creation for open source
  software through collaboration. For example, we have created close to 100
  Python Textbook Companions, while about 100 more are in progress. The FOSSEE
  project promotes Scilab, Python, OpenFOAM, COIN-OR and Oscad (a locally
  developed Electronic Design Automation tool) and Sandhi (a locally developed
  open source alternative to LabVIEW). Python is used extensively in Oscad and
  Sandhi.

  Our team is coordinating the work on Aakash, the world’s lowest cost
  computing device. We have ported Linux on to Aakash and hence, most programs
  that run on Linux are also available on Aakash. We have also ported the
  following applications that are of special interest to schools: spoken
  tutorials, animations, school books and open source accounting software.
  Aakash is a versatile tool that can be made available to most children,
  because of its low cost.

.. class:: keywords

    Python, spoken tutorials, FOSSEE

Introduction
=============


Textbook Companion Project
==========================

One of the major shortcomings of FOSS tools is the lack of
documentation. Proprietary software creators can deploy a lot of money
and other resources to develop good documentation. In this project, we
decided to address this important issue through Textbook Companions.

We wanted to create documents for FOSS using our abundantly available
work force, namely, students. Unfortunately, creating a document
requires mature people. Students are good in writing programs, not
documents. We addressed this by solving the inverse problem: ask the
students to write programs for existing documents. Textbooks can be
considered as good documents.

Textbook companion (TBC) activity creates code for solved examples of
standard textbooks using FOSS. These are created by students and the
faculty of colleges from different parts of India. Students who create
these books are given an honorarium of Rs. 10,000 for each companion and
their teachers who helped review are given an honorarium of Rs. 5,000
per companion.

If anyone wants to understand what a program does, all that they have to
do is to go through the corresponding example in the associated
textbook. If TBCs are available for all textbooks used in our
educational programmes, we would not need proprietary software, at least
for classroom use.

Anyone can contribute to the Python Textbook Companion (PTC) activity;
from students to corporates, teachers and freelancers. Participants
choose any textbook of their choice from any engineering or science
background. They may convert all the solved examples of a particular
textbook into Python codes. Upon successful completion of creating a
PTC, the participant is awarded with a certificate and a handsome
honorarium. PTCs are presented in the form of IPython Notebooks.

The PTC interface (http://tbc-python.fossee.in/) displays all the
completed books together with a screen-shot of code snippets, so that
the user can easily download the PTC of their interest. The interface
also allows the users to view all the codes of a chapter as a IPyton
notebook, which makes learning python much easier. We have the textbook
companions of the following category of books:

-  Fluid Mechanics,

-  Chemical Engineering,

-  Thermodynamics,

-  Mechanical Engineering,

-  Signal Processing,

-  Digital Communications,

-  Electrical Technology,

-  Mathematics & Pure Science,

-  Analog Electronics,

-  Computer Programming and others.

Currently, there are 89 completed PTCs and 87 are in progress. PTCs so
created are available for free download at http://tbc-python.fossee.in/.

By creating textbook Companions in python, participants are able to
learn Python in a practical and effective way. Following are the
testimonials from few of the participants:

    I experienced that even an inexperienced person can do
    coding/programming. I gradually got to advance my skills in python
    as I approached further in it. I got the IIT-B certificate, plus i
    got paid a handsome amount of cheque after completion which was good
    enough for me at then.

    The FOSSE-Textbook Companion Project has been a scintillating
    point in my career. It has been instrumental in fine tuning my
    programming and presentation skills. It provided an ideal forum for
    me to learn Python and contribute to the open source community.

    I learnt python from Spoken-Tutorials available on the
    website.The Python TBC team also helped me a lot in starting my
    internship. Till now,I have completed 3 TBCs and now,I know pretty
    much about python.I plan to take this project forward and Python is
    really helping me shine by resume.

    This internship provided me a perfect platform and environment to
    learn python.It helped me to incorporate both my ideas and practical
    work skills to the best.Especially,those concepts of C which are not
    present in python gave me an awesome experience.Moreover, experience
    gained from it will make me capable of facing and overcoming the
    upcoming challenges under its applications.

We would like to point out some of the processes we have followed in the
creation of PTC. Initially we tried to use the Sprint route to create
PTCs. Unfortunately the progress was slow, as there was no ownership. In
contrast, the Scilab group used the approach explained in this section
and found it to be more effective, and more productive. As a result, the
Python group also changed the strategy for the creation of PTCs and has
yielded good results, as explained above, in a short time.

Spoken Tutorials
================

A Spoken Tutorial is an audio - video instructional material created for
self learning through the Screencast technology. The objectives of the
ST effort are:

#. To create documentation for Free and Open Source Software (FOSS).
   Restriction to FOSS promotes active learning, along with other
   benefits. In a typical spoken tutorial of ten minute duration, there
   are about one hundred screen transitions. It is a lot more difficult
   to create an equally effective pdf document using screenshots.
   Generally, the Screencasts are known to be more effective to
   understand a new topic [11].

#. To make every ST suitable for self learning, given that there is a
   big shortage of good teachers in India.

#. To conduct workshops using ST, so as to make it easy for students to
   access the instructional material.

#. To make IT literacy training accessible to students weak in English,
   without affect- ing their employment potential.

#. To come up with a mechanism to conduct tests for the participants of
   ST based workshops and to issue certificates to the passing students.

All of the above should be made available free of cost to the learners.

The Python Team created a set of 14 Spoken Tutorials on Python at the
beginning. On using these tutorials, it was found that the pace of some
tutorials were fast and that some topics were left out. A fresh set of
37 Spoken Tutorials were created. These have also been dubbed into a few
Indian languages. At present, we have the following Python Spoken
Tutorials at the basic level:

Getting started with ipython

Using the plot command interactively Embellishing a plot

Saving plots

Multiple plots

Additional features of IPython Loading data from files

Plotting the data

Other types of plots

Getting started with sage notebook Getting started with symbolics

Using Sage

Using sage to teach

At the intermediate level, we have the following tutorials:

Getting started with lists

Getting started with for

Getting started with strings

Getting started with files

Parsing data

Statistics

Getting started with arrays

Accessing parts of arrays

Matrices

Least square fit

Basic datatypes and operators IO

Conditionals

Loops

Manipulating lists

Manipulating strings

Getting started with tuples

Dictionaries

Sets

At the advanced level, we have the following tutorials:

Getting started with functions

Advanced features of functions

Using python modules

Writing python scripts

Testing and debugging

The Python Spoken tutorials help in training students and faculty across
the nation through SELF workshops conducted remotely from IIT Bombay.
This is easy, because the tutorials are created for self learning. We
have stream lined the processes involved in the conduct of the workshops
so that any student or teacher or any other volunteer can conduct these
workshops in their premises. Their role is to ensure that the correct
processes are followed. Actual learning takes place through the Spoken
Tutorials only. As a result, there is no dilution of learning, which is
possible in a typical train the trainer type of methods.

After the workshop, the students are encouraged to download the
tutorials and to practise by themselves. An online exam is conducted a
few weeks after the workshop and the participants who pass the exam are
provided with a certificate for successful completion of training. After
workshop(s), students post queries, if any, on Spoken Tutorial Forum
(http://forums.spoken-tutorial.org/) and are given quick answers from
the domain experts.

We present some statistics of the people who have undergone Python SELF
workshops. The number of SELF workshops conducted until now is 417,
training close to 19,000 students. It is interesting to note that 9,300
of them are girls. It is interesting because generally girls take up
programming in much less percentage, here it is almost 50%. Python SELF
workshops have taken place in 23 states of India. The number of
workshops and the participants tuple in 2011, 2012, 2013 and 2014,
respectively are, (21, 945), (144,6562), (116,4857) and (136,6439). One
can see that but for 2013, there is a steady growth. It should be
pointed out that less than one half of the year is over in 2014.

The Python SELF workshops are effective. We have the following
testimonials:

    Through this workshop one can easily understand the basics of
    python,which in turn can develop an interest in one’s mind to learn
    more about python. Thank you very much for this wonderful workshop.

    Got the initiative of how to work on python that makes the
    programming comparative easy.Apart that the graphical representation
    of mathematical formulation are quite good.

    It is a very efficient way of learning new languages as the videos
    seem to be more practical and helps the learning of the language
    along with the examples.

Acknowledgements
----------------

The work reported in this article has been done by the 100+ staff
members of the FOSSEE and Spoken Tutorial teams. The author wishes to
acknowledge the contributions of the Principal Investigators of these
projects, Prabhu Ramachandran and Madhu Belur, and the full time staff
members of the FOSSEE project, Jovina and Hardik, in writing this paper.
