.. raw:: latex

   \newcommand{\DUadmonitionnote}[1]{
     \begin{center}
        \fbox{\parbox{\columnwidth}{#1}}
      \end{center}
   }

:author: David I. Ketcheson
:email: david.ketcheson@kaust.edu.sa
:institution: King Abdullah University of Science and Technology

-------------------------------------------------------------------------
Teaching numerical methods with IPython notebooks
-------------------------------------------------------------------------

.. class:: abstract

Students in a numerical methods course should learn both the mathematical theory
of numerical analysis and the craft of implementing numerical algorithms.
The IPython notebook provides a single medium in which mathematics,
explanations, executable code, and visualizations can be combined, and
with which the student can interact in order to learn both the theory and the
craft of numerical methods.  The use of notebooks lends itself natural to
inquiry-based learning and flipped classrooms.
I discuss the motivation and methodology, of 
teaching such course based on the use of IPython notebooks, including
some specific practical aspects.
The discussion is based on my experience teaching a Masters-level course
in numerical analysis at KAUST, but is intended to be useful for those
who teach at other levels or in industry.

.. class:: keywords

   IPython, IPython notebook, teaching, numerical methods, inquiry-based learning

Teaching numerical methods
==========================
Any course in numerical methods should enable students to:

- Understand relevant mathematical concepts like complexity, stability, and convergence
- Implement a numerical algorithm
- Test and debug an implementation

These skills will allow them to select and use existing numerical software responsibly
and efficiently, and to create or extend such software when necessary.
The first of these objectives, being primarily theoretical, is well suited to a
traditional university course format, with a textbook and lectures.  
Usually, this is the only one of the three objectives that is actually mentioned
in the course syllabus, and in some courses it is the only one students achieve.
But the other two objectives are at least equally likely to be of value to students
in the various careers for which they may be preparing.  These latter skills
are more practical and teaching them properly is in some ways more
akin to teaching a craft (hence the term *software carpentry*).  Crafts, of course,
are not generally taught through lectures and textbooks; rather, one learns a 
craft by *doing*.

As mentioned already, in some courses students are not required to implement
or test anything; only to perform theoretical analysis of algorithms.
Implementation, testing, and debugging are viewed as "mundane" tasks that anyone
should be able to pick up incidentally.  This is connected to (and perhaps at the
root of) the low value placed on algorithmic implementation and software development
in the scientific community.

In other courses, some programming is required in order to complete the homework
assignments.  Usually, no class time is spent on programming, so students learn
it on their own, usually poorly and with much difficulty, due to the lack of
instruction.

Perhaps the most important reason for teaching implementation, testing, and debugging
is that these can and should be used to reinforce the theory.  The student who
learns about numerical instability by reading in a textbook will forget it
after the exam.  The student who discovers numerical instability by implementing
an apparently correct (but actually unstable) algorithm by himself and subsequently
learns how to implement a stable algorithm will remember and understand it much better.
Similarly, implementing an explicit solver for a stiff problem and then seeing the
speedup obtained with an appropriate implicit solver makes a lasting impression.

Finally, there are courses (often called "laboratory" courses) that do focus
on the implementation of numerical algorithms, generally using MATLAB, Mathematica,
or Maple.  Unfortunately, these courses are sometimes for less credit than a typical
mathematics or engineering course, with an attendant reduction in the time spent.

.. note::

    I have carefully used the term *numerical methods* here; many university courses
    (including those I teach) use instead the term *numerical analysis* in their titles.
    It is my contention that such courses, to be of the most use for students,
    should also include some introduction to all three of the skills above,
    except in the case of some very advanced (doctoral-level) graduate courses
    that may be purely theoretical.

Languages for teaching numerical methods
========================================
As mentioned already, the teacher of numerical methods has several choices for 
programming language.  These can broadly be categorized as specialized high-level
languages (MATLAB, Mathematica, Maple) or general-purpose compiled languages (C, C++, Fortran).
High-level languages, especially MATLAB, are used most widely and with good reason:
the code that students will write looks very similar to the formulas in the book;
the learning curve for students is shorter; and debugging is much simpler.
The main drawback is that such languages often do not provide the necessary performance
to solve large research or industrial problems.  This may be a handicap for students
if they are never exposed to compiled languages.

Python strikes a middle ground between these options.  It is a high-level language
with relatively intuitive mathematical syntax and high-level libraries for everything
needed in a course on numerical methods.  At the same time, it is a general purpose 
language.  Although it is slow, Python makes it relatively easy to develop fast
code by using tools like Cython or f2py.


The IPython notebook as a textbook medium
=========================================
Many print and electronic textbooks for numerical methods include code, either
printed on the page or available online (or both).  Some of my favorite
examples are [Tre00]_ and [LeV07]_.  Such books have become more and more common,
as the importance of exposing students to the craft of numerical methods has become
more apparent.  These books are an important step forward from older texts that only
talked *about* numerical methods.  I view the IPython notebook as the next step
in this evolution.  It combines in a single document

- Mathematics (using LaTeX)
- Text (using Markdown)
- Code (using Python or other languages)
- Visualizations (figures and animations from file, from the internet, or produced from code)

It should be noted that media like the IPython notebook have existed for many years;
for instance, Mathematica, Maple, and (more recently) SAGE have document formats
with similar capabilities.  The SAGE worksheet is very similar to the IPython notebook
(indeed, the two projects have strongly influenced each other), so most of what
I will say about the IPython notebook applies also to the SAGE worksheet.

The notebook has some important advantages over Mathematica and Maple documents:

- It can be viewed, edited, and executed using only free software
- It uses a text-based format, which allows notebooks to be diff'ed, merged, etc.
- There are free cloud services for viewing and running notebooks
- It allows the use of languages other than its native one
 
Perhaps the most important advantage of the notebook is the community
in which it has developed -- a community in which openness and collaboration are the norm.
Because of this, those who develop teaching and research materials with IPython notebooks
usually make them freely available to anyone under extremely permissive licenses;
see for example Lorena Barba's AeroPython course [Bar14] or a huge number of books, tutorials,
and lessons listed at [Ipy14].  Due to this culture, the volume and quality of
available materials for teaching with the notebook is quickly overtaking those
of the older formats.  It should be mentioned that the notebook is also being used
widely as a medium for publishing research, both in open notebook science and for
full articles.


Getting students started with the notebook
==========================================
One historical disadvantage of using Python for a course was the
difficulty of ensuring that all students had properly installed the
required packages.  Indeed, when I began teaching with Python 5 years ago,
this was still a major hassle even for a course with twenty students.
If even a few percent of the students have installation problems, it
can create an overwhelming amount of work for the instructor.

This situation has improved dramatically and is now not a major issue.
I have successfully used two strategies, described in the next two subsections.

Local installation
------------------
It is useful for students to have a local installation of all the software
on their own computer or a laboratory machine.  The simplest way to achieve 
this is to install either Anaconda_ or Canopy_.  Both are free and include
Python, IPython, and all of the other Python packages likely to be used
in any scientific course.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Canopy: https://www.enthought.com/products/canopy/


Cloud platforms
---------------
In order to avoid potential installation issues altogether, or as a
fallback option, it is possible to run a course where students only
need access to a computer with a web browser.  Two free platforms
exist for running IPython notebooks:

- `Sage Math Cloud <http://cloud.sagemath.org>`_
- `Wakari <http://wakari.io>`_

Both services are relatively new and are developing rapidly.
Both include all relevant Python packages by default.
I have used both of them successfully, though I have more experience
with Sage Math Cloud (SMC) as its interface seems somewhat more intuitive
to students.  Each SMC project is a complete sandboxed Unix environment, so it
is possible for the user to install additional software if necessary.
On SMC, it is even possible for multiple users to collaboratively edit notebooks
at the same time.


Teaching Python
---------------
Since students of numerical methods do not usually have much prior
programming experience, and what they have is usually in another
language, it is important to give students a solid foundation in Python
at the beginning of the course.  In the graduate courses I teach, I find
that most students have previously programmed in MATLAB and are easily
able to adapt to the similar syntax of Numpy.  However, some aspects of
Python syntax are much less intuitive.  Fortunately, a number of excellent
Python tutorials geared toward scientific users are available.
I find that a 1-2 hour laboratory session at the beginning of the course
is sufficient to acquaint students with the necessary basics; further
details can be introduced as needed later in the course.


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
.. [LeV07] R. J. LeVeque. *Finite Difference Methods for Ordinary and Partial Differential Equations*,
           Society for Industrial and Applied Mathematics, 2007.

.. [Tre00] L. N. Trefethen. *Spectral Methods in MATLAB*,
           Society for Industrial and Applied Mathematics, 2000.

.. [Bar14] L. A. Barba, O. Mesnard. *AeroPython*,  10.6084/m9.figshare.1004727. Code repository,
            Set of 11 lessons in classical Aerodynamics on IPython Notebooks. April 2014.

.. [ipy14] *A gallery of interesting IPython notebooks*,
           https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks.
