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

A course in numerical methods should teach both the mathematical theory
of numerical analysis and the craft of implementing numerical algorithms.
The IPython notebook provides a single medium in which mathematics,
explanations, executable code, and visualizations can be combined, and
with which the student can interact in order to learn both the theory and the
craft of numerical methods.  The use of notebooks also lends itself naturally
to inquiry-based learning and the flipped classroom methodology.
I discuss the motivation and practice of teaching a course based on the use of
IPython notebooks and inquiry-based leanring, including some specific practical aspects.
The discussion is based on my experience teaching a Masters-level course
in numerical analysis at KAUST, but is intended to be useful for those
who teach at other levels or in industry.

.. class:: keywords

   IPython, IPython notebook, teaching, numerical methods, inquiry-based learning

Teaching numerical methods
==========================
Any course in numerical methods should enable students to:

- Understand relevant mathematical concepts like complexity, stability, and convergence
- Select an appropriate method for a given problem
- Implement the selected numerical algorithm
- Test and debug the numerical implementation

In other words, students should develop all the skills necessary to go from
a mathematical model to reliably-computed solutions.
These skills will allow them to select and use existing numerical software responsibly
and efficiently, and to create or extend such software when necessary.
The first two of the four objectives above, being primarily theoretical, are well suited to a
traditional university course format, with a textbook and lectures.  
Usually, only the first of these objectives is actually mentioned
in the course syllabus, and in some courses it is the only one taught.
But the other three objectives are likely to be of at least as much value to students
in their careers.  The last two skills are practical ones, and teaching them
properly is in some ways akin to teaching a craft.  Crafts, of course, are not
generally taught through lectures and textbooks; rather, one learns a craft by
*doing*.

.. As mentioned already, in some courses students are not required to implement
.. or test anything; only to perform theoretical analysis of algorithms.

Too often, implementation, testing, and debugging are viewed as "mundane" tasks that anyone
should be able to pick up incidentally.  
Some programming is required in order to complete the homework assignments.  
But usually no class time is spent on programming, so students learn
it on their own -- usually poorly and with much difficulty, due to the lack of
instruction.  This is connected to (and perhaps at the
root of) the low value placed on algorithmic implementation and software development
in the scientific community.


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
The teacher of numerical methods has several choices of 
programming language.  These can broadly be categorized as specialized high-level interpreted
languages (MATLAB, Mathematica, Maple) or general-purpose compiled languages (C, C++, Fortran).
High-level languages, especially MATLAB, are used most widely and have several advantages.
Namely, the syntax is very similar to the mathematical formulas themselves,
the learning curve is short, and debugging is relatively simple.
The main drawback is that such languages often do not provide the necessary performance
to solve large research or industrial problems.  This may be a handicap for students
if they are never exposed to compiled languages.

Python strikes a middle ground between these options.  It is a high-level language
with relatively intuitive mathematical syntax and high-level libraries for everything
needed in a course on numerical methods.  At the same time, it is a general purpose 
language.  Although it is slow (like MATLAB), Python makes it relatively easy to develop fast
code by using tools such as Cython or f2py.


The IPython notebook as a textbook medium
=========================================
Many print and electronic textbooks for numerical methods include code, either
printed on the page or available online (or both).  Some of my favorite
examples are [Tre00]_ and [LeV07]_.  Such books have become more and more common,
as the importance of exposing students to the craft of numerical methods -- and 
the value of experimentation in learning the theory -- has become
more apparent.  These books are an important step forward from older texts that only
talked *about* numerical methods.  I view the IPython notebook as the next step
in this evolution.  It combines in a single document

- Mathematics (using LaTeX)
- Text (using Markdown)
- Code (using Python or other languages)
- Visualizations (figures and animations that can be embedded from a file, from
  the internet, or produced from code)

It should be noted that media like the IPython notebook have existed for many years;
for instance, Mathematica, Maple, and (more recently) SAGE have document formats
with similar capabilities.  The SAGE worksheet is very similar to the IPython notebook
(indeed, the two projects have strongly influenced each other), so most of what
I will say about the IPython notebook applies also to the SAGE worksheet.

The notebook has some important advantages over Mathematica and Maple documents:

- It can be viewed, edited, and executed using only free software (in fact, with only a web browser)
- It is stored as text, which allows it to be version controlled, diff'ed, merged, etc.
- It allows the use of multiple programming languages
- It can be collaboratively edited by multiple users at the same time
 
Perhaps the most important advantage of the notebook is the community
in which it has developed -- a community in which openness and collaboration are the norm.
Because of this, those who develop teaching and research materials with IPython notebooks
usually make them freely available to anyone under extremely permissive licenses;
see for example Lorena Barba's AeroPython course [Bar14] or 
`this huge list of books, tutorials, and lessons <https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks>`_.
Due to this culture, the volume and quality of
available materials for teaching with the notebook is quickly surpassing what is
available in older proprietary formats.  It should be mentioned that the
notebook is also being used as a medium for publishing research, both in
open notebook science and for full articles.


Getting students started with the notebook
==========================================
One historical disadvantage of using Python for a course was the
difficulty of ensuring that all students had properly installed the
required packages.  Indeed, when I began teaching with Python 5 years ago,
this was still a major hassle even for a course with twenty students.
If even a few percent of the students have installation problems, it
can create an overwhelming amount of work for the instructor.

This situation has improved dramatically and is no longer a significant issue.
I have successfully used two strategies: local installation and cloud platforms.

Local installation
------------------
It can be useful for students to have a local installation of all the software
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
Students should be strongly encouraged to work together in developing
their programming skills.


Mechanics of an interactive, notebook-based course
==================================================
I have successfully used IPython notebooks as a medium of instruction in
both

- semester-length university courses; and
- short 1-3 day tutorials

I will focus on the mechanics of teaching a university course, but
much of what I will say applies also to short tutorials.
The notebook is especially advantageous in the context of a tutorial
because one does not usually have the luxury of requiring that students
purchase a textbook.  The notebooks for the course can comprise a complete,
self-contained curriculum.

The flipped classroom
---------------------
The term *flipped classroom* is becoming fairly well known in higher
education.  It refers to a teaching approach in which students read and
listen to recorded lectures outside of class.  Class time is then used
not for lectures but for more interactive pursuits, such as discussion, 
exercises, and quizzes.  The use of IPython notebooks and the teaching 
of the craft of numerical methods in general lends itself naturally to
the flipped classroom approach, since in-class time can be used for students
to work on implementing, testing, and understanding the methods.

Typically I have used a partially-flipped approach, in which half of the
class sessions are traditional lectures and the other half are *lab sessions*
in which the students spend most of the time programming and discussing
their programs.


What to do during lab sessions
------------------------------
At the beginning of each lab session, the students open a new notebook
that contains some explanations and exercises.  Generally they have already
been introduced to the algorithm in question, and the notebook simply 
provides a short review.  Early in the course, most of the code is provided
to the students already; the exercises consist mainly of extending or
modifying the provided code.  As the course progresses and students develop
their programming skills, they are eventually asked to implement some algorithms
or subroutines from scratch (or by starting from codes they have written previously).
Furthermore, the specifity of the instructions is gradually decreased as
students develop the ability to fill in the intermediate steps.

It is essential that students arrive to the lab session already prepared, 
through completing assigned readings or recordings.
I have found it very useful to administer a quiz at the beginning of class
to provide extra motivation.  Quizzes can also be administered just before
students begin a programming exercise, in order to check that they have a
good plan for completing it, or just after, to see how successful they were.

The main advantage of having students program in class (rather than at
home on their own) is that they can talk to the instructor and to other students
as they go.  Most students are extremely reluctant to do this at first,
and it is necessary to force them to explain to one another what their code
does (or is intended to do).  This can be accomplished by having them program
in pairs (alternating, with one programming while the other makes comments and 
suggestions).  Another option is to have them compare and discuss their codes
after completing an exercise.

When assisting students during the lab sessions, it is important not
to give too much help; i.e., don't immediately tell them what is wrong
or how to fix it.  Ask questions.  Help them learn to effectively read a
traceback and diagnose their code.  Even if they seem to have no problems, it's
worthwhile to discuss their code and help them develop good programming style.

Drawbacks of the interactive approach
-------------------------------------
Programming even simple algorithms takes a lot of time, especially for
students.  Therefore, the amount of material that can be covered in a
semester-length course on numerical methods is substantially less under the
interactive or flipped model.  This is true for inquiry-based learning
techniques in general, but even more so for a course in numerical methods.

.. e.g., to make it run faster, to handle more interesting


Designing effective notebooks
------------------------------
Prescribing how to structure the notebooks themselves is like 
stipulating the style of a textbook or lecture notes.  Each instructor
will have his or her own preferences.  So I will merely share some
principles I have found to be effective.

1. **Help students to discover concepts on their own first.**

2. **Don't make things too easy.**

3. **A picture is worth a thousand words.  And an animation is worth a million words.**


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
