.. raw:: latex

   \newcommand{\DUadmonitionnote}[1]{
     \begin{center}
        \fbox{\parbox{\columnwidth}{#1}}
      \end{center}
   }

:author: David I. Ketcheson
:email: david.ketcheson@kaust.edu.sa
:institution: King Abdullah University of Science and Technology

----------------------------------------------------------------------------
Teaching numerical methods with IPython notebooks and inquiry-based learning
----------------------------------------------------------------------------

.. class:: abstract

A course in numerical methods should teach both the mathematical theory
of numerical analysis and the craft of implementing numerical algorithms.
The IPython notebook provides a single medium in which mathematics,
explanations, executable code, and visualizations can be combined, and
with which the student can interact in order to learn both the theory and the
craft of numerical methods.  The use of notebooks also lends itself naturally
to inquiry-based learning and the flipped classroom methodology.
I discuss the motivation and practice of teaching a course based on the use of
IPython notebooks and inquiry-based learning, including some specific practical aspects.
The discussion is based on my experience teaching a Masters-level course
in numerical analysis at KAUST, but is intended to be useful for those
who teach at other levels or in industry.

.. class:: keywords

   IPython, IPython notebook, teaching, numerical methods, inquiry-based learning

Teaching numerical methods
==========================
Any course in numerical methods should enable students to:

- **Understand** relevant mathematical concepts like complexity, stability, and convergence
- **Select** an appropriate method for a given problem
- **Implement** the selected numerical algorithm
- **Test** and debug the numerical implementation

In other words, students should develop all the skills necessary to go from
a **mathematical model** to **reliably-computed solutions**.
These skills will allow them to select and use existing numerical software responsibly
and efficiently, and to create or extend such software when necessary.
Usually, only the first of these objectives is actually mentioned
in the course syllabus, and in some courses it is the only one taught.
But the other three objectives are likely to be of at least as much value to students
in their careers.  The last two skills are practical, and teaching them
properly is in some ways akin to teaching a craft.  Crafts, of course, are not
generally taught through lectures and textbooks; rather, one learns a craft by
*doing*.

.. The first two of the four objectives above, being primarily theoretical, are well suited to a traditional university course format, with a textbook and lectures.  
.. As mentioned already, in some courses students are not required to implement
.. or test anything; only to perform theoretical analysis of algorithms.

Over the past few years, I have shifted the emphasis of my own numerical courses
in favor of addressing all four objectives.  In doing so, I have drawn
on ideas from *inquiry-based learning* and made heavy use of SAGE worksheets
and IPython notebooks as an instructional medium.  I've found this approach
to be enjoyable, and students have told me (often a year or more after completing the
course) that it was particularly helpful to them.  Because the approach described here
works best if the instructor is able to interact directly at times with each
student, I have found this approach to be worthwhile when the number of the
students in the course is less than twenty.


The value of practice in computational mathematics
==================================================
Too often, implementation, testing, and debugging are viewed as "mundane" tasks that anyone
should be able to pick up incidentally.  In most courses,
some programming is required in order to complete the homework assignments.  
But usually no class time is spent on programming, so students learn
it on their own -- often poorly and with much difficulty, due to the lack of
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

It should be noted that many universities have courses (often called
"laboratory" courses) that do focus on the implementation or application of
numerical algorithms, generally using MATLAB, Mathematica, or Maple.
Such courses may end up being those of most lasting usefulness to many students. 
The tools and techniques discussed in this article could very aptly be applied therein.
Unfortunately, these courses are sometimes for less credit than a normal
university course, with an attendant reduction in the amount of material that
can be covered.

Inquiry-based learning
======================
    *The best way to learn is to do; the worst way to teach is to talk.*
    --P. R. Halmos [Hal75]_

Many great teachers of mathematics (most famously, R.L. Moore) have argued
against lecture-style courses, in favor of an approach in which the students
take more responsibility and there is much more in-class interaction.
The many related approaches that fit this description have come to be called
*inquiry-based learning* (IBL).  In an inquiry-based mathematics course, students
are expected to find the proofs for themselves -- with some assistance from the
instructor.
For a very recent review of what IBL is and the evidence for
its effectiveness, see [Ern14a]_, [Ern14b]_ and references therein.
If an active, inquiry-based approach is appropriate for the teaching of
theoretical mathematics, then certainly it is even more appropriate for
teaching the practical craft of computational mathematics.

A related notion is that of the *flipped classroom*.
It refers to a teaching approach in which students read and
listen to recorded lectures outside of class.  Class time is then used
not for lectures but for more active, inquiry-based learning through things like discussions, 
exercises, and quizzes.  

As we will see, the use of IPython notebooks and the teaching 
of the craft of numerical methods in general lends itself naturally to
inquiry-based learning and flipped classrooms.

, since in-class time can be used for students
to work on implementing, testing, and understanding the methods.


Languages for teaching numerical methods
========================================
The teacher of numerical methods has several choices of 
programming language.  These can broadly be categorized as 

- specialized high-level interpreted languages (MATLAB, Mathematica, Maple) 
- general-purpose compiled languages (C, C++, Fortran).

High-level languages, especially MATLAB, are used most widely and have several advantages.
Namely, the syntax is very similar to the mathematical formulas themselves,
the learning curve is short, and debugging is relatively simple.
The main drawback is that such languages often do not provide the necessary performance
to solve large research or industrial problems.  This may be a handicap for students
if they never gain experience with compiled languages.

Python strikes a middle ground between these options.  It is a high-level language
with relatively intuitive mathematical syntax and high-level libraries for everything
needed in a course on numerical methods.  At the same time, it is a general purpose 
language.  Although it can be relatively slow [VdP14]_ (like MATLAB), Python makes it
relatively easy to develop fast code by using tools such as Cython or f2py.
For the kinds of exercises used in most courses, pure Python code is sufficiently fast.

IPython [Per07]_ is a tool for using Python interactively.  One of its most
useful components is the IPython notebook: a document format containing text, code,
and images that can be viewed, written, and executed in a web browser.

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
- Figures and animations

It should be noted that media like the IPython notebook have existed for many years;
for instance, Mathematica, Maple, and (more recently) SAGE have document formats
with similar capabilities.  The SAGE worksheet is very similar to the IPython notebook
(indeed, the two projects have strongly influenced each other), so most of what
I will say about the IPython notebook applies also to the SAGE worksheet.

The notebook has some important advantages over Mathematica and Maple documents:

- It can be viewed, edited, and executed using only **free** software (in fact, with only a web browser)
- It is stored as **text**, which allows it to be version controlled, diff'ed, merged, etc.
- It allows the use of multiple programming languages
- It can be collaboratively edited by multiple users at the same time
 
Perhaps the most important advantage of the notebook is the community
in which it has developed -- a community in which openness and collaboration are the norm.
Because of this, those who develop teaching and research materials with IPython notebooks
often make them freely available under permissive licenses;
see for example Lorena Barba's AeroPython course [Bar14] or 
`this huge list of books, tutorials, and lessons <https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks>`_.
Due to this culture, the volume and quality of
available materials for teaching with the notebook is quickly surpassing what is
available in older proprietary formats.  It should be mentioned that the
notebook is also being used as a medium for publishing research, both as
open notebook science and full articles.


Getting students started with the notebook
==========================================
One historical disadvantage of using Python for a course was the
difficulty of ensuring that all students had properly installed the
required packages.  Indeed, when I began teaching with Python 5 years ago,
this was still a major hassle even for a course with twenty students.
If just a few percent of the students have installation problems, it
can create an overwhelming amount of work for the instructor.

This situation has improved dramatically and is no longer a significant issue.
I have successfully used two strategies: local installation and cloud platforms.

Local installation
------------------
It can be useful for students to have a local installation of all the software
on their own computer or a laboratory machine.  The simplest way to achieve 
this is to install either Anaconda_ or Canopy_.  Both are free and include
Python, IPython, and all of the other Python packages likely to be used
in any scientific course.  Both can easily be installed on Linux, Mac, and
Windows systems.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Canopy: https://www.enthought.com/products/canopy/


Cloud platforms
---------------
In order to avoid potential installation issues altogether, or as a
secondary option, it is possible to run a course where students only
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
<<<<<<< Updated upstream
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
because one does not usually have the luxury of ensuring that students
have a textbook.  The notebooks for the course can comprise a complete,
self-contained curriculum.

Typically I have used a partially-flipped approach, in which half of the
class sessions are traditional lectures and the other half are *lab sessions*
in which the students spend most of the time programming and discussing
their programs.  Others have used IPython notebooks with a fully-flipped
approach; see for example [Bar13]_.


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
Furthermore, the specificity of the instructions is gradually decreased as
students develop the ability to fill in the intermediate steps.

It is essential that students arrive to the lab session already prepared, 
through completing assigned readings or recordings.
This doesn't mean that they already know everything contained in the notebook
for that day's session; on the contrary, class time should be an opportunity
for guided discovery.
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
suggestions).  Another option is to have them compare and discuss their code
after completing an exercise.

When assisting students during the lab sessions, it is important not
to give too much help; i.e., don't immediately tell them what is wrong
or how to fix it.  Ask questions.  Help them learn to effectively read a
traceback and diagnose their code.  Let them struggle a bit to figure out
why the solution blows up.  Even if they seem to grasp things immediately, it's
worthwhile to discuss their code and help them develop good programming style.

Typically, in an 80-minute class session the students spend 50-60 minutes
working (thinking and programming) and 20-30 minutes
listening to explanations, proposing ideas, discussing their solutions, and
taking quizzes.  During the working time, the instructor should assess and help
students one-on-one as needed.


Designing effective notebooks
=============================
Prescribing how to structure the notebooks themselves is like 
stipulating the style of a textbook or lecture notes.  Each instructor
will have his or her own preferences.  So I will merely share some
principles I have found to be effective.

Make sure that they type code
-----------------------------
This goes without saying, but it's especially important early in the course.
It's possible to write notebooks where all the code involved is
already completely provided.  That's fine if students only need
to understand the output of the code, but not if they need to 
understand the code itself (which they generally do).  The plain truth
is that nobody reads code provided to them unless they have to,
and when they do they understand only a fraction of it.
Typing code, like writing equations, dramatically increases the
degree to which students internalize concepts.  At the very
beginning of the course, it may be helpful to have students
work in an IPython session and type code from a notebook into
the IPython prompt.


Help students to discover concepts on their own
-----------------------------------------------
This is the central principle of inquiry-based learning.
Students are more motivated, gain more understanding, and retain
knowledge better when they discover things through their own
effort.  In a numerical methods course, the traditional approach is
to lecture about instability or inaccuracy, perhaps showing an example
of a method that behaves poorly.  In the flipped approach, you can instead
allow the students to implement and experiment in class with naive algorithms
that seem reasonable but may be inaccurate or unstable.  Have them discuss what
they observe and what might be responsible for it.  Ask them how they think the
method might be improved.

Tailor the difficulty to the students' level
--------------------------------------------
Students will quickly lose interest if either they are not challenged
or they find the first exercise insurmountable.  It can be difficult
to accommodate the varying levels of experience and skill presented by
students in a course.  For students who struggle with programming, peer
interaction in class is extremely helpful.  For students who advance
quickly, the instructor can provide additional, optional, more challenging
questions.

Gradually build up complexity 
-----------------------------
In mathematics, one learns to reason about highly abstract objects by
building up intuition with one layer of abstraction at a time.
Numerical algorithms should be developed and understood in the same
way, with the building blocks first coded and then encapsulated as
subroutines for later use.  For instance, when teaching multigrid
I have students code things in the following sequence:

1. Jacobi's method 
2. Under-relaxed Jacobi
3. A two-grid method,
4. The V-cycle
5. Full multigrid

In each step, the code from the previous step becomes a subroutine.
In addition to being an aid to learning, this approach teaches students
how to code well.

Use animations liberally
------------------------
Solutions of time-dependent problems are naturally depicted as
animations.  Printed texts must restrict themselves to waterfall
plots or snapshots, but electronic media can show solutions in the
natural way.  Students learn more -- and have more fun -- when they
can visualize the results of their work in this way.  I have used
Jake Vanderplas' JSAnimation package [VdP13]_ to easily create such animations.
The latest release of IPyton (version 2.0) directly includes interactive
widgets that can be used to animate simulation results.

But time-dependent solutions are not the only things you can animate.
How does the solution change after each algorithmic iteration?  
What effect does this parameter have on the results?
Such questions can be answered most effectively through the use of
an animation.

Drawbacks
==========
The approach proposed here differs dramatically from a traditional course
in numerical methods.  I have tried to highlight the advantages of this
approach, but of course there are also some potential negatives.

Material covered
-----------------
The most substantial drawback I have found relates to the course coverage.
Programming even simple algorithms takes a lot of time, especially for
students.  Therefore, the amount of material that can be covered in a
semester-length course on numerical methods is substantially less under the
interactive or flipped model.  This is true for inquiry-based learning
techniques in general, but even more so for courses that involve programming.
I believe that showing students the joy, beauty, and usefulness of
numerical mathematics has more impact than the length of the syllabus
on their long-term learning.

Nonlinear notebook execution
-------------------------------
Code cells in the notebook can be executed (and re-executed) in any
order, any number of times.  This can lead to different results than
just executing all the cells in order, which can be confusing to students.
I haven't found this to be a major problem, but students should be
aware of it.

Opening notebooks
-----------------
Perhaps the biggest inconvenience of the notebook is that opening one
is not as simple as clicking on the file.  Instead, one must
open a terminal, go to the appropriate directory, and launch the ipython
notebook.  This is fine for users who are used to UNIX, but is non-intuitive
for some students.  With IPython 2.0, one can also launch the notebook from any
higher-level directory and then navigate to a notebook file within the
browser.

It's worth noting that on SMC one can simply click on a notebook file to
open it.

Lengthy programs: editing and running
-------------------------------------
Programming in the browser means you don't have all the niceties of your
favorite text editor.  This is no big deal for small bits of code, but can
impede development for larger programs.  I also worry that using the notebook
too much may keep students from learning to use a good text editor.
Finally, running long programs from the browser is problematic since you can't detach the process.

Usually, Python programs for a numerical methods course can be broken up into
fairly short functions that each fit on a single screen and run in a reasonable
amount of time.

Interactive plotting
---------------------
In my teaching notebooks, I use Python's most popular plotting
package, Matplotlib [Hun07]_.  It's an extremely useful package, whose
interface is immediately familiar to MATLAB users, but
it has a major drawback when used in the IPython notebook.
Specifically, plots that appear inline in the notebook are not
interactive -- for instance, they cannot be zoomed or panned.  There are 
a number of efforts to bring interactive plots to the notebook
(such as Bokeh and Plotly) and I expect this weakness will soon be an area of
strength for the IPython ecosystem.  I plan to incorporate one of these
tools for plotting in the next course that I teach.


More resources
==============
Many people are advocating and using the IPython notebook as a teaching tool,
for many subjects.  For instance, see:

- `Teaching with the IPython Notebook <http://nbviewer.ipython.org/gist/jiffyclub/5165431>`_ by Matt Davis
- `How IPython Notebook and Github have changed the way I teach Python <http://peak5390.wordpress.com/2013/09/22/how-ipython-notebook-and-github-have-changed-the-way-i-teach-python/>`_ by Eric Matthes
- `Using the IPython Notebook as a Teaching Tool <http://www.software-carpentry.org/blog/2013/03/using-notebook-as-a-teaching-tool.html>`_ by Greg Wilson
- `Teaching with ipython notebooks -- a progress report <http://ivory.idyll.org/blog/teaching-with-ipynb-2.html>`_ by C. Titus Brown

To find course actual course materials (in many subjects!),
the best place to start is this curated list: `A gallery of interesting IPython Notebooks 
<https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks>`_.


Acknowledgments
===============
I am grateful to Lorena Barba for helpful discussions (both online and offline)
of some of the ideas presented here.
This work was supported by the King Abdullah University of Science and Technology (KAUST).

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
.. [LeV07] R. J. LeVeque. *Finite Difference Methods for Ordinary and Partial Differential Equations*, Society for Industrial and Applied Mathematics, 2007.

.. [Tre00] L. N. Trefethen. *Spectral Methods in MATLAB*, Society for Industrial and Applied Mathematics, 2000.

.. [Bar14] L. A. Barba, O. Mesnard. *AeroPython*,  10.6084/m9.figshare.1004727. Code repository, Set of 11 lessons in classical Aerodynamics on IPython Notebooks. April 2014.

.. [Bar13] L. A. Barba.  *CFD Python: 12 steps to Navier-Stokes*, http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/, 2013.

.. [Hal75] P. R. Halmos, E. E. Moise, and G. Piranian.  *The problem of learning how to teach*, The American Mathematical Monthly, 82(5):466--476, 1975.

.. [Ern14a] D. Ernst. *What the heck is IBL?*, Math Ed Matters blog, http://maamathedmatters.blogspot.com/2013/05/what-heck-is-ibl.html, May 2014

.. [Ern14b] D. Ernst. *What's So Good about IBL Anyway?*, Math Ed Matters blog, http://maamathedmatters.blogspot.com/2014/01/whats-so-good-about-ibl-anyway.html, May 2014.

.. [VdP14] J. VanderPlas. *Why Python is Slow: Looking Under the Hood*, Pythonic Perambulations blog, http://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/, May 2014. 

.. [VdP13] J. VanderPlas. *JSAnimation*, https://github.com/jakevdp/JSAnimation, 2013.

.. [Per07] F. Pérez, B. E. Granger. *IPython: A System for Interactive Scientific Computing*, Computing in Science and Engineering, 9(3):21-29, 2007. http://ipython.org/

.. [Hun07] J. D. Hunter.  *Matplotlib: A 2D graphics environment*, Computing in Science and Engineering, 9(3):90--95, 2007. http://matplotlib.org/
