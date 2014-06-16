:author: Jennifer Klay
:email: jklay@calpoly.edu
:institution: California Polytechnic State University San Luis Obispo

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

---------------------------------------------------------------------
Project-based introduction to scientific computing for physics majors
---------------------------------------------------------------------

.. class:: abstract

   blah blah 

.. class:: keywords

   physics, scientific computing, undergraduate education

Introduction
------------


Computational tools and skills are as critical to the training of physics majors as calculus and math, yet they receive much less emphasis in the undergraduate curriculum. One-off courses that introduce programming and basic numerical problem-solving techniques with commercial software packages for topics that appear in the traditional physics curriculum are insufficient to prepare students for the computing demands of modern technical careers. Yet tight budgets and rigid degree requirements constrain the ability to expand computational course offerings for physics majors.

This paper presents an overview of a recently revamped course at California Polytechnic State University San Luis Obispo (Cal Poly) that uses Python and associated scientific computing libraries to introduce the fundamentals of open-source tools, version control systems, programming, numerical problem solving and algorithmic thinking to undergraduate physics majors. The spirit of the course is similar to the bootcamps organized by Software Carpentry [SWC]_ for researchers in science but is offered as a ten-week for-credit course. In addition to having a traditional in-class component, students learn the basics of Python by completing tutorials on Codecademy's Python track [Codecademy]_ and practice their algorithmic thinking by tackling Project Euler problems [ProjectEuler]_. This approach of incorporating online training may provide a different way of thinking about the role of MOOCs in higher education. The early part of the course focuses on skill-building, while the second half is devoted to application of these skills to an independent research-level computational physics project. Examples of recent projects and their results will be presented.
 
Background
----------

California Polytechnic State University San Luis Obispo (Cal Poly) is one of the 23 campuses of the California State University system.  The university has a "learn by doing" emphasis for the educational experience of its predominantly undergraduate population of approximately 19,000 students, encapsulated in its motto *discere faciendo*.  Part of the University's mission is to provide students the opportunity to get directly involved in research at the frontiers of knowledge through interaction with faculty.  The university is also committed to enhancing opportunities for under-represented groups and is committed to fostering a diverse student body.

The College of Engineering enrolls the largest fraction of Cal Poly undergraduates (~28%), followed by the Colleges of Agriculture (~21%), Liberal Arts (~15%), Business (~13%), Architecture & Environmental Design (~8%), and Science & Mathematics (~15%).  Due to the large number of engineering undergraduates at Cal Poly, the distribution of male (~54%) and female (~46%) students is opposite that of the national average.

The Department of Physics, in the College of Science & Mathematics, offers Bachelor of Science and Arts degrees in Physics, and minors in astronomy and geology, with approximately 150 students enrolled.  There are roughly 30 tenure-track faculty, for a current student-to-faculty ratio of 1:5.  In addition, there are typically 5-10 full-time lecturers and fifteen part-time and retired faculty teaching courses in physics and geology.  The size of the department reflects the need to educate a large number of students in introductory physics across all technical disciplines and the university's commitment to small class sizes.  A typical introductory physics course for scientists and engineers has 48 students, in contrast to typical class sizes at large public universities.  The teaching load for tenure-track faculty is 12 weighted teaching units per quarter, or approximately 20 student contact hours (including office hours) per week.  However, the university has some discretionary funds allocated to subsidize teaching release time for faculty members engaged in research projects with undergraduate students, particularly for those who have procured external grants.

The curriculum for physics majors includes a Senior Project which is often the continuation of paid summer internships undertaken with faculty members in the department who have funding to support student assistants.  This support can come from external research grants or from internal programs, such as "College-Based Fees".  The College-Based Fee program was established by the students of Cal Poly to provide additional funds to support university-wide projects that benefit students through improving quality of education, promoting student learning and progress toward degree completion and increasing institutional productivity.  The allocation of funds for specific projects is guided by a College-Based Fee Board in consultation with deans, departments, faculty and students.  In the physics department, some of the College-Based Fees are used to support summer student assistantships for faculty research projects and limited support for travel for students and faculty to conferences.

Cal Poly has one of the largest (in terms of degrees granted) and most successful undergraduate physics programs in the United States.  Only about 5% of all physics programs in the United States regularly award more than 15 degrees per year, and most of those are at Ph.D. granting institutions.  In 2013-2014, 28 B.S. and 1 B.A. degrees were awarded.  The Cal Poly Physics Department is uniquely successful among four-year colleges.  As a result, Cal Poly was one of 21 departments deemed to be "thriving" and profiled in 2002 by the SPIN-UP study (Strategic Programs for INnovation in Undergraduate Physics) sponsored by the American Association of Physics Teachers, the American Physical Society,and the American Institute of Physics [SPIN-UP]_. The external reviewers from SPIN-UP made special mention of the strong faculty-student interactions and of the success of the physics lounge (known as "h-bar") at making students feel welcome and at home in an intense academic environment. Cal Poly hosted the SPIN-UP Western Regional Workshop in June 2010 where faculty teams from 15 western colleges and universities came to learn how to strengthen their undergraduate physics programs.

Computational physics at Cal Poly
---------------------------------

The physics department has a strong record of preparing students for advanced degrees in physics, often at top tier research institutions.  Between 2005 and 2009, at least 20% of Cal Poly physics graduates entered Ph.D. programs in physics and related disciplines with another 10% seeking advanced degrees in engineering, mathematics, law, and business.

The Cal Poly physics program provides a strong base in theoretical physics with the standard traditional sequence of courses while providing excellent experimental training of students in the laboratory, with a full year of upper division modern physics experiments and several additional specialty lab courses offered as advanced physics electives.  Unfortunately, the department has not yet developed as cohesive and comprehensive of a program in computational physics.  There has been one course "Physics and the Computer" offered for physics majors on computational methods since 1996.  The original catalog deascription of the course was 

   *Introduction to microcomputer tools for physics. Graphics, plotting, use of spreadsheets, integration, differential equations, simulations, statistical techniques, non-linear equations. Applications to problems in physics. 3 lectures*

In 1998 the course was renamed "Physics on the Computer", expanded to 4 lectures, and the description was modified to

   *Introduction to computer algebra systems for solving problems in physics: differential equations, matrix manipulations, simulations and numerical techniques, nonlinear dynamics. 4 lectures.*

The original pre-requisites for the course were General Physics III: Electricity and Magnetism and Linear Analysis I (MATH), although in 1998 concurrent enrollment for Linear Analysis was allowed and in 2001 the phrase "and computer literacy" was added to the pre-requisites.  In 2003, the math pre-requisites were changed to "Vector Calculus or Linear Analysis (preferred)", to allow students behind in math to stay on track in the physics course sequence.  The present description was added in 2009:

   *Introduction to using computers for solving problems in physics: differential equations, matrix manipulations, simulations and numerical techniques, nonlinear dynamics. 4 lectures.*

The catalog remained the same until the registrar transitioned it fully online in 2011 and the pre-requisites were truncated to just General Physics III and Vector Calculus.  This last change was not sanctioned by the department and was only recently discovered to be a mistake which will be corrected for the 2015-17 catalog.  

Despite the desire for students to come to this course with some "computer literacy", no traditional computer science courses have been required for physics majors (although they can be counted as free technical electives in the degree requirements).  Each instructor selects the tools and methods used to implement the course.  Early on, many numerical topics were covered using Excel because students typically had acces
s and experience with it.  Interactive computer algebra systems such as Maple and MATLAB were also commonly employed, but no open-source standard high level programming languages were used.  Between 2007 and 2012 MATLAB was the preferred framework, although some use of Excel for introductory tasks was also included.

Beyond simple data analysis and graphing tasks needed for upper division laboratories, there is no concerted effort to include computational or numerical techniques in upper division theory courses.  Instructors choose to include this material at their own discretion.  There is also currently no upper division computational physics elective in the catalog.

When I joined the faculty of Cal Poly in 2007 I quickly obtained external funding from the National Science Foundation to involve Cal Poly physics undergraduates in research at the CERN Large Hadron Collider with the ALICE experiment.  My background in particle and nuclear physics has been very software intensive, owing to the enormous and complex datasets generated in heavy nucleus collisions.  I have served as software coordinator for one of the ALICE detector sub-systems and I am the architect and lead developer of the offline analysis framework for the Neutron Induced Fission Fragment Tracking Experiment (NIFFTE).  Most of my scientific software is written in C/C++, although I have experience with Pascal, Fortran, Java and shell scripting.  I found it extremely challenging to engage students in my research because of the steep learning curve for the software tools.  

After my first year, I proposed adding an introductory computer science course to the physics B.S. degree requirements but was unable to convince my colleagues of the benefits.  I initiated a faculty journal club to read and discuss the literature on nationwide efforts to improve the undergraduate computational physics curriculum and subsequently led a multi-year effort to investigate ways to improve our program.

In 2012 I became interested in learning Python and decided to offer an independent study course called "Python 4 Physicists" so students could learn it with me.  Over 30 eager students signed up for the course.  We followed Allen Downey's "Think Python" book [ThinkPython] for six weeks, largely on our own, but met weekly for one hour to discuss issues and techniques.  For the second half of the course, the students were placed in groups of 3 and assigned one of two projects, either a cellular automaton model of traffic flow or a 3-D particle tracking algorithm for particle collision data reconstruction.  All code and projects were version controlled with git and uploaded to github.  Examples can be found at [Traffic]_ and [3DTracker]_.  At the end of the quarter the groups presented their projects to the class.  

Not all groups were able to successfully complete the projects but this is likely due to their competing priorities consuming their available coding time.  Nevertheless, they were excited to work on a research-level problem and to be able to use their newly developed programming skills to do so.  Most of them gained basic programming proficiency and some students reported that the course helped them secure summer internships.  It became clear to me that Python is an effective and accessible language for teaching physics majors how to program.  When my opportunity to teach "Physics on the Computer" came in 2013-14, I decided to make it a project-based Python programming course that would teach best practices for scientific software development, including version control and creation of publication quality graphics while giving a broad survey of the major problems in computational physics.


Course Organization
-------------------

The learning objectives for the course are a subset of those developed and adopted by the Cal Poly physics department in 2013 for students completing a degree in physics:

Upon completion of the course you should be able to:

* Use basic coding concepts such as loops, control statements, variable types, arrays, array operations, and boolean logic. (LO1)
* Write, run and debug programs in a high level language. (LO2)
* Carry out basic operations (e.g. cd, ls, dir, mkdir, ssh) at the command line. (LO3)
* Maintain a version controlled repository of your files and programs. (LO4)
* Create publication/presentation quality graphics, equations. (LO5)
* Visualize symbolic analytic expressions - plot functions and evaluate their behavior for varying parameters. (LO6)
* Use numerical algorithms (e.g. ODE solvers, FFT, Monte Carlo) and be able to identify their limitations. (LO7)
* Code numerical algorithms from scratch and compare with existing implementations. (LO8)
* Read from and write to local or remote files. (LO9)
* Analyze data using curve fitting and optimization. (LO10)
* Create appropriate visualizations of data, e.g. multidimensional plots, animations, etc. (LO11)

The course schedule and learning objective map are summarized in table :ref:`schedtable`.  For the first two weeks the students followed the Python track at Codecademy to learn basic syntax and coding concepts such as loops, control statements, variable types, arrays, array operations, and boolean logic.  In class, they were instructed about the command line, ssh, the UNIX shell and version control.  These topics were demonstrated and discussed as instructor-led activities in which they entered commands along with me.  The iPython notebook was introduced in the second week and their first programming exercise outside of Codecademy was to pair-program a solution to Project Euler Problem 1.  They created their own github repository for the course and were guided through the workflow at the start and end of class for the first several weeks to help them get acclimated.  We built on their foundations by taking the Battleship game program they wrote in Codecademy and combining it with iPythonBlocks to make it more visual.  We revisited the Battleship code again in week 4 when we learned about error handling and a subset of the students used iPythonBlocks as the basis for their final project on the Schelling Model of segregation.  The introduction, reinforcement and advanced application of programming techniques was employed to help students build lasting competency with fundamental coding concepts.

Each week the students were provided a "tour" of a specific topic for which they were instructed to read and code along in their own iPython notebook.  They were advised not to copy/paste code, but to type their own code cells, thinking about the commands as they went to develop a better understanding of the material.  After finishing a tour they completed exercises on the topic as homework.  Along with these exercises, they completed a Project Euler problem each week to practice efficient basic programming and problem solving.

A single midterm exam was administered in the fifth week to motivate the students to stay on top of their skill-building and to assess their learning at the midway point.  The questions on the midterm were designed to be straightforward and completable within the two-hour class time.  

Following the midterm exam one class period was set aside for presenting three project possibilities and assigning them.  Two of the projects came from Stanford's NIFTY asignment database [Nifty]_ - the Schelling Model of Segregration by Frank McCown [McCown]_ and Estimating Avogadro's Number by Kevin Wayne [Wayne]_.  The Schelling Model project required students to use iPython widgets and iPythonBlocks to create a grid of colored blocks that move according to a set of rules governing their interactions.  Several recent physics publications on the statistical properties of Schelling Model simulations and their application to physical systems were used to define research questions for the students to answer using their programs.[Schelling1]_,[Schelling2]_  For estimating Avogadro's number, the students coded a particle identification and tracking algorithm that they could apply to the frames of a movie showing Brownian motion of particles suspended in fluid.  The initial test data came from the Nifty archive, but at the end of the quarter the students collected their own data using a microscope in the biology department to image milkfat globules suspended in water.  The challenges of adapting their code to the peculiarities of a different dataset was part of the learning experience.  They used code from a tour and exercise they did early in the quarter, based on the MultiMedia programming lesson on Software Carpentry, which had them filter and count stars in a Hubble image.

The third project was to simulate galaxy mergers by solving the restricted N-body problem.  The project description was developed for this course and was based on a 1972 paper by Toomre and Toomre [Toomre1972]_.  They used SciPy's `odeint` to solve the differential equations describing the motion of a set of massless point particles (stars) orbiting a main galaxy core as a disrupting galaxy core passed in a parabolic trajectory.  The students were not instructed on solving differential equations until week 7, so they were advised to begin setting up the initial conditions and visualization code until they had the knowledge and experience to apply `odeint`.
 
.. table:: Course schedule of topics and learning objectives :label:`schedtable`

   +-----------------------------+-----------------------+
   | Week: Title                 | Learning Objectives   |
   +-----------------------------+-----------------------+
   | 1: Programming Bootcamp     | LO1, LO2, LO3, LO4    |
   +-----------------------------+-----------------------+
   | 2: Programming Bootcamp     | LO1-4, LO11           |
   +-----------------------------+-----------------------+
   | 3: Intro to NumPy/SciPy,    | LO1-4, LO9, LO11      |
   |    Data I/O                 |                       |
   +-----------------------------+-----------------------+
   | 4: Graphics, Animation and  | LO1-4, LO5, LO6, LO11 |
   |    Error handling           |                       |
   +-----------------------------+-----------------------+
   | 5: Midterm Exam, Projects   | LO1-4, LO5, LO6, LO9  |
   |    and Program Design       |                       |
   +-----------------------------+-----------------------+
   | 6: Interpolation and        | LO1-4, LO5, LO6, LO7, |
   |    Differentiation          | LO8, LO11             |
   +-----------------------------+-----------------------+
   | 7: Numerical Integration,   | LO1-4, LO5, LO6, LO7, |
   |    Ordinary Differential    | LO8, LO11             |
   |    Equations (ODEs)         |                       |
   +-----------------------------+-----------------------+
   | 8: Random Numbers and       | LO1-4, LO5, LO6, LO7, |
   |    Monte-Carlo Methods      | LO8, LO11             |
   +-----------------------------+-----------------------+
   | 9: Linear Regression and    | LO1-11                |
   |    Optimization             |                       |
   +-----------------------------+-----------------------+
   | 10: Symbolic Analysis,      | LO1-4, LO5, LO6, LO11 |
   |     Project Hack-a-thon!    |                       |
   +-----------------------------+-----------------------+
   | Finals: Project Demos       | LO1-11                |
   +-----------------------------+-----------------------+

References
----------
.. [SWC] http://software-carpentry.org/

.. [Codecademy] http://www.codecademy.com/

.. [ProjectEuler] https://projecteuler.net/

.. [SPIN-UP] http://www.aapt.org/Projects/ntfup/index.cfm

.. [ThinkPython] http://www.greenteapress.com/thinkpython/thinkpython.html

.. [Traffic] Link to project

.. [3DTracker] Link to project

.. [Nifty] http://nifty.stanford.edu/

.. [McCown] http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/

.. [Wayne] http://nifty.stanford.edu/2013/wayne-avogadro.html

.. [Toomre1972] Toomre paper link

.. [Schelling1] link to paper 1

.. [Schelling2] link to paper 2

