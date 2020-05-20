:author: Sebastian Benthall
:email: spb413@nyu.edu
:institution: NYU School of Law

:author: Mridul Seth
:email: mridul@seth.com
:institution: Econ-ARK

:video: http://www.youtube.com/watch?v=dhRUe-gz690

:bibliography: mybib

-------------------------------------------------------------------
Aligning Across Use Cases for Domain Specific Scientific Software
-------------------------------------------------------------------

.. class:: abstract

   An abstract

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
------------

...

Literature review
------------------

Ever since :cite:`papert1980mindstorms` introduced
'constructionist' learning using computers, educators
have been enticed by the possibility that students could
learn valuable knowledge by playing with software.
While originally used as a tool for teaching mathematics,
it was not long before Papert's Logo tool was also
used in scientific education, teaching students not just
about the abstract mathematical sphere, but about the
physical world. :cite:`resnick1990lego`.
The legacy of Logo is alive and well in NetLogo :cite:`tisue2004netlogo`,
which is used by both students and researchers alike in the study of
complex and agent-based systems, and the Python ABM toolkit
Mesa :cite:`masad2015mesa`.

Since, the ubiquity of computing and its increasingly
central role in industry has prompted the spread of
ideas that were once specific to "computer science"
into other disciplines. :cite:`wing2006computational`
coined the term "computational thinking" for the
general skills of managine abstraction, modularity,
scalability, and robustness of systems.
Now it refers to the cross-disciplinary use of these
computational concepts :cite:`guzdial2008education`
:cite:`settle2013beyond`.
The question raised by "computational thinking" is 
how much computer science education is necessary for
these cross-disciplinary uses of computation.
Logo, after all, not only introduced students to mathematics,
but also programming.
But did it teach "computational thinking"?

The industrial demand for students educated in handling
"Big Data" systems has since prompted a popularization
of statistics beyond its discipline in a way that's analogous
to the popularization of computer science. 
:cite:`jordan2016computational` discusses this new industry
demand as "inferential thinking".
Together, computational thinking and inferential thinking
have been reimagined by some as the foundation for a
new form of cross-disciplinary "data science" curriculum
:cite:`adhikari2017computational`
:cite:`van2019accelerating`.

Open source scientific software development has benefited
from the influx of capital due to industry interest in
"data science" applications. Software packages such as Numpy (cite)
Pandas (cite) and Scikit-learn (cite) have become popular as
infrastructure for data science--computational and inferential thinking.
At the same time, these tools have provided a foundation and aspirational
example for more domain specific scientific libraries, such as (need a list: astropy, what else?).
Scientific educators continue to see potential in the use of these
tools to support the education of their students not only
*about computation*, but *about the world* :cite:`barba2016computational`,
in a return to Papert's constructionist paradigm.

This vision of scientific research and education supported by
open source domain specific scientific libraries faces two
significant obstacles.
The first is the development and sustainability of the software
itself.
Open source software projects in general are not guaranteed to
succeed; most fail to gain wide adoption or reach sustainability.
:cite:`schweik2012internet`
In addition to these general difficulties, scientific software
suffers from the fac that researchers, who are often required to
write and modify software, do not have formal training in software
development. As a result, scientific software is often hampered
by technical debt. These problems are mitigated by national
initiatives to train scientists in software engineering skills,
such as the UK's Software Sustainability Institute, as well as
Software Carpentry :cite:`wilson2014software`.
There is further work to be done in institutional design
around filling this skills gap :cite:`katz2016report`.
But it is known that "computational thinking" skills alone
are not sufficient for successful scientific software.
Software engineering skills are necessary to produce
software that is usable beyond the lab or research group
that originates it, which is a necessary path towards
software sustainability :cite:`benthall_2019`.

A second obstacle integrating software tools into
scientific practice is that software-based learning
requires additional education infrastructure.
:cite:`suen2018equity` document the challenges in providing
JupyterHub with automatic grading extensions at universities
and colleges; they find that many institutions do not have the
resources or deep IT expertise necessary to build and maintain this
infrastructure. Cloud-based computational notebooks for assignments
and exploration are coming to be necessary for computation-based
scientific education, with implications for social equity in scientific
education.



Discipline Specifics
---------------------

The Econ-ARK project :cite:`carroll2018econ`
is a toolkit for the structural
modeling of optimizing economic choices by heterogenous agents.
A primary goal of its flagship software library, HARK (Heterogenous
Agent Research toolKit) is to support economic research
into heterogeneous agent (HA) modeling, which became a research priority
after the 2008 financial crisis revealed the weaknesses in the
then-dominant representative agent (RA) based paradigm.
It has been designed so that researchers and students can take a hands-on approach to economic modeling in software :cite:`carroll2018hands`.
It lies roughly in the Papertian educational tradition, similar to
other agent-based modeling software such as NetLogo :cite:`tisue2004netlogo` and Mesa :cite:`masad2015mesa`.
However, in Econ-ARK models, agents that optimize their behavior strategically with respect to predicted effects over time.
In this respect, Econ-ARK has some characteristics of a reinforcement learning or "AI" toolkit.

(Example of a problem, here: bellman equations, etc.)

Models in HARK are, at a certain level of mathematical abstraction,
equivalent to Markov Decision Problems (MDP).
However, generic MDP software is not adequate for research in this
field, for several reasons.

- **Substantive, policy-oriented model-building.**
  Unlike many recent fields of "data science", in which generic
  model-fitting and machine-learning techniques are applied to
  a large data set for the purpose of maximizing predictive potential,
  this branch of economics operates with relatively scarce data and
  a drive for model veracity. Besides the academic field of researchers,
  the intended audience for these models are national central banks
  and other policy-makers. For example, one policy application of these
  models is predicting the impact of the CARES stimulus bill on
  consumption. :cite:`carroll2020modeling` These models are scientifically
  valued for their ability to approximate real social dynamics, and
  for their ability to build consensus towards policy-making, in addition
  to their goodness of fit to available data.
- **Analytical results informing solvers.** Like many other sciences,
  this branch of economics has a theoretical component consisting in
  mathematical proofs about the models in question.
  In addition to providing
  "interpretable" insight into the invariant properties of a model,
  these results also inform the design of model solvers and
  the user experience.
  For example, a mathematical result might reveal under what parameter
  conditions a model has a degenerate solution; the software will warn the
  user if they attempt to solve the model in such a case. Elsewhere,
  an analytical result might provide a shortcut such that it is possible
  to write a solution algorithm with lower computational complexity than a
  generic one would have.
- **Continuous space decisions.** Most MDP solvers and simulators
  assume a discrete control and state space. The economic
  problems studied using HARK are most often defined with continuous
  control and state spaces, and with continuous random variables as
  exogenous shocks. HARK therefore includes a variety of discretization
  and interpolation tools that support the transformation between
  discrete and continuous representations.

The upshot of these conditions is that Econ-ARK software is not only
a tool for researchers doing empirical scientific work.
Rather, its software is an encoding of substantive research results
in mathematical theory. This entails that the success of Econ-ARK
will imply a practical change to the research field: students will
study models that have been published by researchers in Python
in order to learn insights about the economy. This blending of roles,
between researchers, students, and software engineers, leads to
complicates the software architecture of the toolkit.

Case Study: Econ-ARK Use Cases
------------------------------------

Roles
  - Researcher
  - Teacher/Student
  - Publication
  - Software engineer
    


Case Study: Econ-ARK infrastructure
------------------------------------------

Discussion
--------------------


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
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


