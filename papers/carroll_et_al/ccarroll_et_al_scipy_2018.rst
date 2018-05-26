:author: Christopher D. Carroll
:email: FOO
:institution: FOO
:institution: FOO2
:corresponding:

:author: Alexander M. Kaufman
:email: FOO
:institution: FOO

:author: Jacqueline Kazil

:email: FOO
:institution: FOO

:author: Alexander M. Kaufman
:email: FOO
:institution: FOO

:author: Alexander M. Kaufman
:email: FOO
:institution: FOO

:author: Alexander M. Kaufman
:email: FOO
:institution: FOO

:author: Alexander M. Kaufman
:email: FOO
:institution: FOO


------------------------------------------------------------------------------------------
The Econ-ARK Open Source Tools for Computational Economics
------------------------------------------------------------------------------------------

.. class:: abstract

   FOO FOO FOO

.. class:: keywords

   FOO, FOO, FOO

Introduction
=============

Academic research in statistics has standardized on the use of the ‘R’
modeling language for scholarly communication, and on a suite of tools
and standards of practice (the use of R-markdown, e.g.) that allow
statisticians to communicate their ideas easily to each other. Many
other scholarly fields have similarly developed computational and
communication tools that allow scholars easily and transparently to
exchange quantitative ideas and computational results without anyone
having to master idiosyncratic details of anyone else’s hand-crafted
computer code.

The only branch of economics in which something similar has happened is
representative agent macroeconomics, which (to some degree) has
standardized on the use of the DYNARE toolkit.

Our aim is to provide a high quality set of tools and standards whose
existence will help bring the rest of economics out of the (comparative)
wilderness. Part of the reason we are confident the goal is feasible is
that the tools that are now available – Python, Github, and Jupyter
notebooks among them – have finally reached a stage of maturity that can
handle the communication of almost any message an economist might want
to transmit. (See the recent blog post by Paul Romer, `“Jupyter,
Mathematica, and the Future of the Research
Paper” <https://paulromer.net/jupyter-mathematica-and-the-future-of-the-research-paper/>`__
for a fuller statement of the point).

We face two challenges. The first is to develop a set of resources and
examples and standards of practice for communication that are
self-evidently a major improvement on the way economists exchange ideas
now. The second is to persuade scholars to converge on using those
tools.

The `Econ-ARK <http://econ-ark.org>`__ is the vehicle by which we hope
to achieve these objectives. We have begun with the creation of a
toolkit for Heterogeneous Agent (HA) macroeconomics, in part because
that is a field where the need for improvement in standards of
transparency, openness, and reproducibility is particularly manifest,
and because it is a field where important progress seems particularly
feasible.

The traditional approach in macroeconomics has been to assume that
aggregate behavior can adequately be understood by modeling the behavior
of a single ‘representative agent.’ HA macroeconomics instead starts by
constructing models of the behavior of individual microeconomic agents
(a firm or a consumer, e.g.) that match key facts (e.g., some people are
borrowers and others are savers) from the rich microeconomic evidence
about the behavior and circumstances of such agents. With that solid
foundation in place, macroeconomic outcomes are constructed by
aggregating the behavior of the idiosyncratic agents subject to sensible
requirements on the characteristics of the aggregate (such as, in a
stock market, that the number of shares sold must match the number of
shares bought).

The Heterogeneous-Agent Resources toolKit (HARK) is a modular
programming framework for solving, estimating, and simulating
macroeconomic models in which economic agents can be heterogeneous in a
large number of ways. Models that allow heterogeneity among agents have
proven to be useful for policy and research purposes. For example,
recent work by has shown that changes in interest rates (caused, for
example, by monetary policy actions) affect the economy in large part by
reallocating income flows across different types of households
(borrowers versus lenders, e.g.) rather than by causing every household
to change their behavior in the same way (as, implicitly, in a
traditional RA model). C. Carroll, Slacalek, et al. () show that the
response to fiscal policy (e.g., stimulus payments, or tax cuts) depends
crucially on how such payments are distributed across different groups
(an extension of unemployment benefits has a bigger effect on spending
than a cut in the capital gains tax). Geanakoplos () outlines how
heterogeneity drives the leverage cycle, and Geanakoplos et al. ()
applies these insights to large-scale model of the housing and mortgage
markets.

HA models of the kind described above have had a major intellectual
impact over the past few years. But the literature remains small, and
contributions have come mostly from a few small groups of researchers
with close connections to each other.

In large part, this reflects the formidable technical challenges
involved in constructing such models. In each case cited above, the
codebase underlying the results is the result of many years of
construction of hand-crafted code that has not been meaningfully vetted
by researchers outside of the core group of contributors. This is not
mostly because researchers have refused to share their code; instead, it
is because the codebases are so large, so idiosyncratic, and (in many
cases) so poorly documented and organized as to be nearly
incomprehensible to anyone but the original authors and their
collaborators. Researchers with no connections to the pioneering
scholars have therefore faced an unpalatable choice between investing
years of their time reinventing the wheel, or investing years of their
time deciphering someone else’s peculiar and idiosycratic code.

The HARK project addresses these concerns by providing a set of
well-documented code modules that can be combined to solve a range of
heterogeneous-agent models. Methodological advances in the computational
literature allow many types of models to be solved using similar
approaches – the HARK project simply brings these together in one place.
The key is identifying methodologies that are both “modular” (in a sense
to be described below) as well as robust to model misspecification.
These include both solution methods as well as estimation methods.

In addition to these methodological advances, the HARK project adopts
modern software development practices to ease the burden of code
development, code review, code sharing, and collaboration for
researchers dealing in computational methods. Researchers who must
review the scientific and technical code written by others are keenly
aware that the time required to review and understand another’s code can
dwarf the time required to simply re-write the code from scratch
(conditional on understanding the underlying concepts). This can be
particularly important when multiple researchers may need to work on
parts of the same codebase, either across time or distance.

Because these problems are generic (and not specific to computational
economics), the software development community, and particularly the
open-source community, has spent decades perfecting tools for
programmers to quickly consume and understand code written by others,
verify that it is correct, and to contribute back to a large and diverse
codebase without fear of introducing bugs. The tools used by these
professional developers include formal code documentation, unit testing
structures, modern versioning systems for automatically tracking changes
to code and content, and low-cost systems of communicating ideas, such
as interactive programming notebooks that combine formatted mathematics
with executable code and descriptive content. These tools operate
particularly well in concert with one another, constituting an
environment that can greatly accelerate project development for both
individuals and collaborative teams. These technical tools are not new –
the HARK project simply aims to apply the best of them to the
development of code in computational economics in order to increase
researcher productivity, particularly when interacting with other
researchers’ code.

The rest of this paper will first outline the useful concepts we adopt
from software development, with examples of each, and then demonstrate
how these concepts are applied in turn to the key solution and
estimation methods required to solve general heterogeneous-agent models.
The sections are organized as follows: discusses the natural modular
structure of the types of problems HARK solves and overviews the code
structure that implements these solutions. outlines details of the core
code modules used by HARK. outlines two example models that illustrate
models in the HARK framework. summarizes and concludes.

HARK Structure 
===============


` <Methodological%20Framework%20...%20of%20the%20HARK%20Framework>`__

The class of problems that HARK solves is highly modular by
construction. There are approximately these steps in creating a
heterogneous-agents rational model:

#. Write down individual agent problem

#. Solve the individual agent problem

#. For general equilibrium, also solve for aggregate interations and
   beliefs

#. Estimate the model using Simulated Method of Moments (SMM)

Under the solution and estimation method used by HARK, each of these
steps is highly modular. The structure of the solution method suggests a
natural division of the code. The rest of this section outlines the code
structure HARK employs, and the next section outlines the theory behind
these models.

The following example will illustrate the usage of some key commands in
HARK. ``CRRAutility`` is the function object for calculating CRRA
utility supplied by ``HARK.utilities`` module. ``CRRAutility`` is called
attributes of the module ``HARK.utilities``. In order to calculate CRRA
utility with a consumption of 1 and a coefficient of risk aversion of 2
we run:

import HARKutilities as Hutil

Hutil.CRRAutility(,)

Python modules in HARK can generally be categorized into three types:
tools, models, and applications. **Tool modules** contain functions and
classes with general purpose tools that have no inherent “economic
content,” but that can be used in many economic models as building
blocks or utilities. Tools might include functions for data analysis
(e.g. calculating Lorenz shares from data, or constructing a
non-parametric kernel regression), functions to create and manipulate
discrete approximations to continuous distributions, or classes for
constructing interpolated approximations to non-parametric functions.
Tool modules generally reside in HARK’s root directory and have names
like ``HARK.simulation`` and ``HARK.interpolation``. The core
functionality of HARK is in the tools modules; these will be discussed
in detail in the following section.

**Model modules** specify particular economic models, including classes
to represent agents in the model (and the “market structure” in which
they interact) and functions for solving the “one period problem” of
those models. For example, ``ConsIndShockModel.py`` concerns
consumption-saving models in which agents have CRRA utility over
consumption and face idiosyncratic (**Ind**\ ividual) shocks to
permanent and transitory income. The module includes classes for
representing “types” of consumers, along with functions for solving
(several flavors of) the one period consumption-saving problem. When
run, model modules might demonstrate example specifications of their
models, filling in the model parameters with arbitrary values. When
``ConsIndShockModel.py`` is run, it specifies an infinite horizon
consumer with a particular discount factor, permanent income growth
rate, coefficient of relative risk aversion and other parameters, who
faces lognormal shocks to permanent and transitory income each period
with a particular standard deviation; it then solves this consumer’s
problem and graphically displays the results. [1]_ Model modules
generally have ``Model`` in their name. The two examples discussed in
the “microeconomic” and “macroeconomic” sections below come from “Model
modules.”

**Application modules** use tool and model modules to solve, simulate,
and/or estimate economic models *for a particular purpose*. While tool
modules have no particular economic content and model modules describe
entire classes of economic models, applications are uses of a model for
some research purpose. For example,
``/SolvingMicroDSOPs/StructEstimation.py`` uses a consumption-saving
model from ``ConsIndShockModel.py``, calibrating it with age-dependent
sequences of permanent income growth, survival probabilities, and the
standard deviation of income shocks (etc); it then estimates the
coefficient of relative risk aversio n and shifter for an age-varying
sequence of discount factors that best fits simulated wealth profiles to
empirical data from the Survey of Consumer Finance. A particular
application might have multiple modules associated with it, all of which
generally reside in one directory. Particular application modules will
not be discussed in this paper further; please see the Github page and
associated documentation for references to the application modules.

Tool Modules 
=============

HARK’s root directory contains the following tool modules, each
containing a variety of functions and classes that can be used in many
economic models, or even for mathematical purposes that have nothing to
do with economics. We expect that all of these modules will grow
considerably in the near future, as new tools are “low hanging fruit”
for contribution to the project.

HARK.core
---------

This module contains core classes used by the rest of the HARK
ecosystem. A key goal of the project is to create modularity and
interoperability between models, making them easy to combine, adapt, and
extend. To this end, the ``HARK.core`` module specifies a framework for
economic models in HARK, creating a common structure for them on two
levels that can be called “microeconomic” and “macroeconomic”, which are
described in detail in the next section.

Beyond the model frameworks, ``HARK.core`` also defines a
“supersuperclass” called ``HARK.object``. When solving a dynamic
economic model, it is often required to consider whether two solutions
are sufficiently close to each other to warrant stopping the process
(i.e. approximate convergence). HARK specifies that classes should have
a ``distance`` method that takes a single input and returns a
non-negative value representing the (generally dimensionless) distance
between the object in question and the input to the method. As a
convenient default, ``HARK.object`` provides a “universal distance
metric” that should be useful in many contexts. [2]_ When defining a new
subclass of ``HARK.object``, the user simply defines the attribute
distance\_criteria as a list of strings naming the attributes of the
class that should be compared when calculating the distance between two
instances of that class. See
`here <https://econ-%20ark.github.io/HARK/generated/HARK.core.html>`__
for online documentation.



.. [1]
   Running ``ConsIndShockModel.py`` also demonstrates other variations
   of the consumption-saving problem, but their description is omitted
   here for brevity.

.. [2]
   Roughly speaking, the universal distance metric is a recursive
   supnorm, returning the largest distance between two instances, among
   attributes named in ``distance_criteria``. Those attributes might be
   complex objects themselves rather than real numbers, generating a
   recursive call to the universal distance metric.

.. [3]
   Interpolation methods currently implemented in HARK include
   (multi)linear interpolation up to 4D, 1D cubic spline interpolation,
   2D curvilinear interpolation over irregular grids, a 1D “lower
   envelope” interpolator, and others.

.. [4]
   See C. Carroll, Kaufman, et al. () for a much more thorough
   discussion.

.. [5]
   See C. Carroll, Kaufman, et al. () for a much more thorough
   discussion.

.. [6]
   For a much more detailed discussion please see Carroll et al. (2017).

.. [7]
   The solution to a dynamic optimal control problem is a set of policy
   functions and a value functions, one for each period. The policy
   function for this consumption-savings problem is how much to consume
   :math:`C_t` for a given amount of market resources :math:`M_t`.