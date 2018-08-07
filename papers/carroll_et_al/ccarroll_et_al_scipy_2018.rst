:author: Christopher D. Carroll
:email: ccarroll@jhu.edu
:institution: Johns Hopkins University
:corresponding:

:author: Alexander M. Kaufman
:email: akaufman10@gmail.com
:institution: Woodrow Wilson School of Public Policy

:author: Jacqueline L. Kazil
:email: jacqueline.kazil@capitalone.com
:institution: Capital One

:author: Nathan M. Palmer
:email: npalmer.professional@gmail.com
:institution: Econ-ARK

:author: Matthew N. White
:email: mnwecon@udel.edu
:institution: University of Delaware

:video: https://youtu.be/1ytEhrnwu6A

------------------------------------------------------------------------------------------
The Econ-ARK and HARK: Open Source Tools for Computational Economics
------------------------------------------------------------------------------------------

.. class:: abstract

The Economics Algorithmic Repository and toolKit (`Econ-ARK <http://econ-ark.org>`__) aims
to become a focal resource for computational economics. Its first ‘framework,’ the
Heterogeneous Agent Resources and Toolkit (`HARK <http://github.com/econ-ark/HARK>`__),
provides a modern, robust, transparent set of tools to solve a class of macroeconomic models
whose usefulness has become increasingly apparent both for economic policy and for research
purposes, but whose adoption has been limited because the existing literature derives from
idiosyncratic, hand-crafted, and often impenetrable legacy code. We expect future Econ-ARK
frameworks (e.g., for analysis of the transmission of beliefs through agents' social networks)
will draw heavily on key elements of the existing HARK framework, including the API, the
structure, and documentation standards.

.. class:: keywords

 Heterogeneous-Agent Resources toolKit, econ-ark, computational economics, economic modeling

.. class:: disclaimer

 *Disclaimer:* Views expressed herein do not necessarily reflect the views of the respective institutions that employ the respective authors.


Introduction
=============

The Economics Algorithmic Repository and toolKit (`Econ-ARK <http://econ-ark.org>`__)
is a modular programming framework for solving and estimating macroeconomic
and macro-financial models in which economic agents can exhibit significant heterogeneity. [1]_
Models with extensive heterogeneity among agents can be  extremely useful for policy and
research purposes. However, the most commonly published macroeconomic and macro-finance
models have very limited heterogeneity or none at all, in large part because these are
the only models that can be easily solved with existing toolkits such as DYNARE [Adjemian2011].

In contrast, models with extensive heterogeneity among agents have no central
toolkit and must be solved in a bespoke way. This requires a significant
investment of time and human capital before a researcher can produce usable work.
This results in needless code duplication, increasing the chance for error and wasting
valuable research time. The Econ-ARK project addresses these concerns by providing
a set of well-documented code modules that can be composed together to solve a
range of heterogeneous-agent models. Methodological advances in the computational
literature allow many types of models to be solved using similar approaches;
the Econ-ARK project simply brings these pieces together in one place. HARK is
written in Python 2.7, with a pull request underway at the time of this writing
to make it fully compatible with both Python 2.7 and 3.6.

Academic research in statistics has standardized on the use of the ‘R’ modeling language
for scholarly communication, and on a suite of tools and standards of practice (the use
of R-markdown, e.g.) that allow statisticians to communicate their ideas easily to each
other. Many other scholarly fields have similarly developed suites of tools that allow
scholars to easily and transparently exchange quantitative ideas and computational results
without anyone having to master idiosyncratic details of anyone else’s hand-crafted computer
code.

The only branch of economics in which anything similar has happened is representative agent
(RA) macroeconomics, which (to some degree) has standardized on the use of the DYNARE [Adjemian2011]
toolkit for solving representative agent dynamic stochastic general equilibrium models.

We face two primary challenges. The first is to develop a set of resources and
examples and standards of practice for communication that are
self-evidently a major improvement on the way economists exchange ideas
now. The second is to persuade scholars to adopt those tools.

The `Econ-ARK <http://econ-ark.org>`__ is the vehicle by which we hope
to achieve these objectives. We have begun with the creation of a
toolkit for heterogeneous agents (HA) macroeconomics, in part because
that is a field where the need for improvement in standards of
transparency, openness, and reproducibility is particularly manifested,
and partly because it is a field where important progress seems particularly
feasible.  `QuantEcon <https://quantecon.org/>`__ is the most similar
project to Econ-ARK and makes use of open source coding tools. However,
that project focuses largely on foundational material appropriate for an
introductory graduate course on numeric methods in macroeconomics, whereas
the Econ-ARK is geared toward the production of new research. [2]_

The traditional approach in macroeconomics has been to assume that
aggregate behavior can be understood by modeling the behavior
of a single 'representative agent' -- the 'representative consumer' or
'representative firm'. HA macroeconomics instead starts by
constructing models of the behavior of individual microeconomic agents
(a firm or a consumer, e.g.) that match key facts (say, that some people are
borrowers and others are savers) from the rich microeconomic evidence
about the behavior and circumstances of such agents. With that solid
foundation in place, macroeconomic outcomes are constructed by
aggregating the behavior of the individual agents subject to sensible
requirements on the characteristics of the aggregate (such as that the aggregate
amount borrowed cannot exceed a function of the aggregate amount saved). For a
broad review of representative agent and
heterogeneous agents economic modeling, see the discussion by
[Guvenen2011] and [Kirman1992]. More broadly, the branch of agent-based
macroeconomics explores the issues of emergence and complexity.
The interested reader is directed to the Handbooks of Computational Economics,
Volumes 2 and 4: [Tesfatsion2006] and [Hommes2018]. The most recent volume in
particular outlines similarities and differences between more traditional
heterogeneous agents macroeconomics and so-called "agent-based methods," inspired
from fields such as physics and ecology.

The Heterogeneous-Agent Resources toolKit (HARK) is a modular
programming framework for solving, estimating, and simulating
macroeconomic models with heterogeneous agents. Agents in HARK can be heterogeneous in
a large number of ways, such as in wealth, income processes, preferences, or
expectations. Models with heterogeneity among agents have
proven to be increasingly useful for policy and research purposes.

For example, recent work by [Kaplan2018] has shown that changes in interest rates affect the
economy in large part by reallocating income flows across different types of
households rather than by causing every household to change their behavior in
the same way. The latter implicitly occurs in a traditional rational expectations
model, but may be misleading regarding the underlying channel of the effect.
[Carroll2017a] shows that the response to fiscal policy (such as stimulus payments
or tax cuts) depends crucially on how such payments are distributed across
different groups. For example, an extension of unemployment benefits has a bigger
effect on spending than a cut in the capital gains tax. [Geanakoplos2010] outlines how
heterogeneity drives the leverage cycle, and [Geanakoplos2012]
applies these insights to large-scale model of the housing and mortgage
markets.

HA models of the kind described above have had a major intellectual
impact over the past few years. But the literature remains small, and
contributions have come mostly from a few small groups of researchers
with close connections to each other. An excellent overview of this literature
can be found in the most recent volume of the Handbooks of Computational
Economics [Hommes2018] and works cited therein.

In large part, this reflects the formidable technical challenges
involved in constructing such models. In each case cited above, the
codebase underlying the results is the result of many years of
construction of hand-crafted code that has not been meaningfully vetted
by researchers outside of the core group of contributors. This is not
because researchers have refused to share their code; instead, it
is because the codebases are so large, so idiosyncratic, and (in many
cases) so poorly documented and organized as to be nearly
incomprehensible to anyone but the original authors and their
collaborators. Researchers with no connections to the pioneering
scholars have therefore faced an unpalatable choice between investing
years of their time reinventing the wheel, or investing years of their
time deciphering someone else’s peculiar and idiosyncratic code.

Researchers who must review the scientific and technical code written by others
are keenly aware that the time required to review and understand another’s code
can dwarf the time required to simply re-write the code from scratch
(conditional on understanding the underlying concepts). This can be
particularly important when multiple researchers may need to work on
parts of the same codebase, either across time or distance.

The HARK project addresses these concerns by providing a set of
well-documented code modules that can be combined to solve a range of
heterogeneous-agent models. Methodological advances in the computational
economics literature allow many types of models to be solved using similar
approaches; the key for HARK is to identify methodologies that are “modular”
(in a sense to be described below).

In addition to these methodological advances, the HARK project adopts
modern software development practices to ease the burden of code
development, code review, code sharing, and collaboration for
researchers dealing with computational methods.

Because these problems are generic (and not specific to computational
economics), the software development community, and particularly the
open-source community, has spent decades developing tools for
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
individuals and collaborative teams. These technical tools are not new--
the HARK project simply aims to apply the best of them to the
development of code in computational economics in order to increase
researcher productivity, particularly when interacting with other
researchers’ code.

The rest of this paper will first outline the useful concepts we adopt
from software development, with examples of each, and then demonstrate
how these concepts are applied in turn to the key solution and
estimation methods required to solve heterogeneous-agent models.
The sections are organized as follows: Section 1 discusses the natural modular
structure of the types of problems HARK solves and provides an overview
of the code structure that implements these solutions. Section 2 provides
details of the core code modules in HARK. Section 3 outlines two examples
that illustrate models in the HARK framework. Section 4 summarizes and concludes.

1. HARK Structure
=================

The class of problems that HARK solves is highly modular by
construction. There are approximately these steps in solving a
rational heterogeneous agents model:

#. Specify the problem faced by an individual agent

#. Specify how the actions and states of individual agents collectively generate aggregate outcomes or processes

#. For given beliefs about aggregate processes, solve the individual agent's problem

#. Simulate the behavior of agents, generating a "history" of aggregate outcomes

#. Formulate new beliefs about the aggregate processes based on that history

#. Iterate on steps 3-5 until beliefs converge

In isolation, steps 1 and 3 constitute the solution to a "microeconomic" model in HARK:
how an individual agent should optimally act, treating all inputs to his problem as fixed.
The inclusion of steps 2, 4, 5, and 6 embeds the microeconomic model in a "macroeconomic"
model, requiring consistency among agents' individual behavior, the outcomes that result
from the aggregation of these choices, and agents' beliefs about aggregate processes.
The assumption of rationality is imposed by having the beliefs formulated in step 5 be
justified given the history of aggregate outcomes; agents correctly interpret (a hypothetical)
history when forming their new beliefs.  Economists call such a solution a "rational
expectations equilibrium", as agents' expectations are fulfilled by reality, and they
have no reason to update these expectations or beliefs. [3]_

In the section below titled "Sample Model: Perfect Foresight Consumption-Saving,"
we directly illustrate a microeconomic model in HARK; a full example of a
macroeconomic model is outlined in [Carroll2017b].

To *estimate* a model for some research purpose, the economist tries to find the "deep"
or "structural" parameters that make model outcomes best match particular features of
some dataset.  That is, the model is mathematically specified in steps 1 and 2 above,
but the economist does not know the values of some vector of model parameters; the objective
of the estimation is to find the parameters that make the model best "match" real data.
As the dataset, features or moments to match, and particular estimation method (e.g.
simulated method of moments or maximum likelihood estimation) are idiosyncratic to each
research project, we will not elaborate further here.

In HARK, each of the solution steps is highly modular, and the structure of the solution method
suggests a natural division of the code. (The solution method is dynamic programming
and fixed point iteration, and the estimation method is Simulated Method of Moments.
These are described in detail in [Carroll2012].)

Python modules in HARK can generally be categorized into three types:
tools, models, and applications. **Tool modules** contain functions and
classes with general purpose tools that have no inherent “economic
content,” but that can be used in many economic models as building
blocks or utilities. Tools might include functions for data analysis
(e.g. calculating Lorenz shares from data, or constructing a
non-parametric kernel regression), functions to create and manipulate
discrete approximations to continuous distributions, or classes for
constructing interpolated approximations to non-parametric functions.
Tool modules reside in the "top level" of HARK and have names like
``HARK.simulation`` and ``HARK.interpolation``. The core
functionality of HARK is in the tools modules; these will be discussed
in detail in the following section.

**Model modules** specify particular economic models, including classes
to represent agents in the model and the “market structure” in which
they interact, and functions for solving the “one period problem” of
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
problem and graphically displays the results. [4]_ Model modules
generally have ``Model`` in their name. There are two broad types of models
solved by HARK, "microeconomic" models and aggregate or "macroeconomic" models.
In a microeconomic problem, agents solve their problem taking their environment
as a given -- the "macro" environment is fixed exogenously. A macroeconomic
problem is typically composed of a number of agents solving their own
microeconomic problems, whose interactions affect the macroeconomic
environment. Thus the aggregate processes that describe the agents' environment
is endogenous to the individual-level decisions made by each agent. The two
examples illustrate this in the “microeconomic” and “macroeconomic” sections below.

**Application modules** use tool and model modules to solve, simulate,
and/or estimate economic models *for a particular purpose*. While tool
modules have no particular economic content and model modules describe
entire classes of economic models, applications are uses of a model for
some research purpose. For example,
``/SolvingMicroDSOPs/StructEstimation.py`` uses a consumption-saving
model from ``ConsIndShockModel.py``, calibrating it with age-dependent
sequences of permanent income growth, survival probabilities, and the
standard deviation of income shocks (etc); it then estimates the
coefficient of relative risk aversion and shifter for an age-varying
sequence of discount factors that best fits simulated wealth profiles to
empirical data from the Survey of Consumer Finance. A particular
application might have multiple modules associated with it, all of which
generally reside in one directory. Particular application modules will
not be discussed in this paper further; please see `the GitHub page and
associated documentation <https://github.com/econ-ark/HARK>`__
for references to the application modules.

2. Tool Modules
===============

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
levels that can be called “microeconomic” and “macroeconomic”.

Beyond the model frameworks, ``HARK.core`` also defines a
"superclass" called ``HARKobject``. When solving a dynamic
economic model, it is often required to consider whether two solutions
are sufficiently close to each other to warrant stopping the process
(i.e. approximate convergence). HARK specifies that classes should have
a ``distance`` method that takes a single input and returns a
non-negative value representing the (generally dimensionless) distance
between the object in question and the input to the method. As a
convenient default, ``HARKobject`` provides a “universal distance
metric” that should be useful in many contexts. [5]_ When defining a new
subclass of ``HARKobject``, the user simply defines the attribute
distance\_criteria as a list of strings naming the attributes of the
class that should be compared when calculating the distance between two
instances of that class. See
`here <https://econ-%20ark.github.io/HARK/generated/HARK.core.html>`__
for online documentation.


HARK.utilities
--------------

The ``HARK.utilities`` module carries a double meaning in its name, as
it contains both utility functions (and their derivatives, inverses, and
combinations thereof) in the economic modeling sense as well as
utilities in the sense of general tools. Utility functions include
constant relative risk aversion (CRRA) and constant absolute risk
aversion (CARA). Other functions in ``HARK.utilities`` include data
manipulation tools, functions for constructing discrete state space
grids, and basic plotting tools. The module also includes functions for
constructing discrete approximations to continuous distributions and
manipulating these representations.

HARK.interpolation
------------------

The ``HARK.interpolation`` module defines classes for representing
interpolated function approximations. Interpolation methods in HARK all
inherit from a superclass such as ``HARKinterpolator1D`` or
``HARKinterpolator2D``, wrapper classes that ensure interoperability
across interpolation methods. These classes all inherit from ``HARKobject``,
so that they come equipped with the default distance metric. [6]_

**HARK.simulation**
`````````````````````

The HARK.simulation module provides tools for generating simulated data
or shocks for post-solution use of models. Currently implemented
distributions include normal, lognormal, Weibull (including
exponential), uniform, Bernoulli, and discrete.

**HARK.estimation**
````````````````````

Methods for optimizing an objective function for the purposes of
estimating a model can be found in ``HARK.estimation``. As of this
writing, the implementation includes minimization by the Nelder-Mead
simplex method, minimization by a derivative-free Powell method variant,
and two tools for resampling data (e.g., for a bootstrap). Future
functionality will include global search methods, including genetic
algorithms, simulated annealing, and differential evolution.

3. Model Modules
================

*Microeconomic* models in HARK use the ``AgentType`` class to represent
agents with an intertemporal optimization problem. Each of these models
specifies a subclass of ``AgentType``; an instance of the subclass
represents agents who are ex-ante homogeneous (they have common values
for all parameters that describe the problem, such as risk aversion).
The ``AgentType`` class has a ``solve`` method that acts as a “universal
microeconomic solver” for any properly formatted model, making it easier
to set up a new model and to combine elements from different models; the
solver is intended to encompass any model that can be framed as a
sequence of one period problems. [7]_

*Macroeconomic* models in HARK use the ``Market`` class to represent a
market or other mechanisms by which agents' (i.e. instances of ``AgentType`` subclasses)
interactions are aggregated to produce “macro-level” outcomes. For example,
the market in a consumption-saving model might combine the individual asset holdings of
all agents in the market to generate aggregate savings and capital in
the economy, which in turn produces the interest rate that agents care
about. Agents then learn the aggregate capital level and interest rate,
which affects their future actions. In this way, objects that *microeconomic*
agents treat as exogenous when solving their individual-level problems
(such as the interest rate) are made *endogenous* at at the
macroeconomic level through the ``Market`` aggregator. Like
``AgentType``, the ``Market`` class also has a ``solve`` method, which
seeks out a dynamic general equilibrium rule governing the
aggregate processes.

Microeconomics: the AgentType Class
-----------------------------------

The core of our microeconomic dynamic optimization framework is a
flexible object-oriented representation of economic agents. Each microeconomic
model defines a subclass of ``AgentType``, specifying additional
model-specific features and methods while inheriting the methods of the
superclass. This section provides a brief example of a problem solved by a microeconomic
instance of ``AgentType``.

**Sample Model: Perfect Foresight Consumption-Saving**
``````````````````````````````````````````````````````````

To provide a concrete example of how the AgentType class works, consider
the very simple case of a perfect foresight consumption-saving model.
The agent has time-separable, additive CRRA preferences over consumption
:math:`C_t`, discounting future utility at a constant rate. He receives
a particular stream of labor income :math:`Y_t` each period and knows
the interest rate :math:`{R}` on assets :math:`A_t` that he holds
from one period to the next. His decision about how much to consume :math:`C_t` in a
particular period out of total market resources :math:`M_t`
can be expressed in Bellman form as:

.. math::

 \begin{aligned}
 V_t(M_t) &= \max_{C_t} \; \mathrm{u}(C_t)  + \beta  (1-{D}_{t+1}) E [V_{t+1}(M_{t+1}) ], \\
 A_t &= M_t - C_t, \\
 M_{t+1} &= {R} A_t + Y_{t+1}, \\
 Y_{t+1} &= \Gamma_{t+1} Y_t, \\
 \mathrm{u}(C) &= \frac{C^{1-\rho}}{1-\rho}.
 \end{aligned}

The agent’s problem is thus characterized by values of :math:`\rho`,
:math:`{R}`, and :math:`\beta`, plus sequences of survival
probabilities :math:`(1-{D}_{t+1})` and income growth factors
:math:`\Gamma_{t+1}` for :math:`t = 0, ... ,T-1`. This problem has an
analytical solution for both the value function and the consumption function.

The ``ConsIndShockModel.py`` module defines the class
``PerfForesightConsumerType`` as a subclass of ``AgentType`` and
provides ``solver`` classes for several variations of a
consumption-saving model, including the perfect foresight problem. A
HARK user could specify and solve a ten period perfect foresight model
with the following two commands (the first command is split over
multiple lines) :

.. code-block:: python

  MyConsumer = PerfForesightConsumerType(
      time_flow=True, cycles=1, AgentCount = 1000,
      CRRA = 2.7, Rfree = 1.03, DiscFac = 0.98,
      LivPrb = [0.99,0.98,0.97,0.96,0.95,0.94,0.93,
                0.92,0.91,0.90],
      PermGroFac = [1.01,1.01,1.01,1.01,1.01,1.02,
                    1.02,1.02,1.02,1.02] )

  MyConsumer.solve()

The first line makes a new instance of ConsumerType, specifies that time
is currently “flowing” forward, specifies that the sequence of periods
happens exactly once, and that, if the model is simulated after it is solved,
there are 1000 agents with these exact characteristics. The next five lines
(all part of the same command) set the time-invariant (``CRRA`` is :math:`\rho`,
``Rfree`` is :math:`{R}`, and ``DiscFac`` is :math:`\beta`) and time-varying
parameters (``LivPrb`` is :math:`(1-{D}_{t+1})`, ``PermGroFac`` is :math:`\Gamma_{t+1}`). After
running the ``solve method``, ``MyConsumer`` will have an attribute
called ``solution``, which will be a list with eleven
``ConsumerSolution`` objects, representing the period-by-period solution
to the model. [8]_

The consumption function for a perfect foresight consumer is a linear
function of market resources-- not terribly exciting. The marginal
propensity to consume out of wealth doesn’t change whether the consumer
is rich or poor. When facing *uncertain* income, however, the
consumption function is concave: the marginal propensity to consume is
very high when agents are poor, and lower when they are rich. Moreover,
agents facing income risk save more than agents under
certainty. However, as agents facing uncertainty get richer, their
consumption function converges to the perfect foresight consumption
function-- rich but uncertain agents act like agents who face no income risk.
In Figure 1, the solid blue line is consumption under certainty, while the
dashed orange line is consumption under uncertainty. The inset plot
demonstrates that these two functions converge as the horizontal axis of this
plot is extended.

.. figure:: ./consumption_functions.png
 :alt: Consumption Functions

 Consumption Functions

Macroeconomics: the Market Class
--------------------------------

The modeling framework of ``AgentType`` is called “microeconomic”
because it pertains only to the dynamic optimization problem of
individual agents, treating all inputs of the problem from their
environment as exogenously fixed. In what we label as “macroeconomic”
models, some of the inputs for the microeconomic models are endogenously
determined by the collective states and choices of other agents in the
model. In a rational dynamic general equilibrium, there must be
consistency between agents’ beliefs about these macroeconomic objects,
their individual behavior, and the realizations of the macroeconomic
objects or processes that result from individual choices.

The ``Market`` class in ``HARK.core`` provides a framework for such
macroeconomic models, with a ``solve`` method that searches for a
rational dynamic general equilibrium. An instance of ``Market`` includes
as an attribute a list of ``AgentType`` objects that compose the economy, a method for
transforming microeconomic outcomes (states, controls, and/or shocks)
into macroeconomic outcomes, and a method for interpreting a history or
sequence of macroeconomic outcomes into a new “dynamic rule” for agents
to believe. Agents treat the dynamic rule as an input to their
microeconomic problem, conditioning their optimal policy functions on
it. A dynamic general equilibrium is a fixed point dynamic rule: when
agents act optimally while believing the equilibrium rule, their
individual actions generate a macroeconomic history consistent with the
equilibrium rule.

**Down on the Farm**
`````````````````````

The ``Market`` class uses a farming metaphor to conceptualize the
process for generating a history of macroeconomic outcomes in a model.
Suppose all ``AgentType`` agents in the economy believe in some dynamic rule
(i.e. the rule is stored as attributes of each ``AgentType``, which
directly or indirectly enters their dynamic optimization problem), and
that they have each found the solution to their microeconomic model
using their ``solve`` method. Further, the macroeconomic and
microeconomic states have been reset to some initial orientation.

To generate a history of macroeconomic outcomes, the ``Market``
repeatedly loops over the following steps a set number of times:

#. ``sow``: Distribute the macroeconomic state variables to all
   ``AgentTypes`` in the market.

#. ``cultivate``: Each ``AgentType`` executes their ``marketAction``
   method, often corresponding to simulating one period of the
   microeconomic model.

#. ``reap``: Microeconomic outcomes are gathered from each ``AgentType``
   in the market.

#. ``mill``: Data gathered by ``reap`` is processed into new
   macroeconomic states according to some “aggregate market process”.

#. ``store``: Relevant macroeconomic states are added to a running
   history of outcomes.

This procedure is conducted by the ``makeHistory`` method of ``Market``
as a subroutine of its ``solve`` method. After making histories of the
relevant macroeconomic variables, the market then executes its
``calcDynamics`` function with the macroeconomic history as inputs,
generating a new dynamic rule to distribute to the ``AgentType`` agents in the
market. The process then begins again, with the agents solving their
updated microeconomic models given the new dynamic rule; the ``solve``
loop continues until the “distance” between successive dynamic rules is
sufficiently small.

Each subclass of ``Market`` has its own ``mill`` and ``calcDynamics`` methods, and designates which variables
are to be gathered ``reap`` and distributed by ``sow``, thus specifying what it means to generate "aggregate
outcomes" and "form beliefs" in that particular model. We believe that the ``Market``
framework is general enough to encompass a very wide range of disparate models, from
standard models in which individual assets are aggregated into productive capital,
to models of choice over health insurance contracts with adverse selection and moral hazard,
to models of direct agent-to-agent interaction more commonly seen in other scientific fields.

4. Summary and Conclusion
=========================

The Econ-ARK project's broadest aim is to provide a platform for improving
communication and collaboration among economists on technical and computational
questions. Its first framework, the HARK project, is a modular code library for
constructing microeconomic and macroeconomic models with agents who differ from
each other in serious ways: in dimensions whose consequences cannot be
captured by analyzing the behavior of a single agent with average
characteristics.

The HARK project is the starting point because it is an area where both the need
and opportunities for improvement are great. In particular, existing code to
solve HA models tends to be bespoke and idiosyncratic, with the consequence that
tools are often reinvented by different researchers working on similar problems.
Researchers should spend their valuable time producing research, not reinventing
wheels. The HARK toolkit already provides a useful set of industrial strength,
reliable, reusable wheels, constructed using a simple and easily extensible
framework with clear documentation and testing regimens.

Part of the reason we are confident our goal is feasible is
that the tools now available – Python, GitHub, and Jupyter
notebooks among them – have finally reached a stage of maturity that can
handle the communication of almost any message an economist might want
to convey. [9]_

The longer-term goals of the Econ-ARK project are to create a collaborative
codebase that can serve the entire discipline of economics, employing the best
of modern software development tools to accelerate understanding and
implementation of cutting edge research tools. The solution methods employed in
HARK are not the only methods available, and those who have additional
methodological suggestions are strongly encouraged to contribute. The interested
user should check the Econ-ARK GitHub page, particularly the `HARK sub-page <https://github.com/econ-ark/HARK/>`__.
There you will find a README and documentation. For the interested contributor,
the `issues page <https://github.com/econ-ark/HARK/issues>`__ outlines the future
improvements in progress. Issues labeled with "help wanted" are particularly good
for getting started with contributing.

Acknowledgements
================

The Econ-ARK project is supported by a generous grant from the Alfred P. Sloan Foundation,
with fiscal sponsorship from NumFOCUS.  The authors would like to thank both organizations
for their time, resources, and expertise.



Bibliography
============


:math:`\;\;\;\;\;\;` [Adjemian2011] Adjemian, Stephane, Houtan Bastani, Michel Juillard, Ferhat Mihoubi, George Perendia, Marco Ratto, and Sebastien Villemot. 2011. "Dynare: Reference Manual, Version 4." *Dynare Working Papers* 1, CEPREMAP. `RePEc: cpmdynare/001 <https://econpapers.repec.org/paper/cpmdynare/001.htm>`__ .

[Carroll2012] Carroll, Christopher. 2012. "Solving Microeconomic Dynamic Stochastic Optimization Problems." *Lecture Notes, Johns Hopkins University.* `url <https://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs/>`__

[Carroll2017a] Carroll, Christopher, Jiri Slacalek, Kiichi Tokuoka, and Matthew N
White. 2017. "The Distribution of Wealth and the Marginal Propensity to
Consume." *Quantitative Economics* 8 (3). Wiley Online Library:
977–1020. `doi:10.3982/QE694 <https://doi.org/10.3982/QE694>`__

[Carroll2017b] Carroll, Christopher, Alexander Kaufman, David Low, Nathan Palmer, and
Matthew White. 2017. "A User’s Guide for Hark: Heterogeneous Agents
Resources and toolKit." *Econ ARK.* `url <https://github.com/econ-ark/HARK/blob/master/Documentation/HARKmanual.pdf>`__

[Geanakoplos2010] Geanakoplos, John. 2010. "The Leverage Cycle." *NBER Macroeconomics
Annual* 24 (1). The University of Chicago Press: 1-66. `doi:10.1086/648285 <https://doi.org/10.1086/648285>`__

[Geanakoplos2012] Geanakoplos, John, Robert Axtell, J Doyne Farmer, Peter Howitt, Benjamin
Conlee, Jonathan Goldstein, Matthew Hendrey, Nathan M. Palmer, and
Chun-Yi Yang. 2012. "Getting at Systemic Risk via an Agent-Based Model
of the Housing Market." *American Economic Review* 102 (3): 53-58. `doi:10.1257/aer.102.3.53 <https://www.aeaweb.org/articles?id=10.1257/aer.102.3.53>`__

[Guvenen2011] Guvenen, Fatih. 2011. "Macroeconomics with Heterogeneity: A Practical Guide,"
*Economic Quarterly, Federal Reserve Bank of Richmond* 97 (3): 255-326. `doi:10.3386/w17622 <https://www.nber.org/papers/w17622>`__

[Hommes2018] Hommes, Cars, and Blake LeBaron, eds. 2018. "Handbook of Computational Economics,
Vol 4: Heterogeneous Agent Modeling," *Handbook of Computational Economics*, Elsevier, Vol 4: 2-796. `doi:10.1016/S1574-0021(18)30018-2 <https://doi.org/10.1016/S1574-0021(18)30018-2>`__

[Kaplan2018] Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy
According to HANK." *American Economic Review* 108 (3): 697-743. `doi:10.1257/aer.20160042 <https://www.aeaweb.org/articles?id=10.1257/aer.20160042>`__

[Kirman1992] Kirman, Alan P. 1992. "Whom or What Does the Representative
Individual Represent?" *Journal of Economic Perspectives* 6 (2): 117-136. `doi:10.1257/jep.6.2.117 <https://www.aeaweb.org/articles?id=10.1257/jep.6.2.117>`__

[Tesfatsion2006] Tesfatsion, Leigh, Kenneth L. Judd, eds. 2006. "Handbook of Computational Economics,
Vol 2: Agent-Based Computational Economics," *Handbook of Computational Economics*, Elsevier, Vol 2: 829-1660. `doi:10.1016/S1574-0021(05)02039-3 <https://doi.org/10.1016/S1574-0021(05)02039-3>`__


.. [1]
 In this context, "heterogeneity" refers to both ex post heterogeneity--
 agents attaining different states or making different choices because
 they have experienced different random shocks in the model-- and ex ante
 heterogeneity-- agents differing in their preferences, beliefs, or other
 innate attribute before the model "begins".

.. [2]
 It is possible that some of the foundational tools from QuantEcon could
 be incorporated into the Econ-ARK, with the permission of its project leads.
 Our teams are in communication, and their advice has been valuable.

.. [3]
 HARK does not impose the assumption of rationality; we use it here for
 exposition because it is the standard assumption in economics.  The
 modular structure of the toolkit makes it easy to remove this assumption
 by, e.g., having agents misperceive their own problem, imperfectly process
 information, or form beliefs about aggregate processes that are not
 "justified" by the history.

.. [4]
 Running ``ConsIndShockModel.py`` also demonstrates other variations
 of the consumption-saving problem, but their description is omitted
 here for brevity.

.. [5]
 Roughly speaking, the universal distance metric is a recursive
 supnorm, returning the largest distance between two instances, among
 attributes named in ``distance_criteria``. Those attributes might be
 complex objects themselves rather than real numbers, generating a
 recursive call to the universal distance metric.

.. [6]
 Interpolation methods currently implemented in HARK include
 (multi)linear interpolation up to 4D, 1D cubic spline interpolation,
 2D curvilinear interpolation over irregular grids, a 1D “lower
 envelope” interpolator, and others.

.. [7]
 See [Carroll2017b] for a much more thorough discussion.

.. [8]
 The solution to a dynamic optimal control problem is a set of policy
 functions and a value function, for each period. The policy
 function for this consumption-saving problem is how much to consume
 :math:`C_t` for a given amount of market resources :math:`M_t`.
 The eleventh and final element of ``solution`` represents the trivial
 solution to the terminal period of the problem. For a much more detailed
 discussion, please see [Carroll2017b].

.. [9]
 See the recent blog post by Paul Romer, `“Jupyter, Mathematica, and the
 Future of the Research Paper” <https://paulromer.net/jupyter-mathematica-and-the-future-of-the-research-paper/>`__
 for a fuller argument).
