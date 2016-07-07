
:author: Andrew M. Fraser
:email: afraser@lanl.gov
:institution: XCP-8, Los Alamos National Laboratory
:corresponding:

:author: Stephen A. Andrews
:email: saandrews@lanl.gov
:institution: XCP-8, Los Alamos National Laboratory

.. latex::

    \newcommand{\dmudc}{\left(\frac{\partial \mu(c)}{\partial c} \right)}
    \newcommand{\normal}[2]{{\cal N}\left( #1,#2 \right)}
    \newcommand{\normalexp}[3]{ -\frac{1}{2}
      (#1 - #2)^T #3^{-1} (#1 - #2) }
    \newcommand{\La}{{\cal L}}
    \newcommand{\fnom}{\tilde f}
    \newcommand{\fhat}{\hat f}
    \newcommand{\COST}{\cal C}
    \newcommand{\LL}{{\cal L}}
    \newcommand{\Fisher}{{\cal I}}
    \newcommand{\Prob}{\text{Prob}}
    \newcommand{\field}[1]{\mathbb{#1}}
    \newcommand\REAL{\field{R}}
    \newcommand\Z{\field{Z}}
    \newcommand{\partialfixed}[3]{\left. \frac{\partial
      #1}{\partial#2}\right|_#3}
    \newcommand{\partiald}[2]{\frac{\partial #1}{\partial #2}}
    \newcommand{\argmin}{\operatorname*{argmin}}
    \newcommand{\argmax}{\operatorname*{argmax}}
    \newcommand\norm[1]{\left|#1\right|}
    \newcommand\bv{\mathbf{v}}
    \newcommand\bt{\mathbf{t}}
    \newcommand\vol{v}        % specific volume
    \newcommand{\pressure}{p}
    \newcommand{\eos}{f}
    \newcommand{\eosnom}{\tilde f}
    \newcommand{\EOS}{{\cal F}}
    \newcommand{\data}{y}
    \newcommand{\DATA}{{\cal Y}}
    \newcommand\Vfunc{\mathbb{V}}
    \newcommand\Vt{\mathbf{V}}
    \newcommand\vexp{V_{\rm{exp}}}
    \newcommand\texp{T_{\rm{exp}}}
    \newcommand\cf{c_f}
    \newcommand\cv{c_v}
    \newcommand\fbasis{b_f}
    \newcommand\vbasis{b_v}
    \newcommand\tsim{{\mathbf t}_{\rm{sim}}}
    \newcommand\DVDf{\partiald{\Vt}{f}}
    \newcommand\Lbb{\mathbb{L}}
    \newcommand\epv{\epsilon_v}
    \newcommand\epf{\epsilon_f}
    \newcommand{\EV}{\field{E}}
	      
==========================================================
 Functional Uncertainty Constrained by Law and Experiment
==========================================================

.. [1] LA-UR-16-23717

.. class:: abstract
		  
   Many physical processes are modeled by unspecified functions.
   Here, we introduce the `F_UNCLE` project which uses the Python
   ecosystem of scientific software to develop and explore techniques
   for estimating such unknown functions and our uncertainty about
   them.  The work provides ideas for quantifying uncertainty about
   functions given the constraints of both laws governing the
   function's behavior and experimental data.  We present an analysis
   of pressure as a function of volume for the gases produced by
   detonating an imaginary explosive, estimating a *best* pressure
   function and using estimates of *Fisher information* to quantify
   how well a collection of experiments constrains uncertainty about
   the function.  A need to model particular physical processes has
   driven our work on the project, and we conclude with a plot from
   such a process.
     
.. class:: keywords

   python, uncertainty quantification, Bayesian inference, convex
   optimization, reproducible research, function estimation, equation
   of state, inverse problems

Introduction
============

Some tasks require one to quantitatively characterize the accuracy of
models of physical material properties which are based on existing
theory and experiments.  If the accuracy is inadequate, one must then
evaluate whether or not proposed experiments or theoretical work will
provide the necessary information.  Faced with several such tasks, we
have chosen to first work on the equation of state (EOS) of the gas
produced by detonating an explosive called PBX-9501 because it is
relatively simple.  In particular Hixson et al. [hixson2000]_ describe
a model form that roughly defines its properties in terms of an
unknown one dimensional function (pressure as a function of volume on
a special path) and simple constraints.  This EOS estimation problem
shares the following challenges with many of the other material models
that we must analyze:

#. The uncertain object is a *function*.  In principal it has an
   infinite number of degrees of freedom.  In order to implement a
   Bayesian analysis one must define and manipulate probability
   measures on sets in function space.  We do not know how to define a
   probability measure on sets in function space, and we do not know
   how to compare the utility of different families of parametric
   approximations.

#. Understanding the constraints on the unknown function and the
   connection between it and experimental measurements requires
   understanding some detailed physics.

#. Simulations of some of the experiments run for more than a few
   minutes on high performance computers.  The job control is unwieldy
   as are the mechanisms for expressing trial instances of the unknown
   functions and connecting them to the simulations.

We are organizing our efforts to address those challenges under the
title `F_UNCLE` (Functional UNcertainty Constrained by
Law and Experiment).  We work in two parallel modes as we
develop ideas and algorithms.  We write code for a surrogate problem
that runs in a fraction of a minute on a PC, and we write code for
fitting a model to PBX-9501 in a high performance computing
environment.  Our focus shifts back and forth as we find and resolve
problems.  As we have progressed, we have found that improving our software
practices makes it easier to express ideas, test them on PCs and
implement them on the HPCs.  In this paper, we introduce the
[F_UNCLE]_ code, the surrogate problem we have developed for the EOS and
our analysis of that problem.

We are also using the project to learn and demonstrate *Best Practices
for Scientific Computing* (eg, [wilson2014]_) and *Reproducible
Research* (eg, [fomel2009]_).  The work is designed to be modular,
allowing a wide range of experiments and simulations to be used in an
analysis.  The code is self documenting, with full docstring coverage,
and is converted into online documentation using [sphinx]_.  Each
class has a test suite to allow unit testing.  Tests are collected and
run using [nose]_.  Each file is also tested using [pylint]_ with all
default checks enabled to ensure it adheres to Python coding
standards, including PEP8.  Graphics in this paper were generated
using [matplotlib]_ and the code made use of the [numpy]_ and [scipy]_
packages.  Among the reasons we chose the Python/SciPy ecosystem, the
most important are:

Readable
   Writing in Python helps us implement the most important point in
   [wilson2014]_ : "Write programs for people, not computers."

Versatile
   The Python Standard Library lets easily us connect our scripts to
   other code, eg, submitting HPC jobs and wrapping libraries written
   in other languages.

Community support
   Because of the large number of other users, it is easy to get
   answers to questions.

Numerical packages
   We use a host of modules from Numpy, SciPy and other sources. 

Portable
   With the Python/SciPy ecosystem, it is easy to write code that
   runs on our desktops and also runs in our HPC environment.

The task of mapping measurements to estimates of the characteristics
of models for the physical processes that generated them is called an
*inverse problem*.  Classic examples include RADAR, tomography and
image estimation.  Our problems differ from those in the diverse and
indirect nature of the measurements, the absence of translation
invariance and in the kinds of constraints.  [F_UNCLE]_ uses
constrained optimization and physical models with many degrees of
freedom to span a large portion of the allowable function space while
strictly enforcing constraints.  The analysis determines the function
that maximizes the a posteriori probability (the MAP estimate) using
simulations to match :math:`K` data-sets.  We characterize how each
experiment constrains our uncertainty about the function in terms of
its *Fisher information*.

As a surrogate problem, we have chosen to investigate the equation of
state (EOS) for the products-of-detonation of a hypothetical High
Explosive (HE).  The EOS relates the pressure to the specific volume
of the products-of-detonation mixture.  We follow traditional practice
(eg, [ficket2000]_) and constrain the function to be positive,
monotonically decreasing and convex.  To date we have incorporated two
examples of experiments: The detonation velocity of a *rate stick* of
HE and the velocity of a projectile driven by HE. The behavior of both
these experiments depend sensitively on the EOS function.

The following sections describe the choices made in modeling the EOS
function, the algorithm used for estimating the function and the use
of the Fisher information to characterize the uncertainty about the
function.  Results so far indicate optimization can find good
approximations to the unknown functions and that analysis of Fisher
information can quantify how various experiments constrain uncertainty
about their different aspects.  While these preliminary results are
limited to an illustration of the ideas applied to synthetic data and
simple models, the approach can be applied to real data and complex
simulations.  A plot from work on estimating the EOS of the high
explosive PBX-9501 appear in the concluding section.

Fisher Information and a Sequence of Quadratic Programs
=======================================================
:label:`sequence`


Our analysis is approximately Bayesian and Gaussian. We suppose that:

#. Experiments provide data :math:`x=[x_0,\ldots,x_n]`, where
   :math:`x_k` is the data from the :math:`k^{th}` experiment

#. We have a likelihood function :math:`p_l(x|\theta) = \prod_k
   p_l(x_k|\theta)` in which the data from different experiments are
   conditionally independent given the parameters :math:`\theta`

#. We have a prior on the parameters :math:`p_p(\theta)`

From those assumptions, one can write the *a posteriori* distribution of
the parameters as

.. math::
   :label: eq-bayes

   p(\theta|x) = \frac{p_l(x|\theta) p_p(\theta)}{\int p_l(x|\phi) p_p(\phi) d\phi}.

Rather than implement Equation (:ref:`eq-bayes`) exactly, we use a
Gaussian approximation calculated at

.. math::
   :label: eq-map

   \hat \theta \equiv {\operatorname*{argmax}}_{\theta} p(\theta|x).

Since :math:`\theta` does not appear in the denominator on the right
hand side of Equation (:ref:`eq-bayes`), in a Taylor series expansion
of the log of the *a posteriori* distribution about :math:`\hat \theta`
the denominator only contributes a constant added to expansions of the
log of the likelihood and the log of the prior.

.. math::
   :type: align

   &\log \left( p(\theta|x) \right) = \log \left( \frac{p_l(x|\hat \theta)
         p_p(\hat \theta)}{\int p_l(x|\phi) p_p(x) d\phi} \right) \nonumber \\
     &\qquad~+ \frac{1}{2}
     \left( \theta - \hat \theta \right)^T \left(
       \frac{d^2 \log\left( p_l(x|\phi) \right) }{d\phi^2} +
       \frac{d^2 \log \left( p_p(\phi) \right) }{d\phi^2} 
     \right)_{\phi=\hat \theta} \left( \theta - \hat \theta \right)\\&\qquad + R \nonumber \\
     \label{eq:taylor}
     &\qquad\equiv C + \frac{1}{2}
     \left( \theta - \hat \theta \right)^T H \left( \theta - \hat \theta \right)
     + R

Dropping the higher order terms in the remainder :math:`R` in leaves
the normal or Gaussian approximation

.. math::
   :type: align

   \theta|x &\sim {{\cal N}\left( \hat \theta,\Sigma = H^{-1} \right)}\\
     p(\theta|x) &= \frac{1}{\sqrt{(2\pi)^{\text{dim}}|\Sigma|}} \exp\left(
       -\frac{1}{2}(\theta-\hat\theta)^\mathrm{T}\Sigma^{-1}
        (\theta-\hat\theta) \right).

With this approximation, experiments constrain the a posteriori
distribution by the second derivative of their log likelihoods.

Quoting Wikipedia: “If :math:`p(x|\theta)` is twice differentiable with
respect to :math:`\theta`, and under certain regularity conditions, then
the Fisher information may also be written as

.. math::
   :label: eq-fisher

   \mathcal{I}(\theta) = - \EV_X
     \left[\left. \frac{\partial^2}{\partial\theta^2} \log
         p(X;\theta)\right|\theta \right].

[...] The Cramér–Rao bound states that the inverse of the Fisher
information is a lower bound on the variance of any unbiased
estimator”

Our simulated measurements have Gaussian likelihood function in which
the unknown function only influences the mean.  Thus we calculate the
second derivative of the log likelihood as follows:

.. math::
   :type: align
   
   L &\equiv -\frac{1}{2} \left(x - \mu(\theta) \right)^T \Sigma^{-1}
	  \left(x - \mu(\theta) \right) +C \\
	  \partiald{L}{\theta} &= \left(x - \mu(\theta) \right)^T
	  \Sigma^{-1} \partiald{\mu}{\theta}  \\
	  \frac{\partial^2}{\partial \theta^2} L &=
	  -\left(\partiald{\mu}{\theta}\right)^T \Sigma^{-1}
	  \left(\partiald{\mu}{\theta})\right) + \left(x - \mu(\theta)
	  \right)^T \Sigma^{-1} \frac{\partial^2 \mu}{\partial
      \theta^2} \\
      &\quad \text{and} \\
   \EV_X \frac{\partial^2}{\partial \theta^2} L &\approx -\left(\partiald{\mu}{\theta}\right)^T
	  \Sigma^{-1} \left(\partiald{\mu}{\theta}\right).
   
The approximation is valid if either :math:`\frac{\partial^2
\mu}{\partial \theta^2}` is roughly constant or :math:`\left(x -
\mu(\theta) \right)` is small (both of which are probably true).


Iterative Optimization
----------------------

We maximize the *log* of the a posteriori probability as the objective
function which is equivalent to :ref:`eq-map`.  Dropping terms that
do not depend on :math:`\theta`, we write the cost function as follows:

.. math::
   :type: align

     C(\theta) &\equiv -\log(p(\theta)) - \sum_k \log(p(x_k|\theta)) \\
     &\equiv \frac{1}{2} (\theta-\mu)^T \Sigma^{-1} (\theta-\mu) - 
     \sum_k \log(p(x_k|\theta)),

where :math:`k` is an index over a set of independent experiments. We
use the following iterative procedure to find :math:`\hat \theta`, the
*Maximum A posteriori Probability* (MAP) estimate of the parameters:

#. Set :math:`i=0` and :math:`\theta_i[j] = \mu[j]`, where :math:`i` is the
   index of the iteration and :math:`j` is index of the components of
   :math:`\theta`.

#. Increment :math:`i`

#. Estimate :math:`P_i` and :math:`q_i` defined as

      .. math::
	 :type: align

	 q_i^T &\equiv \left. \frac{d}{d\theta} C(\theta)\right|_{\theta=\theta_{i-1}} \\
	     P_i &\equiv \left. \frac{d^2}{d\theta^2} C(\theta)\right|_{\theta=\theta_{i-1}}
	 

   Since the experiments are independent, the joint likelihood is the
   product of the individual likelihoods and the log of the joint
   likelihood is the sum of the logs of the individual likelihoods,
   ie,

   .. math::
      :type: align

      q_i^T &\equiv (\theta_{i-1}-\mu)\Sigma^{-1} + \sum_k
            \left. \frac{d}{d\theta} \log(p(x_k|\theta)\right|_{\theta=\theta_{i-1}} \nonumber \\
            & \equiv (\theta_{i-1}-\mu)\Sigma^{-1} + \sum_k q_{i,k}^T \\
            P_i &\equiv \Sigma^{-1} + \sum_k
            \left. \frac{d^2}{d\theta^2} \log(p(x_k|\theta)\right|_{\theta=\theta_{i-1}}\nonumber \\
            &\equiv \Sigma^{-1} + \sum_k P_{i,k}

   where in :math:`P_{i,k}` and :math:`q_{i,k}`, :math:`i` is the
   iteration number and :math:`k` is the experiment number.

#. Calculate the matrix :math:`G_i` and the vector :math:`h_i` to
   express the appropriate constraints [2]_.

#. Calculate :math:`\theta_i = \theta_{i-1} + d` by solving the
   quadratic program

   .. math::
      :type: align

      \text{Minimize } & \frac{1}{2} d^T P_i d + q^T d \\
      \text{Subject to } & G_id \preceq h_i
      
   where :math:`\preceq` means that for each component the left hand
   side is less than or equal to the right hand side.
      
#. If not converged go back to step 2.

.. [2] For our surrogate problem, we constrain the function at the
       last knot to be positive and have negative slope.  We also
       constrain the second derivative to be positive at every knot.
       See the [F_UNCLE]_ code and documentation for more details.

This algorithm differs from modern SQP methods as each QP sub-problem
is has no knowledge of the previous iteration. This choice is
justified as the algorithm converges in less than 5 outer loop
iterations. This unconventional formulation helps accelerate
convergence as the algorithm does not need multiple outer loop
iterations to obtain a good estimate of the Hessian, as in modern SQP
methods.

.. figure:: scipy2016_figure6.pdf

   Convergence history of a typical solution to the MAP optimization problem
   
The assumption that the experiments are statistically independent
enables the calculations for each experiment :math:`k` in to be done
independently. In the next few sections, we describe both the data
from each experiment and the procedure for calculating :math:`P_i[k]`
and :math:`q_i[k]`. 

Equation of State
=================
:label:`eos`

For our surrogate problem, we say that the thing we want to estimate,
:math:`\theta`, represents the equation of state (EOS) of a gas.  We
also say that the state of the gas in experiments always lies on an
isentrope [3]_ and consequently the only relevant data is the pressure as a
function of specific volume (ml/gram) of the gas.  For physical
plausibility, we constrain the function to have the following
properties:

* Positive
* Monotonic
* Convex

.. [3] In an *isentropic* expansion or compression there is no heat
       conduction.  Our isentropic approximation relies on the
       expansion being so rapid that there is not enough time for heat
       conduction.
   
Here, let us introduce the following notation: 
  
* :math:`\vol` Specific volume
* :math:`p` Pressure
* :math:`\eos` An EOS that maps specific volume to pressure,
  :math:`\eos: \vol \mapsto \pressure`.
* :math:`v_0` The minimum relevant volume.
* :math:`v_1` The maximum relevant volume.
* :math:`\EOS` The set of possible EOS functions, :math:`p(v), v_0 \leq v
  \leq v_1`

    
Cubic Splines
-------------

While no finite dimensional coordinate scheme can represent every
element of :math:`\EOS`, the flexibility of cubic splines lets us get
close to any element of :math:`\EOS` using a finite number of
parameters.  (An analysis of the efficiency of various representations
is beyond the scope of this paper.)

Constraining :math:`\eos` to be positive and to be a convex function
of :math:`\vol` is sufficient to ensure that it is also monotonic.
Although we are working on a definition of a probability measure on a
sets of functions that obeys those onstraints and is further
constrained by :math:`\frac{\left| \eos(\vol) -
\mu_\eos(\vol)\right|}{\mu_\eos(\vol)} \leq \Delta`, for now, we
characterize the prior as Gaussian.  As we search for the mean of the
a posteriori distribution, we enforce the constraints, and the result
is definitely not Gaussian.  For the remainder of the present work we
ignore that inconsistency and use a prior defined in terms of spline
coefficients.  We start with a nominal EOS

.. math::
   :label: eq-nom
	 
   \eosnom(\vol) = \frac{F}{\vol^3}, \text{ where } F \leftrightarrow
   2.56\times10^9 \text{Pa} \text{ at one cm}^{3}\text{g}^{-1}

and over a finite domain we approximate it by a cubic spline with
coefficients :math:`\left\{\tilde \cf[i] \right\}`.  Thus :math:`c`,
the vector of spline coefficients, is the set of unknown parameters
that we have previously let :math:`\theta` denote.  Then we assign a
variance to each coefficient:

.. math::
  :label: eq-3

  \sigma^2[i] = \left( \cf[i] \Delta \right)^2.

We set :math:`\Delta = 0.05`.  These choices yield:

.. math::
   :type: align
	  
   \mu_\eos &\leftrightarrow \left\{\tilde c[i] \right\} \\
   \Sigma_\eos[i,j] &= \tilde \sigma^2[i] \delta_{i,j}

Thus we have the following notation for splines and a prior
distribution over :math:`\EOS`.

* :math:`\cf,\fbasis` Vector of coefficients and cubic spline basis
  functions that define an EOS.  We will use :math:`cf[i]` and
  :math:`\fbasis[i]` to denote components.
* :math:`\mu_\eos, \Sigma_\eos` Mean and covariance of prior
  distribution of EOS.  In a context that requires coordinates, we let
  :math:`\mu_\eos = \left( \cf[0], \cf[1], \ldots , \cf[n] \right)^T`.


The Nominal and *True* EOS
--------------------------

For each experiment, data comes from a simulation using a *true*
function and each optimization starts from the nominal EOS which is
the mean of the prior given in :ref:`eq-nom`.  We've made the *true*
EOS differ from the nominal EOS by a sum of Gaussian bumps.  Each bump
is characterized by a center volume :math:`v_k`, a width :math:`w_k`
and a scale :math:`s_k`, with:

.. math::

   b_k(v) = \frac{s_k F}{v_k^3} e^{- \frac{(v-v_k)^2}{2w_k^2}}

Throughout the remainder of this paper, the *true* EOS that we have
used to generate pseudo-experimental data is:

.. math::
   :label: eq-actual
   :type: align
	  
   f(v)&= \frac{F}{v^3} + b_0(v) + b_1(v)


where: :math:`v_0 = .4\, \text{cm}^3\text{g}^{-1}`, :math:`~w_0 = .1\,
\text{cm}^3\text{g}^{-1}`, :math:`~s_0 = .25`, :math:`v_1 = .5\,
\text{cm}^3\text{g}^{-1}`, :math:`w_1 = .1\, \text{cm}^3\text{g}^{-1}`,
and :math:`s_1 = -.3`.

.. figure:: scipy2016_figure1eos.pdf

   The prior and nominal *true* equation of state function. The two
   models differ most at a specific volume of 0.4 g cm :math:`^{-1}`
   

A Rate Stick
============

The data from this experiment represent a sequence of times that a
detonation shock is measured arriving at locations along a stick of HE
that is so thick that the detonation velocity is not reduced by
curvature.  The code for the pseudo data uses the average density and
sensor positions given by Pemberton et al.  [pemberton2011]_ for their
*Shot 1*.

.. figure:: stick.pdf

   The rate stick experiment showing the detonation wave propagating
   through the rate stick at the CJ velocity. Detonation velocity
   is measured by the arrival time of the shock at the sensors placed
   along the stick.

   
Implementation
--------------

The only property that influences the ideal measurements of rate stick
data is the HE detonation velocity.  Code in
`F_UNCLE.Experiments.Stick` calculates that velocity following Section
2A of Fickett and Davis [ficket2000]_ (entitled *The Simplest
Theory*).  The calculation solves for conditions at what is called the
*Chapman Jouguet* (CJ) state.  The CJ state is defined implicitly by a
line (called the *Rayleigh line*) in the :math:`(p,v)` plane that goes
through :math:`(p_0,v_0)`, the pressure and volume before detonation,
and :math:`(p_\text{CJ},v_{CJ})`.  The essential requirement is that
the Rayleigh line be tangent to the isentrope or EOS curve in the
:math:`(p,v)` plane.  The slope of the Rayleigh line that satisfies
those conditions defines the CJ velocity, :math:`V` in terms of the
following equation:

.. math::
   	   
   \frac{V^2}{v_0^2} = \frac{p_\text{CJ}-p_0}{v_0-v_\text{CJ}}.

For each trial EOS, the `F_UNCLE` code uses the
`scipy.optimize.brentq` method in a nested loop to solve for
:math:`(p_\text{CJ},v_{CJ})`.  Figure :ref:`fig-cj-stick` shows the
EOS and both the Rayleigh line and the CJ point that the procedure
yields.

.. The `scipy.optimize.brentq` was chosen as it did not require an
   initial estimate of detonation velocity but rather used the bounds
   of the detonation velocity, which could be estimated *a
   priori*. With good estimates of the detonation velocity bounds, the
   algorithm was sufficiently robust to be used within the MAP
   optimization procedure described previously.

.. figure:: scipy2016_figure1.pdf
   :align: center  
	   
   Isentropes, a Rayleigh line and the CJ conditions. Starting from the
   isentrope labeled *Prior EOS* and using data from simulated
   experiments based on the isentrope labeled *True EOS*, the
   optimization algorithm described in the Algorithm section produced
   the estimate labeled *Fit EOS*.  Solving for the CJ state of *Fit
   EOS* isentropes yields a Rayleigh line.  The data constrains the
   isentrope only at :math:`v_\text{CJ}`. :label:`fig-cj-stick`

Comparison to Pseudo Experimental Data
--------------------------------------

The previous section explained how to calculate the detonation
velocity, :math:`V_{\text{CJ}}(\eos)`, but the *experimental* data are
a series of times when the shock reached specified positions on the
rate-stick.  The simulated detonation velocity is related to these
arrival times by:

.. math::

   t[j] = \frac{x[j]}{V_{\text{CJ}}(\eos)}.

where :math:`x[j]` are the locations of each sensor measuring arrival
time.

We let :math:`D` denote the sensitivity of the set of simulated
arrival times to the spline coefficients governing the equation of
state, and write:

.. math::
   
  D[j,i] \equiv \frac{\partial t[j]}{\partial c[i]}.

We use finite differences to estimate :math:`D`.

The Gun
=======

The data from this experiment are a time series of measurements of a
projectile's velocity as it accelerates along a gun barrel driven by
the expanding products-of-detonation of HE.  Newton's equation

.. math::
   
   F = ma

determines the velocity time series.  The pressure from the EOS times
the area of the barrel cross section is the force.

.. figure:: gun.pdf

   The gun experiment. The projectile of a given mass and
   cross-sectional area is accelerated along the barrel by the
   expanding products of combustion from the high explosives in the
   barrel.

   
Implementation
--------------

The position and velocity history of the projectile is generated by
the `scipy.integrate.odeint` algorithm. This method solves the
differential equation for the projectile position and velocity as it
is accelerated along the barrel.

.. math::
   :label: eq-gun-difeq
   :type: align
      
   \frac{\mathrm{d}x(t)}{\mathrm{d}t} & = v(t) \\
   \frac{\mathrm{d}v(t)}{\mathrm{d}t} & = \frac{A}{m_{proj}} \eos\left( \frac{x(t) A}{m_{HE}} \right)

where:

* :math:`t` is time from detonation (assuming the HE burns instantly)
* :math:`x(t)` is the position of the projectile along the barrel  
* :math:`v(t)` is the velocity of the projectile
* :math:`A` is the cross-sectional area of the barrel
* :math:`m_{HE}` is the initial mass of high explosives
* :math:`m_{proj}` is the mass of the projectile  
* :math:`\eos` is the equation of state which relates the pressure to
  the specific volume of the HE products-of-detonation

The acceleration is computed based the projectile's mass and the force
resulting from the uniform pressure acting on the projectile. This
pressure is related to the projectile's position by the EOS, assuming
that the projectile perfectly seals the barrel so the mass of
products-of-detonation behind the projectile remains constant.

Comparison to Pseudo Experimental Data
--------------------------------------

We generated *experimental data* using our simulation code with the
nominal *true* EOS described previously. These experimental data were
a series of times and corresponding velocities. To compare the
experiments to simulations, which may use a different time
discretization, the simulated response was represented by a spline,
and was compared to the experiments at each experimental time stamp.

.. math::
   :label: gun_sens
	   
   D[j,i] = \partiald{\hat{v}(t_{exp}[j])}{\cf[i]}

where:

* :math:`\hat{v}` is the velocity given from the spline fit to
  simulated :math:`v(t)` data
* :math:`t_{exp}` is the times where experimental data were available
    
Numerical Results
=================

The algorithm was applied to the sets of simulation results and pseudo
experimental data for both the rate-stick and gun models. Figure
:ref:`fig-opt-stick` shows the improved agreement between the
simulated and *experimental* arrival times after the algorithm adjusts the
equation of state. Similar results are shown in Figure
:ref:`fig-fve-gun` , where the significant error in velocity history
at early times is reduced by an order of magnitude with the optimized
EOS model.

.. figure:: scipy2016_figure3.pdf
   :align: center   

   Fitting an isentrope to rate stick data.  Green
   +'s denote measured shock arrival time at various positions.  The blue
   line represents the shock velocity calculated from the nominal EOS,
   and the black line is the result of the optimization algorithm described
   in the text. :label:`fig-opt-stick`


.. figure:: scipy2016_figure4.pdf
   :align: center	   

   Estimation of the maximum *a posteriori* probability
   parameters of the gun experiment.  The *True EOS* appears
   in the upper plot, and the optimization starts with the *Prior EOS*
   and ends with *Fit EOS*.  The corresponding velocity for the gun as a
   function of position appears in the lower plot. The estimation also
   used experimental data from the rate stick. :label:`fig-fve-gun`


Fisher Information Matrix
-------------------------

The Fisher information matrix characterizes how tightly the
experimental data constrain the spline coefficients. This matrix can
be better understood through a spectral decomposition to show the
magnitude of the eigenvalues and the eigenvector behavior.

The eigenvalues and eigenvectors of the Fisher information matrix of
the rate-stick experiment are shown in Figure
:ref:`fig-info-stick`. Only the CJ point on the EOS influences the
forecast data, :math:`\mu(c)`.  Thus only one degree of freedom in the
model influences the likelihood and Fisher Information matrix,
:math:`\Fisher`, should have a rank of one. Figure
:ref:`fig-info-stick` illustrates characteristics of the optimization
procedure and :math:`\Fisher(\hat c)`.  The largest eigenvalue
:math:`\Fisher(\hat c)` is :math:`10^{4}` larger than the next
largest. We expected the rank to be one to within numerical precision.

.. figure:: scipy2016_figure2.pdf
   :align: center
   :class: w	   

   Fisher Information of the Rate Stick Experiment. The largest three
   eigenvalues of :math:`\Fisher(\hat c)` appear in the upper plot and
   the eigenfunction corresponding to these three eigenvalues appears in
   he lower plot. :label:`fig-info-stick`
   

The Fisher information matrix of the gun experiment is more complex as
changes to the EOS affect the entire time history of the projectile
velocity. In Figure :ref:`fig-info-gun` There is no clear *dominating*
eigenvalue, the largest eigenvalue corresponds to an eigenvector which
is more influential at smaller projectile displacements while the next
three largest eigenvalues correspond to eigenvectors which are more
influential across the range of displacements.

.. figure:: scipy2016_figure5.pdf
   :align: center	    

   Fisher Information of the Gun Experiment.  The largest four
   eigenvalues of :math:`\Fisher(\hat c)`
   appear in the upper plot and the eigenfunctions corresponding to
   the largest four eigenvalues appear in the lower plot. :label:`fig-info-gun`

These preliminary investigations of the Fisher information matrix show
how this matrix can be informative in describing the uncertainty
associated with the optimal EOS function determined by the [F_UNCLE]_
algorithm.  Notice that the eigenvectors of the matrix describe
functions that are  zero for states not visited by the gun
experiment.
   
Conclusion, Caveats and Future Work
===================================

We have described an iterative procedure for estimating functions
based on experimental data in a manner that enforces chosen
constraints.  The [F_UNCLE]_ code implements the procedure, and we
used it to make the figures in the previous sections.  The code runs
on a modest desktop computer and makes the figures in a fraction of a
minute.  That speed and simplicity allows one to easily try out new
ideas and code.  We have relied on the [F_UNCLE]_ code to guide work
with real experimental data and simulations on high performance
computers that use proprietary software.  Figure :ref:`fig-pbx` is the
result of applying the ideas presented here to the physical
experiments described in [pemberton2011]_.

.. figure:: fit_v.pdf
   :align: center
	
   Improvement of match between true experiments on PBX-9501 and
   simulations on a high performance computer.  The mean of the
   experimental data is labeled :math:`\mu`, and the optimization
   scheme yields the EOSs that produce the traces labeled
   :math:`fit_n`. :label:`fig-pbx`

The [F_UNCLE]_ code has been useful for us, and while we believe it
could be useful for others, we emphasize that it is a work in
progress.  In particular:

* The prior is inconsistent.  We hope to analyze and perhaps mitigate
  the effects of that inconsistency in future work.
* The choice of splines is not justified.  We plan to compare the
  performance of coordinate system options in terms of quantities such
  as bias and variance in future work.
* The optimization procedure is ad hoc and we have not considered
  convergence or stability.  We have already begun to
  consider other optimization algorithms.

We have designed the [F_UNCLE]_ code  so that one can easily
use it to model any process where there is a simulation which depends
on a model with an unknown functional form. The self documenting
capabilities of the code and the test suites included with the source
code will help others integrate other existing models and simulations
into this framework to allow it to be applied to many other physical
problems.

References
==========

.. [cvxopt] Andersen, M. and Vandenberghe, L.. "cvxopt\: Convex
            Optimization Package" `<http://cvxopt.org/>`_ [Online;
            accessed 2016-05-27].

.. [ficket2000] Ficket, W. and
                Davis, W. C., 2000. "Detonation". University of
                California Press: Berkeley, CA.

.. [fomel2009] Fomel, Sergey, and Jon F. Claerbout. "Reproducible
	       research." Computing in Science & Engineering 11.1
	       (2009): 5-7.

.. [F_UNCLE] "F_UNCLE: Functional Uncertainty Constrained by Law and
             Experiment" `https://github.com/fraserphysics/F_UNCLE
             <https://github.com/fraserphysics/F_UNCLE>`_ [Online;
             accessed 2016-05-27].

.. [hill1997] Hill, L. G., 1997. "Detonation Product Equation-of-State
              Directly From the Cylinder Test". Proc. 21st
              Int. Symp. on Shock Waves, Great Keppel Insland,
              Australia.

.. [hixson2000] Hixson, R. S. et al., 2000. "Release isentropes of
                overdriven plastic-bonded explosive PBX-9501."
                *J. Applied Physics* **88** (11) pp. 6287-6293

.. [matplotlib] Hunter, J. D.. "Matplotlib\: A 2D Graphics
                Environment", Computing in Science & Engineering,
                **9**, 90-95 (2007), `DOI:10.1109/MCSE.2007.55
                <https://doi.org/10.1109/MCSE.2007.55>`_

.. [nose] "nose: Nose Extends Unittest to Make Testing Easier"
          `<https://pypi.python.org/pypi/nose/1.3.7>`_ [Online;
          accessed 2016-05-27].

.. [numpy] van der Walt, S. , Colbert, C. S.  and Varoquaux, G.. "The
           NumPy Array\: A Structure for Efficient Numerical
           Computation", Computing in Science \& Engineering, **13**,
           22-30 (2011), `DOI:10.1109/MCSE.2011.37
           <https://doi.org/10.1109/MCSE.2011.37>`_

.. [pemberton2011] Pemberton et al. "Test Report for Equation of State
                   Measurements of PBX-9501". LA-UR-11-04999, Los
		   Alamos National Laboratory, Los Alamos, NM.

.. [pylint] "pylint\: Python Code Static Checker"
            `<https://www.pylint.org/>`_ [Online; accessed
            2016-05-27].

.. [scipy] Jones, E., Oliphant, E., Peterson, P., et al. "SciPy\: Open
	   Source Scientific Tools for Python", 2001-,
	   `<http://www.scipy.org/>`_ [Online; accessed 2016-05-27].


.. [sphinx] "sphinx\: Python Documentation Generator"
            `<http://www.sphinx-doc.org/>`_ [Online; accessed
            2016-05-27].

.. .. [vaughan2014] Vaughan, D. E. and Preston, D. L. "Physical Uncertainty
		 Bounds (PUB)". LA-UR-14-20441, Los Alamos National
		 Laboratory, Los Alamos, NM.

.. [wilson2014] Wilson, Greg, et al. "Best practices for scientific
		computing." PLoS Biol 12.1 (2014): e1001745.
		



..
   Local Variables:
   mode: rst
   compile-command: "cd ../..; ./make_paper.sh papers/andrew_fraser"
   End:
