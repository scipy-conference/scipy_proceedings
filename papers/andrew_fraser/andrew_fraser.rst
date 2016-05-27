
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
    \newcommand{\partialfixed}[3]{\left. \frac{\partial #1}{\partial#2}\right|_#3}
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
	      
==========================================================
 Functional Uncertainty Constrained by Law and Experiment
==========================================================

.. [1] LA-UR-16-23717

.. class:: abstract

   Many physical processes can be represented by functions with known
   general behavior which however we cannot specify precisely. The
   `F_UNCLE` project develops theory and code for understanding the
   uncertainty about such functions given the constraints of both laws
   governing the function's behavior and experimental data. Here, we
   present a procedure for determining both the most likely function
   to represent a physical process as well as understanding how this
   function is constrained by the given experimental data. An example
   application of this process is given for estimating the equation of
   state for the products-of-combustion of a high explosive. Simulated
   data are compared to reduced order models for two different
   experiments and the best estimate for the equation of state is
   given as well as information about how informative each experiment
   is in determining the best equation of state model.
     
.. class:: keywords

   uncertainty quantification, Bayesian inference, convex
   optimization, reproducible research, function estimation, equation
   of state

Introduction
============

There are many physical process whose general behavior is known though
not an exact mathematical representation. Such epistemic uncertainty
can arise in processes which occur at extreme regimes where direct
measurement is challenging. Existing approaches for estimating the
functional form of such processes included Gaussian Process Modeling
(GPM) and parametric models. Traditional GPM approaches may not be
able to account for the known constraints on the functional form
because they allow physically impossible functions. Many parametric
approaches overly constrain the function and do not span the allowable
function-space. The approach presented in this paper uses constrained
optimization and physical models with many degrees of freedom to span
a large portion of the allowable function space while disallowing
functions which do not follow the known constraints on the function.

This approach, demonstrated in the [F_UNCLE]_ project, provides a way
to describe the uncertainty in the functional form of such a physical
process. The analysis determines the function which maximizes the
probability of :math:`K` different simulations matching :math:`K`
corresponding data-sets while meeting all constraints given by *a
priori* knowledge of the functional form.  We characterize our
uncertainty about this function using the Fisher information matrix of
the likelihood function.

To ensure that the results of this project are verifiable and
reproducible, we distribute the source code for the [F_UNCLE]_
project.  Additionally, this project is intended to demonstrate good
software development practices within Los Alamos National
Laboratory. The project is designed to be modular, allowing a wide
range of experiments and simulations to be used in the analysis. The
code is self documenting, with full docstring coverage, and is
converted into a user manual using [sphinx]_. Each class has a test
suite to allow unit testing. Tests are collected and run using
[nose]_. Each file is also tested using [pylint]_ with all default
checks enabled to ensure it adheres to python coding standards,
including PEP8.  Graphics in this paper were generated using
[matplotlib]_ and the code made use of the [numpy]_ package

In this paper, the functional form under investigation is the equation
of state (EOS) for the products-of-combustion of a High Explosive
(HE). The EOS relates the pressure to the specific volume of the
products-of-combustion mixture. Previous work in this field
[ficket2000]_ has shown this function to be positive, monotonically
decreasing and convex. However, the extreme pressures and temperatures
of HE products-of-combustion preclude experimental measurements of the
EOS directly, and its behavior must be inferred.  Two examples of
experiments are given: the detonation velocity of a *rate stick* of HE
and the velocity of a projectile driven by HE. The behavior of both
these experiments is highly dependent on the EOS model.

The following sections describe the choices made in modeling the EOS
function, the algorithm used for estimating the function and the use
of the Fisher information to characterize the uncertainty in the
function.  We describe two sets of simulations and synthetic
experimental data and present an EOS function fit to represent both
these experiments as well as a spectral analysis of the Fisher
information matrix.  While the results are limited to an illustration
of the [F_UNCLE]_ project applied to synthetic data and simple models,
the [F_UNCLE]_ approach can be applied to real data and complex finite
difference simulations. Some preliminary results from work on
estimating the EOS of the high explosive PBX-9501 appear in the
concluding section.  In doing that work we rely on [F_UNCLE]_ for
developing and testing code and ideas.


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

   p(\theta|x) = \frac{p_l(x|\theta) p_p(\theta)}{\int p_l(x|\phi) p_p(x) d\phi}.

Rather than implement Equation (:ref:`eq-bayes`) exactly, we use a
Gaussian approximation calculated at

.. math::
   :label: eq-map

   \hat \theta \equiv {\operatorname*{argmax}}_{\phi} p(\theta|x).

Since :math:`\theta` does not appear in the denominator on the right
hand side of Equation (:ref:`eq-bayes`), in a Taylor series expansion
of the log of the a posteriori distribution about :math:`\hat \theta`
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
the normal or Gaussian

.. math::
   :type: align

   \theta|x &\sim {{\cal N}\left( \hat \theta,\Sigma = H^{-1} \right)}\\
     p(\theta|x) &= \frac{1}{\sqrt{(2\pi)^{k}|\Sigma|}} \exp\left(
       -\frac{1}{2}(\theta-\hat\theta)^\mathrm{T}\Sigma^{-1}
        (\theta-\hat\theta) \right).

With this approximation, experiments constrain the a posteriori
distribution by the second derivative of their log likelihoods.

Quoting Wikipedia: “If :math:`p(x|\theta)` is twice differentiable with
respect to :math:`\theta`, and under certain regularity conditions, then
the Fisher information may also be written as”

.. math::

   \mathcal{I}(\theta) = - \operatorname{E}
     \left[\left. \frac{\partial^2}{\partial\theta^2} \log
         p(X;\theta)\right|\theta \right].

Thus if the second derivative in is constant with respect to :math:`x`
(As it would be for a Gaussian likelihood), then one may say that an
experiment constrains uncertainty through its Fisher Information.

Iterative Optimization
----------------------

We use the log of the a posteriori probability as the objective function.
Dropping terms that don't depend on :math:`\theta`, we write the cost function
as follows:

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
	 

   Since the experiments are independent the joint likelihood is the
   product of the individual likelihoods and the log of the joint
   likelihood is the sum of the logs of the individual likelihoods, ie,

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

#. Calculate :math:`G_i` and :math:`h_i` to express the appropriate
   constraints

#. Calculate :math:`\theta_i = \theta_{i-1} + d` by solving the
   quadratic program

   .. math::
      :type: align

      \text{Minimize } & \frac{1}{2} d^T P_i d + q^T d \\
      \text{Subject to } & G_id \preceq h_i
      
   where :math:`\preceq` means that for each component the left hand
   side is less than or equal to the right hand side.
      
#. If not converged go back to step 1.

The assumption that the experiments are statistically independent
enables the calculations for each experiment :math:`k` in to be done
independently. In the next few sections, we describe both the data
from each experiment and the procedure for calculating :math:`P_i[k]`
and :math:`q_i[k]`.

The following sections describe the examples currently implemented in
F_UNCLE.  The components are the model parameters :math:`\theta`
which define an unknown EOS function and two experiments, namely a gun
and a rate stick.

Equation of State
=================
:label:`eos`

For the present work, we say that the thing we want to
estimate, :math:`\theta`, represents the equation of state (EOS)
of a gas.  We also say that the state of the gas in experiments
always lies on an isentrope and consequently the only relevant
data is the pressure as a function of specific volume (ml/gram)
of the gas.  For physical plausibility, we constrain the function to
have the following properties:

* Positive
* Monotonic
* Convex

Here, let us introduce the following notation:

* :math:`\vol` Specific volume
* :math:`p` Pressure
* :math:`\eos` An EOS that maps specific volume to pressure, :math:`\eos: \vol \mapsto \pressure`.
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
sets of functions that obeys those constraints and is further
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
   2.56\times10^9 \text{Pa} \text{ at one gram/cc}

and over a finite domain we approximate it by a cubic spline with
coefficients :math:`\left\{\tilde \cf[i] \right\}`.  Thus :math:`c`,
the vector of spline coefficients is the set of unknown parameters
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

Thus we have the following notation for splines and an a prior
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


where:

.. math::
   :type: align

   v_0 &= .4 \frac{\text{cm}^3}{\text{g}}   &  v_1 &= .5 \frac{\text{cm}^3}{\text{g}}  \\
   w_0 &= .1 \frac{\text{cm}^3}{\text{g}} &    w_1 &= .1 \frac{\text{cm}^3}{\text{g}}\\
   s_0 &= .25 &  s_1 &= -.3 


A Rate Stick
============

The data from this experiment represent a sequence of times that a
detonation shock is measured arriving at locations along a stick of HE
that is so thick that the detonation velocity is not reduced by
curvature.  The code for the pseudo data uses the average density and
sensor positions given by Pemberton et al.  [pemberton2011]_ for their
*Shot 1*.

Implementation
--------------

The only property of the HE that this ideal rate stick measures is the
detonation velocity.  Code in `F_UNCLE.Experiments.Stick` derives that
velocity following Section 2A of Fickett and Davis [ficket2000]_
(entitled *The Simplest Theory*).  At the Chapman Jouguet (CJ) state,
the following three curves are tangent in the :math:`p,v` plane:

* The Rayleigh line which gives a relation implied by conservation
  laws between pressure and density (or specific volume) before and
  after a shock.
* The Hugoniot curve, which is not used in this analysis.
* An isentrope.  erally one must use the Hugoniot to determine which
  isentrope goes through the CJ state, but it is assumed that each
  isentrope considered goes through the CJ state.

On page 17 of Fickett and Davis [ficket2000]_, Equation 2.3 expresses
the Rayleigh line as,

.. math::
   :label: eq-rayleigh
	   
   \rho_0^2 V^2 - (p-p_0)/(v_0-v) = 0,

where:

* :math:`\rho_0` is the initial density (before detonation wave arrives)
* :math:`v_0\equiv\frac{1}{\rho_0}` is the initial specific volume
* :math:`p_0` is the initial pressure
* :math:`V` is the velocity of the detonation wave
* :math:`p` is the pressure at positions behind the wave
* :math:`v` is the specific volume at positions behind the wave.

Rearranging the terms in :ref:`eq-rayleigh` yields this relation
between pressure and volume after the shock,

.. math::
   
   p = R(v,V) \equiv p_0 + \frac{V^2(v_0-v)}{v_0^2}.

The detonation velocity can be located by solving for the velocity
where Rayleigh line is tangent to the isentrope, known as the Chapman
Jouguet (CJ) point.

.. math::
   :type: align
	  
   F(v,V) &= \eos(v) - R(v,V)\\
   F'(v,V) &= \frac{d \eos}{d v} - \frac{V^2}{v_0^2},

At the CJ point:

.. math::	  
   :label: eq-fcond
	   
   F(v,V) = 0 

.. math::
   :label: eq-dfcond

   F'(v,V) = 0.

For a given value of :math:`V`, the `scipy.optimize.brentq` method is
used to solve :ref:`eq-dfcond` for :math:`v`.  Letting :math:`v(V)`
denote that solution, we write :ref:`eq-fcond` as,

.. math::
   :label: eq-fv

   F(v(V),V) = 0.

The code now solved for the root of :ref:`eq-fv` using
`scipy.optimize.brentq` to get :math:`V_{\text{CJ}}` and then assigns
:math:`v_{\text{CJ}} = v(V_{\text{CJ}})`. Figure :ref:`fig-cj-stick`
depicts three isentropes and the results of solving :ref:`eq-fv` for
the two curves labeled *experiment* and *fit*.

.. figure:: CJ_stick.pdf
   :align: center  
	   
   Isentropes, Rayleigh lines and CJ conditions. Starting from the
   isentrope labeled *nominal* and using data from a simulated
   experiment based on the isentrope labeled *experiment*, the
   optimization algorithm described in the Algorithm section produced
   the estimate labeled *fit*.  Solving Eqn. :ref:`eq-fv` for the
   *experiment* and *fit* isentropes yields the two Rayleigh lines
   that appear.  They are are nearly identical because the detonation
   velocities (and hence the experimental and fit data) are given by
   their slopes.  Outside of the CJ points where the Rayleigh lines
   are tangent to the isentropes, the data does not constrain the
   isentropes, and in fact they are quite
   different. :label:`fig-cj-stick`

Comparison to Pseudo Experimental Data
--------------------------------------

The previous simulation calculated the detonation velocity,
:math:`V_{\text{CJ}}(\eos)`, while experimental data were a series of
times when the shock reached a given position on the rate-stick. The
simulated detonation velocity could be related to these arrival times
using:

.. math::

   t[j] = \frac{x[j]}{V_{\text{CJ}}(\eos)}.

where :math:`x[j]` were the locations of each sensor measuring arrival time.

The sensitivity of the simulated response at the set of arrival times
to the spline coefficients governing the equation of state is given
by:

.. math::
   
  D[j,i] \equiv \frac{\partial t[j]}{\partial c[i]}

where the derivative was evaluated using finite differences.

The Gun
=======

The data from this experiment are a time series of measurements of a
projectile's velocity as it accelerates down a gun barrel driven by
the expanding products-of-combustion of HE.

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
  the specific volume of the HE products-of-combustion

The acceleration is computed based the projectile's mass and the force
resulting from the uniform pressure acting on the projectile. This
pressure is related to the projectile's position by the EOS, assuming
that the projectile perfectly seals the barrel so the mass of
products-of-combustion behind the projectile remains constant.

Comparison to Psudo Experimental Data
-------------------------------------

The experimental data were also the result of this simulation but
performed using the nominal *true* EOS described previously. These
experimental data were a series of times and corresponding
velocities. To compare the experiments to simulations, which may use a
different time discretization, the simulated response was represented
by a spline, and was compared to the experiments at each experimental
time stamp.

.. math::
   :label: gun_sens
	   
   D[j,i] = \partiald{\hat{v}(t_{exp}[j])}{\cf[i]}

where:

* :math:`\hat{v}` is the velocity given from the spline fit to simulated :math:`v(t)` data
* :math:`t_{exp}` is the times where experimental data were available

 
    
Numerical Results
=================

The algorithm was applied to the sets of simulation results and pseudo
experimental data for both the rate-stick and gun models. Figure
:ref:`fig-opt-stick` shows the improved agreement between the
simulated and *experimental* arrival times as the algorithm adjust the
equation of state. Similar results are shown in Figure
:ref:`fig-fve-gun` , where the significant error in velocity history
at early times is reduced by and order of magnitude as the optimized
EOS model approached the *true* EOS.

.. figure:: opt_stick.pdf
   :align: center   

   Fitting an isentrope to rate stick data.  In the upper plot, black
   +'s denote measured shock arrival time at 7 positions.  The blue
   line represents the shock velocity calculated from the nominal EOS,
   and the other lines come from the sequence of isentropes that the
   optimization algorithm described in the text generates as it seeks
   an isentrope that will produce a simulation that matches the data.
   That sequence of isentropes appears in the lower
   plot. :label:`fig-opt-stick`


.. figure:: fve_gun.pdf
   :align: center	   

   Sequential estimation of the maximum *a posteriori* probability
   parameters of :math:`f`.  The *true* EOS appears as *experimental*
   in the upper plot, and the optimization starts with the *nominal*
   and ends with *fit*.  The corresponding velocity for the gun as a
   function of position appears in the middle plot, and the sequence
   of errors in the forecast velocity time series after each step in
   the optimization appears in the lower plot. The estimation also
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
:math:`\Fisher(\hat c)` is :math:`10^{16}` larger than the next
largest, ie, the rank of :math:`\Fisher(\hat c)` is one to within
machine precision.

.. figure:: info_stick.pdf
   :align: center
   :class: w	   

   Fisher Information of the Rate Stick Experiment.  The sequence of
   log likelihoods produced by the optimization procedure appear in
   the upper left, and the corresponding isentropes appear in the
   upper right.  The largest three eigenvalues of :math:`\Fisher(\hat
   c)` appear in the lower left and the eigenfunction corresponding to
   the largest eigenvalue appears in the lower
   left. :label:`fig-info-stick`
   

The Fisher information matrix of the gun experiment is more complex as
changes to the EOS affect the entire time history of the projectile
velocity. In Figure :ref:`fig-info-gun` There is no clear *dominating*
eigenvalue, the largest eigenvalue corresponds to an eigenvector which
is more influential at larger projectile displacements while the next
three largest eigenvalues correspond to eigenvectors which are more
influential at the start of the experiment.

.. figure:: info_gun
   :align: center	    

   Fisher Information of the Gun Experiment.  The sequence of log
   likelihoods produced by the optimization procedure appear in the
   upper left, and the corresponding isentropes appear in the upper
   right.  The largest nine eigenvalues of :math:`\Fisher(\hat c)`
   appear in the lower left and the eigenfunctions corresponding to
   the largest four eigenvalues appear in the lower
   left. :label:`fig-info-gun`

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
characteristics.  The code [F_UNCLE]_ implements the procedure and we used
it to make the figures in the previous sections.  [F_UNCLE]_ runs on a
modest desktop computer and makes the figures in a few minutes.  That
speed and simplicity allows one to easily try out new ideas and code.
We have relied on [F_UNCLE]_ to guide work with real experimental data and
simulations on high performance computers that use proprietary
software.  Figure :ref:`fig-pbx` is the result of applying the ideas
presented here to the physical experiments described in
[pemberton2011]_.

.. figure:: fit_v.pdf
   :align: center
	
   Improvement of match between true experiments on PBX-9501 and
   simulations on a high performance computer.  The mean of the
   experimental data is labeled :math:`\mu`, and the optimization
   scheme yields the EOSs that produce the traces labeled
   :math:`fit_n`. :label:`fig-pbx`

[F_UNCLE]_ has been useful for us, and while we believe it could be useful
for others, we emphasize that it is a work in progress.  In
particular:

* The prior is inconsistent.  We hope to analyze and perhaps mitigate
  the effects of that inconsistency in future work.
* The choice of splines is not justified.  We plan to compare the
  performance of coordinate system options in terms of quantities such
  as bias and variance in future work.
* The optimization procedure is ad hoc.  We have already begun to
  consider other optimization algorithms.

	
	   

References
==========

.. [pemberton2011] Pemberton et al. "Test Report for Equation of State
                   Measurements of PBX-9501". LA-UR-11-04999, Los
                   Alamos National Laboratory, Los Alamos, NM.

.. [ficket2000] Ficket, W. and
                Davis, W. C., 2000. "Detonation". University of
                California Press: Berkeley, CA.

.. [F_UNCLE] "F_UNCLE: Functional Uncertainty Constrained by Law and
             Experiment" `https://github.com/fraserphysics/F_UNCLE
             <https://github.com/fraserphysics/F_UNCLE>`_ [Online;
             accessed 2016-05-27].

.. [Scipy] Jones, E., Oliphant, E., Peterson, P., et al. "SciPy\: Open
           Source Scientific Tools for Python", 2001-,
           `<http://www.scipy.org/>`_ [Online; accessed 2016-05-27].

.. [matplotlib] Hunter, J. D.. "Matplotlib\: A 2D Graphics
                Environment", Computing in Science & Engineering,
                **9**, 90-95 (2007), `DOI:10.1109/MCSE.2007.55
                <https://doi.org/10.1109/MCSE.2007.55>`_

.. [numpy] van der Walt, S. , Colbert, C. S.  and Varoquaux, G.. "The
           NumPy Array\: A Structure for Efficient Numerical
           Computation", Computing in Science \& Engineering, **13**,
           22-30 (2011), `DOI:10.1109/MCSE.2011.37
           <https://doi.org/10.1109/MCSE.2011.37>`_

.. [cvxopt] Andersen, M. and Vandenberghe, L.. "cvxopt\: Convex
            Optimization Package" `<http://cvxopt.org/>`_ [Online;
            accessed 2016-05-27].

.. [sphinx] "sphinx\: Python Documentation Generator"
            `<http://www.sphinx-doc.org/>`_ [Online; accessed
            2016-05-27].

.. [pylint] "pylint\: Python Code Static Checker"
            `<https://www.pylint.org/>`_ [Online; accessed
            2016-05-27].

.. [nose] "nose: Nose Extends Unittest to Make Testing Easier"
          `<https://pypi.python.org/pypi/nose/1.3.7>`_ [Online;
          accessed 2016-05-27].
   
       
       
	     
.. .. [hill1997] Hill, L. G., 1997. "Detonation Product Equation-of-State Directly From the Cylinder Test". Proc. 21st Int. Symp. on Shock Waves, Great Keppel Insland, Australia.

..
   Local Variables:
   mode: rst
   compile-command: "cd ../..; ./make_paper.sh papers/andrew_fraser"
   End:
