:author: Konstantinos Vamvourellis
:email: k.vamvourellis@lse.ac.uk
:institution: London School of Economics and Political Science
:corresponding:

:author: Marianne Corvellec
:email: marianne.corvellec@igdore.org
:institution: Institute for Globally Distributed Open Research and Education (IGDORE)

:bibliography: library

:video: https://www.youtube.com/watch?v=Gt73VNaZLXA

--------------------------------------------------
A Bayesian’s journey to a better research workflow
--------------------------------------------------

.. class:: abstract

   TODO Proofread abstract.

.. class:: keywords

   Bayesian statistics, life sciences, clinical trials, probabilistic programming, Stan, PyStan

Introduction
------------

TODO Proofread introduction.

A Bayesian Workflow
-------------------

We present a Bayesian workflow for statistical modeling. We recognize that each
project is unique and the research process can be too complex to summarize
in a recipe-style list. However, we believe that following a workflow, such as
suggested here, can help researchers, especially beginners. We have found that
setting up a workflow increases productivity. It also helps make research
projects more reproducible, as we discuss in the last section.

We propose a simple workflow made of the following steps:

1. Scope the problem;
2. Specify the likelihood and priors;
3. Generate fake data that resemble the true data to a reasonable degree;
4. Fit the model to the fake data;

   a. Check that the true values are recovered;
   b. Check the model fit;

5. Fit the model to the real data.

An advanced workflow, which is  beyond the scope of this paper, could be
extended to include the following steps:

6. Check the predictive accuracy of the model;
7. Select best model among different candidates (model selection);
8. Specify and integrate the utility function over the posterior distribution;
9. Perform a sensitivity analysis.

In what follows, we will use :math:`\mathcal{M}(\theta)` to denote the model as
a function of its parameter :math:`\theta` (:math:`\theta` is either a scalar
or a vector representing a set of parameters).
Data usually consist of observable outcomes [#]_ :math:`y`
and covariates [#]_ :math:`x`, if any. We will distinguish between the two when
necessary; otherwise, we will denote all data together by :math:`\mathcal{D}`.
We use :math:`p(\cdot)` to denote either probability distributions or probability
densities, even though it is not rigorous notation.

.. [#] Depending on their field, readers may want to think ‘dependent variables’ or ‘labels’.
.. [#] Depending on their field, readers may want to think ‘independent variables’ or ‘features’.

*1) Scope the problem*

The main goal of this workflow is to achieve successful Bayesian inference.
That is, correctly retrieving samples from the posterior distribution of the
parameter values, which are typically unknown before the analysis, using the
information contained in the data. The posterior distribution
:math:`p(\theta | \mathcal{D})` forms the basis of the Bayesian approach from
which we derive all quantities of interest.

Why do we need statistical inference in the first place? We need it to answer
our questions about the world. Usually, our questions refer to an implicit or
explicit parameter :math:`\theta` in a statistical model, such as:

* What values of :math:`\theta` are most consistent with the data?
* Do the data support a certain condition (e.g., for :math:`\theta` a scalar, :math:`\theta > 0`)?
* How can we predict the future outcome of an experiment?

To proceed, we need to define a model. Choosing a model is usually
tied to the exact research questions we are interested in. We can choose to
start with a postulated data generation process and then decide on how to interpret
the parameters in relation to the research question. Alternatively, it is
equally valid to start from the research question and design the model in a
way such that the model parameters are directly connected to the specific
questions we wish to answer. In the next section, we illustrate with an example
how to design a model to answer a specific research question.

Note that the question of prediction depends directly on inferring successfully
the parameter values. We shall come back to this at the end of this section.

*2) Specify the likelihood and priors*

Once we have defined the scope of the problem, we need to specify the design of
the model which is captured in the *likelihood* function
:math:`f(\mathcal{D} | \theta, \mathcal{M})`.
Usually, argument :math:`\mathcal{M}` is dropped for notational
simplicity, the model being chosen and assumed known. When the
model include covariates, the more accurate expression is
:math:`f(\mathcal{D} | \theta, x)`.
This function ties together the ingredients of
statistical inference and allows information to flow from the data
:math:`\mathcal{D}` to the parameters :math:`\theta`.

The second ingredient of Bayesian inference is the prior distribution
:math:`p(\theta)`. Priors are inescapably part of the Bayesian approach and, hence,
have to be considered carefully. The goal of Bayesian inference is to combine
the prior knowledge we have with the evidence contained in the data, to derive the
posterior distribution of :math:`\theta`. It is difficult to predict how sensitive the final
results will be to a change in the priors. However, it is important to note
that the impact of priors progressively diminishes as the number of observations
increases.

The ideal scenario for applying the Bayesian approach is when prior knowledge is
available, in which case the prior distribution can and should capture that
knowledge. But, sometimes, we might want to avoid expressing prior knowledge,
especially when such knowledge is not available. How are we supposed to
choose priors then? Constructing default priors is an active area of research
that is beyond the scope of this work. Here, we provide a high-level overview and refer
the interested reader to various sources for further reading.

Priors which express very little or no prior knowledge are called vague or
*uninformative priors*. Such priors are deliberately constructed in a way which
minimizes their impact on the resulting inference, relative to the information
brought in by the likelihood.  In fact, Bayesian inference technically works
even when the prior is not a proper distribution but a function that assumes all
values are equally likely, referred to as *improper prior*. However, it is
generally advisable to avoid improper priors, especially in settings beyond
just inference. If no prior knowledge is available, a very vague normal
distribution with large variance is still a better default prior than a uniform
distribution. It is important to note that improper or even vague priors are not
appropriate for model selection.

Additional considerations can impact the choice of priors,
especially when chosen together with the likelihood.
From a computational perspective, the most convenient priors are called
*conjugate priors*, because they mimic the structure of the likelihood function
and lead to a closed-form posterior distribution. Priors can have additional
benefits when used deliberately with a certain goal in mind. For example,
priors can be used to guard against overfitting by pulling the
parameters away from improbable values, or help with feature selection (e.g., see
horse-shoe priors).

Bayesian critics often see priors as a weakness, whereas in reality they are
an opportunity. Notably, priors give us the opportunity to employ our
knowledge to guide the inference in the absence of evidence from the data.
Also, it is important to remember that, in a scientific research context,
we rarely have absolutely no prior knowledge and
we typically do not consider any parameter value to be equally likely.

*3) Generate fake data*

Once we have agreed on a generative process, i.e., a model :math:`\mathcal{M}`,
we can use it to simulate data :math:`\mathcal{D'}`. We can choose reasonable
parameter values :math:`\theta_0` and use :math:`\mathcal{M}` to generate data
based on these values. Alternatively,
instead of coming up with reasonable parameter values, we can sample
these values from the prior distributions:

.. math::

   \theta_0 \sim p(\theta)

The fake data can then be interpreted as our prior distribution of the data.
Hence, by inspecting the fake data, we can reflect back on our choices for the
likelihood and priors. This inspection can be used for high-level
characteristics of the model, such as the scale of the outcome values or the
shape of the distributions. However, with this approach, we should make
sure that our priors are not uninformative, which would likely produce
unreasonable fake data.

Note how the model :math:`\mathcal{M}` is a hypothesized process and comes with
necessary assumptions and simplifications. It is highly unlikely that the real
world would follow exactly :math:`\mathcal{M}`. That being said, if
:math:`\mathcal{M}` is close enough to the real generative process, it can
still be very useful to help us understand something about the world. All
models are wrong, but some models are useful.

*4) Fit the model to the fake data*

If simulating data using our generative process :math:`\mathcal{M}` is the forward
direction, statistical inference is the reverse direction by which we find what
parameter values could have produced such data, under :math:`\mathcal{M}`.

The most popular statistical inference algorithm is maximum likelihood
estimation (MLE), which finds the parameter values that maximize the likelihood
given the observed data. Under the Bayesian approach, we treat the parameters
:math:`\theta` as random variables and express our prior knowledge about :math:`\theta` with
the prior probability distribution :math:`p(\theta)`. Bayesian inference is the process of
updating our beliefs about :math:`\theta` in light of the data :math:`\mathcal{D}`. The
updating process uses Bayes’ theorem and results in the conditional distribution :math:`p(\theta|
\mathcal{D})`, the posterior distribution. Bayesian inference is
generally a hard problem. In most cases, we cannot derive the mathematical form
of the posterior distribution; instead, we settle for an algorithm that returns
samples from the posterior distribution.

When we fit the model to fake data, we want to check two things, i.e., the correctness
of the inference algorithm and the quality of our model.

a. Much like in software testing, we want to check if the inference process
works by starting simple and advance progressively to the real challenge. By
fitting the model to fake data generated from the same model, we, effectively,
rule out issues of mismatch between our model and the real data. Testing the
inference algorithm under these ideal conditions allows us to perfect the
inference algorithm in a controlled environment, before trying it on the real data. In
our experience, this step brings to the surface many bugs in the code as well as
issues about the model in general.
It offers an added benefit, later on, when
we critique the fit of our model :math:`\mathcal{M}` to the real data
:math:`\mathcal{D}`. Having confidence in the correctness of our inference process
allows us to attribute any mismatch issues to the choice of the model.

By fitting the model to fake data, we recover samples from the posterior
distribution of the model parameters. There are various model fit tests to
choose from. At a minimum,
we need to check that the 95% posterior confidence intervals cover the true
parameter values :math:`\theta_0` that were used to generate the fake data. We should
tolerate a few misses, since 95% intervals will not cover the true values 5% of the
time, even if the algorithm is perfectly calibrated. Success at this stage is
not sufficient guarantee that the model will fit well to the real data, but it is
a necessary condition for proceeding further.

b. Equipped with a correct inferential algorithm for our model :math:`\mathcal{M}`,
it is time to critique it and ask if it is appropriate for the application. More
generally, this is a good time to check the model fit and decide if we need to
make any changes to the model. This step is usually specific to each
application. There is no limit as to how many tests we can do at this stage. It is
up to us to decide which tests are necessary to build confidence in
the model. If we choose a different model :math:`\mathcal{M'}`, we need to go
back to step 2 and start again. A more comprehensive evaluation of the model
should include checking the fit to the real data as well.

*5. Fit the model to the real data*

This is the time we have been waiting for. Once we have finalized the design of
our model and have tested it on fake data, we are ready to fit it to the real
data and get the results. Usually, we are interested in a specific quantity that
is derived from the posterior samples, as illustrated by our
case study in the next section.

At this point, we are expected to evaluate the model again if
necessary, depending on the application. For example, the model
may capture the average of the quantity but fail to capture the behavior at
the tails of the distribution.  This step is highly application-specific and
requires a combination of statistical expertise and subject-matter expertise. It is
important to build confidence in the power of our inference algorithm before we
proceed to interpreting the results, in order to be able to separate, to the extent
possible, inference issues from model issues. At this stage, it is likely that we
will come up with a slightly updated model :math:`\mathcal{M'}`. We then have to go
back and start again from the beginning.

*Posterior Predictive Checks and Model Evaluation*

One way to evaluate a model is to check how well it predicts unknown observable
data :math:`\tilde{y}`, where unknown means that the model was not fit
to :math:`\tilde{y}`. The Bayesian posterior predictive distribution is given
by the following formula:

.. math::
   :type: eqnarray

   p (\tilde{y} | \mathcal{D} ) &=& \int p( \tilde{y}, \theta | \mathcal{D}) d\theta \\
   &=& \int p( \tilde{y} |  \theta) p(\theta | \mathcal{D}) d\theta

In practice, we approximate the integral using samples from the posterior
distributions, by mapping each parameter posterior sample

.. math::

   \theta_0 \sim p(\theta|\mathcal{D})

to the corresponding sample of the posterior predictive distribution

.. math::

   \tilde{y} \sim p (\tilde{y} | \mathcal{D}).

Posterior predictive accuracy is useful even outside the strict scope of a
predictive task. Posterior predictive checks, evaluating the predictive accuracy
of a model, can be a good method to evaluate a model, especially in exploratory
analyses. A model that predicts well is a model that fits the data well. Model
evaluation is an extensive area of research with a rich literature, which is
beyond the scope of this contribution.

*Further reading*

For a concise overview of statistical modeling and inference, including a high-level
comparison with the frequentist approach, see :cite:`Wood15`. For a more
extended treatment of the Bayesian approach, including utility functions, see
:cite:`robert2007bayesian`. For an accessible Bayesian modeling primer,
especially for beginner Bayesians, see :cite:`McElreath15` and
:cite:`Marin2006`. For a complete treatment of Bayesian data analysis, including
many workflow-related discussions, see :cite:`gelman2013bayesian` [#]_.

.. [#] And for an example implementation of a complete workflow with PyStan,
       see https://github.com/betanalpha/jupyter_case_studies/tree/master/pystan_workflow.

A Case Study in Clinical Trial Data Analysis
--------------------------------------------

We propose a Bayesian model to extract insights from clinical trial datasets.
We are interested in understanding the effect of a treatment on the patients.
Our goal is to use the data to predict the effect of the treatment on a new
patient. We apply our method on artificially created data, for illustration
purposes only.

*1) Scope the problem*

Regulators focus on a few key effects when deciding whether a drug is fit for
market. In our case we will assume, for simplicity, that there are three
effects, where two are binary variables and the other is a continuous variable.

Our data is organized as a table, with one patient (subject) per row and one effect per column. For
example, if our clinical trial dataset records three effects per subject,
‘Hemoglobin Levels’ (continuous), ‘Nausea’ (yes/no), and ‘Dyspepsia’ (yes/no),
the dataset looks like Table :ref:`mtable`.

.. table:: Toy clinical trial data. :label:`mtable`

   +------------+------------+------------------+-----------+--------+
   | Subject ID | Group Type | Hemoglobin Level | Dyspepsia | Nausea |
   +============+============+==================+===========+========+
   | 123        | Control    | 3.42             | 1         | 0      |
   +------------+------------+------------------+-----------+--------+
   | 213        | Treatment  | 4.41             | 1         | 0      |
   +------------+------------+------------------+-----------+--------+
   | 431        | Control    | 1.12             | 0         | 0      |
   +------------+------------+------------------+-----------+--------+
   | 224        | Control    | -0.11            | 1         | 0      |
   +------------+------------+------------------+-----------+--------+
   | 224        | Treatment  | 2.42             | 1         | 1      |
   +------------+------------+------------------+-----------+--------+

The fact that the effects are of mixed data types, boolean and
continuous, makes it harder to model their interdependencies. To address this
challenge, we propose a latent variable  structure. Then, the expected value of
the latent variables will correspond to the average effect of the treatment.
Similarly, the correlations between the latent variables will correspond to the
the correlations between the effects. Knowing the distribution of the latent
variables will give us a way to predict what the effect will be on a new
patient.

*2) Specify the model, likelihood, and priors*

a. Model

Let :math:`Y` be a :math:`N\times K` matrix where each column represents an effect and each
row refers to an individual subject. This matrix contains our observations,
it is our clinical trial dataset. We distinguish between treatment and placebo
(control) subjects by considering separately :math:`Y^T` (resp. :math:`Y^{C}`),
the subset of :math:`Y` containing only treatment subjects (resp. control subjects).
Since the model for :math:`Y^T` and :math:`Y^{C}` is identical, for convenience,
we suppress the notation into :math:`Y` in the
remainder of this section. Recall that the important feature of
the data is that each column in :math:`Y` may be measured on different scales, i.e.,
binary, count, continuous, etc. The main purpose of this work is to extend the
current framework so that it can incorporate interdependencies between
different features, both discrete and continuous.

We consider the following general latent variable framework. We assume subjects
are independent and wish to model the dependencies between the effects.
The idea is to bring all columns to a common scale :math:`(-\infty, \infty)`.
The continuous effects are observed directly and are already on this scale.
For the binary effects, we apply appropriate transformations on their
parameters via user-specified link functions :math:`h_{j}(\cdot)`, in order to
bring them to the :math:`(-\infty, \infty)` scale.
Let us consider the :math:`i`-th subject. Then, if the :math:`j`-th effect is
measured on the binary scale, the model is

.. math::
   :type: eqnarray

   Y_{ij} &\sim& \text{Bernoulli}(\eta_j)\\
   h_{j}(\eta_j) &=& Z_{ij},

where the link function can be the logit, probit, or any other bijection from
:math:`[0, 1]` to the real line. Continuous data are assumed to be observed
directly and accurately (without measurement error), and modeled as follows:

.. math::

   Y_{ij} = Z_{ij} \quad \text{for}\; i=1, \dots, N.

In order to complete the model, we need to define the
:math:`N\times K` matrix :math:`Z`.
Here, we use a :math:`K`-variate normal distribution
:math:`\mathcal{N}_K(\cdot)` on each :math:`Z_{i \cdot}` row, such that

.. math::

   Z_{i\cdot} \sim \mathcal{N}_{K}(\mu, \Sigma),

where :math:`\Sigma` is a :math:`K\times K` covariance matrix, :math:`\mu` is a row
:math:`K`-dimensional vector, and :math:`Z_{i\cdot}` are independent for all :math:`i`.

In the model above, the vector :math:`\mu=(\mu_{1},\dots,\mu_K)` represents
the average treatment effect in the common scale. In our example, the first
effect is directly observed whereas the other effects can only be
inferred via the corresponding binary observations. Note that the variance of
the non-observed latent variables is non-identifiable :cite:`Chib1998a,Talhouk2012a`,
so we need to fix it to a known constant to fully specify
the model. We do this by decomposing the covariance into correlation and
variance: :math:`\Sigma = DRD`, where :math:`R` is the correlation matrix and :math:`D` is a
diagonal matrix of variances :math:`D_{jj} = \sigma_j^2` for the :math:`j`-th effect.

b. Likelihood

The likelihood function can be expressed as

.. math::
   :type: eqnarray

   f(Y | Z, \mu, \Sigma) &=& f(Y|Z) \cdot p(Z| \mu, \Sigma)\\
   &=& \prod_{j \in J_b} \prod_{i=1}^N h^{-1}(Z_{ij})^{Y_{ij}} (1-h^{-1}(Z_{ij}))^{(1-Y_{ij})} \cdot p(Z| \mu, \Sigma)\\
   &=& \prod_{j \in J_b} \prod_{i=1}^N \eta_{ij}^{Y_{ij}} (1-\eta_{ij})^{(1-Y_{ij})} \cdot N(Z| \mu , \Sigma),\\

where :math:`J_b` is the index of effects that are binary and
:math:`N(Z| \mu , \Sigma)` is the probability density function (pdf)
of the multivariate normal distribution.

c. Priors

In this case study, the priors should come from previous studies of the treatment
in question or from clinical judgment. If there was no such option,
then it would be up to us to decide on an appropriate prior. We use
the following priors for demonstration purposes:

.. math::
   :type: eqnarray

   \mu_i \; & \sim \; N(0,10) \\
   R \; & \sim \; \text{LKJ}(2) \\
   \sigma_j \; & \sim \; \text{Cauchy}(0,2)  \; \text{for} \; j \not\in J_b \\
   Z_{ij} \; & \sim \; N(0,1) \; \text{for} \; j \in J_b. \\

This will become more transparent in the next section, when we come back to
the choice of priors.
Let us note that our data contain a lot of information, so the final outcome
will be relatively insensitive to the priors.

*3) Generate fake data*

To generate fake data, we choose reasonable parameter values :math:`(\mu, \Sigma)`
and generate 200 samples of underlying latent variables
:math:`Z_{i \cdot} \sim N(\mu,\Sigma)`.
The observed fake data :math:`Y_{ij}` are defined to be equal to
:math:`Z_{ij}` for the effects that are continuous. For the binary effects, we sample
Bernoulli variables with probability equal to the inverse logit of the
corresponding :math:`Z_{ij}` value.

A Bayesian model with proper informative priors, such as the one above, can also
be used directly to sample fake data. As explained in the previous section,
we can sample all the parameters according to the prior distributions.
The fake data can then be interpreted as our prior distribution on the data.

References
----------

