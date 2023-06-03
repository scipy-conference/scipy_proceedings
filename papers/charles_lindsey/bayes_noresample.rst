:author: Charles Lindsey
:email: charles.lindsey@revionics.com
:institution: Revionics, an Aptos Company
:bibliography: mybib

--------------------------------------------------------
Bayesian Statistics with Python, No Resampling Necessary
--------------------------------------------------------

.. class:: abstract

TensorFlow Probability is a powerful library for statistical analysis in Python. Using TensorFlow Probability’s implementation of Bayesian methods, modelers can incorporate prior information and obtain parameter estimates and a quantified degree of belief in the results. Resampling methods like Markov Chain Monte Carlo can also be used to perform Bayesian analysis. As an alternative, we show how to use numerical optimization to estimate model parameters, and then show how numerical differentiation can be used to get a quantified degree of belief. How to perform simulation in Python to corroborate our results is also demonstrated.

.. class:: keywords

   Bayesian statistics, resampling, maximum likelihood, numerical differentiation 

Introduction
------------

Some machine learning algorithms output only a single number or decision.  It can be useful to have a measure of confidence in the output of the algorithm, a quantified degree of belief.  Bayesian statistical methods can be used to provide both estimates and confidence for users.

A model with parameters :math:`\boldsymbol{\theta}` governs the process we are investigating.  We begin with a prior belief about the probability distribution of :math:`\boldsymbol{\theta}`, the density :math:`\pi(\boldsymbol{\theta})`.

Then the data we observed gives us a refined belief about the distribution :math:`\boldsymbol{\theta}`. We obtain the posterior density :math:`\displaystyle \pi(\boldsymbol{\theta}\vert {\bf x})`.

We can estimate values of :math:`\boldsymbol{\theta}` with the posterior mode of :math:`\displaystyle \pi(\boldsymbol{\theta}\vert {\bf x})`, :math:`\widehat{\boldsymbol \theta}`.

Then we can estimate the posterior variance of :math:`\boldsymbol{\theta}`, and with some knowledge of :math:`\displaystyle \pi(\boldsymbol{\theta}\vert {\bf x})` obtain confidence in our estimate :math:`\widehat{\boldsymbol \theta}`.


Normal Approximation to the Posterior
-------------------------------------

We will use numerical optimization to obtain the posterior mode :math:`\widehat{\boldsymbol \theta}`, maximizing the posterior :math:`\displaystyle \pi(\boldsymbol{\theta}\vert {\bf x})`.

The posterior is proportional (where the scaling does not depend on :math:`\boldsymbol{\theta}`) to the prior and likelihood (or density of the data).

.. math::

   \pi(\boldsymbol{\theta}\vert {\bf x}) \propto L(\boldsymbol{\theta}\vert {\bf x}) \pi(\boldsymbol{\theta})

As in maximum likelihood, we directly maximize the log-posterior, :math:`\log \pi(\boldsymbol{\theta}\vert {\bf x})` because it is more numerically stable.

Now, as described in section 4.1 of :cite:`gelman2013bayesian` , we can approximate :math:`\ln \pi(\boldsymbol{\theta}\vert {\bf x})` using a second order Taylor Expansion around :math:`\widehat{\boldsymbol \theta}`.

.. math::
   :type: eqnarray

   \log \pi(\boldsymbol{\theta}\vert {\bf x}) &\approx& \log \pi(\widehat{\boldsymbol \theta}\vert{\bf x}) + (\boldsymbol{\theta} - \widehat{\boldsymbol \theta} )^TS({\boldsymbol \theta})\vert_{{\boldsymbol \theta}={\widehat{\boldsymbol \theta}}} \\
        & &  + \frac{1}{2}(\boldsymbol{\theta} - \widehat{\boldsymbol \theta} )^T H(\widehat{\boldsymbol \theta}) (\boldsymbol{\theta} - \widehat{\boldsymbol \theta} )

Where :math:`S(\boldsymbol{\theta})` is the score function

.. math::

   S(\boldsymbol{\theta}) = \frac{\delta}{\delta \boldsymbol{\theta}} \log \pi(\boldsymbol{\theta}\vert {\bf x})

and :math:`H(\boldsymbol{\theta})` is the Hessian function.

.. math::

   H(\boldsymbol{\theta}) = \frac{\delta}{\delta \boldsymbol{\theta}^T} S(\boldsymbol{\theta})

We assume that :math:`\widehat{\boldsymbol \theta}` is in the interior of the parameter space (or support) of :math:`\boldsymbol{\theta}`.  Also, :math:`\pi(\boldsymbol{\theta}\vert {\bf x})` is a continuous function of :math:`\boldsymbol{\theta}`.

Finally the Hessian matrix, :math:`H(\boldsymbol{\theta})` is negative definite, so :math:`-H(\boldsymbol{\theta})` is positive definite. This means that we can invert :math:`-H(\boldsymbol{\theta})` and get a matrix that is a valid covariance.

With these assumptions, as the sample size :math:`n\to\infty` the quadratic approximation for :math:`\log \pi(\boldsymbol{\theta}\vert {\bf x})` becomes more accurate. At the posterior mode :math:`{\boldsymbol \theta}={\widehat{\boldsymbol \theta}}`, :math:`\log \pi(\boldsymbol{\theta}\vert {\bf x})` is maximized and :math:`0=S({\boldsymbol \theta})\vert_{{\boldsymbol \theta}={\widehat{\boldsymbol \theta}}}`.

Given this, we can exponentiate the approximation to get

.. math::

   \pi(\boldsymbol{\theta}\vert {\bf x}) \approx \pi(\widehat{\boldsymbol \theta}\vert{\bf x}) \exp(\frac{1}{2} (\boldsymbol{\theta} - \widehat{\boldsymbol \theta} )^T H(\widehat{\boldsymbol \theta}) (\boldsymbol{\theta} - \widehat{\boldsymbol \theta} ))

So for large :math:`n`, the posterior distribution of :math:`{\boldsymbol \theta}` is approximately proportional to a multivariate normal density with mean :math:`\widehat{\boldsymbol{\theta}}` and covariance :math:`-H(\widehat{\boldsymbol{\theta}})^{-1}`.

.. math::

   {\boldsymbol \theta} \vert x \approx_D N(\widehat{\boldsymbol{\theta}}, -H(\widehat{\boldsymbol{\theta}})^{-1})

Another caveat for this result is that the prior should be proper, or at least lead to a proper posterior.  Our asymptotic results are depending on probabilities integrating to 1.

We could get a quantified degree of beief by using resampling methods like Markov chain Monte Carlo (MCMC) :cite:`gelman2013bayesian` directly.  We would have to use fewer assumptions. However, resampling can be computationally intensive.

Parameter Constraints and Transformations
-----------------------------------------

Optimization can be easier if the parameters are defined over the entire real line.  Parameters that do not follow this rule are plentiful. Variances are only positive. Probabilities are in [0,1].

We can perform optimization over the real line by creating unconstrained parameters from the original parameters of interest.  These are continuous functions of the constrained parameters, which may be defined on intervals of the real line.

For example, the unconstrained version of a standard deviation parameter :math:`\sigma` is :math:`\psi= \log \sigma`. The parameter :math:`\psi` is defined over the entire real line.

It will be useful for us to consider the constrained parameters as being functions of the unconstrained parameters.  So :math:`\sigma=exp(\psi)` is our constrained parameter of :math:`\psi`.

So the posterior mode of the constrained parameters :math:`{\boldsymbol{\theta_c}}` is :math:`\widehat{\boldsymbol \theta}_{\boldsymbol c} = g(\widehat{\boldsymbol \theta})`.  We will call :math:`g` the **constraint** function.

Then we can use the delta method :cite:`oehlert1992delta` on :math:`g` to get the posterior distribution of the constrained parameters.

A first-order Taylor approximation of :math:`g({\boldsymbol \theta})` at :math:`\widehat{\boldsymbol \theta}` yields

.. math::

   g({\boldsymbol \theta}) \approx g(\widehat{\boldsymbol \theta}) + \left\{\frac{\delta}{\delta \widehat{\boldsymbol{\theta}}} g(\widehat{\boldsymbol{\theta}})\right\} ({\boldsymbol \theta} - \widehat{\boldsymbol{\theta}})

Remembering that the posterior of :math:`\boldsymbol \theta` is approximately normal, the rules about linear transformations for multivariate normal random vectors tell us that

.. math::
   :type: eqnarray
   
      & & {\boldsymbol{\theta_c}}\vert x = g({\boldsymbol \theta}) \vert x \approx_D  \\
      & &  N \left\lbrack g(\widehat{\boldsymbol{\theta}}), \left\{\frac{\delta}{\delta \widehat{\boldsymbol{\theta}}} g(\widehat{\boldsymbol{\theta}})\right\}^T \left\{-H(\widehat{\boldsymbol{\theta}})^{-1}\right\} \left\{\frac{\delta}{\delta \widehat{\boldsymbol{\theta}}} g(\widehat{\boldsymbol{\theta}})\right\}\right\rbrack

This involved a first-order approximation of :math:`g`.  Earlier we used a second order approximation for taking the numeric derivative. Why would we just do a first-order here?  Traditionally the delta-method is taught and used as only a first-order method.  Usually the functions used in the delta method are not incredibly complex. It is *good enough* to to use the first-order approximation.

Hessian and Delta Approximation
-------------------------------

To be able to use the normal approximation, we need :math:`\widehat{\boldsymbol{\theta}}`, :math:`H(\widehat{\boldsymbol{\theta}})^{-1}`, and :math:`\frac{\delta}{\delta \widehat{\boldsymbol{\theta}}} g(\widehat{\boldsymbol{\theta}})`.  As mentioned before, we use numerical optimization to get :math:`\widehat{\boldsymbol{\theta}}`. Ideally, we would have analytic expressions for :math:`H` and the derivatives of :math:`g`.

This can be accomplished with automatic differentiation :cite:`baydin2018automatic`, which will calculate the derivatives analytically. We can also perform numerical differentiation to get the Hessian and the gradient of the constraint function :math:`g`. This will be less accurate than an analytic expression, but may be less computationally intensive in large models.

But once you learn how to take one numeric derivative, you can take the numeric derivative of anything. So using numerical differentiation is a very flexible technique that we can easily apply to all the models we would use.

Numerical Differentiation
-------------------------

So numeric derivatives can be very pragmatic, and flexible.  How do you compute them? Are they accurate? We use section 5.7 of :cite:`press2007numerical` as a guide.

The derivative of the function :math:`f` with respect to :math:`x` is

.. math::

    f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}

To approximate :math:`f'(x)` numerically, couldn't we just plugin a small value for :math:`h` and compute the scaled difference?   Yes. And that is basically what happens.  We do do a little more work to choose :math:`h` and use a second-order approximation instead of a first-order.

We can see that the scaled difference is a first-order approximation by looking at the Taylor series expansion around :math:`x`.
   
Taylor's theorem with remainder gives

.. math::
   :type: eqnarray

   f(x+h) &=& f(x) + ((x+h)-x)f'(x) + .5((x+h)-x)^2 f''(\epsilon) \\
    &=& f(x) + -h f'(x) + .5 h^2 f''(\epsilon) \\


where :math:`\epsilon` is between :math:`x` and :math:`x+h`.

Now we can rearrange to get

.. math::

    \frac{f(x+h)-f(x)}{h} - f'(x) = .5 h f''(\epsilon)
    
The right hand side is the truncation error, :math:`\epsilon_t` since it's linear in :math:`h`, the bandwidth we call the this approximation a first order method.

We can do second-order approximations for :math:`f(x+h)` and :math:`f(x-h)` and get a more accurate second order method of approximation for :math:`f'(x)`.

.. math::
   :type: eqnarray

    f(x+h) &=& f(x) + ((x+h)-x)f'(x) \\
            & & + \frac{((x+h)-x)^2 f''(x)}{2!} + \frac{((x+h)-x)^3f'''(\epsilon_1)}{3!} \\
    f(x-h) &=& f(x) + ((x-h)-x)f'(x)  \\
            & & + \frac{((x-h)-x)^2 f''(x)}{2!} + \frac{((x-h)-x)^3f'''(\epsilon_2)}{3!}

were :math:`\epsilon_1` is between :math:`x` and :math:`x+h` and   :math:`\epsilon_2` is between :math:`x-h` and :math:`x`.

Then we have

.. math::

    \frac{f(x+h) - f(x-h)}{2h} - f'(x) = h^2 \frac{f'''(\epsilon_1)+ f'''(\epsilon_2)}{12}

This is quadratic in :math:`h`.  The first term takes equal input from both sides of :math:`x`, so we call it a centered derivative.

So we choose a small value of :math:`h` and plug it into :math:`\frac{f(x+h) - f(x-h)}{2h}` to approximate :math:`f'(x)`.

Our derivation used a single input function :math:`f`. The idea applies to partial derivatives of multi-input functions as well. The inputs that you aren't taking the derivative with respect to are treated as fixed parts of the function.

Choosing a Bandwidth
--------------------

In practice, second order approximation actually involves two sources of error.  Roundoff error, :math:`\epsilon_r` arises from being unable to represent :math:`x` and :math:`h` or functions of them with exact binary represetation.

.. math::

   \epsilon_r \approx \epsilon_f\frac{\mid{f(x)}\mid}{h}

where :math:`\epsilon_f` is the fractional accuracy with which :math:`f` is computed. This is generally machine accuracy.  If we are using NumPy :cite:`numpy` this would be

.. math::
    
    \epsilon_f = \mbox{np.finfo(float).eps}
   
Minimizing the roundoff error and truncation error, we obtain
   
.. math::

   h \sim \epsilon_f^{1/3} \left(\frac{f}{f'''}\right)^{1/3}

where :math:`\left(f / f'''\right)^{1/3}` is shorthand for the ratio of :math:`f(x)` and the sum of :math:`f'''(\epsilon_1)+ f'''(\epsilon_2)`.

We use shorthand here because because we are not going to approximate :math:`f'''` (we are already approximating :math:`f'`), so there is no point in writing it out.

Call this shorthand

.. math::
    \left(\frac{f}{f'''}\right)^{1/3}=x_c

the curvature scale, or characteristic scale of the function :math:`f`.

There are several algorithms for choosing an optimal scale.  The better the scale chosen, the more accurate the approximation is.  A good rule of thumb, which is computationally quick, is to just use the absolute value of :math:`x`.

.. math::
   x_c = \mid{x}\mid

Then we would use

.. math::
    h = \epsilon_f^{1/3} \mid{x}\mid

But what if :math:`x` is 0?  This is simple to handle, we just add :math:`\epsilon_f^{1/3}` to :math:`x_c = \mid x \mid`

.. math::
   h = \epsilon_f^{1/3} ( \mid{x}\mid + \epsilon_f^{1/3})

Now, Press et al. also suggest performing a final sequence of assignment operations that ensures :math:`x` and :math:`x+h` differ by an exactly representable number. You assign :math:`x+h` to a temporary variable :math:`temp`. Then :math:`h` is assigned the value of :math:`temp-h`.

In Python, the code would look like

.. code-block:: python

    temp = x + h
    h = temp - x


Estimating Confidence Intervals after Optimization
--------------------------------------------------


With the posterior mode, variance, and normal approximation to the posterior. It is simple to create confidence (credible) intervals for the parameters.

Let's talk a little bit about what these intervals are.  For the parameter :math:`\gamma` we want a :math:`(1-\alpha)` interval :math:`(u,l)` (defined on the observed data generated by a realization of :math:`\gamma`) to be defined such that

.. math::

   \mbox{Pr}(\gamma \in (u,l)) = 1-\alpha

The frequentist confidence interval does not meet this criteria.   :math:`\gamma` is just one fixed value, so it is either in the interval, or it isn't!  The probability is 0 or 1.  A credible interval (Bayesian confidence interval) can meet this criteria.

Suppose that we are able to use the normal approximation for :math:`\gamma \vert \bf{x}`

.. math::
    \gamma\vert{\bf{x}} \approx_D N(\hat\gamma,\hat\sigma_\gamma^2)


Then we have

.. math::
    :type: eqnarray
    
    1-\alpha &=& \mbox{Pr}(l \leq \gamma \leq u \vert{\bf{x}} ) \\
             &=& \mbox{Pr}(l - \hat{\gamma} \leq  \gamma - \hat{\gamma} \leq u - \hat{\gamma}\vert{\bf{x}} ) \\
             &=& \mbox{Pr}\left(\frac{l - \hat{\gamma}}{{\hat\sigma_\gamma}} \leq  \frac{\gamma - \hat{\gamma}}{{\hat\sigma_\gamma}} \leq \frac{u - \hat{\gamma}}{{\hat\sigma_\gamma}}\vert{\bf{x}} \right)
    
Now, :math:`(\gamma - \hat{\gamma}) / \hat\sigma_\gamma^2` is :math:`N(0,1)`, standard normal. So we can use the standard normal quantiles in solving for :math:`l` and :math:`u`.

The upper :math:`\alpha/ 2` quantile of the standard normal distribution, :math:`z_{\alpha/ 2}` satisfies

.. math::
    \mbox{Pr}(Z \geq z_{\alpha/ 2}) = \alpha / 2

for standard normal :math:`Z`.

Noting that the standard normal is symmetric, if we can find :math:`l` and :math:`u` to satisfy

.. math::
   :type: eqnarray
   
   \frac{l - \hat{\gamma}}{\hat\sigma_\gamma} &=& - z_{\alpha/ 2} \\
    \frac{u - \hat{\gamma}}{\hat\sigma_\gamma} &=& z_{\alpha/ 2}

then we have a valid Bayesian confidence interval.

Simple calculation shows that the solutions are

.. math::
    :type: eqnarray
    
    l &=& -z_{\alpha/2}\hat\sigma_\gamma + \hat{\gamma} \\
    u &=& z_{\alpha/2}\hat\sigma_\gamma + \hat{\gamma}

The :math:`z_{\alpha/ 2}` quantile can be easily generated using **scipy.stats** from SciPy :cite:`scipy`.  We can also adjust the intervals for inference on many parameters by using Bonferroni correction :cite:`bonferroni1936teoria`.

Now we know how to estimate the posterior mode. We also know how to estimate the posterior variance after computing the posterior mode. And we have seen how confidence intervals are made based on this posterior variance, mode, and the normal approximation to the posterior.  Let's discuss some tools that will enable us to perform these operations.

TensorFlow Probability
----------------------

Now we will introduce TensorFlow Probability, a Python library that we can use to perform the methods we have been discussing.  TensorFlow Probability is library built using TensorFlow, a leading software library for machine learning and artificial intelligence :cite:`tensorflow2015-whitepaper`.

TensorFlow Probability is a probabilistic programming language.  This lets us build powerful models in a modular way and estimate them automatically.  At the heart of TensorFlow Probability is the **Distribution** class.  In theory, a probability distribution is the set of rules that govern the likelihood of how a random variable (vector, or even general tensor) takes its values.

In TensorFlow Probability, distribution rules for scalars and vectors are parametrized, and these are expanded for higher dimensions as independent samples.  A distribution object corresponds to a random variable or vector.  The parts of a Bayesian model can be represented using different distribution objects for the parameters and observed data.

Example Distribution
--------------------
As an example, let's examine a linear regression with a :math:`\chi^2` prior for the intercept a and a normal prior for the slope :math:`\beta`. Our observed outcome variable is :math:`y` with a normal distribution and the predictor is :math:`x`.

.. math::
    y_i \sim \mbox{Normal}(x_i\beta + \alpha, 1)

We can store the distribution objects in a dictionary for clear organization. The prior distribution of :math:`\beta` is Normal with mean 1 and variance 1, :math:`N(1,1)`. We use the **Normal** distribution subclass to encode its information in our dictionary.

.. code-block:: python

    tfd = tfp.distributions
    dist_dict = {}
    dist_dict['beta'] = tfd.Normal(1,1)

The :math:`\beta` parameter can range over the real line, but the intercept, :math:`\alpha` should be nonnegative. The **Chi2** distribution sublcass has support on only the nonegative reals.  However, if we are performing optimization on the :math:`\alpha` parameter, we may take a step where it became negative. We can avoid any complications like this if we use a **TransformedDistribution**. Transformed distributions can be used together with a **Bijector** object that represents the transforming function.

For :math:`\alpha`, we will model an unconstrained parameter, :math:`\alpha^u = \log \alpha`. The natural logarithm can take values over the real line.

.. code:: python
    
    tfb = tfp.bijectors
    dist_dict['unconstrained_alpha'] = \
    tfd.TransformedDistribution(tfd.Chi2(4),tfb.Log())
    
We can use the **sample** method on the distribution objects we created to see random realizations. Before we do that we should setthe seed, so that we can replicate our work.

.. code:: python
    
    tf.random.set_seed(132)
    sample_ex=dist_dict['unconstrained_alpha'].sample(10)
    sample_ex
    
.. container:: output execute_result

      ::

         <tf.Tensor: shape=(10,), dtype=float32, numpy=
         array([ 2.050956  , 0.56120026,  1.8559402,
                -0.05669071, ... ], dtype=float32)>

We see that the results are stored in a **tf.Tensor** object.  This has an easy interface with NumPy, as you can see by the **numpy** component. We see that the unconstrained :math:`\alpha`, :math:`\alpha^u` takes positive and negative values.

We can evaluate the density, or it's natural logarithm using class methods as well. Here is the log density for the sample we just drew.

.. code:: python

    dist_dict['unconstrained_alpha'].log_prob(sample_ex)

.. container:: output execute_result

    ::

         <tf.Tensor: shape=(10,), dtype=float32, numpy=
         array([-1.1720479 , -1.1402813 , -0.8732692 ,
                -1.9721189 , ...], dtype=float32)>

Now we can get :math:`\alpha` from :math:`\alpha^u` by using a callable and the **Deterministic** distribution.

.. code-block:: python

    dist_dict['alpha'] = \
        lambda unconstrained_alpha: \
            tfd.Deterministic(\
                loc= tfb.Log().inverse( \
                    unconstrained_alpha))
      
Now we've added all of the parameters to **dist_dict**.  We just need to handle the observed variables :math:`y` and :math:`x`.  In this example :math:`x` is **exogenous**, which means it can be treated as fixed and nonrandom in estimating :math:`\alpha` and :math:`\beta` in the model for :math:`y`. :math:`y` is **endogenous**, which means it is a response variable in the model, the outcome we are trying to estimate.

We will define :math:`x` separately from our dictionary of distributions. For the example we have to generate values of :math:`x`, but once this is done we will treat it as fixed and exogenous

The observed variable :math:`x` will have a standard normal distribution. We will start by giving it a sample size of 100.

.. code-block:: python

      n = 100
      x_dist = tfd.Normal(tf.zeros(n),1)
      x = x_dist.sample()

The distribution of :math:`y`, which would give us the likelihood, can be formulated using a callable function of the parameters and the fixed value of :math:`x` we just obtained.

.. code-block:: python

    dist_dict['y'] = \
        lambda alpha, beta: \
            tfd.Normal(loc = alpha + beta*x,scale=1)

With a dictionary of distributions and callables indicating their dependencies, we can work with the joint density. This will correspond to the posterior distribution of the model, augmenting the priors with the likelihood.

The **JointDistributionNamed** class takes a dictionary as input and behaves similarly to a regular univariate distribution object.  We can take samples, which are returned as dictionaries keyed by the parameter and observed variable names.  We can also compute log probabilities, which gives us the posterior density.

.. code-block:: python

    posterior = tfd.JointDistributionNamed(dist_dict)

Now we have a feel for how TensorFlow Probability can be used to store a Bayesian model.  We have what we need to start performing optimization and variance estimation.

Maximum A Posteriori (MAP) with SciPy
-------------------------------------
   
We can use SciPy's implementation of the Limited memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) :cite:`Flet87` algorithm to estimate the posterior mode. This is a Quasi-Newton optimization method that does not need to store the entire Hessian matrix during the algorithm, so it can be very fast. If the Hessian was fully stored we could just use it directly in variance estimation, but it would be slower.  We do to take advantage of automatic differentiation to calculate the score function, the first derivative of the posterior.  TensorFlow Probability provides this through the **value_and_gradient** function of its **math** library.

We will use **minimize** from the **optimize** SciPy library, which operates on a loss function that takes a vector of parameters as input.  We will optimize on **unconstrained_alpha** and **beta**, the unconstrained space parameters of the model.  In the joint distribution representation, they are separate tensors.  But in optimization, we will need to evaluate a single tensor.

We will use the first utility function from the **bayes_mapvar** library, which will be available with this paper, to accomplish this.  The **par_vec_from_dict** function unpacks the tensors in a dictionary into a vector.

Within our loss function, we must move the vector of input parameters back to a dictionary of tensors to be evaluated by TensorFlow probability.  The **par_dict_from_vec** function moves the unconstrained parameters back into a dictionary, and the constrained parameters are generated by the **get_constrained** function.  Then the posterior density is evaluated by augmenting this dictionary of constrained parameters with the observed endogenoous variables. The get_constrained function is also used to get the final posterior model estimates from the SciPy optimization.

Variance Estimation with SciPy
------------------------------

Once the posterior mode is estimated we can estimate the variance.  The first step
is calculating the bandwidths.  The **get_bandwidths** function handles this.

.. code-block:: python

    def get_bandwidths(unconstrained_par_vec):
        abspars = abs(unconstrained_par_vec)
        epsdouble = np.finfo(float).eps
        epsdouble = epsdouble**(1 / 3)
        scale = epsdouble * (abspars + epsdouble)
        scaleparmstable = scale + abspars
        return scaleparmstable - abspars

With the bandwidths calculated, we step through the parameters and create the Hessian and Delta matrices that we need for variance estimation.  The **get_hessian_delta_variance** function use numeric differentation to calculate the Hessian, based on numeric derivaties of the automatic derivatives computed by TensorFlow probability for the score function.  The Delta matrix is calculated using numeric differentation of the constrained parameter functions.

Simulation
----------

We evaluated our methodology with a simulation based on the :math:`\alpha` and :math:`\beta` parameter setting discussed earlier.  This was an investigation into how well we estimated the posterior mode, variance, and distribution using the methods of TensorFlow Probability, SciPy, and bayes_mapvar.

To evaluate the posterior distributions of the parameters we used the MCMC capabilities of TensorFlow Probability.  Particulary the the No-U-Turn Sampler :cite:`hoffman2011nouturn`.  We were careful to thin the sample based on effective sample size so that autocorrelation would not be a problem.  This was accomplished using TensorFlow Probability's  **effective_sample_size** function from its **mcmc** library.

We drew :math:`n_{pre} = 1000` observations from the unconstrained prior parameter distribution for :math:`\alpha_i` and :math:`\beta_i`.  For each of these prior draws, we drew a posterior sample of :math:`{\bf y}_i` and :math:`{\bf x}_i`. :math:`{\bf y}_i` and :math:`{\bf x}_i` were :math:`n_{post} = 600` samples based on each `\alpha_i` and :math:`\beta_i`. The posterior mode and variance were estimated, and :math:`n_{MCMC}=500` posterior draws from MCMC were made.  The mean was used in the MCMC draws since it sould coincide with the mode if our assumptions are correct.
    
To check the distributional results, we used the  Anderson-Darling test :cite:`STEP1974`.  This is given by **anderson** in **scipy.stats**.  We stored a record of whether the test rejects normality at the .05 significance level for each of the :math:`n_{pre}` draws.  This test actually checks the mean and variance assumptions as well, since it compares to a standard normal and we are standardizing based on the MAP and **get_hessian_delta_variance** estimates.

.. table:: Simulation Results, :math:`n_{pre} = 1000`, :math:`n_{post} = 600`, :math:`n_{MCMC}=500`. :label:`mtable`

   +------------------------------+-------+-------+
   | Statistic                    | Mean  | S.D.  |
   +==============================+=======+=======+
   | :math:`\alpha_{MAP}` mean    | 4.141 | 2.475 |
   +------------------------------+-------+-------+
   | :math:`\alpha_{MCMC}` mean   | 3.989 | 2.765 |
   +------------------------------+-------+-------+
   | :math:`\alpha_{MAP}` S.E.    | 0.037 | 0.004 |
   +------------------------------+-------+-------+
   | :math:`\alpha_{MCMC}`  S.E.  | 0.041 | 0.001 |
   +------------------------------+-------+-------+
   | :math:`\alpha` A.D. Reject   | 0.042 | 0.201 |
   +------------------------------+-------+-------+
   | :math:`\beta_{MAP}` mode     | 1.013 | 0.504 |
   +------------------------------+-------+-------+
   | :math:`\beta_{MCMC}` mean    | 1.022 | 1.003 |
   +------------------------------+-------+-------+
   | :math:`\beta_{MAP}` S.E.     | 0.029 | 0.001 |
   +------------------------------+-------+-------+
   | :math:`\beta_{MCMC}` S.E.    | 0.041 | 0.002 |
   +------------------------------+-------+-------+
   | :math:`\beta` A.D. Reject    | 0.045 | 0.207 |
   +------------------------------+-------+-------+


The results of the simulation are shown in :ref:`mtable`.We use Standard Error (S.E.) to refer to the 1000 estimates of posterior standard deviations from **get_hessian_delta_variance** and the MCMC sample standard deviations.  The Standard Deviation (S.D.) column represents the statistics calculated over the 1000 estimates.  The standard errors are not far from each other, and neither are the modes and means.  The rejection rates for the Anderson Darling test are not far from .05 either.

We can perform a hypothesis test of whether the rejection rate is .05 by checking whether .05 is in the confidence interval for the proportion. We will use the **proportion_confint** function from **statsmodels** :cite:`seabold2010statsmodels`.  In :ref:`mtable2`, we see that .05 is comfortably within intervals for both parameters.  Our simulation successfully corroborated our assumptions about the model and the consistency of our method for estimating the posterior mode, variance, and distribution.
 
 .. table:: A.D. Confidence Intervals, :math:`n_{pre} = 1000`, :math:`n_{post} = 600`, :math:`n_{MCMC}=500`. :label:`mtable2`
    
    +---------------------------+-------+-------+
    | Statistics                | Lower | Upper |
    +===========================+=======+=======+
    | :math:`\alpha` AD Reject  | 0.030 | 0.056 |
    +---------------------------+-------+-------+
    | :math:`\beta` A.D. Reject | 0.033 | 0.060 |
    +---------------------------+-------+-------+

Conclusion
----------

We have explored how Bayesian analysis can be performed without resampling and still obtain
full inference.  With adequate amounts of the data, the posterior mode can be estimated with numeric optimization and the posterior variance can be estimated with numeric or automatic differentation.  The asymptotic normality of the posterior distribution enables simple calculation of posterior probabilities and confidence (credible) intervals as well.

Bayesian methods let us use data from past experience, subject matter expertise, and different levels of certainty to solve data sparsity problems and provide a probabilistic basis for inference.  Retail Price Optimization benefits from historical data and different granularities of information.  Other fields may also take advantage of access to large amounts of data and be able to use these approximation techniques.  These techniques and the tools implementing them can be used by practicioners to make their analysis more efficient and less intimidating.
