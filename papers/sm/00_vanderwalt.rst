:author: Gaius Caesar
:email: jj@rome.it
:institution: Senate House

:author: Mark Anthony
:email: mark37@rome.it
:institution: Egyptian Embassy

------------------------------------------------
A Numerical Perspective to Terraforming a Desert
------------------------------------------------

.. class:: abstract

   A short version of the long version that is way too long to be written as a
   short version anyway.  Still, when considering the facts from first
   principles, we find that the outcomes of this introspective approach is
   compatible with the guidelines previously established.

   In such an experiment, it is then clearl that the potential for further
   development not only depends on previous relationships found but also on
   connections made during exploitation of this novel new experimental
   protocol.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
------------

Twelve hundred years ago, in a galaxy just across the hill...

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
justo vitae odio tincidunt tempus at vitae tortor.

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
   :linenos:

   int main() {
       for (int i = 0; i < 10; i++) {
           /* do something */
       }
       return 0;
   }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
   :linenos:
   :linenostart: 2

   for (int i = 0; i < 10; i++) {
       /* do something */
   }
 
Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

   g(x) = \int_0^\infty f(x) dx

The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas diam turpis, placerat at adipiscing ac,
pulvinar id metus.

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+-------+
   | Material   | Units |
   +------------+-------+
   | Stone      | 3     |
   +------------+-------+
   | Water      | 12    |
   +------------+-------+

We show the different quantities of materials required in Table
:ref:`mtable`.

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.


Introduction
------------

Time Series comprises observations that are ordered along one
dimension, that is time, which imposes specific stochastic structures
on the data. Our current models assume that observations are
continuous, that time is discrete and equally spaced and that we do
not have missing observations. This type of observations is very
common in many fields, in economics and finance for example, national
output, labor force, prices, stock market values, sales volumes, just
to name a few.

In the following we briefly discuss some statistical properties of the
estimation with time series data, and then illustrate and summarize
what is currently available in statsmodels.


Ordinary Least Squares (OLS)
----------------------------

The simplest linear model assumes that we observe an endogenous
variable y and a set of regressors or explanatory variables x, where y
and x are linked through a simple linear relationship plus a noise or
error term

.. math::

   y_t = x_t \beta + \epsilon_t

In the simplest case, the errors are independently and identically
distributed. Unbiasedness of OLS requires that the regressors and
epsilon are uncorrelated. If the errors are additionally normally
distributed and the regressors are non-random, then the resulting OLS
or Maximum Likelihood estimator of beta is also normally distributed
in small samples. We obtain the same resut, if we consider consider
the distributions as conditional on :math:`x_t` when they are exogenous random
variables. So far this is independent whether t indexes time or any
other index of observations.

When we have time series, there are two possible extensions that come
from the intertemporal linkage of observations. In the first case,
past values of the endogenous variable influence the expectation or
distribution of the current endogenous variable, in the second case
the errors :math:`\epsilon_t` are correlated over time. If we have either one
case, we can still use OLS or generalized least squares GLS to get a
consistent estimate of the parameters. If we have both cases at the
same time, then OLS is not consistent anymore, and we need to use a
non-linear estimator. This case is essentially what ARMA does.

Linear Model with autocorrelated error (GLSAR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model assumes that the explanatory variables, regressors, are
uncorrelated with the error term. But the error term is an
autoregressive process, i.e.

.. math::

   E(x_t, \epsilon_t) = 0

.. math::

   \epsilon_t = a_1 \epsilon_{t-1} + a_2 \epsilon_{t-1} + ... + a_k \epsilon_{t-k}


An example will be presented in the next section.

Linear Model with lagged dependent variables (OLS, AR, VAR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This group of models assume that past dependent variables, $y_{t-i},
are included among the regressors, but that the error term are not
serially correlated

.. math::

   E(\epsilon_t, \epsilon_s) = 0, \text{ for } t \neq s 

.. math::

   y_t = a_1 y_{t-1} + a_2 y_{t-1} + ... + a_k y_{t-k} + x_t \beta + \epsilon_t


Dynamic processes like autoregressive processes depend on observations
in the past. This means that we have to decide what to do with the
initial observations in our sample where we do nnt observe any past
values.

The simplest way is to treat the first observation as fixed,
and analyse our sample starting with the k_th observation. This leads
to conditional least squares or conditional maximum likelihood
estimation. For conditional least squares we can just use OLS to
estimate, adding past `endog` to the `exog`. The vector autoregressive
model (VAR) has the same basic statistical structure except that we
consider now a vector of endogenous variables at each point in time,
and can also be estimated with OLS conditional on the initial
information. (The stochastic structure of VAR is richer, because we
now also need to take into account that there can be contemporaneous
correlation of the errors, i.e. correlation at the same time point but
across equations, but still uncorrelated across time.) The second estimation
method that is currently available in statsmodels is maximum likelihood
estimation. Following the same approach, we can use the likelihood function that
is conditional on the first observations. If the errors are normaly distributed,
then this is essentially equivalent to least squares. However, we can easily
extend conditional maximum likelihood to other models, for example GARCH, linear
models with generalized autoregressive conditional heteroscedasticity, where
the variance depends on the past, or models where the errors follow a non-normal
distribution, for example Student-t distributed which has heavier tails and is
sometimes more appropriate in finance.

The second way to treat the problem of intial conditions is to model them together
with other observations, usually under the assumption that the process has started
far in the past and that the initial observations are distributed according to
the long run, i.e. stationary, distribution of the observations. This exact
maximum likelihood estimator is implemented in statsmodels for the autoregressive
process in statsmodels.tsa.AR, and for the ARMA process in statsmodels.ARMA.

Autoregressive Moving average model (ARMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARMA combines an autoregressive process of the dependent variable with a error
term, moving-average or MA, that includes the present and a linear combination
of past error terms, an ARMA(p,q) is defined as

.. math::

   E(\epsilon_t, \epsilon_s) = 0, \text{ for } t \neq s


.. math::

   y_t = \mu + a_1 y_{t-1} + ... + a_k y_{t-p} +
   \epsilon_t + b_1 \epsilon_{t-1} + ... + b_q \epsilon_{t-q}

As a simplified notation, this is often expressed in terms of lag-polynomials as

.. math::
    
   \phi (L) y_t = \psi (L) \epsilon_t

where

.. math::

  \phi (L) = 1 - a_1 L^1 - a_2 L^2 - ... - a_k L^p
  
.. math::

  \psi (L) = 1 + b_1 L^1 + b_2 L^2 + ... + b_k L^q

:math:`L` is the lag or shift operator, :math:`L^i x_t = x_{t-i}, L^0 = 1`. This is the same
process that scipy.lfilter uses. Forecasting with ARMA models has become popular
since the 1970's as Box-Jenkins methodology, since it often showed better
forecast performance than more complex, structural models.

Using OLS to estimate this process, i.e. regressing :math:`y_t` on past :math:`y_{t-i}`, does not
provide a consistent estimator. The process can be consistently estimate using
either conditional least squares, which in this case is a non-linear estimator,
or conditional maximum likelihood or with exact maximum likelihood. The difference
between conditional methods and exact MLE is the same as described before.
statsmodels provides estimators for both methods in tsa.ARMA which will be
described in more detail below.

Time series analysis is a vast field in econometrics with a large range of models
that extend on the basic linear models with the assumption of normally distributed
errors in many ways, and provides a range of statistical tests to identify an
appropriate model specification or test the underlying assumptions.

Besides estimation of the main linear time series models, statsmodels also provides
a range of descriptive statistics for time series data and associated statistical
tests. We include an overview in the next section before describing AR, ARMA and
VAR in more details. Additional results that facilitate the usage and interpretation of the
estimated models, for example impulse response functions, are also available.

OLS, GLSAR and serial correlation
---------------------------------

Suppose we want to model a simple linear model that links the stock of money
in the economy to real GDP and consumer price index CPI, example in Greene
(2003, ch. 12). We import numpy and statsmodels, load the variables from
the example dataset included in statsmodels, transform the data and fit
the model with OLS:

.. code-block:: python

   import numpy as np
   import scikits.statsmodels.api as sm
   tsa = sm.tsa  # as shorthand

   mdata = sm.datasets.macrodata.load().data
   endog = np.log(mdata['m1'])
   exog = np.column_stack([np.log(mdata['realgdp']),
                           np.log(mdata['cpi'])])
   exog = sm.add_constant(exog, prepend=True)

   res1 = sm.OLS(endog, exog).fit()

`print res1.summary()` provides the basic overview of the regression results.
We skip it here to safe on space. The Durbin-Watson statistic that is included
in the summary is very low indicating that there is a strong autocorrelation
in the residuals. Plotting the residuals shows a similar strong autocorrelation.

As a more formal test we can calculate the autocorrelation, the Ljung-Box
Q-statistic for the test of zero autocorrelation and the associated p-values:

.. code-block:: python

    acf, ci, Q, pvalue = tsa.acf(res1.resid, nlags=4, confint=95,
                                 qstat=True, unbiased=True)
    acf
    #array([ 1.   ,  0.982,  0.948,  0.904,  0.85 ])
    pvalue
    #array([  3.811e-045,   2.892e-084,   6.949e-120,   2.192e-151])

To see how many autoregressive coefficients might be relevant, we can also look
at the partial autocorrelation coefficients

.. code-block:: python

   tsa.pacf(res1.resid, nlags=4)
   #array([ 1.   ,  0.982, -0.497, -0.062, -0.227])

Similar regression diagnostics, for example for heteroscedasticity, are
available in `scikits.statsmodels.stats.diagnostic`. Details on these
functions and their options can be found in the documentation and docstrings.

The strong autocorrelation indicates that either our model is misspecified or
there is strong autocorrelation in the errors. If we assume that the second
is correct, then we can estimate the model with GLSAR. As an example, let us
assume we consider four lags in the autoregressive error.

.. code-block:: python

   mod2 = sm.GLSAR(endog, exog, rho=4)
   res2 = mod2.iterative_fit()

`iterative_fit` alternates between estimating the autoregressive process of
the error term using tsa.yule_walker, and feasible sm.GLS. Looking at the
estimation results shows two things, the parameter estimates are very
different between OLS and GLS, and the autocorrelation in the residual is
close to a random walk:

.. code-block:: python

   res1.params
   #array([-1.502,  0.43 ,  0.886])
   res2.params
   #array([-0.015,  0.01 ,  0.034])

   mod2.rho
   #array([ 1.009, -0.003,  0.015, -0.028])

This indicates that the short run and long run dynamics might be very different
and that we should consider a richer dynamic model, and that the variables
might not be stationary and that there might be unit roots.

Stationarity, Unit Roots and Cointegration
------------------------------------------

Loosely speaking, stationarity means here that the mean, variance and
intertemporal correlation structure remains constant over time.
Non-stationarities can either come from deterministic changes like trend or
seasonal fluctuations, or the stochastic properties of the process, if for
example the autoregressive process has a unit root, that is one of the roots
of the lag polynomial is on the unit circle. In the first case, we can remove
the deterministic component by detrending or deseasonalization. In the second
case we can take first differences of the process,

.. math:

   dy_t = (1-L)y_t = y_t - y_{t-1}

Differencing is a common approach in the Box-Jenkins methodology and gives
rise to ARIMA, where the I stands for integrated processes, which are made
stationary by differencing. This lead to a large literature in econometrics
on unit-root testing that tries to distinguish deterministic trends from
unit roots or stochastic trends. statsmodels provides the augmented
Dickey-Fuller test. Monte Carlo studies have shown that it is often the most
powerful of all unit roots test.

To illustrate the results, we just show two results. Testing the log of the
stock of money with a null hypothesis of unit roots against an alternative
or stationarity around a linear trend, shows that a adf-statistic of -1.5
and a p-value of 0.8 so we are far away from rejecting the unit root
hypothesis

.. code-block:: python

   tsa.adfuller(endog, regression="ct")[:2]
   (-1.561, 0.807)

If we test the differenced series, growth rate of monestock, with a Null
hypothesis of Random Walk with drift, then we can strongly reject the
hypothesis that the growth rate has a unit root (p-value 0.0002)

.. code-block:: python

   tsa.adfuller(np.diff(endog), regression="c")[:2]
   (-4.451, 0.00024)









References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


