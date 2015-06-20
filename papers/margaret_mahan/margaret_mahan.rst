:author: Margaret Y Mahan
:email: mahan027@umn.edu
:institution: Brain Sciences Center, Minneapolis VA Health Care System & University of Minnesota

:author: Chelley R Chorn
:institution: Brain Sciences Center, Minneapolis VA Health Care System & University of Minnesota

:author: Apostolos P Georgopoulos
:institution: Brain Sciences Center, Minneapolis VA Health Care System & University of Minnesota

-------------------------------------------------------------------------------------------------------------
White Noise Test: detecting serial correlation and nonstationarities in long time series after ARIMA modeling
-------------------------------------------------------------------------------------------------------------

.. class:: abstract

Time series analysis has been a dominant technique for assessing relationships within temporally derived datasets and is becoming increasingly prevalent in the scientific community; for example, assessing brain networks by calculating pairwise correlations of time series generated from different areas of the brain. The assessment of these relationships relies, in turn, on the proper calculation of interactions between time series, which is achieved by rendering each individual series stationary and nonautocorrelated (i.e., white noise, or to “prewhiten” the series). This ensures that the relationships computed subsequently are due to the interactions between the series and do not reflect internal dependencies of the series themselves. An established method for prewhitening time series is to apply an Autoregressive (AR, *p*) Integrative (I, *d*) Moving Average (MA, *q*) model (ARIMA) and retain the residuals. To diagnostically check whether the model orders (*p*, *d*, *q*) are sufficient, both visualization (ACF & PACF plots) and tests (Ljung-Box, KPSS, ADF, etc.) of the residuals are performed. However, these tests are not robust for high-order models in long time series. Additionally, as dataset size increases (i.e., number of time series to model) it is not feasible to visually inspect each series independently. As a result, there is a need for robust alternatives to diagnostic evaluations of ARIMA modeling. Here, we demonstrate how to perform ARIMA modeling of long time series using statsmodels, a library for statistical analysis in Python. Then, we present a new algorithm (White Noise Test) to detect serial correlation and nonstationarities in prewhitened time series to establish the series does not differ significantly from white noise. This test was validated using time series collected from magnetoencephalography recordings. Overall, our White Noise Test provides a robust alternative to diagnostic checks of ARIMA modeling for long time series.

.. class:: keywords

   Time series, statsmodels, ARIMA

Introduction
------------

Time series are discrete, stochastic realizations of underlying data generating processes [Yaffee00]. They are ubiquitous in any field where monitoring of data is involved. For example, time series can be environmental, economic, or medical. In addition, time series can provide information about trends (e.g., broad fluctuations in values) and cycles (e.g., systematic, periodic fluctuations in values). Time series analysis can also be used to predict the next value in the series, given some model of its history. This is of special importance in environmental and econometric studies where forecasting the next set of values (e.g., the weather or a stock price) may have serious practical consequences. In other fields, time series provide crucial information about an evolving process (e.g., rate of spread of a disease or changing pollution levels) with implications about the effect of interventions. Finally, time series can provide fundamental information about the process that generates them, leading to a scientific understanding of that process (e.g., brain network analysis).

In time series analysis, there are two main investigative methods: frequency-domain and time-domain. Within the time-domain, typically crosscorrelation analysis utilized as a measure of the relation between time series. Now, it is commonly the case that a time series contains some level of serial correlation, meaning values in the time series influence previous values. It is also common for a time series to exhibit nonstationarities, such as drifts or trends over time. In either case, the crosscorrelation function calculated between two series containing either serial correlation or nonstationarities will give misleading results. To circumvent this, time series are modeled to remove such characteristics, as in the case of prewhiteining.

Prewhitening
------------

A white noise process is a continuous time series of random shocks, normally and independently distributed, with a zero mean and constant variance. If after modeling a time series the residuals are white noise, then we say the series has been prewhitened. An established method for prewhitening time series is to apply an Autoregressive (AR) Integrative (I) Moving Average (MA) model (ARIMA) and retain the residuals [Box76]. The full specification of an ARIMA model comprises the orders of each component, (*p*, *d*, *q*), where *p* is the number of preceding values in the autoregressive component, *d* is the number of differencing, and *q* is the number of preceding values in the moving average component. Most importantly, the ARIMA method requires the input time series to be: (1) equally spaced over time, (2) of sufficient length, (3) continuous (i.e., no missing values), and (4) stationary in the second or weak sense.

Prewhitening using ARIMA modeling takes three main steps. First, identify and select the model, by detecting and removing factors that influence the time series, such as nonstationarities or seasonalities, and identifying the AR and MA components (i.e., model orders). Second, estimate parameters, by deriving the parameter values (model coefficients). Third, evaluate the model, by checking the model’s adequacy through establishing that the series has been rendered stationary and nonautocorrelated. This time series modeling is iterative, successively refining the model until white noise residuals are obtained. Overall, a good model serves three purposes: providing the background information for further research on the process that generated the time series; enabling accurate forecasting of future values in the series; and yielding the white noise residuals necessary to evaluate accurately associations between time series, since they are devoid of any dependencies stemming from within their own series.

Model Identification and Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are several factors that can influence a value in a time series, which arise from previous values in the series, variability in these values, or nonstationarities (trend, drift, changing variance, or random walk). It is important to properly remove these factors by modeling the time series. To identify the model orders for an ARIMA(*p*,*d*,*q*), the autocorrelation (ACF) and partial autocorrelation (PACF) functions are used extensively.

First, nonstationarities need to be removed before ARMA modeling. A nonstationary process is identified by an ACF that does not tail away to zero quickly or cut-
off after a finite number of steps. If the time series is nonstationary, then a first differencing of the series is computed. This process is repeated until the time series is stationary, which determines the value of d. Two of the most frequently used tests for nonstationarities are the augmented Dickey-Fuller test [Said84] and the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test [Kwiatkowski92]. The ADF is a unit root test for the null hypothesis that a time series is I(1) while the KPSS is a stationarity test for the null hypothesis that a time series is I(0). These tests are complementary and can be used together. We implement the ADF test using statsmodels and the KPSS test using the arch python package.

To choose p and q orders, ACF and PACF show typical patterns, based on which a tentative ARIMA model can be postulated. There are three main patterns. A pure MA(*q*) process will have an ACF that cuts off after *q* and a PACF that tails off. A pure AR(*p*) process will have an ACF that tails off and a PACF that cuts off after *p*. For a mixed-model ARMA(*p*,*q*) process, both the ACF and PACF will tail off. The initial model selection begins with the minimum orders. 

Parameter Estimation
^^^^^^^^^^^^^^^^^^^^

ARIMA modeling has been implemented in Python with the statsmodels package [McKinney11],[Seabold10]. It includes parameter estimation and model evaluation procedures. After the model orders have been selected, the model parameters can be estimated with the ``statsmodels.tsa.arima_model.ARIMA.fit()`` function to maximize the likelihood that these coefficients describe the data. First, initial estimates of the parameters are used to get close to the desired parameters. Second, optimization functions are applied to adjust the parameters to maximize the likelihood by minimizing the negative loglikelihood function. If adequate initial parameter estimates were selected, a local optimization algorithm will find the local loglikelihood minimum near the parameter estimates, which will be the global minimum.

In statsmodels, default starting parameter estimations are calculated using the Hannan-Rissanen method [Hannan82] and these parameters are checked for stationarity and invertibility. If ``method`` is set to ``css-mle``, starting parameters are estimated further with conditional sum of squares methods. Parameters estimated in this way are not guaranteed to be stationary, so starting parameters may be set as an input variable (``start_params``) to ``ARIMA.fit()``. We have implemented a custom starting parameter selection method, which forces stationarity and invertibility, if necessary. In addition, the Hannan-Rissanen method uses an initial AR model with an order selected by minimizing BIC; then it estimates ARMA using the residuals from that model. This initial AR model is required to be larger than max(p,q) of the desired ARIMA model, which is not guaranteed with an AR selected by BIC criterion. We have implemented a method similar to Hannan-Rissanen, the long AR method, which is equivalent to Hannan-Rissanen except the initial AR model is set to be large (AR = 300). This results in an initial AR model order which is guaranteed to be larger than max(*p*, *q*), and starting parameter selection is more time efficient since fitting multiple AR model orders to optimize BIC is not required.

To fit ARIMA models, statsmodels has options for methods and solvers. The chosen method will determine the type of likelihood for estimation, where ``mle`` is the exact likelihood maximization, ``css`` is the conditional sum of squares minimization, and ``css-mle`` involves first estimating the starting parameters with css followed by an mle fit. The solver variable in ARIMA.fit() designates the optimizer from ``scipy.optimize`` for minimizing the negative loglikelihood function. Optimization solvers ``nm`` (Nelder-Mead) and ``powell`` are the most time efficient because they do not require a score, gradient, or Hessian. The next fastest solvers, ``lbfgs`` (limited memory Broyden-Fletcher-Goldfarb-Shanno), ``bfgs`` (Broyden-Fletcher-Goldfarb-Shanno), ``cg`` (conjugate gradient), and ``ncg`` (Newton conjugate-gradient), require a score or gradient, but no Hessian. The ``newton`` (Newton-Raphson) solver requires a score, gradient, and Hessian. Lastly, a global solver ``basinhopping``, displaces parameters randomly before minimizing with another local optimizer. For more information about these solvers, see ``statsmodels.base.model.GenericLikelihoodModel``.

Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^

The parameters of a specific model include coefficients for the (p, q) terms applied to the original series (if *d* = 0) or to the differenced series (if *d* > 0). There are two components in evaluating an ARIMA model, namely model stability and model adequacy. For the model to be stable, the absolute values of all *p* and *q* coefficients should be < 1, i.e., within bounds of stationarity (for the *p* coefficients) and invertibility (for the *q* coefficients). For the model to be adequate, the residual time series should not be significantly different from white noise. If either model stability or adequacy have not been established, then model identification and selection should be revised.   

Inspecting the *p* and *q* coefficients for being within bounds checks model stability and model adequacy is checked by examining the time-varying mean of the residuals (should be close to zero), their variance (should not differ appreciably along time), and their autocorrelation (should not be different from chance). Finally, the ACF and PACF of the residuals should not contain statistically significant terms more than the number expected by chance. This number depends on the number of lags; for example, if k = 40 lags, one would expect 2 values (5% of 40) to exceed their standard error (approximates the inverse of the square root of the length of the series, when the series is long [Bartlett46]).

Magnetoencephalography (MEG) Dataset
------------------------------------

To evaluate the functional brain, MEG is the optimal technique because it measures magnetic fluctuations generated by synchronized neural activity in the brain noninvasively and at high temporal resolution. For the applications below, MEG recordings were collected using a 248-channel axial gradiometer system (Magnes 3600WH, 4-D Neuroimaging, San Diego, CA) sampled at ~1 kHz from 50 cognitively healthy women (40 - 93 years, 70.58 ± 14.77, mean ± std dev) in a task-free state (i.e., resting state). The data were time series consisting of 50,000 values per subject and channel. Overall, the full MEG data matrix contains 50 samples x 248 channels x 50,000 time points.

Performing ARIMA Modeling
-------------------------

Method-solver implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The length and quantity of time series has a direct impact on the ease of modeling. Therefore, we aim to implement an iterative approach to ARIMA modeling while keeping focus on model reliability and validity of residuals, along with incorporating an efficiency cost (i.e., constraints on allowed processing time). The goal for this stage is to determine which method-solver is most appropriate for the application dataset, not necessarily to achieve white noise residuals.

To accomplish this, we randomly select 5% (round to nearest integer) of the channels from each sample for the MEG test dataset (N = 600). Next, we select a range of model orders: AR = {10, 20, 30, 40, 50, 60}, I = {1}, MA = {1, 3, 5}. Taking combinations of these model orders, we have 18 total model order combinations. Using each method-solver group (16 total) and model order combination, we now have 288 testing units. For each of the testing units, ARIMA modeling is performed on each channel in the test dataset. If 2% of the test dataset channels have a processing time > 5 minutes per channel, then the testing unit is withdrawn from further analysis. Otherwise, for each channel, four measures are retained. The first measure is the AICc (Akaike Information Criterion with correction for finite sample sizes), which describes the quality of the statistical models performance. The second and third measures calculate the ACF and PACF, respectively, up to AR lags and a count of the number of lags exceeding alpha = 0.01/AR is retained. The final measure is the processing time, which is measured on each channel and is the time (in seconds) it takes to start the ARIMA modeling process until residuals are produced. For all measures, lower values indicate better performance. Then, for each channel and model order, ranks for the first three measures are calculated across the method-solver groups, with tied ranks getting the same rank number.

Cumulative distribution functions (CDFs) of each method-solver group are calculated and plotted in Figure :ref:`egfig`. In this plots, larger area under the curve indicates better performance. In Table 1, the mean time per channel for each method except withdrawn methods (css-basinhopping, mle-bfgs, mle-newton, mle-cg, mle-ncg, mle-powell, mle-basinhopping) is given, along with the highest order able to be modeled. Mean ranks were calculated for each method-solver, shown in Table :ref:`mtable`, and used for the final rank calculation. In the MEG test dataset, the css-lbfgs method-solver outperformed all others while maintaining a reasonable time per channel (91.47 seconds). The css-lbfgs method-solver was retained for all further MEG analysis.

.. figure:: figure1.png
   :align: center

   MEG CDF Ranks :label:`egfig`

.. table:: MEG Method-Solver Attributes :label:`mtable`

   +------------+----------+-----------+--------+-------+
   |Method-     | Mean     | Highest   | Mean   | Final |
   |Solver      | Time (s) | Model     | Ranks  | Rank  |            
   +============+==========+===========+========+=======+
   |css-lbfgs   | 91.47    |60-1-3     |1.32    |1      |
   +------------+----------+-----------+--------+-------+
   |css-bfgs    |115.22    |60-1-3     |2.23    |2      |
   +------------+----------+-----------+--------+-------+
   |css-powell  | 54.47    |60-1-5     |3.25    |3      |
   +------------+----------+-----------+--------+-------+
   |css-cg      |132.78    |50-1-1     |3.77    |4      |
   +------------+----------+-----------+--------+-------+
   |css-nm      | 39.55    |60-1-3     |4.29    |5      |
   +------------+----------+-----------+--------+-------+
   |css-ncg     |138.97    |20-1-3     |6.9     |6      |
   +------------+----------+-----------+--------+-------+
   |mle-nm      | 85.71    |30-1-5     |7.31    |7      |
   +------------+----------+-----------+--------+-------+
   |mle-lbfgs   | 57.7     |10-1-5     |8.29    |8      |
   +------------+----------+-----------+--------+-------+
   |css-newton  |235.11    |20-1-1     |8.36    |9      |
   +------------+----------+-----------+--------+-------+

Selecting model orders
^^^^^^^^^^^^^^^^^^^^^^

Before selecting the differencing model order, *d*, each series is inspected for extreme values. For each raw series, the 25th and 75th percentiles, along with the inter quartile range (IQR = 75th – 25th) are calculated. Using these values, the upper fence = 75th + 3*IQR and the lower fence = 25th – 3*IQR are calculated. Then, the values below the lower fence and above the upper fence are counted. If this count is greater than 5, the series is removed from further consideration when selecting model orders. The remaining series are first differenced (*d* = 1). Next we check the series for stationarity; recall, an appropriately differenced process should be stationary. Both the KPSS stationarity test and ADF unit root test are calculated. Their values plotted against each other are shown in Figure :ref:`egfig2`. The KPSS statistic ranges from 0 to 0.28; since all KPSS test statistics calculated are less than the critical value of 0.743 at the p = 0.01, the null hypothesis of stationarity cannot be rejected. The ADF statistic ranges from -16.19 to -58.32; since all ADF test statistics calculated are more negative than the critical value of -3.43 at the p = 0.01, the null hypothesis of a unit root is rejected. Taken together, we have established stationarity for our test dataset.

.. figure:: figure2.png
   :align: center

   Stationarity (KPSS) and Unit Root (ADF) Tests :label:`egfig2`

Taking the differenced series, the ACF and PACF are calculated for ±60 lags. The median correlation value for each lag is plotted in Figure :ref:`egfig3`. From this figure, a mixed-model ARMA(*p*,*q*) process is seen since both the ACF and PACF tail off. To decide on the *p* and *q* orders, we calculate the minimum lag where the correlation value is less than two times the standard error.  For the PACF, this occurs at lag 30 and for the ACF, this occurs at lag 13. To avoid possible overfitting the data (although not as critical since we are not forecasting from the model), we choose to begin the White Noise Test with an ARIMA of (20,1,3).

.. figure:: figure3.png
   :align: center

   ACF and PACF of MEG data after first differencing :label:`egfig3`

White Noise Test
-----------------

*Unmodeled data*: Channels unable to be modeled using the css-lbfgs with the given model order were excluded from further analysis. Channels with extreme values, as calculated during the differencing step, were also excluded from further analysis.

*Normality*: Each residual series was tested for normality using the Kolmogorov–Smirnov test.

*Zero mean*: A one-sample t-test was calculated for each residual series to test whether the mean is significantly different from zero. In addition, each series was split into 10% nonoverlapping windows (i.e. 5 windows). For each window, a one-sample t-test was again calculated. A count of the number of windows with means significantly different from zero was retained for each residual series (maximum value = 5). 

*Constant variance*: For each residual series, the windows were also tested for equal variances using Bartlett’s test. A count of the number of window-pairs with unequal variances was retained for each residual series (maximum value = 9). 

*Uncorrelated with all other realizations*: The ACF and PACF were calculated for each residual series up to AR lags (i.e., p used in the model to calculate the residuals). The t-statistic = absolute value of the ACF / standard error (df = N-1) at each lag, k, is calculated used for a two-tailed test; a cumulative count of those exceeding alpha = 0.01/AR were retained. If the count is greater than 5% of the AR order for either the ACF or PACF, there is significant serial correlation within the residuals.

REVIEWERS: we are finishing up this section with figures.

Conclusion
----------

REVIEWERS: we are finishing up the remaining sections completed by Monday 6/22/15. Thanks!!