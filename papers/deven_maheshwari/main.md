---
# Ensure that this title is the same as the one in `myst.yml`
title: "Climate is not a straight line: Scalable Python-based GAMM Workflows for Wildlife Conservation"
abstract: |
  Wildlife conservation organizations are collecting growing volumes of ecological data, but
  technical barriers limit their effective use for decision-making. We present a Python-based trend
  fitting framework designed for non-technical conservationists that operationalizes Generalized
  Additive Mixed Models (GAMMs) within a scalable and reproducible pipeline. The framework
  automates model fitting, evaluation, and hyperparameter selection, and is fine-tuned for
  conservation use cases. Through a primary case study on nine Kenyan forest cover sites, and a
  secondary robustness check on elephant movement telemetry, we demonstrate how GAMMs improve
  data fitting compared to other common statistical baselines. We conclude by discussing how we implemented
  our framework within the Ecoscope library, which enabled the large-scale deployment of our
  analysis pipeline worldwide.
---

## Introduction

### Background

Environmental research organizations are recording an increasing amount of data on animal
movements and landscape changes. These real-world monitoring systems generate sparse and
irregular data across species, terrain, and sensors. Statistical tools, such as trend fitting
analysis, can leverage these datasets to help conservationists better anticipate how ecosystem
dynamics may evolve over time. Their widespread adoption, however, remains constrained by
technical barriers. Access to qualified personnel capable of analyzing these datasets and
extracting robust conclusions remains limited. This challenge highlights an urgent need for
analysis tools capable of transforming noisy and complex datasets into insights without
requiring the user to have advanced expertise in statistics and programming.

We present a Python-based trend fitting framework targeted at non-technical wildlife
conservation practitioners. Building on the SciPy [@scipy], Scikit-Learn [@sklearn1; @sklearn2],
and StatsModels [@statsmodels] libraries, our pipeline modularizes model fitting, hyperparameter
selection, and evaluation. We then
present interpretable results through an interactive and reproducible dashboard. At its core, our
framework utilizes Generalized Additive Mixed Models (GAMMs), a powerful statistical approach
for modeling complex, random effect relationships. We introduce a robust GAM/GAMM Regressor
class that automates smoothing parameter selection over large matrices of data. Through a primary
case study on forest cover across nine monitoring sites, we show how GAMMs can be a scalable and
interpretable improvement on other common time series models. This is especially
applicable in noisy, low-data, and multi-entity regimes often encountered in conservation databases.
A secondary case study on elephant movement speed, tracking over the course of one year, demonstrates the robustness of this
pipeline by applying it unmodified to a dataset that differs from the forest series in sampling
frequency, response variable, and noise structure. We show the relative improvements GAMs offer in
comparison to Generalized Linear Models (GLMs) and standard linear regression models (OLS), as
well as the challenges of considering deep learning regression analysis for ecological data.

We further discuss how our design decisions were shaped by trade offs between interpretability,
robustness, and predictive performance. This includes topics like mitigating the risks of overfitting and maintaining user trust.

Finally, we explain how we implemented our framework within Ecoscope, an open-source Python
library for conservation analytics. We provide a brief overview of the package architecture and
demonstrate how Ecoscope enables the successful real-world deployment of our trend fitting
pipeline within the EarthRanger monitoring platform, which is currently being used by more than
800 wildlife conservation sites across the globe.

## Related Work

### EarthRanger and the Ecoscope Library

EarthRanger is an open-source software platform developed by the Allen Institute for AI and
Wildlife Dynamics. It is used to support protected area managers, ecologists, and wildlife
biologists in analysis. The system consists of seven main components — Core Server, API,
Storage, Gundi, Web App, Mobile App, and Ecoscope — providing functionality for data handling
and storage, real-time and post-collection analysis, visualization, and sharing. EarthRanger
handles a variety of data including GPS telemetry, ranger patrol data, camera traps, and remote sensing.

Ecoscope is the open-source Python analytics layer of EarthRanger. Its modules are used for analysis workflows related to wildlife
movement and conservation datasets. It can handle data from major Earth reporting
data lakes including EarthRanger, Google Earth Engine, MoveBank, and GeoPandas. The library
supports conservationists as a visualization tool with established workflows unique to each
study, and has been integral in reducing analysis time in a wide range of research and conservation
fields.

The forest cover data used in this study is derived from the University of Maryland Hansen
Global Forest Change dataset, accessed via Google Earth Engine. This dataset provides annual,
30-meter resolution estimates of forest cover, loss, and gain globally from 2000 onward, enabling
consistent longitudinal analysis across geographically dispersed conservation sites [@hansen2013].
The elephant walking speed dataset is provided by Wildlife Dynamics and field researchers from
the Mara Elephant Project, and consists of hourly GPS relocations for a single collared bull
elephant recorded between March 2008 and April 2009.

### Trend Analysis for Ecological Datasets

Long term trend estimation in conservation is a fundamental challenge due to data
sparseness, irregularity, and comparability. Measurement noise is common due to tool errors and
interaction from other abiotic and biotic factors. Additionally, there exists a massive obstacle in
communicating findings to sites exhibiting similar traits.

In choosing a well fitting and robust regression model we consider the relative strengths and increased
complexity of four curves: standard linear regression (OLS), Generalized Linear Models (GLM), Generalized
Additive Models (GAM), and Generalized Additive Mixed Models (GAMMs).

Standard linear regression, while widely used, assumes a constant rate of change over time and a Gaussian error structure.
These assumptions are routinely violated in ecological datasets exhibiting nonlinear dynamics and regime changes,
such as natural disasters, climate changes, or an influx of human influence.

Generalized Linear Models (GLMs) extend the linear framework to accommodate non-Gaussian
response distributions by using a link function and exponential family distribution
[@mccullagh1989]. GLMs have been applied across ecology for modeling count data (Poisson
family), presence-absence data (binomial family), and positive continuous measurements (gamma
family). However, the linearity constraint on the link function again limits its ability to
represent trends that do not follow a linear relationship.


### Modeling Non-linear Dynamics

Generalized Additive Models (GAMs) address the linearity limitation of GLMs by replacing the
parametric linear predictor with a sum of smooth nonparametric functions estimated from data
[@hastie1990; @wood2017]. GAMs have seen increasing application in ecology for modeling
species distribution responses to environmental and phenological trends. Since they produce
smoother curves with quantified uncertainty, they are particularly useful for deployment in
field researcher use. It is important to note that all three discussed models are site specific.

The extension to Generalized Additive Mixed Models (GAMMs) introduces random effects to
the GAM framework, separating a smooth trend shared across all sites from site-specific
deviations. This allows for
multiple correlated observational units — sites, subjects, or
repeated measurements — to improve statistical strength. Forrest et al. [@forrest2020]
demonstrated that GAMMs substantially reduce bias in trend estimation compared to fixed effects
models in a biomedical context. This motivates their application to ecological time series with
analogous properties. Baayen and Linke [@baayen2021] provide a detailed outline of GAMM
fitting in the context of linguistics, highlighting the role of random smooth terms in capturing
deviations from population level trends.

Recent work has also explored the intersection of additive models and modern machine learning.
Knieper et al. [@knieper2025] proposed gradient boosting for generalized additive mixed models,
showing improved predictive performance in iterative sites. Agarwal et al. [@agarwal2021]
introduced Neural Additive Models, which replace spline basis functions with neural network
components while preserving the additive interpretability structure of GAMs. Xiong et al.
[@xiong2019] extended mixed effects modeling to neural networks, demonstrating that random
effect structures can be incorporated into deep learning architectures. These approaches represent
natural future extensions of the framework presented here, though their data requirements exceed
what is currently available at these EarthRanger sites.

### Model Progression

Each model in this progression relaxes exactly one assumption from its predecessor, enabling
progressively more flexible representations of ecological time series data.

**Ordinary Least Squares (OLS):** The standard linear regression model assumes a Gaussian
response with an identity link and a strictly linear relationship between the predictor and the
conditional mean, $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$, with parameters estimated by
minimizing the residual sum of squares.

**Generalized Linear Model (GLM):** The GLM extends OLS by introducing a link function
$g(\cdot)$ relating a linear predictor to the conditional mean $\mu_i$, and by allowing the response
to follow any distribution in the exponential family. In this work we use a Gaussian family with a
log link, giving $\log(\mu_i) = \beta_0 + \beta_1 x_i$.

**Generalized Additive Model (GAM):** The GAM replaces the linear predictor with a smooth
nonparametric function $f(x_i)$, so that $g(\mu_i) = \beta_0 + f(x_i)$, where $f$ is represented as a
weighted sum of $K$ B-spline basis functions. To prevent overfitting, a smoothing penalty scaled by
parameter $\alpha \geq 0$ is applied.

**Generalized Additive Mixed Model (GAMM):** The GAMM extends the GAM to multi-site
data by decomposing the smooth trend into a global component shared across all sites and
site specific smooth deviations modeled as random effects:

```{math}
:label: eq:gamm

g(\mu_{ij}) = \beta_0 + f(x_{ij}) + b_j, \qquad b_j \sim \mathcal{N}(0, \sigma_b^2)
```

where $i$ indexes observations, $j$ indexes sites, $f(x_{ij})$ is the population-level smooth, and
$b_j$ is a site-specific random intercept shrunk toward zero.

## Regression Class Design for Ecoscope

The trend fitting module in Ecoscope is built on top of a set of scikit-learn compatible regressor
classes [@sklearn1; @sklearn2]. Each of the four models follows the standard `fit` and `predict`
interface, which includes metrics for estimating model fit strength. The four classes —
`LinearRegressionRegressor`, `GLMRegressor`, `GAMRegressor`, and `GAMMRegressor` — share
common characteristics in handling preprocessing, normalization, and hyperparameterization
internally. Conservationists interact only with the raw data inputs and interpretable outputs using
the workflow GUI.

This section describes the key design decisions behind these classes, evaluating the tradeoffs
between model flexibility and stability. Primary design decisions include normalization,
smoothing parameter selection, and evaluation techniques.

(alpha-normalization)=
### Alpha Selection and Normalization

The GAM smoothing parameter $\alpha$ controls the balance between curve flexibility and
smoothness. Values near $10^{-6}$ produce maximally flexible curves that closely follow every
fluctuation in the data (also described as "wiggly"). Values near $10^4$ produce nearly
linear curves. Selecting an appropriate $\alpha$ is critical: too low causes overfitting, too high
produces underfitting that misses nonlinear dynamics.

Prior to fitting, all inputs are normalized within each regressor class. The time variable $X$ is
shifted to start at zero by subtracting its minimum value, and for Gaussian-family fits the response
variable $y$ is standardized to mean zero and unit variance. Shifting the time axis is required
because knot boundaries are computed in shifted coordinates, and it additionally improves the
conditioning of the spline design matrix. The same transformations apply to the elephant dataset,
where the time axis is derived from raw timestamps stored as nanoseconds since the Unix epoch and
converted to days since the first GPS fix, so the series begins at day zero by construction.

Normalization is applied internally and
automatically inverted during prediction, so users always interact with inputs and outputs in their
original units.

Once axes are standardized, the optimal $\alpha$ is selected via grid search over 100 candidate
values drawn from a logarithmic grid spanning $[10^{-6}, 10^4]$. For each candidate value, a GAM
is fit and scored using the evaluation metric specified by the user. The candidate producing the
best score is selected and the final model is refitted on the full training data using that value.

For datasets with ten or fewer observations, leave-one-out cross-validation [@arlot2010] is used
to maximize the use of available data. For larger datasets, time-series-aware cross-validation
[@arlot2010; @sklearn1] via scikit-learn's `TimeSeriesSplit` is used with five splits. This ensures
that test observations always occur after their corresponding training observations and ensures
realistic forecasting performance.

During rolling origin cross-validation, the GAM is trained on a temporal subset of the data but
predicts observations within the full time range. To prevent out of bounds errors in spline
evaluation, the lower and upper knot boundaries are set to the full dataset range rather than the
training subset range. Boundaries passed as raw year values are converted to the normalized scale
before being passed to the spline constructor. A sliding window trains the model on $N$ years and
predicts values for the next 3 years.

### Evaluation Measures

Four evaluation measures are implemented across all regressor classes.

**Akaike Information Criterion (AIC)** rewards model fit while penalizing wiggliness. Lower AIC
indicates a better model given its complexity. AIC is only meaningful when comparing models fit
on the same dataset and should not be used to compare models across sites.

```{math}
AIC = -2\ell(\hat{\beta}) + 2p
```

where $\ell(\hat{\beta})$ is the maximized log-likelihood and $p$ is the number of effective parameters.

**Bayesian Information Criterion (BIC)** applies a more severe penalty for model complexity
than AIC, scaled by the log of the sample size. Like AIC, it is only valid for within-dataset
comparisons.

```{math}
BIC = -2\ell(\hat{\beta}) + p\log(n)
```

**Coefficient of Determination ($R^2$)** measures the proportion of variance in the response.
$R^2 = 1.0$ indicates a perfect fit; $R^2 = 0.0$ indicates the model performs no better than
predicting the mean. Since $R^2$ is normalized, it can be used across sites.

```{math}
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
```

where $SS_{res}$ is the sum of squared prediction errors and $SS_{tot}$ is the sum of squared
deviations from the mean.

**Mean Squared Error (MSE)** measures the average squared difference between predicted and
actual values. MSE is used as the scoring metric for $\alpha$ selection during cross-validation.

```{math}
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

Note that in the rolling origin evaluation context, MAE (mean absolute error) is additionally
reported due to its frequent usage by field rangers.

### Rolling Origin Evaluation

Standard cross-validation strategies such as $k$-fold are inappropriate for time series data
because they allow test observations to occur before their corresponding training observations
[@bergmeir2012].
Rolling origin evaluation addresses this by enforcing chronological ordering throughout the
evaluation pipeline.

A minimum training size is specified (15 years for the forest cover work; not applied to the single-year elephant dataset). At each step, the model is
trained on all observations up to a cutoff year and evaluated on the subsequent $N$ years of
observations (3 years in this work). The cutoff is then advanced by one year and the process repeats
until the end of the series is reached. For each evaluation window, MAE and RMSE are computed
between predicted and observed values in the test horizon. These are averaged across all windows to
produce a single score per site per model. Spline knot boundaries are held fixed to the full
observation window at every origin so that the basis is identical across folds.

For the GAMM, rolling origin evaluation is conducted across all sites simultaneously. At each
cutoff, the model is trained on the first $N$ years of data for every site jointly, and evaluated on
the subsequent three years for each site. This preserves the multi-site structure of the GAMM
while still evaluating out-of-sample forecasting performance per site. The model always has
access to all sites during training.

### Robust Data Handling

This framework is designed to handle datasets encountered across EarthRanger monitoring sites
globally and relies on built-in Ecoscope functionality for I/O routing and plotting.

The trend analysis module enforces guards for a variety of data reporting issues. First, not all
sites have the same number of observations. The cross-validator selection described in
{ref}`alpha-normalization` adapts automatically to sample size with either leave-one-out or
time-series-aware validation. A minimum observation warning is raised when any site has fewer than
ten data points. Second, time intervals can be inconsistent or missing in reports. The framework is
not restricted to annual data: the time variable $X$ is treated as a continuous numeric input and
normalized to start at zero. Missing years are handled gracefully because the spline basis is defined
over the full observed range and predictions can be made at any point within that range.

For the GAMM, all sites are combined into a single DataFrame with a site identifier column before
fitting. The random intercept term requires that site identities be consistent between
training and prediction. For forecasts at unseen sites, the population level smooth is
returned as a fallback with all site random effects set to zero.

## Workflow Dashboard Generation

### Forest Cover

We apply the trend fitting pipeline to nine Kenyan forest monitoring sites spanning 2001–2024.
Forest survival area in acres is extracted from the Hansen Global Forest Change dataset
[@hansen2013] via Google Earth Engine, retaining only pixels whose year-2000 tree cover exceeds
60%. Annual loss within each site is accumulated and subtracted from the baseline forested area to
give a survival curve. OLS, GLM, and GAM are fit independently per site. The GAMM is fit jointly
across all nine sites as a shared cubic B-spline trend with a per-site random intercept, estimated
by Markov chain Monte Carlo through Bambi [@bambi] and PyMC [@pymc].

:::{figure} figure1.png
:label: fig:dashboard
:align: center
:width: 70%
Sample forest cover workflow dashboard generated from the publicly accessible workflow. This is
the mock used for the Mount Kenya site for planning. It includes a map of the region, a plot
displaying the three site-specific regression curves, and the multi-site GAMM fit. All show a
general decline in forest cover with a relatively large drop-off between 2016–2018 and close
alignment with the GAM curve.
:::

(elephant-speed)=
### Elephant Walking Speed

To test whether the framework transfers beyond annual remote sensing data, we apply the identical
pipeline to GPS telemetry from a single collared elephant monitored by the Mara Elephant
Project between March 2008 and April 2009. The record contains 8,355 hourly movement segments,
which was summarized by a daily mean travel speed to give 367 observations.

The contrast between model classes is more significant than in the forest cover case. OLS fits a
nearly flat line at 0.55 km h$^{-1}$ whose slope is statistically indistinguishable from zero. The
GAM, at a selected $\alpha$ of 1.45, instead recovers variation with $R^2 = 0.40$.

Refitting on the unaggregated hourly segments (@fig:elephant, panel b) displays a similar smooth
despite a doubling of observation level standard deviation (0.66 against 0.30 km h$^{-1}$ for daily
means) and an order of magnitude more observations. The stability of the recovered trend across
aggregation levels indicates that the $\alpha$ search responds to signal rather than to sampling
resolution, which supports the framework's applicability to the irregular, high-frequency telemetry
typical of EarthRanger deployments. The purpose of this case study is to show that the pipeline
behaves correctly on data structurally unlike the forest series, not to establish an ecological
result.

:::{figure} figure3.png
:label: fig:elephant
:align: center
:width: 100%
Fitted OLS and GAM curves with 95% confidence intervals for elephant travel speed over one year.
(a) Daily mean speed, 367 observations. (b) The same models fitted to the 8,355 unaggregated hourly
movement segments. The linear fit is flat in both panels, while the GAM recovers comparable
multi-modal structure at both aggregation levels.
:::

### In-Sample Fit

To understand how well each model type fits the nine datasets and how effective the smoothing
parameter grid search is for the `GAMRegressor` class, we ran the deployed trend analysis module
on each site. @tbl:insample reports in-sample fit metrics for the three single site models.

GAM achieves the lowest AIC on every site, confirming that flexible smooth curves provide better
fit than linear alternatives even after penalizing for complexity. GAM $R^2$ exceeds 0.98 on all
sites. OLS $R^2$ ranges from 0.80 to 0.98, and GLM $R^2$ from 0.81 to 0.99.

Selected $\alpha$ values are ecologically interpretable. Eburu receives $\alpha = 0.0014$ reflecting its
sharp single-year drop. Samburu receives $\alpha = 95.5$ consistent with its near-flat trend.
MaraConservancies receives $\alpha = 0.035$ capturing its sharp 2012–2014 cliff, and Marmanet
receives $\alpha = 14.85$ despite a visible regime change. This shows how  rolling origin
cross-validation favors smoother curves.

:::{table} In-sample fit metrics comparing the smoothing parameter, AIC, and $R^2$ for each site.
:label: tbl:insample

<table>
<tr>
  <th>Site</th>
  <th>GAM α</th><th>GAM AIC</th><th>GAM R²</th>
  <th>GLM R²</th>
  <th>OLS AIC</th><th>OLS R²</th>
</tr>
<tr><td>EasternMau</td><td>0.226</td><td>-48.7</td><td>0.997</td><td>0.961</td><td>3.8</td><td>0.942</td></tr>
<tr><td>Eburu</td><td>0.0014</td><td>-51.9</td><td>0.999</td><td>0.812</td><td>31.7</td><td>0.805</td></tr>
<tr><td>Loita</td><td>0.112</td><td>-27.9</td><td>0.994</td><td>0.844</td><td>26.3</td><td>0.846</td></tr>
<tr><td>MaraConservancies</td><td>0.035</td><td>-91.3</td><td>1.000</td><td>0.863</td><td>15.2</td><td>0.907</td></tr>
<tr><td>Marmanet</td><td>14.850</td><td>-21.4</td><td>0.984</td><td>0.935</td><td>16.1</td><td>0.903</td></tr>
<tr><td>MountKenya</td><td>0.722</td><td>-62.4</td><td>0.998</td><td>0.946</td><td>1.4</td><td>0.947</td></tr>
<tr><td>MountLondiani</td><td>0.572</td><td>-66.3</td><td>0.998</td><td>0.932</td><td>2.0</td><td>0.946</td></tr>
<tr><td>Narok</td><td>47.508</td><td>-53.4</td><td>0.995</td><td>0.992</td><td>-28.7</td><td>0.985</td></tr>
<tr><td>Samburu</td><td>95.455</td><td>-21.5</td><td>0.982</td><td>0.952</td><td>-1.1</td><td>0.953</td></tr>
</table>
:::

### Site-Level Analysis

The nine sites span a wide range of trend shapes, and the sites with abrupt transitions are where the
model classes diverge most. Eburu declines sharply around 2010 and then stabilizes, so the linear
models continue a decline through a period when loss had effectively stopped. MaraConservancies is
the most extreme case, losing 22% of its remaining cover in the two years from 2012 to 2014. A
straight line cannot represent the stable phases on either side of that cliff. Loita and Marmanet
show a similar pattern, with rate changes in 2018 and 2011 respectively that the
linear models absorb into a single averaged slope. At the opposite end, Narok declines almost
linearly and Samburu loses less than 1% of coverage
over 24 years, and these are precisely the sites where the GAM's advantage narrows and its selected
$\alpha$ rises to 47.5 and 95.5 to produce a nearly straight fit.

:::{figure} figure2.png
:label: fig:sites
:align: center
:width: 100%
Fitted OLS, GLM, and GAM curves with 95% confidence intervals for each site. Generally declining
forest cover is observed across all measured sites. There are slight deviations from the linear
pattern in each, with more significant drops at Eburu 2010, Loita 2018, MaraConservancies 2013,
and Marmanet 2011.
:::

### Forecasting Performance

@tbl:forecast reports rolling origin forecasting metrics. GAM or GAMM achieves lower MAE than
OLS on seven of nine sites. The two exceptions are notable: at Samburu, OLS and GAMM are
essentially equivalent (92 vs 94 acres) given the near-linear trend; at MaraConservancies, OLS
achieves lower MAE (304 vs 406) because after the 2012–2014 cliff the post-break trend is
approximately linear.

The GLM falls between OLS and GAM on most sites. It improves on OLS where decline accelerates over
time, as at Marmanet and Narok, but cannot represent a discrete break: at Eburu its MAE is 312 acres
versus GAM 13. The proportional decline assumption fits gradual change but not abrupt dynamics.

The GAMM outperforms the per-site GAM on five of nine sites. Sites where GAM is superior —
Eburu, EasternMau, MountLondiani, and Narok — share highly individualized dynamics not well
represented by the global smooth trend. The GAMM's pooling slightly constrains these sites
relative to a fully independent fit, while benefiting the remaining five through information sharing.

:::{table} Rolling origin forecasting MAE (acres) for each of the nine forest cover sites.
:label: tbl:forecast

<table>
<tr>
  <th>Site</th>
  <th>GAMM MAE</th><th>GAM MAE</th><th>GLM MAE</th><th>OLS MAE</th>
</tr>
<tr><td>EasternMau</td><td>1534</td><td>1528</td><td>1713</td><td>2463</td></tr>
<tr><td>Eburu</td><td>30</td><td>13</td><td>312</td><td>320</td></tr>
<tr><td>Loita</td><td>180</td><td>323</td><td>342</td><td>341</td></tr>
<tr><td>MaraConservancies</td><td>379</td><td>406</td><td>502</td><td>304</td></tr>
<tr><td>Marmanet</td><td>213</td><td>441</td><td>858</td><td>1257</td></tr>
<tr><td>MountKenya</td><td>1422</td><td>1701</td><td>1476</td><td>1455</td></tr>
<tr><td>MountLondiani</td><td>2032</td><td>1834</td><td>2117</td><td>1922</td></tr>
<tr><td>Narok</td><td>5147</td><td>3730</td><td>4059</td><td>7478</td></tr>
<tr><td>Samburu</td><td>94</td><td>170</td><td>92</td><td>92</td></tr>
</table>
:::

## LLM-Based Hyperparameterization Experiment

With the increasing computing resources allocated to training large language models (LLMs),
there was a question of how they may serve in hyperparameterizing the aforementioned regression
classes. This section presents a preliminary empirical investigation into whether an LLM can
reduce the computational cost of GAM smoothing parameter selection by proposing a narrowed
$\alpha$ search range informed by ecological site characteristics.

The current pipeline selects $\alpha$ via a grid search of over 100 candidate values. This can be
computationally expensive at scale when fitting 100 GAMs per site across 800+ EarthRanger
deployments. The research question: can an LLM propose a credible and narrower search range
(threshold of 20 values) when fed site characteristics? For each of the nine sites, a prompt was
constructed describing quantitative characteristics computed from the data: site name, year range,
total forest loss, mean annual loss, variance of annual loss, and the presence and timing of a
regime change. Claude Sonnet 4, was selected as the model of choice due to its widespread use in
industry [@anthropic2025]. Rounds of full grid search, LLM-guided search, and a random control
group of 20 $\alpha$ values in a random range of equal log-width to the LLM suggestion were used
to generate a comparison table.

The LLM's proposed range contained the optimal $\alpha$ in 2 of 9 sites, compared to 3 of 9 for the
random control group. The LLM suggested $\alpha = 0.01$ for 8 of 9 sites regardless of the
site-specific characteristics. It did correctly identify low-$\alpha$ sites such as Eburu and
MaraConservancies, where regime changes are observed. The LLM failed on high-$\alpha$ sites such
as Marmanet ($\alpha = 14.85$), Narok ($\alpha = 47.5$), and Samburu ($\alpha = 95.5$).

:::{table} Optimal alpha values and LLM prediction accuracy for the GAM smoothing parameter.
:label: tbl:llm

<table>
<tr>
  <th>Site</th><th>Optimal α</th><th>LLM Guess</th><th>LLM Range</th><th>LLM Hit</th><th>Random Hit</th>
</tr>
<tr><td>EasternMau</td><td>0.226</td><td>0.001</td><td>[0.0001, 0.01]</td><td>N</td><td>N</td></tr>
<tr><td>Eburu</td><td>0.0014</td><td>0.001</td><td>[0.0001, 0.01]</td><td>Y</td><td>N</td></tr>
<tr><td>Loita</td><td>0.112</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>N</td></tr>
<tr><td>MaraConservancies</td><td>0.035</td><td>0.010</td><td>[0.001, 0.1]</td><td>Y</td><td>Y</td></tr>
<tr><td>Marmanet</td><td>14.850</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>N</td></tr>
<tr><td>MountKenya</td><td>0.722</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>N</td></tr>
<tr><td>MountLondiani</td><td>0.572</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>N</td></tr>
<tr><td>Narok</td><td>47.508</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>Y</td></tr>
<tr><td>Samburu</td><td>95.455</td><td>0.010</td><td>[0.001, 0.1]</td><td>N</td><td>Y</td></tr>
</table>
:::

Overall, the context window of LLMs is not currently suited for hyperparameterization when
compared to the brute force grid search implemented in Ecoscope. There is limited current
reasoning about the nonlinear dynamics of each site as well as model awareness of regime changes
in the data. The LLM fails to represent the relationship between the ecological site factors and the
smoothing penalty under cross-validation. For example, Marmanet has a visible drop in the data
where the LLM interprets a low $\alpha$, but evaluation determines a smooth curve can be used
across the whole time range.

This represents some of the first applications of LLM-based hyperparameter search to
domain-specific statistical models in the conservation field, and defines concrete gaps in current
LLM context windows. A multi-shot prompt is worth exploring with labeled examples of validated
$\alpha$ parameters as a way to teach the model nuanced selection. Applying Bayesian optimization
as a preprocessing tool may also reduce experimental error.

## Discussion

### Predicting Regime Changes

GAMs reliably detect and represent abrupt regime changes that linear models cannot.
MaraConservancies, Eburu, Loita, Marmanet, and MountLondiani all exhibit distinct deforestation
phases separated by sudden transitions. OLS and GLM produce a single averaged rate that
misrepresents both the stable and rapidly changing phases, while the GAM correctly fits each
phase independently. A ranger using OLS at MaraConservancies in 2011 would have forecast
gradual continued loss and missed the 35% collapse that followed.

The GAMM can add a further benefit by learning shared change patterns across the nine sites.
Areas undergoing similar transitions in the same period benefit from pooled information, improving
estimates particularly where local data is sparse around the transition. This is a unique case of
model improvement and cannot be applied to all nine sites evenly, which is why the per-site
regression curves are also displayed to the researcher.

### Use Cases for Ecological Prediction

The forest cover results demonstrate that the GAM/GAMM framework is well suited to generalized
ecological time series due to its handling of nonlinear and unpredictable dynamics. The GAMM is
particularly valuable in regional analyses where adjacent sites share ecological trends, as
demonstrated by the five sites where cross-site pooling improved forecasting performance over the
per-site GAM.

### LLM vs Grid Search for Alpha Selection

The LLM initialization experiment shows that Claude Sonnet does not outperform random search
for $\alpha$ selection on the nine forest cover sites. The model suggested $\alpha = 0.01$ for eight of
nine sites regardless of input characteristics, correctly identifying low-$\alpha$ sites with strong
regime changes but failing entirely on smooth high-$\alpha$ sites including Marmanet, Narok, and
Samburu. The current grid search implementation remains the appropriate default. The gap in
ecological intuition provides a concrete target for future work via few-shot prompting or Bayesian
optimization.

## Conclusion

This work presents a Python-based trend fitting framework for wildlife conservation practitioners,
implemented within the Ecoscope library and evaluated on nine Kenyan forest monitoring sites
spanning 2001–2024. The framework operationalizes four regression models — OLS, GLM, GAM,
and GAMM — each relaxing one assumption from its predecessor. Pre-processing is carried out so
that practitioners interact only with raw data and interpretable outputs.

The primary empirical finding is that the GAM or GAMM outperforms OLS on seven of nine sites for
rolling origin forecasting, with the advantage largest at sites exhibiting abrupt regime changes.
The GAMM with per-site random intercepts outperforms the per-site GAM on five of nine sites by
leveraging shared trend structure across the monitoring network. Sites with highly individual
dynamics retain a per-site GAM advantage. A secondary case study on a year of hourly elephant
movement telemetry shows that the same pipeline transfers without modification to a far denser and
noisier series, recovering a stable nonlinear trend where the linear model reports no change.

The LLM hyperparameter initialization experiment yields a negative result that is informative for
future work. Current frontier models do not encode the relationship between ecological site
characteristics and optimal smoothing parameters well. This defines a concrete research gap and
motivates few-shot and Bayesian approaches as future directions.

Together these contributions address the growing need for widespread trend analysis tooling in
ecological spaces — providing conservation organizations with statistical tools that are powerful
enough to capture genuine environmental complexity, robust enough to perform reliably on sparse
and irregular data, and accessible enough to deploy without requiring advanced statistical
expertise. The implementation within Ecoscope ensures that the framework is available to the
more than 800 EarthRanger sites currently operating worldwide.

## Code

The full implementation is available across three public repositories:

- **Ecoscope library:** <https://github.com/wildlife-dynamics/ecoscope>
- **Trend fitting module:** <https://github.com/wildlife-dynamics/ecoscope/blob/master/ecoscope/analysis/trend_analysis.py>
- **Forest cover workflow:** <https://github.com/wildlife-dynamics/wt-hansen-deforestation>
