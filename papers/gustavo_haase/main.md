---
# Ensure that this title is the same as the one in `myst.yml`
title: "PanelBox: A Comprehensive Python Library for Panel Data Econometrics"
abstract: |
  Panel data econometrics is ubiquitous in economic and social-science
  research, yet Python has lacked a comprehensive library for advanced panel
  analysis: researchers have had to fall back on proprietary Stata or on R,
  breaking otherwise Python-native, reproducible workflows. PanelBox grew out
  of our model-validation practice in credit risk, where econometric models
  running in production Python systems had to be exported to Stata or R for
  estimation and validation. We introduce
  PanelBox, the first all-in-one Python library for panel data econometrics,
  implementing more than 70 models across 11 families — static linear models,
  dynamic GMM estimators (Arellano–Bond, Blundell–Bond with the Windmeijer
  correction), discrete choice, count data, spatial econometrics, stochastic
  frontier analysis, quantile regression, panel VAR/VECM, censored and
  selection models, and the first comprehensive Python implementation of panel
  unit-root and cointegration tests. The library adds 50+ diagnostic tests,
  11 standard-error estimators, four bootstrap methods, an integrated
  HTML/LaTeX/Markdown reporting system with interactive Plotly visualizations,
  and 103 bundled datasets, all on top of the scientific Python stack
  (NumPy, pandas, SciPy, statsmodels). We validate PanelBox against Stata's
  `xtabond2` and R's `plm`, obtaining coefficient differences below 0.01%, and
  illustrate it with a canonical dynamic labor-demand application. With 3,900+
  tests and an MIT license, PanelBox lets researchers run sophisticated panel
  analysis entirely in Python.
---

## Introduction

PanelBox did not begin as a general-purpose econometrics library. We work in
model validation for credit risk, where econometric models estimated on panel
data must be independently re-estimated, stress-tested, and documented before
they are allowed into production. Our production systems run on Python; the
reference implementations of the methods we needed did not. Every validation
cycle meant exporting data to Stata or R, re-importing results, and
reconciling discrepancies by hand — slow, error-prone, and impossible to
automate end to end. What started as an internal toolkit for validating panel
models in pure Python grew, model family by model family, into PanelBox.

The data at the center of this workflow are *panel data* — repeated
observations on the same entities over time, such as firms observed annually
or countries tracked across decades. Panels are the empirical backbone of
modern economics and the social sciences [@baltagi2021; @wooldridge2010], and
for a simple reason: every firm or country has stable characteristics that no
dataset records — management quality, institutions, culture. A cross-sectional
regression silently attributes the effect of these unobserved traits to the
variables it does observe, biasing its conclusions. Because a panel follows
each entity over time, it can compare each entity with itself, removing the
influence of anything constant — the *unobserved heterogeneity* that panel
methods are built to neutralize. Combined with the time dimension, this lets
researchers estimate genuinely dynamic relationships, from labor-market
adjustment [@arellano1991] to production functions [@blundell1998] and
cross-country growth [@bond2001].

The gap we encountered, it turned out, is far wider than our use case: the
tooling for these methods is unevenly distributed across computing
environments. Stata's `xtabond2` command [@roodman2009stata] is the
de facto standard for dynamic panel GMM, and R offers the mature `plm`
package [@croissant2008plm] for linear panels and specialized packages such as
`pdynmc` [@fritsch2019pdynmc] for dynamic ones. Python — now dominant in data
science and increasingly in quantitative social science thanks to
NumPy [@harris2020numpy], pandas [@mckinney2011pandas], and
SciPy [@scipy] — has lagged conspicuously behind in panel econometrics.

Existing Python libraries cover only fragments of the workflow.
`linearmodels` [@linearmodels2019] provides static fixed- and random-effects
models — the workhorse estimators that remove, or explicitly model, the
stable entity traits discussed above — and some support for instrumental
variables (proxy regressors used in place of a variable that is correlated
with the error term), but no dynamic GMM, the estimator required when the
outcome depends on its own past, introduced in detail below. `pyfixest`
excels at high-dimensional fixed effects but offers neither GMM nor dynamic
panels; and `statsmodels` [@seabold2010statsmodels] has time-series tools but
no panel-specific GMM. Critically, none of them implement panel unit-root or
cointegration tests, which are routine in Stata (`xtunitroot`, `xtcointtest`)
and R (`plm::purtest`). The consequence is a broken workflow: analysts either
limit themselves to basic models in Python, or export their data to Stata or
R, forfeiting reproducibility and integration with the rest of the Python
stack.

PanelBox closes this gap. Version 1.0 implements more than 70 models across 11
families in a single package, validated numerically against established
software. Beyond breadth, it contributes three things that are new to Python:
a validated dynamic-GMM implementation with the Windmeijer finite-sample
correction; the first comprehensive panel unit-root and cointegration testing
suite; and an intelligent algorithm for unbalanced panels that retains far
more data than naive complete-case approaches. The library is open source
(MIT), integrates natively with pandas, and ships with interactive
visualization, publication-ready reporting, and 103 datasets for immediate
experimentation. The paper follows the workflow a researcher actually
traverses — specify and estimate a model, probe its assumptions, and report
the results — with the depth reserved for the three novel contributions;
@sec:breadth summarizes the full catalogue, and @sec:using describes how the
library is used and extended in practice.

## Library design: one workflow from data to report

PanelBox is organized around the estimation workflow rather than around a
catalogue of methods. The user starts from a pandas `DataFrame` in long
format; a common data layer wraps it together with entity and time
identifiers, and everything downstream consumes that layer. Estimators accept
R-style formulas via `patsy` and return a result object following the
`statsmodels` conventions — `summary()`, `conf_int()`, `to_latex()`,
`to_html()` — so that the same result flows into the shared inference layer
(11 standard-error types, four bootstrap methods), the diagnostics layer (50+
tests), and the reporting layer that renders self-contained HTML, LaTeX, or
Markdown. @fig:architecture summarizes this pipeline.

:::{figure} figures/fig1_architecture.png
:label: fig:architecture
PanelBox as a dataflow. A long-format pandas `DataFrame` enters the shared
data layer, flows through the 11 estimator families — with the inference
engine supplying standard errors and bootstraps at `fit()` time — and the
resulting objects feed the diagnostics and reporting layers. Solid colored
boxes are the user-facing API; dashed gray boxes are internal infrastructure.
:::

Because the API follows conventions the community already knows —
`statsmodels`-style formulas and result objects, with an sklearn-like
construct-then-fit pattern — a minimal estimation holds no surprises:

```python
import panelbox as pb

data = pb.load_dataset("arellano_bond_employment")
fe = pb.FixedEffects(
    data, formula="n ~ w + k",
    entity_var="firm_id", time_var="year",
)
result = fe.fit(cov_type="clustered", cluster="entity")
print(result.summary())
```

The workflow's entry point is the static linear family, which implements
pooled OLS, fixed effects (within), random effects (feasible GLS), the
between estimator, and first differences for the model

```{math}
:label: static
y_{it} = \mathbf{x}_{it}'\boldsymbol{\beta} + \alpha_i + \lambda_t + \epsilon_{it},
\qquad i = 1,\dots,N,\; t = 1,\dots,T,
```

where $\alpha_i$ are entity effects — absorbing everything specific to entity
$i$ that is constant over time, such as a firm's management quality — and
$\lambda_t$ are time effects, absorbing shocks common to all entities in
period $t$, such as a recession year. Fixed
effects are computed by within-transformation using sparse matrix operations
and pandas `groupby`, so the estimator scales to large $N$ and supports
unbalanced panels with entity-specific $T_i$.

## Estimating dynamics: panel GMM

Many economic outcomes depend on their own past. Employment adjusts slowly
because hiring and firing are costly; this year's GDP is anchored to last
year's. Capturing such persistence means adding the lagged outcome
$y_{i,t-1}$ as a regressor:

```{math}
:label: dynamic
y_{it} = \alpha\, y_{i,t-1} + \mathbf{x}_{it}'\boldsymbol{\beta} + \eta_i + \epsilon_{it}.
```

That innocent-looking addition breaks both estimators of the previous
section. The lagged outcome is itself driven by the entity effect $\eta_i$, so
it is correlated with the composite error: pooled OLS is biased upward, and
the within estimator downward (the Nickell bias, @nickell1981) — the true
dynamics lie somewhere in between, and neither estimator can find them. The
resolution, due to @arellano1991 and @blundell1998, is to transform away
$\eta_i$ and instrument the problematic regressor with its own older lags:
old enough to be uncorrelated with today's shock, recent enough to remain
informative. This machinery — dynamic panel GMM — is PanelBox's flagship
capability, and previously had no validated Python implementation.

**Difference GMM** [@arellano1991] first-differences {ref}`dynamic` to remove
$\eta_i$ and exploits the moment conditions
$\mathrm{E}[y_{i,t-s}\,\Delta\epsilon_{it}] = 0$ for $s \ge 2$, so lagged levels
instrument the differenced regressors. **System GMM** [@blundell1998] augments
the differenced equations with the level equations, instrumenting them with
lagged differences; this restores efficiency and reduces finite-sample bias
when $\alpha$ is close to unity and lagged levels are weak instruments.
PanelBox implements both, in one-step and two-step variants — one-step GMM
weights the moment conditions with a fixed matrix, while two-step re-estimates
the weights from the first step's residuals to gain asymptotic efficiency —
and generates the instrument sets automatically:

```python
gmm = pb.SystemGMM(
    data, dep_var="n", lags=1,
    id_var="firm_id", time_var="year",
    exog_vars=["w", "k"],
    collapse=True, two_step=True,
)
res = gmm.fit()
print(f"Hansen J p-value: {res.hansen_j.pvalue:.3f}")
print(f"AR(2) p-value:    {res.ar2_test.pvalue:.3f}")
```

**Windmeijer correction.** Two-step GMM is asymptotically efficient but its
standard errors are severely biased downward in finite samples — by 30–80% in
typical applications. @windmeijer2005 derives a finite-sample correction that
accounts for the estimated weight matrix; PanelBox applies it by default,
following the algorithm in @roodman2009stata.

**Instrument proliferation.** Because the instrument count grows quadratically
in $T$, it can exceed $N$ and overfit the endogenous regressors
[@roodman2009hansen]. PanelBox warns automatically when the instrument count
approaches $N$ and implements the `collapse` option, which combines the
per-period instruments for each variable into a single column while preserving
the moment conditions.

### Intelligent handling of unbalanced panels

Missing observations are a recurring practical obstacle in dynamic GMM: because
each instrument depends on a specific lag (e.g. $y_{i,t-2}$ instruments
$\Delta y_{it}$), a single gap can invalidate instruments. Existing tools
handle this bluntly — `xtabond2` silently discards observations with missing
instruments, and many practitioners drop every entity with any missing value,
causing severe sample loss.

PanelBox instead validates instruments observation-by-observation. For each
entity $i$ and period $t$ it (1) identifies the required instrument lags, (2)
checks their availability, (3) builds an entity-specific instrument set from
whatever lags exist, and (4) assembles the block-diagonal instrument matrix
$\mathbf{Z}$ from variable-sized blocks (@fig:unbalanced). An observation is
kept whenever valid instruments exist, rather than discarding the whole
entity. On the Arellano–Bond employment data this retains 72% of observations,
against roughly 40% for `xtabond2` and 0% for naive complete-case analysis,
shrinking standard errors without compromising instrument validity.

:::{figure} figures/fig2_unbalanced_algorithm.png
:label: fig:unbalanced
The unbalanced-panel algorithm: instruments are validated per observation and
entity-specific instrument sets are assembled into a block-diagonal matrix,
retaining observations that fixed-template approaches discard.
:::

## Testing assumptions: unit roots and cointegration

Estimation is only the middle of the workflow; its validity rests on
assumptions that must themselves be tested. For macroeconomic panels the
first question is whether each variable has a stable long-run level or
wanders without one (a *unit root*): regressing one wandering series on
another produces convincing-looking but spurious correlations, so testing for
unit roots — and, when variables do wander, for a genuine shared long-run
relationship (*cointegration*) — is the standard safety check before
estimating relationships among variables such as GDP, prices, or exchange
rates. This capability is unique to PanelBox among Python panel libraries,
despite being standard in macroeconomics and finance.

Three panel unit-root tests build on the entity-specific augmented
Dickey–Fuller regression, testing $H_0: \rho_i = 0$. The LLC test
[@levin2002] assumes a common $\rho$ and uses a pooled adjusted $t$-statistic;
the IPS test [@improsan2003] allows heterogeneous $\rho_i$ and averages the
individual $t$-statistics into an asymptotically normal $W$; and Fisher-type
tests [@maddala1999] combine the per-entity $p$-values. For $I(1)$ variables,
the Pedroni [@pedroni1999; @pedroni2004] and Kao [@kao1999] tests assess
cointegration; Pedroni reports seven statistics and resolves them by majority
vote.

```python
from panelbox.validation.unit_root import LLCTest, IPSTest

llc = LLCTest(data, variable="log_gdp",
              entity_var="country", time_var="year", trend="ct")
print(llc.test().pvalue)
```

Together these enable a complete workflow — test for unit roots, test for
cointegration when variables are $I(1)$, then estimate the appropriate model —
illustrated in the growth application of @sec:application.

(sec:breadth)=
## Breadth: eleven model families

Linear panels are only part of applied practice: economic data routinely
involve binary outcomes, counts, censoring, spatial dependence, and
efficiency frontiers. PanelBox covers these in the same interface;
@tbl:families maps each family to its closest Stata and R equivalents, and we
highlight below only the capabilities not available elsewhere in Python. The
complete catalogue, with an executable example notebook per family, is in the
official documentation.

```{list-table} The 11 model families in PanelBox and their closest equivalents in Stata and R. A dash means no established equivalent.
:label: tbl:families
:header-rows: 1
* - Family
  - Representative estimators in PanelBox
  - Stata
  - R
* - Static linear
  - Pooled OLS, FE, RE, between, first differences
  - `xtreg`
  - `plm`
* - Dynamic GMM
  - Difference/system GMM, Windmeijer correction, Anderson–Hsiao, LSDVC
  - `xtabond2`
  - `plm::pgmm`, `pdynmc`
* - Unit-root & cointegration tests
  - LLC, IPS, Fisher; Pedroni, Kao
  - `xtunitroot`, `xtcointtest`
  - `plm::purtest`
* - Discrete choice
  - Pooled/FE/RE logit & probit, multinomial, ordered, dynamic binary
  - `xtlogit`, `xtprobit`
  - `bife`, `pglm`
* - Count data
  - Poisson (pooled/FE/QML), negative binomial, zero-inflated, PPML
  - `xtpoisson`, `ppmlhdfe`
  - `pglm`, `fixest`
* - Censored & selection
  - Panel tobit, sample-selection models
  - `xttobit`, `xtheckman`
  - `censReg`
* - Quantile regression
  - Pooled, FE, Canay, Machado–Santos Silva
  - `xtqreg`
  - `rqpd`
* - Spatial panels
  - SAR, SEM, SDM, GNS, dynamic spatial
  - `spxtregress`
  - `splm`
* - Stochastic frontier
  - Battese–Coelli, Greene TFE/TRE, four-component
  - `sfpanel`
  - `frontier`
* - Panel VAR/VECM
  - IRFs, FEVD, Granger causality
  - `pvar`
  - `panelvar`
* - Heterogeneous panels
  - Mean-group estimators, SUR
  - `xtmg`, `sureg`
  - `plm::pmg`, `systemfit`
```

Several entries go beyond their Stata/R counterparts. The discrete-choice
family includes dynamic binary models with the @wooldridge2005
initial-conditions correction and conditional maximum-likelihood fixed
effects [@chamberlain1980], with average and representative-value marginal
effects computed with delta-method standard errors; the count family includes
the PPML estimator [@santossilva2006] widely used for gravity models. The
spatial family estimates five models by quasi-maximum likelihood [@leeyu2010]
and decomposes results into direct and indirect (spillover) effects via the
spatial multiplier $(\mathbf{I}_N - \rho\mathbf{W})^{-1}\beta_k$. The frontier
family implements the @kumbhakar2014 four-component model separating
*persistent* from *transient* inefficiency [@battesecoelli1992;
@battesecoelli1995; @greene2005], and the quantile family includes the
@machadosantossilva2019 location-scale estimator, which guarantees
non-crossing quantiles [@koenker2004; @canay2011] — all three previously
unavailable in Python.

## Inference, diagnostics, and reporting

Once a model is estimated, the workflow turns to defending it. All standard
errors use the sandwich form
$\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) =
(\mathbf{X}'\mathbf{X})^{-1}\,\hat{\boldsymbol{\Omega}}\,(\mathbf{X}'\mathbf{X})^{-1}$,
differing in the inner "meat" matrix $\hat{\boldsymbol{\Omega}}$ (the term of
art for the filling of the sandwich form): HC0–HC3 for
heteroskedasticity, one-way and two-way clustering, and the Driscoll–Kraay and
Newey–West HAC estimators. Four bootstraps — pairs, wild, block, and
residual — cover small-sample and non-standard cases.

The diagnostic suite includes the Hausman test [@hausman1978] comparing FE and
RE, and the GMM trio that governs dynamic-panel validity: the Hansen $J$ test
of overidentifying restrictions [@hansen1982], and the Arellano–Bond AR(1) and
AR(2) tests for serial correlation, where AR(1) is expected to reject and AR(2)
to not reject.

Result objects render publication-ready output. `comparison_table` aligns
several models side by side, and the reporting layer produces self-contained
HTML with embedded interactive Plotly charts (28+ chart types across residual
diagnostics, model comparison, and econometric tests) [@plotly], or static
LaTeX/Markdown:

```python
pb.comparison_table(
    [ols_result, fe_result, diffgmm_result, sysgmm_result],
    labels=["Pooled OLS", "FE", "Diff-GMM", "Sys-GMM"],
)
```

(sec:using)=
## Using and extending PanelBox

PanelBox makes no assumptions about where data come from. Any long-format
pandas `DataFrame` with an entity column and a time column works directly —
loaded from CSV, Parquet, SQL, or an API via the usual pandas readers — and
unbalanced panels are handled natively, so no reshaping or gap-filling is
required. The 103 bundled datasets are conveniences for teaching,
benchmarking, and replication, not a requirement.

All estimators share one contract: construct with the data and a
specification, call `fit()`, receive a results object with the same methods
everywhere. Comparing a fixed-effects model with a system-GMM model is a
one-line change of class, and `comparison_table` aligns any set of results.
Within this contract, the supported customization points are the covariance
estimator and bootstrap scheme (per `fit()` call), the instrument design for
GMM (`gmm_lags`, `collapse`), arbitrary variable transformations through
`patsy` formulas, marginal-effects options for nonlinear models, and the
output format of every report (HTML, LaTeX, Markdown).

Two honest boundaries: there is currently no public API for user-defined
moment conditions or custom estimators — internally all models subclass a
common base model and results class, and a documented, stable extension
interface is on the roadmap — and GMM is restricted to linear dynamic panels.
The official documentation includes a gallery of executable Jupyter notebooks
covering every model family end to end, which is the recommended starting
point for adapting the library to a new use case.

(sec:application)=
## Application: dynamic labor demand

To illustrate the capabilities of the library, we use it to reproduce the
canonical application of @arellano1991: a dynamic labor-demand
equation on UK firm-level data, a balanced panel of $N = 140$ firms over
$T = 9$ years (1979–1987). The model

```{math}
:label: labor
n_{it} = \alpha\, n_{i,t-1} + \beta_w w_{it} + \beta_k k_{it}
         + \eta_i + \lambda_t + \epsilon_{it}
```

relates log employment $n_{it}$ to log real wages $w_{it}$ and log capital
$k_{it}$; $\alpha$ captures employment adjustment costs. We estimate it four
ways to expose the bias–consistency trade-off:

```python
ols = pb.PooledOLS(data, formula="n ~ L(n,1) + w + k").fit()
fe = pb.FixedEffects(
    data, formula="n ~ L(n,1) + w + k",
    entity_var="firm_id", time_var="year",
).fit(cov_type="clustered", cluster="entity")
diffgmm = pb.DifferenceGMM(
    data, dep_var="n", lags=1, id_var="firm_id", time_var="year",
    exog_vars=["w", "k"], gmm_lags=[2, 9], two_step=True,
).fit()
sysgmm = pb.SystemGMM(
    data, dep_var="n", lags=1, id_var="firm_id", time_var="year",
    exog_vars=["w", "k"], collapse=True, two_step=True,
).fit()
```

The estimates trace the textbook pattern (@fig:coefficients). Pooled OLS yields
$\hat{\alpha} = 0.937$, biased upward; fixed effects gives $\hat{\alpha} =
0.548$, biased downward by the Nickell bias; the GMM estimators fall in between,
with system GMM at $\hat{\alpha} = 0.694$ (standard error 0.137) versus
difference GMM at $0.686$ (0.152), confirming the efficiency gain from the
additional moment conditions. The wage elasticity is consistently negative
($\hat{\beta}_w \approx -0.57$) and capital enters positively
($\hat{\beta}_k \approx 0.38$). Diagnostics validate the specification: the
Hansen $J$ test does not reject ($p = 0.172$), AR(1) rejects as expected
($p = 0.003$), AR(2) does not ($p = 0.671$), and the 91 instruments stay well
below the 140 entities. The results match @arellano1991 to within 0.01%.

:::{figure} figures/fig5_coefficient_comparison.png
:label: fig:coefficients
Coefficient estimates across pooled OLS, fixed effects, difference GMM, and
system GMM for the labor-demand model, illustrating the dynamic-panel bias
pattern: OLS biased up, FE biased down, GMM in between.
:::

The same building blocks support a full nonstationary-panel workflow on the
Penn World Table: LLC and IPS tests fail to reject a unit root in log GDP
(both $I(1)$), Pedroni's test then rejects no-cointegration on six of seven
statistics, and a system-GMM growth regression returns a significantly negative
coefficient on lagged log GDP ($\hat{\beta}_1 = -0.034$, SE $0.008$),
confirming conditional convergence at about 3.4% per year — consistent with the
growth literature [@mankiw1992].

## Validation and performance

We validate PanelBox against Stata 18's `xtabond2` [@roodman2009stata] and R's
`plm` [@croissant2008plm], estimating identical specifications and computing
relative differences $\Delta = |\text{PanelBox} - \text{reference}| /
|\text{reference}|$. For the Arellano–Bond system-GMM specification, every
coefficient and standard error matches `xtabond2` to four decimal places, and
the Hansen $J$, AR(1), and AR(2) statistics are identical (@tbl:validation).
Across 25 specifications on 10 datasets — difference and system GMM, one- and
two-step, with and without collapse, balanced and unbalanced — the mean
coefficient difference is 0.0003% and the maximum 0.008%. Against `plm`, fixed-
and random-effects coefficients, the Hausman statistic, and LLC/IPS unit-root
statistics agree to machine precision: these estimators are closed-form,
deterministic linear algebra, so two correct implementations differ only in
floating-point rounding, below any reported digit.

```{list-table} Numerical validation against Stata xtabond2, system GMM on the Arellano–Bond employment data. Standard errors and p-values in parentheses.
:label: tbl:validation
:header-rows: 1
* - Quantity
  - Stata
  - PanelBox
  - $\Delta$ (%)
* - `L.n` coef. (s.e.)
  - 0.6861 (0.1366)
  - 0.6861 (0.1366)
  - 0.00
* - `w` coef. (s.e.)
  - −0.5685 (0.1398)
  - −0.5685 (0.1398)
  - 0.00
* - `k` coef. (s.e.)
  - 0.3838 (0.0619)
  - 0.3838 (0.0619)
  - 0.00
* - Hansen J
  - 93.56 (0.172)
  - 93.56 (0.172)
  - 0.00
* - AR(1)
  - −2.93 (0.003)
  - −2.93 (0.003)
  - 0.00
* - AR(2)
  - −0.42 (0.671)
  - −0.42 (0.671)
  - 0.00
```

PanelBox is implemented in pure Python on NumPy/SciPy yet remains within
20–30% of Stata's compiled `Mata` for system-GMM estimation and runs roughly
3–4× faster than R's `plm` (@fig:performance). For typical panels
($N \approx 500$, $T \approx 10$) estimation completes in 1–2 seconds; memory
scales linearly in $NT$ thanks to sparse matrices, and bootstrap inference
parallelizes with near-linear speedup.

:::{figure} figures/fig8_performance.png
:label: fig:performance
System-GMM execution time across panel sizes. PanelBox tracks Stata closely and
substantially outperforms R's `plm`.
:::

## Conclusion

PanelBox brings comprehensive panel data econometrics to Python: 70+ models
across 11 families, 50+ diagnostic tests, 11 standard-error estimators, an
interactive visualization and reporting system, and 103 bundled datasets, in a
single MIT-licensed package built on the scientific Python stack. Its dynamic
GMM is validated against `xtabond2` to within 0.01% and adds an unbalanced-panel
algorithm that retains 72% of observations where existing tools retain far
fewer, and it offers capabilities — panel unit-root and cointegration tests,
four-component stochastic frontiers, non-crossing quantile regression, panel
VAR/VECM — previously unavailable in Python. The unusual emphasis on numerical
validation throughout this paper is not incidental: it reflects the library's
origin in production model validation, where matching the reference
implementation is the requirement, not a nicety.

Current limitations include spatial-model scalability beyond $N \approx 2{,}000$
(mitigated by Chebyshev approximation), GMM restricted to linear dynamic panels,
and no GPU acceleration. Planned work includes FMOLS/DOLS for cointegrated
panels, double/debiased machine learning for panel data, GPU acceleration via
JAX/CuPy, and Bayesian panel models. By eliminating the need to leave Python for
sophisticated panel analysis, PanelBox makes empirical economic research more
reproducible, more integrated with the machine-learning ecosystem, and more
accessible. It is available on PyPI (`pip install panelbox`) and on GitHub at
<https://github.com/PanelBox-Econometrics-Model/panelbox> [@panelbox2025],
with documentation, tutorials, example notebooks for every model family, and
full replication materials for this paper.

## Generative AI disclosure

Portions of this work were assisted using generative AI tools, specifically
Anthropic's Claude. The tools were used to help draft and refine the prose of
the manuscript, to improve clarity and phrasing, to suggest edits to the
LaTeX/MyST markup, and to suggest code for parts of the PanelBox library and
its replication scripts. Generative AI was **not** used to produce the
numerical results, statistical validations (e.g., the comparisons against
`xtabond2` and `plm`), or references reported here: these derive from actual
executions of the software and were checked against the reference tools. All
outputs were reviewed, verified, and revised by the author, who takes full
responsibility for the accuracy and integrity of the final content, including
all technical claims, results, and references.
