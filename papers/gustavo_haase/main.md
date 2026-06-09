---
# Ensure that this title is the same as the one in `myst.yml`
title: "PanelBox: A Comprehensive Python Library for Panel Data Econometrics"
abstract: |
  Panel data econometrics is ubiquitous in economic and social-science
  research, yet Python has lacked a comprehensive library for advanced panel
  analysis: researchers have had to fall back on proprietary Stata or on R,
  breaking otherwise Python-native, reproducible workflows. We introduce
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
  illustrate it with a canonical dynamic labor-demand application. With 3{,}900+
  tests and an MIT license, PanelBox lets researchers run sophisticated panel
  analysis entirely in Python.
---

## Introduction

Panel data — repeated observations on the same entities over time — is the
empirical backbone of modern economics and the social sciences
[@baltagi2021; @wooldridge2010]. By combining cross-sectional and temporal
variation, panel methods let researchers control for unobserved heterogeneity
while estimating dynamic relationships, from labor-market adjustment
[@arellano1991] to production functions [@blundell1998] and cross-country
growth [@bond2001].

The tooling for these methods, however, is unevenly distributed across
computing environments. Stata's `xtabond2` command [@roodman2009stata] is the
de facto standard for dynamic panel GMM, and R offers the mature `plm`
package [@croissant2008plm] for linear panels and specialized packages such as
`pdynmc` [@fritsch2019pdynmc] for dynamic ones. Python — now dominant in data
science and increasingly in quantitative social science thanks to
NumPy [@harris2020numpy], pandas [@mckinney2011pandas], and
SciPy [@scipy] — has lagged conspicuously behind in panel econometrics.

Existing Python libraries cover only fragments of the workflow.
`linearmodels` [@linearmodels2019] provides static fixed- and random-effects
models and some instrumental-variables support but no dynamic GMM; `pyfixest`
excels at high-dimensional fixed effects but offers neither GMM nor dynamic
panels; and `statsmodels` [@seabold2010statsmodels] has time-series tools but
no panel-specific GMM. Critically, *none* of them implement panel unit-root or
cointegration tests — routine in Stata (`xtunitroot`, `xtcointtest`) and R
(`plm::purtest`). The consequence is a broken workflow: analysts either limit
themselves to basic models in Python, or export their data to Stata or R,
forfeiting reproducibility and integration with the rest of the Python stack.

PanelBox closes this gap. Version 1.0 implements more than 70 models across 11
families in a single package, validated numerically against established
software. Beyond breadth, it contributes three things that are new to Python:
a validated dynamic-GMM implementation with the Windmeijer finite-sample
correction; the first comprehensive panel unit-root and cointegration testing
suite; and an intelligent algorithm for unbalanced panels that retains far more
data than naive complete-case approaches. The library is open source (MIT),
integrates natively with pandas, and ships with interactive visualization,
publication-ready reporting, and 103 datasets for immediate experimentation.

## Library overview and design

@fig:architecture summarizes the architecture. A common data layer wraps a
pandas `DataFrame` together with entity and time identifiers; every estimator
consumes this layer, supports R-style formulas via `patsy`, and returns a
result object following the `statsmodels` convention, with `summary()`,
`conf_int()`, `to_latex()`, and `to_html()` methods. On top sit the 11 model
families, a shared inference layer (11 standard-error types, four bootstrap
methods), a diagnostics layer (50+ tests), and a reporting layer that renders
self-contained HTML, LaTeX, or Markdown.

:::{figure} figures/fig1_architecture.png
:label: fig:architecture
High-level architecture of PanelBox: the 11 model families build on a shared
data layer and feed a common inference, diagnostics, visualization, and
reporting infrastructure.
:::

A minimal estimation looks like idiomatic scientific Python:

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

The static family implements pooled OLS, fixed effects (within), random effects
(feasible GLS), the between estimator, and first differences for the model

```{math}
:label: static
y_{it} = \mathbf{x}_{it}'\boldsymbol{\beta} + \alpha_i + \lambda_t + \epsilon_{it},
\qquad i = 1,\dots,N,\; t = 1,\dots,T,
```

where $\alpha_i$ are entity effects and $\lambda_t$ are time effects. Fixed
effects are computed by within-transformation using sparse matrix operations
and pandas `groupby`, so the estimator scales to large $N$ and supports
unbalanced panels with entity-specific $T_i$.

## Dynamic panel GMM

The flagship capability is dynamic panel GMM. With a lagged dependent variable,

```{math}
:label: dynamic
y_{it} = \alpha\, y_{i,t-1} + \mathbf{x}_{it}'\boldsymbol{\beta} + \eta_i + \epsilon_{it},
```

both OLS and fixed effects are inconsistent, because $y_{i,t-1}$ is correlated
with the composite error: OLS is biased upward and the within estimator
downward (the Nickell bias, @nickell1981). The GMM framework of @arellano1991
and @blundell1998 resolves this by transforming away $\eta_i$ and using lagged
values as instruments.

**Difference GMM** [@arellano1991] first-differences {ref}`dynamic` to remove
$\eta_i$ and exploits the moment conditions
$\mathrm{E}[y_{i,t-s}\,\Delta\epsilon_{it}] = 0$ for $s \ge 2$, so lagged levels
instrument the differenced regressors. **System GMM** [@blundell1998] augments
the differenced equations with the level equations, instrumenting them with
lagged differences; this restores efficiency and reduces finite-sample bias
when $\alpha$ is close to unity and lagged levels are weak instruments.
PanelBox implements both, in one-step and two-step variants, and generates the
instrument sets automatically:

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
kept whenever *valid instruments exist*, rather than discarding the whole
entity. On the Arellano–Bond employment data this retains 72% of observations,
against roughly 40% for `xtabond2` and 0% for naive complete-case analysis,
shrinking standard errors without compromising instrument validity.

:::{figure} figures/fig2_unbalanced_algorithm.png
:label: fig:unbalanced
The unbalanced-panel algorithm: instruments are validated per observation and
entity-specific instrument sets are assembled into a block-diagonal matrix,
retaining observations that fixed-template approaches discard.
:::

## Panel unit-root and cointegration tests

A capability unique to PanelBox among Python panel libraries is comprehensive
unit-root and cointegration testing — standard in macroeconomics and finance
but previously absent from the ecosystem.

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
illustrated in the growth application below.

## Breadth: nonlinear, spatial, and time-series families

While linear models cover many applications, economic data routinely involve
binary outcomes, counts, censoring, spatial dependence, and efficiency
frontiers. PanelBox implements these in a consistent interface.

**Discrete choice and count data.** Pooled, fixed-effects (conditional
maximum likelihood, @chamberlain1980), and random-effects logit/probit, plus
multinomial, ordered, and dynamic binary models with the @wooldridge2005
initial-conditions correction. For counts, the Poisson family (pooled,
conditional FE, QML), negative binomial for overdispersion, zero-inflated
models, and the PPML estimator [@santossilva2006] widely used for gravity
models. Average marginal effects, marginal effects at means, and at
representative values are available with delta-method standard errors.

**Spatial econometrics.** Five spatial panel models — SAR, SEM, the spatial
Durbin model (SDM), the general nesting model (GNS), and dynamic spatial
panels — estimated by quasi-maximum likelihood [@leeyu2010]. The
log-determinant $\ln|\mathbf{I}_N - \rho\mathbf{W}|$ is computed by
eigenvalue decomposition, sparse LU factorization, or Chebyshev approximation
depending on $N$, and results are decomposed into direct and indirect
(spillover) effects via the spatial multiplier
$(\mathbf{I}_N - \rho\mathbf{W})^{-1}\beta_k$.

**Stochastic frontier analysis.** The frontier
$\ln y_{it} = \mathbf{x}_{it}'\boldsymbol{\beta} + v_{it} - u_{it}$ with
one-sided inefficiency $u_{it} \ge 0$, supporting four distributions for
$u_{it}$, the Battese–Coelli time-varying and determinant specifications
[@battesecoelli1992; @battesecoelli1995], Greene's true fixed/random effects
[@greene2005], and a four-component model [@kumbhakar2014] that separates
*persistent* from *transient* inefficiency — a policy-relevant decomposition
not available in other Python libraries.

**Quantile regression and panel VAR.** Pooled, fixed-effects [@koenker2004],
Canay two-step [@canay2011], and the @machadosantossilva2019 location-scale
estimator, which guarantees non-crossing quantiles. Panel VAR/VECM provides
impulse-response functions with bootstrap bands, forecast-error variance
decomposition, and Granger-causality tests.

## Inference, diagnostics, and reporting

Valid inference is central. All standard errors use the sandwich form
$\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) =
(\mathbf{X}'\mathbf{X})^{-1}\,\hat{\boldsymbol{\Omega}}\,(\mathbf{X}'\mathbf{X})^{-1}$,
differing in the meat $\hat{\boldsymbol{\Omega}}$: HC0–HC3 for
heteroskedasticity, one-way and two-way clustering, and the Driscoll–Kraay and
Newey–West HAC estimators. Four bootstraps — pairs, wild, block, and
residual — cover small-sample and non-standard cases.

The diagnostic suite includes the Hausman test [@hausman1978] comparing FE and
RE, and the GMM trio that governs dynamic-panel validity: the Hansen $J$ test
of overidentifying restrictions [@hansen1982], and the Arellano–Bond AR(1) and
AR(2) tests for serial correlation, where AR(1) is expected to reject and AR(2)
to *not* reject.

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

## Application: dynamic labor demand

We reproduce the canonical application of @arellano1991: a dynamic labor-demand
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
statistics agree exactly.

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
VAR/VECM — previously unavailable in Python.

Current limitations include spatial-model scalability beyond $N \approx 2{,}000$
(mitigated by Chebyshev approximation), GMM restricted to linear dynamic panels,
and no GPU acceleration. Planned work includes FMOLS/DOLS for cointegrated
panels, double/debiased machine learning for panel data, GPU acceleration via
JAX/CuPy, and Bayesian panel models. By eliminating the need to leave Python for
sophisticated panel analysis, PanelBox makes empirical economic research more
reproducible, more integrated with the machine-learning ecosystem, and more
accessible. It is available on PyPI (`pip install panelbox`) and on
GitHub [@panelbox2025], with documentation, tutorials, and full replication
materials for this paper.
