---
# Ensure that this title is the same as the one in `myst.yml`
title: 'Right Predictions, Wrong Reasons: Explanation Drift Monitoring in Production'
abstract: |
  Machine learning models in production can keep making correct predictions
  while silently changing the reasoning behind them. Because traditional
  monitoring watches input distributions and evaluation metrics, this shift in
  reasoning often goes undetected until the model's performance visibly degrades.
  By then, the model may have served many flawed decisions. We describe
  *explanation drift monitoring*: tracking the distribution of per-feature
  SHapley Additive exPlanations (SHAP) attributions over time and alerting when
  it moves. Explanation drift is a complement to input feature drift and to
  performance monitoring, not a replacement; it is most valuable in the common
  situation where ground-truth labels arrive with a long delay. We discuss the
  signals that make explanation drift measurable, i.e., per-feature Population
  Stability Index (PSI), a multivariate adversarial-validation score, and changes
  in attribution magnitude, rank, and sign, and we relate them to three recurring
  production failure patterns: concept drift, model-version regressions that
  preserve accuracy, and data-pipeline errors that input monitoring cannot see.
  We ground the approach in a worked case study on the UCI Adult Income dataset,
  where a model's F1 score holds steady across a demographic shift
  while its SHAP attributions change dramatically. Finally, we present `shap-monitor`, an
  open-source Python package that implements this workflow with low
  instrumentation overhead, and we discuss the practical challenges of running it
  in production.
---

## Introduction and motivation

A model that keeps reporting good accuracy is easy to trust and easy to ignore.
Yet a model can produce the *right predictions for the wrong reasons*: it can
hold its headline metrics steady while the features driving its decisions change
underneath it. When that happens, the model has quietly become a different model.
A model whose behavior was never validated will eventually fail, and the
eventual failure tends to be sudden and hard to diagnose.

Most production ML monitoring is built around two families of signal. The first
is **performance monitoring**: accuracy, F1, AUC, calibration, and business KPIs
computed once ground-truth labels are available. The second is **input drift
monitoring**: statistical tests on incoming feature distributions, watching for
the data to move away from what the model was trained on. Both are valuable and
both are necessary. Both also have a blind spot.

Performance monitoring is the ground truth, but it is frequently *lagging*. In
many real-world systems labels are delayed by weeks or months: loan-default
labels mature over a repayment horizon, fraud labels require investigation, and
churn is only confirmed after a customer has already left. Until the labels
arrive, performance monitoring is blind. Input drift monitoring does not need
labels, but it answers the question *did the data change?* rather than
*did the model's use of the data change?*.

This paper is about a third signal that sits between the two: **explanation
drift**, the change over time in *how* a model attributes its predictions to
input features. We measure it by logging per-feature SHAP attributions
[@shap_nips] for sampled production traffic and tracking their distributions over
time. When the model starts relying on different features, or relying on the same
features in a different direction, the SHAP distributions move and can be detected
without waiting for labels.

Our contributions are:

1. A practical framing of explanation drift as a monitoring signal that
   complements input drift and performance monitoring, with an explicit account
   of what each signal can and cannot catch.
2. A description of concrete, low-overhead detection signals, including per-feature PSI,
   a multivariate adversarial-validation score, and attribution magnitude/rank/sign changes,
   and how to interpret them.
3. A mapping from these signals to three failure patterns that recur in production:
   concept drift, accuracy-preserving model-version regressions, and data-pipeline
   errors invisible to input monitoring.
4. An open-source reference implementation, `shap-monitor` [@shapmonitor], and a
   reproducible case study demonstrating the blind spot it closes.

## Background

### SHAP attributions

SHAP (SHapley Additive exPlanations) assigns each feature of a single prediction
a contribution value derived from cooperative game theory, distributing the gap
between a model's output and a baseline expectation across the input features
[@shap_nips]. For a prediction $f(x)$, the attributions $\phi_i$ satisfy an
additive decomposition,

```{math}
:label: shap-additivity
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i,
```

where $\phi_0$ is the base (expected) value and $\phi_i$ is the contribution of
feature $i$. A positive $\phi_i$ pushes the prediction above the baseline; a
negative one pulls it below. For tree ensembles, `TreeExplainer` computes these
attributions exactly and efficiently [@shap_treeexplainer], which makes per-
prediction explanation practical at production volumes.

For monitoring we are not interested in any single explanation. We are interested
in the *distribution* of $\phi_i$ for each feature across many predictions, and
in how that distribution evolves over time. Two summaries are central. The
**mean absolute attribution** $\mathbb{E}[|\phi_i|]$ is a global importance
measure that quantifies how much feature $i$ moves predictions, regardless of direction. The
**mean signed attribution** $\mathbb{E}[\phi_i]$ captures the typical *direction*
of the feature's effect. A change in either of these is a change in the model's reasoning.

### What existing monitoring catches, and what it misses

Consider a tabular classifier serving live traffic. Input drift monitoring tests
whether each incoming feature's distribution has shifted from a reference window,
typically with a univariate statistic such as PSI or a Kolmogorov-Smirnov (KS)
test, and tools such as Evidently [@evidently] and Alibi Detect [@alibi_detect]
make this routine. Performance monitoring, once labels land, tells us whether the
model is still accurate. Neither directly observes the model's decision process.

The gap is the case where inputs look acceptable, labels have not yet arrived,
and yet the model's behavior has changed. This is precisely where explanation
drift is informative: it observes the model's *output reasoning*, which is a
function of both the inputs and the learned model, and therefore reacts to
changes that a feature-only view can miss.

## Explanation drift

We define **explanation drift** as a statistically detectable change, over time,
in the distribution of a model's SHAP attributions on production inputs. It is
distinct from input feature drift in an important way. Input drift asks whether
$P(X)$ has changed. Explanation drift asks whether $P(\phi(X))$ has changed,
where $\phi$ is the explanation function induced by the *current model*. The
attribution distribution couples the data and the model, so explanation drift can
surface for at least three reasons:

- the input distribution moved in a way that changed which features the model
  leans on (covariate shift with a behavioral consequence);
- the relationship between features and target changed, so the same inputs are
  now used differently (concept drift); or
- the model itself changed, i.e., a new version, or a silent change in an upstream
  transformation, so attributions shift even on identical inputs.

Crucially, none of these requires labels to detect. That makes explanation drift
a *leading* signal in delayed-label regimes, where it can flag a problem during
the window when performance monitoring is still blind.

Explanation drift is a complement, not a replacement. Input drift remains the
right tool for raw data-quality and distribution questions; performance
monitoring remains the arbiter of whether the model is actually still good. The
value of explanation drift is that it narrows the blind spot between them, and
that because attributions are per-feature it points at *which* part of the
model's reasoning moved, which is a strong lead for root-cause analysis.

## Detecting explanation drift

Given logged attributions for a reference period and a current period, several
signals make drift measurable. They differ in whether they look at one feature at
a time or at the joint distribution, and they are most useful read together.

### Per-feature distribution shift (PSI)

The Population Stability Index is a long-standing measure of how much a
distribution has moved between two samples, widely used in credit risk to monitor
scorecards [@psi_siddiqi]. It bins a reference distribution, compares the
proportion of mass each bin receives under the current distribution, and sums the
divergence:

```{math}
:label: psi
\mathrm{PSI} = \sum_{b=1}^{B} (c_b - r_b)\,\ln\!\frac{c_b}{r_b},
```

where $r_b$ and $c_b$ are the reference and current proportions in bin $b$.
Applied to the SHAP values of a feature, PSI quantifies how much that feature's
*contribution* distribution has shifted. The conventional reading is that PSI below 0.1
is stable, 0.1–0.25 warrants investigation, and above 0.25 indicates significant
shift and transfers directly. PSI is cheap, interpretable, and per-feature, which
makes it a good default for a first-pass alarm and for localizing drift to
specific features. A KS test is a reasonable alternative univariate statistic;
PSI's advantage here is the familiar, calibrated banding.

### Multivariate shift (adversarial validation)

Per-feature tests miss joint shifts: combinations of attributions can move even
when no single feature's marginal PSI looks alarming. *Adversarial validation*
captures this. We label reference-period attribution vectors as class 0 and
current-period vectors as class 1, train a classifier to tell them apart, and
measure its cross-validated AUC [@adversarial_validation]. An AUC near 0.5 means
the two periods are statistically indistinguishable; an AUC approaching 1.0 means
the model's reasoning in the two periods is easily separable, i.e., strong evidence of
drift. The classifier's feature importances additionally rank *which* attribution
dimensions drive the separation, giving a multivariate complement to per-feature
PSI.

### Magnitude, rank, and sign changes

Beyond distributional distance, three interpretable summaries describe the
*nature* of a shift. The **change in mean absolute attribution** says whether a
feature became more or less influential overall. The **change in importance
rank** says whether the model's priority ordering of features was reshuffled. And
a **sign flip**, i.e., a change in the typical direction of a feature's contribution,
is the most striking: it means a feature that used to push predictions one way now
pushes them the other. Sign flips on important features are a high-signal
indication that relationships learned during training no longer hold.

Read together, these signals answer complementary questions: PSI asks *how much*
each feature's contribution moved, adversarial AUC asks *whether the joint
reasoning is distinguishable at all*, and the magnitude/rank/sign summaries
describe *in what way* the reasoning changed.

## Major failure patterns

Explanation drift is worth monitoring because it catches failure modes that the
other signals are structurally poor at catching. Three recur often enough to be
worth naming.

### Concept drift: same features, shifted relationships

In concept drift the relationship between features and target changes while the
inputs may look similar [@gama2014survey]. A feature that was once weakly
predictive becomes strongly predictive, or reverses direction. Input monitoring
can be quiet, because the marginal feature distributions need not move much.
Performance monitoring eventually catches it, once labels arrive. Explanation
drift sits in between: as the model encounters inputs whose learned relationships
no longer match reality, its attribution magnitudes and signs shift, and PSI on
the affected SHAP distributions rises before the labels confirm the damage.

### Model-version regression: same accuracy, different reasoning

Model retraining and redeployment is routine, and validation usually gates on
aggregate metrics. A new version can match the old one's accuracy on a holdout set
while arriving at its predictions through materially different reasoning, e.g., leaning
on a leak-prone proxy feature, say, or down-weighting a feature the business
considers essential. Because the headline metric is preserved, this passes
standard checks silently. Comparing the SHAP attribution distributions of the two
versions on the *same* inputs surfaces it directly: a high adversarial-validation
AUC between versions, or large per-feature rank and sign changes, reveals that the
reasoning regressed even though the accuracy did not.

### Data-pipeline errors invisible to input monitoring

Some of the most damaging production incidents are upstream data bugs: a feature
silently defaulting to a constant, a units change, a join that drops rows, a
transformation applied in the wrong order. When such a bug pins a feature to a
plausible value, its marginal distribution can stay within range and input
monitoring stays quiet, but the model's *use* of that feature collapses or
distorts, which the attribution distribution registers immediately. Because
explanation drift is per-feature, it both detects the problem and points at the
feature whose pipeline to inspect.

## Case study: a stable score hiding a reasoning shift

To make the blind spot concrete and reproducible, we walk through a covariate
shift on the UCI Adult Income dataset [@uci_adult], a standard tabular benchmark
predicting whether annual income exceeds \$50K. We construct a deliberate
demographic shift: we train on the younger population (age $\le 45$) and treat the
older population (age $> 45$) as drifted "production" traffic. This mirrors a
model developed on one population and deployed to serve another. We fit a LightGBM
classifier [@lightgbm], use `TreeExplainer` for exact attributions
[@shap_treeexplainer], and compute all drift statistics with `shap-monitor`
[@shapmonitor].

The first thing to notice is that **performance barely moves**. The model's F1
score is 0.71 on the reference (younger) validation data and 0.72 on the drifted
(older) production data. Had we been watching F1 alone — and remembering that in
production we would not even have these labels for weeks — we would have seen
nothing worth investigating.

The explanation signal tells a completely different story. @fig:blindspot places
the two side by side: F1 is essentially flat across the shift, while the
adversarial-validation drift score over the SHAP attributions jumps from 0.50 (a
no-drift control, where reference data is split in half) to 0.97 (reference vs.
drifted). An AUC of 0.97 means the model's reasoning in the two regimes is almost
perfectly separable, i.e., the model is making similarly accurate predictions through
substantially different reasoning.

:::{figure} figure1.png
:label: fig:blindspot
The monitoring blind spot. The model's F1 score is flat across the demographic
shift (0.71 → 0.72), so performance monitoring sees nothing, and in a delayed-
label setting these labels would not be available yet anyway. An
adversarial-validation drift score computed over the model's SHAP attributions
rises from 0.50 (no-drift control) to 0.97, revealing that the model's reasoning
changed substantially even though its accuracy did not.
:::

Per-feature PSI localizes the shift. @fig:psi shows the PSI of each feature's SHAP
distribution between the reference and drifted periods, on a log scale because the
values span more than two orders of magnitude. The `age` feature's attribution PSI
is 11.07 — far above the 0.25 "significant shift" threshold — and several other
features (`race`, `marital-status`, `workclass`) also cross the alert line. The
model has reorganized which features drive its decisions for the older population.

:::{figure} figure2.png
:label: fig:psi
Per-feature PSI of the SHAP attribution distributions, reference vs. drifted
period (log scale). The dashed guides mark the conventional 0.1 (warn) and 0.25
(alert) thresholds. The `age` attribution distribution moves by a PSI of 11.07,
and several other features cross the alert threshold — the model is attributing
its predictions very differently under the shift.
:::

Looking inside the most-drifted feature confirms the picture. @fig:dist overlays
the distribution of `age` SHAP values in the two periods. In the reference period
the attributions are broad and bimodal; in the drifted period they collapse into a
tight, strongly positive cluster — for the older population, `age` has become a
consistent, large positive driver of the high-income prediction. Alongside this,
`shap-monitor` reports a +33% increase in `age` mean absolute attribution, an +82%
increase for `workclass`, and sign flips on `marital-status` and `workclass`,
meaning those features reversed their typical contribution direction. None of this
is visible in the F1 score.

:::{figure} figure3.png
:label: fig:dist
Distribution of `age` SHAP values, reference (blue) vs. drifted (orange). The
reference distribution is broad and bimodal; under the shift it collapses to a
tight, strongly positive cluster (dashed lines mark the means). For the older
population, `age` has become a consistent large positive driver of the prediction.
:::

The lesson generalizes beyond this dataset. A flat headline metric is not evidence
that a model is behaving as it did at validation time. Explanation drift makes the
difference observable, and does so without waiting for labels.

## Open-source implementation

`shap-monitor` [@shapmonitor] is an open-source (Apache-2.0) Python package that
implements this workflow with a deliberately small surface area: a logger to
capture attributions in production, a storage backend, and an analyzer to compute
drift. It builds on SHAP [@shap_nips], scikit-learn [@sklearn1; @sklearn2], NumPy
[@numpy], and pandas [@pandas2].

### Logging attributions in production

The `SHAPMonitor` wraps any SHAP explainer and logs attributions for a configurable
fraction of traffic. Sampling keeps the overhead bounded: explanation cost scales
with the number of explained rows, so a sample rate well below 1.0 is appropriate
for high-volume services.

```python
from shapmonitor import SHAPMonitor
import shap

monitor = SHAPMonitor(
    explainer=shap.TreeExplainer(model),
    data_dir="/var/log/shap",
    sample_rate=0.05,          # explain 5% of traffic
    model_version="prod-v2.1",
)

# In the prediction path:
predictions = model.predict(X)
monitor.log_batch(X, predictions)   # samples, explains, and persists
```

`log_batch` samples the batch, computes SHAP values for the sampled rows, and
persists them. The logging call is best wrapped so that a monitoring failure can
never take down the prediction path.

### Storage

Attributions are written as Parquet [@apache_arrow], partitioned by date in a
Hive-style layout (`date=YYYY-MM-DD/`) so that time-range queries prune to the
relevant partitions, with optional partitioning by model version. Each record
carries a timestamp, a batch UUID, the model version, the per-feature SHAP values,
and optionally the feature values and predictions. The columnar format keeps
storage compact and is directly readable by pandas, DuckDB, and Spark. The storage
layer is a small protocol, so a different backend (an object store or a warehouse)
can be substituted without touching the logging or analysis code.

### Analyzing drift

The `SHAPAnalyzer` reads logged attributions and computes the signals described
above. Comparisons can be made between two time periods, two specific batches, or
two model versions:

```python
from datetime import datetime
from shapmonitor.analysis import SHAPAnalyzer
from shapmonitor.backends import ParquetBackend

analyzer = SHAPAnalyzer(ParquetBackend("/var/log/shap"))

# Per-feature PSI, magnitude/rank/sign changes between two windows.
drift = analyzer.compare_time_periods(
    (datetime(2026, 5, 1), datetime(2026, 5, 8)),    # reference
    (datetime(2026, 5, 8), datetime(2026, 5, 15)),   # current
)
alerts = drift[drift["psi"] >= 0.25]          # significant shifts
flips = drift[drift["sign_flip"]]             # reversed contribution direction

# Multivariate adversarial-validation score for the same windows.
adv = analyzer.compare_adversarial(
    (datetime(2026, 5, 1), datetime(2026, 5, 8)),
    (datetime(2026, 5, 8), datetime(2026, 5, 15)),
)
print(adv.attrs["adversarial_auc"])           # 0.5 = no drift, 1.0 = max drift
```

The comparison returns, per feature, its PSI, its mean absolute attribution in
each period, the percentage change in importance, the change in importance rank,
and whether its contribution direction flipped, exactly the quantities reported
in the case study. To compare two deployed models on the same traffic, the
version-regression pattern — `compare_versions` performs the same analysis keyed
on the logged model version.

### Command line and alerting

For operational use the same analysis is available from a command-line interface,
which is convenient for cron-driven reports and for piping machine-readable output
into an alerting system:

```bash
# Human-readable drift report between two windows.
shapmonitor report drift --data-dir /var/log/shap \
    --ref last-14d..last-7d --curr last-7d..now

# Machine-readable output for an alerting rule.
shapmonitor report drift --data-dir /var/log/shap --json \
    | jq '.features[] | select(.psi >= 0.25)'
```

A simple and effective alerting policy is threshold-based: page when any
important feature's SHAP PSI crosses 0.25, or when the adversarial-validation AUC
between the current window and a fixed baseline exceeds a chosen level (for
example 0.8). Restricting per-feature alerts to features above a minimum mean
absolute attribution avoids noisy alarms on features the model barely uses.

### Batch versus streaming, and overhead

The dominant cost is computing the explanations, not storing or analyzing them.
For tree models, `TreeExplainer` is fast enough to run inline on a sampled
fraction of requests; for expensive explainers, or very high request rates,
attribution is better done out of band, i.e., log the inputs and explain them in a
periodic batch job — so that the prediction path is never blocked. Drift analysis
itself is a cheap offline computation over the logged Parquet and is naturally run
on a schedule (hourly or daily) rather than per request.

## Challenges and limitations

Explanation drift monitoring is a useful signal, but it is not free and not a
panacea.

**Explainer cost.** Exact attributions are cheap for tree models but expensive for
model-agnostic explainers such as `KernelExplainer`. Sampling and out-of-band
computation mitigate this, but the cost must be budgeted, and very low sample rates
trade statistical power for cheapness.

**Baseline choice.** Drift is always measured *relative to a reference*. A poorly
chosen baseline — too short, unrepresentative, or itself already drifted —
produces misleading comparisons. Seasonal traffic in particular calls for a
seasonally aware reference rather than a naive trailing window.

**Threshold tuning.** The PSI bands are conventions, not laws, and the right
alerting thresholds depend on the model, the feature, and the tolerance for false
alarms. Thresholds should be calibrated against a quiet period and revisited as the
system evolves.

**Attribution stability and correlation.** SHAP attributions can be sensitive to
correlated features and to explainer configuration, so a portion of measured drift
may reflect attribution variance rather than genuine reasoning change. Adequate
sample sizes per window and a fixed explainer configuration reduce, but do not
eliminate, this.

**It is a leading indicator, not truth itself.** Explanation drift tells you the
model's reasoning changed; it does not, by itself, tell you whether that change is
harmful. It is a high-quality lead that warrants investigation, ideally
triangulated against input drift and, once available, performance, not an
automatic trigger for rollback.

## Conclusion and future work

Production models can keep their accuracy while quietly changing the reasoning
behind their predictions, and the standard monitoring stack, which includes input drift plus
delayed-label performance has a structural blind spot for exactly this. Tracking
the distribution of SHAP attributions over time closes much of that gap. It is a
label-free, leading signal that complements existing monitoring, and because it is
per-feature it both detects drift and localizes it. Our case study showed a model
holding F1 steady (0.71 → 0.72) across a demographic shift while its
adversarial-validation drift score rose from 0.50 to 0.97 and the `age` attribution
PSI reached 11, a reasoning change that was invisible to accuracy and would have
been invisible for weeks in a realistic delayed-label setting.

The `shap-monitor` package provides an open-source reference implementation, and
the technique itself is implementation-agnostic. Promising directions for future
work include asynchronous and streaming logging to further reduce overhead,
additional storage backends for cloud-native deployments, and richer alerting that
fuses explanation drift with input-drift and performance signals into a single,
better-calibrated alarm. We hope explanation drift monitoring becomes a standard
third pillar of production ML observability, alongside the two it complements.
