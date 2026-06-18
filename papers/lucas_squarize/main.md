---
# Ensure that this title is the same as the one in `myst.yml`
title: "The Missing Lever in Machine Learning Deployment: Threshold Tuning using Regression Discontinuity"
abstract: |
  When deploying machine learning models into production, a critical but often overlooked
  lever is the decision threshold that converts predictions into actions. In practice,
  thresholds are typically chosen arbitrarily and left unchanged, leaving substantial value
  on the table despite heavy investment in model development.

  We propose a systematic framework for optimizing thresholds using Regression Discontinuity
  Design (RDD), an econometric method that exploits the natural experiments created by
  decision cutoffs. We operationalize this approach into a Python package that automates
  bandwidth selection, local linear estimation, robustness checks, visualization, and
  optimization routines. Validated on simulated data and applied within a large-scale
  production system at Intuit, RDD-based threshold tuning consistently outperforms
  alternative methods, improving target metrics without harming guardrails and enabling
  principled trade-off analysis between competing business objectives.
---

## Motivation

When deploying machine learning models into production, one critical lever is routinely
overlooked: the threshold that converts predictions into actions. A recommender system must
decide which items to display, a churn model who to target, and a fraud model when to
trigger an alert. These thresholds often matter as much for outcomes as the underlying model
itself. Yet in practice, they are usually chosen arbitrarily: 50% relevance to show a
recommendation, 70% fraud risk to flag a case as fraud, 90% churn probability to trigger a
discount. Once set, they are often left unchanged throughout the model's lifecycle.

As a result, firms invest heavily in model development but leave major value untapped by
relying on static, guesswork thresholds. We propose a systematic way to optimize thresholds
using Regression Discontinuity Design (RDD) [@cattaneo2020], an econometric method that
leverages arbitrary policy or decision cutoffs to uncover causal relationships.

## Methodology: Regression Discontinuity to Tune Thresholds

Thresholds create natural experiments: units just above and just below a cutoff are nearly
identical, except that one receives a treatment (e.g., a recommendation or a discount) and
the other does not. RDD exploits this discontinuity to estimate the causal effect of the
treatment for units that are close to the threshold (@fig:rdd-analysis displays an example).

This turns an arbitrary cutoff into a measurable and optimizable decision rule. By focusing
on outcomes near the threshold, RDD can both diagnose whether the current cutoff is adding
or destroying value, and optimize by recommending whether the threshold should be raised,
lowered, or kept constant to maximize outcomes while preserving guardrails.

We operationalized this approach into a practical methodology that combines causal inference
with optimization. Building on recent econometrics research [@marinescu2022], we use fitted
values from RDD models to find alternative values for the thresholds in order to maximize
business outcomes, with or without guardrail constraints. For example, in @fig:rdd-analysis,
the positive impact of a recommendation in the main outcome metric suggests that a reduction
in the threshold would generate additional gain.

To scale adoption, we developed a Python package that automates RDD-based threshold
optimization. The package includes bandwidth selection, local linear estimation, robustness
checks, visualization tools, and optimization routines that identify the best threshold for
different scenarios. This enables seamless integration into production ML pipelines without
requiring new randomized experiments or uplift modeling. With online iterations, we guarantee
that our decision thresholds are always at an optimized level. We recommend the RDD-based
threshold tuning as a crucial workflow for ML model deployment (@fig:workflow).

:::{figure} figure1.png
:label: fig:rdd-analysis
RDD-based threshold tuning. In the graphs, the estimated score from a ML model is on the
horizontal axis, and the average outcome (e.g., conversion) is on the vertical axis. The
graphs show the fitted local regressions to the left (red) and right (blue) of a threshold
for a recommendation (grey shades are confidence intervals). **a)** RDD analysis estimates
near the threshold show that the recommendation produces a gain in the main outcome.
**b)** Using the fitted values we recommend optimized thresholds aimed at maximizing the
business outcome. For illustration, the figures were constructed using simulated data.
:::

:::{figure} figure2.png
:label: fig:workflow
Workflow for RDD-based threshold tuning. Predictions from an ML model are translated into
actions using a threshold. RDD analysis estimates causal effects near the threshold and
recommends optimized thresholds that improve business outcomes, considering both the main
outcome and possible guardrails.
:::

## Applications and Impact

First, we validated the method on simulated data and then applied it within a large-scale
system at Intuit. For instance, in a sequential recommendation setting, we originally defined
thresholds at 50% for topic recommendations, which still left potential gains on the table.
We then tested different approaches to threshold tuning in a real experiment. In one recipe,
thresholds were optimized using our RDD-based methodology, while others relied on alternative
methods such as arbitrary cutoffs or precision and recall optimization. Results showed that
thresholds optimized with RDD consistently performed better than alternative methods,
improving our target metrics without harming guardrail metrics.

Another application involved a premium feature recommendation with a trade-off between
revenue and units. This is similar to a discount policy that offers a price discount to
customers with a high probability of churning, increasing their probability of conversion
but reducing the expected revenue per customer. We used our RDD-based method to produce
trade-off curves between revenue and conversion, allowing leaders to steer outcomes more
scientifically depending on their business goals (@fig:tradeoff).

:::{figure} figure3.png
:label: fig:tradeoff
Business outcome as a function of threshold. RDD-based threshold tuning provides the
trade-off between two or more conflicting metrics: the main outcome (e.g., conversion) and
a guardrail (e.g., revenue). In this case, a reduction in the threshold increases conversion
but reduces revenue, while an increase in the threshold increases revenue but reduces
conversion. These curves present leaders the trade-off associated with their business
decision regarding the threshold level. For illustration, the figure was constructed using
simulated data.
:::

Across simulated and real-world deployments, the approach has led to measurable improvements
in engagement, conversions, and revenue, while preserving quality guardrails. Importantly,
it provides a lightweight, repeatable way to adapt thresholds as customer behavior and
business priorities evolve.

## Contributions and Implications

The examples in the previous section illustrate how the framework translates statistical
insights into practical levers for decision-making, without requiring new experiments. Our
contributions are fourfold:

- **Conceptual**: Elevating threshold decisions as critical, often-overlooked levers in machine learning deployment.
- **Methodological**: Showing how RDD can be repurposed from evaluation to optimization.
- **Practical**: Delivering a package and workflow that make RDD tuning accessible and scalable.
- **Empirical**: Demonstrating impact in production deployments without costly new experiments.

More broadly, this work positions threshold tuning as a causal optimization problem. By
embedding RDD into production pipelines, firms can move from arbitrary cutoffs to adaptive,
principled business rules that maximize value at scale.
