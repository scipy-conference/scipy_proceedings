---
# Ensure that this title is the same as the one in `myst.yml`
title: Detecting Anomalies in Scientific Data Using SciPy's Statistical and Signal Tools
abstract: |
  This paper describes a practical anomaly-detection workflow for scientific data
  using SciPy's statistical and signal-processing routines. We focus on clear
  diagnostics, stable calibration, and validation practices that help researchers
  distinguish instrumentation artifacts from physically meaningful deviations.
---

## Introduction

Anomaly detection in scientific data often blends statistical testing, robust
summary statistics, and signal-processing heuristics. While many workflows are
ad hoc, reproducibility improves when the detection logic is documented, the
assumptions are explicit, and the diagnostics are logged. This paper organizes a
SciPy-centered workflow and highlights steps that reduce false positives in
noisy or nonstationary settings.

We build on core routines in SciPy for statistical testing and signal analysis
[@scipy], using NumPy arrays for data handling [@numpy].

## Problem Setting

We focus on scalar time series and multivariate measurements collected from
instruments with occasional transient faults. The target is to identify
unexpected deviations that violate stationarity or distributional assumptions
without overfitting to normal measurement noise.

## Methods

Our workflow has three stages:

1. Data preparation and robust scaling using median and MAD statistics.
2. Candidate anomaly scoring using windowed z-scores, rank statistics, and
   frequency-domain energy shifts.
3. Post-hoc validation with holdout segments and sensitivity checks.

We report thresholds, window sizes, and smoothing parameters to ensure that
results are reproducible across repeated runs.

## Results

We summarize detection rates, false-positive rates, and runtime for each method
variant. The evaluation highlights the tradeoff between sensitivity and
stability, particularly for short-window detectors in noisy signals.

## Discussion

Robust scaling and validation reduce the risk of false discoveries when the
signal distribution drifts or contains heavy tails. Diagnostics such as residual
plots and band-limited energy ratios help connect anomalies to physical causes.

## Conclusion

A small set of consistent preprocessing, scoring, and validation steps can make
anomaly detection workflows more reliable. Future work will expand the study to
multivariate covariance changes and streaming detection scenarios.
