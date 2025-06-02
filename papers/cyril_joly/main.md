---
# Ensure that this title is the same as the one in `myst.yml`
title: "OptiMask: Efficiently Finding the Largest NaN-Free Submatrix"
abstract: |
  When working with tabular data, certain processes cannot handle missing values. While imputation strategies can be employed, another approach is to use a NaN-free submatrix. Simply dropping every row or column containing NaN values can result in a significantly reduced dataset, potentially leaving no data at all. OptiMask is a heuristic designed to address the following optimization problem: identifying the largest (in terms of the number of elements) not necessarily contiguous submatrix without missing data. OptiMask calculates the sets of rows and columns to remove from the input matrix, providing a solution or a close approximation to the optimal solution for this problem.
---

## Introduction
