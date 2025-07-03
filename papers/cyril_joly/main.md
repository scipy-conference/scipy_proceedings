---
# Ensure that this title is the same as the one in `myst.yml`  
title: "OptiMask: Efficiently Finding the Largest NaN-Free Submatrix"  
abstract: |  
  When working with tabular data, many processes cannot handle missing values. While imputation strategies are commonly used, another approach is to extract a NaN-free submatrix. Simply dropping every row or column containing NaN values can drastically reduce the dataset, potentially leaving no usable data. OptiMask is a heuristic designed to solve the following optimization problem: identifying the largest (in terms of the number of elements) not necessarily contiguous submatrix without missing data. OptiMask determines the sets of rows and columns to remove from the input matrix, providing either an exact or near-optimal solution.
---

## Introduction  

Missing data is a common challenge in data analysis, often represented as NaN (Not a Number) values in matrices or DataFrames.
Many algorithms and statistical methods require complete datasets, necessitating effective handling of missing values [@little2019statistical; @rubin2004multiple].
Traditional approaches include imputation (replacing missing values with estimates) and complete-case analysis (discarding rows/columns with any NaN) [@schafer1997analysis; @van2018flexible].
However, imputation can introduce bias [@white2011multiple], while complete-case analysis may discard excessive data, especially when missing values are widespread [@enders2010applied].  

An alternative strategy is to identify the largest possible submatrix without missing values, preserving the original data unaltered.
This reduces to an optimization task: remove the minimal set of rows and columns to yield a NaN-free submatrix of maximum size (i.e., maximizing the product of its dimensions).  

This problem is computationally challenging, as the search space grows exponentially with the number of rows and columns containing NaN.
Exact solutions (e.g., linear programming) guarantee optimality but are prohibitively expensive for large matrices.
Heuristic methods like OptiMask provide near-optimal solutions efficiently.  

OptiMask is a heuristic method which closely, efficiently approximates these exact solutions.
It iteratively permutes rows and columns to isolate NaN values along a frontier, simplifying the search for the largest contiguous NaN-free submatrix.
By combining randomization with multiple restarts, it reliably finds high-quality solutions.
This paper explores the OptiMask algorithm, theoretical foundations, and practical performance across diverse datasets, including large and structured matrices. It also discusses the `optimask` Python package (<https://pypi.org/project/optimask/>), which enables applying the algorithm to matrix-like data structures popular with Python programmers.

## Problem Formalization and Challenges  

### Mathematical Definition  

Given an $ m \times n $ matrix $ A $ with missing values (NaN), the goal is to find:  

- A subset of rows $ \mathcal{R} \subseteq \{1, \dots, m\} $ and columns $ \mathcal{C} \subseteq \{1, \dots, n\} $ such that the submatrix $ A[\mathcal{R}, \mathcal{C}] $ contains no NaN values.  
- The solution maximizing the size $ |\mathcal{R}| \times |\mathcal{C}| $ of the submatrix.  

### The Fundamental Trade-off  

When handling missing values in a matrix, we must decide whether to remove affected rows, columns, or a combination of both.
The optimal choice depends on the matrix's dimensions and NaN distribution:  

1. **Single NaN Case**:  
   - In tall matrices (`rows > columns`), removing the problematic row typically preserves more data.  
   - In wide matrices (`columns > rows`), removing the affected column is usually preferable.  

   :::{figure} figures/at_hand_two  
   :alt: Data to process  
   :width: 400 px
   :align: left
   A toy example: removing the row containing the two NaNs yields the largest submatrix, rather than removing the two columns.  
   :::  

2. **Multiple NaNs**:  
   - When a row contains several NaNs, removing the entire row may be more efficient than removing multiple columns.  
   - Conversely, if NaNs are clustered in columns, removing those columns could be optimal.  
   - In complex cases, the optimal solution requires removing specific combinations of rows and columns, necessitating a general algorithmic approach.  

### Linear Programming Formulation  

The problem can be formulated using integer linear programming [@wolsey2020integer], defining decision variables for removing rows, columns, and individual cells, subject to constraints ensuring all NaN values are handled.
The objective is to minimize the total number of effectively removed cells, equivalent to maximizing the area of the remaining NaN-free submatrix.  

**Given:**  

- A matrix $A$ of shape $m \times n$ with elements $a_{i,j}$.  
- Decision variables for $i \in \{1, \dots, m\}$ and $j \in \{1, \dots, n\}$:  
  - $r_i \in \{0,1\}$: 1 if row $i$ is removed, 0 otherwise.  
  - $c_j \in \{0,1\}$: 1 if column $j$ is removed, 0 otherwise.  
  - $e_{i,j} \in \{0,1\}$: 1 if cell $(i,j)$ is effectively removed (i.e., part of a removed row or column), 0 otherwise.  

**Constraints:**  

For all $i \in \{1, \dots, m\}$ and $j \in \{1, \dots, n\}$:  

- If $a_{i,j}$ is NaN, then $e_{i,j} = 1$.  
- $r_i + c_j \geq e_{i,j}$.  
- $e_{i,j} \geq r_i$.  
- $e_{i,j} \geq c_j$.  

**Objective Function:**  

Minimize the total number of effectively removed cells:  

$$
\min \sum_{i=1}^{m} \sum_{j=1}^{n} e_{i,j}  
$$  

This formulation can be solved using integer linear programming solvers (e.g., GLPK [@makhorin2012glpk], Gurobi [@gurobi2023gurobi], CPLEX [@cplex2009v12]), often interfaced via modeling languages like Pyomo [@hart2017pyomo] or PuLP [@mitchell2011pulp] in Python for data science practioners.
However, its primary disadvantage is computational cost: for an $m \times n$ matrix, the formulation uses $m \times n + m + n$ binary variables, which becomes prohibitive for large matrices.

## Algorithm  

OptiMask is a heuristic designed to provide near-approximations of the optimal solutions to the problem.
The core idea is to compute row and column permutations such that the search for the largest non-contiguous NaN-free submatrix reduces to finding a contiguous one.  

### Core Approach  

:::{figure} figures/algo_data  
:alt: Data to process  
:width: 400 px
:align: left
An example matrix illustrating the algorithm's steps. Grey cells represent missing values.  
:::  

OptiMask employs an iterative permutation-based algorithm to identify the largest NaN-free submatrix through these key steps:  

1. **Problem Reduction**:  
   - Isolate rows and columns containing at least one NaN value (rows or columns without NaNs are preserved, as there is no reason to remove them).  
   - Create a boolean mask matrix where True represents NaN positions.  

2. **Frontier Detection**:  
   - Compute `hx`: column-wise highest NaN index (from the bottom).  
   - Compute `hy`: row-wise rightmost NaN index (from the left).  
   - These define the current "NaN frontier" of the matrix.  

   :::{figure} figures/algo_0  
   :alt: Step #1 and #2  
   :width: 400 px
   :align: left
   Steps #1 and #2: isolating rows and columns with NaNs and computing `hx` and `hy`.  
   :::  

3. **Permutation Phase**:  
   - Alternately sort rows and columns to push NaN values toward a Pareto frontier.  
   - Even iterations: Sort columns by descending `hx`.  
   - Odd iterations: Sort rows by descending `hy`.  
   - Track all permutations applied during this process.  
   - Repeat until both `hx` and `hy` form non-increasing sequences, indicating an optimal NaN frontier.  

   :::{figure} figures/algo_iterations  
   :alt: Permutation steps  
   :width: 800 px
   :align: left
   Three alternate permutations lead to a Pareto frontier of NaNs.  
   :::  

4. **Submatrix Extraction**:  
   - Identify the largest contiguous NaN-free rectangle in the permuted space.  

   :::{figure} figures/algo_result_permuted_space  
   :alt: OptiMask result in permuted space  
   :width: 400 px
   :align: left
   OptiMask result in permuted space: the black-dotted rectangles are candidates for the largest contiguous NaN-free submatrix, with the red-dotted one selected for its maximal area.  
   :::  

   - Apply inverse permutations to map back to original row/column indices.  

   :::{figure} figures/algo_result.png
   :alt: OptiMask result  
   :width: 400 px
   :align: left
   OptiMask result: red indicates removed rows and columns, blue marks the computed NaN-free submatrix.  
   :::  

## Python Package  

A Python implementation of the algorithm is available on PyPI (<https://pypi.org/project/optimask/>) and conda-forge (<https://anaconda.org/conda-forge/optimask>) and can be used as follows: 

```python
import numpy as np
from optimask import OptiMask
from optimask.utils import generate_mar

# Generate a Missing At Random matrix with 2% NaN values
x = generate_mar(m=100_000, n=1_000, ratio=0.02)
rows, cols = OptiMask().solve(x)
np.isnan(x[np.ix_(rows, cols)]).any()  # False
len(rows), len(cols)  # (38031, 48)
```  

This computation takes approximately ~200ms on an average personal computer.
The library uses Numba [@lam2015numba] for speed, and accepts inputs several popular input formats, including NumPy arrays [@harris2020array], pandas DataFrames [@mckinney2010data], and Polars DataFrames [@vink2023polars].

## Conclusion

OptiMask provides a scalable heuristic for finding the largest NaN-free submatrix in large datasets where exact methods like linear programming become computationally impractical.
By strategically permuting rows and columns to isolate missing values, it offers a practical solution that preserves maximal data without imputation.
The Python implementation supports common data structures (NumPy, pandas, Polars) and delivers results efficiently even for big matrices.

Future work will explore theoretical guarantees on the approximation quality and extensions to weighted optimization problems.
The implementation will continue to be optimized for speed in subsequent versions.
Finally, an OptiMask-based algorithm for tabular imputation will be developed and benchmarked against Multiple Imputation by Chained Equations ("MICE") to evaluate whether it can achieve better or faster results.

## Aknowledgements

This work was funded by Airparif.
I'd like to thank Paul Catala (Université de Lorraine) and Alexis Lebeau (RTE) for their assistance and review.

::: {#refs}
:::
