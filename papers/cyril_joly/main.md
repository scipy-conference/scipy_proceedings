---
# Ensure that this title is the same as the one in `myst.yml`
title: "OptiMask: Efficiently Finding the Largest NaN-Free Submatrix"
abstract: |
  When working with tabular data, certain processes cannot handle missing values. While imputation strategies can be employed, another approach is to use a NaN-free submatrix. Simply dropping every row or column containing NaN values can result in a significantly reduced dataset, potentially leaving no data at all. OptiMask is a heuristic designed to address the following optimization problem: identifying the largest (in terms of the number of elements) not necessarily contiguous submatrix without missing data. OptiMask calculates the sets of rows and columns to remove from the input matrix, providing a solution or a close approximation to the optimal solution for this problem.
---

## Introduction

Missing data is a common challenge in data analysis, often represented as NaN (Not a Number) values in matrices or DataFrames. Many algorithms and statistical methods require complete datasets, necessitating effective handling of missing values. Traditional approaches include imputation (replacing missing values with estimates) or complete-case analysis (discarding rows/columns with any NaN). However, imputation can introduce bias, while complete-case analysis may discard too much data, especially when missing values are widespread.

An alternative strategy is to identify the largest possible submatrix without missing values. This preserves the original data without alteration. The problem reduces to an optimization task: remove the minimal set of rows and columns to yield a NaN-free submatrix of maximum size (i.e., maximizing the product of its dimensions).

This problem is computationally challenging, as the search space grows exponentially with the number of rows and columns containing NaN. Exact solutions (e.g., linear programming) guarantee optimality but are intractable for large matrices. Heuristic methods like OptiMask provide near-optimal solutions efficiently.

OptiMask iteratively permutes rows and columns to isolate NaN values along a frontier, simplifying the search for the largest contiguous NaN-free submatrix. By combining randomization with multiple restarts, it reliably finds high-quality solutions. This paper explores OptiMask’s algorithm, theoretical foundations, and practical performance across diverse datasets, including large and structured matrices.

## Problem Formalization and Challenges  

### Mathematical Definition  

Given an $ m \times n $ matrix $ A $ with missing values (NaN), the goal is to find:  

- A subset of rows $ \mathcal{R} \subseteq \{1, \dots, m\} $ and columns $ \mathcal{C} \subseteq \{1, \dots, n\} $ such that the submatrix $ A[\mathcal{R}, \mathcal{C}] $ contains no NaN values.  
- The solution maximizing the size $ |\mathcal{R}| \times |\mathcal{C}| $ of the submatrix.  

This is equivalent to minimizing the number of deleted rows ($ m - |\mathcal{R}| $) and columns ($ n - |\mathcal{C}| $) while ensuring all remaining NaN-free entries are retained.  

### Computational Complexity  

The problem is NP-hard, as it generalizes the *Bipartite Vertex Cover* problem. Exhaustive search is infeasible for even modestly sized matrices:  

- For a matrix with $ k $ NaN-containing rows/columns, there are $ 2^k $ possible removal combinations.  
- Linear programming (LP) formulations scale poorly due to $ O(mn) $ binary variables.  

### Key Challenges  

1. **Trade-off Between Rows and Columns**: Removing a row may eliminate multiple NaN values but could discard more data than removing specific columns (and vice versa).  
2. **Structured Missingness**: Real-world datasets often exhibit structured NaN patterns (e.g., block-wise or periodic), which heuristics must exploit.  
3. **Scalability**: Solutions must handle large matrices (e.g., $ 10^5 \times 10^3 $) efficiently without exact methods.

## Core Problem and Intuition

### The Fundamental Trade-off

When facing missing values in a matrix, we must decide whether to remove affected rows, columns, or a combination of both. The optimal choice depends on the matrix's dimensions and NaN distribution:

1. **Single NaN Case**:
   - In tall matrices (rows > columns), removing the problematic row typically preserves more data
   - In wide matrices (columns > rows), removing the affected column is usually better

2. **Multiple NaNs**:
   - When a row contains several NaNs, removing the entire row may be more efficient than removing multiple columns
   - Conversely, if NaNs are clustered in certain columns, removing those columns could be optimal
   - In complex cases, the optimal solution requires removing specific combinations of both rows and columns, necessitating a general algorithmic approach

### Handling Different Missingness Patterns

OptiMask effectively handles various NaN distributions:

- **Missing At Random (MAR)**: Randomly scattered missing values
- **Structured patterns**: Block-wise, periodic, or clustered missing data
- **Mixed patterns**: Combinations of random and structured missingness

### Visualizing the Solution

The algorithm identifies the optimal combination of rows and columns to remove, maximizing the retained data:

```python
# For any m×n matrix with arbitrary NaN pattern
rows, cols = OptiMask().solve(x)  # Finds maximal NaN-free submatrix
```

The solution adapts to both the matrix's aspect ratio and the specific NaN distribution, whether random or structured.

## Algorithm

### Core Approach

```{image} figures/algo_data
:alt: Data to process
:class: bg-primary mb-1
:width: 300px
:align: center
```

OptiMask employs an iterative permutation-based algorithm to identify the largest NaN-free submatrix through these key steps:

1. **Problem Reduction**:
   - Isolate rows and columns containing at least one NaN value
   - Create a boolean mask matrix where True represents NaN positions

2. **Frontier Detection**:
   - Compute `hx`: column-wise highest NaN index (from bottom)
   - Compute `hy`: row-wise rightmost NaN index (from left)
   - These define the current "NaN frontier" of the matrix
  
   ```{image} figures/algo_0
   :alt: Step #1 and #2
   :class: bg-primary mb-1
   :width: 300px
   :align: center
   ```

3. **Permutation Phase**:
   - Alternately sort rows and columns to push NaN values toward a Pareto frontier
   - Even iterations: Sort columns by descending `hx`
   - Odd iterations: Sort rows by descending `hy`
   - Track all permutations applied during this process
   - Repeat until both `hx` and `hy` form non-increasing sequences
   - This indicates an optimal NaN frontier has been established

   | | | |
   |-|-|-|
   | ![I1](figures/algo_1) | ![I2](figures/algo_2) | ![I3](figures/algo_3) |

4. **Submatrix Extraction**:
   - Identify largest contiguous NaN-free rectangle in permuted space

   ```{image} figures/algo_result_permuted_space
   :alt: Optimask result in permuted space
   :class: bg-primary mb-1
   :width: 300px
   :align: center
   ```

   - Apply inverse permutations to map back to original row/column indices

   ```{image} figures/algo_result
   :alt: Optimask result
   :class: bg-primary mb-1
   :width: 300px
   :align: center
   ```
