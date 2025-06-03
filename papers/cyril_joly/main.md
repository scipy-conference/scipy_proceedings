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

### Overview
OptiMask employs a heuristic approach to efficiently approximate the largest NaN-free submatrix problem. The algorithm combines iterative permutation strategies with randomized restarts to navigate the solution space, avoiding the computational intractability of exact methods while maintaining high solution quality.

### Key Steps

1. **Problem Reduction**:
   - Isolate rows and columns containing NaN values (reducing the problem size)
   - Convert the input matrix into a boolean mask where `True` represents NaN positions

2. **Permutation and Frontier Formation**:
   - Randomly permute rows and columns to redistribute NaN positions
   - Iteratively sort rows/columns by their "NaN height" (the position of the last NaN in each row/column)
   - Alternate between row and column permutations until a Pareto-optimal frontier of NaN values emerges (a monotonically decreasing pattern)

3. **Rectangle Identification**:
   - Once the frontier is established, find the largest contiguous rectangle in the upper-right corner of the permuted matrix that contains no NaN values
   - This rectangle corresponds to the optimal rows/columns to keep in the original matrix

4. **Random Restarts**:
   - Repeat the permutation process multiple times (`n_tries`) with different random seeds
   - Track and return the best solution found across all trials

### Mathematical Formulation

For a matrix `A` with NaN positions marked in boolean matrix `M`:

1. **Height Vectors**:
   - Column heights: `h_x[j] = max{i | M[i,j] = True}` for each column `j`
   - Row heights: `h_y[i] = max{j | M[i,j] = True}` for each row `i`

2. **Pareto Condition**:
   The algorithm seeks permutations where both `h_x` and `h_y` are monotonically decreasing, creating a staircase-like NaN frontier.

3. **Objective Function**:
   Maximize the area:  
   `max (m - k) × (n - l)`  
   where `k` rows and `l` columns are removed, subject to the remaining submatrix being NaN-free.

### Implementation Details

The algorithm leverages several optimizations:

1. **Numba Acceleration**:
   - Critical operations (permutation applications, height calculations) are compiled via Numba for performance
   - Parallel processing for independent operations

2. **Efficient Data Structures**:
   - Sparse representation of NaN positions
   - In-place permutations to minimize memory usage

3. **Early Termination**:
   - The iteration stops when either:
     - The Pareto condition is satisfied
     - A maximum number of steps (`max_steps`) is reached

### Pseudocode

```python
def solve(matrix):
    best_area = 0
    best_solution = None
    
    for trial in range(n_tries):
        # Random initialization
        permuted_matrix = random_permutation(matrix)
        
        # Iterative optimization
        while not pareto_optimal(permuted_matrix) and steps < max_steps:
            if step % 2 == 0:
                permuted_matrix = sort_columns_by_height(permuted_matrix)
            else:
                permuted_matrix = sort_rows_by_height(permuted_matrix)
            
            current_area = calculate_max_rectangle(permuted_matrix)
            
            if current_area > best_area:
                best_area = current_area
                best_solution = extract_solution(permuted_matrix)
    
    return best_solution
```

### Complexity Analysis

- **Time Complexity**:
  - Each trial requires O(k × max_steps) operations, where k is the number of NaN-containing rows/columns
  - Typical performance is linear in the number of NaN cells for sparse matrices

- **Space Complexity**:
  - O(m + n) additional space for tracking permutations and heights
  - Original matrix is not modified

### Advantages Over Exact Methods

1. **Scalability**:
   - Handles matrices up to 10^5 × 10^3 efficiently
   - Memory usage remains practical for large datasets

2. **Adaptability**:
   - Automatically adjusts to matrix aspect ratios
   - Effective for both random and structured NaN patterns

3. **Tunable Precision**:
   - Solution quality can be improved by increasing `n_tries`
   - Provides explicit control over computation time/quality trade-off

### Visualization of the Process

The algorithm's progression can be visualized through the NaN frontier at each step:

1. **Initial Random Permutation**:
   ```python
   plot(xp, title="Initial NaN Distribution")
   ```

2. **After Column Sorting**:
   ```python
   plot(xp, title="After Column Permutation")
   ```

3. **Final Pareto Frontier**:
   ```python
   plot(xp, title="Optimal NaN Frontier")
   ```

The final frontier shows a clear descending pattern, enabling straightforward identification of the maximal NaN-free rectangle.
