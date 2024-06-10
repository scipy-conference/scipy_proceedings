---
# Ensure that this title is the same as the one in `myst.yml`
title: multinterp
subtitle: A Unified Interface for Multivariate Interpolation in the Scientific Python Ecosystem
abstract: |
  Multivariate interpolation is a fundamental tool in scientific computing, yet the Python ecosystem offers a fragmented landscape of specialized tools. This fragmentation hinders code reusability, experimentation, and efficient deployment across diverse hardware. To address this challenge, I've developed the `multinterp` package. It provides a unified interface for regular/rectilinear interpolation, supports serial (NumPy/SciPy), parallel (Numba), and GPU (CuPy, PyTorch, JAX) backends, and includes tools for multivalued interpolation and interpolation of derivatives.
exports:
  - format: pdf
---

## Introduction

The scientific Python ecosystem has a number of diverse tools for multivariate interpolation. However, these tools are scattered across multiple packages, each constructed for a specific purpose that prevents them from being easily used in other contexts.

Currently, there is no unified and comprehensive interface for interpolation with structured data (such as regular, rectilinear, and curvilinear) and unstructured data (as in irregular or scattered) that can be used with different hardware backends (e.g., serial, in parallel, or on a gpu) and software (numpy, numba, cupy, pytorch, jax) backends. The lack of this common platform makes it difficult for users to switch between different interpolation methods and backends, leading to inefficiencies and inconsistencies in their research.

This project aims to develop a comprehensive framework for multivariate interpolation for structured and unstructured data that can be used with different software technologies and hardware backends.

## Grid Interpolation

Functions are powerful mappings between sets of inputs and outputs, indicating how one set of values is related to another. Functions, however, are also infinitely dimensional, in that the inputs can range over an infinite number of values each mapping 1-to-1 (typically) to an infinite number of outputs. This makes it difficult to represent non-analytic functions in a computational environment, as we can only store a finite number of values in memory. For this reason, interpolation is a powerful tool in scientific computing, as it allows us to represent functions with a finite number of values and to approximate the function's behavior between these values.

The set of input values on which we know the function's output values is called a **grid**. A grid (or input grid) of values can be represented in many ways, depending on its underlying structure. The broadest categories of grids are regular or structured grids, and irregular or unstructured grids. Regular grids are those where the input values are arranged in a regular pattern, such as a triangle or a quadrangle. Irregular grids are those where the input values are not arranged in a particularly structured way and can seem to be scattered randomly across the input space.

As we might imagine, interpolation on regular grids is much easier than interpolation on irregular grids as we are able to exploit the structure of the grid to make predictions about the function's behavior between known values. Irregular grid interpolation is much more difficult, and often requires *regularizing* and/or *regression* techniques to make predictions about the function's behavior between known values. `multinterp` aims to provide a comprehensive set of tools for both regular and irregular grid interpolation, and we will discuss some of these tools in the following sections.

```{list-table} Grids and structures implemented in "multinterp".
:label: tbl:grids
:header-rows: 1
* - Grid
  - Structure
  - Geometry
* - Rectilinear
  - Regular
  - Rectangular mesh
* - Curvilinear
  - Regular 
  - Quadrilateral mesh
* - Unstructured
  - Irregular
  - Random 
```

## Rectilinear Interpolation

A *rectilinear* grid is a regular grid where the input values are arranged in a *rectangular* (in 2D) or *hyper-rectangular* (in higher dimensions) pattern. Moreover, they can be represented by the tensor product of monotonically increasing vectors along each dimension. For example, a 2D rectilinear grid can be represented by two 1D arrays of increasing values, such as $x = [x_0, x_1, x_2, \cdots, x_n]$ and $y = [y_0, y_1, y_2, \cdots, y_m]$, where $x_i > x_j$ and $y_i > y_j$ $\forall i > j$, and the input grid is then represented by $x \times y$ of dimensions $n \times m$. This allows for a very simple and efficient interpolation algorithm, as we can easily find and use the nearest known values to make predictions about the function's behavior in the unknown space.

```{figure} figures/BilinearInterpolation
:label: bilinear
:alt: A non-uniformly spaced rectilinear grid can be transformed into a uniformly spaced coordinate grid (and vice versa).
:align: center

A non-uniformly spaced rectilinear grid can be transformed into a uniformly spaced coordinate grid (and vice versa).
```

### Multilinear Interpolation

`multinterp` provides a simple and efficient implementation of *multilinear interpolation* for various backends (`numpy` (@Harris2020), `scipy` (@Virtanen2020), `numba` (@Lam2015), `cupy` (@Okuta2017), `pytorch` (@Paszke2019), and `jax` (@Bradbury2018)) via its `multinterp` function. From the remaining of this section, `multinterp` refers to the `multinterp` function in `multinterp` package, unless otherwise specified.

The main workhorse of `multinterp` is `scipy.ndimage`'s `map_coordinates` function. This function takes an array of **input** values and an array of **coordinates**, and returns the interpolated values at those coordinates. More specifically, the `input` array is the array of known values on the coordinate (index) grid, such that `input[i,j,k]` is the known value at the coordinate `(i,j,k)`. The `coordinates` array is an array of fractional coordinates at which we wish to know the values of the function, such as `coordinates[0] = (1.5, 2.3, 3.1)`. This indicates that we wish to know the value of the function between input index $i \in [1,2]$, $j \in [2,3]$, and $k \in [3,4]$. While `map_coordinates` is a powerful tool for coordinate grid interpolation, a typical function in question may not be defined on a coordinate grid. For this reason, we first need to find a mapping between the functional input grid and the coordinate grid, and then use `map_coordinates` to interpolate the function on the coordinate grid.

To do this, the `multinterp` package provides the function `get_coordinates`, which creates a mapping between the functional input grid that may be defined on the real numbers, and the coordinate grid which is defined on the positive integers including zero. In short, `multinterp` consists of two main functions: `get_coordinates` and `map_coordinates`, which together provide a powerful and flexible framework for multilinear interpolation on rectilinear grids.

An additional advantage of `scipy`'s `map_coordinates` is that it has been extended and integrated to various backends, such as `cupy` ([`cupyx.scipy.ndimage.map_coordinates`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.map_coordinates.html)) and `jax` ([`jax.scipy.ndimage.map_coordinates`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html)). This allows for easy and efficient interpolation on GPUs and TPUs, which can be orders of magnitude faster than CPU interpolation. For wider compatibility, `multinterp` also provides a `numba` and `pytorch` implementation of `map_coordinates`, which broadens the range of hardware that can be used for interpolation.

```{include} notebooks/Multivariate_Interpolation.ipynb
```

### Derivatives

The `multinterp` package also allows for the calculation of derivatives of the interpolated function defined on a rectilinear grid. This is done by using the function `get_grad`, which wraps numpy's `gradient` function to calculate the gradient of the interpolated function at the given coordinates.

```{include} notebooks/Multivariate_Interpolation_with_Derivatives.ipynb
```

### Multivalued Interpolation

Finally, the `multinterp` package allows for multivalued interpolation on rectilinear grids via the `MultivaluedInterp` class.

```{include} notebooks/Multivalued_Interpolation.ipynb
```

## Curvilinear Interpolation

A *curvilinear* grid is a regular grid whose input coordinates are *curved* or *warped* in some regular way, but can nevertheless be transformed back into a regular grid by simple transformations. That is, every quadrangle in the grid can be transformed into a rectangle by a remapping of its verteces. There are two approaches to curvilinear interpolation in `multinterp`: the first requires a "point location" algorithm to determine which quadrangle the input point lies in, and the second requires a "dimensional reduction" algorithm to generate an interpolated value from the known values in the quadrangle.

```{figure} figures/CurvilinearInterpolation
:label: curvilinear
:alt: A curvilinear grid can be transformed into a rectilinear grid by a simple remapping of its vertices.
:align: center

A curvilinear grid can be transformed into a rectilinear grid by a simple remapping of its vertices.
```

```{include} notebooks/Curvilinear_Interpolation.ipynb
```

## Unstructured Interpolation

```{figure} figures/UnstructuredInterpolation
:label: unstructured
:alt: Unstructured grids are irregular and often require a triangulation step which might be computationally expensive and time-consuming.
:align: center

Unstructured grids are irregular and often require a triangulation step which might be computationally expensive and time-consuming.
```

```{include} notebooks/Unstructured_Interpolation.ipynb
```

## Conclusion

Multivariate interpolation is a cornerstone of scientific computing, yet the Python ecosystem (@Oliphant2007) presents a fragmented landscape of tools. While individually powerful, these packages often lack a unified interface. This fragmentation makes it difficult for researchers to experiment with different interpolation methods, optimize performance across diverse hardware, and handle varying data structures (regular, rectilinear, curvilinear, unstructured).

The `multinterp` project seeks to change this. Its goal is to provide a unified, comprehensive, and flexible framework for multivariate interpolation in Python. This framework will streamline workflows by offering:

- Unified Interface: A consistent API for interpolation, regardless of data structure or desired backend, reducing the learning curve and promoting code reusability.
- Hardware Adaptability: Seamless support for CPU (NumPy, SciPy), parallel (Numba), and GPU (CuPy, PyTorch, JAX) backends, empowering users to optimize performance based on their computational resources.
- Broad Functionality: Tools for regular/rectilinear interpolation, multivalued interpolation, and derivative calculations, addressing a wide range of scientific problems.

The multinterp package (<https://github.com/alanlujan91/multinterp>) is currently in its beta stage.  It offers a strong foundation but welcomes community contributions to reach its full potential.  We invite collaboration to improve documentation, expand the test suite, and ensure the codebase aligns with the highest standards of Python package development.
