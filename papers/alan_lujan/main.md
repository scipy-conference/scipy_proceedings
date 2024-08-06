---
# Ensure that this title is the same as the one in `myst.yml`
title: multinterp
subtitle: A Unified Interface for Multivariate Interpolation in the Scientific Python Ecosystem
abstract: |
  Multivariate interpolation is a fundamental tool in scientific computing, yet the Python ecosystem offers a fragmented landscape of specialized tools. This fragmentation hinders code reusability, experimentation, and efficient deployment across diverse hardware. The `multinterp` package was developed to address this challenge. It provides a unified interface for regular/irregular interpolation, supports serial (using `numpy` and `scipy`), parallel (using `numba`), and GPU (using `cupy`, `pytorch`, and `jax`) backends, and includes tools for multivalued interpolation and interpolation of derivatives. This paper introduces `multinterp`, demonstrates its capabilities, and invites the community to contribute to its development.
exports:
  - format: pdf
---

## Introduction

The scientific Python ecosystem has a number of specialized tools for multivariate interpolation. However, these tools are scattered across multiple packages, each constructed for a specific purpose that prevents them from being easily used in other contexts. For example, `scipy.interpolate` provides a comprehensive set of interpolation tools, but these are limited to data and algorithms running on the CPU, and do not support GPU acceleration. On the other hand, `cupy` and `jax` provide GPU acceleration for some interpolation methods, but lack the comprehensive set of interpolation tools available in `scipy.interpolate`.

Currently, there is no unified and comprehensive interface for interpolation with structured data (such as regular, rectilinear, and curvilinear) and unstructured data (as in irregular or scattered) that can be used with different hardware backends (e.g., serial, in parallel, or on a gpu) and software (`numpy`, `numba`, `cupy`, `pytorch`, `jax`) backends. The lack of this common platform makes it difficult for researchers and practitioners to switch between different interpolation methods and backends, leading to inefficiencies and inconsistencies in their research.

This project aims to develop a comprehensive framework for multivariate interpolation for structured and unstructured data that can be used with different software technologies and hardware backends.

## Grid Interpolation

Functions are powerful mappings between sets of inputs and sets of outputs, indicating how one set of values is related to another. Functions, however, can also be thought of as infinitely dimensional, in that the inputs can range over an infinite number of values, each mapping (typically) 1-to-1 to an infinite number of outputs. For example, even the simple function $f(x) = x$ maps the uncontably infinite set of the reals to itself. Fortunately, this function is analytic, and can be represented summarily by the expression $f(x) = x$. This is also true of more general polynomials, trigonometric functions ($\sin$, $\cos$, $\tan$), other trancendental functions ($\log$, $\exp$), and their combinations. However, there are functions that can not be represented symbolicaly, and require explicit mappings. This makes it difficult to represent non-analytic functions in a computational environment, as we can only store a finite number of values in memory. For this reason, interpolation is a powerful tool in scientific computing, as it allows us to represent functions with a finite number of values and to approximate the function's behavior between these values.

The set of input values on which we know the function's output values is called a **grid**. A grid (or input grid) of values can be represented in many ways, depending on its underlying structure. The broadest categories of grids are regular or structured grids, and irregular or unstructured grids. Regular grids are those where the input values are arranged in a regular pattern, such as a triangle or a quadrangle. Irregular grids are those where the input values are not arranged in a particularly structured way and can seem to be scattered randomly across the input space.

As we might imagine, interpolation on regular grids is much easier than interpolation on irregular grids, as we are able to exploit the structure of the grid to make predictions about the function's behavior between known values. Irregular grid interpolation is much more difficult, and often requires _regularizing_ and/or _regression_ techniques to make predictions about the function's behavior between known values. `multinterp` aims to provide a comprehensive set of tools for both regular and irregular grid interpolation, and we will discuss some of these tools in the following sections.

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

A _rectilinear_ grid is a regular grid where the input values are arranged in a _rectangular_ (in 2D) or _hyper-rectangular_ (in higher dimensions) pattern. Moreover, they can be represented by the tensor product of monotonically increasing vectors along each dimension. For example, a 2D rectilinear grid can be represented by two 1D arrays of increasing values, such as $x = [x_0, x_1, x_2, \cdots, x_n]$ and $y = [y_0, y_1, y_2, \cdots, y_m]$, where $x_i < x_j$ and $y_i < y_j$ for all $i < j$, and the input grid is then represented by $x \times y$ of dimensions $n \times m$. This allows for a very simple and efficient interpolation algorithm, as we can easily find and use the nearest known values to make predictions about the function's behavior in the unknown space.

```{figure} figures/BilinearInterpolation
:label: bilinear
:alt: A non-uniformly spaced rectilinear grid can be transformed into a uniformly spaced coordinate grid (and vice versa).
:align: center

A non-uniformly spaced rectilinear grid can be transformed into a uniformly spaced coordinate grid (and vice versa).
```

### Multilinear Interpolation

`multinterp` provides a simple and efficient implementation of _multilinear interpolation_ for various backends (`numpy` (@Harris2020), `scipy` (@Virtanen2020), `numba` (@Lam2015), `cupy` (@Okuta2017), `pytorch` (@Paszke2019), and `jax` (@Bradbury2018)) via its `multinterp` function. From the remaining of this section, `multinterp` refers to the `multinterp` function in `multinterp` package, unless otherwise specified.

The main workhorse of `multinterp` is `scipy.ndimage`'s `map_coordinates` function. This function takes an array of **input** values and an array of **coordinates**, and returns the interpolated values at those coordinates. More specifically, the `input` array is the array of known values on the coordinate (index) grid, such that `input[i,j,k]` is the known value at the coordinate `(i,j,k)`. The `coordinates` array is an array of fractional coordinates at which we wish to know the values of the function, such as `coordinates[0] = (1.5, 2.3, 3.1)`. This indicates that we wish to know the value of the function between input index $i \in [1,2]$, $j \in [2,3]$, and $k \in [3,4]$. While `map_coordinates` is a powerful tool for coordinate grid interpolation, a typical function in question may not be defined on a coordinate grid. For this reason, we first need to find a mapping between the functional input grid and the coordinate grid, and then use `map_coordinates` to interpolate the function on the coordinate grid.

To do this, the `multinterp` package provides the function `get_coordinates`, which creates a mapping between the functional input grid that may be defined on the real numbers, and the coordinate grid which is defined on the positive integers including zero. In short, `multinterp` consists of two main functions: `get_coordinates` and `map_coordinates`, which together provide a powerful and flexible framework for multilinear interpolation on rectilinear grids.

An additional advantage of `scipy`'s `map_coordinates` is that it has been extended and integrated to various backends, such as `cupy` ([`cupyx.scipy.ndimage.map_coordinates`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.map_coordinates.html)) and `jax` ([`jax.scipy.ndimage.map_coordinates`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html)). This allows for easy and efficient interpolation on GPUs and TPUs, which can be orders of magnitude faster than CPU interpolation. For wider compatibility, `multinterp` also provides a `numba` and `pytorch` implementation of `map_coordinates`, which broadens the range of hardware that can be used for interpolation.

Below we demonstrate the use of the object oriented wrapper for `multinterp`, which we call `MultivariateInterp`, and compare timings to `scipy.interpolate`'s `RegularGridInterpolator`.

Suppose we are trying to approximate the following function at a set of points:

```python
def squared_coords(x, y):
    return x**2 + y**2
```

Our points will lie on a regular or rectilinear grid. A rectilinear grid may not be evenly spaced, but it can be reproduced by the cross product of $n$ 1-dimensional vectors. For example, let's assume we know the value of the function at the following points:

```python
x_grid = np.geomspace(1, 11, 11) - 1
y_grid = np.geomspace(1, 11, 11) - 1
x_mat, y_mat = np.meshgrid(x_grid, y_grid, indexing="ij")

z_mat = squared_coords(x_mat, y_mat)
```

Notice that the points are not evenly spaced, which is achieved with the use of `np.geomspace`. So now, we know the value of the function `squared_coords` and have labeled them as `z_mat`. Now suppose that we would like to know the value of the function at the points `x_new` and `y_new` which create an evenly spaced regular grid.

```python
x_new, y_new = np.meshgrid(
    np.linspace(0, 10, 11),
    np.linspace(0, 10, 11),
    indexing="ij",
)
```

We can use scipy's `RegularGridInterpolator` to interpolate the function at these new points and then we can plot the results.

```python
interp = RegularGridInterpolator([x_grid, y_grid], z_mat)
z_interp = interp(np.column_stack((x_new.ravel(), y_new.ravel()))).reshape(x_new.shape)
```

```python
%%timeit
z_interp = interp(np.column_stack((x_new.ravel(), y_new.ravel()))).reshape(x_new.shape)
```

```{embed} #fig:multivariate_regular
:remove-input: true
```

Here we introduce `MultivariateInterp`, which brings additional features and speed improvements. The key feature of `MultivariateInterp` is its `backend` parameter, which can be set to `scipy`, `numba`, or `cupy`, among others. This allows the user to specify the backend device for the interpolation. Using `MultivariateInterp` mirrors the use of `RegularGridInterpolator` very closely.

```python
mult_interp = MultivariateInterp(z_mat, [x_grid, y_grid])
z_mult_interp = mult_interp(x_new, y_new)
```

```python
%%timeit
z_mult_interp = mult_interp(x_new, y_new)
```

```{embed} #fig:multivariate_interp
:remove-input: true
```

As we see above, `MultivariateInterp` is faster than `RegularGridInterpolator`, even with the default `backend="scipy"`. Moreover, the speed of `MultivariateInterp` is highly dependent on the number of points in the grid and the backend device. For example, for a large number of points, `MultivariateInterp` with `backend='numba'` can be shown to be significantly faster than `RegularGridInterpolator`.

```python
gpu_interp = MultivariateInterp(z_mat, [x_grid, y_grid], backend="cupy")
z_gpu_interp = gpu_interp(x_new, y_new).get()  # Get the result from GPU
```

```python
%%timeit
z_gpu_interp = gpu_interp(x_new, y_new).get()  # Get the result from GPU
```

We can test the results of `MultivariateInterp` against `RegularGridInterpolator`, and we see that the results are almost identical.

```python
np.allclose(z_interp - z_gpu_interp, z_mult_interp - z_gpu_interp)
```

To experiment with `MultivariateInterp` and evaluate the conditions which make it faster than `RegularGridInterpolator`, we can create a grid of data points and interpolation points and then time the interpolation on different backends.

```python
n = 35
grid_max = 300
grid = np.linspace(10, grid_max, n, dtype=int)
fast = np.empty((n, n))
scipy = np.empty_like(fast)
parallel = np.empty_like(fast)
gpu = np.empty_like(fast)
```

We will use the following function to time the execution of the interpolation.

```python
def timeit(interp, x, y, min_time=1e-6):
    if isinstance(interp, RegularGridInterpolator):
        start = time()
        points = np.column_stack((x.ravel(), y.ravel()))
        interp(points).reshape(x.shape)
    else:
        interp.compile()
        start = time()
        interp(x, y)

    elapsed_time = time() - start
    return max(elapsed_time, min_time)
```

For different number of data points and approximation points, we can time the interpolation on different backends and use the results of `RegularGridInterpolator` to normalize the results. This will give us a direct comparison of the speed of `MultivariateInterp` and `RegularGridInterpolator`.

```python
for i, j in product(range(n), repeat=2):
    data_grid = np.linspace(0, 10, grid[i])
    x_cross, y_cross = np.meshgrid(data_grid, data_grid, indexing="ij")
    z_cross = squared_coords(x_cross, y_cross)

    approx_grid = np.linspace(0, 10, grid[j])
    x_approx, y_approx = np.meshgrid(approx_grid, approx_grid, indexing="ij")

    fast_interp = RegularGridInterpolator([data_grid, data_grid], z_cross)
    time_norm = timeit(fast_interp, x_approx, y_approx)
    fast[i, j] = time_norm

    scipy_interp = MultivariateInterp(z_cross, [data_grid, data_grid], backend="scipy")
    scipy[i, j] = timeit(scipy_interp, x_approx, y_approx) / time_norm

    par_interp = MultivariateInterp(z_cross, [data_grid, data_grid], backend="numba")
    parallel[i, j] = timeit(par_interp, x_approx, y_approx) / time_norm

    gpu_interp = MultivariateInterp(z_cross, [data_grid, data_grid], backend="cupy")
    gpu[i, j] = timeit(gpu_interp, x_approx, y_approx) / time_norm
```

```{embed} #fig:multivariate_speed
:remove-input: true
```

As we can see from the results, `MultivariateInterp` is faster than `RegularGridInterpolator` depending on the number of points and the backend device. A value of 1 represents the same speed as `RegularGridInterpolator`, while a value less than 1 is faster (in red) and a value greater than 1 is slower (in blue).

For `backend="scipy"`, `MultivariateInterp` is (much) slower when the number of approximation points that need to be interpolated is very small, as seen by the deep blue areas. When the number of approximation points is moderate to large, however, `MultivariateInterp` is about as fast as `RegularGridInterpolator`.

For `backend="numba"`, `MultivariateInterp` is slightly faster when the number of data points with known function value are greater than the number of approximation points that need to be interpolated. However, `backend='parallel'` still suffers from the high overhead when the number of approximation points is small.

For `backend="cupy"`, `MultivariateInterp` is much slower when the number of data points with known function value are small. This is because of the overhead of copying the data to the GPU. However, `backend='numba'` is significantly faster for any other case when the number of approximation points is large regardless of the number of data points.

### Derivatives

The `multinterp` package also allows for the calculation of derivatives of the interpolated function defined on a rectilinear grid. This is done by using the function `get_grad`, which wraps numpy's `gradient` function to calculate the gradient of the interpolated function at the given coordinates.

Consider the following function along with its analytical derivatives:

```python
def trig_func(x, y):
    return y * np.sin(x) + x * np.cos(y)


def trig_func_dx(x, y):
    return y * np.cos(x) + np.cos(y)


def trig_func_dy(x, y):
    return np.sin(x) - x * np.sin(y)
```

First, we create a sample input gradient and evaluate the function at those points. Notice that we are not using the analytical derivatives to create the interpolation function. Instead, we will use these to compare the results of the numerical derivatives.

```python
x_grid = np.geomspace(1, 11, 1000) - 1
y_grid = np.geomspace(1, 11, 1000) - 1
x_mat, y_mat = np.meshgrid(x_grid, y_grid, indexing="ij")

z_mat = trig_func(x_mat, y_mat)
```

Now, we generate a different grid which will be used as our query points.

```python
x_new, y_new = np.meshgrid(
    np.linspace(0, 10, 1000),
    np.linspace(0, 10, 1000),
    indexing="ij",
)
```

Now, we can compare our interpolation function with the analytical function, and see that these are very close to each other.

```python
mult_interp = MultivariateInterp(z_mat, [x_grid, y_grid], backend="cupy")
z_mult_interp = mult_interp(x_new, y_new).get()
z_true = trig_func(x_new, y_new)
```

```{embed} #fig:multivariate
:remove-input: true
```

To evaluate the numerical derivatives, we can use the method `.diff(argnum)` of `MultivariateInterp` which provides an object oriented way to compute numerical derivatives. For example, calling `mult_interp.diff(0)` returns a `MultivariateInterp` object that represents the numerical derivative of the function with respect to the first argument on the same input grid.

We can now compare the numerical derivatives with the analytical derivatives, and see that these are indeed very close to each other.

```python
dfdx = mult_interp.diff(0)
z_dfdx = dfdx(x_new, y_new).get()
dfdx_true = trig_func_dx(x_new, y_new)
```

```{embed} #fig:multivariate_dx
:remove-input: true
```

Similarly, we can compute the derivatives with respect to the second argument, and see that it produces an accurate result.

```python
dfdy = mult_interp.diff(1)
z_dfdy = dfdy(x_new, y_new).get()
dfdy_true = trig_func_dy(x_new, y_new)
```

```{embed} #fig:multivariate_dy
:remove-input: true
```

The choice of returning object oriented intepolation functions for the numerical derivatives is very useful, as it allows for re-usability without re-computation and easy chaining of operations. For example, we can compute the second derivative of the function with respect to the first argument by calling `mult_interp.diff(0).diff(0)`.

### Multivalued Interpolation

Finally, the `multinterp` package allows for multivalued interpolation on rectilinear grids via the `MultivaluedInterp` class.

Consider the following multivalued function:

```python
def squared_coords(x, y):
    return x**2 + y**2


def trig_func(x, y):
    return y * np.sin(x) + x * np.cos(y)


def multivalued_func(x, y):
    return np.array([squared_coords(x, y), trig_func(x, y)])
```

As before, we can generate values on a sample input grid, and create a grid of query points.

```python
x_grid = np.geomspace(1, 11, 1000) - 1
y_grid = np.geomspace(1, 11, 1000) - 1
x_mat, y_mat = np.meshgrid(x_grid, y_grid, indexing="ij")

z_mat = multivalued_func(x_mat, y_mat)

x_new, y_new = np.meshgrid(
    np.linspace(0, 10, 1000),
    np.linspace(0, 10, 1000),
    indexing="ij",
)
```

`MultivaluedInterp` can easily interpolate the function at the query points and avoid repeated calculations.

```python
from multinterp.rectilinear._multi import MultivaluedInterp

mult_interp = MultivaluedInterp(z_mat, [x_grid, y_grid], backend="cupy")
z_mult_interp = mult_interp(x_new, y_new).get()
z_true = multivalued_func(x_new, y_new)
```

```{embed} #fig:multivalued
:remove-input: true
```

## Curvilinear Interpolation

A _curvilinear_ grid is a regular grid whose input coordinates are _curved_ or _warped_ in some regular way, but can nevertheless be transformed back into a regular grid by simple transformations. That is, every quadrangle in the grid can be transformed into a rectangle by a remapping of its verteces. There are two approaches to curvilinear interpolation in `multinterp`: the first requires a "point location" algorithm to determine which quadrangle the input point lies in, and the second requires a "dimensional reduction" algorithm to generate an interpolated value from the known values in the quadrangle.

```{figure} figures/CurvilinearInterpolation
:label: curvilinear
:alt: A curvilinear grid can be transformed into a rectilinear grid by a simple remapping of its vertices.
:align: center

A curvilinear grid can be transformed into a rectilinear grid by a simple remapping of its vertices.
```

Suppose we have a collection of values for an unknown function along with their respective coordinate points. For illustration, assume the values come from the following function:

```python
def function_1(x, y):
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2) ** 2
```

The points are randomly scattered in the unit square and therefore have no regular structure. This is achieved by randomly shifting a well structured grid at every point.

```python
rng = np.random.default_rng(0)
warp_factor = 0.01
x_list = np.linspace(0, 1, 20)
y_list = np.linspace(0, 1, 20)
x_temp, y_temp = np.meshgrid(x_list, y_list, indexing="ij")
rand_x = x_temp + warp_factor * (rng.random((x_list.size, y_list.size)) - 0.5)
rand_y = y_temp + warp_factor * (rng.random((x_list.size, y_list.size)) - 0.5)
values = function_1(rand_x, rand_y)
```

Now suppose we would like to interpolate this function on a rectilinear grid, which is known as "regridding".

```python
grid_x, grid_y = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100),
    indexing="ij",
)
```

To do this, we use `multinterp`'s `Warped2DInterp` and `Curvilinear2DInterp` classes. The class takes the following arguments:

- `values`: an ND-array of values for the function at the points
- `grids`: a list of ND-arrays of coordinates for the points
- `backend`: the backend to use for interpolation, currently only `scipy` and `numba` are supported for `Warped2DInterp` and only `scipy` is supported for `Curvilinear2DInterp`

```python
from multinterp.curvilinear import Curvilinear2DInterp, Warped2DInterp

warped_interp = Warped2DInterp(values, (rand_x, rand_y), backend="numba")
warped_interp.warmup()
```

Once we create the interpolator objects, we can evaluate the functions on the query grids and compare their time performance.

```python
start = time()
warped_grid = warped_interp(grid_x, grid_y)
print(f"Warped interpolation took {time() - start:.5f} seconds")
```

```python
curvilinear_interp = Curvilinear2DInterp(values, (rand_x, rand_y))
start = time()
curvilinear_grid = curvilinear_interp(grid_x, grid_y)
print(f"Curvilinear interpolation took {time() - start:.5f} seconds")
```

Now we can compare the results of the interpolation with the original function. Below we plot the original function and the sample points that are known. Notice that the points are almost rectilinear, but have been randomly shifted to create a more challenging interpolation problem.

```{embed} #fig:curvilinear_original

```

Then, we can look at the result for each method of interpolation and compare it to the original function.

```{embed} #fig:curvilinear_result

```

In short, `multinterp`'s `Warped2DInterp` and `Curvilinear2DInterp` classes are useful for interpolating functions on curvilinear grids which have a quadrilateral structure but are not perfectly rectangular.

## Unstructured Interpolation

```{figure} figures/UnstructuredInterpolation
:label: unstructured
:alt: Unstructured grids are irregular and often require a triangulation step which might be computationally expensive and time-consuming.
:align: center

Unstructured grids are irregular and often require a triangulation step which might be computationally expensive and time-consuming.
```

Suppose we have a collection of values for an unknown function along with their respective coordinate points. For illustration, assume the values come from the following function:

```python
def function_1(u, v):
    return u * np.cos(u * v) + v * np.sin(u * v)
```

The points are randomly scattered within a square and therefore have no regular structure.

```python
rng = np.random.default_rng(0)
rand_x, rand_y = rng.random((2, 1000)) * 3
values = function_1(rand_x, rand_y)
```

Now suppose we would like to interpolate this function on a rectilinear grid, which is known as "regridding".

```python
grid_x, grid_y = np.meshgrid(
    np.linspace(0, 3, 100),
    np.linspace(0, 3, 100),
    indexing="ij",
)
```

To do this, we use `multinterp`'s `UnstructuredInterp` class. The class takes the following arguments:

- `values`: an ND-array of values for the function at the points
- `grids`: a list of ND-arrays of coordinates for the points
- `method`: the interpolation method to use, with options "nearest", "linear", "cubic" (for 2D only), and "rbf". The default is `'linear'`.

The `UnstructuredInterp` class is an object oriented wrapper around `scipy.interpolate`'s functions for multivariate interpolation on unstructured data, which are `NearestNDInterpolator`, `LinearNDInterpolator`, `CloughTocher2DInterpolator`, and `RBFInterpolator`. The advantage of using `multinterp`'s `UnstructuredInterp` class is that it provides a consistent interface for all of these methods, making it easier to switch between them and other interpolators in the `multinterp` package.

```python
nearest_interp = UnstructuredInterp(values, (rand_x, rand_y), method="nearest")
linear_interp = UnstructuredInterp(values, (rand_x, rand_y), method="linear")
cubic_interp = UnstructuredInterp(values, (rand_x, rand_y), method="cubic")
rbf_interp = UnstructuredInterp(values, (rand_x, rand_y), method="rbf")
```

Once we create the interpolator objects, we can use them using the `__call__` method which takes as many arguments as there are dimensions.

```python
nearest_grid = nearest_interp(grid_x, grid_y)
linear_grid = linear_interp(grid_x, grid_y)
cubic_grid = cubic_interp(grid_x, grid_y)
rbf_grid = rbf_interp(grid_x, grid_y)
```

Now we can compare the results of the interpolation with the original function. Below we plot the original function and the sample points that are known.

```{embed} #fig:unstructured_original
:remove-input: true
```

Then, we can look at the result for each method of interpolation and compare it to the original function.

```{embed} #fig:unstructured_interpolated
:remove-input: true
```

Finally, `multinterp` also provides a set of interpolators organized around the concept of _regression_. As a demonstration, below we use a `RegressionUnstructuredInterp` interpolator which uses a Gaussian Process regression model from `scikit-learn` (@Pedregosa2011) to interpolate the function defined on the unstructured grid. The `RegressionUnstructuredInterp` class takes the same arguments as the `UnstructuredInterp` class, but it additionally requires the user to specify the regression `model` to use.

```python
from multinterp import RegressionUnstructuredInterp

gaussian_interp = RegressionUnstructuredInterp(
    values,
    (rand_x, rand_y),
    model="gaussian-process",
    std=True,
)

gaussian_grid = gaussian_interp(grid_x, grid_y)


```

```{embed} #fig:unstructured_gp
:remove-input: true
```

## Conclusion

Multivariate interpolation is a cornerstone of scientific computing, yet the Python ecosystem (@Oliphant2007) presents a fragmented landscape of tools. While individually powerful, these packages often lack a unified interface. This fragmentation makes it difficult for researchers to experiment with different interpolation methods, optimize performance across diverse hardware, and handle varying data structures (regular, rectilinear, curvilinear, unstructured).

The `multinterp` project seeks to change this. Its goal is to provide a unified, comprehensive, and flexible framework for multivariate interpolation in Python. This framework will streamline workflows by offering:

- Unified Interface: A consistent API for interpolation, regardless of data structure or desired backend, reducing the learning curve and promoting code reusability.
- Hardware Adaptability: Seamless support for CPU (NumPy, SciPy), parallel (Numba), and GPU (CuPy, PyTorch, JAX) backends, empowering users to optimize performance based on their computational resources.
- Broad Functionality: Tools for regular/rectilinear interpolation, multivalued interpolation, and derivative calculations, addressing a wide range of scientific problems.

The multinterp package (<https://github.com/alanlujan91/multinterp>) is currently in its beta stage. It offers a strong foundation but welcomes community contributions to reach its full potential. We invite collaboration to improve documentation, expand the test suite, and ensure the codebase aligns with the highest standards of Python package development.
