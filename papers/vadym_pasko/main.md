---
# Ensure that this title is the same as the one in `myst.yml`
title: "Enhancing Curve Fitting with SciPy: Interactive Spline Modeling and Reproducibility with SplineCloud"
abstract: |
  Curve fitting is a fundamental task in data science, engineering, and scientific computing, enabling researchers to extract meaningful relationships from data. However, selecting and tuning the right fitting model for complex, noisy, or multidimensional data remains a significant challenge. SciPy plays a critical role in addressing these challenges by providing robust spline fitting methods that offer flexibility and precision. Yet, fine-tuning spline parameters, ensuring stability in extrapolation, and sharing fitted models for reproducibility remain open problems.
  
  In order to address these challenges, we developed SplineCloud - an open platform that provides interactive spline fitting capabilities and uses SciPy on the backend. SplineCloud allows the construction, analysis, and exchange of spline-based regression models using SciPy’s `interpolate` module. SplineCloud’s curve fitting tool extends the capabilities of SciPy spline fitting methods by enabling researchers to fine-tune spline parameters: knot vector, control points, and degree interactively, instantly analyzing the accuracy of models. Models constructed on the platform obtain their unique identifiers and become reusable in code, fostering better collaboration and knowledge transfer. Reusability of spline curves and underlying datasets in code is enabled via the open-source SplineCloud client library called `splinecloud-scipy`, which is also based on SciPy. The proposed approach of interactive cloud-based fitting improves data processing workflows, allows separating data preparation and approximation routines from the main code, and brings FAIR principles to the curve fitting, enabling researchers to construct and share libraries of empirical data relations.
---

## Introduction

The adoption of FAIR (Findable, Accessible, Interoperable, Reusable) data principles has fundamentally transformed research practices across scientific disciplines. Initially designed to enhance research data management and increase transparency, these principles now extend to a broad range of digital scientific artifacts, including computational models, source code, 3D models, and other digital objects. Widespread adoption of FAIR principles fosters improved reproducibility and facilitates the integration and extension of scientific results. However, not all digital objects achieve the same level of FAIRness, and many platforms lack support for the diverse range of data formats and objects used in science and engineering.

This paper focuses on enhancing reusability—and thereby reproducibility—in the context of regression modeling, a technique used to mathematically describe the relationship between dependent and independent variables. This process, commonly known as curve fitting, is widely employed in data science, engineering, and scientific computing, particularly for processing experimental data and extracting interpretable models for further analysis. Of particular practical interest is spline fitting, which effectively models complex empirical relationships without explicit underlying mathematical laws. A formal overview of splines and spline fitting is provided in Sections 2 and 3.

SciPy’s `interpolate` module offers extensive functionality for constructing splines, from simple interpolating curves to advanced smoothing splines and parametric representations. Its spline fitting methods provide robust algorithms widely used to model complex, noisy, or multidimensional data dependencies. Nonetheless, the capabilities of splines cannot be fully leveraged through purely programmatic interfaces. In contrast, modern graphical modeling software with interactive spline manipulation affords a level of control and intuitive adjustment not achievable by traditional automatic fitting methods. This, together with a problem of reusability of obtained models, their accessibility, and interoperability, leaves a way for thorough considerations towards alternative approaches to the curve fitting process. 

These considerations eventually led to the development of a dedicated platform for interactive spline fitting, which incorporates FAIR principles and adapts them to the regression models. Built upon SciPy’s routines, SplineCloud enables collaborative, transparent, and reusable curve fitting. Sections 4 through 8 of this paper detail the similarities and distinctions between SciPy’s and SplineCloud’s approaches, illustrating a novel paradigm for collaborative data fitting.


## Some Theoretical Background Behind Splines

Splines are piecewise-defined functions used extensively in numerical analysis, computer-aided geometric design, and data fitting. The fundamental idea behind spline fitting process is to construct a smooth function that matches a set of data points or satisfies a set of constraints, while preserving computational efficiency and numerical stability. Splines can be represented in several forms, each suitable for different applications: Piecewise Polynomial, Hermite, Bézier, B-spline, NURBS (Non-Uniform Rational B-Splines). Amogst these forms a special place takes B-Spline representation - where spline function (or curve) is expressed as a linear combination of basis functions.

A distinction must be made between spline functions and parametric spline curves:

- **A spline function** is a scalar-valued function $S(x)$, defined over a single independent variable $x$, typically used in interpolation or regression of scalar data.

- **A parametric spline curve** defines a vector-valued mapping from a scalar parameter $t$ to a multidimensional space:

```{math}
\mathbf{C}(t) = \left( x(t), y(t), z(t), \dots \right)

```
where each coordinate function $x(t), y(t), \dots$ is a spline function in $t$. This is the common representation in geometric modeling and computer graphics.

A particularly efficient and widely used spline representation is the B-spline (basis spline). B-splines are defined via a set of control points ${ \mathbf{P}_i }$ and a knot vector ${ t_i }$, and provide local control, numerical stability, and efficient evaluation. A B-spline curve of degree $k$ is defined as:

```{math}
:label: b-spline
\mathbf{C}(t) = \sum_{i=0}^{n} \mathbf{P}_i B_{i,k}(t)
```
where $\mathbf{P}_i$ are control points, $B_{i,k}(t)$ are the B-spline basis functions of degree $k$, defined recursively by the Cox–de Boor formula [@doi:10.1093/imamat/10.2.134; @doi:10.1007/978-1-4612-6333-3]:

```{math}
:label: cox-de-boor
B_{i,k}(t) =
\frac{t - t_i}{t_{i+k} - t_i} B_{i,k-1}(t)
+
\frac{t_{i+k+1} - t}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(t)
```

In {ref}`cox-de-boor` possible divisions by zero are resolved by the convention that ‘anything divided by zero is zero’ [@lyche-morken]. This formulation separates geometry (control points) from basis functions, allowing flexible manipulation and efficient computation of spline curves.

### Applications and Benefits of Parametric Splines

Parametric splines are widely used in computer graphics, computer-aided desig (CAD), and more general geometric modeling, where the goal is to design and manipulate accurate and smooth curves and surfaces in two or three dimensions. Their unique properties provide the following advantages:

- **Geometric Flexibility**. Since parametric curves are not constrained to be functions in the $y = f(x)$ form, they can represent vertical segments, loops, cusps, and other geometries that a function cannot.

- **Smooth Multi-dimensional Representation**. By treating each coordinate as an independent spline, parametric curves offer uniform control over the curve shape in all spatial directions.

- **Local Control**. In representations such as B-splines or NURBS, moving one control point affects only a portion of the curve, enabling precise local edits without changing the global shape.

- **Uniform Parameterization**. The parameter $t$ typically varies over a fixed interval $[t_0, t_n]$, which makes operations like subdivision, evaluation, and rendering more efficient and robust.

These properties make parametric splines ideal for applications in: 3D modeling and animation, font and character design, surface generation (via tensor product surfaces), and industrial design (automotive, aerospace, etc.).


An intuitive visual explanation of splines, different forms of their representation and unique properties is given in the video by Freya Holmér [@holmer-continuity-of-splines].

Despite their geometric advantages, parametric splines are rarely used in statistical data fitting and regression analysis. The primary limitation is the parameterization problem: for observed data points $(x_i, y_i)$, there is no natural way to assign parameter values $t_i$ that correspond meaningfully to the underlying data relationship. Additionally, parametric splines do not provide the direct functional relationship $y = f(x)$ that is typically required for prediction and statistical inference in data analysis contexts. This is why spline functions are often a more popular choice. 

## Spline Fitting With SciPy

SciPy provides a robust and flexible set of spline fitting tools for both **interpolation** and **approximation** through the `scipy.interpolate` module.

In term of univariate fitting, SciPy supports several spline fitting methods, including:

**Interpolating splines** - exact fit to the data {ref}`fig:scipy-spline-fitting` (a).

This constructs a spline $S(x)$ such that:

```{math}
S(x_i) = y_i \quad \text{for all } i
```
and ensures continuity of first and second derivatives ($C^2$ continuity for cubic splines).


**Smoothing splines** - approximate fit with smoothness penalty {ref}`fig:scipy-spline-fitting` (b).

The smoothing spline minimizes the penalized least-squares objective:
```{math}
\min_S \left\{ \sum_{i=1}^n \left( y_i - S(x_i) \right)^2 + \lambda \int_a^b \left( S''(x) \right)^2 dx \right\}
```
where $\lambda$ is a regularization parameter related to s.

**Least squares splines** - approximate fit with squared residual minimization penalty {ref}`fig:scipy-spline-fitting` (c). 

LSQ spline is constructed to minimize the sum of squared residuals:

```{math}
\min_S \sum_{i=1}^n \left( y_i - S(x_i) \right)^2
```

This method is useful when knot positions reflect known features or transitions in the data. It can also be used in a pair with custom algorithms that select the best knot position solving optimization problem to minimize the sum of residuals or another objective function that reflects a measure of the fit quality.

**Parametric splines**. Used to fit looped curve, isolines, or 3D curves.

:::{figure}
:label: fig:scipy-spline-fitting
SciPy spline fitting.
<table>
<tr>
<td style="text-align: center;"><img src="interp_spline_fitting_scipy.png" height="220px"/>(a) Simple interpolaing cubic spline.</td>
<td style="text-align: center;"><img src="smooth_spline_fitting_scipy.png" height="220px"/>(b) Smoothing cubic spline.</td>
<td style="text-align: center;"><img src="lsq_spline_fitting_scipy.png" height="220px"/>(c) Least-Squares cubic spline.</td>
</tr>
</table>
:::

### Pain Points of Pure Programmable Spline Fitting

SciPy provides a programmatic interface to robust spline fitting methods suitable for a variety of tasks, particularly when working with complex experimental or statistical data. However, despite offering multiple fitting methods and adjustable parameters, selecting an appropriate combination can be challenging. As noted in [@pasko-blog-post-2015], issues such as overfitting ({ref}`fig:overfitting_extrapolation` a) and extrapolation control ({ref}`fig:overfitting_extrapolation` b) lack straightforward solutions and often require multiple iterations, either with visual inspection or advanced scripting.

:::{figure}
:label: fig:overfitting_extrapolation
Simple interpolaing cubic spline.
<table>
<tr>
<td style="text-align: center;"><img src="overfitting.png" height="230px"/>(a) Typical overfitting issue - RMSE is minimal, but interpolation error is high.</td>
<td style="text-align: center;"><img src="bad_extrapolation.png" height="230px"/>(b) Typical extrapolation issue - curve behavior is not following the data trend beyound the given interval.</td>
</tr>
</table>
:::

The mathematical formulation of B-splines—especially parametric B-splines and NURBS—enables fine control over curve shapes through manipulation of control points and knot vectors. While this functionality is widely leveraged in computer graphics and CAD applications, it is not supported in SciPy.

Introducing interactive control over curve geometry can improve fitting accuracy and reduce the time spent tuning parameters in search of acceptable results from automatic fitting routines.

Another limitation is the lack of portability and reproducibility of fitted models. SciPy spline objects are tightly coupled to the Python runtime and local environment. Although spline parameters (order, knot vector, and coefficients) can be exported, reconstructing models from these components requires an understanding of B-spline structure. Model serialization is also possible but may not be practical in all scenarios.

These limitations hinder reproducibility and lead to duplicated effort, where researchers repeatedly perform the same fitting procedures on shared datasets, reimplementing logic and parameters independently. Even when code and data are shared, fitted models often remain environment-specific and are difficult to reuse across projects.

Overcoming these challenges requires workflows in which spline models can be edited interactively, stored independently from source code and raw data, and reused across tools and teams.

These needs motivated the development of SplineCloud, a platform that extends SciPy’s spline fitting with interactivity, transparency, and full model lifecycle support.


## Introducing SplineCloud

SplineCloud is an open platform for formalized knowledge exchange (free to use for everyone). It is a cloud solution designed to make technical data, like results of simulations, tests and modeling more accessible and reusable in computer code. The platform has integrated tools that allow formalizing and organizing data in topical repositories. Its architecture and instruments help reduce the routine overhead of data collection, processing, and model reconstruction in code.

SplineCloud’s main features are: plot digitizer, advanced spline fitting tool, open API, and client libraries for Python and MATLAB (support of other languages is planned for future implementation). It addresses all four problems of FAIR data: findability, accessibility, interoperability, and reusability for a specific type of data: experimental, simulation, and statistical results, which can have either tabular or graphical form. 

The philosophy behind the platform is in representing data relations in the form of spline models, which can be reused in code to omit tedious and repetitive operations on data extraction and fitting, which usually complicate mathematical modeling processes. In this way, SplineCloud can be considered as a repository-based open library of functional relations. The typical user journey on the platform is given on the {ref}`fig:workflow_scheme`.

:::{figure} workflow_scheme.png
:label: fig:workflow_scheme
:width: 350px
SplineCloud workflow scheme
:::

The platform’s data structure is given in {ref}`fig:class_diagram`. This data structure models a hierarchical system for managing technical datasets within user-owned repositories. Each repository contains multiple data files, which serve as sources for datasets; these in turn produce subsets of structured (clean) data used for construction of data relations. Relationships between columns in subsets are captured as data relation objects, which can be fitted with spline curves to model underlying patterns.

:::{figure} class_diagram.png
:label: fig:class_diagram
:width: 800px
SplineCloud objects relationship diagram
:::

Spline fitting tool, a main instrument of the platform, is built on top of SciPy’s interpolate module and provides a visual interface to three main spline fitting methods described in section 3: Interpolating Splines, Smoothing Splines, Least Squares Splines.

The web interface of the spline fitting tool is powered by [D3.js](https://d3js.org/) and JavaScript build of [verb-nurbs](https://github.com/pboyer/verb) - an open-source library for NURBS modeling. The RESTful API and an open-source Python client library ([splinecloud-scipy](https://github.com/nomad-vagabond/splinecloud-scipy)) allow the reuse of data objects and spline curves in code. In this way, SplineCloud addresses the key limitations of traditional spline fitting workflows by offering the following core capabilities:

 - **Interactive Spline Modeling**. Users can construct spline curves using an interface to the SciPy’s interpolate module. Fine-tuning mode enables manual transformation of knot vectors and control points with real-time feedback to control smoothness, continuity, and fitting tolerance of the curves.

 - **Reusability and Interoperability of Models**. Fitted models can be accessed programmatically through the API or client libraries, eliminating the need to rerun fitting scripts for each new use case.

 - **Model Traceability**. The platform tracks authorship, data provenance, and dependencies between objects — helping preserve the context and credibility of published models.

 - **Collaborative Sharing**. Open models and datasets are discoverable and citable (via unique object UID). This supports collaborative workflows and reduces duplication of effort by reusing existing clean subsets and regression models.

## Interactive Spline Fitting Workflow

As it was mentioned in the prior sections, conventional programmatic approaches to curve fitting — such as those available in SciPy’s interpolate module — require iterative selection of fitting parameters. Usually, this means manual parameter tuning and replotting results to assess smoothness and fit quality. Alternatively, custom optimization scripts can be written to run through different combinations of parameters to minimize mean squared error (MSE), root mean squared error (RMSE), or another objective function. However, this complicates the process and does not allow for estimation of possible overfitting ({ref}`fig:overfitting_extrapolation` a) and extrapolation ({ref}`fig:overfitting_extrapolation` b) issues.

Interactivity in the curve fitting process significantly simplifies and accelerates model construction. It enables users to identify overfitting, discontinuities, or extrapolation issues early, and make real-time adjustments. This is especially valuable in parametric spline fitting, where fine-tuning the curve shape and knot configuration often requires iterative, visual feedback.

Despite this, many existing interactive curve fitting applications are either commercial and closed-source, lack support for parametric splines, or does not provide the ability to export and reuse models in code. This limits reproducibility and integration with modern data workflows.

In this section, we will take a look at how these capabilities are implemented in SplineCloud and how the interactive curve fitting approach can be complementary to the programmatic data processing workflows.


### Data Preparation

According to the workflow presented in {ref}`fig:workflow_scheme`, data has to be uploaded to the existing or new repository. It can be a text file, a spreadsheet, or an image containing a plot. In a case of text data, a default dataset will be created automatically and a subset of data can be identified by adjusting data loading options. Datasets will be created automatically for each sheet in the spreadsheet source file. An interactive plot digitizer tool will be displayed for the image file to help extract data ranges from plots ({ref}`fig:splinecloud_datasets`).

:::{figure} splinecloud_datasets.png
:width: 650px
:label: fig:splinecloud_datasets
SplineCloud datasets extracted from tabular data file and plots
:::

After identifying clean subsets, a default Data Relation object will be created after entering a curve fitting mode (either by clicking Fit Curve button or going into the Relations tab).

### Automatic Spline Fitting

By adding a first curve, an initial spline fit is generated using one of the SciPy fitting methods. The default choice is often a smoothing spline (implemented via `UnivariateSpline`), where a smoothing factor controls the trade-off between fidelity to the data and smoothness of the resulting curve ({ref}`fig:splinecloud-curve-fitting`). 

:::{figure} splinecloud-curve-fitting.png
:width: 550px
:label: fig:splinecloud-curve-fitting
Selecting a spline fitting method
:::

The important difference here is that SplineCloud’s smoothness parameter is a relative parameter used to calculate the actual SciPy’s (FITPACK’s) smoothing factor $s$ defined as:

```{math}
\sum_{i=1}^{n} w_i \left( y_i - S(x_i) \right)^2 \leq s
```

The transition from the relative smoothing parameter to  the SciPy’s s-factor is implemented in three steps:
1) build the least squares fit with the minimal possible number of knots (zero internal knots);
2) calculate the actual smoothing factor $s_{max}$ for this fit using formula (8);
3) multiply the relative smoothing parameter by this value: $s_{\text{scipy}} = s_{\text{max}} \cdot s_{\text{rel}}$

This approach improves user experience - instead of guessing each time the correct absolute value (which depends on the scale of data points), it is more intuitive to use relative values. By selecting several values for one data range the developed feedback instructs the more appropriate values for another curve of a different scale.

Least Squares fitting is implemented as an alternative to smoothing splines and is built on top of SciPy’s `LSQUnivariateSpline` class. For simplicity, a uniform knot vector is constructed and passed to the class constructor using the number of internal knots from the user input. However, there is an option to adjust the knot vector interactively and use least squares fitting for the given non-uniform knot vector. This capability is implemented in the fine-tuning mode (see Section 5.3).

For the cases when the curve should pass through the data points, SplineCloud has its implementation of the interpolating splines. This method is also implemented by using SciPy’s `UnivariateSpline` with hardcoded $s=0$.

### Fine-Tuning. Interactive Adjustments of Control Points and Knot Vector

A principal enhancement over the automatic fitting approach is SplineCloud’s *Fine-Tuning* functionality. It provides the ability to visually adjust control points and knot vectors of fitted splines. As proved in many cases, this interface enables users to achieve curve refinements that exceed the capabilities of SciPy’s automatic fitting algorithms in terms of smoothness and accuracy. More of that, the visual interface to knot vectors provides control over curve continuity in the specific regions. This helps in modelling complex data behavior with steep changes in main trends.

In SplineCloud, all spline curves are represented as parametric B-splines or NURBS. As mentioned in Section 2, a parametric B-spline curve is defined as a vector-valued function {ref}`b-spline`. This function, however, can be decomposed into its scalar components:

```{math}
\begin{aligned}

S^{(x)}(t) &= \sum_{i=0}^{n} c_{i}^{(x)} B_{i,k}(t) \\
S^{(y)}(t) &= \sum_{i=0}^{n} c_{i}^{(y)} B_{i,k}(t)
\end{aligned}
```

In simpler words, parametric spline curves can be defined by two distinct univariate spline functions, $x(t)$ and $y(t)$, sharing a common knot vector ({ref}`fig:spline_curve_as_two_functions`). This representation allows for the modeling of complex geometries, including loops, sharp transitions, and high-curvature regions.

:::{figure} spline_curve_as_two_functions.png
:width: 500px
:label: fig:spline_curve_as_two_functions
Parametric spline curve as a combination of two spline functions.  *Black triangles along parameter axes represent knots. Control polygons and control points are displayed in green color*
:::

#### Interactive Editing of Control Points

The beauty of parametric splines is in the ability to have an intuitive control over the curve shape by modifying the control polygon: moving, adding, and removing control points, increasing or decreasing their weights. This is impossible for spline functions, since they are defined through scalar coefficients, not vectors.

However, it is important to mention that for parametric spline curves coordinates of the control points $P_i = [x_i, y_i]$ are mathematically equivalent to the spline coefficients for the component functions $S^{(x)}(t)$ and $S^{(y)}(t)$, so that:

```{math}
\mathbf{C}(t) = \sum_{i=0}^{n} \mathbf{P}_i \, B_{i,k}(t), \quad \text{where } \mathbf{P}_i = \begin{bmatrix} c_i^{(x)} \\ c_i^{(y)} \end{bmatrix}
```

Due to the local support property of B-spline basis functions, moving a single control point influences the shape of the curve only within a limited range of the parameter domain, providing localized editing capabilities.

In the Fine-Tune mode, users can interactively drag control points to reshape the curve ({ref}`fig:fine-tune-control-points`). This mirrors the behavior of spline modeling in professional CAD environments, where designers sculpt geometry directly. The influence of each control point depends not only on its position but also on the associated basis function and, optionally, its weight. For example:
 - Control points clustered more closely create tighter curvature and sharper transitions;
 - Distant spacing results in smoother, flatter regions of the curve;
 - Assigning higher weights to a control point increases its pull on the curve, bringing it closer to the point's location, converting the B-Spline to NURBS.

:::{figure} fine-tune-control-points.png
:width: 600px
:label: fig:fine-tune-control-points
Adjusting control points of the spline curve in the Fine-Tune mode
:::

This form of user interaction provides precise control over the spline's shape and is particularly advantageous when fitting complex datasets where automated routines produce unsatisfactory results.

#### Dynamic Knot Vector Adjustment

The knot vector plays a central role in determining the structure and properties of the resulting spline curve. Given a spline of degree k, the knot vector $\{ t_0, t_1, \ldots, t_{m} \}$ is a non-decreasing sequence of real numbers, typically ranging from 0 to 1 (but not necessarily). Each interval $[t_i, t_{i+k+1})$ corresponds to a region over which a particular B-spline basis function $B_{i,k}(t)$ has support, meaning that each basis function is non-zero over at most $k+1$ knot spans.

The location and multiplicity of knots affect several critical properties of the resulting spline:

- **Continuity**. The number of continuous derivatives at a knot $t_i$ is $k−m_i$​, where $m_i$ is the multiplicity of that knot. Repeated knots reduce the smoothness of the spline at that knot location. Specifically, if a knot has multiplicity $m$, then the continuity of the spline at that knot is reduced to $C^{k - m}$, where $k$ is the degree of the spline. That is, the spline remains $(k - m)$-times continuously differentiable, and all higher derivatives are discontinuous.

- **Flexibility**. Adding more knots increases the local adaptability of the spline, allowing it to better follow variations in the data. In the B-spline formulation, the number of basis functions, and therefore the number of control points $n$ is always $T−k−1$, where $k$ is the spline degree, $T$ is the number of knots. As more interior knots are introduced, the number of basis functions increases, providing additional degrees of freedom for shaping the curve while maintaining the required continuity.

In traditional fitting methods such as `LSQUnivariateSpline` in SciPy, the knot vector must be either provided manually or generated heuristically, which typically requires some kind of iterative approach in finding an optimal (usually quasi-optimal) knot vector. SplineCloud removes this complexity by exposing the knot vector as an editable structure in the Fine-Tune mode, where users can manipulate knots directly and observe their impact on the spline in real time ({ref}`fig:fine-tuning-knot-vector`).

:::{figure} fine-tuning-knot-vector.png
:width: 600px
:label: fig:fine-tuning-knot-vector
Adjusting knot vector of the spline curve in the Fine-Tune mode
:::

In particular, SplineCloud enables users to:
 - Insert new knots to increase the flexibility of the spline in localized regions.
 - Remove knots to enforce greater smoothness and reduce overfitting.
 - Relocate knots to shift the spatial distribution of curve flexibility, optimizing the placement of inflection points or areas of curvature.
 - Add and remove duplicate knots to control continuity and introduce geometric features such as cusps, kinks, or plateaus.

This level of control is critical for modeling non-uniform data, such as step functions, relations with discontinuities, or empirical data sampled for distinct regimes or environments. However, compared to adjusting control points, manipulating the knot vector is often less intuitive.

Uniformly spaced knot vectors typically produce curves with more predictable and symmetric behavior, which is easier to interpret visually and adjust interactively. In contrast, non-uniform vectors can help in fitting local irregularities with higher precision. For example, a knot vector with tightly spaced knots in a transition zone and widely spaced knots elsewhere can fit complex behavior without sacrificing smoothness in the remaining domain.


### Fitting Errors

Comparison of the different fitted models requires a well-defined error metric that quantifies the discrepancy between the predicted and actual data. The fitting accuracy estimators tell how closely the curve approximates the given data. Out of many different metrics it is worth to mention the most common.


**Mean Absolute Error (MAE)**

The Mean Absolute Error (MAE) is used to evaluate the accuracy of a fitted model by measuring the average magnitude of the errors between predicted and observed values. It is defined as:

```{math}
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - S(x_i) \right|
```

where $y_i$​ are the actual data values, $S(x_i)$​ are the corresponding predicted values from the curve, $n$ is the total number of data points.

MAE provides an intuitive measure of model performance: it tells how far, on average, the predictions are from the actual observations.  Unlike squared error estimators, MAE is less sensitive to large individual errors (outliers), making it a useful complementary metric for assessing fit quality, especially when robustness is important.

**Mean Squared Error (MSE)**

The mean squared error measures the average of the squared differences between observed values $y_i$​ and corresponding predicted values $S(x_i)$. It is defined as:

```{math}
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - S(x_i))^2
```

This metric penalizes larger deviations more heavily, making it sensitive to outliers.

**Root Mean Squared Error (RMSE)**

The root mean squared error is the square root of the MSE and provides an error measure in the same units as the data:

```{math}
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - S(x_i))^2}
```

#### Estimating Fitting Errors in SciPy

In SciPy's spline fitting, smoothing splines support weighted data points, with fitting errors estimated using the corresponding weighted residuals. If weights $w_i$​ are provided (or implicitly set to 1), the weighted squared residuals are computed as:

```{math}
R_i = (w_i \left( y_i - S(x_i) \right))^2
```

The smoothing condition is formulated as:

```{math}
\sum_{i=1}^{n} R_i \leq s
```

where $s$ is the smoothing factor. This formulation is used internally by functions such as `UnivariateSpline` and `splrep`, where the goal is to balance curve fidelity with smoothness.

The `UnivariateSpline` class includes a method called `get_residual()`, which returns the weighted sum of squared residuals used in the spline construction process. Here is the example of estimating the fitting error using built-in `get_residual()` and custom estimator function:


```python
import numpy as np
from scipy.interpolate import UnivariateSpline

# Generate synthetic data
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.2 * np.random.randn(100)

# Fit smoothing spline
spline = UnivariateSpline(x, y, s=5.0)

# Evaluate spline at input points
y_fit = spline(x)

# RMSE using SciPy's get_residual()
residual = spline.get_residual()
rmse_scipy = np.sqrt(residual / len(x))

# RMSE computed manually
rmse_manual = np.sqrt(np.mean((y - y_fit)**2))

print(f"RMSE (SciPy):  {rmse_scipy:.5f}")
print(f"RMSE (Manual): {rmse_manual:.5f}")
```

```bash
RMSE (SciPy):  0.22361
RMSE (Manual): 0.22361
```

#### Estimating Fitting Errors in SplineCloud

In SplineCloud, the RMSE is calculated automatically for all curves. The `Fit accuracy` hint updates after each curve modification ({ref}`fig:fit_accuracy` a, b). This allows tracking the change of fitting error while applying different fitting parameters, comparing different models and fine-tuned curves against automatically fitted models.

:::{figure}
:label: fig:fit_accuracy
Simple interpolaing cubic spline.
<table>
<tr>
<td style="text-align: center;"><img src="fit_accuracy.png" height="230px"/>(a) RMSE estimation in SplineCloud.</td>
<td style="text-align: center;"><img src="fit_accuracy_parametric.png" height="230px"/>(b) RMSE valuated via shortst distance residuals.</td>
</tr>
</table>
:::

For complex curve shapes, especially those with closed loops ({ref}`fig:fit_accuracy` b), the error between a data point and the curve should not be measured only along the vertical axis. Instead, the true discrepancy is the Euclidean distance from each data point to the nearest point on the curve. This approach is implemented in SplineCloud to estimate ftting errors of parametric curves that approximate data in non-ascending order via $RMSE_{sd}$ - root mean squared shortest distance error:

```{math}
\text{RMSE}_{sd} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (w_i d_i)^2 }
```

where $d_i$ is the shortest distance between a data point and spline curve. Evaluation of this distance is implemented via SciPy's `optimize` module, and particularly a [direct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html) method is used as the one of the most stable and fastest global optimization algorithms (based on the in-house analyses).


## Reusability and Reproducibility with SplineCloud

**Reusability** and reproducibility are closely related concepts in scientific research, both essential for ensuring that data-driven results can be reliably verified, extended, and applied across different contexts and by independent researchers.
Reusability refers to the capacity of data, models, or computational results to be used beyond their original context, either by the same researcher at a later time or by others pursuing related work. For scientific outputs (data, code, models, etc.) to be reusable, they must be well-documented, accessible in a standardized format, and independent of specific software or environments. In the context of regression models, reusability implies that the model can be extracted, stored, and later reintegrated into different analytical workflows, ideally without the need to rerun the original fitting procedure or reaccess the raw data.

**Reproducibility** denotes the ability of independent researchers to replicate the results of a study using the same input data, methods, and conditions. In computational science, this typically involves the complete transparency of the modeling pipeline, including data preprocessing, parameter tuning, and evaluation metrics. For regression models, reproducibility requires that all aspects of the fitting process are recorded and available so that the same model output can be regenerated deterministically.

In conventional curve fitting workflows implemented via a programmable interface, fitted models are often tightly coupled with the original data and code used to generate them. While the source data and fitting scripts may be published, the resulting models themselves are rarely stored or shared as independently reusable objects. As a result, anyone wishing to replicate or build upon a previous curve fitting problem must re-execute the entire fitting process, including selecting the method, tuning parameters, and validating the fit.

This leads to three key limitations:
 - **Redundant effort**. Researchers across teams or domains often repeat fitting procedures for the same datasets.
 - **Poor reproducibility**. Deviations in fitting parameters and algorithms across various programming environments may cause discrepancies in reproduced models and, as a consequence, discrepancies in modelling results.
 - **Poor interoperability**. There is no widely adopted convention for serializing and sharing regression models across tools and environments. As a result, transferring models typically requires manual extraction of fitting parameters or curve construction data (like spline parameters). This process is error-prone and demands significant additional effort to reuse models accurately in code.

By extending the principles of FAIR data (Findable, Accessible, Interoperable, Reusable) to regression models, it becomes clear that it is not sufficient to publish raw data and code alone - intermediate or final research outputs such as fitted models (or response surace models) should also be accessible in standardized, referenceable form. Without this, the fitted model becomes an opaque byproduct rather than a verifiable and citable result.

For example, in engineering, a response surface models are widely used to approximate simulation results or material properties. These models are often used in systems engineering modeling processes, optimizations, or other analyses. If those models are not independently accessible, it becomes difficult to trace their provenance, assess their quality, or adapt them in related studies.


### Improving Reusability and Reproducibility of Spline Models

SplineCloud addresses these challenges by decoupling the spline model from the fitting code and treating it as a first-class, shareable object. The platform is open for open data and supports persistent storage of fitted curves as structured entities that include:
 - Unique IDs and web links;
 - API links that return spline data and related datasets and subsets;
 - A metadata layer, including authorship, tags, units, and other associated context (improvements in progress).

Once users create spline models interactively, they become instantly accessible to the broad public, including anonymous users. Reusability and reproducibility are ensured by both the open REST API and client libraries. 

An official Python client, `splinecloud-scipy`, is based on SciPy and allows for fetching spline data using the spline UIDs and recreating the model in code. Reusable spline model has critical methods for evaluating spline in the form $y=f(x),$ loading underlying data, and assessing fit accuracy (using one of the methods, listed in Section 5.4). This enables a new level of interoperability: a spline curve fitted by one researcher can be imported and evaluated in another researcher’s codebase, without accessing the original data and refitting it.

Let’s first take a look at the client library basic usage scenarios and then discuss the library structure and how splines a recreated and evaluated. 

#### Installation and basic usage

The `splinecloud-scipy` library is lightweight and can be installed from the Python Package Index:

```bash
pip install splinecloud-scipy
```

As of now, the library provides two main functions: `load_spline()` and `load_subset()`, which should be used to fetch spline models and underlying subsets:

```python
from splinecloud_scipy import load_spline
spline = load_spline(<curve_uid>)
```

The curve UID can be taken from SplineCloud - an API link dropdown on the Curves toolbox ({ref}`fig:curve_api_link`)

:::{figure} curve_api_link.png
:width: 600px
:label: fig:curve_api_link
Accessing curve API link
:::

`load_spline()` returns an instance of the `ParametricUnivariateSpline` class, which allows for evaluating the spline as a function of $x$. The structure and usage of this class a given below in section 6.1.2.

```python
import numpy as np
X = np.linspace(0, 20, 100)
Y = spline.eval(X, extrapolate=True)
```

:::{figure} evaluated_curve.png
:width: 400px
:label: fig:evaluated_curve
Reproduced spline curve
:::

The spline object allows for loading underlying data:

```python
columns, table = spline.load_data()
```

The same result can be achieved by explicitly loading a corresponding subset via its UID:

```python
from splinecloud_scipy import load_subset
columns, table = load_subset(<subset_uid>)
```

Similarly to spline curve, the subset UID can be taken from SplineCloud - an API link dropdown on the table header for tabular data, and from the Subsets toolbox for data extracted from plots ({ref}`fig:subset_api_link`)


:::{figure} subset_api_link.png
:width: 700px
:label: fig:subset_api_link
Accessing subset API link
:::

Both methods return a tuple with a list of column names and a NumPy array with subset data.

More information on the library usage can be found in the library code [repository](https://github.com/nomad-vagabond/splinecloud-scipy) README section.


#### The structure of the client library for Python

The `splinecloud-scipy` library is built around two core classes: `ParametricUnivariateSpline` - for the construction of the parametric splines from the curve data (degree, control points, and knot vector) and `PPolyInvertible` - a helper class defined to enable solving the spline curve as a function of $x$ values. Both classes extend SciPy’s classes, namely `interpolate.UnivariateSpline` and `interpolate.PPoly`.

`ParametricUnivariateSpline` is the main class of the `splinecloud-scipy` client library. Its instance is returned by the `load_spline()` function. It defines a 2D parametric spline curve based on two univariate spline functions — one for the $x(t)$ and one for the $y(t)$ dependencies, that share a common knot vector and spline degree.

Such representation allows reusing properties of `UnivariateSpline` and evaluating the inverse relation $t(x)$ by using a piecewise polynomial representation of the spline function $x(t)$.

Initialization of `ParametricUnivariateSpline` takes a tuple or list of the form: `(t: knots, cx: x-coefficients, cy: y-coefficients, k: degree)`


The class instance stores: curve degree, knot vector and its normalized version, coefficients for *x-* and *y-splines*. The `ParametricUnivariateSpline` object is callable - it takes a parameter value and returns the corresponding $x$ and y values by calling internal objects `self.spline_x()` and `self.spline_y()` - instances of SciPy’s `UnivariateSpline` class. This part, however, requires refactoring, since it uses private methods to construct `UnivariateSpline` from knot vector, coefficients, and degree:

```python
 self.spline_x = si.UnivariateSpline._from_tck(tck_x)
 self.spline_y = si.UnivariateSpline._from_tck(tck_y)
```
There is a plan to use the `BSpline` class instead.


Evaluation of the spline in the form $y(x)$ is implemented in the `eval()` method. Inside this method, a piecewise polynomial representation of spline functions is used to find the polynomial for the interval containing the *x-value*, then the polynomial is solved for the *t-value*. This parameter value is then used in the spline function $y(t)$ to find the desired *y-value*. 

Polynomial representations are constructed inside the private method `_build_ppolyrep()`, called on the `ParametricUnivariateSpline` initialization.

```python
  def _build_ppolyrep(self):
      self.spline_x.ppoly = PPolyInvertible.from_splinefunc(self.spline_x, extrapolate=True)
      self.spline_y.ppoly = PPolyInvertible.from_splinefunc(self.spline_y, extrapolate=True)
```

These representations are based on the custom class `PPolyInvertible` that extends SciPy’s `PPoly` class. Its main purpose is to find the corresponding interval for the *x-value* and solve for the *t-value* on this interval. This logic is implemented inside `evalinv()` method called from the `ParametricUnivariateSpline.eval()` method for the *x-spline* function. Inside `evalinv()`, a *t-value* is found using SciPy’s [optimize.brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html) method - a root-finding algorithm that finds a zero of a continuous function within a specified interval.

The `ParametricUnivariateSpline` class also provides a method for evaluating fit accuracy using one of the three methods discussed in Section 5.4.

```python
RMSE = spline.fit_accuracy(table, method="RMSE")
```

The client library introduces basic capabilities that allow for fetching, recreating, and evaluating spline models in code. Its structure, however, is a bit complicated: conversion to polynomial representation may be omitted in future versions, and other approaches may be used to evaluate the inverse relation $t(x)$ or solve the relation $y(x)$ directly. Contribution for this and other improvements is welcome.

Nevertheless, the implemented approach is covered with tests, supports extrapolation, and shows good performance (no reportable benchmarking was performed).


### Implications for Reusability and Reproducibility of Spline Models

By transforming spline models and datasets into shareable, code-native objects, and by introducing a client library for Python, SplineCloud eliminates the need to re-run fitting scripts or re-import raw data. From a practical point of view, this means:
 - **Immediate reuse**. Models can be invoked like native functions in mathematical modeling, numerical analysis, optimization, or visualization processes.
 - **Traceable origin**. Each model retains provenance information. Spline objects are associated with subsets, datasets, source data files, and authors. This provides transparency and enables attribution.
 - **Consistent integration and referencing**.  Anyone can retrieve the exact same spline instance and integrate it into their code, leaving the data fitting process outside the main code. Whether in Jupyter notebooks, software libraries, command-line tools, or a scientific article, the same model can be referenced by its unique identifier.

SplineCloud enables the transition of curve fitting from a script-bound operation to a persistent, shareable, and reproducible modeling activity. This approach minimizes redundant work and enhances the reliability and continuity of model-based research outputs.
The platform and its client library for Python implement principles of FAIR data and reproducible workflows by allowing fitted models to be handled as independent, discoverable entities that can be consistently integrated, reused, and exchanged across various models, analyses, and software packages.

### Limitations

While SplineCloud offers a substantial improvement in the usability, reusability, and reproducibility of spline-based regression models, the platform currently does not support other classes of regression models such as polynomial fits, rational functions, exponential models, or machine learning-based regressors. This limitation is not technical in nature but a deliberate, temporary design decision, rooted in both theoretical and practical considerations.

Spline models, particularly B-splines and their parametric forms, possess a standardized and well-defined mathematical representation. This makes them ideally suited for platform-independent storage, manipulation, and code-level reuse. Their local support, smoothness properties, and flexibility in representing arbitrary empirical relations enable consistent behavior across computational environments and programming languages. These features align with the core objectives of SplineCloud — namely, enabling transparent and reproducible modeling workflows.

In contrast, extending support to arbitrary analytical or statistical regression models would require substantial generalization of the platform’s core architecture. Such models often rely on complex formulations and domain-specific assumptions that are difficult to standardize or serialize reliably. Moreover, handling custom models would require the development of thicker client libraries to support reusability. This, along with significant effort for the platform frontend and backend modifications, constrains their adoption by SplineCloud (at least until proved necessary).

From a pragmatic perspective, spline-based models are especially effective for representing empirical data derived from physical experiments, simulations, and measurements — scenarios where an accurate and differentiable fit is often more valuable than an interpretable analytical expression. SplineCloud is thus particularly suited for constructing surrogate models, processing and interpolating results of numerical simulations (like wind tunnel tests), and reusing such models in downstream analysis, control, or optimization tasks.

However, users requiring analytical model validation (e.g., fitting custom functions to verify physical laws or derive closed-form expressions) will find SplineCloud insufficient for their needs. In such cases, traditional tools, including SciPy, remain more appropriate.

Despite this limitation, SplineCloud introduces the novel capability of building shareable repositories of empirical relations. The catalogs of such relations can serve as reference data and model sources in experimental domains such as fluid dynamics, structural mechanics, or thermodynamics, where precomputed curves and data-driven models are frequently reused. For example, aerodynamic design processes often rely on aggregated wind tunnel results; SplineCloud provides an infrastructure to formalize, store, and exchange such data in a reproducible and programmatically accessible way.


## Use Cases and Applications (TODO)


## Summary and Future Directions (TODO)

