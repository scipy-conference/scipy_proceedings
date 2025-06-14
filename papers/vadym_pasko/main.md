---
# Ensure that this title is the same as the one in `myst.yml`
title: "Enhancing Curve Fitting with SciPy: Interactive Spline Modeling and Reproducibility with SplineCloud"
abstract: |
  Curve fitting is a fundamental task in data science, engineering, and scientific computing, enabling researchers to extract meaningful relationships from data. However, selecting and tuning the right fitting model for complex, noisy, or multidimensional data remains a significant challenge. SciPy plays a critical role in addressing these challenges by providing robust spline fitting methods that offer flexibility and precision. Yet, fine-tuning spline parameters, ensuring stability in extrapolation, and sharing fitted models for reproducibility remain open problems.
  
  In order to address these challenges we developed SplineCloud - an open platform that provides interactive spline fitting capabilities and is using SciPy on the backend. SplineCloud allows the construction, analysis, and exchange of complex regression models using SciPy’s `interpolate` module. With the help of the SplineCloud interactive tool, which extends the basic capabilities of SciPy researchers can fine-tune spline parameters, adjust knot vectors interactively, analyze the accuracy of models, and make them reusable in code, fostering better collaboration and knowledge transfer. Reusability of curves in code is enabled via the open-source SplineCloud client library called `splinecloud-scipy`, which is also based on SciPy. The approach of interactive cloud-based fitting allows to build improved data processing workflows, separate data processing from the main code, and make curve fitting more accessible and reproducible.
---

## Introduction

Curve fitting is a well-known technique in science and engineering, used to capture and formalize empirical or statistical relationships in data, coming either from physical experiments, numerical simulations, or observational studies. Besides a pure visual representation of the trends in data,  curve fitting enables model-driven decision making, allowing to: reduce the number of numerical simulations, smoothen noisy observations, intepolate or extrapolate missing data points.

Various curve fitting approaches have been developed and used in different fields of science and engineering, including but not limited to polynomial regression, exponential models, radial basis functions, and neural approximators. Among these, spline models stand out for their balance between flexibility and smoothness. Unlike global polynomial fits, splines are piecewise polynomials that offer local control, making them highly adaptable to complex or noisy data.

SciPy has long served as a foundational toolkit for curve fitting in research and engineering. With methods like `scipy.optimize.curve_fit` and the methods of `scipy.interpolate` module researchers have a power to fi complex relations in data with functional relations. Particularly `UnivariateSpline`, `InterpolatedUnivariateSpline`, `LSQUnivariateSpline`, `splrep/splev`, `make_interp_spline` methods provide reliable and high-performance interfaces to construct, evaluate, and differentiate spline models. These methods, however (due to the nature of the SciPy library), provide only programmable interfaces, which limits the ease of experimentation and tuning. More of that, no standard way for sharing and reproducing fitted models in code has been suggested so far.

In practice, engineers and scientists face several persistent pain points:
 - Choosing the right model or spline type;
 - Tuning smoothing factors and knot positions;
 - Ensuring meaningful extrapolation behavior;
 - Reusing fitted models across projects or teams;
 - Reproducing results without needing to share raw data or fitting scripts.

To address these issues, we developed SplineCloud: an open platform that brings interactivity, reusability, and collaboration to data fitting, while remaining tightly coupled with SciPy’s interpolate module. SplineCloud enhances traditional curve fitting by providing an interactive web interface, shareable model storage, and a Python client library (also based on SciPy) that simplifies access and reuse of fitted models in code.

In this article, we provide an overview of the spline fitting problem, outline its foundational implementation within SciPy, and introduce SplineCloud — a cloud-based platform for constructing and sharing regression models using parametric splines. We also examine several practical applications to illustrate its capabilities in real-world contexts.


## Some Theoretical Background Behind Splines (WIP)

Splines are piecewise-defined functions used extensively in numerical analysis, computer-aided geometric design, and data fitting. The fundamental idea behind spline interpolation or approximation is to construct a smooth function that matches a set of data points or satisfies a set of constraints, while preserving computational efficiency and numerical stability.


A spline function $S(x)$ of degree $k$ over an interval $[a, b]$ is a piecewise polynomial function such that:

- On each subinterval $[x_i, x_{i+1}]$, $S(x)$ is a polynomial of degree $k$,
- $S(x)$ is $C^{k-1}$-continuous on $[a, b]$, i.e., it has continuous derivatives up to order $k-1$.

A commonly used example is the cubic spline ($k = 3$), which ensures $C^2$ continuity.

Splines can be represented in several ways, each suitable for different applications:

- **Piecewise Polynomial Form**: The spline is given explicitly by polynomials on each interval:

```{math}
S(x) =
\begin{cases}
P_1(x), & x_0 \leq x < x_1 \\\\
P_2(x), & x_1 \leq x < x_2 \\\\
\vdots \\\\
P_n(x), & x_{n-1} \leq x \leq x_n
\end{cases}
```

- **Hermite Form**: Based on interpolation of function values and derivatives at each knot.

- **Bézier Form**: Uses Bernstein polynomials and control points, typically for single intervals in computer graphics and CAD.

- **B-spline Form**: Offers a powerful and stable representation for splines over multiple intervals, especially in numerical methods and modeling.


A distinction must be made between spline functions and parametric spline curves:

- **A spline function** is a scalar-valued function $S(x)$, defined over a single independent variable $x$, typically used in interpolation or regression of scalar data.

- **A parametric spline curve** defines a vector-valued mapping from a scalar parameter $t$ to a multidimensional space:

```{math}
\mathbf{C}(t) = \left( x(t), y(t), z(t), \dots \right)

```
where each coordinate function $x(t), y(t), \dots$ is a spline function in $t$. This is the common representation in geometric modeling and computer graphics.

A particularly efficient and widely used spline representation is the B-spline (basis spline). B-splines are defined via a set of control points ${ \mathbf{P}_i }$ and a knot vector ${ t_i }$, and provide local control, numerical stability, and efficient evaluation.

A B-spline curve of degree $k$ is defined as:

```{math}
\mathbf{C}(t) = \sum_{i=0}^{n} \mathbf{P}_i B_{i,k}(t)
```
where $\mathbf{P}_i$ are control points, $B_{i,k}(t)$ are the B-spline basis functions of degree $k$, defined recursively by the Cox–de Boor formula [@doi:10.1093/imamat/10.2.134; @doi:10.1007/978-1-4612-6333-3]:

```{math}
B_{i,k}(t) =
\frac{t - t_i}{t_{i+k} - t_i} B_{i,k-1}(t)
+
\frac{t_{i+k+1} - t}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(t)
```

In (5) possible divisions by zero are resolved by the convention that ‘anything divided by zero is zero’. The function $B_{i,k}=B_{i,k,\mathbf{t}}$ is called a B-spline of degree k (with knots $\mathbf{t}$).

This formulation separates geometry (control points) from basis functions, allowing flexible manipulation and efficient computation of spline curves.

### Applications and Benefits of Parametric Splines

Parametric splines are widely used in computer graphics, computer-aided desig (CAD), and more general geometric modeling, where the goal is to design and manipulate accurate and smooth curves and surfaces in two or three dimensions. Their unique properties provide the following advantages:

- **Geometric Flexibility**. Since parametric curves are not constrained to be functions in the $y = f(x)$ form, they can represent vertical segments, loops, cusps, and other geometries that a function cannot.

- **Smooth Multi-dimensional Representation**. By treating each coordinate as an independent spline, parametric curves offer uniform control over the curve shape in all spatial directions.

- **Local Control**. In representations such as B-splines or NURBS, moving one control point affects only a portion of the curve, enabling precise local edits without changing the global shape.

- **Uniform Parameterization**. The parameter $t$ typically varies over a fixed interval $[t_0, t_n]$, which makes operations like subdivision, evaluation, and rendering more efficient and robust.

These properties make parametric splines ideal for applications in: 3D modeling and animation, font and character design, surface generation (via tensor product surfaces), and industrial design (automotive, aerospace, etc.).


An intuitive visual explanation of splines, different forms of their representation and unique properties is given in the video by Freya Holmér: 

[📺 The Continuity of Splines](https://www.youtube.com/watch?v=jvPPXbo87ds)


## Spline Fitting With SciPy (WIP)

SciPy provides a robust and flexible set of spline fitting tools for both **interpolation** and **approximation** through the `scipy.interpolate` module.

In ternm of univariate fitting, SciPy supports several spline fitting methods, including:

- **Interpolating splines** (exact fit to the data)
- **Smoothing splines** (approximate fit with smoothness penalty)
- **Least squares splines** (approximate fit with squared residual minimization penalty)
- **Parametric splines** (e.g., 2D or 3D curves with respect to a parameter)

### Basic Spline Interpolation Example

To fit a spline through a set of points $(x_i, y_i)$ exactly, use the `InterpolatedUnivariateSpline` or `make_interp_spline` functions:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Sample data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
y = np.array([12, 8, 11, 7, 5, 2, 3, 5, 6, 4, 5, 7, 8, 13, 19, 22, 25])

# Create a cubic spline interpolant
spline = make_interp_spline(x, y, k=3)

# Evaluate spline on a fine grid
x_spl = np.linspace(x.min(), x.max(), 200)
y_spl = spline(x_spl)

# Plot
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_spl, y_spl, label='Cubic spline')
plt.legend()
plt.title("Cubic Spline Interpolation with SciPy")
plt.show()
```
:::{figure} interp_spline_fitting_scipy.png
:label: fig:stream
Simple interpolaing cubic spline.
:::

This constructs a spline $S(x)$ such that:

```{math}
S(x_i) = y_i \quad \text{for all } i
```
and ensures continuity of first and second derivatives ($C^2$ continuity for cubic splines).


### Smoothing Splines

When data contains noise, it is often preferable to use a smoothing spline, which balances fidelity to the data with smoothness. SciPy provides UnivariateSpline, which takes a smoothing factor s:

```python
from scipy.interpolate import UnivariateSpline

# Fit smoothing spline with smoothing factor s
spline = UnivariateSpline(x, y, s=18)

x_spl = np.linspace(x.min(), x.max(), 200)
y_spl = spline(x_spl)

plt.plot(x, y, 'o', label='Noisy data')
plt.plot(x_spl, y_spl, label='Smoothing spline (s=18)')
plt.legend()
plt.title("Smoothing Spline Fit")
plt.show()
```
:::{figure} smooth_spline_fitting_scipy.png
:label: fig:stream
Simple interpolaing cubic spline.
:::
The smoothing spline minimizes the penalized least-squares objective:

```{math}
\min_S \left\{ \sum_{i=1}^n \left( y_i - S(x_i) \right)^2 + \lambda \int_a^b \left( S''(x) \right)^2 dx \right\}
```
where $\lambda$ is a regularization parameter related to s.







### Critical Pain Points (TODO)


## Introducing SplineCloud (TODO)


## Interactive Spline Fitting Workflow (TODO)


## Reusability and Reproducibility with SplineCloud (TODO)


## Use Cases and Applications (TODO)


## Summary and Future Directions (TODO)


## Bibliographies, citations and block quotes (WIP)

Bibliography files and DOIs are automatically included and picked up by `mystmd`.
These can be added using pandoc-style citations `[@doi:10.1109/MCSE.2007.55]`
which fetches the citation information automatically and creates: [@doi:10.1109/MCSE.2007.55].
Additionally, you can use any key in the BibTeX file using `[@citation-key]`,
as in [@hume48] (which literally is `[@hume48]` in accordance with
the `hume48` cite-key in the associated `mybib.bib` file).
Read more about [citations in the MyST documentation](https://mystmd.org/guide/citations).

If you wish to have a block quote, you can just indent the text, as in:

> When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication.
>
> -- @hume48

Other typography information can be found in the [MyST documentation](https://mystmd.org/guide/typography).

### DOIs in bibliographies

In order to include a DOI in your bibliography, add the DOI to your bibliography
entry as a string. For example:

```{code-block} bibtex
:emphasize-lines: 7
:linenos:
@book{hume48,
  author    =  "David Hume",
  year      = {1748},
  title     = "An enquiry concerning human understanding",
  address   = "Indianapolis, IN",
  publisher = "Hackett",
  doi       = "10.1017/CBO9780511808432",
}
```

### Citing software and websites

Any paper relying on open-source software would surely want to include citations.
Often you can find a citation in BibTeX format via a web search.
Authors of software packages may even publish guidelines on how to cite their work.

For convenience, citations to common packages such as
Jupyter [@jupyter],
Matplotlib [@matplotlib],
NumPy [@numpy],
pandas [@pandas1; @pandas2],
scikit-learn [@sklearn1; @sklearn2], and
SciPy [@scipy]
are included in this paper's `.bib` file.

In this paper we not only terraform a desert using the package terradesert [@terradesert], we also catch a sandworm with it.
To cite a website, the following BibTeX format plus any additional tags necessary for specifying the referenced content is recommended.
If you are citing a team, ensure that the author name is wrapped in additional braces `{Team Name}`, so it is not treated as an author's first and last names.

```{code-block} bibtex
:emphasize-lines: 2
:linenos:
@misc{terradesert,
  author = {{TerraDesert Team}},
  title  = {Code for terraforming a desert},
  year   = {2000},
  url    = {https://terradesert.com/code/},
  note   = {Accessed 1 Jan. 2000}
}
```

## Source code examples

No paper would be complete without some source code.
Code highlighting is completed if the name is given:

```python
def sum(a, b):
    """Sum two numbers."""

    return a + b
```

Use the `{code-block}` directive if you are getting fancy with line numbers or emphasis. For example, line-numbers in `C` looks like:

```{code-block} c
:linenos: true

int main() {
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
    return 0;
}
```

Or a snippet from the above code, starting at the correct line number, and emphasizing a line:

```{code-block} c
:linenos: true
:lineno-start: 2
:emphasize-lines: 3
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
```

You can read more about code formatting in the [MyST documentation](https://mystmd.org/guide/code).

## Figures, Equations and Tables

It is well known that Spice grows on the planet Dune [@Atr03].
Test some maths, for example $e^{\pi i} + 3 \delta$.
Or maybe an equation on a separate line:

```{math}
g(x) = \int_0^\infty f(x) dx
```

or on multiple, aligned lines:

```{math}
\begin{aligned}
g(x) &= \int_0^\infty f(x) dx \\
     &= \ldots
\end{aligned}
```

The area of a circle and volume of a sphere are given as

```{math}
:label: circarea

A(r) = \pi r^2.
```

```{math}
:label: spherevol

V(r) = \frac{4}{3} \pi r^3
```

We can then refer back to Equation {ref}`circarea` or
{ref}`spherevol` later.
The `{ref}` role is another way to cross-reference in your document, which may be familiar to users of Sphinx.
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas[^footnote-1] diam turpis, placerat[^footnote-2] at adipiscing ac,
pulvinar id metus.

[^footnote-1]: On the one hand, a footnote.
[^footnote-2]: On the other hand, another footnote.

:::{figure} figure1.png
:label: fig:stream
This is the caption, sandworm vorticity based on storm location in a pleasing stream plot. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/arrays/streamplot.html).
:::

:::{figure} figure2.png
:label: fig:em
This is the caption, electromagnetic signature of the sandworm based on remote sensing techniques. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/stats/hist2d.html).
:::

As you can see in @fig:stream and @fig:em, this is how you reference auto-numbered figures.
To refer to a sub figure use the syntax `@label [a]` in text or `[@label a]` for a parenhetical citation (i.e. @fig:stream [a] vs [@fig:stream a]).
For even more control, you can simply link to figures using `[Figure %s](#label)`, the `%s` will get filled in with the number, for example [Figure %s](#fig:stream).
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

```{list-table} This is the caption for the materials table.
:label: tbl:materials
:header-rows: 1
* - Material
  - Units
* - Stone
  - 3
* - Water
  - 12
* - Cement
  - {math}`\alpha`
```

We show the different quantities of materials required in
@tbl:materials.

Unfortunately, markdown can be difficult for defining tables, so if your table is more complex you can try embedding HTML:

:::{table} Area Comparisons (written in html)
:label: tbl:areas-html

<table>
<tr><th rowspan="2">Projection</th><th colspan="3" align="center">Area in square miles</th></tr>
<tr><th align="right">Large Horizontal Area</th><th align="right">Large Vertical Area</th><th align="right">Smaller Square Area<th></tr>
<tr><td>Albers Equal Area   </td><td align="right"> 7,498.7   </td><td align="right"> 10,847.3  </td><td align="right">35.8</td></tr>
<tr><td>Web Mercator        </td><td align="right"> 13,410.0  </td><td align="right"> 18,271.4  </td><td align="right">63.0</td></tr>
<tr><td>Difference          </td><td align="right"> 5,911.3   </td><td align="right"> 7,424.1   </td><td align="right">27.2</td></tr>
<tr><td>Percent Difference  </td><td align="right"> 44%       </td><td align="right"> 41%       </td><td align="right">43%</td></tr>
</table>
:::

or if you prefer LaTeX you can try `tabular` or `longtable` environments:

```{raw} latex
\begin{table*}
  \begin{longtable*}{|l|r|r|r|}
  \hline
  \multirow{2}{*}{\bf Projection} & \multicolumn{3}{c|}{\bf Area in square miles} \\
  \cline{2-4}
   & \textbf{Large Horizontal Area} & \textbf{Large Vertical Area} & \textbf{Smaller Square Area} \\
  \hline
  Albers Equal Area   & 7,498.7   & 10,847.3  & 35.8  \\
  Web Mercator        & 13,410.0  & 18,271.4  & 63.0  \\
  Difference          & 5,911.3   & 7,424.1   & 27.2  \\
  Percent Difference  & 44\%      & 41\%      & 43\%  \\
  \hline
  \end{longtable*}

  \caption{Area Comparisons (written in LaTeX) \label{tbl:areas-tex}}
\end{table*}
```

Perhaps we want to end off with a quote by Lao Tse[^footnote-3]:

> Muddy water, let stand, becomes clear.

[^footnote-3]: $\mathrm{e^{-i\pi}}$
