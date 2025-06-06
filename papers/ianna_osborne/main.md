---
title: Advancing High Energy Physics Data Analysis with Julia -- A Case for JuliaHEP
abstract: |
  High-energy physics (HEP) research relies on efficient data analysis 
  frameworks to process vast amounts of structured and unstructured data. 
  Traditionally, Python has been the dominant language for data analysis 
  in HEP, supported by libraries such as NumPy, Awkward Array, and Uproot. 
  However, the Julia programming language offers a compelling alternative 
  with its high-performance capabilities and ease of use. This paper introduces 
  JuliaHEP, an initiative to leverage Julia for HEP data analysis, providing 
  a high-level overview of its benefits, current developments, and integration 
  with existing Python-based tools.

---

## Introduction

The high-energy physics (HEP) community has long relied on Python and C++ for data analysis and simulation. While Python provides ease of use and a rich scientific ecosystem, it struggles with performance for large-scale analyses. C++, on the other hand, offers speed but comes with increased complexity and slower development cycles. Julia, a modern language designed for scientific computing, promises the best of both worlds: high-level expressiveness with near-native execution speed.

In this paper, we introduce JuliaHEP, an emerging set of tools and libraries designed to facilitate HEP data analysis in Julia. We discuss how JuliaHEP integrates with Python-based frameworks such as Awkward Array, Uproot, and Numba while also leveraging Julia’s strengths, such as just-in-time (JIT) compilation, multiple dispatch, and seamless GPU acceleration.

We first examine how Awkward Array, a key library for handling complex and jagged HEP data, is integrated into Julia workflows. By using Vector of Vectors, a Julia native type that provides similar capabilities, we compare performance and usability against the traditional Python stack. Additionally, we explore Julia’s interoperability with Numba and its native compilation strategies to optimize HEP computations.

A key focus of this paper is the performance benefits of Julia for HEP workloads. We showcase benchmarks comparing JuliaHEP tools against Python and C++ implementations, highlighting cases where Julia offers significant speedups with less boilerplate code. We also discuss Julia’s potential for parallel computing and its built-in support for GPUs, which provides an efficient pathway for scaling HEP computations.

Beyond performance, we address practical aspects of integrating Julia into existing HEP workflows. We demonstrate how physicists can gradually adopt Julia without abandoning Python-based tools, thanks to interlanguage operability. Case studies from CMS and other experimental collaborations illustrate Julia’s real-world applicability in tasks like event processing, data transformation, and simulation.

## Bibliographies, citations and block quotes

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
