---
# Ensure that this title is the same as the one in `myst.yml`
title: Pipeline-level differentiable programming for the real world
abstract: |
  TBD
---

## Introduction

Countless problems in science and engineering can be framed as tuning tasks, where parameters of a complex process &mdash; such as a physical simulation or a lab experiment &mdash; are iteratively tuned to maximize an objective. For such tasks, differentiable programming (DP) has emerged as a powerful tool, due to its ability to automate and accelerate gradient-based operations, enabling "differentiable physics" by codifying the use of gradient information to optimize, correct, or control physical systems. At the core of DP is the technique known as automatic differentiation (autodiff, AD) to compute partial derivatives of computer programs without the need to spell out explicit forms of said derivatives. Autodiff techniques enjoy great success in the fields of artificial intelligence and machine learning (ML) thanks to proliferation of deep learning software frameworks such as TensorFlow, JAX, and PyTorch [@baydin2018automatic].

However, applications of AD in differentiable physics are largely untested at industrial scale, and ML frameworks are rarely designed for nor tested on practical science and engineering _systems_ (as opposed to single components). Building pipelines that propagate gradients effortlessly across components introduces unique challenges. Real-world pipelines often span diverse technologies, frameworks (e.g., JAX, TensorFlow, PyTorch, Julia), computing environments (local vs. distributed clusters; CPU vs. GPU), and teams with varying expertise. Additionally, legacy systems and non-differentiable components often need to coexist with modern AD-enabled frameworks.

Recognising the need for a robust system-level software support of Autodiff-capable pipelines, we put forth the design, implementation, and validation of a novel system engineering approach to AD-driven physics: "Differentiable Physics Programming" (DPP). DPP resolves the above challenges via autodiff-native software containerization and dataflow-based orchestration, built to be highly modular and interoperable with physics simulation tools and engineering data types, namely computational fluid dynamics (CFD) and computer-aided engineering (CAE) broadly. Such a system enables scientists and engineers of diverse backgrounds to build complex workflows centered around simulation and data-driven surrogate models, and propagate gradients throughout the entire workflow, thus unleashing the potential of AD on end-to-end applications.

To demonstrate the DPP system in action we present Tesseract, a software ecosystem that provides pipeline-level AD and unlocks DPP at scale. We give detailed description of Tesseract's software design and functionality, and demonstrate its use on a non-trivial problem of the minimization of compliance of a parametric structure made of a linear elastic material. Our aim in developing Tesseract is present the community with tools necessary to scale up the capabilities of Autodiff-native scientific workflows.

## General intro into tesseracts ecosystem
Maybe transfer from our docs

## Tessearct core
Couple of paragraphs about design, API, usage. Nice diagram.

## Tesseract jax
Couple of paragraphs about design, API, usage. Nice diagram if it exists.

## Use case: Parametric Topology Optimization

![Figure 1](illustration.png)

As a proof of concept of Tesseract's capabilities, to simplify differentiable physics programing, we present a case study of parametric topology optimization. The idea is to construct a parametric geometry using a standard 3D geometry library. We then compute a SDF (signed distance field) and apply a sigmoid function to transform it into a density field. The density field is then fed into a finite element solver, which then computes the compliance. Since the mapping from the design space parameters to the SDF field is not implemented in way that enables automatic differentiation, we implement a custom AD endpoint using finite differences. Using tesseract-core we implement the parameter to sdf field function and the density field to compliance function as tesseract components. We then leverage tesseract-jax and use the standard gradient computation function from jax to compute the total gradient of the compliance with respect to the design parameters. 

| Parametric Optimization (Ours) | Free Form Topology Optimization |
|-------------------------|---------------------------------|
| ![param](rho_optim.gif) | ![param](free_form.gif)         |

We compare our solution with the free form topology optimization solution that is implemented in the jax-fem library [@xue2023jax]. We can observe that our solution constructs a structure that is suprisingly similar to the free form topology optimization solution, even though in our case the design space is parametrized by a small number of parameters. Doing the above without using Tesseract is challenging due to the following reasons:

- **Hetereogeneity of gradient computation**: In this pipeline some components rely on automatic differentiation, while others rely on finite differences. With tesseracts and tesseract-jax we can define the AD endpoints for each component and then use the standard gradient computation function from JAX to compute the total gradient of the compliance with respect to the design parameters.

- **Modularity**: The components of the pipeline are implemented as tesseract components, which allows us to easily swap out components and reuse them in other pipelines. For example, we could replace the design space tesseract relying on PyVista with a design space tesseract relying on OpenSCAD.

- **Dependency management**: Tesseract components are containerized, which allows us to easily manage dependencies and ensure that the pipeline runs in a consistent environment. 

- **Computing ressources**: Tesseract components can be run on different computing resources, which allows us to easily scale the pipeline and run it on different machines. For example, we could run the finite element solver on a GPU machine and the design space tesseract on a CPU.

## Related work

Tesseract offers a unique combination of containerised runtime for scientific computing and native AD capabilities. Considered separately, both these areas are rich with existing tools and frameworks.

**Containerised runtime.** There are several projects that provide containerisedruntime infrastructure for scientific computations. Some, including MLServer [@MLServer], BentoML [@BentoML], or Triton Inference Server [@Triton_Inference_Server], target primarily machine learning workloads. Other projects, such as UM-Bridge [@UMBridge] or Singularity [@kurtzer2017singularity], cover a broader scope of scientific domains. A key feature that separates Tesseract ecosystem from these project is its strong focus on gradient computation.

**Automatic differentiation.** Given how critical is automatic differentiation (AD) to modern scientific workflows [@baydin2018automatic], it is not surprising that there is a wide range of software tools providing AD capabilities. Major deep learning frameworks PyTorch [@paszke2017automatic] and TensorFlow [@abadi2016tensorflow] both implement AD and make extensive use of it for training of ML models. AD is also one of the main features of JAX, Python library for high performance numerical computing [@jax2018github], and its implementation in JAX strongly influenced design of Tesseract's API. Crucially, the majority of these frameworks support AD on a program level, and component-scale AD is not nearly as common. While composite solutions are possible (e.g. combining AD-capable backend with an HTTP service), to the best of our knowledge Tesseract is the first software project that natively supports AD on the component level.

## Conclusions

----- What comes below is the example, to use for markup nits

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
