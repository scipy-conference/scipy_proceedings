---
# Ensure that this title is the same as the one in `myst.yml`
title: Orchestrating Bioinformatics Workflows Across a Heterogeneous Toolset with Flyte
abstract: |
  While Python excels at prototyping and iterating quickly, it’s not always performant enough for whole-genome scale data processing. Flyte, an open-source Python-based workflow orchestrator, presents an excellent way to tie together the myriad tools required to run bioinformatics workflows. Flyte is a k8s native orchestrator, meaning all dependencies are captured and versioned in container images. It also allows you to define custom types in Python representing genomic datasets, enabling a powerful way to enforce compatibility across tools. Computational biologists, or any scientists processing data with a heterogeneous toolset, stand to benefit from a common orchestration layer that is opinionated yet flexible.
---

## Introduction

Since the sequencing of the human genome [@doi:10.1126/science.1058040], and as other wet lab processes have scaled in the last couple decades, computational approaches to understanding the living world have exploded. The firehose of data generated from all these experiments led to algorithms and heuristics developed in low-level high-performance languages such as C and C++. Later on, industry standard collections of tools like GATK [@doi:10.1002/0471250953.bi1110s43] were written in Java. A number of less performance intensive offerings such as MultiQC [@doi:10.1093/bioinformatics/btw354] are written in Python; and R is used extensively where it excels: visualization and statistical modeling. Finally, newer AI models and Rust based components are entering the fray.

Different languages also come with different dependencies and approaches to dependency management, interpreted versus compiled languages for example handle this very differently. They also need to be installed correctly and available in the user’s PATH for execution. Moreover, compatibility between different tools in bioinformatics often falls back on standard file types expected in specific locations on a traditional filesystem. In practice this means searching through datafiles or indices available at a particular directory and expecting a specific naming convention or filetype.

In short, bioinformatics suffers from the same reproducibility crisis [@doi:10.1038/533452a] as the broader scientific landscape. Standardizing the interfaces, orchestration and encapsulation of these different tools in a flexible and future-proof way is of paramount importance on this unrelenting march towards larger and larger datasets.

## Methods

Solving these problems using Flyte is accomplished by capturing dependencies flexibly with dynamically generated container images, defining custom types to enforce at the task boundary, and wrapping tools in Flyte tasks. Before diving into the finer points, a brief primer on Flyte is required. While the [introduction](https://docs.flyte.org/en/latest/introduction.html) in the docs is a worthwhile read before continuing, I'll skip to a more salient *hello world* example.

```python
from flytekit import task, workflow

@task
def greet() -> str:
  return "Hello"

@task
def say(greeting: str, name: str) -> str:
    return f"{greeting}, {name}!"

@workflow
def hello_world_wf(name: str = 'world') -> str:
    greeting = greet()
    res = say(greeting=greeting, name=name)
    return res
```

Tasks are the most basic unit of work in Flyte. They are pure-python functions and their interface is strongly typed in order to compose the workflow. The workflow itself is actually a DSL that statically compiles a directed-acyclic graph (DAG) based on the dependencies between the different tasks. There are different flavors of tasks and workflows as we'll see later, but this is the core concept.

### Images

While it's possible to run Flyte tasks and workflows locally in a Python virtual environment, production executions in Flyte run on a Kubernetes cluster. As a k8s native orchestrator, all tasks run in their own (typically single-container) pods using whatever image is specified in the task decorator. Capturing dependencies in container images has been a gold-standard for some time now, but this is taken a step further with [ImageSpec](https://docs.flyte.org/en/latest/user_guide/customizing_dependencies/imagespec.html#imagespec). ImageSpec lets you easily define a base image and additional dependencies right alongside your task and workflow code. These additional dependencies can include apt, python or conda packages. While [envd](https://github.com/tensorchord/envd) is the default builder, other backends like Docker are available should the need arise. These ImageSpec definitions are loosely coupled to your workflow code and are built automatically when tasks are registered or run on a Flyte cluster. ImageSpec reduces the complexity inherent in manually authoring a Dockerfile and enables a more streamlined approach to building images without the need for an additional build step and configuration update to reference the latest image. This coupling and reduced complexity makes it easier to build single-purpose images instead of throwing everything into a single monolith - here are a few:

```python
main_img = ImageSpec(
    name="main",
    packages=["flytekit"],
    python_version="3.11",
    conda_channels=["bioconda"],
    conda_packages=[
        "samtools",
        "bcftools",
        "bwa",
        "fastp",
        "hisat2",
        "bowtie2",
        "gatk4",
        "fastqc",
        "htslib",
    ],
    registry="ghcr.io/pryce-turner",
)

folding_img = ImageSpec(
    name="protein",
    platform="linux/amd64",
    python_version="3.11",
    packages=["flytekit", "transformers", "torch"],
    conda_channels=["bioconda", "conda-forge"],
    conda_packages=[
        "prodigal",
        "biotite",
        "biopython",
        "py3Dmol",
        "matplotlib",
    ],
    registry="ghcr.io/pryce-turner",
)
```

The `main` image has a lot of functionality and could arguably be pared down. It contains a number of very common low-level tools, along with GATK and a couple aligners. The `protein` image on the other hand only contains a handful of tools related to a very specific protein folding and visualization workflow.

When a new Flyte project is initialized, a default Dockerfile is created with Flyte dependencies built in. It’s easy to extend this with whatever dependencies you might need. ImageSpec is an extension of this, allowing you to specify dependencies on top of an existing image right inline with your task code. This image will be built and uploaded when your tasks and workflows are registered to a Flyte cluster.

### Datatypes

Having rich data types to enforce compatibility at the task boundary is essential to these wrapped tools working together. Flyte supports arbitrary data types through Python’s dataclasses library. Data types representing raw reads and alignment files allow us to reason about these files and their metadata more easily across tasks, as well as enforce naming conventions. Importantly, Flyte abstracts the object store, allowing you to load these assets into pods wherever is most convenient for your tool. This not only makes it easier to work with these files, but also safer as you’re working with ephemeral storage during execution instead of a full production filesystem.

### Tasks

While Flyte tasks are written in Python, there are a couple of ways to wrap arbitrary tools. ShellTasks are one such way, allowing you to define scripts as multi-line strings in Python. For added flexibility around packing and unpacking data types before and after execution, Flyte also ships with a subproc_execute function which can be used in vanilla Python tasks.

## Results

Tying it all together with a workflow.

## Conclusion

Different steps in a bioinformatics pipeline often require tools with significantly different characteristics. As such, different languages are employed where their strengths are best leveraged. Being able to wrap these executables and the data they operate over into a common orchestration layer presents an enormous benefit to the developer experience and consequently the reproducibility and extensibility of the research project as a whole.













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
