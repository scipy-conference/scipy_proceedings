---
# Ensure that this title is the same as the one in `myst.yml`
Title: Training a Supervised Cilia Segmentation Model from Self-Supervision
Abstract: |
  Cilia are organelles found on the surface of some cells in the human body that sweep rhythmically to transport substances. Dysfunctional cilia are indicative of diseases that can disrupt organs such as the lungs and kidneys. Understanding cilia behavior is essential in diagnosing and treating such diseases, but is often a labor and time-intensive task. A significant bottleneck to automating cilia analysis is a lack of automated segmentation. We develop a robust, self-supervised framework exploiting the visual similarity of normal and dysfunctional cilia.
---

## Introduction

Cilia are hair-like membranes that extend out from the surface of the cells and are present on a variety of cell types such as lungs and brain ventricles and can be found in the majority of vertebrate cells. Categorized into motile and primary, motile cilia can help the cell to propel, move the flow of fluid, or fulfill sensory functions, while primary cilia act as signal receivers, translating extracellular signals into cellular responses. Ciliopathies is the term commonly used to describe diseases caused by ciliary dysfunction. These disorders can result in serious issues such as blindness, neurodevelopmental defects, or obesity [@doi:10.1140/epje/s10189-021-00031-y]. Motile cilia beat in a coordinated manner with a specific frequency and pattern [@doi:10.1016/j.compfluid.2011.05.016]. Stationary, dyskinetic, or slow ciliary beating indicates ciliary defects. Ciliary beating is a fundamental biological process that is essential for the proper functioning of various organs, which makes understanding the ciliary phenotypes a crucial step towards understanding ciliopathies and the conditions stemming from it [@zain2022low].

Identifying and categorizing the motion of cilia is an essential step towards understanding ciliopathies. However, this is generally an expert-intensive process. Studies have proposed methods that automate the ciliary motion assessment. These methods rely on large amounts of labeled data that are annotated manually which is a costly, time-consuming, and error-prone task. Consequently, a significant bottleneck to automating cilia analysis is a lack of automated segmentation. Segmentation has remained a bottleneck of the pipeline due to the poor performance of even state-of-the-art models on some datasets. These datasets tend to exhibit significant spatial artifacts (light diffraction, out-of-focus cells, etc.) which confuse traditional image segmentation models.

Video segmentation techniques tend to be more robust to such noise, but still struggle due to the wild inconsistencies in cilia behavior: while healthy cilia have regular and predictable movements, unhealthy cilia display a wide range of motion, including a lack of motion altogether. This lack of motion especially confounds movement-based methods which otherwise have no way of discerning the cilia from other non-cilia parts of the video. Both image and video segmentation techniques tend to require expert-labeled ground truth segmentation masks. Image segmentation requires the masks in order to effectively train neural segmentation models to recognize cilia, rather than other spurious textures. Video segmentation, by contrast, requires these masks in order to properly recognize both healthy and diseased cilia as a single cilia category, especially when the cilia show no movement.

To address this challenge, we propose a two-stage image segmentation model designed to obviate the need for expert-drawn masks. We first build a corpus of segmentation masks based on optical flow thresholding over a subset of healthy training data with guaranteed motility. We then train a semi-supervised neural segmentation model to identify both motile and immotile data as a single segmentation category, using the flow-generated masks as “pseudolabels”. These pseudolabels operate as “ground truth” for the model while acknowledging the intrinsic uncertainty of the labels. The fact that motile and immotile cilia tend to be visually similar in snapshot allows us to generalize the domain of the model from motile cilia to all cilia. Combining these stages results in a semi-supervised framework that does not rely on any expert-drawn ground-truth segmentation masks, paving the way for full automation of a general cilia analysis pipeline.

The rest of this article is structured as follows: The Background section enumerates the studies relevant to our methodology, followed by a detailed description of our approach in the Methodology section. Finally, the next section delineates our experiment and provides a discussion of the results obtained.

## Background



## Methodology

Dynamic textures, such as sea waves, smoke, and foliage, are sequences of images of moving scenes that exhibit certain stationarity properties in time [@doi:10.1023/A:1021669406132]. Similarly, ciliary motion can be considered as dynamic textures for their orderly rhythmic beating. Taking advantage of this temporal regularity in ciliary motion, optical flow (OF) can be used to compute the flow vectors of each pixel of high-speed videos of cilia. In conjunction with OF, autoregressive (AR) parameterization of the optical flow property of the video yields a manifold that quantifies the characteristic motion in the cilia. The low dimension of this manifold contains the majority of variations within the data, which can then be used to segment the motile ciliary regions.

### Optical Flow Properties

Taking advantage of this temporal regularity in ciliary motion, we use OF to compute the flow vectors of each pixel using optical flow to obtain motion vectors of the region where cilia reside. We begin by extracting flow vectors of the video recording of cilia, under the assumption that pixel intensity remains constant throughout the video

```{math}
:label: of_formula
I(x,y,t)=I(x+u\delta t,y+v\delta t,t+\delta t)
```

Where $I(x,y,t)$ is the pixel intensity at position $(x,y)$ a time $t$. Here, $(u\delta t, v\delta t)$ are small changes in the next frame taken after $\delta t$ time, and $(u,v)$, respectively, are the optical flow components that represent the displacement in pixel positions between consecutive frames in the horizontal and vertical directions at pixel location $(x, y)$.

:::{figure} frame_out.png
:label: fig:frame_out
A sample frame of a video in our cilia dataset
:::
:::{figure} ground_truth.png
:label: fig:ground_truth
Manually labeled ground truth
:::
:::{figure} sample_OF.png
:label: fig:sample_OF
Representation of rotation (curl) component of optical flow at a random time
:::

### Autoregressive Modeling

@fig:sample_OF shows a sample of the optical flow component at a random time. From OF vectors, elemental components such as rotation are derived, which highlights the ciliary motion by capturing twisting and turning movements. To model the temporal evolution of these vectors, an autoregressive (AR) model [@doi:10.5244/C.21.76] is formulated:

```{math}
:label: AR
y_t =C\vec{x_t} + \vec{u} 
```

```{math}
:label: AR_state
\vec{x}_t = A_1\vec{x}_{t-1} + A_2\vec{x}_{t-2} + ... + A_d\vec{x}_{t-d} + \vec{v_t}
```

In equation {ref}`AR`, $\vec{y}_t$ represents the appearance of cilia at time $t$ influenced by noise $\vec{u}$. Equation {ref}`AR_state` represents the state $\vec{x}$ of the ciliary motion in a low-dimensional subspace defined by an orthogonal basis $C$ at time $t$, plus a noise term $\vec{v}_t$
and how the state changes from $t$ to $t + 1$.

Equation {ref}`AR_state` is a decomposition of each frame of a ciliary motion video $\vec{y}_t$ into a low-dimensional state vector $\vec{x}_t$
using an orthogonal basis $C$. This equation at position ${x}_t$ is a function of the sum of $d$ of its previous positions $\vec{x}_{t−1}, \vec{x}_{t−2}, \vec{x}_{t−d}$, each multiplied by its corresponding coefficients $A = {A_1, A_2, ..., A_d}$. The noise terms $\vec{u}$ and $\vec{v}$ are used to represent the residual difference between the observed data and the solutions to the linear equations. The variance in the data is predominantly captured by a few dimensions of $C$, simplifying the complex motion into manageable analyses.

In our experiments, we chose $d = 5$ as the order of our autoregressive model.
We can then create raw masks from this lower-dimensional subspace, and further enhance them with thresholding to remove the remaining noise.

:::{figure} AR matrices.png
:label: fig:sample_AR
A 5-order AR model of the OF component of the video
:::
In @fig:sample_AR, the first-order AR parameter is showing the most variance in the video, which corresponds to the frequency of motion that cilia exhibit. The remaining orders have correspondence with other different frequencies in the data caused by, for instance, camera shaking. Evidently, simply thresholding the first-order AR parameter is adequate to produce an accurate mask, however, in order to get a more refined result we subtracted the second order from the first one, followed by a normalization of pixel intensities. We used adaptive thresholding to extract the mask on all videos of our dataset. The generated masks exhibited under-segmentation in the ciliary region, and sparse over-segmentation in other regions of the image. To overcome this, we adapted a Gaussian blur filter followed by an Otsu thresholding to restore the under-segmentation and remove the sparse over-segmentation. @fig:thresholding illustrates the steps of the process.
:::{figure} thresholding.png
:label: fig:thresholding
The process of computing the masks. a) Subtracting the second-order AR parameter from the first-order, followed by b) Adaptive thresholding, which suffers from under/over-segmentation. c) A Gaussian blur filter, followed by d) An Otsu thresholding eliminates the under/over-segmentation.
:::

### Training the model

In this study, we employed a Feature Pyramid Network (FPN) [@kirillov2017unified] architecture with a ResNet-34 encoder. The model was configured to handle grayscale images with a single input channel and produce binary segmentation masks. We utilized Binary Cross-Entropy Loss for training and the Adam optimizer with a learning rate of $10^-3$. To evaluate the model's performance, we calculated the Dice score during training and validation. Data augmentation techniques, including resizing, random cropping, and rotation, were applied to enhance the model's generalization capability. The implementation was done using a library [@Iakubovskii:2019] based on PyTorch Lightning to facilitate efficient training and evaluation. The next section discusses the results of the experiment and the performance of the model in detail.

## Results

@fig:out_sample shows the predictions of the model on 5 sample videos of dyskinetic cilia. The dyskinetic samples were not used in the training or validation phases. These predictions were generated after only 5 epochs of training with the FPN architecture.

:::{figure} out sample.png
:label: fig:out_sample
The model predictions on 5 dyskinetic cilia samples. The first column shows a frame of the video, the second column shows the manually labeled ground truth, the third column is the model's prediction, and the last column is a thresholded version of the prediction.
:::
@tbl:metrics contains a summary of the model's performance.

:::{table} The performance of the model in training, validation, and testing phases.
:label: tbl:metrics
<table>
    <tr>
        <th rowspan="2">Phases</th>
        <th colspan="4" align="center">Metrics</th>
    </tr>
    <tr>
        <th align="center">IoU per image</th>
        <th align="center">IoU over dataset</th>
        <th align="center">Sensitivity</th>
        <th align="center">Specificity</th>
    </tr>
    <tr>
        <td align="left">Training </td>
        <td align="center">0.804</td>
        <td align="center">0.810</td>
        <td align="center">0.998</td>
        <td align="center">0.974</td>
    </tr>
    <tr>
        <td align="left">Validation </td>
        <td align="center">0.760</td>
        <td align="center">0.778</td>
        <td align="center">0.997</td>
        <td align="center">0.968</td>
    </tr>
    <tr>
        <td align="left">Testing </td>
        <td align="center">0.760</td>
        <td align="center">0.778</td>
        <td align="center">0.997</td>
        <td align="center">0.968</td>
</table>
:::

## Conclusions

In this paper, we introduced a self-supervised framework for cilia segmentation that bypasses the need for expert-labeled ground truth segmentation masks. Our method leverages the inherent visual similarities between healthy and unhealthy cilia to generate pseudolabels from optical flow-based motion segmentation of motile cilia. These pseudolabels serve as the ground truth for training a robust semi-supervised neural network capable of distinguishing motile and dyskinetic cilia regions with high accuracy. The proposed methodology addresses the challenge of automated cilia analysis by mitigating the reliance on labor-intensive and error-prone manual annotations.

Taking advantage of the orderly beating pattern in healthy cilia, we employed a two-stage segmentation process that effectively handles the inconsistencies and noise present in cilia microscopy videos. The deep learning model trained on the generated masks shows improved performance in accurately detecting ciliary regions, thus paving the way for more reliable automated cilia analysis pipelines. This advancement has the potential to significantly accelerate research in ciliopathies, facilitating better diagnostic and therapeutic strategies.

<!--  53244/53244 [35:33<00:00, 24.95it/s, loss=0.79, v_num=26, train_loss=0.732, valid_loss=0.777, valid_per_image_iou=0.854, valid_dataset_iou=0.864, valid_dice_score=0.927, valid_sensitivity=0.998, valid_specificity=0.982, train_per_image_iou=0.837, train_dataset_iou=0.851, train_dice_score=0.919, train_sensitivity=0.998, train_specificity=0.980] -->

<!-- ################################################################################# -->
<!-- ################################################################################# -->
<!-- ################################################################################# -->
<!-- ################################################################################# -->

## WHAT'S BELOW IS PART OF THE TEMPLATE

Twelve hundred years ago — in a galaxy just across the hill...

This document should be rendered with MyST Markdown [mystmd.org](https://mystmd.org),
which is a markdown variant inspired by reStructuredText. This uses the `mystmd`
CLI for scientific writing which can be [downloaded here](https://mystmd.org/guide/quickstart).
When you have installed `mystmd`, run `myst start` in this folder and
follow the link for a live preview, any changes to this file will be
reflected immediately.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
justo vitae odio tincidunt tempus at vitae tortor.

## Bibliographies, citations and block quotes

Bibliography files and DOIs are automatically included and picked up by `mystmd`.
These can be added using pandoc-style citations `[@doi:10.1109/MCSE.2007.55]`
which fetches the citation information automatically and creates: [@doi:10.1109/MCSE.2007.55].
Additionally, you can use any key in the BibTeX file using `[@citation-key]`,
as in [@hume48] (which literally is `[@hume48]` in accordance with
the `hume48` cite-key in the associated `mybib.bib` file).
Read more about [citations in the MyST documentation](https://mystmd.org/guide/citations).

If you wish to have a block quote,] you can just indent the text, as in:

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
