---
# Ensure that this title is the same as the one in `myst.yml`
title: Performing Object Detection on Drone Orthomosaics with Meta's Segment Anything Model (SAM)
abstract: |
  Accurate and efficient object detection and spatial localization in remote sensing imagery is a persistent challenge. In the context of precision agriculture, the extensive data annotation required by conventional deep learning models poses additional challenges. This paper presents a fully open source workflow leveraging Meta AI's Segment Anything Model (SAM) for zero-shot segmentation, enabling scalable object detection and spatial localization in high-resolution drone orthomosaics without the need for annotated image datasets. Model training and/or fine-tuning is rendered unnecessary in our precision agriculture-focused use case. The presented end-to-end workflow takes high-resolution images and quality control (QC) check points as inputs, automatically generates masks corresponding to the objects of interest (empty plant pots, in our given context), and outputs their spatial locations in real-world coordinates. Detection accuracy (required in the given context to be within 3 cm) is then quantitatively evaluated using the ground truth QC check points and benchmarked against object detection output generated using commercially available software. Results demonstrate that the open source workflow achieves superior spatial accuracy — producing output `20% more spatially accurate`, with `400% greater IoU` — while providing a scalable way to perform spatial localization on high-resolution aerial imagery (with ground sampling distance, or GSD, < 30 cm).
---

## Acknowledgments

We gratefully acknowledge the contributions of the open source community — thank you to the giants on whose shoulders we stand.

This work was funded by FiOR Innovations and Woodburn Nursery & Azaleas. We deeply appreciate their support and partnership.

A special thanks to Paniz Herrera, MBA, MSIST, Ryan Marinelli, PhD Fellow at the University of Oslo, and Danny Clifford for their help proofreading. 

## Introduction

Image segmentation is a critical task in geospatial analysis, enabling the identification and extraction of relevant features from high resolution remote sensing imagery [@wu23]. However, extracting actionable information (i.e., object detection and spatial localization) can be constrained by the need for large, labeled datasets to train deep learning models in order to then perform inference and (hopefully) produce the desired output. This bottleneck is particularly acute in agricultural domains, where variability in conditions and object types complicates manual annotation [@osco23].

Recent advances in foundation models, such as Meta AI’s Segment Anything Model (SAM), offer a promising path forward. SAM is designed for promptable “zero-shot” segmentation. “Prompt engineering”, in this context, involves using points and bounding boxes to focus the model’s efforts on more efficiently generating masks corresponding to objects of interest [@mayladan23]. Providing these prompts allows accurate masks to be generated for novel objects (ones not included in SAM’s training corpus), without domain-specific training. Masks can also be generated automatically with no such prompting. SAM’s automatic mask generator will effectively “detect” everything using open source model checkpoints and generate masks for each object in a provided image [@kirillov23].

While SAM’s ability to generalize is impressive [@kirillov23; @osco23], its performance on remote sensing imagery and fine-grained features requires careful workflow integration and evaluation [@wu23]. This paper describes a comprehensive, open source workflow for object detection and spatial localization in high-resolution remote sensing imagery, built around SAM and widely used geospatial Python libraries [@gdal; @shapely; @geopandas; @rasterio]. The complete process is delineated, from data loading and preprocessing to mask generation, post-processing, and quantitative accuracy assessment, culminating in a rigorous comparison against the results produced using the proprietary software (see [code](#code)). Precision, accuracy, F1 score, mean deviation (in cm), and Intersection-over-Union (IoU) are calculated in order to quantify the relative quality of the output produced using each workflow[^footnote-1].


[^footnote-1]: Output evaluation details are discussed in the [Appendix](#accuracy-evaluation-methodology).

## Motivation

Precision agriculture relies on accurate object detection for tasks such as plant counting, health monitoring, and the automated control of heavy equipment. Traditional deep learning approaches can become hindered by the cost and effort of generating carefully annotated data, limiting scalability and accessibility. Proprietary solutions, while effective, can be expensive and opaque, impeding reproducibility and customization.

:::{figure} trimmer.jpg
:label: fig:trimmer
The derived centroids of the objects detected in drone orthomosaics are used to automate this nursery trimmer.
:::

SAM’s zero-shot segmentation capability directly addresses the data annotation bottleneck, enabling rapid deployment in novel contexts. By developing an open source workflow around SAM, an end-to-end pipeline is created which allows for the quantitative evaluation of spatial accuracy with respect to objects detected in high-resolution aerial imagery. This modular workflow can also be repurposed as an automated data annotation pipeline for downstream model training/fine-tuning, if required[^footnote-2].

[^footnote-2]: See [this Colab notebook](https://colab.research.google.com/drive/1WNZFDr8bMMi51fOo5vv-TrVSBGW56yvA?usp=sharing#sandboxMode=true) for details.

## Approach

Our approach integrates SAM’s segmentation strengths with traditional geospatial data processing techniques, which lends itself to our precision agriculture use case. The workflow, like any other, can be thought of as a sequence of steps (visualized above and described below), each with their own sets of substeps:

- **Data Ingestion**: Loading GeoTIFF orthomosaics and QC point CSVs, extracting spatial bounds and coordinate reference systems (CRS) using Rasterio or GDAL.
- **Preprocessing**: Filtering QC points to those within image bounds, standardizing coordinate columns, and saving filtered data for downstream analysis.
- **Mask Generation**: Tiling large images for efficient processing, running SAM’s automatic mask generator (**_ViT-H_** variant) on each tile, and filtering masks by confidence.
- **Post-Processing**: Converting masks to polygons, filtering by area and compactness, merging overlapping geometry, and extracting centroids.
- **Accuracy Evaluation**: Calculating point-to-centroid deviations (in centimeters) between detected objects and QC points, compiling results, and generating visual and tabular reports.
- **Benchmarking**: Quantitatively comparing SAM-based results against the evaluated output using identical evaluation metrics (precision, recall, IoU, etc.; see [Appendix](#accuracy-evaluation-methodology) for details).

It should be noted that there are no model training or fine-tuning steps included in our workflow, as we are using a foundation model to generate masks. This is analogous to using ChatGPT to generate text, which does not require users to train or fine-tune the underlying foundation model in order to do so.

This approach is carried out entirely using open source Python libraries, ensuring transparency and extensibility.

## Methodology

### Data and Environment

- **Imagery**: High-resolution GeoTIFF orthomosaic.
  - **Image size**: 18,200 x 55,708; 1,013,885,600 pixels
  - **Total area**: 545,243 sq ft; 12.5 acres
  - **GSD**: 0.71 cm
- **Ground Truth**: QC points in CSV format, containing spatial coordinates and unique identifiers.
- **Coordinate Reference System (CRS) Transformations**: All spatial operations are performed using the NAD83 CRS (EPSG:6859), with reprojection to the WGS84 CRS (EPSG:4326) for downstream reporting and nursery trimmer automation.
- **Dependencies**[^footnote-3]: 
  - GDAL [@gdal]
  - GeoPandas [@geopandas]
  - Matplotlib [@matplotlib]
  - NumPy [@numpy]
  - OpenCV [@opencv]
  - OpenPyXL [@openpyxl]
  - Pandas [@pandas1; @pandas2]
  - Pillow [@pillow]
  - Rasterio [@rasterio]
  - Segment Anything[^footnote-4] [@kirillov23]
  - Shapely [@shapely]
  - Torch [@pytorch]

[^footnote-3]: See [requirements.txt](https://raw.githubusercontent.com/nickmccarty/scipy-2025/refs/heads/main/requirements.txt) for version details.

[^footnote-4]: Inference was accelerated using `CUDA 12` (`cuDF 25.2.1`) on a `T4` GPU within our Colab notebook environment.

### Workflow

1. [Data Ingestion](#data-ingestion-and-preprocessing)
2. [Mask Generation](#mask-generation)
3. [Post Processing](#post-processing)
4. [Accuracy Evaluation](#accuracy-evaluation)
5. [Benchmarking](#benchmarking)

(data-ingestion-and-preprocessing)=
#### Data Ingestion and Preprocessing

```{math}
\begin{array}{ll}
\textbf{a.} & \text{Load GeoTIFF file(s)} \\
\quad & \text{Extract image bounds: } (\text{min}_x, \text{min}_y, \text{max}_x, \text{max}_y) \\
\quad & \text{Extract coordinate reference system (CRS)} \\
\\
\textbf{b.} & \text{Load QC point CSV} \\
\quad & \text{Clean and standardize column names} \\
\quad & \text{Assign unique IDs to each QC point} \\
\quad & \text{Reproject QC points to match image CRS (if needed)} \\
\\
\textbf{c.} & \text{For each QC point:} \\
\quad & \textbf{if} \text{ point within image bounds} \textbf{ then} \\
\quad\quad & \text{Keep point} \\
\quad & \textbf{else} \\
\quad\quad & \text{Discard point} \\
\\
\textbf{d.} & \text{Save filtered QC points to output CSV}
\end{array}
```

(mask-generation)=
#### Mask Generation

```{math}
\begin{array}{ll}
\textbf{a.} & \text{Initialize SAM (ViT-H checkpoint)} \\
\quad & \text{Use GPU if available} \\
\\
\textbf{b.} & \text{Tile ("chip") input images} \\
\quad & \text{Iteratively generate masks using SAM} \\
\\
\textbf{c.} & \text{Filter masks with confidence } \geq 80\% \\
\\
\textbf{d.} & \text{Convert masks to polygons} \\
\quad & \text{Aggregate results and export as GeoJSON}
\end{array}
```

(post-processing)=
#### Post-Processing

```{math}
\begin{array}{ll}
\textbf{a.} & \text{Filter polygons (e.g., based on area threshold)} \\
\\
\textbf{b.} & \text{Merge overlapping polygons} \\
\quad & \text{Extract individual (non-overlapping) polygons} \\
\\
\textbf{c.} & \text{Extract centroids from polygons} \\
\end{array}
```

(accuracy-evaluation)=
#### Accuracy Evaluation

```{math}
\begin{array}{ll}
\textbf{a.} & \text{Load merged geometry and filtered QC points} \\
\\
\textbf{b.} & \text{Iteratively compute geometric deviation from QC points} \\
\quad & \text{Output deviations in centimeters}
\end{array}
```

See [Appendix](#accuracy-evaluation-methodology) for accuracy evaluation methodology details.

(benchmarking)=
#### Benchmarking 

```{math}
\begin{array}{ll}
\textbf{a.} & \text{Apply accuracy evaluation steps to proprietary software output} \\
\\
\textbf{b.} & \text{Compare workflows using IoU, precision, recall, and F1 scores}
\end{array}
```

See [code](#code) for benchmarking methodology details.

## Results

### Proprietary Workflow

:::{figure} proprietary-workflow-output-evaluation.png
:label: fig:proprietary-workflow-output-evaluation
19 false positives (FP) and hundreds of sliver polygons were observed in the output produced using the proprietary software.
:::

The bounding boxes that were output using this workflow (against which we are benchmarking ours) can be viewed as a layer overlain onto the GeoTIFF orthomosaic using GIS software[^footnote-5]. What was of particular use to us is the fact that zero false negatives (FN) were observed in the output, though 19 false positives (FP) were. This empirical knowledge equips us with something not usually possessed in use cases such as this: the number of true positives (TP), which allows us to leverage such metrics as precision, recall, and the harmonic mean of the two, F1 score, to perform a rigorous comparison against our open source workflow (see [code](https://colab.research.google.com/drive/1pwnb14s2i7n_VAlfwhBqzDQ0cOb9oGs-?usp=sharing#sandboxMode=true&scrollTo=VNWvzNKU-ePt)). 

[^footnote-5]: We used open source QGIS [@qgis] as our selected data viewer.

### Open Source Workflow

:::{figure} open-source-false-positive-collage.png
:label: fig:open-source-workflow-output-evaluation
18 FP were observed in the output produced using the open source workflow. 
:::

Merging the overlapping geometry and filtering out the empirically observed FP allowed for us to ascertain exactly how many TP (18,736) there are in the benchmark output and derive how many FP (18) and FN (65) there are in our workflow output, which enabled us to conduct our performance comparison.

:::{table} Performance Comparison  
:label: tbl:performance-comparison

<table>
<tr>
  <th rowspan="2" align="center">Workflow</th>
  <th colspan="3" align="center">Detection Quality Metrics</th>
  <th colspan="2" align="center">Localization Accuracy Metrics</th>
</tr>
<tr>
  <th align="center">Precision</th>
  <th align="center">Recall</th>
  <th align="center">F1 Score</th>
  <th align="center">Mean Deviation (cm)</th>
  <th align="center">IoU</th>
</tr>
<tr>
  <td align="center"><b><i>Proprietary</i></b></td>
  <td align="center">0.9990</td>
  <td align="center">1.0000</td>
  <td align="center">0.9995</td>
  <td align="center">1.39</td>
  <td align="center">0.18</td>
</tr>
<tr>
  <td align="center"><b><i>Open Source</i></b></td>
  <td align="center">0.9990</td>
  <td align="center">0.9956</td>
  <td align="center">0.9973</td>
  <td align="center">1.20</td>
  <td align="center">0.74</td>
</tr>
</table>
:::

It can be observed that empty plant pots tend to be ~64 pixels (px) wide and tall; with QC points corresponding to actual pot centroids, we were able to create 64-by-64px boxes to facilitate our IoU calculations (see [code](https://colab.research.google.com/drive/1NqDTYw0V9yRnZtoT6Pc7ZJ3ATe7CTue8?usp=sharing#sandboxMode=true)). These calculations further allow us to assess the relative alignment between the detection output geometry and our "ground truth" geometry.

This work makes it easy to identify down to the individual QC point ID level which detection centroids deviate from said point by more than 3 cm, which is the tolerance specified by our client. In aggregate, we were able to gain a quantified sense of the mean deviation (in cm) of the output produced by each workflow. However, visual inspection revealed that some detection geometry was flagged as having out-of-tolerance centroids when the QC points were themselves off-center. This is to say that some detections from both workflows were flagged as being out-of-tolerance when they observably were not.

:::{figure} qc-point-91-collage.png
:label: fig:qc-point-91
Visual inspection of the detected centroids relative to QC point 91 reveal that the QC point is off-center.
:::

Visual inspection also reveals that our detections (the pink circle) and those produced using the commercial software (the beige square) have greater overall coverage with respect to the QC geometry (the grey square). This provides intuition as to why the IoU calculations revealed a 400% increase in coverage with respect to the geometry produced using SAM's automatic mask generator, zero-shot.

## Discussion

### Key Findings

The open-source workflow powered by Meta AI’s Segment Anything Model (SAM) outperformed a commercial alternative in object detection and spatial localization on high-resolution drone imagery. It achieved `20% higher spatial accuracy` (1.20 cm vs. 1.39 cm deviation) and a `400% higher Intersection-over-Union (IoU)` (0.74 vs. 0.18), indicating stronger alignment between the detections and the actual object boundaries. Both methods had near-perfect precision, but the open-source approach showed slightly lower recall due to 65 false negatives. It should be noted, however, that these FN were a direct result of the filtering substep in our workflow, which excludes detections outside of the provided geometry area and compactness thresholds; see [code](https://colab.research.google.com/drive/1pwnb14s2i7n_VAlfwhBqzDQ0cOb9oGs-?usp=sharing#sandboxMode=true&scrollTo=240nXaT5-EqM). 

Nevertheless, its overall performance supports its suitability for precision agriculture and downstream automation.

### Precision Agriculture Challenges

Our work began with an eye toward tackling a major challenge in agricultural remote sensing: the need for extensive manual annotation. SAM’s zero-shot segmentation enables accurate object detection without domain-specific training, making it scalable and adaptable for new use cases with minimal setup.

### Benefits of Open Source

Built entirely on open source geospatial tools, our open source workflow offers transparency, reproducibility, and flexibility. It can be tailored to suit various tasks, even automating annotation for model training, thereby supporting broader adoption with respect to high-resolution remote sensing imagery, in general.

### Practical Impact

Meeting professional-grade tolerance requirements (e.g., < 3 cm) enables real-world applications, such as automating heavy equipment, based on precise object localization. This demonstrates how automated workflows can reduce manual labor and support more efficient agricultural practices.

### Limitations and Future Work

Our approach to tiling ("chipping") high-resolution orthomosaics, processing 588 individual 1280-by-1280px tiles at an average pace of 11 seconds per tile, required a total processing time of ~110 minutes running on a Colab single `T4` GPU instance. It is important to note that an overlap of 25% (320px) between tiles during processing was required to ensure that geometry was not produced containing "holes" or malformations; merging overlapping polygons after filtering (based on area and compactness calculations, in this case) helped us ensure the overall quality of the geometric output.

Future work will be centered on building an open source CLI and Python package[^footnote-6], which will allow users to pass orthomosaics as inputs and get geometry meeting desired spatial charactersitics as an output.

[^footnote-6]: We have since open-sourced the [`orthomasker`](https://pypi.org/project/orthomasker) Python package and CLI; work on a GUI is currently underway.

## Conclusion

We present a robust, open source workflow for object detection and spatial localization in high-resolution drone orthomosaics, leveraging SAM’s zero-shot segmentation capabilities. Our quantitative evaluation demonstrates improved accuracy over a commercially available software solution, underscoring the potential of foundation models and open source tools to advance scalable, cost-effective feature extraction. This work provides a template for further research and deployment in diverse contexts.

To our knowledge, this is the first comparative evaluation of an open source segmentation model against commercial software in a context requiring high (< 3 cm) spatial accuracy. Our results demonstrate that the workflow not only matches but in some cases exceeds performance metrics with respect to the evaluated output.

## Conflicts of Interest

The author declares no conflicts of interest.

## AI Usage Disclosure

AI tools (ChatGPT, Perplexity, and NotebookLM) were used:

- in writing portions of the workflow integration code,
- to generate Matplotlib subplots, process flow diagrams, pseudocode, <span style="font-family: serif;">L<span style="vertical-align: 0.4ex; font-size: 0.8em;">A</span>T<span style="vertical-align: -0.3ex; font-size: 0.8em;">E</span>X</span>, etc.
- for proofreading and light revision to reduce potential publication errors.

(code)=
## Code

Data and code required to replicate our approach can be found using the links below:

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nickmccarty/scipy-2025)
<a href="https://colab.research.google.com/drive/1pwnb14s2i7n_VAlfwhBqzDQ0cOb9oGs-?usp=sharing#sandboxMode=true">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28px">
</a>

## Appendix

(accuracy-evaluation-methodology)=
### Accuracy Evaluation Methodology

The evaluation focuses on two primary categories of metrics: ***localization accuracy*** and ***detection quality***; the employed methodology relies on the following data:

- **Ground Truth Quality Control (QC) Points** ($P_{QC}$), defined as a set of $N_{QC}$ known spatial coordinates, $P_{QC} = \{p_j\}_{j=1}^{N_{QC}}$, where each $p_j = (x_j, y_j)$ represents the centroid (in real-world coordinates) of our object of interest (empty plant pots), serving as the ground truth for spatial localization. 
- **Detected Object Centroids** ($C_{Det}$) are used, which are a set of $N_{Det}$ centroids, $C_{Det} = \{c_k\}_{k=1}^{N_{Det}}$, where each $c_k = (x'_k, y'_k)$ is the centroid extracted from a polygon representing an object detected by the workflow. 
- **Detected Object Polygons** ($G_{Det}$) are included, representing a set of $N_{Det}$ polygons, $G_{Det} = \{g_k\}_{k=1}^{N_{Det}},$ where each $g_k$ is a polygon generated from a SAM-produced mask after post-processing. 
- **Ground Truth Polygons** ($G_{GT}$) the calculation of Intersection over Union (IoU) implies the existence of a corresponding set of $N_{QC}$ ground truth polygons, $G_{GT} = \{g'_j\}_{j=1}^{N_{QC}}$, where each $g'_j$ delineates the extent of the object associated with ground truth point $p_j$. This allows for the quantification of the spatial alignment between the detection bounding boxes and those associated with the QC points, in aggregate. The provided [code](#code) details how we created this geometry and performed the calculations.