---
# Ensure that this title is the same as the one in `myst.yml`
title: 'Ocetrac: An Object-Based Framework for Tracking and Quantifying Climate Extremes in Gridded Datasets'
abstract: |
  Climate extremes such as marine heatwaves and ocean acidification events appear as dynamic, irregular structures in gridded datasets. Software tools are available for detecting these events. However, there is a need for standardized methods for tracking their spatiotemporal evolution and evolving characteristics (such as intensity, shape, and duration). Ocetrac is an open-source Python package that bridges this gap by integrating morphological image processing (via scikit-image) with xarray-based data workflows to track and quantify amorphous two-dimensional climate features in space and time. The tracking core identifies and tracks objects that split and merge. A second submodule, Measures, computes shape-, motion-, and intensity-based measures. These measures enable comparison between similar events that occur at different times (e.g., via clustering), machine learning integration such as feature extraction for classification/regression (e.g., scikit-learn), and process-oriented analysis such as linking object behavior to physical drivers (e.g., surface heat fluxes). Designed for the Scientific Python ecosystem, Ocetrac simplifies workflows from detection to downstream analysis, supporting applications in climate science, oceanography, and atmospheric research. 
---
## Introduction
Gridded climate datasets are important for studying phenomena like marine heatwaves (MHWs), deoxygenation zones, and atmospheric blocking events among others. These events, or structures, exhibit complex spatiotemporal behavior including splitting, merging, and  complex evolution. The ecological impact of these events are closely tied to their spatial and temporal characteristics.

Conventional analysis approaches, including gridpoint-based statistics, fixed-region summaries, and empirical orthogonal functions (EOFs), offer valuable insight but fall short in characterizing events as coherent and evolving structures. Gridpoint statistical analysis obscures spatial connectivity (CITE) while EOFs lack object-level granularity (CITE). Simple threshold-based detection (e.g., `TempestExtremes`, CITE) is optimized for compact features like cyclones, but these algorithms perform poorly for  amorphous structures and lack the ability to capture the spatial and temporal evolution of events. For applications like attribution studies, understanding how an event evolves, not just where it occurs or how intense it is, is essential (CITE).

Object-based approaches treat climate features as coherent objects in space $(x, y)$ and time $(y)$.  Object-based approaches thus enable direct evaluation of properties like shape, connectivity, and persistence. These added metrics allow for aggregating analysis of multiple events within large datasets that easily allows further statistical analyses. 

Here, we describe Ocetrac [Ocetrac](https://ocetrac.readthedocs.io/en/latest/), an open-source Python package designed for detecting, tracking, and quantifying irregular, evolving features in gridded climate data. Ocetrac combines morphological image processing (via `scikit-image`) with labelled array workflows (via `xarray`) to characterize irregular structures. Its modular design supports: 
1. Detection: Ocetrac smooths spatially connected regions
2. Tracking: Ocetrac labels objects across time steps, even when they split or merge
2. Quantification: Ocetrac computes shape (e.g., compactness), motion (e.g., centroid velocity), and intensity (e.g., thermal anomaly) metrics
3. Integration: Ocetrac is compatible with the Scientific Python ecosystem (`dask`; `scikit-learn`)

Originally developed for MHW analysis, Ocetrac can be used for other applications, from atmospheric rivers to hypoxic zones and bloom patches in the ocean. Its modular design allows users to customize detection thresholds, tracking logic, and metrics of tracked objects. This paper details its design, demonstrates several use cases, and outlines future directions. 

## Goals and Motivations
### Software Landscape
### Motivating the Measures Submodule

## Ocetrac: Tracking and Measures Submodules
Ocetrac is a flexible tracking framework designed for geophysical feature analysis. While we demonstrate its application using SST anomalies to identify MHWs, the algorithm itself is variable-agnostic. Ocetrac can be applied to any two-dimensional spatiotemporal field that can be thresholded into spatially coherent features that is defined on a regular grid (after interpolation if necessary), and has sufficient temporal resolution to resolve event evolution. The core tracking methodology is described briefly below and in detail in Scannell et al. (2024) (CITE) where it was used to examine MHWs throughout the globe. To support event characterization beyond simple tracking, we also introduce the Measures submodule, which enables the calculation of a suite of geometric, intensity-based, and motion-related diagnostics for each tracked event. The Ocetrac workflow is demonstrated below in [Figure %s](#fig:ocetrac).

:::{figure} ocetrac_workflow.jpg
:label: fig:ocetrac
Ocetrac workflow diagram. 
:::

### Input Specifications and Preprocessing
Ocetrac requires input data as a 3D `xarray.DataArray` with dimensions ordered as $(t, y, x)$, where $(y, x)$ correspond to regular spatial coordinates, typically latitude and longitude, and $t$ corresponds to time. The time dimension should be uniformly spaced to ensure optimal tracking performance of features through the temporal sequence. The time dimension should also have high enough spatial and temporal resolution to capture the temporal evolution of the target feature.  Temporal gaps in the data can be filled using linear temporal interpolation to maintain continuity in the time series. The spatial grid should be regular (i.e., evenly spaced in both $y$ and $x$). Datasets with irregular or unstructured spatial grids must be interpolated to a regular grid prior to Ocetrac. Methods for interpolating spatial fields include bilinear interpolation and nearest-neighbor interpolation. 

An optional binary land mask (1 for valid grid cells, 0 for excluded regions such as land or sea ice) can be provided to omit specific areas from detection and tracking. All preprocessing, such as detrending and anomaly calculation, thresholding, must be performed before using Ocetrac. Common thresholding approaches include: percentile-based (e.g., values exceeding the 90th percentile of anomalies), absolute value thresholds (e.g., values exceeding 28°C), or statistical significance thresholds (e.g., values exceeding two standard deviations from the mean). Ocetrac is agnostic to the thresholding method, as long as the input is a binary spatiotemporal field.

### Tracking Submodule
The process begins with converting anomalies into binary maps using a threshold (typically the 90th percentile for MHWs). Grid cells exceeding the threshold are marked as features (active = 1); all others are inactive (0).

To refine features, Ocetrac applies sequential morphological operations: 
1. Closing (dilation followed by erosion): Fills small holes within features and connects nearby regions belonging to the same object
2. Opening (erosion followed by dilation): Removes isolated pixels, smooths feature boundaries, and cleans residual artifacts introduced by closing

Empirically, this sequence (closing followed by opening) optimizes feature integrity while minimizing fragmentation. The intermediate results of morphological operations are shown and described in greater detail in Scannell et al. (2024) (CITE).

The operations use a circular structuring element with radius R (in grid cells). For 0.25° resolution data:
- R = 4 to 6 grid cells (1 - 1.5°): Preserves smaller-scale features while removing noise
- R = 6 to 8 grid cells (1.5 to 2°): Emphasizes larger, more coherent structures
- R > 8 grid cells: May merge distinct features or even fail to identify features as valid objects altogether

The choice of R represents a trade-off between feature completeness (higher R retains more connected areas) and spatial precision (lower R preserves finer-scale details) ([Figure %s](#fig:radius_size)). Users should validate this parameter against known feature scales in their domain. Features, after undergoing morphological operations, are now referred to as objects.

Ocetrac then applies size-based filtering to focus on the most spatially coherent objects. It calculates the area of each detected object in grid cells, then removes objects smaller than a user-defined threshold. By default, it keeps objects larger than 75% of all detected features (the 75th percentile), meaning the smallest 25% of objects are filtered out. This filtering step helps eliminate noise and isolate the most physically meaningful objects in the dataset.

:::{figure} radius_var.jpg
:label: fig:radius_size
Objects outlined (purple outlines) detected using varying R values (R ranging from 1° to 7°). The background shows sea surface temperature anomalies (°C) from April 1979 of an ensemble member of the CESM2-LENS dataset.
:::

Ocetrac identifies objects in each time step by grouping together adjacent pixels, including diagonal connections (8-connectivity). Then, it matches objects that appear in either the exact same grid cell or directly adjacent cells (up/down/left/right) in the next timestep (6-connectivity). This conservative approach allows objects to move gradually and prevents discontinuous jumps. The system maintains consistent ID numbers for objects as they evolve and handles global datasets where the map wraps around the globe at 0°/360° longitude, and tracks when objects merge together or split apart. The only requirement is that objects must overlap by at least one grid cell from one time step to the next to be in the same event.

**Implementation example**
```python
# Initialize tracker with user-defined parameter
obj_Tracker = ocetrac.Tracker(
    binary_out_afterlandmask,   # Thresholded binary field
    newmask,                    # Land/ice mask (optional)
    radius=3,                   # R in grid cells
    min_size_quartile= 0.75,    # P (determines minimum object size)
    timedim = 'time',           # Time dimension name
    xdim = 'lon',               # Longitude dimension name
    ydim='lat',                 # Latitude dimension name
    positive=True)              # Track positive anomalies

# Execute tracking and morphological operations
blobs = obj_Tracker.track()

mo = obj_Tracker._morphological_operations()
```
Here Ocetrac provides basic diagnostics (e.g., initial/final object counts, area retention percentage) to evaluate tracking performance and feature characteristics. 

### Measures Submodule
The output of the tracker is a `xarray.Dataset` with many labelled events. The Measures submodule allows users to process large numbers of tracked events by automating repetitive calculations. Its standardized outputs allow for direct comparison across different events and datasets, which streamlines analysis workflows. The submodule supports both individual case studies and ensemble analyses. In brief, the Measures submodule processes the output of Ocetrac’s core tracking algorithm and provides quantitative metrics to characterize detected events. These measures fall into five categories: shape, motion, intensity, and temporal, and contextual (Table 1). Shape measures quantify geometric properties including area, perimeter, and circularity that are relevant for quantifying structural evolution. Motion measures track movement via centroids and intensity-weighted centers of mass, while also handling longitudinal wrapping. Intensity measures capture magnitude variations through spatial statistics (mean, maximum, percentile extremes) of the underlying anomaly field. Temporal measures record lifecycle characteristics including duration, initiation timing, and timing peaks of intensity and area. Some of the measures can also be used to create contextual measures (e.g., object counts per timestep) which capture splitting and merging objects. Together, these measures enable analysis of event dynamics. The measures are summarized in [Table %s](#tbl:event-measures).

```{raw} latex
\begin{table*}
  \begin{longtable*}{|l|l|l|l|}
  \hline
  \textbf{Category} & \textbf{Measure} & \textbf{Definition} & \textbf{Interpretation} \\
  \hline
  \endfirsthead
  
  \hline
  \textbf{Category} & \textbf{Measure} & \textbf{Definition} & \textbf{Interpretation} \\
  \hline
  \endhead
  
  \hline
  \multirow{5}{*}{SHAPE} & Area (Extent) & Spatial coverage in km² or grid cells. Calculated as the sum of cell areas within bounds. & Larger values = greater spatial influence \\
  \cline{2-4}
  & Perimeter & Boundary length (km), computed via geodesic contours. Calculated as Haversine sum of contour points. & Complex shapes yield high perimeters. \\
  \cline{2-4}
  & Circularity & Deviations from a perfect circle. Calculated as $4\pi\frac{Area}{Perimeter^2}$. & 1 = perfect circle; 0 = highly irregular \\
  \cline{2-4}
  & Deformation & Shape stability between timesteps. Calculated as $1 - \frac{Shared\ Area}{Total\ Area}$. & 0 = no change; 1 = complete deformation \\
  \cline{2-4}
  & Convex Hull Area & Object area relative to its convex hull area. Calculated as $\frac{Area}{Convex\ Hull\ Area}$. & Lower values = more concave/irregular \\
  \hline
  
  \multirow{4}{*}{MOTION} & Centroids per timestep & Geometric centers of all objects at each timestep, handling longitudinal wrapping (0°-360°). & Used to identify object locations. \\
  \cline{2-4}
  & Centroid displacement & Distance (km) between centroids across timesteps. Calculated using Haversine distance. & Tracks movement paths; large values = faster movement \\
  \cline{2-4}
  & Center-of-mass coordinates & Intensity-weighted mean position per timestep. & Reflects mass distribution shifts \\
  \cline{2-4}
  & Center-of-mass displacement & Distance between center-of-mass positions across timesteps. & Quantifies intensity-weighted movement \\
  \hline
  
  \multirow{5}{*}{INTENSITY} & Cumulative & Sum of intensity values across space per timestep & Total event magnitude \\
  \cline{2-4}
  & Mean & Spatial average intensity & Baseline event strength \\
  \cline{2-4}
  & Max. & Spatial maximum intensity & Peak local magnitude \\
  \cline{2-4}
  & Std. Dev. & Spatial variability & Higher values = more heterogeneous \\
  \cline{2-4}
  & Percentile & Threshold-exceeding intensity (e.g., 90th) & Robust extreme value detection \\
  \hline
  
  \multirow{4}{*}{TEMPORAL} & Initial Detection Time & First timestep of event occurrence & Event onset \\
  \cline{2-4}
  & Duration & Total timesteps the event persists & Distinguishes transient vs. persistent \\
  \cline{2-4}
  & Peak Intensity Timing & When max intensity occurs (\% duration) & Early peak = rapid intensification \\
  \cline{2-4}
  & Peak Area Timing & When maximum extent occurs & Early peak = rapid growth \\
  \hline
  
  CONTEXTUAL & Object counts per timestep & Number of distinct objects detected & Higher counts indicate fragmentation \\
  \hline
  
   \caption{Event measures in the Measures Submodule \label{tbl:event-measures}}
  \end{longtable*}
\end{table*}
```

The computed measures are designed for application across both individual events and multiple events. Each measure is stored in a structured Python dictionary keyed by event ID, supporting single-event analysis and multi-event comparison, where ensemble statistics can be calculated when aggregating across multiple event IDs. The submodule includes built-in visualization tools that generate plots of trajectory maps (centroid/center-of-mass paths with arrow markers indicating directionality). The nested dictionary output structure accommodates mixed dimensionality, where scalars (e.g., maximum intensity) coexist with vectors (e.g., centroid path) and expansion, where new measures can be added without restructuring existing outputs. This approach is particularly valuable for comparative analyses, where aggregating specific measures across events needs only simple dictionary comprehensions. The Measures submodule allows for user-defined extension, which is an important feature for domain-specific applications.

**Implementation example**
```python
object_ids = [8., 11., 16.]           # Example IDs from Ocetrac tracking output

# Toggle which measures to compute for the objects (True/False flags)
run_shape_flag = True       
run_motion_flag = False               # Example: turn off motion measures
run_temporal_flag = True
run_intensity_flag = True


results_for_objects = process_objects_and_calculate_measures(
   object_ids,                        # List of object IDs to process
   blob_data = labelled_field,        # Ocetrac's labeled output (tracked objects)
   intensity_data = intensity_field,  # Intensity data (e.g., SST anomalies)
   run_shape=run_shape_flag,
   run_motion=run_motion_flag,
   run_temporal=run_temporal_flag,
   run_intensity=run_intensity_flag,
   lon_resolution_value=lon_resolution_value,
   lat_resolution_value=lat_resolution_value
)
```

### Flexibility and Extensibility
Ocetrac’s modularity supports diverse applications beyond MHWs. This tracking framework could be used to analyze atmospheric systems (with geopotential height data), precipitation events, ocean eddies (using sea surface height anomalies), and freshwater plumes (via salinity fields), amongst others. Ocetrac integrates with xarray for labeled data handling, dask for parallel processing, and common geospatial toolkits, making the package suitable for studying different geophysical phenomena.  

## Example use cases
To demonstrate Ocetrac's broad applicability, we present several example use cases focused on MHWs. These examples illustrate Ocetrac’s capabilities in analyzing global patterns, investigating regional dynamics, placing events within a historical context, and evaluating forecast skill.

### Application to Observational Data
We apply Ocetrac to the NOAA 1/4° Optimum Interpolation Sea Surface Temperature (OISST) dataset (LINK) (CITE) to demonstrate its functionality for real-world MHW analysis. This methodology for anomaly calculation and results is described in greater detail in Scannell et al. (2024) (CITE). This study demonstrates Ocetrac successfully identifying known MHW events and their spatial structure, temporal evolution, and intensity distributions from the observational record. This shows how Ocetrac captures both large-scale and regional MHW events in observational products. The workflow is shown in [Figure %s](#fig:JTECH). 

:::{figure} jtech_workflow.jpg
:label: fig:JTECH
Ocetrac workflow for MHW analysis in an observational data product (in this example, the NOAA 1/4° Optimum Interpolation Sea Surface Temperature). The diagram illustrates the processing pipeline from model output through event detection, tracking, and characterization of observational MHWs. 
:::

### Characterizing Events in a Large Ensemble
This use case demonstrates how Ocetrac allows for comprehensive analysis of MHWs across a large ensemble of climate simulations. By processing many ensemble members, we generate an extensive event catalog that is orders of magnitude larger than the observational record. This provides robust statistics for MHW characteristics calculated using the Measures submodule, This expanded dataset supports clustering analyses to identify recurrent event types and their associated physical drivers.

The large number of simulated events allows us to address key questions about MHW diversity: What canonical patterns emerge? How do their dynamical drivers differ? Such analyses would not be possible using the observational record alone. The approach is particularly valuable for characterizing rare but high-impact events, where large ensembles provide sufficient samples to assess their physical rarity and driving mechanisms. The workflow is shown in [Figure %s](#fig:LENS). 

:::{figure} lens_workflow.jpg
:label: fig:LENS
Ocetrac workflow for MHW analysis in climate ensembles. This approach allows for statistical analysis of event properties.
:::

### Comparing Observed Marine Heatwaves Against Many Simulated Events
This example uses Ocetrac to compare MHWs in the observational record (i.e., ‘The Blob’ MHW in 2014 to 2016) with events simulated in a large ensemble of climate models (i.e., CESM2 LENS (LINK) (CITE)). The tool provides a probabilistic framework for assessing their rarity, spatial structure, and dynamical drivers. By applying consistent object-based detection and tracking to both observations and model output, Ocetrac allows direct comparisons that address questions such as (1) How unusual was the observed event in the context of internal climate variability?, (2) What is the modeled likelihood of similar or more intense MHWs?, and (3) What types of MHWs does the model suggest are possible but have not yet been observed? An analysis workflow is illustrated in [Figure %s](#fig:compare). 

:::{figure} comparison_workflow.jpg
:label: fig:compare
Ocetrac workflow for comparing observed and simulated MHWs.
:::

### Evaluating Marine Heatwave Predictions Using Object-Based Metrics

## Discussion
### Implications for Use
### Future Directions

## Conclusion

This document should be rendered with MyST Markdown [mystmd.org](https://mystmd.org),
which is a markdown variant inspired by reStructuredText. This uses the `mystmd`
CLI for scientific writing which can be [downloaded here](https://mystmd.org/guide/quickstart).
When you have installed `mystmd`, run `myst start` in this folder and
follow the link for a live preview, any changes to this file will be
reflected immediately.

## Bibliographies, citations and block quotes

Bibliography files and DOIs are automatically included and picked up by `mystmd`.
These can be added using pandoc-style citations `[@doi:10.1109/MCSE.2007.55]`
which fetches the citation information automatically and creates: [@doi:10.1109/MCSE.2007.55].
Additionally, you can use any key in the BibTeX file using `[@citation-key]`,
as in [@hume48] (which literally is `[@hume48]` in accordance with
the `hume48` cite-key in the associated `mybib.bib` file).
Read more about [citations in the MyST documentation](https://mystmd.org/guide/citations).

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

You can read more about code formatting in the [MyST documentation](https://mystmd.org/guide/code).

As you can see in @fig:stream and @fig:em, this is how you reference auto-numbered figures.
To refer to a sub figure use the syntax `@label [a]` in text or `[@label a]` for a parenhetical citation (i.e. @fig:stream [a] vs [@fig:stream a]).
For even more control, you can simply link to figures using `[Figure %s](#label)`, the `%s` will get filled in with the number, for example [Figure %s](#fig:stream).
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

## Acknowledgments
We gratefully acknowledge the software and methods contributions to Ocetrac from Ryan Abernathey, Julius Busecke, David John Gagne, and Daniel Whitt (listed alphabetically). The work was supported by funding from NSF, NSF NCAR, the Leonardo DiCaprio Foundation Foundation, Microsoft, the Gordon and Betty Moore Foundation, and the University of Washington eScience Institute.