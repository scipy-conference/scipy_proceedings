---
title: Cyanobacteria detection in small, inland water bodies with CyFi

abstract: |

  Harmful algal blooms (HABs) pose major health risks to human and aquatic life. Remote sensing-based methods exist to automatically detect large, slow-moving HABs in the ocean, but fall short for smaller, more dynamic blooms in critical inland water bodies like lakes, reservoirs, and rivers.
  
  CyFi is an open-source Python package that enables detection of cyanobacteria in inland water bodies using 10-30m Sentinel-2 imagery and a computationally efficient tree-based machine learning model. CyFi enables water quality and public health managers to conduct high level assessments of water bodies of interest and identify regions in which to target monitoring and responsive actions to protect public health. 
  
  CyFi was developed in three phases. A machine learning competition leveraged the diverse skills and creativity of data science experts to surface promising approaches for cyanobacteria detection from remote sensed data. Subsequent user interviews and model iteration resulted in a deployment-ready open-source package designed to meet user workflow needs and decision-making priorities. This process illustrates a replicable pathway for developing powerful machine learning tools in domain-specific areas.

---

# Introduction

Inland water bodies provide a variety of critical services for both human and aquatic life, including drinking water, recreational and economic opportunities, and marine habitats. Harmful algal blooms (HABs) pose a significant risk to these inland bodies, producing toxins that are poisonous to humans and their pets and threatening marine ecosystems by blocking sunlight and oxygen. Such threats require water quality managers to monitor for the presence of HABs and to make urgent decisions around public health warnings and closures when they are detected.

The most common source of HABs in freshwater environments is cyanobacteria, or blue-green algae [@algae]. While there are established methods for using satellite imagery to detect cyanobacteria in larger water bodies like oceans, detection in small inland lakes, reservoirs, and rivers remains a challenge. Manual water sampling is accurate, but is too time and resource intensive to perform continuously at scale. Machine learning models, on the other hand, can generate estimates in seconds. Automatic detection enables water managers to better prioritize limited manual sampling resources and can provide a birds-eye view of water conditions across a region [@doi:10.1007/s10661-020-08631-5]. Machine learning is particularly well-suited to this task because indicators of cyanobacteria are visible in free, routinely collected satellite imagery.

We present CyFi, short for Cyanobacteria Finder, an open-source Python package that uses satellite imagery and machine learning to estimate cyanobacteria levels in inland water bodies [@cyfi_docs]. CyFi helps decision makers protect the public by flagging the highest-risk areas in lakes, reservoirs, and rivers quickly and easily. CyFi represents a significant advancement in environmental monitoring, providing higher-resolution detection capabilities that can pinpoint areas at risk of cyanobacterial contamination. Key strengths of CyFi compared to other tools include:
- Features derived from high-resolution Sentinel-2 satellite data
- A fast and computationally efficient boosted tree machine learning algorithm
- A straightforward command line interface
- A unique training dataset of almost 13,000 cyanobacteria ground measurements across the continental U.S.

This paper presents a detailed examination of the development of CyFi, from its origins in a machine learning competition to an open source package. The methods section explores the setup of the prize competition, the subsequent model experimentation phase that built on winning approaches, and the end user interviews that helped shape CyFi with real-world context. The results section provides insights on the machine learning methods that proved most effective for detecting inland HABs, and details CyFi's underlying methodology, core capabilities, and model performance. Finally, the discussion section reflects the primary ways CyFi can augment human decision making workflows to protect public health and notes areas for future research.

# Motivation

> There are tens of thousands of lakes that matter for recreation and drinking water. Cyanobacterial blooms pose real risks in many of them, and we really don't know when or where they show up, except in the largest lakes.
>
> -- Dr. Rick Stumpf, Oceanographer, NOAA, National Centers for Coastal Ocean Science[^footnote-1]

Harmful algal blooms are a pressing environmental and public health issue, characterized by the rapid and excessive growth of algae in water bodies. These blooms can produce toxins, such as microcystins and anatoxins, that pose severe risks to human health, aquatic ecosystems, and local economies [@doi:10.15406/icpjl.2016.02.00062].

The toxins released by HABs can contaminate drinking water supplies, leading to acute and chronic health problems for communities [@doi:10.1038/s41893-021-00770-y]. Exposure to these toxins through ingestion, skin contact, or inhalation can result in a variety of acute and chronic health issues, including gastrointestinal illnesses, liver damage, neurological effects, and even death in extreme cases [@doi:10.3390/life12030418].

Ecologically, HABS can create hypoxic (low oxygen) conditions in water bodies, resulting in massive fish kills and the disruption of aquatic food webs [@doi:10.1016/j.hal.2016.04.010]. HABs can form dense algal mats that block sunlight, inhibiting the growth of submerged vegetation essential for aquatic habitats. Furthermore, the decomposition of large algal blooms consumes significant amounts of dissolved oxygen, exacerbating oxygen depletion and leading to dead zones where most aquatic life cannot survive.

These ecological impacts can have devastating economic consequences for local industries reliant on water resources, such as fisheries, tourism, and recreation. Beaches and lakeside areas affected by algal blooms often face closures, leading to a loss of revenue. The cost of managing and mitigating the effects of HABs, including water treatment and healthcare expenses, places additional financial burdens on affected communities [@doi:10.1289/ehp.0900645; @doi:10.1007/BF02804908].

Despite the severe consequences of HABs, existing monitoring tools and methods are often insufficient. Traditional approaches, such as manual water sampling and laboratory analysis, are time-consuming, labor-intensive, and provide only localized snapshots of water quality.

Existing satellite-based monitoring tools offer broad coverage but fall short of the spatial resolution needed for small inland water bodies. Most are aimed at monitoring blooms in the ocean, which are larger and slower moving. Many of the leading satellite-based methods for cyanobacteria detection rely on the [Ocean and Land Colour Instrument](https://oceancolor.gsfc.nasa.gov/about/projects/cyan/) on Sentinel-3 [@doi:10.3390/rs13214347]. However, the 300m resolution of Sentinel-3 is too coarse to pick up many inland water bodies and therefore is not able to provide the data needed for effective early warning and rapid response to HAB outbreaks in lakes, reservoirs, and rivers.

:::{figure} resolution_sentinel_2.webp
:label: fig:10m
:width: 500px
An example of a water body at 10m resolution
:::

:::{figure} resolution_sentinel_3.webp
:label: fig:30m
:width: 500px
An example of the @fig:10m image at 30m resolution
:::

Effectively monitoring inland HABs and protecting public health requires developing new innovative tools that capture a higher spatial resolution, can be run quickly and frequently, and are accessible to decision makers. CyFi aims to fill this gap by incorporating higher-resolution satellite imagery, an efficient tree-based model, and a user-friendly command line interface.

# Methods

## Machine learning competition

The machine learning approach in CyFi was originally developed as part of the Tick Tick Bloom: Harmful Algal Detection Challenge, which ran from December 2022 to February 2023 [@ttb_results]. Machine learning competition can harness the power of community-driven innovation and rapidly test a wide variety of possible data sources, model architectures, and features [@doi:10.48550/arXiv.1606.07781; @doi:10.1093/rasti/rzae009; @doi:10.1073/pnas.2011362118]. Tick Tick Bloom was created by DrivenData on behalf of NASA and in collaboration with NOAA, the U.S. Environmental Protection Agency, the U.S. Geological Survey, the U.S. Department of Defense Defense Innovation Unit, Berkeley AI Research, and Microsoft AI for Earth.

In the Tick Tick Bloom challenge, over 1,300 participants competed to detect cyanobacteria blooms in small, inland water bodies using publicly available [satellite](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/650/#satellite-imagery), [climate](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/650/#climate-data), and [elevation](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/650/#elevation-data) data. Models were trained and evaluated using a set of manually collected water samples that had been analyzed for cyanobacteria density. Labels were sourced from 14 data providers across the U.S., shown in @fig:ttb_datasets. The full dataset containing 23,570 in situ cyanobacteria measurements is publicly available through the SeaBASS data archive [@seabass]. Each observation in the dataset is a unique combination of date, latitude, and longitude.

:::{figure} ttb_datasets.webp
:label: fig:ttb_datasets
Labeled samples used in the Tick Tick Bloom competition colored by dataset provider.
:::

The labels were divided into train and set sets, where train labels were provided to participants and test labels were used to evaluate model performance and kept confidential from participants. Lakes in close proximity can experience similar bloom-forming conditions, presenting a risk of leakage. To address this, clustering methods were used to maximize the distance between every train set point and every test set point, decreasing the likelihood that participants could gain insight into any test point density based on the training set. Scikit-learn's DBSCAN [@sklearn; @dbscan] was used to divide all data points into spatial clusters. Each cluster was then randomly assigned to either the train or test dataset, such that no test data point was within 15 kilometers of a train data point.

Participants predicted a severity category for a given sampling point as shown in @tbl:severity_categories. These ranges were informed by EPA and WHO guidelines [@epa_guidelines; @who_guidelines].

```{list-table} Severity categories used in the Tick Tick Bloom competition
:label: tbl:severity_categories
:header-rows: 1
* - Severity level
  - Cyanobacteria density range (cells/mL)
* - 1
  - $\lt$20,000
* - 2
  - 20,000 – $\lt$100,000
* - 3
  - 100,000 – $\lt$1,000,000
* - 4
  - 1,000,000 – $\lt$10,000,000
* - 5
  - $\ge$10,000,000
```

Predictions were evaluated using region-averaged root mean squared error. Averaging across regions incentivized models to perform well across the continental U.S., rather than in certain states that were over-represented in the competition dataset (such as California and North Carolina). Over 900 submissions across 115 teams were made over the course of the competition.

## Carrying forward competition models

Machine learning competitions are excellent for crowd-sourcing top approaches to complex predictive modeling problems. Over a short period of time, a large community of solvers tests a broad feature space including possible data sources, model architectures, and model features. The result is an [open-source github repository](https://github.com/drivendataorg/tick-tick-bloom) with code from the most effective approaches, trained model weights, and write-ups of winning methods.

However, transforming this research code into production code requires significant additional work. There are a few key differences between competition-winning research approaches and deployable code:
1. The competition relies on static data exported and processed once while deployment requires repeated, automatic use with new data.
2. Winning models are relatively unconstrained by the size and cost of their solutions. For ongoing use, efficiency matters.
3. Competition code is validated once with anticipated, clean data and static versions of Python package dependencies. In the real world things break and change; use requires basic robustness, testing and configurability.
4. There is substantial variability in the clarity and organization of competition-winning code. Usable code requires others to be able to understand, maintain, and build on the codebase.

The end goal is regularly generated predictions of cyanobacteria levels surfaced in user-friendly dashboards to water quality managers. To achieve this, the intermediate requirement is a deployment-ready code package. This package is CyFi, a configurable, open-source Python library capable of generating cyanobacteria predictions on new input data.

### Model experimentation

CyFi was developed through an additional model experimentation phase, which combined and iterated on the most useful pieces from competition-winning models, and simplified and restructured code to transform it into a runnable pipeline. Additional model testing helped determine which winning approaches were the most robust, accurate, and generalizable outside of the competition setting. 

The table below summarizes the matrix of experiments that were conducted. Model experimentation informed key decisions around which data sources were used, how satellite imagery was selected and processed, and which target variable was predicted.

:::{figure} model_experimentation.webp
:label: fig:experiments
:width: 100%
Model experimentation summary, with final selections in bold.
:::

During experimentation, the model was trained on roughly 13,000 samples and evaluated on a holdout validation set of roughly 5,000 samples. Performance was evaluated based on a combination of root mean squared error, mean absolute error, mean absolute percentage error, and regional root mean squared error, along with manual review and visualizations of predictions. Standard best practices were used to inform hyperparameters tuning for the final model.

### User interviews

To design a package that optimally addresses on-the-ground user needs, we conducted human-centered design (HCD) interviews with subject matter experts and end users. Interviewees included water quality and public health experts from California, New York, Georgia, Louisiana, Florida, and Michigan. Representatives from these states were selected to understand workflows and priorities, and capture a diversity of geographic locations, number of water bodies in the region, HAB severity, investment in HABs monitoring, and technical sophistication of current approaches. User interviews focused on understanding current water quality decision-making processes, including the data and tools used to support those decisions. Learnings were used to inform the format for surfacing predictions, priorities in model performance, and computational constraints. @tbl:interview_takeaways summarizes the core design decisions for CyFi that were rooted in insights from user interviews.

# Results

## Competition takeaways

The overarching goal of the [Tick Tick Bloom: Harmful Algal Bloom Detection Challenge](#machine-learning-competition) was to identify the most useful data sources, features, and modeling methods for cyanobacteria estimation in small, inland water bodies. There was particular interest around the use of Sentinel-2 data, which is has significantly higher resolution than Sentinel-3 and is more suited to smaller water bodies. However, Sentinel-2 does not contain sensors that can spectrophotometrically measure chlorophyll, which is how most Sentinel-3-based cyanobacteria estimates are derived.

The competition showed that Sentinel-2 contains sufficient information for generating accurate cyanobacteria estimates. Below is a summary of which datasets were used by winners.

:::{table} Data sources used by Tick Tick Bloom competition winners
:label: tbl:winner-data-sources
<table>
    <tr>
        <th></th>
        <th style="background-color: #e0f7e9; padding-left: 16px; padding-right: 16px;"><b>Landsat</b><br><i>Satellite</i></th>
        <th style="background-color: #e0f7e9; padding-left: 16px; padding-right: 16px;"><b>Sentinel 2</b><br><i>Satellite</i></th>
        <th style="background-color: #fef4e0; padding-left: 16px; padding-right: 16px;"><b>HRRR</b><br><i>Climate data</i></th>
        <th style="background-color: #ede0f7; padding-left: 16px; padding-right: 16px;"><b>Copernicus DEM</b><br><i>Elevation</i></th>
        <th style="background-color: #f7f7f7; padding-left: 16px; padding-right: 16px;"><b>Metadata</b><br><i>Time, location</i></th>
    </tr>
    <tr>
        <td>1st Place</td>
        <td></td>
        <td>✔<br><small>Color value statistics</small></td>
        <td>✔<br><small>Temperature</small></td>
        <td></td>
        <td>✔<br><small>Region<br>Location</small></td>
    </tr>
    <tr>
        <td>2nd Place</td>
        <td></td>
        <td>✔<br><small>Color value statistics</small></td>
        <td></td>
        <td>✔</td>
        <td>✔<br><small>Clustered location</small></td>
    </tr>
    <tr>
        <td>3rd Place</td>
        <td>✔<br><small>Color value statistics</small></td>
        <td>✔<br><small>Color value statistics</small></td>
        <td>✔<br><small>Temperature<br>Humidity</small></td>
        <td></td>
        <td>✔<br><small>Longitude</small></td>
    </tr>
</table>
:::

All winners used Level-2 satellite imagery instead of Level-1, likely because it already includes useful atmospheric corrections. Sentinel-2 data is higher resolution than Landsat, and proved to be more useful in modeling.

All winners also used gradient boosted decision tree models such as LightGBM [@lightgbm], XGBoost [@doi:10.48550/arXiv.1603.02754], and CatBoost [@doi:10.48550/arXiv.1810.11363]. First place explored training a CNN model but found the coarse resolution of the satellite imagery overly constraining, particularly when using Landsat imagery.

## Model experimentation takeaways

The [model experimentation](#model-experimentation) phase did not explore alternate model architectures given how clearly the competition surfaced the success of a gradient boosted tree model [@ttb_winners_announcement]. It did however extensively iterate on other parts of the pipeline. Over 30 configurations were tested to identify the optimal setup for training a robust, generalizable model. Below are the core decisions that resulted from model experimentation and retraining.

### Data decisions

:::{table} Data decisions from model experimentation
:label: tbl:data-decisions
<table>
  <tr>
    <th>Decision</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Filter points farther than 550m from a water body</td>
    <td>A small amount of noise in the competition dataset was caused by a combination of human error, GPS device error, or a lack of adequate precision in recorded latitude and longitude. Excluding points that are farther than 500m from a water body helps ensure that the model learns from real-world environmental characteristics of cyanobacteria blooms rather than patterns in human error (see below for additional details).</td>
  </tr>
  <tr>
    <td>Use Sentinel-2 as the sole satellite source</td>
    <td>Landsat data primarily only added value for time period prior to July 2015, when Sentinel-2 data became available. Most applications of CyFi will be forward looking, meaning Sentinel-2 data will be available. The slowest part of the prediction process is downloading satellite data, imposing a significant efficiency cost for incorporating Landsat as a second source. To rely only on Sentinel-2, any samples prior to the launch of Sentinel-2 were removed from the training and evaluation sets. This decreased the train set size from 11,299 to 8,979, and the test set size from 4,938 to 4,035.</td>
  </tr>
  <tr>
    <td>Exclude climate and elevation features</td>
    <td>Climate and elevation features primarily provided value for data points prior to the launch of Sentinel-2 and so are not used in the final CyFi model. Climate and elevation likely do have an impact on how cyanobacteria blooms form, and more sophisticated feature engineering with these data sources may add value in the future. This is a direction for <a href="#future-directions">future research</a>.</td>
  </tr>
  <tr>
    <td>Incorporate land cover</td>
    <td>Including a land cover map, even at a coarse 300m resolution, added to model accuracy. The land cover map likely captures farmland areas with fertilizer runoff that contributes to blooms. A static map from 2020 is used rather than a real-time satellite-derived product, as this reduces the compute time and patterns in land use do not fluctuate daily. Land cover is also an effective balance between reflecting regional characteristics, and avoiding overfitting to the small number data providers in the training set.</td>
  </tr>
</table>
:::

One of the risks in a machine learning competition is overfitting to the test set. Competition models may pick up on patterns specific to the competition data, rather than patterns of environmental cyanobacteria conditions that generalize outside of the competition. The experimentation phase worked to identify and remove competition artifacts that would hamper the generalizability of the model in an open source package. For example, all winning solutions used a "longitude" feature in their models, which captured some underlying differences in sampling procedures by the 14 data providers for the competition. For example, data-providing organizations in California only conduct toxin analysis for suspected blooms, leading to an over-representation of high density samples among competition data points in California. Predicting high severity for all points in California served well in the competition setting, but would not generalize to the real world. As a result, geographic features like longitude, state, and region were not used for the deployed CyFi model.

Competitions can also surface data quality issues. A number of winners pointed out that upon inspection of satellite imagery, some competition data points appeared to be outside of any water body. A small amount of noise in the competition dataset was caused by a combination of human error, GPS device error, or a lack of adequate precision in recorded latitude and longitude. Including these noisy data points in a training pipeline could result in a model that predicts error, rather than one based on environmental conditions.

GPS coordinates are often recorded from a dock or parking lot near a sampling location. In these cases, the bounding box used to generate features would still pick up on relevant water-based characteristics. Filtering out samples that are far from any water body, and keeping points that are on land but *near* water pixels, is the best method to separate relevant data from incorrect coordinates.

The distance between each sample and the nearest water body was calculated using the European Space Agency (ESA) [WorldCover 10m 2021](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) product on Google Earth Engine. Samples farther than 550m from a water body were excluded to help ensure that the relevant water body fell within the portion of the satellite image from which features were calculated. The WorldCover dataset was chosen over Sentinel-2's scene classification band as the water classification appeared to be more reliable based on visual review of samples.

### Satellite feature engineering decisions

:::{table} Satellite feature engineering decisions from model experimentation
:label: tbl:satellite-decisions
<table>
  <tr>
    <th>Decision</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Filter to water area and use a large bounding box</td>
    <td>Land pixels are filtered out because they are usually greener than water areas, and can generate falsely high cyanobacteria estimates. Sentinel-2's scene classification band is not perfectly accurate, but is sufficient for masking non-water pixels. Since ground sampling points are often on land but <em>near</em> water (taken from the shore or the dock), a large bounding box of 2,000m is used to ensure that relevant water pixels are included.</td>
  </tr>
  <tr>
    <td>Use a large look-back window and filter to images with almost no clouds</td>
    <td>When selecting relevant imagery, CyFi uses Sentinel-2's scene classification band to calculate the percent of clouds in the bounding box. Any imagery that has greater than 5% clouds is not used. CyFi combines a relatively large look-back window of 30 days before the sample with this strict cloud threshold to increase the chances of finding a cloud-free image.</td>
  </tr>
  <tr>
    <td>Use only one image per sample point</td>
    <td>Some winning solutions averaged predictions over multiple satellite images within a specified range. We find that this favors static blooms. We use only the most recent cloud-free image to better detect short-lived blooms.</td>
  </tr>
</table>
:::

### Target variable decisions

:::{table} Target variable decisions from model experimentation
:label: tbl:target-variable-decisions
<table>
  <tr>
    <th>Decision</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Estimate density instead of severity</td>
    <td>We learned during user interviews that states use different thresholds for action, so predicting density instead of severity categories supports a broader range of use cases. The winning competition models were trained to predict severity. During experimentation, we validated that there was sufficient signal to predict at the higher granularity of exact density.</td>
  </tr>
  <tr>
    <td>Train the model to predict log density</td>
    <td>We find transforming density into a log scale for model training and prediction yields better accuracy, as the underlying data is highly skewed. About 75% of samples have a density less than 400,000 cell/mL, but there are extreme outliers with densities as high as 800,000,000 cells/mL. A log scale helps the model learn that incorrectly estimating a density of 100,000 when the true density is 0 is much more important than incorrectly estimating a density of 1,100,000 when the true density is 1,000,000. The estimate a user sees has been converted back into (non-log) density.</td>
  </tr>
</table>
:::

## User interview takeaways

Technical experimentation alone is insufficient to build a tool that effectively addresses a real-world problem. Understanding user needs and day-to-day processes helps enable integration with existing workflows and increases the likelihood of adoption. The table below synthesizes key insights gleaned from [user interviews](#user-interviews), and outlines how each insight supported the development of a user-friendly package.

```{list-table} CyFi design decisions rooted in HCD interviews
:label: tbl:interview_takeaways
:header-rows: 1
* - Interview insight
  - CyFi design decision
* - States tend to have designated sampling locations or locations of reported blooms. Coverage of the full area of a water body is nice but not necessary.
  - CyFi will expect sampling points as input rather than polygons, and the output will be point-estimates rather than a gridded heatmap.
* - Thresholds are not universal and actions vary by state.
  - Prediction will be a density value rather than severity category.
* - While blooms in small water bodies can change quickly, the maximum cyanobacteria estimation cadence is daily.
  - A sampling point will be a unique combination of date, latitude, and longitude. Additional time granularity is not needed.
* - Many states include a visual review of imagery (satellite or submitted photo) as part of the decision-making process.
  - CyFi will include a way to see the underlying satellite data for a given prediction point, to help users build confidence and intuition around the CyFi model.
* - States have their own tools for managing water quality data (e.g. ground samples and lab results).
  - CyFi will output a simple CSV file that includes identifying columns for joining with external data.
```

## CyFi

The culmination of the machine learning competition, subsequent model experimentation, and user interviews is CyFi. CyFi, short for Cyanobacteria Finder, is an open-source Python package that uses satellite imagery and machine learning to detect cyanobacteria levels, one type of HAB. CyFi can help decision makers protect the public by flagging the highest-risk areas in lakes, reservoirs, and rivers quickly and easily. CyFi incorporates open-source best practices, including tests and continuous integration, and is ready for use in state-level dashboards and decision-making processes.

### Data sources

CyFi relies on two data sources as input:
1. Sentinel-2 satellite imagery
2. Land cover classifications

**Sentinel-2** is a wide-swath, high-resolution, multi-spectral imaging mission. The Sentinel-2 Multispectral Instrument (MSI) samples [13 spectral bands](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#available-bands-and-data): four bands at 10 meters, six bands at 20 meters, and three bands at 60 meters spatial resolution. The mission provides global coverage of the Earth's land surface every 5 days. Sentinel-2 data is accessed through Microsoft's [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).

CyFi uses high-resolution Sentinel-2 satellite imagery (10-30m) to focus on smaller water bodies with rapidly changing blooms. This is a significant improvement in resolution over Sentinel-3, which is used by most existing satellite-based cyanobacteria detection tools and has a resolution of 300-500m.

The Climate Research Data Package **Land Cover Gridded Map** (2020) categorizes land surface into 22 classes, which have been defined using the United Nations Food and Agriculture Organization's Land Cover Classification System (LCCS). The map is based on data from the Medium Resolution Imaging Spectrometer (MERIS) sensor on board the polar-orbiting Envisat-1 environmental research satellite by the European Space Agency. CyFi accesses the data using the CCI-LC database hosted by the ESA Climate Change Initiative's [Land Cover project](https://www.esa-landcover-cci.org/?q=node/164).

### Feature processing

Each observation (or "sampling point") is a unique combination of date, latitude, and longitude. Feature generation for each observation is as follows:

  1. Identify relevant Sentinel-2 tiles based on a bounding box of 2,000m around the sampling point and a time range of 30 days prior to (and including) the sampling date.
  2. Select the most recent image that has a bounding box containing less than 5% cloud pixels. If none of the images meet this criteria, no prediction is made for that sampling point.
  3. Filter the pixels in the bounding box to the water area using the scene classification (SCL) band.
  4. Generate band summary statistics (e.g., mean, 95th percentile) and ratios (e.g, green-blue ratio, NDVI) using 15 different Sentinel-2 bands. The full list of satellite image features is here: https://github.com/drivendataorg/cyfi/blob/ad239c8569d6ef48b8769b3bebe98029ea6ecb6f/cyfi/config.py#L93-L121
  5. Look up static land cover map data for the sampling point, and combine land cover information with satellite features.
  
:::{figure} feature_creation.webp
:label: fig:feature_creation
:width: 600px
Mock up of satellite data selection and processing. The dot represents the sample point; the square represents the 2,000m bounding box around the sample point. The multiple squares outlined in black represents the multiple satellite image contenders within the lookback period. The orange outlined square indicates the selected, most-recent satellite image. The blue shaded area indicates the water pixels in the bounding box from which features are calculated.

Note that not all features are represented in the columns.
:::

### Model

Cyanobacteria estimates are generated by a gradient-boosted decision tree algorithm built with LightGBM [@lightgbm]. The hyperparameters can be found here: https://github.com/drivendataorg/cyfi/blob/ad239c8569d6ef48b8769b3bebe98029ea6ecb6f/cyfi/config.py#L188-L196

The model was trained and evaluated using "in situ" labels collected manually by many organizations across the U.S. The train and test split was maintained from the [competition data](#machine-learning-competition), with stricter filtering on [distance to a water body](#data-decisions).

:::{figure} train_test.webp
:label: fig:train_test
Location and distribution of training and evaluation data for CyFi.
:::

### Performance

CyFi was evaluated using 2,880 ground measurements from 12 data providers spanning the time range August 2015 to December 2021. Given that CyFi relies on Sentinel-2 imagery, the earliest date in the evaluation set aligns with the launch of Sentinel-2 (mid 2015). Of these points, 1,153 were low severity, 504 were moderate severity, and 1,223 were high severity according to ground measurement data. Some states only conduct toxin analysis when blooms are suspected, which may account for the large number of high-severity observations in the evaluation set.

:::{figure} eval_data_providers.webp
:label: fig:eval_data
:width: 700px
Counts of cyanobacteria measurements by data provider in the evaluation set.
:::

We use the following categories based on @who_guidelines for evaluation:
- **Non-bloom:** Cyanobacteria density is less than 20,000 cells/mL
- **Bloom:** Cyanobacteria density is at least 20,000 cells/mL
- **Severe bloom:** Cyanobacteria density is greater than 100,000 cells/mL. Severe blooms are a subset of blooms

On this evaluation dataset, CyFi detects 48% of **non-blooms** with 63% accuracy. Being able to detect places *not* likely to contain blooms enables ground sampling staff to de-prioritize low-risk sampled locations and better allocate limited resources.

CyFi detects 81% of **blooms** with 70% accuracy. Based on user interviews, moderate blooms are important to identify because they should be prioritized for sampling. There may be negative public health impacts and more precise toxin analysis is needed.

Lastly, CyFi detect 53% of **severe blooms** with 71% accuracy. These locations pose the highest risk of severe negative health impacts, and are critical to flag for decision makers to prioritize for public health action (e.g., issuing advisories). In the most severe cases, additional visual inspection of the satellite imagery used by CyFi may be sufficient to issue an advisory without additional sampling. CyFi enables this step with its [CyFi Explorer](#cyfi-explorer) functionality.

Model accuracy can vary based on bloom severity as well as location and other attributes of the sampling point, so the performance metrics above will vary based on the distribution in the evaluation set.

### Benchmark comparison

An apples-to-apples comparison with one of the leading tools for cyanobacteria estimation from satellite imagery is provided as a more objective benchmark of performance. The Cyanobacteria Index proxies chlorophyll absorption with a spectral shape algorithm using MERIS bands 7, 8 and 9 and was developed through the [Cyanobacteria Assessment Network (CyAN)](https://oceancolor.gsfc.nasa.gov/about/projects/cyan/) [@doi:10.1016/j.hal.2017.06.001; @doi:10.4319/lo.2010.55.5.2025; @doi:10.1038/s41598-019-54453-y; @doi:10.1080/01431160802007640].

Due to lower resolution of satellite imagery as well as missing data, CyAN's Cyanobacteria Index is only able to provide estimates for 30% of points in the evaluation set described in @fig:eval_data (756 points). A major advantage of CyFi is coverage of small water bodies. Over half of the points in the evaluation set were identified as "land" by CyAN due to the coarse resolution of Sentinel-3 imagery. An additional 18% of points had "no data" likely due to clouds or bad imagery.

Among the portion of the evaluation set captured by CyAN, CyFi detects blooms with slightly higher accuracy. Using a cutoff of 10,000 cells/mL per @doi:10.1016/j.scitotenv.2021.145462, we find CyFi has a presence/absence accuracy of 72% compared to 66% for CyAN. The improved accuracy is largely due to a higher correct classification of true positive cases (blooms).

:::{figure} cyan_comparison.webp
:label: fig:cyan_comparison
:width: 350px
A comparison of CyFi and CyAN model accuracy on 756 ground sampled data points from across the U.S. A true positive (bloom presence) is where cyanobacteria density > 10,000 cells/mL.
:::

### Using CyFi

Comprehensive instructions for using CyFi can be found in the [CyFi docs](https://cyfi.drivendata.org/). The below provides an overview of some of CyFi's key functionality.

CyFi is designed to be simple to use. To get started, users can install CyFi with pip.

```bash
$ pip install cyfi
```

Cyanobacteria predictions can then be generated with a single command. The only information needed to generate a prediction is a location (latitude and longitude) and date.

```bash
$ cyfi predict-point --lat 35.6 --lon -78.7 --date 2023-09-25

SUCCESS  | Estimate generated:
date                    2023-09-25
latitude                      35.6
longitude                    -78.7
density_cells_per_ml        22,836
severity                  moderate

```

For each sampling point, CyFi downloads recent cloud-free Sentinel-2 data from Microsoft's Planetary Computer, calculates a set of summary statistics using the spectral bands for the portion of the image around the sampling point, and passes those features to a LightGBM model which produces an estimated cyanobacteria density.

CyFi also makes it easy to generate cyanobacteria estimates for many points at once. Users can input a CSV with columns for date, latitude, and longitude.

:::{table} Example input csv (`samples.csv`) containing the sampling points where cyanobacteria estimates are needed
:label: tbl:cyfi_preds
<table>
  <thead>
        <tr>
            <th>latitude</th>
            <th>longitude</th>
            <th>date</th>
        </tr>
  </thead>
  <tbody>
        <tr>
            <td>41.424144</td><td>-73.206937</td><td>2023-06-22</td>
        </tr>
        <tr>
            <td>36.045</td><td>-79.0919415</td><td>2023-07-01</td>
        </tr>
        <tr>
            <td>35.884524</td><td>-78.953997</td><td>2023-08-04</td>
        </tr>
  </tbody>
</table>
:::


A CSV input can also be processed with a single command.

```bash
$ cyfi predict samples.csv

SUCCESS  | Loaded 3 sample points (unique combinations of date, latitude, and longitude) for prediction
SUCCESS  | Downloaded satellite imagery
SUCCESS  | Cyanobacteria estimates for 4 sample points saved to preds.csv

```

Cyanobacteria estimates are saved out as a CSV that can be plugged into any existing decision-making process. For each point, the model provides an estimated density in cells per mL for detailed analysis. Densities are also discretized into severity levels based on World Health Organization (WHO) guidelines [@who_guidelines].

:::{table} CyFi outputted csv (`preds.csv`) containing predictions
:label: tbl:cyfi_preds
  <table>
    <thead>
      <tr>
          <th>sample_id</th>
          <th>date</th>
          <th>latitude</th>
          <th>longitude</th>
          <th>density_cells_per_ml</th>
          <th>severity</th>
      </tr>
    </thead>
    <tbody>
      <tr>
          <td>7ff4b4a56965d80f6aa501cc25aa1883</td>
          <td>2023-06-22</td>
          <td>41.424144</td>
          <td>-73.206937</td>
          <td>34,173</td>
          <td>moderate</td>
      </tr>
      <tr>
          <td>882b9804a3e28d8805f98432a1a9d9af</td>
          <td>2023-07-01</td>
          <td>36.045</td>
          <td>-79.0919415</td>
          <td>7,701</td>
          <td>low</td>
      </tr>
      <tr>
          <td>10468e709dcb6133d19a230419efbb24</td>
          <td>2023-08-04</td>
          <td>35.884524</td>
          <td>-78.953997</td>
          <td>4,053</td>
          <td>low</td>
      </tr>
    </tbody>
  </table>
:::

:::{table} WHO Recreational Guidance/Action Levels for Cyanobacteria [@who_guidelines]
:label: tbl:who_guidelines
<table>
  <thead>
    <tr>
      <th>Relative Probability of Acute Health Effects</th>
      <th>Cyanobacteria (cells/mL)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Low</td>
      <td>&lt; 20,000</td>
    </tr>
    <tr>
      <td>Moderate</td>
      <td>20,000–100,000</td>
    </tr>
    <tr>
      <td>High</td>
      <td>&gt;100,000–10,000,000</td>
    </tr>
    <tr>
      <td>Very High</td>
      <td>&gt; 10,000,000</td>
    </tr>
  </tbody>
</table>
:::


### CyFi Explorer

CyFi also comes with a visualization tool called [CyFi Explorer](https://cyfi.drivendata.org/explorer/). CyFi Explorer surfaces the corresponding Sentinel-2 imagery for each cyanobacteria estimate. The explorer runs a Gradio app locally on the user's machine and is intended to enable visual inspection of where the model is performing well, as well as edge cases or failure modes. It is not intended to replace more robust data analytics tools and decision-making workflows.

:::{figure} cyfi_explorer.webp
:label: fig:cyfi_explorer
:width: 700px
Screenshot of CyFi Explorer, a visualization tool that surfaces the underlying satellite imagery used to generate the cyanobacteria estimate.
:::

# Discussion

CyFi's progression from a machine learning competition that surfaced promising approaches, through subsequent user interviews and model iteration, to a deployment-ready open source package illustrates a replicable pathway for developing powerful machine learning tools in domain-specific areas.

We find that CyFi performs at least as well as a leading Sentinel-3 based tool, but has 10 times the coverage of water bodies across the U.S. due to the higher resolution of Sentinel-2 data. This dramatically expands the applicability of remote sensing-based estimates as a tool for management of HABs.

### Implications for use

CyFi works best as an enhancement to existing decision-making processes through its ability to surface high and low priority areas. At its current accuracy level, we believe CyFi should be used to inform human workflows rather than triggering automatic actions.

Based on discussions with end users, a few common use cases for CyFi are listed below. Overall, CyFi supports more widespread and timely public health actions, better allocation of ground sampling resources, and more informed impairment and regulatory monitoring.

1) Flag high severity blooms for public health action

High concentrations of cyanobacteria often merit public health interventions. Having daily estimates at designated sampling points can quickly and easily surface worrisome conditions. States have the flexibility to design their own processes for how to use this information. For example, some states may choose to prioritize these locations for ground sampling where advisory levels are dependent upon toxin analysis results. Other states may choose to take action such as issuing a press release based on visual review of imagery alone.

2) Identify locations where ground sampling can be deprioritized

Identifying water bodies that are *not* experiencing blooms can be just as helpful as identifying water bodies that are. Ground sampling is time and labor intensive, and CyFi enables water quality managers to deprioritize sampling in the areas least likely to contain blooms.

3) Confirm publicly reported blooms with more data

Multiple states rely on visual inspection of a submitted photo to confirm a bloom. CyFi can both generate a cyanobacteria density estimate and show the most recent, cloud free 10m satellite imagery for that location.

4) Provide a birds-eye view of lake conditions across the state

Many states track [impaired and threatened waters](https://www.epa.gov/tmdl) in accordance with the Clean Water Act and develop total maximum daily loads (TMDLs), which specify the maximum amount of pollutant allowed to enter a water body. Routine predictions from CyFi can help monitor the progression in water bodies where cyanobacteria is a primary concern.

### Future directions

While CyFi represents a significant step forward in detecting cyanobacteria from satellite imagery, challenges remain. CyFi is the least reliable for the following cases:
- In very narrow or small waterways
- When there are clouds obscuring the area around a sampling point
- Where multiple water bodies are nearby in the same satellite image

Model performance could be improved by retraining with additional ground measurements for true negative cases, adding water body segmentation to exclude pixels from non-contiguous water bodies, and adding cloud segmentation to remove cloud pixels from feature calculations. Additionally, incorporating more sophisticated time-series climate features may enhance model accuracy. To support users who desire comprehensive estimates across an entire water body, a pre-processing step could be added that accepts a water body polygon as input and transforms this into a grid of sample points.

As decision-makers begin experimenting with CyFi, we recommend calculating historical estimates and comparing these against prior ground measurements to get a baseline accuracy for CyFi's performance. Using CyFi Explorer to review predictions can provide further insight into water bodies that may be particularly challenging for CyFi.

# Conclusion

CyFi is a powerful tool for identifying high and low levels of cyanobacteria, and enables humans to make more timely and targeted decisions when issuing public health guidances around current cyanobacteria levels. Areas with low-density cyanobacteria counts can be excluded from ground sampling to better prioritize limited resources, while areas with high-density cyanobacteria counts can be prioritized for public health action. The development of CyFi illustrates the utility of machine learning competitions as a first step toward open source tools. CyFi's primary use cases show how machine learning can be incorporated into human workflows to enable more efficient and more informed decision making.

[^footnote-1]: @ttb_results
