---
# Ensure that this title is the same as the one in `myst.yml`
title: "Echostack: A flexible and scalable open-source software toolbox for echosounder data processing"
abstract: |
  Water column sonar data collected by echosounders are essential for marine ecosystem research, allowing the detection, classification, and quantification of fish and zooplankton from many different ocean observing platforms. However, broad usage of these data has been hindered by the lack of software tools that allow intuitive and transparent data access, processing, and interpretation. We address this gap by developing Echostack, a toolbox of open-source packages leveraging distributed computing and cloud-interfacing libraries in the scientific Python ecosystem. These tools can be used individually or orchestrated together, which we demonstrate in example use cases in a common application scenario for a fisheries acoustic-trawl survey. 
---

## Introduction

Echosounders are high-frequency sonar systems optimized for sensing fish and zooplankton in the ocean. By transmitting sounds and analyzing the returning echoes, fishery and ocean scientists use echosounders to “image” the distribution and infer the abundance of these animals in the water column [@Medwin1998] [@fig:echo_data A]. As a remote sensing tool, echosounders are uniquely suitable for efficient, continuous biological monitoring across time and space, especially when compared to net trawls that are labor-intensive and discrete in nature, or optical imaging techniques that are limited in range due to the strong absorption of light in water [REF].

In recent years, echosounders have been installed widely on many ocean observing platforms (fig:echo_data B), resulting in a deluge of data accumulating at an unprecedented speed from all corners of the ocean. These extensive datasets contain crucial information that can help scientists better understand the marine ecosystems and their response to the changing climate. However, the volume of the data (100s of GBs to TBs [REF]) and the complexity of the problem (e.g., how large- scale ocean processes drive changes in acoustically observed marine biota [REF]) naturally call for a paradigm shift in the data analysis workflow.


:::{figure} fig_echo_data.png
:label: fig:echo_data
:width: 700 px
:align: center
(A) Echograms at two different frequencies. Echo strength variation across frequency is useful for inferring scatterer identity. (B) The variety of ocean observing platforms with echosounders installed.
:::


It is crucial to have software tools that are developed and shared openly, scalable in response to data size and computing platforms, easily interoperable with diverse analysis tools and different types of oceanographic data, and straightforward to reproduce to facilitate iterative modeling, parameterization, and mining of the data. These requirements are challenging to meet by the conventional echosounder data analysis workflows, which rely heavily on manual analysis on mostly closed-source software packages designed to be used with Graphic User Interface (GUI) on a single computer [REF]. Similarly, rather than continuing to store data in manufacturer-specific binary format, making echosounder data widely available in a standardized, machine-readable format will expand the use of these data beyond the original applications in fisheries surveys and specific research cruises.

In this paper, we introduce Echostack, an open-source Python software toolbox aimed at addressing these needs by providing the fisheries acoustics and ocean sciences communities with a suite of open tools for intuitive and transparent access, organization, processing, and visualization of these data. Echostack is a domain-specific adoption of the Pandata stack of Python libraries [REF] that streamlines the composition and execution of common echosounder data workflow, thereby allowing researchers to focus on the key interpretive stage of scientific data analysis. While it is possible for individual researchers to directly use Pandata tools for the same functionalities, we took the modularization approach and created the Echostack libraries, to: 1) enable and facilitate broader code reuse, especially for routine processing steps that are often common across echosounder data workflows [REF], 2) streamline interlace domain-specific operations with more general computational tools, such as the machine learning (ML) libraries, and 3) provide a friendlier on-ramp for researchers who are not already familiar with the scientific Python software ecosystem but possess domain expertise and can quickly benefit from “out-of-the-box” capabilities such native cloud access and distributed computing support. 

Below, we will discuss what the Echostack tools aim to achieve in Design considerations, outline the functionalities of individual libraries in The Echostack packages, demonstrate how Echostack tools can be leveraged in two example use cases, and conclude the discussion by looking forward in the Future work section.
