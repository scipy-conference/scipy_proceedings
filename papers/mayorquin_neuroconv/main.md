---
# Ensure that this title is the same as the one in `myst.yml`
title: NeuroConv. Streamlining Neurophysiology Data Conversion to the NWB Standard
keywords: Neurodata Without Borders, NWB, Neurophysiology, Data standardization, Data conversion, DANDI, Python, Scientific software, Large-scale data
abstract: |
  Converting diverse neurophysiology data to the standardized Neurodata Without Borders (NWB) format remains a significant barrier to data sharing and reuse. We present NeuroConv, a software that enables converting 44 distinct data formats while handling high-volume data as well as extracting meaningful metadata. The library has enabled the standardization of over 300 datasets totaling more than 300 TB in the DANDI archive and has been deployed to create automated conversion pipelines for 40 laboratories with unique experimental paradigms. By reducing technical barriers to NWB adoption, our tools accelerate progress toward reproducible neuroscience research through standardized data sharing.
---

## Introduction

Modern neurophysiology research generates increasingly complex, multimodal, and large-scale datasets that demand robust standards for organization, sharing, and archival. The [Neurodata Without Borders (NWB)](https://nwb.org/)[@nwb_2015; @nwb_2022] format has emerged as a comprehensive solution, storing contextual metadata alongside primary data objects to ensure long-term interpretability. A key strength of NWB is its ability to unify diverse experimental modalities—such as simultaneous behavioral and electrophysiology recordings that usually reside in scattered files and formats. By consolidating all relevant information about an experimental session, NWB enables robust reanalysis and improves scientific reproducibility.

Despite these advantages, several significant challenges impede widespread NWB adoption. First, researchers must develop at least a basic understanding of NWB's data organization principles which can be a daunting prospect given the format's comprehensive scope. This learning curve presents a substantial barrier to integrating NWB standards into daily laboratory practices. Additionally, utilizing NWB's data Application Programming Interfaces (APIs), such as [pyNWB](https://pynwb.readthedocs.io/en/stable/) and (MatNWB)[https://nwb.org/matnwb/], requires programming expertise that may exceed the capabilities of many researchers. This technical barrier often restricts NWB adoption to those with coding proficiency, dedicated technical staff, or places an outsized burden on junior lab members who possess the necessary skills.

Labs frequently face two distinct conversion challenges: processing existing backlogged data and establishing automated pipelines for future data collection. The latter is particularly crucial for minimizing duplicated work, as labs aim to automatically convert newly generated data into NWB format. However, developing robust conversion pipelines presents significant challenges due to the diversity of neurophysiology data formats. A single lab may employ multiple modalities such as voltage recording, optical imaging, optogenetics, and behavioral tracking, each with its own software-dependent formats lacking standardization.

## Background and Motivation

The challenges outlined above stem from fundamental characteristics of the neurophysiology data landscape. The field encompasses diverse experimental modalities, including microscopy for optical imaging, extracellular and intracellular electrophysiology for neural activity recording, and a wide array of behavioral tracking approaches, each with its own specialized requirements and methodological considerations [@nwb_2015; @MEF3_format_2016; @nwb_2022]. Within each modality, researchers rely on dozens of acquisition systems, each typically recording data in its own proprietary format. These formats prioritize different aspects of data handling: some optimize for write speed during acquisition, others for storage efficiency or cross-platform compatibility. This layered diversity across both modalities and acquisition systems creates a complex landscape where formats vary widely in their efficiency, support longevity, metadata richness, and cross-platform compatibility.

Converting existing datasets to NWB presents several interconnected challenges that compound the adoption barriers:

:::{table} Key challenges in converting neurophysiology datasets to NWB format, illustrating the multifaceted barriers that laboratories face when adopting data standardization practices.
:label: tbl:nwb-challenges

<table>
<tr><th>Challenge</th><th>Description</th></tr>
<tr><td>Format diversity</td><td>Source data formats span both proprietary and open standards, with many formats existing in multiple versions and internal variations</td></tr>
<tr><td>Metadata complexity</td><td>Metadata requirements vary substantially across experimental paradigms, with critical information often stored inconsistently or incompletely</td></tr>
<tr><td>Scale challenges</td><td>Dataset sizes frequently reach hundreds of gigabytes to terabytes, requiring specialized handling for memory-efficient processing</td></tr>
<tr><td>Expertise requirements</td><td>Following NWB best practices requires considerable knowledge of both the source formats and the NWB standard itself</td></tr>
</table>
:::

While NWB has emerged as a unifying standard that addresses many common pitfalls of proprietary formats, the conversion process itself remains a significant bottleneck. This process requires deep knowledge of experimental design, source data formats, and the NWB standard—expertise that is rarely concentrated in a single individual or even within a single laboratory.

## NeuroConv Architecture and Design

To address these multifaceted challenges [NeuroConv](https://neuroconv.readthedocs.io/en/stable/index.html), a library that automates the ingestion and conversion of neurophysiology data from diverse formats into NWB. The development of NeuroConv required solving the following fundamental challenges:

```{list-table} Key technical challenges addressed by NeuroConv's architecture, highlighting the multifaceted requirements for automated neurophysiology data conversion.
:label: tbl:neuroconv-challenges
:header-rows: 1
* - Challenge
  - Solution Approach
* - Handling format and metadata diversity
  - Abstracting the complexity of 44+ distinct data formats while preserving format-specific metadata and ensuring NWB compliance
* - Managing high-volume data efficiently  
  - Processing datasets that exceed available RAM through streaming and chunked operations
* - Supporting complex experimental setups
  - Accommodating multi-stream conversions where different modalities are recorded simultaneously in different formats
```

The following sections detail how NeuroConv's architecture addresses each of these challenges through a modular, extensible design that maintains both flexibility and ease of use.


### Handling Diverse Data Formats

The challenge of format diversity in neurophysiology extends beyond their sheer number. Many formats, such as Neuralynx, exist in multiple versions, while others, like TIFF, exhibit significant internal variability in how labs use them. NeuroConv addresses this complexity through a modular architecture built around DataInterface classes. Each supported format has a dedicated DataInterface that handles data and metadata extraction, with specialized implementations like SpikeGLXRecordingInterface for raw voltage recordings, PhySortingInterface for spike-sorted data, and DeepLabCutInterface for behavioral tracking data. The critical contribution is a common interface that abstracts the format internal details and allows users to build conversions in a consistent manner, regardless of source format.

The central object of NeuroConv is the DataInterface, which serves as the medium between the source data format and the NWB file. It provides a unified API for extracting metadata, adding data to an NWB file, and running the conversion process. Each DataInterface is designed to handle a specific source format, encapsulating the logic required to read and convert that format's data while adhering to NWB best practices.

The basic piece of code in NeuroConv looks like this:

```python

from neuroconv import DataInterface

# Initialize the DataInterface for a specific format
interface = DataInterface(file_path="path/to/data/file")
# Extract the metadata from the source format
metadata = interface.get_metadata()
# 
# Code to modify the metadata as needed
# 
# Add the data to an NWB File and write it to disk 
interface.run_conversion(nwbfile_path="path/to/nwbfile.nwb", metadata=metadata)
```

This core pattern abstracts away the complexities of each source format while providing a consistent interface for users. The user initializes a DataInterface for a specific source format, extracts metadata, modifies it as needed, and then runs the conversion process to create an NWB file.


Currently supporting 44 distinct input formats (Table 1), each DataInterface is comprehensively documented and demonstrated in the [Conversion Gallery](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/index.html), where users can find complete examples requiring only ~5 lines of code to perform full data conversion. Throughout the conversion process, NeuroConv enforces NWB Best Practices for metadata and data organization while at the same time optimizes data storage for both archival purposes and cloud computing requirements.

| **Category** | **Subcategory** | **Format** |
|--------------|-----------------|------------|
| **Extracellular Electrophysiology** | Recording | AlphaOmega |
| | Recording | Axona |
| | Recording | Biocam |
| | Recording | Blackrock |
| | Recording | European Data Format (EDF) |
| | Recording | Intan |
| | Recording | MaxOne |
| | Recording | MCSRaw |
| | Recording | MEArec |
| | Recording | Neuralynx |
| | Recording | NeuroScope |
| | Recording | OpenEphys |
| | Recording | Plexon |
| | Recording | Plexon2 |
| | Recording | Spike2 |
| | Recording | Spikegadgets |
| | Recording | SpikeGLX |
| | Recording | Tucker-Davis Technologies (TDT) |
| | Recording | White Matter |
| | Sorting | Blackrock |
| | Sorting | Cell Explorer |
| | Sorting | KiloSort |
| | Sorting | Neuralynx |
| | Sorting | NeuroScope |
| | Sorting | Phy |
| | Sorting | Plexon |
| **Intracellular Electrophysiology** | | ABF |
| **Optical Physiology** | Imaging | Bruker |
| | Imaging | HDF5 |
| | Imaging | Micro-Manager |
| | Imaging | Miniscope |
| | Imaging | Scanbox |
| | Imaging | ScanImage |
| | Imaging | Thor |
| | Imaging | Tiff |
| | Segmentation | Caiman |
| | Segmentation | CNMFE |
| | Segmentation | EXTRACT |
| | Segmentation | Suite2P |
| | Fiber Photometry | TDT Fiber Photometry |
| **Behavior** | Motion Tracking | DeepLabCut |
| | Motion Tracking | FicTrac |
| | Motion Tracking | LightningPose |
| | Motion Tracking | Neuralynx NVT |
| | Motion Tracking | SLEAP |
| | Audio/Video | Videos |
| | Operant Conditioning | MedPC |
| **General Data** | Image | Image (png, jpeg, tiff, etc) |
| | Text/Tabular | CSV |
| | Text/Tabular | Excel |
| | Text/Tabular | Text |

*Table 1: Comprehensive list of data formats supported by NeuroConv, organized by experimental modality and data type.*

Each format example in our documentation includes basic code snippets that demonstrate how to use the DataInterface for that specific format, including metadata extraction, modification, and the conversion process. This modular approach allows users to easily adapt examples to their specific needs while providing a consistent interface across different data formats. For example, converting amplifier data acquired with Intan requires only these steps:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from neuroconv.datainterfaces import IntanRecordingInterface

file_path = f"{ECEPHY_DATA_PATH}/intan/intan_rhd_test_1.rhd" # This can also be .rhs
interface = IntanRecordingInterface(file_path=file_path, verbose=False)

# Extract what metadata we can from the source files
metadata = interface.get_metadata()
# session_start_time is required for conversion. If it cannot be inferred
# automatically from the source files you must supply one.
session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
metadata["NWBFile"].update(session_start_time=session_start_time)

nwbfile_path = f"{path_to_save_nwbfile}"  # This should be something like: "./saved_file.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
```


As NeuroConv's format support has expanded to 44+ formats, dependency management has become increasingly complex. Each format often requires specialized libraries with potentially conflicting version requirements, creating dependency resolution challenges that can make installation difficult or impossible. For example, different electrophysiology formats may depend on incompatible versions of numerical libraries, while imaging formats might require conflicting versions of image processing packages. Additionally, installing all dependencies simultaneously would create an unnecessary and inefficient environment with hundreds of packages, many of which users never need.

To address these challenges, we rely on [installation extras](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras) to manage installation complexity. Users can specify only the formats they need during installation:


```python
pip install "neuroconv[spikeglx,phy,deeplabcut]"
```

This approach aggregates only the required dependencies for selected formats, avoiding dependency conflicts while ensuring that the installation remains lightweight and manageable for end users. Users working with a specific subset of formats can maintain clean, minimal environments without the overhead of unused dependencies.

### Handling Multi Stream Conversions

Neurophysiology experiments typically involve multiple simultaneous data streams from different modalities, such as raw electrophysiology recordings, spike-sorted data, and behavioral video. Each stream may be recorded in a different format, leading to complex conversion requirements. NeuroConv's architecture supports multi-stream conversions through the aggregation of DataInterfaces within a Converter framework.

[Diagram of converter worfklow should be here]

The converter pattern enables combining multiple DataInterface instances into a single conversion workflow. This allows users to convert all relevant data streams from an experiment into a single NWB file, ensuring that all data is properly aligned and associated with the correct metadata. The Converter class handles the orchestration of multiple DataInterfaces, managing the order of operations and resolving any conflicts in metadata or data organization.

```python
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import SpikeGLXRecordingInterface, PhySortingInterface, DeepLabCutInterface

# Initialize the DataInterfaces for each data stream
recording_interface = SpikeGLXRecordingInterface(file_path="path/to/recording/file")
sorting_interface = PhySortingInterface(file_path="path/to/sorting/file")
behavior_interface = DeepLabCutInterface(file_path="path/to/behavior/file")

# Create the ConverterPipe with the DataInterfaces
data_interfaces = [recording_interface, sorting_interface, behavior_interface]
converter = ConverterPipe(
    data_interfaces=data_interfaces
)

metadata = converter.get_metadata()
# Modify metadata as needed

# Run the conversion to create an NWB file
converter.run_conversion(nwbfile_path="path/to/nwbfile.nwb", metadata=metadata)
```
Note that the same pattern used for single interfaces extends seamlessly to multi-stream conversions. The user initializes multiple DataInterfaces for each data stream, aggregates them into a ConverterPipe, extracts metadata, modifies it as needed, and then runs the conversion process to create an NWB file. This modular approach allows NeuroConv to handle complex experimental setups with multiple data streams while maintaining a consistent interface for users.

### Handling High-Volume Data

Modern acquisition systems, such as multi-probe Neuropixel recordings or whole-brain optical imaging, generate massive volumes of data that continue to grow year over year [@neuropixels_2018; @optical_physiology_methods_2022]. These volumes of data pose a variety of challenges both for conversion and for long-term storage. Moreover, as cloud computing emerges as a solution [@amazon_scientific_workflows_2009; @ome_ngff_2021] for managing and storing large datasets, ensuring efficient accessibility for the scientific community becomes a critical consideration.

A critical feature is the ability to process datasets larger than available RAM. NeuroConv inherits from the work performed by the NWB core group with [iterative writing](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/plot_iterative_write.html#sphx-glr-tutorials-advanced-io-plot-iterative-write-py) to stream data in manageable chunks, with configurable chunk sizes based on available resources. We have extended this approach to support chunked reading from SpikeInterface, enabling buffered processing of numerous extracellular electrophysiology formats, such as SpikeGLX, Neuralynx, and Plexon. We have also implemented an iterative writing approach for roiextractors that allows buffered writing of large imaging datasets, such as those generated by whole-brain calcium imaging. Furthermore, we have implemented chunked solutions for other data-intensive formats such as video. This approach enables processing of arbitrarily large files and has been successfully tested on 100+ GB files using computers with only 8 GB of RAM.

For storage optimization, NeuroConv leverages HDF5 and Zarr's support for chunked, compressed datasets. The current supported backends in NWB are HDF5 and Zarr. Compression algorithms represent a trade-off between storage space and access speed [@alessio_compression_2023]. NeuroConv exposes an easy-to-use [API](https://neuroconv.readthedocs.io/en/stable/user_guide/backend_configuration.html) for configuring chunking and compression settings at the dataset level, allowing for quick experimentation while providing sensible defaults that work for most users.

Determining optimal chunk parameters presents complex tradeoff [@doi:10.1002/essoar.10511054.2][@nguyen2023impact] Large chunks minimize the number of read operations but may require decompressing unnecessary data. Small chunks provide more precise access but increase overhead, particularly for cloud storage where each chunk requires a separate range request. In general, appropriate chunking requires knowledge of the most common access patterns of the data. As Neurosphysilogy has to some degree common analysis and visualizations it is possible to implement heuristics for chunk size for the most common data types (such as voltage recordings and imaging data).


### Cloud Deployment

NeuroConv supports both local installation (Linux, Windows, or macOS) and [cloud deployment](https://neuroconv.readthedocs.io/en/stable/user_guide/aws_demo.html) through a maintained [Docker image](https://neuroconv.readthedocs.io/en/stable/user_guide/docker_demo.html) containing all dependencies. We've developed a YAML-based specification language for defining conversion pipelines, validated through JSON schema. This specification can fully describe multi-subject, multi-session conversions with custom metadata at each level, enabling automated conversion through containerized NeuroConv deployments. 

## Testing and Quality Assurance

Ensuring reliable conversion across diverse neurophysiology data formats requires a robust testing infrastructure. Our testing framework rests on two pillars: an automated continuous integration (CI) pipeline built using [GitHub Actions](https://github.com/features/actions)  and a comprehensive test data library.

Our continuous integration pipeline ensures code quality and maintains compatibility across operating systems through standard software engineering practices. Failed tests block pull request merging, maintaining code quality standards while facilitating rapid development. The pipeline runs on every pull request and includes several key components: unit tests covering the core functionality of the library; integration tests ensuring the library works as expected with real data; cross-platform testing to verify compatibility across all supported operating systems; documentation builds to ensure up-to-date documentation; [doctest](https://docs.python.org/es/3.13/library/doctest.html) functionality to verify that our conversion gallery works with the current version of the code, preventing documentation drift; code style checks ensuring consistency and adherence to best practices; and test coverage analysis to ensure the code is well-tested and maintainable. The code coverage of NeuroConv stands at 90%, which exceeds standard practices [@code_coverage_google; @coverage_continuous_integration_theater_2019].


The test data libraries contain a curated collection of example files spanning all supported data formats. We currently organize our testing libraries across three domains: [extracellular electrophysiology][https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/] in collaboration with the NEO and SpikeInterface development teams, [optical physiology][https://gin.g-node.org/CatalystNeuro/ophys_testing_data], and [behavior](https://gin.g-node.org/CatalystNeuro/behavior_testing_data). These files are selected to cover common usage patterns for each format, including different format versions, edge cases, missing streams, and various experimental configurations. The test data is hosted in a public version control system ([G-Node](https://gin.g-node.org)) and uses git-annex technology to efficiently manage large files. This approach allows us to maintain a lightweight repository while providing access to the full set of test data files. The test data library is automatically downloaded during the CI pipeline and cached using GitHub Actions, ensuring that tests run with the most up-to-date and relevant data with minimal transfer overhead.


## Community and Ecosystem

### NeuroConv in the NWB Conversion Landscape

NeuroConv occupies a strategic position within the broader NWB ecosystem, bridging the gap between low-level programming interfaces and high-level user tools. The conversion landscape offers different approaches suited to varying user needs and technical expertise levels.

:::{figure} assets/conversion_comparisons.png
:label: fig:assets/conversion_comparisons
This illustrates where NeuroConv stands in regard to other conversion tools. For low-level, high-precision control, users can employ the NWB APIs directly (PyNWB in Python, MatNWB in MATLAB). For a guided GUI-based experience, the NWB GUIDE offers the best option but may be too rigid for complex workflows. NeuroConv stands as a middle ground, automating the conversion of a large number of formats while still allowing for customization and flexibility.
:::

At the foundational level, PyNWB and MatNWB provide direct programmatic access to the NWB specification, offering maximum flexibility but requiring deep understanding of the standard. NWB GUIDE offers a graphical interface that democratizes access to NWB conversion through guided workflows, making it accessible to users without programming experience. NeuroConv complements both approaches by providing programmatic automation while maintaining the flexibility needed for complex, multi-modal experimental setups.

This ecosystem approach ensures that researchers can choose the conversion method that best matches their technical expertise and experimental complexity, while all approaches converge on the same standardized output format.

### Integration with Data Archives and Visualization

The Distributed Archives for [Neurophysiology Data Integration (DANDI)](https://dandiarchive.org/) platform complements NWB by providing free hosting for NWB-formatted datasets up to terabytes in size. DANDI offers researchers a pathway to meet NIH data sharing requirements while effectively archiving their data and leveraging an expanding ecosystem of visualization and analysis tools. The platform supports versioned datasets, comprehensive metadata, and API access, making it an ideal complement to NeuroConv's conversion capabilities.

Neurosift [@doi:10.21105/joss.06590] provides web-based visualization tools specifically designed for NWB files, enabling researchers to explore their converted datasets without requiring local software installation. This browser-based approach facilitates data sharing and collaborative analysis, particularly important for large datasets that benefit from cloud-based access patterns.

Together, NWB, NeuroConv, DANDI, and Neurosift create a comprehensive ecosystem that spans the entire data lifecycle from acquisition to publication and reuse, supporting emerging software domains from electrophysiological spike sorting to calcium imaging segmentation and behavioral pose estimation.

### Integration with the Neuroscience and Wider Software Ecosystem

NeuroConv pursues a “no-wheel-reinvention” strategy: wherever a mature open-source library already parses a file format or implements a preprocessing step, NeuroConv delegates that responsibility instead of duplicating it. This design choice concentrates development effort on the NWB hand-off layer while reducing maintenance burden and maximising compatibility with community standards.

For electrophysiology workflows, NeuroConv relies on NEO[@neo] for format parsing and on SpikeInterface[@spikeinterface] for unified access to raw and spike-sorted data and probe interface [@probeinterface] for selected probe metadata. By layering its NWB exporters directly on top of these libraries, NeuroConv inherits support for acquisition systems ranging from Neuropixels to legacy multi-electrode arrays while ensuring that any preprocessing, sorting, or quality-control metrics performed in SpikeInterface are transferred losslessly to the final NWB file.

For optical physiology, NeuroConv maintains roiextractors to provide a single API spanning Suite2P, CaImAn, EXTRACT, and CNMF-E segmentation outputs. roiextractors, in turn, depends on tifffile[@doi:10.5281/zenodo.6795860] to decode the heterogeneous TIFF variants produced by modern microscopes. This layered approach ensures that raw image stacks, ROI masks, fluorescence traces, and deconvolved events are all represented consistently in NWB.

Behavioural data integration combines several specialised libraries. Key-point trajectories are ingested through the Python APIs of sleap io [@deeplabcut], and neuroconv adapted for DeepLabCut and  Lightning Pose[@lightningpose]; audio waveforms are read with SciPy[@scipy]; and high-definition video frames are handled via OpenCV, which delegates codec support to FFmpeg. We also benefit from the [pymatreader](https://pymatreader.readthedocs.io/en/latest/) library for reading MATLAB .mat files, which are commonly used in neuroscience for storing experimental data and analysis results. This library provides a unified interface for reading MATLAB files, allowing NeuroConv to seamlessly integrate data stored in this format into NWB files.

Crucially, NeuroConv is not a passive consumer of these dependencies. Large-scale conversions expose edge cases—unexpected metadata tags, off-by-one timestamps, floating-point overflows—that formal test suites rarely capture. NeuroConv developers file reproducible issue reports, submit pull requests with fixes, add regression tests, and participate in release discussions across the aforementioned projects. We believe that this reciprocal workflow ensures that improvements made during NWB conversion propagate upstream, strengthening the wider neuroscience software ecosystem while continuously enhancing NeuroConv’s own reliability.

<!-- Additionally, we support specialized neuroscience data types through targeted integrations: TiffFile [@doi:10.5281/zenodo.6795860] for the complex TIFF variants common in microscopy, PyMatReader [@pymatreader] for MATLAB-based analysis outputs, and format-specific readers for acquisition systems like Bruker, ScanImage, and Miniscope. This neuroscience-focused approach ensures that NeuroConv addresses the actual data formats and analysis workflows used in contemporary systems neuroscience.

Our effort is built on the work of other packages in the neuroscience scientific community. Specifcally, for extracellular electrophysiology we leverage [NEO](https://neo.readthedocs.io/en/latest/) [@neo] through [SpikeInterface](https://spikeinterface.readthedocs.io/en/stable/) [@spikeinterface] for raw extracellular electrophysiology and spike-sorted data. For optical imaging, we have developed and maintain [roiextractors](https://roiextractors.readthedocs.io/en/latest/index.html), which provides a unified interface for both raw imaging data and the output of popular processing pipelines like suite2p and CaImAn. Here we also relying strongly on other packagest like the [tifffile](https://github.com/cgohlke/tifffile/) [@doi:10.5281/zenodo.6795860] python library . Behavior, being more heterogenous requires a more scattered approach, for handling audio we relying on scipy [@scipy] and the python standard library, for video we use [opencv](https://opencv.org/) whicn in turns uses [ffmpeg](https://ffmpeg.org/). 

[pymatreader](https://pymatreader.readthedocs.io/en/latest/)  -->


## Current Limitations

While NeuroConv has significantly improved data standardization processes, some challenges remain:

- **Format Coverage**: Despite supporting 44 formats, new acquisition systems and format versions continually emerge. While users can develop custom DataInterfaces, these require understanding both the source format and NeuroConv's architecture.
- **Custom Lab Formats**: Many labs store data in custom formats, often as MATLAB .mat files or custom csv files. These formats tend to be highly variable and rapidly evolving, making automated conversion challenging. NeuroConv works best with data in its original acquisition format or standardized processing output.
- **Programming Prerequisites**: While NeuroConv substantially reduces the coding burden, it still requires basic programming knowledge, including object-oriented concepts.

## Future Directions

* **Enhanced User Experience**: We are implementing comprehensive improvements to examples, tutorials, and documentation. We aim to slowly transition to the Diataxis [@diataxis] framework which will provide clearer learning pathways for users with different backgrounds and goals.* Keep in line with the latest developments of the schema and the NWB standard. 
* **NWB Standard Evolution**: We maintain alignment with NWB schema developments through active participation and close follow-up of the [NWB Extensions Proposals](https://github.com/nwb-extensions/nwbep-review/). Current developments include improved schemas for describing experimental events (NWBEP001), extracellular electrophysiology (NWBEP002), and optical physiology (NWBEP003, NWBEP004). This ensures that NeuroConv users benefit from the latest standard improvements.
* **AI-Assisted Conversion**: We are exploring large language model integration to advance our core mission of automating neurophysiology data conversion. This includes using LLMs to generate DataInterfaces from source format documentation, automatically extract metadata from experimental protocols, and generate custom conversion pipelines based on natural language requirements (conversion agent).
* **Cloud-Optimized Storage**: We are implementing improved chunking patterns and compression strategies for large datasets to enhance cloud access performance. This includes systematic experimentation through the [NWB Benchmarks Project](https://nwb-benchmarks.readthedocs.io/en/latest/) to determine optimal chunk sizes and compression algorithms [@doi:10.48550/arXiv.1601.07028]. Our goal is to implement evidence-based heuristics that ensure efficient and performant data storage as well as having a friendly and clear API that allows cutomization and flexibility for users to adapt to their specific needs.

 
## Closing Remarks


NeuroConv represents a critical step toward realizing the vision of FAIR neurophysiology data. By automating the conversion of diverse data formats into a common standard, we enable researchers to focus on scientific discovery rather than data wrangling. The success of this approach depends not only on technical implementation but on fostering a community that values standardization, reproducibility, and open science.

Our work demonstrates that effective scientific software development requires balancing automation with flexibility, standardization with customization, and ease of use with powerful capabilities. The challenges we have addressed—format diversity, metadata complexity, and scale—are not unique to neurophysiology but represent broader issues in scientific computing that require community-driven solutions.

The broader implications extend beyond neurophysiology to any scientific domain grappling with data heterogeneity and the need for standardization. Our approach of abstracting format complexity through unified interfaces, while maintaining extensibility through modular architecture, provides a template for similar challenges in other fields.

As the neurophysiology community continues to generate increasingly complex and voluminous datasets, tools like NeuroConv become essential infrastructure for scientific progress. By lowering barriers to data standardization and sharing, we contribute to a future where scientific data is truly FAIR—findable, accessible, interoperable, and reusable—accelerating discovery and enhancing reproducibility across the field.



