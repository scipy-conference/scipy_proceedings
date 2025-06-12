---
# Ensure that this title is the same as the one in `myst.yml`
title: NeuroConv. Streamlining Neurophysiology Data Conversion to the NWB Standard
keywords: Neurodata Without Borders, NWB, Neurophysiology, Data standardization, Data conversion, DANDI, Python, Scientific software, Large-scale data
abstract: |
  Converting diverse neurophysiology data to the standardized Neurodata Without Borders (NWB) format remains a significant barrier to data sharing and reuse. We present NeuroConv, a software that enables converting 47 distinct data formats while handling high-volume data as well as extracting meaningful metadata. The library has enabled the standardization of over 350 datasets totaling more than 350 TB in the DANDI archive and has been deployed to create automated conversion pipelines for 40 laboratories with unique experimental paradigms. By reducing technical barriers to NWB adoption, our tools accelerate progress toward reproducible neuroscience research through standardized data sharing.
---

## Introduction

Modern neurophysiology research generates increasingly complex, multimodal, and large-scale datasets that demand robust standards for organization, sharing, and archival. The [Neurodata Without Borders (NWB)](https://nwb.org/) [@nwb_2015; @nwb_2022] format has emerged as a comprehensive solution, storing contextual metadata alongside primary data objects to ensure long-term interpretability. A key strength of NWB is its ability to unify diverse experimental modalities such as simultaneous behavioral and electrophysiology recordings that usually reside in scattered files and formats. By consolidating all relevant information about an experimental session, NWB enables robust reanalysis and improves scientific reproducibility.

Despite these advantages, several significant challenges impede widespread NWB adoption. First, researchers must develop at least a basic understanding of NWB's data organization principles which can be a daunting prospect given the format's comprehensive scope. This learning curve presents a substantial barrier to integrating NWB standards into daily laboratory practices. Additionally, utilizing NWB's data Application Programming Interfaces (APIs), such as [pyNWB](https://pynwb.readthedocs.io/en/stable/) and [MatNWB](https://nwb.org/matnwb/), requires programming expertise that may exceed the capabilities of many researchers. This technical barrier often restricts NWB adoption to those with coding proficiency, dedicated technical staff, or places an outsized burden on junior lab members who possess the necessary skills.

Labs frequently face two distinct conversion challenges: processing existing backlogged data and establishing automated pipelines for future data collection. The latter is particularly crucial for minimizing duplicated work, as labs aim to automatically convert newly generated data into NWB format. However, developing robust conversion pipelines presents significant challenges due to the diversity of neurophysiology data formats. A single lab may employ multiple modalities such as voltage recording, optical imaging, optogenetics, and behavioral tracking, each with its own software-dependent formats lacking standardization.

## Background and Motivation

The challenges outlined above stem from fundamental characteristics of the neurophysiology data landscape. The field encompasses diverse experimental modalities, including microscopy for optical imaging, extracellular and intracellular electrophysiology for neural activity recording, and a wide array of behavioral tracking approaches, each with its own specialized requirements and methodological considerations [@nwb_2015; @MEF3_format_2016; @nwb_2022]. Across these modality, researchers rely on dozens of acquisition systems, each typically recording data in its own proprietary format. These formats prioritize different aspects of data handling, often optimizing for write speed during acquisition over storage efficiency or cross-platform compatibility. This layered diversity across both modalities and acquisition systems creates a complex landscape where formats vary widely in their efficiency, support longevity, metadata richness, and cross-platform compatibility.

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

While NWB has emerged as a unifying standard that addresses many common pitfalls of proprietary formats, the conversion process itself remains a significant bottleneck. This process requires deep knowledge of experimental design, source data formats, and the NWB standard, expertise that is rarely concentrated in a single individual or even within a single laboratory.

## NeuroConv Architecture and Design

To address these multifaceted challenges, we developed [NeuroConv](https://neuroconv.readthedocs.io/en/stable/index.html), an open source library that automates the ingestion and conversion of neurophysiology data from diverse formats into NWB. The development of NeuroConv required solving the following fundamental challenges:

```{list-table} Key technical challenges addressed by NeuroConv's architecture, highlighting the multifaceted requirements for automated neurophysiology data conversion.
:label: tbl:neuroconv-challenges
:header-rows: 1
* - Challenge
  - Solution Approach
* - Handling format and metadata diversity
  - Abstracting the complexity of 47+ distinct data formats while preserving format-specific metadata and ensuring NWB compliance
* - Managing high-volume data efficiently  
  - Processing datasets that exceed available RAM through streaming and chunked operations
* - Supporting complex experimental setups
  - Accommodating multi-stream conversions where different modalities are recorded simultaneously in different formats, while preserving time synchronization
```

The following sections detail how NeuroConv's architecture addresses each of these challenges through a modular, extensible design that maintains both flexibility and ease of use.


### Handling Diverse Data Formats
The challenge of format diversity in neurophysiology extends beyond their sheer number. Many formats, such as Neuralynx, exist in multiple versions, while others, like TIFF, exhibit significant internal variability in how labs use them. NeuroConv addresses this complexity through a unified architecture centered on the DataInterface abstraction. Each supported format has a dedicated DataInterface that encapsulates the format-specific logic for data reading, metadata extraction, and NWB conversion while presenting a consistent API to users. The DataInterface serves as the fundamental building block of NeuroConv, providing a standardized pathway from diverse source formats to NWB output. This abstraction enables users to work with any supported format using identical code patterns, regardless of the underlying format complexity or internal details. The minimal conversion pipeline is illustrated in {ref}`fig:assets/minimal_conversion_pipeline`:

:::{figure} assets/minimal_conversion_pipeline.png
:label: fig:assets/minimal_conversion_pipeline
The process begins with source data (e.g., binary recordings, metadata, and configuration files). A data-specific Interface object is instantiated and used to extract metadata via the .get_metadata() method. The resulting metadata can be optionally edited by the user to fill in missing or corrected fields. The finalized metadata and source data are then passed to the .run_conversion() method, which writes a complete NWB file compliant with the standard.
:::

Programmatically, this process can be summarized in a few lines of code, as shown below. The DataInterface handles all format-specific complexities internally, from parsing proprietary binary structures to extracting embedded metadata, while ensuring the output adheres to NWB best practices:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from neuroconv import DataInterface

# Initialize the DataInterface for a specific format
interface = DataInterface(file_path="path/to/data/file")

# Extract the metadata from the source format
metadata = interface.get_metadata()

# Modify metadata as needed, add missing or correct existing fields
metadata["NWBFile"]["experimenter"] = ["Baggins, Bilbo"]
metadata["NWBFile"]["experiment_description"] = "Example neurophysiology experiment"
metadata["NWBFile"]["institution"] = "University of Middle Earth"
# session_start_time is required for conversion
metadata["NWBFile"]["session_start_time"] = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("Middle-earth/Shire"))

# Add the data to an NWB File and write it to disk 
interface.run_conversion(nwbfile_path="path/to/nwbfile.nwb", metadata=metadata)
```

This core pattern ensures consistency and simplicity across all supported formats:

1. Instantiate the appropriate DataInterface
2. Extract and modify metadata as needed  
3. Execute the conversion

The DataInterface abstracts away format-specific complexities—from parsing proprietary binary structures to extracting embedded metadata—while automatically enforcing NWB best practices for data organization and storage optimization.

Currently supporting 47 distinct input formats (Table 1), each DataInterface is comprehensively documented and demonstrated in the [Conversion Gallery](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/index.html), where users can find complete examples requiring only ~5 lines of code to perform full data conversion. Throughout the conversion process, NeuroConv enforces NWB Best Practices for metadata and data organization while optimizing data storage for both archival purposes and cloud computing requirements.

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
As NeuroConv's format support has expanded to 47+ formats, dependency management has become increasingly complex. Each format often requires specialized libraries with potentially conflicting version requirements, creating dependency resolution challenges that can make installation difficult or impossible. For example, different electrophysiology formats may depend on incompatible versions of numerical libraries, while imaging formats might require conflicting versions of image processing packages. Additionally, installing all dependencies simultaneously would create an unnecessary and inefficient environment with hundreds of packages, many of which users never need.

To address these challenges, we rely on [installation extras](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras) to manage installation complexity. Users can specify only the formats they need during installation:


```python
pip install "neuroconv[spikeglx,phy,deeplabcut]"
```

This approach aggregates only the required dependencies for selected formats, avoiding dependency conflicts while ensuring that the installation remains lightweight and manageable for end users. Users working with a specific subset of formats can maintain clean, minimal environments without the overhead of unused dependencies.

### Handling Multi Stream Conversions

Neurophysiology experiments typically involve multiple simultaneous data streams from different modalities, such as raw electrophysiology recordings, spike-sorted data, and behavioral video. Each stream may be recorded in a different format, leading to complex conversion requirements. NeuroConv's architecture supports multi-stream conversions through the aggregation of DataInterfaces within a Converter framework as illustrated {ref}`fig:assets/diagram_converter`

:::{figure} assets/diagram_converter.png
:label: fig:assets/diagram_converter
The Converter orchestrates multiple specialized interfaces (A, B, C), each handling different data types (electrophysiology, imaging, and behavior). Individual interfaces extract metadata from their respective data sources, which the Converter combines into a single, user-editable metadata structure through its get_metadata() method. The Converter's run_conversion() method then coordinates all interfaces to produce a unified NWB file containing all data modalities. This design pattern enables flexible, modular integration of heterogeneous neuroscience data into a single standardized format.
:::

The converter pattern enables combining multiple DataInterface instances into a single conversion workflow. This allows users to convert all relevant data streams from an experiment into a single NWB file, ensuring that all data is properly aligned and associated with the correct metadata. The Converter class handles the orchestration of multiple DataInterfaces, managing the order of operations and resolving any conflicts in metadata or data organization. Here's an example of conversion for a multi-modal experimental session:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import SpikeGLXRecordingInterface, PhySortingInterface, DeepLabCutInterface

# Initialize the DataInterfaces for each data stream
recording_interface = SpikeGLXRecordingInterface(file_path="path/to/recording/file")
sorting_interface = PhySortingInterface(file_path="path/to/sorting/file")
behavior_interface = DeepLabCutInterface(file_path="path/to/behavior/file")

# Create the ConverterPipe with the DataInterfaces
data_interfaces = [recording_interface, sorting_interface, behavior_interface]
converter = ConverterPipe(data_interfaces=data_interfaces)

# Extract metadata from all interfaces
metadata = converter.get_metadata()

# Modify metadata as needed, add missing or correct existing fields
metadata["NWBFile"]["experimenter"] = ["Baggins, Bilbo"]
metadata["NWBFile"]["experiment_description"] = "Multi-modal neurophysiology experiment"
metadata["NWBFile"]["institution"] = "University of Middle Earth"
# session_start_time is required for conversion
metadata["NWBFile"]["session_start_time"] = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("Middle-earth/Shire"))

# Run the conversion to create an NWB file
converter.run_conversion(nwbfile_path="path/to/nwbfile.nwb", metadata=metadata)
```
Note that the same pattern used for single interfaces extends seamlessly to multi-stream conversions. The user initializes multiple DataInterfaces for each data stream, aggregates them into a ConverterPipe, extracts metadata, modifies it as needed, and then runs the conversion process to create an NWB file. This modular approach allows NeuroConv to handle complex experimental setups with multiple data streams while maintaining a consistent interface for users.

### Handling High-Volume Data

Modern acquisition systems, such as multi-probe Neuropixel recordings or whole-brain optical imaging, generate massive volumes of data that continue to grow year over year [@neuropixels_2018; @optical_physiology_methods_2022; @stringer2024analysis]. These volumes of data pose a variety of challenges both for conversion and for long-term storage. Moreover, as cloud computing emerges as a solution [@amazon_scientific_workflows_2009; @ome_ngff_2021] for managing and storing large datasets, ensuring efficient accessibility for the scientific community becomes a critical consideration.

A critical feature is the ability to process datasets larger than available RAM. NeuroConv inherits from the work performed by the NWB core group with [iterative writing](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/plot_iterative_write.html#sphx-glr-tutorials-advanced-io-plot-iterative-write-py) to stream data in manageable chunks, with configurable chunk sizes based on available resources. We have extended this approach to support chunked reading from SpikeInterface, enabling buffered processing of numerous extracellular electrophysiology formats, such as SpikeGLX, Neuralynx, and Plexon. We have also implemented an iterative writing approach for roiextractors that allows buffered writing of large imaging datasets, such as those generated by whole-brain calcium imaging. Furthermore, we have implemented chunked solutions for other data-intensive formats such as video. This approach enables processing of arbitrarily large files and has been successfully tested on 100+ GB files using computers with only 8 GB of RAM.

For storage optimization, NeuroConv leverages HDF5 and Zarr's -- the current supported backends in NWB -- support for chunked, compressed datasets. Compression algorithms represent a trade-off between storage space and access speed [@alessio_compression_2023]. NeuroConv exposes an easy-to-use [API](https://neuroconv.readthedocs.io/en/stable/user_guide/backend_configuration.html) for configuring chunking and compression settings at the dataset level, allowing for quick experimentation while providing sensible defaults that work for most users.

Determining optimal chunk parameters involves complex tradeoffs [@zarr_performance; @nguyen2023impact]. Large chunks minimize the number of read operations but may require decompressing unnecessary data when accessing small subsets. Small chunks provide more precise access but increase overhead, particularly for cloud storage where each chunk requires a separate HTTP range request. Generally, appropriate chunking requires understanding the most common data access patterns. Since neurophysiology has relatively standardized analysis workflows and visualization patterns, it becomes feasible to implement evidence-based heuristics for chunk sizing across common data types, such as voltage recordings and imaging datasets.

### Multi-modal time synchronization  

Precise temporal alignment across diverse recording modalities is essential for accurate multi-modal data analysis and reproducibility. NeuroConv streamlines this critical process by providing intuitive, unified methods for time synchronization, leveraging common temporal references like hardware clocks or synchronization pulses. In many cases, it automatically detects and reconciles temporal offsets between devices, reducing manual effort and ensuring NWB files maintain internally consistent timestamps across all modalities. This automation enforces best practices for temporal metadata in NWB and enhances downstream analysis integrity.

### Cloud Deployment

NeuroConv supports both local installation (Linux, Windows, or macOS) and [cloud deployment](https://neuroconv.readthedocs.io/en/stable/user_guide/aws_demo.html) through a maintained [Docker image](https://neuroconv.readthedocs.io/en/stable/user_guide/docker_demo.html) containing all dependencies. We've developed a YAML-based specification language for defining conversion pipelines, validated through JSON schema. This specification can fully describe multi-subject, multi-session conversions with custom metadata at each level, enabling automated conversion through containerized NeuroConv deployments. 

### Testing and Quality Assurance

Ensuring reliable conversion across diverse neurophysiology data formats requires a robust testing infrastructure. Our testing framework rests on two pillars: an automated continuous integration (CI) pipeline built using [GitHub Actions](https://github.com/features/actions)  and a comprehensive test data library.

Our continuous integration pipeline ensures code quality and maintains compatibility across operating systems through standard software engineering practices. Failed tests block pull request merging, maintaining code quality standards while facilitating rapid development. The pipeline runs on every pull request and includes several key components: unit tests covering the core functionality of the library; integration tests ensuring the library works as expected with real data; cross-platform testing to verify compatibility across all supported operating systems; documentation builds to ensure up-to-date documentation; [doctest](https://docs.python.org/es/3.13/library/doctest.html) functionality to verify that our conversion gallery works with the current version of the code, preventing documentation drift; code style checks ensuring consistency and adherence to best practices; and test coverage analysis to ensure the code is well-tested and maintainable. The code coverage of NeuroConv stands at 90%, which exceeds standard practices [@code_coverage_google; @coverage_continuous_integration_theater_2019].

Our test data infrastructure comprises carefully curated libraries spanning all supported formats, organized across three specialized domains: [extracellular electrophysiology](https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/) (developed in collaboration with NEO and SpikeInterface teams), [optical physiology](https://gin.g-node.org/CatalystNeuro/ophys_testing_data), and [behavior](https://gin.g-node.org/CatalystNeuro/behavior_testing_data). Each library contains representative files selected to encompass common usage patterns, format variations, edge cases, and diverse experimental configurations. This comprehensive coverage ensures that our testing captures real-world scenarios that users encounter, from legacy format versions to modern acquisition systems with missing data streams or unconventional metadata structures. For continuous integration, we leverage [G-Node](https://gin.g-node.org)'s git-annex technology for efficient large file management, maintaining lightweight repositories while providing full access to the test datasets. During continuous integration, GitHub Actions automatically downloads and caches these libraries, ensuring tests execute with current data while minimizing transfer overhead and maintaining rapid build times.

## NWB Community and Software Ecosystem

### NeuroConv and the NWB Conversion Landscape

NeuroConv occupies a strategic position within the broader NWB ecosystem, bridging the gap between low-level programming interfaces and high-level user tools. The conversion landscape offers different approaches suited to varying user needs and technical expertise levels.

:::{figure} assets/conversion_comparisons2.png
:label: fig:assets/conversion_comparisons
This illustrates where NeuroConv stands in regard to other conversion tools. For low-level, high-precision control, users can employ the NWB APIs directly ([PyNWB](https://pynwb.readthedocs.io) in Python, [MatNWB](https://matnwb.readthedocs.io/) in MATLAB). For a guided GUI-based experience, the [NWB GUIDE](https://nwb-guide.readthedocs.io/) offers the best option but may be too rigid for complex workflows. NeuroConv stands as a middle ground, automating the conversion of a large number of formats while still allowing for customization and flexibility.
:::

At the foundational level, PyNWB and MatNWB provide direct programmatic access to the NWB specification, offering maximum flexibility but requiring deep understanding of the standard. NWB GUIDE offers a graphical interface that democratizes access to NWB conversion through guided workflows, making it accessible to users without programming experience. NeuroConv complements both approaches by providing programmatic automation while maintaining the flexibility needed for complex, multi-modal experimental setups.

This ecosystem approach ensures that researchers can choose the conversion method that best matches their technical expertise and experimental complexity, while all approaches converge on the same standardized output format.

### Integration with Data Archives and Visualization

The [Distributed Archives for Neurophysiology Data Integration (DANDI)](https://dandiarchive.org/) platform complements NWB by providing free hosting for NWB-formatted datasets up to terabytes in size. DANDI offers researchers a pathway to meet NIH data sharing requirements while effectively archiving their data and leveraging an expanding ecosystem of visualization and analysis tools. The platform supports versioned datasets, comprehensive metadata, and API access, making it an ideal complement to NeuroConv's conversion capabilities.

Neurosift [@doi:10.21105/joss.06590] provides web-based visualization tools specifically designed for NWB files, enabling researchers to explore their converted datasets without requiring local software installation. This browser-based approach facilitates data sharing and collaborative analysis, particularly important for large datasets that benefit from cloud-based access patterns. Neurosift is also automatically integrated with DANDI, allowing users to visualize their datasets directly from the archive. This integration provides a seamless experience for researchers, enabling them to share and explore their data without needing to download large files locally.

Together, NWB, NeuroConv, DANDI, and Neurosift showcase a comprehensive pipeline that spans the entire data lifecycle from acquisition to publication and reuse, supporting emerging software domains from electrophysiological spike sorting to calcium imaging segmentation and behavioral pose estimation.

### Integration with the Neuroscience and Wider Software Ecosystem

NeuroConv pursues a “no-wheel-reinvention” strategy: wherever a mature open-source library already parses a file format or implements a preprocessing step, NeuroConv delegates that responsibility instead of duplicating it. This design choice concentrates development effort on the NWB hand-off layer while reducing maintenance burden and maximising compatibility with community standards.

For electrophysiology workflows, NeuroConv leverages NEO [@neo] for format parsing and SpikeInterface [@spikeinterface] for unified access to raw and spike-sorted data, with ProbeInterface [@probeinterface] handling probe geometry metadata. This hierarchical approach enables NeuroConv to inherit comprehensive support for acquisition systems spanning from cutting-edge Neuropixels arrays to legacy multi-electrode platforms, while ensuring that preprocessing pipelines, spike sorting results, and quality metrics from SpikeInterface integrate seamlessly into the final NWB representation.

For optical physiology, NeuroConv maintains [ROIExtractors](https://github.com/catalystneuro/roiextractors), which provides a unified API for optical physiology data across both raw imaging and processed segmentation outputs. For raw imaging data, ROIExtractors relies heavily on TiffFile [@doi:10.5281/zenodo.6795860] to decode the heterogeneous TIFF variants produced by modern microscopy systems, handling diverse acquisition formats and metadata structures. For segmentation data, ROIExtractors integrates outputs from Suite2P [@suite2p], CaImAn [@caiman], EXTRACT [@extract_2017; @extract_2021], and CNMF-E [@cnmfe_e_2016, @cnmf_e_2018] analysis pipelines. This hierarchical approach ensures consistent NWB representation of raw image stacks, ROI masks, fluorescence traces, and deconvolved neural activity signals across different processing workflows.

Behavioral data integration is considerably more heterogeneous in nature and therefore resists a unified API such as the one for electro and optical physiology described previously. To handle this heterogeneity, NeuroConv leverages multiple specialized libraries to handle diverse data types. Pose estimation trajectories are processed through dedicated Python APIs: [SLEAP-IO](https://github.com/talmolab/sleap-io) for SLEAP [@sleap_2022] data, custom adapters for DeepLabCut [@deep_lab_cut_2018] and Lightning Pose [@lightning_pose_2024] outputs. Audio waveforms from auditory experimental setups are processed using SciPy [@scipy], while video streams for behavioral video streams are handled through OpenCV [@bradski2000opencv], which delegates codec operations to [FFmpeg](https://ffmpeg.org/). Additionally, we utilize [PyMatReader](https://pymatreader.readthedocs.io/en/latest/) for MATLAB .mat files, which remain prevalent in neuroscience for storing experimental data and analysis results. This library provides a robust interface for MATLAB file formats, enabling seamless integration of experimental data into NWB files.

Crucially, NeuroConv is not a passive consumer of these dependencies. Large-scale conversions expose edge cases in performance, unexpected metadata tags, off-by-one timestamps, floating-point overflows, that are hard to capture in formal test suites. We follow the policy of giving back by filing reproducible issue reports, submitting pull requests with fixes, adding regression tests, and participating in release discussions across the aforementioned projects. We believe that this reciprocal workflow ensures that improvements made during NWB conversion propagate upstream, strengthening the wider neuroscience software ecosystem while continuously enhancing NeuroConv’s own reliability.

## Current Limitations

While NeuroConv has significantly improved data standardization processes, some major challenges remain in the  :

- **Format Coverage**: Despite supporting 47 formats, the neurophysiology ecosystem continues to evolve with new acquisition systems, format versions, and experimental modalities emerging regularly. Each new system often introduces proprietary formats with unique metadata structures and data organization schemes. While NeuroConv's modular architecture allows users to develop custom DataInterfaces for unsupported formats, this process requires substantial technical expertise in both the source format's internal structure and NeuroConv's interface architecture. Additionally, maintaining custom interfaces as both the source format and NeuroConv evolve presents ongoing maintenance challenges for individual laboratories.

- **Custom Laboratory Formats**: Many laboratories develop custom data storage solutions tailored to their specific experimental workflows, often utilizing MATLAB .mat files, custom CSV schemas, or bespoke binary formats. These custom formats frequently exhibit significant variability not only between laboratories but even within the same laboratory over time as experimental protocols evolve. The heterogeneous nature of these formats—ranging from simple tabular data to complex nested structures with laboratory-specific metadata conventions—makes automated conversion particularly challenging. While NeuroConv can handle some standardized custom formats, the diversity of these approaches often requires manual intervention or custom code development. We recommend that laboratories work with data in its original acquisition format whenever possible, as this provides the richest metadata and most reliable conversion pathway.

- **Metadata Completeness**: While NeuroConv automatically extracts available metadata from source formats, many acquisition systems store incomplete or inconsistent metadata. Critical information such as experimental conditions, subject details, or calibration parameters may be missing, incorrectly formatted, or stored in non-standard locations. This limitation requires manual metadata curation and validation, which can be time-consuming for large datasets. Additionally, some formats store metadata in proprietary or undocumented structures that may not be fully accessible through existing parsing libraries.

- **Complex Experimental Paradigms**: Highly specialized experimental setups—such as those involving custom-built hardware, novel stimulation protocols, or unique behavioral paradigms—may not map directly onto NWB's data organization schemes. While NWB provides extension mechanisms for custom data types, leveraging these extensions through NeuroConv requires additional development effort and deep understanding of the NWB specification. These edge cases often require manual intervention or custom conversion pipelines that may not be easily generalizable to other laboratories.
- **Programming Prerequisites**: While NeuroConv substantially reduces the coding burden, it still requires basic programming knowledge, including object-oriented concepts. While we put a lot of effort into making the library as user-friendly as possible, users must still be comfortable with Python programming to effectively utilize NeuroConv. This includes understanding how to install the library, manage dependencies, and write scripts that leverage the DataInterface classes for conversion.

## Future Directions

* **Enhanced User Experience**: We are implementing comprehensive improvements to examples, tutorials, and documentation. We aim to slowly transition to the Diataxis [@diataxis] framework which will provide clearer learning pathways for users with different backgrounds and goals.
* **NWB Standard Evolution**: We actively participate in NWB schema development and maintain close alignment with emerging standards through engagement with [NWB Extensions Proposals](https://github.com/nwb-extensions/nwbep-review/). By monitoring and contributing to current developments—including enhanced schemas for experimental events (NWBEP001), extracellular electrophysiology (NWBEP002), and optical physiology (NWBEP003, NWBEP004)—we ensure that NeuroConv incorporates the latest standard improvements as they become available for the benefit of our users.
* **AI-Assisted Conversion**: We are exploring large language model integration to advance our core mission of automating neurophysiology data conversion. This includes using LLMs to generate DataInterfaces from source format documentation, automatically extract metadata from experimental protocols, and generate custom conversion pipelines based on natural language requirements (conversion agent).
* **Cloud-Optimized Storage**: We are implementing improved chunking patterns and compression strategies for large datasets to enhance cloud access performance. This includes systematic experimentation through the [NWB Benchmarks Project](https://nwb-benchmarks.readthedocs.io/en/latest/) to determine optimal chunk sizes and compression algorithms [@alessio_compression_2023]. Our goal is to implement evidence-based heuristics that ensure efficient and performant data storage, as well as having a friendly and clear API that allows customization and flexibility for users to adapt to their specific needs. In the future, we would also like to explore the inclusions of cost optimization considerations for cloud storage [@cost_effective_scientific_datasets_2016].

## Closing Remarks

We believe that NeuroConv represents a critical step toward realizing the vision of FAIR (Findable, Accessible, Interoperable, and Reusable) neurophysiology data. By automating the conversion of diverse data formats into a common standard, we enable researchers to focus on scientific discovery rather than data wrangling. The success of this approach depends not only on technical implementation but on fostering a community that values standardization, reproducibility, and open science.

Our work demonstrates that effective scientific software development requires balancing automation with flexibility, standardization with customization, and ease of use with powerful capabilities. The challenges we have addressed such as format diversity, metadata complexity, and scale are not unique to neurophysiology but represent broader issues in scientific computing that require community-driven solutions.

The broader implications extend beyond neurophysiology to any scientific domain grappling with data heterogeneity and the need for standardization. We believe that our approach of abstracting format complexity through unified interfaces, while maintaining extensibility through modular architecture, provides a template for similar challenges in other fields.

As the neurophysiology community continues to generate increasingly complex and voluminous datasets, tools like NeuroConv become essential infrastructure for scientific progress. By lowering barriers to data standardization and sharing, we contribute to a future where scientific data is truly FAIR, accelerating discovery and enhancing reproducibility across the field.
