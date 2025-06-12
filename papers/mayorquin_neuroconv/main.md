---
# Ensure that this title is the same as the one in `myst.yml`
title: NeuroConv. Streamlining Neurophysiology Data Conversion to the NWB Standard
keywords: Neurodata Without Borders, NWB, Neurophysiology, Data standardization, Data conversion, DANDI, Python, Scientific software, Large-scale data
abstract: |
  Converting diverse neurophysiology data to the standardized Neurodata Without Borders (NWB) format remains a significant barrier to data sharing and reuse. We present NeuroConv, a software that enables converting 47 distinct data formats while handling high-volume data as well as extracting meaningful metadata. The library has enabled the standardization of over 390 datasets totaling more than 350 TB in the DANDI archive and has been deployed to create automated conversion pipelines for 40 laboratories with unique experimental paradigms. By reducing technical barriers to NWB adoption, our tools accelerate progress toward reproducible neuroscience research through standardized data sharing.
---

# Introduction, Background and Motivation
Modern neurophysiology produces complex, multimodal, large-scale datasets that require disciplined organization, shareability, and long-term archiving. The [Neurodata Without Borders (NWB)](https://nwb.org/) format has become a leading solution, embedding rich contextual metadata directly alongside primary data to preserve interpretability over time [@nwb_2015; @nwb_2022]. Crucially, NWB can unite disparate modalities such as simultaneous behavioral and electrophysiological recordings typically stored in separate files and formats, thereby enabling rigorous re-analysis and bolstering reproducibility.

Yet several factors still hinder broad NWB adoption. Researchers must first grasp NWB’s detailed organizational schema; a steep learning curve for many laboratories. Using the core APIs, [PyNWB](https://pynwb.readthedocs.io/en/stable/) and [MatNWB](https://nwb.org/matnwb/), also demands coding proficiency that many researchers lack, placing the burden on a handful of technically skilled team members.

Additional obstacles arise from the intrinsic diversity of neurophysiology. Modalities range from optical microscopy to extracellular and intracellular electrophysiology and varied behavioral-tracking techniques, each with specialized requirements [@nwb_2015; @MEF3_format_2016; @nwb_2022]. Dozens of acquisition systems record in proprietary formats optimized for rapid data capture, not for storage efficiency or cross-platform use. Consequently, formats differ widely in efficiency, longevity of support, metadata depth, and interoperability.

Conversion therefore becomes the main bottleneck. Automated pipelines are essential to prevent repetitive effort and to guarantee that future datasets enter NWB seamlessly, but building such pipelines is hard. A single lab might combine voltage recordings, optical imaging, optogenetics, and behavioral tracking, demanding expertise in experimental design, source formats, and NWB itself, expertise rarely concentrated in one person or even one lab.

:::{table} Key challenges in converting neurophysiology datasets to NWB format.
:label: tbl:nwb-challenges

<table>
<tr><th>Challenge</th><th>Description</th></tr>
<tr><td>Format diversity</td><td>Source data formats span both proprietary and open standards, with many formats existing in multiple versions and internal variations</td></tr>
<tr><td>Metadata complexity</td><td>Metadata requirements vary substantially across experimental paradigms, with critical information often stored inconsistently or incompletely</td></tr>
<tr><td>Scale challenges</td><td>Dataset sizes frequently reach hundreds of gigabytes to terabytes, requiring specialized handling for memory-efficient processing</td></tr>
<tr><td>Expertise requirements</td><td>Following NWB best practices requires considerable knowledge of both the source formats and the NWB standard itself</td></tr>
<tr><td>Multi-modal integration</td><td>Experiments often combine multiple recording systems (electrophysiology, imaging, behavior) each storing data in different formats. This also introduces the problem of data alignment and synchronization across modalities.</td></tr>
</table>
:::

# NeuroConv: Architecture and Design

To address these multifaceted challenges described in the previous section we developed [NeuroConv](https://neuroconv.readthedocs.io/en/stable/index.html), an open source library that automates the ingestion and conversion of neurophysiology data from diverse formats into NWB. This section describes in detail how NeuroConv's architecture addresses each of these challenges through a modular, extensible design that maintains both flexibility and ease of use.

## Handling Format Diversity
The challenge of format diversity in neurophysiology extends beyond their sheer number (47 supported at the moment). Many formats, such as Neuralynx, have multiple versions, while others, like TIFF, exhibit significant internal variability in how labs use them. NeuroConv addresses this complexity through a unified architecture centered on the DataInterface abstraction. DataInterface is an abstract class for reading data, and each supported format has a dedicated DataInterface that encapsulates the format-specific logic for data reading, metadata extraction, and NWB conversion while presenting a consistent API to users. The DataInterface serves as the fundamental building block of NeuroConv, providing a standardized pathway from diverse source formats to NWB output. This abstraction enables users to work with any supported format using identical code patterns, regardless of the underlying format complexity or internal details. The minimal conversion pipeline is illustrated in {ref}`fig:assets/minimal_conversion_pipeline`:

:::{figure} assets/minimal_conversion_pipeline.png
:label: fig:assets/minimal_conversion_pipeline
The process begins with source data (e.g., binary recordings, metadata, and configuration files). A data-specific DataInterface object is instantiated and used to extract metadata via the .get_metadata() method. The resulting metadata can be optionally edited by the user to fill in missing or corrected fields. The finalized metadata and source data are then passed to the .run_conversion() method, which writes a complete NWB file compliant with the standard.
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

The DataInterface abstracts away format-specific complexities: from parsing proprietary binary structures to extracting embedded metadata.

Currently supporting 47 distinct input formats (@tbl:formats), each DataInterface is comprehensively documented and demonstrated in the [Conversion Gallery](https://neuroconv.readthedocs.io/en/stable/conversion_examples_gallery/index.html), where users can find complete examples requiring only ~5 lines of code to perform full data conversion. Throughout the conversion process, NeuroConv enforces NWB Best Practices for metadata and data organization while optimizing data storage for both archival purposes and cloud computing requirements.

:::{table} Comprehensive list of data formats supported by NeuroConv, organized by experimental modality and data type.
:label: tbl:formats

<table style="border-collapse: collapse; width: 100%;">
<thead>
<tr style="background-color: #2E5090; color: white;">
<th style="width: 35%; padding: 12px; text-align: left;">Category</th>
<th style="width: 25%; padding: 12px; text-align: left;">Subcategory</th>
<th style="width: 40%; padding: 12px; text-align: left;">Format</th>
</tr>
</thead>
<tbody>
<!-- Extracellular Electrophysiology -->
<tr style="background-color: #E8F2FF">
<td rowspan="26" style="padding: 8px; border-right: 2px solid #B8D4F1;"><strong>Extracellular Electrophysiology</strong></td>
<td rowspan="19" style="padding: 8px; color: #2E5090;"><em>Recording</em></td>
<td style="padding: 8px;">AlphaOmega</td>
</tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Axona</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Biocam</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Blackrock</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">European Data Format (EDF)</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Intan</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">MaxOne</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">MCSRaw</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">MEArec</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Neuralynx</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">NeuroScope</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">OpenEphys</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Plexon</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Plexon2</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Spike2</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Spikegadgets</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">SpikeGLX</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Tucker-Davis Technologies (TDT)</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">White Matter</td></tr>
<tr style="background-color: #D6E7FF">
<td rowspan="7" style="padding: 8px; color: #2E5090;"><em>Sorting</em></td>
<td style="padding: 8px;">Blackrock</td>
</tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">Cell Explorer<sup><a href="#cellexplorer-ref">7</a></sup></td></tr>
<tr style="background-color: #D6E7FF"><td style="padding: 8px;">KiloSort<sup><a href="#kilosort-ref">8</a></sup></td></tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">Neuralynx</td></tr>
<tr style="background-color: #D6E7FF"><td style="padding: 8px;">NeuroScope</td></tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">Phy</td></tr>
<tr style="background-color: #D6E7FF"><td style="padding: 8px;">Plexon</td></tr>

<!-- Intracellular Electrophysiology -->
<tr style="border-top: 3px solid #2E5090;">
<td style="background-color: #E8F2FF; padding: 8px; border-right: 2px solid #B8D4F1;"><strong>Intracellular Electrophysiology</strong></td>
<td style="padding: 8px;">—</td>
<td style="padding: 8px;">ABF</td>
</tr>

<!-- Optical Physiology -->
<tr style="border-top: 3px solid #2E5090; background-color: #E8F2FF">
<td rowspan="13" style="padding: 8px; border-right: 2px solid #B8D4F1;"><strong>Optical Physiology</strong></td>
<td rowspan="8" style="padding: 8px; color: #2E5090;"><em>Imaging</em></td>
<td style="padding: 8px;">Bruker</td>
</tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">HDF5</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Micro-Manager</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Miniscope</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Scanbox</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">ScanImage</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">Thor</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Tiff</td></tr>
<tr style="background-color: #D6E7FF">
<td rowspan="4" style="padding: 8px; color: #2E5090;"><em>Segmentation</em></td>
<td style="padding: 8px;">CaImAn<sup><a href="#caiman-ref">1</a></sup></td></tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">CNMFE<sup><a href="#cnmfe-ref">2</a></sup></td></tr>
<tr style="background-color: #D6E7FF"><td style="padding: 8px;">EXTRACT<sup><a href="#extract-ref">3</a></sup></td></tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">Suite2P<sup><a href="#suite2p-ref">4</a></sup></td></tr>
<tr style="background-color: #E8F2FF">
<td style="padding: 8px; color: #2E5090;"><em>Fiber Photometry</em></td>
<td style="padding: 8px;">TDT Fiber Photometry</td>
</tr>

<!-- Behavior -->
<tr style="border-top: 3px solid #2E5090; background-color: #E8F2FF">
<td rowspan="7" style="padding: 8px; border-right: 2px solid #B8D4F1;"><strong>Behavior</strong></td>
<td rowspan="5" style="padding: 8px; color: #2E5090;"><em>Motion Tracking</em></td>
<td style="padding: 8px;">DeepLabCut<sup><a href="#deeplabcut-ref">5</a></sup></td>
</tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">FicTrac</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">LightningPose</td></tr>
<tr style="background-color: #F5F9FF"><td style="padding: 8px;">Neuralynx NVT</td></tr>
<tr style="background-color: #E8F2FF"><td style="padding: 8px;">SLEAP<sup><a href="#sleap-ref">6</a></sup></td></tr>
<tr style="background-color: #D6E7FF">
<td style="padding: 8px; color: #2E5090;"><em>Audio/Video</em></td>
<td style="padding: 8px;">Videos</td>
</tr>
<tr style="background-color: #F5F9FF">
<td style="padding: 8px; color: #2E5090;"><em>Operant Conditioning</em></td>
<td style="padding: 8px;">MedPC</td>
</tr>

<!-- General Data -->
<tr style="border-top: 3px solid #2E5090; background-color: #E8F2FF">
<td rowspan="4" style="padding: 8px; border-right: 2px solid #B8D4F1;"><strong>General Data</strong></td>
<td style="padding: 8px; color: #2E5090;"><em>Image</em></td>
<td style="padding: 8px;">Image (png, jpeg, tiff, etc)</td>
</tr>
<tr style="background-color: #D6E7FF">
<td rowspan="3" style="padding: 8px; color: #2E5090;"><em>Text/Tabular</em></td>
<td style="padding: 8px;">CSV</td>
</tr>
<tr style="background-color: #E2EFFF"><td style="padding: 8px;">Excel</td></tr>
<tr style="background-color: #D6E7FF"><td style="padding: 8px;">Text</td></tr>
</tbody>
</table>

**References:**

<a id="caiman-ref">1.</a> CaImAn [@caiman]  
<a id="cnmfe-ref">2.</a> CNMFE [@cnmfe_e_2016; @cnmf_e_2018]  
<a id="extract-ref">3.</a> EXTRACT [@extract_2017; @extract_2021]  
<a id="suite2p-ref">4.</a> Suite2P [@suite2p]  
<a id="deeplabcut-ref">5.</a> DeepLabCut [@deep_lab_cut_2018]  
<a id="sleap-ref">6.</a> SLEAP [@sleap_2022]  
<a id="cellexplorer-ref">7.</a> Cell Explorer [@cell_explorer_2021]  
<a id="kilosort-ref">8.</a> KiloSort [@kilosort_2024]

:::
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
As NeuroConv's format support has expanded to 47+ formats, dependency management has become increasingly complex. Each format often requires specialized libraries with potentially conflicting version requirements, creating dependency resolution challenges that can make installation difficult or impossible. For example, libraries for reading different formats may require different versions of Python or different versions of common dependencies such as numpy. Additionally, installing the dependencies of every DataInterface would create an unnecessary and inefficient environment with hundreds of packages, many of which users never need for their conversion.

To address these challenges, we rely on [installation extras](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras) to manage installation complexity. Users can specify only the formats they need during installation:


```python
pip install "neuroconv[spikeglx,phy,deeplabcut]"
```

This approach aggregates only the required dependencies for selected formats, avoiding dependency conflicts while ensuring that the installation remains lightweight and manageable for end users. Users working with a specific subset of formats can maintain clean, minimal environments without the overhead of unused dependencies.

## Multi Stream Conversions

Neurophysiology experiments typically involve multiple simultaneous data streams from different modalities, such as raw electrophysiology recordings, spike-sorted data, and behavioral video. Each stream may be stored in a different format, leading to complex conversion requirements. NeuroConv's architecture supports multi-stream conversions through the aggregation of DataInterfaces within a Converter framework as illustrated {ref}`fig:assets/diagram_converter`

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

## Multi-modal time synchronization
Precise temporal alignment across diverse recording modalities is essential for accurate multi-modal data analysis and reproducibility. NeuroConv streamlines this critical process by providing intuitive, unified methods for time synchronization, leveraging common temporal references like hardware clocks or synchronization pulses. NeuroConv provides convenience functions for extracting times from pulse signals, and for aligning time using a single starting pulse or for regular pulses sent between systems. These alignment strategies can correct for differences in starting time and for temporal drift between systems, and work for systems with very different sampling rates. This automation enforces best practices for temporal metadata in NWB and enhances downstream analysis integrity.


## Handling High-Volume Data

Modern acquisition systems, such as multi-probe Neuropixel recordings or whole-brain optical imaging, generate massive volumes of data that continue to grow year over year [@neuropixels_2018; @optical_physiology_methods_2022; @stringer2024analysis]. These volumes of data pose a variety of challenges both for conversion and for long-term storage. Moreover, as cloud computing emerges as a solution [@amazon_scientific_workflows_2009; @ome_ngff_2021] for managing and storing large datasets, ensuring efficient accessibility for the scientific community becomes a critical consideration.

A critical feature is the ability to process datasets larger than available RAM. NeuroConv inherits from the work performed by the NWB core group with [iterative writing](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/plot_iterative_write.html#sphx-glr-tutorials-advanced-io-plot-iterative-write-py) to stream data in manageable chunks, with configurable buffer sizes based on available resources. We have extended this approach to support chunked reading that allows buffered writing of large datasets. This approach enables processing of arbitrarily large files and has been successfully tested on 100+ GB files using computers with only 8 GB of RAM.

For storage optimization, NeuroConv leverages chunking and compression in HDF5 and Zarr, the currently supported backends in NWB. Chunking allows large datasets to be read in manageable chunk sizes, and lossless compression algorithms allow us to reduce file size without altering the values of the data. There are many options for compression algorithms, and they present a trade-off between storage space and access speed [@alessio_compression_2023]. NeuroConv exposes an easy-to-use [API](https://neuroconv.readthedocs.io/en/stable/user_guide/backend_configuration.html) for configuring chunking and compression settings at the dataset level, allowing for quick experimentation while providing sensible defaults that work for most users.

Determining optimal chunk parameters involves complex tradeoffs [@zarr_performance]. Large chunks minimize the number of read operations but may require decompressing unnecessary data when accessing small subsets. Small chunks provide more precise access but increase overhead, particularly for cloud storage where each chunk requires a separate HTTP range request. Generally, appropriate chunking requires understanding the most common data access patterns. By understanding common analysis workflows and visualization patterns, it becomes feasible to implement evidence-based heuristics for chunk sizing across common data types, such as voltage recordings and imaging datasets.

## Cloud Deployment

NeuroConv supports both local installation (Linux, Windows, or macOS) and [cloud deployment](https://neuroconv.readthedocs.io/en/stable/user_guide/aws_demo.html) through a maintained [Docker image](https://neuroconv.readthedocs.io/en/stable/user_guide/docker_demo.html) containing all dependencies. We've developed a YAML-based specification language for defining conversion pipelines, validated through JSON schema. This specification can fully describe multi-subject, multi-session conversions with custom metadata at each level, enabling automated conversion through containerized NeuroConv deployments. 

## Testing and Quality Assurance

Ensuring reliable conversion across diverse neurophysiology data formats requires a robust testing infrastructure. Our testing framework rests on two pillars: an automated continuous integration (CI) pipeline built using [GitHub Actions](https://github.com/features/actions) and a comprehensive test data library.

Our continuous integration pipeline ensures code quality and maintains compatibility across operating systems through standard software engineering practices. The pipeline runs on every pull request and includes several key components: unit tests covering the core functionality of the library; integration tests ensuring the library works as expected with real data; cross-platform testing to verify compatibility across all supported operating systems; documentation builds to ensure up-to-date documentation; [doctest](https://docs.python.org/es/3.13/library/doctest.html) functionality to verify that our conversion gallery works with the current version of the code, preventing documentation drift; code style checks ensuring consistency and adherence to best practices; and test coverage analysis to ensure the code is well-tested and maintainable. The code coverage of NeuroConv stands at 90%, which exceeds standard practices [@code_coverage_google; @coverage_continuous_integration_theater_2019].  Failed tests block pull request merging, maintaining code quality standards while facilitating rapid development. 

Our test data infrastructure comprises carefully curated libraries spanning all supported formats, organized across three specialized domains: [extracellular electrophysiology](https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/) (developed in collaboration with NEO and SpikeInterface teams), [optical physiology](https://gin.g-node.org/CatalystNeuro/ophys_testing_data), and [behavior](https://gin.g-node.org/CatalystNeuro/behavior_testing_data). Each library contains representative files selected to encompass common usage patterns, format variations, edge cases, and diverse experimental configurations. This comprehensive coverage ensures that our testing captures real-world scenarios, from legacy format versions to modern acquisition systems with missing data streams or unconventional metadata structures. For continuous integration, we leverage [G-Node](https://gin.g-node.org)'s git-annex technology for efficient large file management, maintaining lightweight repositories while providing full access to the test datasets. During continuous integration, GitHub Actions automatically downloads and caches these libraries, ensuring tests execute with current data while minimizing transfer overhead and maintaining rapid build times.

# NWB Community and Software Ecosystem

## NeuroConv and the NWB Conversion Landscape

NeuroConv occupies a strategic position within the broader NWB ecosystem, bridging the gap between low-level programming interfaces and high-level user tools. The conversion landscape offers different approaches suited to varying user needs and technical expertise levels.

:::{figure} assets/conversion_comparisons2.png
:label: fig:assets/conversion_comparisons
This illustrates where NeuroConv stands in regard to other conversion tools. For low-level, high-precision control, users can employ the NWB APIs directly ([PyNWB](https://pynwb.readthedocs.io) in Python, [MatNWB](https://matnwb.readthedocs.io/) in MATLAB). For a guided GUI-based experience, the [NWB GUIDE](https://nwb-guide.readthedocs.io/) offers the best option but may be too rigid for complex workflows. NeuroConv stands as a middle ground, automating the conversion of a large number of formats while still allowing for customization and flexibility.
:::

At the foundational level, PyNWB and MatNWB provide direct programmatic access to the NWB specification, offering maximum flexibility but requiring deep understanding of the standard. NWB GUIDE offers a graphical interface that democratizes access to NWB conversion through guided workflows, making it accessible to users without programming experience. NeuroConv complements both approaches by providing programmatic automation while maintaining the flexibility needed for complex, multi-modal experimental setups.

This ecosystem approach ensures that researchers can choose the conversion method that best matches their technical expertise and experimental complexity, while all approaches converge on the same standardized output format.

## Integration with Data Archives and Visualization

The [Distributed Archives for Neurophysiology Data Integration (DANDI)](https://dandiarchive.org/) Archive complements NWB by providing free hosting for NWB-formatted datasets up to terabytes in size. DANDI offers researchers a pathway to meet NIH data sharing requirements while effectively archiving their data and leveraging an expanding ecosystem of visualization and analysis tools. The platform supports versioned datasets, comprehensive metadata, and API access, making it an ideal complement to NeuroConv's conversion capabilities.

Together, NWB, NeuroConv, DANDI, and Neurosift showcase a comprehensive pipeline that spans the entire data lifecycle from acquisition to publication and reuse, supporting emerging software domains from electrophysiological spike sorting to calcium imaging segmentation and behavioral pose estimation.

## Integration with the Neuroscience and Wider Software Ecosystem

NeuroConv pursues a “no-wheel-reinvention” strategy: wherever a mature open-source library already parses a file format or implements a preprocessing step, NeuroConv delegates that responsibility instead of duplicating it. This design choice concentrates development effort on the NWB hand-off layer while reducing maintenance burden and maximising compatibility with community standards.

For electrophysiology workflows, NeuroConv leverages NEO [@neo] for format parsing and SpikeInterface [@spikeinterface] for unified access to raw and spike-sorted data, with ProbeInterface [@probeinterface] handling probe geometry metadata. This hierarchical approach enables NeuroConv to inherit comprehensive support for acquisition systems spanning from cutting-edge Neuropixels arrays to legacy multi-electrode platforms, while ensuring that preprocessing pipelines, spike sorting results, and quality metrics from SpikeInterface integrate seamlessly into the final NWB representation.

For optical physiology, NeuroConv maintains [ROIExtractors](https://github.com/catalystneuro/roiextractors), which provides a unified API for optical physiology data across both raw imaging and processed segmentation outputs. For raw imaging data, ROIExtractors relies heavily on TiffFile [@doi:10.5281/zenodo.6795860] to decode the heterogeneous TIFF variants produced by modern microscopy systems, handling diverse acquisition formats and metadata structures. For segmentation data, ROIExtractors integrates outputs from Suite2P [@suite2p], CaImAn [@caiman], EXTRACT [@extract_2017; @extract_2021], and CNMF-E [@cnmfe_e_2016, @cnmf_e_2018] analysis pipelines. This hierarchical approach ensures consistent NWB representation of raw image stacks, ROI masks, fluorescence traces, and deconvolved neural activity signals across different processing workflows.

Behavioral data integration is considerably more heterogeneous in nature and therefore resists a unified API such as the one for electro and optical physiology described previously. To handle this heterogeneity, NeuroConv leverages multiple specialized libraries to handle diverse data types. Pose estimation trajectories are processed through dedicated Python APIs: [SLEAP-IO](https://github.com/talmolab/sleap-io) for SLEAP [@sleap_2022] data, custom adapters for DeepLabCut [@deep_lab_cut_2018] and Lightning Pose [@lightning_pose_2024] outputs. Audio waveforms from auditory experimental setups are processed using SciPy [@scipy], while video streams for behavioral video streams are handled through OpenCV [@bradski2000opencv], which delegates codec operations to [FFmpeg](https://ffmpeg.org/). Additionally, we utilize [PyMatReader](https://pymatreader.readthedocs.io/en/latest/) for MATLAB .mat files, which remain prevalent in neuroscience for storing experimental data and analysis results. This library provides a robust interface for MATLAB file formats, enabling seamless integration of experimental data into NWB files.

Crucially, NeuroConv is not a passive consumer of these dependencies. Large-scale conversions expose edge cases in performance, unexpected metadata tags, off-by-one timestamps, floating-point overflows, that are hard to capture in formal test suites. We follow the policy of giving back by filing reproducible issue reports, submitting pull requests with fixes, adding regression tests, and participating in release discussions across the aforementioned projects. We believe that this reciprocal workflow ensures that improvements made during NWB conversion propagate upstream, strengthening the wider neuroscience software ecosystem while continuously enhancing NeuroConv’s own reliability.

# Current Limitations

While NeuroConv has significantly improved data standardization processes, some major challenges remain:

**Format Coverage**: Despite supporting 47 formats, the neurophysiology ecosystem continues to evolve with new acquisition systems, format versions, and processing software. Each new system often introduces proprietary formats with unique metadata structures and data organization schemes. While NeuroConv's modular architecture allows users to develop custom DataInterfaces for unsupported formats, this process requires substantial technical expertise in both the source format's internal structure and NeuroConv's interface architecture. Additionally, maintaining custom interfaces as both the source format and NeuroConv evolve presents ongoing maintenance challenges for individual laboratories.

**Custom Laboratory Formats**: Many laboratories develop custom data storage solutions tailored to their specific experimental workflows, often utilizing MATLAB .mat files, custom CSV schemas, or bespoke binary formats. These custom formats frequently exhibit significant variability not only between laboratories but even within the same laboratory over time as experimental protocols evolve. The heterogeneous nature of these formats—ranging from simple tabular data to complex nested structures with laboratory-specific metadata conventions—makes automated conversion particularly challenging. While NeuroConv can handle some standardized custom formats, the diversity of these approaches often requires manual intervention or custom code development. We recommend that laboratories work with data in its original acquisition format whenever possible, as this provides the richest metadata and most reliable conversion pathway.

**Metadata Completeness**: While NeuroConv automatically extracts available metadata from source formats, many acquisition systems store incomplete or inconsistent metadata. Critical information such as experimental conditions, subject details, or calibration parameters may be missing, incorrectly formatted, or stored in non-standard locations. This limitation requires manual metadata curation and validation, which can be time-consuming for large datasets. Additionally, some formats store metadata in proprietary or undocumented structures that may not be fully accessible through existing parsing libraries.

**Complex Experimental Paradigms**: Highly specialized experimental setups—such as those involving custom-built hardware, novel stimulation protocols, or unique behavioral paradigms—may not map directly onto NWB's data organization schemes. While NWB provides extension mechanisms for custom data types, leveraging these extensions through NeuroConv requires additional development effort and deep understanding of the NWB specification. These edge cases often require manual intervention or custom conversion pipelines that may not be easily generalizable to other laboratories.

**Programming Prerequisites**: While NeuroConv substantially reduces the coding burden, it still requires basic programming knowledge, including object-oriented concepts. While we aim to make the library as user-friendly as possible, users must still be comfortable with Python programming to effectively utilize NeuroConv. This includes understanding how to install the library, manage dependencies, and write scripts that leverage the DataInterface classes for conversion.

# Future Directions

* **Enhanced User Experience**: We are implementing comprehensive improvements to examples, tutorials, and documentation. We aim to slowly transition to the Diataxis [@diataxis] framework which will provide clearer learning pathways for users with different backgrounds and goals.
* **NWB Standard Evolution**: We actively participate in NWB schema development and maintain close alignment with emerging standards through engagement with [NWB Extensions Proposals](https://github.com/nwb-extensions/nwbep-review/). By monitoring and contributing to current developments—including enhanced schemas for experimental events (NWBEP001), extracellular electrophysiology (NWBEP002), and optical physiology (NWBEP003, NWBEP004)—we ensure that NeuroConv incorporates the latest standard improvements as they become available for the benefit of our users.
* **AI-Assisted Conversion**: We are exploring large language model integration to advance our core mission of automating neurophysiology data conversion. This includes using LLMs to generate DataInterfaces from source format documentation, automatically extract metadata from experimental protocols, and generate custom conversion pipelines based on natural language requirements (conversion agent).
* **Cloud-Optimized Storage**: We are implementing improved chunking patterns and compression strategies for large datasets to enhance cloud access performance. This includes systematic experimentation through the [NWB Benchmarks Project](https://nwb-benchmarks.readthedocs.io/en/latest/) to determine optimal chunk sizes and compression algorithms [@alessio_compression_2023]. Our goal is to implement evidence-based heuristics that ensure efficient and performant data storage, as well as having a friendly and clear API that allows customization and flexibility for users to adapt to their specific needs. In the future, we would also like to explore the inclusions of cost optimization considerations for cloud storage [@cost_effective_scientific_datasets_2016].

# Closing Remarks

We believe that NeuroConv represents a critical step toward realizing the vision of FAIR (Findable, Accessible, Interoperable, and Reusable) neurophysiology data. By automating the conversion of diverse data formats into a common standard, we enable researchers to focus on scientific discovery rather than data wrangling. The success of this approach depends not only on technical implementation but on fostering a community that values standardization, reproducibility, and open science.

Our work demonstrates that effective scientific software development requires balancing automation with flexibility, standardization with customization, and ease of use with powerful capabilities. The challenges we have addressed such as format diversity, metadata complexity, and scale are not unique to neurophysiology but represent broader issues in scientific computing that require community-driven solutions.

The broader implications extend beyond neurophysiology to any scientific domain grappling with data heterogeneity and the need for standardization. We believe that our approach of abstracting format complexity through unified interfaces, while maintaining extensibility through modular architecture, provides a template for similar challenges in other fields.

As the neurophysiology community continues to generate increasingly complex and voluminous datasets, tools like NeuroConv become essential infrastructure for scientific progress. By lowering barriers to data standardization and sharing, we contribute to a future where scientific data is truly FAIR, accelerating discovery and enhancing reproducibility across the field.
