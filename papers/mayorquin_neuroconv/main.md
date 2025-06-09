---
# Ensure that this title is the same as the one in `myst.yml`
title: NeuroConv. Streamlining Neurophysiology Data Conversion to the NWB Standard
keywords: Neurodata Without Borders, NWB, Neurophysiology, Data standardization, Data conversion, DANDI, Python, Scientific software, Large-scale data
abstract: |
  Converting diverse neurophysiology data to the standardized Neurodata Without Borders (NWB) format remains a significant barrier to data sharing and reuse. We present NeuroConv, a software that enables converting 44 distinct data formats while handling high-volume data as well as extracting meaningful metadata. The library has enabled the standardization of over 300 datasets totaling more than 300 TB in the DANDI archive and has been deployed to create automated conversion pipelines for 40 laboratories with unique experimental paradigms. By reducing technical barriers to NWB adoption, our tools accelerate progress toward reproducible neuroscience research through standardized data sharing.
---

## Introduction

Modern neurophysiology research generates increasingly complex, multimodal, and large-scale datasets that demand robust standards for organization, sharing, and archival. The [Neurodata Without Borders (NWB)](https://nwb.org/)[@nwb] format has emerged as a comprehensive solution, storing contextual metadata alongside primary data objects to ensure long-term interpretability. A key strength of NWB is its ability to unify diverse experimental modalities—such as simultaneous behavioral and electrophysiology recordings that usually reside in scattered files and formats. By consolidating all relevant information about an experimental session, NWB enables robust reanalysis and reproducibility.

The Distributed Archives for [Neurophysiology Data Integration (DANDI)](https://dandiarchive.org/) platform complements NWB by providing free hosting for NWB-formatted datasets up to terabytes in size. DANDI offers researchers a pathway to meet NIH data sharing requirements while effectively archiving their data and leveraging an expanding ecosystem of visualization and analysis tools. Together, NWB and DANDI create an ideal foundation for emerging software domains, from electrophysiological spike sorting to calcium imaging segmentation and behavioral pose estimation.

Despite these advantages, several significant challenges impede widespread NWB adoption. First, researchers must develop at least a basic understanding of NWB's data organization principles—a daunting prospect given the format's comprehensive scope. This learning curve presents a substantial barrier to integrating NWB standards into daily laboratory practices. Additionally, utilizing NWB's data Application Programming Interfaces (APIs), such as [pyNWB](https://pynwb.readthedocs.io/en/stable/) and (MatNWB)[https://nwb.org/matnwb/], requires programming expertise that may exceed the capabilities of many researchers. This technical barrier often restricts NWB adoption to those with coding proficiency, dedicated technical staff, or places an outsized burden on junior lab members who possess the necessary skills.

Labs frequently face two distinct conversion challenges: processing existing backlogged data and establishing automated pipelines for future data collection. The latter is particularly crucial for minimizing duplicated work, as labs aim to automatically convert newly generated data into NWB format. However, developing robust conversion pipelines presents significant challenges due to the diversity of neurophysiology data formats. A single lab may employ multiple modalities—voltage recording, optical imaging, optogenetics, and behavioral tracking—each with its own software-dependent formats lacking standardization.

## Background and Motivation

Neurophysiology research generates vast amounts of data across diverse formats, from electrophysiology recordings to calcium imaging and behavior tracking. The NWB format has emerged as a community standard for storing and sharing neurophysiology data, promoting reproducibility and collaboration in neuroscience. However, converting existing datasets to NWB presents significant challenges:

- Source data formats are highly diverse (proprietary and open)
- Metadata requirements vary substantially
- Dataset sizes often reach hundreds of gigabytes to terabytes
- Following NWB best practices requires considerable expertise

## NeuroConv Architecture and Design

Modern neurophysiology research faces a significant data standardization challenge. The field relies on dozens of acquisition systems, each typically recording data in its own proprietary format. These formats prioritize different aspects of data handling: some optimize for write speed during acquisition, others for storage efficiency or cross-platform compatibility. This diversity creates a complex landscape where formats vary widely in their efficiency, support longevity, metadata richness, and cross-platform compatibility. While NWB has emerged as a unifying standard that addresses many common pitfalls of proprietary formats, converting data to NWB remains a significant bottleneck. This conversion process requires deep knowledge of experimental design, source data formats, and the NWB standard itself.

To address this challenge, we developed [NeuroConv](https://neuroconv.readthedocs.io/en/stable/index.html), a library that automates the ingestion and conversion of neurophysiology data from diverse formats into NWB. The development of NeuroConv required solving the following fundamental challenges:
* Handling the diversity of source formats and metadata diversity.
* Managing high-volume data efficiently, including datasets that exceed available RAM.
* Accommodating complex experimental setups with multiple simultaneous recordings.



### Handling Diverse Data Formats

The challenge of format diversity in neurophysiology extends beyond their sheer number. Many formats, such as Neuralynx, exist in multiple versions, while others, like TIFF, exhibit significant internal variability in how labs use them. NeuroConv addresses this complexity through a modular architecture built around DataInterface classes. Each supported format has a dedicated DataInterface that handles data and metadata extraction, with specialized implementations like SpikeGLXRecordingInterface for raw voltage recordings, PhySortingInterface for spike-sorted data, and DeepLabCutInterface for behavioral tracking data. The critical contribution is a common interface that abstracts the format internal details and allows the user to builds conversions in a consistent manner, regardless of its source format.

The central object of NeuroConv is the DataInterface. An interface is the medium between the source data format and the NWB file. It provides a unified API for extracting metadata, adding data to an NWB file, and running the conversion process. Each DataInterface is designed to handle a specific source format, encapsulating the logic required to read and convert that format's data while adhering to NWB best practices.

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

This is the core pattern of NeuroConv usage. The user initializes a DataInterface for a specific source format, extracts metadata, modifies it as needed, and then runs the conversion process to create an NWB file. This pattern abstracts away the complexities of each source format while providing a consistent interface for users.

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

Each of the examples contains basic code snippets that demonstrate how to use the DataInterface for a specific format, including how to extract metadata, modify it as needed, and run the conversion process to create an NWB file. This modular approach allows users to easily adapt the examples to their specific needs, while also providing a consistent interface for working with different data formats. For showcase, this is an example of converting amplifier data acquiried with Intan:

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

As NeuroConv's format support has expanded, we rely on [installation extras](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras)  to manage installation complexity. Users can specify only the formats they need during installation:

```python
pip install "neuroconv[spikeglx,phy,deeplabcut]"
```

This approach aggregates only the required dependencies for selected formats ensuring that the installation remains lightweight and manageable for end users.

### Handling Multi Stream Conversions

Neurophysiology experiments typically involve multiple simultaneous data streams from different modalities, such as raw electrophysiology recordings, spike-sorted data, and behavioral video. Each stream may be recorded in a different format, leading to complex conversion requirements. NeuroConv's architecture supports multi-stream conversions through the aggregation of DataInterfaces with a Converter.

The converter pattern allows to combine multiple DataInterface instances into a single conversion workflow. This enables users to convert all relevant data streams from an experiment into a single NWB file, ensuring that all data is properly aligned and associated with the correct metadata. The Converter class handles the orchestration of multiple DataInterfaces, allowing users to specify the order of operations and resolve any conflicts in metadata or data organization. 

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
Note that the pattern of a single interface is repeated here for a multi-stream conversion. The user initializes multiple DataInterfaces for each data stream, aggregates them into a ConverterPipe, extracts metadata, modifies it as needed, and then runs the conversion process to create an NWB file. This modular approach allows NeuroConv to handle complex experimental setups with multiple data streams while maintaining a consistent interface for users.

### Handling High-Volume Data

Modern acquisition systems, such as multi-probe Neuropixel recordings or whole-brain optical imaging, generate massive volumes of data that continue to grow year over year [@doi:10.1016/j.conb.2018.01.009][@doi:10.48550/arXiv.2201.03537]. These volumes of data pose a variety of challenges both for conversion and for long-term storage. Moreover, cloud computing is emerging as a solution  [@doi:10.48550/arXiv.1005.2718] [@doi:doi.org/10.1038/s41592-021-01326-w] for long term managing and storing the data in a way that is efficiently accessible for the scientific community is an important consideration.

A critical feature is the ability to process datasets larger than available RAM. NeuroConv inherits from the work performed by the NWB core group with [iterative writing](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/plot_iterative_write.html#sphx-glr-tutorials-advanced-io-plot-iterative-write-py) to stream data in manageable chunks, with configurable chunk sizes based on available resources. In NeuroConv we have extended this approach to support reading data in chunks from SpikeInterface enabling the buffered reading of a plethora of extracellular electrophysiology formats, such as SpikeGLX, Neuralynx, and Plexon. We have also implemented an iterative writing approach to roiextractors that allows buffered writing of large imaging datasets, such as those generated by whole-brain calcium imaging. Furthermore, we have implemented chunked solutions for other common formats with large data such as video. This approach enables processing of arbitrarily large files, successfully tested on 100+ GB files using computers with only 8 GB of RAM.

For storage optimization, NeuroConv leverages HDF5 and Zarr's support for chunked, compressed datasets. The current supported backends in NWB are HDF5 and Zarr. Compression algorithms represent a trade-off between storage space and access speed [@doi:10.1088/1741-2552/acf5a4]. Neuroconv exposes an easy-to-use API for configuring chunking and compression settings at the dataset level that allows for quick experimentation but at the same time we aim to include sensible defaults that just work for most users. 

Determining optimal chunk parameters presents complex tradeoff [@doi:10.1002/essoar.10511054.2][@nguyen2023impact] Large chunks minimize the number of read operations but may require decompressing unnecessary data. Small chunks provide more precise access but increase overhead, particularly for cloud storage where each chunk requires a separate range request. In general, appropriate chunking requires knowledge of the most common access patterns of the data. As Neurosphysilogy has to some degree common analysis and visualizations it is possible to implement heuristics for chunk size for the most common data types (such as voltage recordings and imaging data).


### Cloud Deployment

NeuroConv supports both local installation (Linux, Windows, or macOS) and [cloud deployment](https://neuroconv.readthedocs.io/en/stable/user_guide/aws_demo.html) through a maintained [Docker image](https://neuroconv.readthedocs.io/en/stable/user_guide/docker_demo.html) containing all dependencies. We've developed a YAML-based specification language for defining conversion pipelines, validated through JSON schema. This specification can fully describe multi-subject, multi-session conversions with custom metadata at each level, enabling automated conversion through containerized NeuroConv deployments. 

## Testing and Quality Assurance

Ensuring reliable conversion across diverse neurophysiology data formats requires a robust testing infrastructure. There are two pillars to this testing infrastructure:  an automated continuous integration (CI) pipeline that we have built using [github actions](https://github.com/features/actions) and a comprehensive test data library.

Our continuous integration pipeline, implemented through GitHub Actions, ensures code quality and maintains compatibility across operating systems. We follow standard software engineering practices for testing. As per usual standard practice, failed tests block pull request merging, maintaining code quality standards while facilitating rapid development. The pipeline runs on every pull request and includes the following key components: unit tests, which cover the basic internal core functionality of the library; integration tests on data which ensure that the library works as expected with real data; cross-platform testing to ensure that the library works on all supported operating systems; documentation build to ensure that the documentation is up-to-date; we leverage [doctest](https://docs.python.org/es/3.13/library/doctest.html) functionality to test that the aforementioned conversion gallery works with the current version of the code to avoid documentation and code drift; code style checks to ensure that the code is consistent and follows best practices; and test coverage to ensure that the code is well-tested and maintainable. The code coverage of NeuroConv stands at 90%, which is well above standard practices [@code_coverage_google][@doi:10.48550/arXiv.1907.01602].


The test data libraries contain a curated collection of example files spanning all supported data formats. At the moment, we divide our testing librari between [extracelullar electrophysilgy][https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/] in collaboration with the NEO and SpikeInterface development teams, [optical physiology][https://gin.g-node.org/CatalystNeuro/ophys_testing_data] and [behavior](https://gin.g-node.org/CatalystNeuro/behavior_testing_data). These files are selected to cover common usage patterns for each format such as different format versions, edge cases, missing streams or experimental configurations. The test data is hosted in a public version control system ([G-Node](https://gin.g-node.org)) and uses git-annex technology to efficiently manage large files. This approach allows us to maintain a lightweight repository while still providing access to the full set of test data files. The test data library is automatically downloaded during the CI pipeline and cached using github actions ensuring that tests run with the most up-to-date and relevant data.


## Community and Ecosystem

Converting to NWB

:::{figure} assets/conversion_comparisons.png
:label: fig:assets/conversion_comparisons
This illustrates where does neuroconv stands in regards to other conversion tools. For low level high precision control you can use the NWB APIs directly (pynwb in Python, matNWB in MATLAB), for a guided GUI based experience the NWB GUIDE is the best option but might be too rigid. Neuroconv stands as a middle ground automatizing the conversion of a large number of formats while still allowing for customization and flexibility.
:::

[add image here] and states how neuroconv stands in regard to the APIs n

Neuroconv and the [Nwb GUIDE](https://nwb-guide.readthedocs.io/en/stable/) .

Neuroconv and the core language APIs

Visualizing with neurosift
Neurosift [@doi:/10.21105/joss.06590]

Upload the data to the archive.

_[Additional content needed: Integration with other Python scientific tools]_
Neuroconv and SpikeInterface 


_[Additional content needed: Contribution guidelines and community involvement]_
_[Additional content needed: Educational resources and documentation]_
Neuroconv documentation.


## Current Limitations and Future Work

While NeuroConv has significantly improved data standardization processes, some challenges remain:

- Format Coverage: Despite supporting 44 formats, new acquisition systems and format versions continually emerge. While users can develop custom DataInterfaces, these require understanding both the source format and NeuroConv's architecture.
- Custom Lab Formats: Many labs store data in custom formats, often as MATLAB .mat files or custom csv files. These formats tend to be highly variable and rapidly evolving, making automated conversion challenging. NeuroConv works best with data in its original acquisition format or standardized processing output.
- Programming Prerequisites: While NeuroConv substantially reduces the coding burden, it still requires basic programming knowledge, including object-oriented concepts. Some features, like temporal alignment, may require advanced numerical computing skills.

_[Additional content needed: Roadmap for future development]_
* Improve the user experience by providing more comprehensive examples, tutorials, and documentation to help users understand how to use NeuroConv effectively. We are slowly but surely moving towards a diataxis [@diataxis] structure were...
* Keep in line with the latest developments of the schema and the NWB standard. The NWB standard has adopted a mechanism for enchacments to the schema the [NWB Extensions Proposals](https://github.com/nwb-extensions/nwbep-review/). At the moment, there is progress on improving the schema description for descriving events, NWBEP00, in experiments, extracellular electrophysiology NWBEP002,  optical physiolog, NWBEP003 and NWBEP004 and . The developers of neuroconv are actively participating in the NWB Extensions Proposals and we aim to support the improved standrads as soon as they are accepeted. This ensure to our users that the data they convert with neuroconv is always up to date with the latest NWB standards.
* Leverage the latest developments in LLMs's to fullfil the library core mission: automating the conversion of neurophysiology data to NWB. This includes using LLMs to generate DataInterfaces from source format documentation, automatically extracting metadata, and even generating custom conversion pipelines based on user requirements.
* Improve the chunking patterns for larger data files to improve cloud access performance. This includes experimentation to determine optimal chunk sizes and compression algorithms. The goal is to implement the best knowldged availalble [nwb benchmarks project](https://nwb-benchmarks.readthedocs.io/en/latest/) and implement heuristics that ensure that the data is stored in a way that is efficient and performant [@doi:10.48550/arXiv.1601.07028]

 
## Closing Remarks

_[Additional content needed: Summary of key contributions]_

_[Additional content needed: Broader implications for scientific software development]_



Our effort is built on the work of other packages in the neuroscience scientific community. Specifcally, for extracellular electrophysiology we leverage [NEO](https://neo.readthedocs.io/en/latest/) [@neo] through [SpikeInterface](https://spikeinterface.readthedocs.io/en/stable/) [@spikeinterface] for raw extracellular electrophysiology and spike-sorted data. For optical imaging, we have developed and maintain [roiextractors](https://roiextractors.readthedocs.io/en/latest/index.html), which provides a unified interface for both raw imaging data and the output of popular processing pipelines like suite2p and CaImAn. Here we also relying strongly on other packagest like the [tifffile](https://github.com/cgohlke/tifffile/) [@doi:10.5281/zenodo.6795860] python library . Behavior, being more heterogenous requires a more scattered approach, for handling audio we relying on scipy [@scipy] and the python standard library, for video we use [opencv](https://opencv.org/) whicn in turns uses [ffmpeg](https://ffmpeg.org/). 

[pymatreader](https://pymatreader.readthedocs.io/en/latest/) 


_[Additional content needed: Call to action for community involvement]_

## References

_[References will be automatically generated from citations in the text and the mybib.bib file]_
