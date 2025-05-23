---
# Ensure that this title is the same as the one in `myst.yml`
title: NeuroConv. Streamlining Neurophysiology Data Conversion to the NWB Standard
abstract: |
  Converting diverse neurophysiology data to the standardized Neurodata Without Borders (NWB) format remains a significant barrier to data sharing and reuse. We present an integrated software suite that streamlines NWB adoption through three tools: NWB Inspector for automated quality control, NeuroConv for converting 44 distinct data formats while handling high-volume data, and NWB GUIDE for code-free data conversion and publication. This suite has enabled the standardization of over 300 datasets totaling more than 300 TB in the DANDI archive and has been deployed to create automated conversion pipelines for 40 laboratories with unique experimental paradigms. By reducing technical barriers to NWB adoption, our tools accelerate progress toward reproducible neuroscience research through standardized data sharing.
---

## Introduction

Modern neurophysiology research generates increasingly complex, multimodal, and large-scale datasets that demand robust standards for organization, sharing, and archival. The Neurodata Without Borders (NWB) format has emerged as a comprehensive solution, storing contextual metadata alongside primary data objects to ensure long-term interpretability. A key strength of NWB is its ability to unify diverse experimental modalities—such as simultaneous behavioral and electrophysiology recordings—that traditionally exist in scattered files and formats. By consolidating all relevant information about an experimental session, NWB enables robust reanalysis and reproducibility.

The Distributed Archives for Neurophysiology Data Integration (DANDI) platform complements NWB by providing free hosting for NWB-formatted datasets up to terabytes in size. DANDI offers researchers a pathway to meet NIH data sharing requirements while effectively archiving their data and leveraging an expanding ecosystem of visualization and analysis tools. Together, NWB and DANDI create an ideal foundation for emerging software domains, from electrophysiological spike sorting to calcium imaging segmentation and behavioral pose estimation.

Despite these advantages, several significant challenges impede widespread NWB adoption. First, researchers must develop at least a basic understanding of NWB's data organization principles—a daunting prospect given the format's comprehensive scope. This learning curve presents a substantial barrier to integrating NWB standards into daily laboratory practices. Additionally, utilizing NWB's data Application Programming Interfaces (APIs), such as pyNWB and MatNWB, requires programming expertise that may exceed the capabilities of many researchers. This technical barrier often restricts NWB adoption to those with coding proficiency or places an outsized burden on junior lab members who possess the necessary skills.

Labs frequently face two distinct conversion challenges: processing existing backlogged data and establishing automated pipelines for future data collection. The latter is particularly crucial for minimizing duplicated work, as labs aim to automatically convert newly generated data into NWB format. However, developing robust conversion pipelines presents significant challenges due to the diversity of neurophysiology data formats. A single lab may employ multiple modalities—voltage recording, optical imaging, optogenetics, and behavioral tracking—each with its own software-dependent formats lacking standardization.

_[Additional content needed: Brief history of neurophysiology data standardization efforts and specific examples of format diversity problems]_

## Background and Motivation

Neurophysiology research generates vast amounts of data across diverse formats, from electrophysiology recordings to calcium imaging and behavior tracking. The NWB format has emerged as a community standard for storing and sharing neurophysiology data, promoting reproducibility and collaboration in neuroscience. However, converting existing datasets to NWB presents significant challenges:

- Source data formats are highly diverse (proprietary and open)
- Metadata requirements vary substantially
- Dataset sizes often reach hundreds of gigabytes to terabytes
- Following NWB best practices requires considerable expertise

_[Additional content needed: Comparison with other standardization efforts in scientific computing and technical details about NWB format structure]_

_[Additional content needed: Examples of typical lab workflows before/after standardization with diagrams]_

## NeuroConv Architecture and Design

Modern neurophysiology research faces a significant data standardization challenge. The field relies on dozens of acquisition systems, each typically recording data in its own proprietary format. These formats prioritize different aspects of data handling—some optimize for write speed during acquisition, others for storage efficiency or cross-platform compatibility. This diversity creates a complex landscape where formats vary widely in their efficiency, support longevity, metadata richness, and cross-platform compatibility. While NWB has emerged as a unifying standard that addresses many common pitfalls of proprietary formats, converting data to NWB remains a significant bottleneck. This conversion process requires deep knowledge of experimental design, source data formats, and the NWB standard itself.

To address this challenge, we developed NeuroConv, a library that automates the ingestion and conversion of neurophysiology data from diverse formats into NWB. The development of NeuroConv required solving three fundamental challenges: handling the diversity of source formats, managing high-volume data efficiently, and accommodating complex experimental setups with multiple simultaneous recordings.

### Handling Diverse Data Formats

The challenge of format diversity in neurophysiology extends beyond their sheer number. Many formats, such as Neuralynx, exist in multiple versions, while others, like TIFF, exhibit significant internal variability in how labs use them. NeuroConv addresses this complexity through a modular architecture built around DataInterface classes. Each supported format has a dedicated DataInterface that handles data and metadata extraction, with specialized implementations like SpikeGLXRecordingInterface for raw voltage recordings, PhySortingInterface for spike-sorted data, and DeepLabCutInterface for behavioral tracking data. 

Currently supporting 44 distinct input formats (Table 1), each DataInterface is comprehensively documented and demonstrated in the Conversion Gallery, where users can find complete examples requiring only ~5 lines of code to perform full data conversion. Throughout the conversion process, NeuroConv enforces NWB Best Practices and optimizes data storage for both archival purposes and cloud computing requirements.

_[Table 1: List of supported data formats - to be added]_

To maximize reliability and maintainability, NeuroConv builds upon established domain-specific libraries that provide unified APIs across multiple formats. It leverages NEO for raw extracellular electrophysiology data and SpikeInterface for spike-sorted data. For optical imaging, we developed and maintain roiextractors, which provides a unified interface for both raw imaging data and the output of popular processing pipelines like suite2p and CaImAn. The framework also incorporates specialized libraries for individual formats, such as pyEDF for the European Data Format, ensuring robust and well-tested data handling across the neurophysiology ecosystem.

As NeuroConv's format support has expanded, we've implemented a modular dependency system to manage installation complexity. Users can specify only the formats they need during installation:

```python
pip install neuroconv[spikeglx,phy,deeplabcut]
```

This approach aggregates only the required dependencies for selected formats. For users needing comprehensive format support, we maintain full installation options for all major operating systems:

```python
pip install neuroconv[full]
```

_[Additional content needed: UML or architectural diagrams showing the component relationships]_

_[Additional content needed: More detailed code examples showing the API usage]_

## Technical Solutions

### Handling High-Volume Data

Modern acquisition systems, such as multi-probe Neuropixel recordings or whole-brain optical imaging, generate massive volumes of data that continue to grow year over year. NeuroConv implements advanced data engineering features across its DataInterface classes to handle these large-scale datasets efficiently.

A critical feature is the ability to process datasets larger than available RAM. NeuroConv employs DataChunkIterator to stream data in manageable chunks, with configurable chunk sizes based on available resources. This approach enables processing of arbitrarily large files, successfully tested on 100+ GB files using computers with only 8 GB of RAM.

For storage optimization, NeuroConv leverages HDF5 and Zarr's support for chunked, compressed datasets. Rather than applying compression to entire files, which would require full decompression for any data access, NeuroConv implements intelligent chunking strategies. Data is divided into pieces that are individually compressed, enabling direct access to specific data regions by decompressing only the relevant chunks. This technique has achieved up to 50% compression while maintaining sufficient read performance for interactive data visualization.

Determining optimal chunk parameters presents complex tradeoffs. Large chunks minimize the number of read operations but may require decompressing unnecessary data. Small chunks provide more precise access but increase overhead, particularly for cloud storage where each chunk requires a separate range request. NeuroConv carefully balances these factors and considers access patterns—for example, avoiding time-dimension chunking that would require reading entire temporal segments when accessing individual channels.

_[Additional content needed: Specific performance metrics and benchmarks comparing different chunking strategies]_

### Handling Complex Experiments

Neurophysiology experiments typically involve multiple simultaneous data streams that require careful temporal alignment. A typical navigational study might combine:
- Raw extracellular electrophysiology
- Processed spike-sorted data
- Raw behavioral video
- Processed behavioral metrics

NeuroConv provides a NWBConverter class to aggregate multiple DataInterface instances into a comprehensive conversion pipeline. This class can handle both raw acquisition data and processed outputs, with features for resolving metadata conflicts and specifying priority between different data sources.

A key challenge is synchronizing data streams from independent acquisition systems. NWB requires all timestamps to reference a common clock, though sampling rates can vary. NeuroConv supports three common synchronization strategies:

1. Starting time offset: A TTL pulse marks the start of secondary recording, establishing a simple offset. While straightforward, this approach doesn't account for clock drift between systems.
2. Timestamps signal: TTL pulses sent at each sample provide precise temporal alignment. This approach handles both offset and drift but may be impractical for high sampling rates.
3. Synchronization signal: Periodic or irregular TTL pulses span multiple samples, enabling timestamp adjustment through interpolation.

The NWBConverter class provides methods for implementing these strategies and supports custom synchronization approaches.

_[Additional content needed: Detailed algorithm descriptions for synchronization with examples]_

### Cloud Deployment

NeuroConv supports both local installation (Linux, Windows, or macOS) and cloud deployment through a maintained Docker image containing all dependencies. We've developed a YAML-based specification language for defining conversion pipelines, validated through JSON schema. This specification can fully describe multi-subject, multi-session conversions with custom metadata at each level, enabling automated conversion through containerized NeuroConv deployments.

_[Additional content needed: Example YAML configuration and deployment workflow]_

## Testing and Quality Assurance

Ensuring reliable conversion across diverse neurophysiology data formats requires a robust testing infrastructure. We've developed two key components to address this challenge: a comprehensive test data library and an automated continuous integration pipeline.

The test data libraries contain a curated collection of example files spanning all supported data formats in collaboration with the NEO and SpikeInterface development teams. These files are carefully selected to cover common usage patterns and edge cases for each format. For proprietary formats, we've created minimal test files that capture essential format features while respecting licensing constraints. The library serves multiple purposes: it enables thorough testing of conversion functionality, provides reference data for debugging user issues, and offers concrete examples for documentation. 

Our continuous integration pipeline, implemented through GitHub Actions, ensures code quality and maintains compatibility across operating systems. The pipeline runs on every pull request and includes:
- Unit Tests: Verify individual components and format-specific DataInterface implementations
- Integration Tests: Check end-to-end conversion workflows and interactions between components
- Cross-Platform Testing: Validate functionality on Linux, Windows, and macOS
- Documentation Build: Ensure documentation remains current and examples are functional
- Type Checking: Enforce static type annotations
- Code Style: Maintain consistent formatting through black and flake8
- Test Coverage: Track code coverage using codecov

The pipeline employs caching strategies to optimize performance, particularly for handling the test data library. We've implemented a matrix testing approach that validates different Python versions and optional dependencies combinations, ensuring reliable operation across diverse user environments. Failed tests block pull request merging, maintaining code quality standards while facilitating rapid development.

_[Additional content needed: Test coverage statistics and examples of caught bugs/regressions]_

_[Additional content needed: Performance metrics of CI pipeline (build times, etc.)]_

## Impact and Adoption

The impact of standardization tools on the neuroscience community has been substantial. Through CatalystNeuro's consulting work, we have implemented automated NWB conversion pipelines for 40 laboratories, each with unique experimental paradigms and data formats. These tools have contributed to the growth of the DANDI archive, which now hosts over 300 NWB datasets comprising more than 300 TB of standardized neurophysiology data. This rapid adoption demonstrates both the effectiveness of our toolchain and the neuroscience community's pressing need for standardized data sharing solutions.

_[Additional content needed: User testimonials or case studies]_

_[Additional content needed: Before/after conversion time comparisons]_

_[Additional content needed: Community contribution statistics (GitHub stars, contributors, etc.)]_

_[Additional content needed: Specific examples of enabled research]_

## Current Limitations and Future Work

While NeuroConv has significantly improved data standardization processes, some challenges remain:

- Format Coverage: Despite supporting 44 formats, new acquisition systems and format versions continually emerge. While users can develop custom DataInterfaces, these require understanding both the source format and NeuroConv's architecture.
- Custom Lab Formats: Many labs store data in custom formats, often as MATLAB .mat files. These formats tend to be highly variable and rapidly evolving, making automated conversion challenging. NeuroConv works best with data in its original acquisition format or standardized processing output.
- Programming Prerequisites: While NeuroConv substantially reduces the coding burden, it still requires basic programming knowledge, including object-oriented concepts. Some features, like temporal alignment, may require advanced numerical computing skills.

_[Additional content needed: Roadmap for future development]_

_[Additional content needed: Integration plans with other tools]_

_[Additional content needed: Community feedback and requested features]_

_[Additional content needed: Technical debt and refactoring plans]_

## Code Examples and Usage

_[Additional content needed: Step-by-step conversion examples]_

_[Additional content needed: Common use case demonstrations]_

_[Additional content needed: Integration with Jupyter notebooks]_

## Performance Analysis

_[Additional content needed: Benchmarks across different data sizes]_

_[Additional content needed: Memory usage profiles]_

_[Additional content needed: Conversion time comparisons]_

## Community and Ecosystem

_[Additional content needed: Integration with other Python scientific tools]_

_[Additional content needed: Contribution guidelines and community involvement]_

_[Additional content needed: Educational resources and documentation]_

## Conclusion

_[Additional content needed: Summary of key contributions]_

_[Additional content needed: Broader implications for scientific software development]_

_[Additional content needed: Call to action for community involvement]_

## References

_[References will be automatically generated from citations in the text and the mybib.bib file]_
