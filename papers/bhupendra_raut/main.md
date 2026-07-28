---
title: "Adapt: Prototyping a Real-Time, Reproducible Data Analysis Framework for Adaptive Radar Scanning"
---

+++ {"part": "abstract"}
*Adapt* is a modular framework for real-time convective cell detection, motion projection, and storm lifecycle analysis, developed to support adaptive radar-scanning operations at the U.S. Department of Energy Atmospheric Radiation Measurement (DOE-ARM) User Facility. Scientific workflows are composed of independently registered modules that communicate through a shared execution context and are assembled at runtime into directed acyclic graphs (DAGs) through automatic dependency resolution. The project aims to enable plug-and-play modules and automated pipeline construction from declared dependencies.
A reference NEXRAD pipeline demonstrates the architecture through data acquisition, detection, analysis, projection and tracking.
A read-only data-access layer provides unified SQL and NetCDF access to pipeline outputs for real-time and archival analysis. Adapt is being developed toward a reusable ecosystem of scientific and machine-learning modules that can be composed into operational and research pipelines. Planned applications include cloud-based nowcasting dashboards and JupyterHub workflows on DOE-ARM computing infrastructure. We welcome contributions in architecture, validation, software design, and governance to strengthen scalability and maintainability. As AI-assisted development becomes increasingly prevalent, a central objective is to build an architecture that is explicit, verifiable, and resilient while supporting both human developers and coding agents.
+++

(sec-intro)=
## Introduction

Adaptive radar scanning enables real-time response to evolving convective storms by focusing radar on dynamically selected sectors [@oue2022optimizing]. Traditional scanning strategies must balance spatial coverage, temporal frequency, and vertical resolution under fundamental physics constraints, frequently undersampling rapid microphysical evolution during convective lifecycles. Real-time feedback control offers a pathway to dramatically improve observations of transient phenomena but demands low-latency analysis, reliable decision support, and infrastructure that remains stable under continuous operation [@lamer2023multisensor; @gupta2025cloud].

The TITAN system demonstrated operational radar data analysis in real time as early as the 1990s [@dixon1993titan]. Modern open-source frameworks such as PyART, tobac, TINT, and CoCo-MET provide powerful scientific algorithms primarily designed for offline batch workflows [@helmus2016python; @sokolowsky2024tobac; @raut2021adaptive; @hahn2025cocomet]. Composing them into a reliable, continuously running, operationally maintainable pipeline introduces requirements that go beyond scientific algorithms: the need is continuous data ingestion, event-driven execution, configurable processing chains without code changes, separation of consumers from producers, and long-term reproducibility.

*Adapt* aims to address these requirements through a layered architecture in which scientific modules are registered, wired by dependency, and executed by a graph-based orchestrator. The current implementation processes NEXRAD Level-II data in real time or archival mode, writing validated outputs to a structured repository that external consumers access through a read-only API.

This paper presents the design of Adapt for scientific developers seeking to extend the framework with new modules, pipelines, or data consumers. {ref}`sec-design` introduces the layered architecture, followed by the module system ({ref}`sec-modules`), execution graph ({ref}`sec-graph`), configuration system ({ref}`sec-config`), and data repository and consumer API ({ref}`sec-repo`). {ref}`sec-nexrad` presents the reference NEXRAD pipeline, {ref}`sec-stack` summarizes the scientific Python software stack, and {ref}`sec-roadmap` outlines future development and contribution opportunities.

(sec-design)=
## System Design

Adapt is structured in layers, each with a single, well-defined responsibility. Layers communicate through explicit interfaces as shown in @fig-flow.

:::{figure} flow.png
:label: fig-flow
:width: 90%
Adapt execution workflow. A two-thread runtime (downloader and processor) drives a DAG-based execution graph. Scientific modules are registered plugins that communicate through a shared context dictionary. All outputs are written to a structured repository accessed through a read-only consumer API.
:::

### Layer Responsibilities

**Module layer** contains all scientific algorithms. Modules are stateless functions wrapped in a standard interface and they do not have I/O, threading, or repository access. Each module declares the named data products it reads from and writes to a shared context dictionary. The graph executor wires and sequences them automatically.

**Execution layer** hosts `GraphBuilder`, which constructs a DAG from module input/output declarations, and `GraphExecutor`, which traverses the graph in topological order, enforcing contracts at every module boundary. The execution layer is the only layer that directly instantiates modules. It does so through the module registry, not by importing module classes.

**Runtime layer** coordinates threads and data flow. It starts and stops the downloader and processor threads, owns the inter-thread queue, manages the two-frame rolling buffer required by the projection module, and delegates each file to the pipeline. No scientific logic resides in the runtime layer.

**Persistence layer** writes and indexes all artifacts to the local data repository. Gridded fields are stored as NetCDF and tabular analysis outputs as Parquet while tracking data and metadata catalogs as SQLite. All file paths are encapsulated within this layer and no other layer constructs paths or directory structure directly.

**Configuration layer** resolves a three-tier hierarchy into a single immutable `InternalConfig` object at startup. No component reads configuration files at runtime.

**Consumer API layer** provides a read-only API (`DataClient`) that wraps catalog queries, DuckDB-powered Parquet SQL, and a polling-based streaming interface. Dashboards, notebooks, and autonomous pipelines can use only this layer and they have no access to the pipeline's internal state.

### Dependency Rules

The import rules are enforced by `import-linter` and checked in CI, so a layering violation fails the build even when every unit test passes.

(sec-modules)=
## Plug-and-Play Module System

The module system will be the primary extension point for scientific developers. New modules can be integrated and tested without modifying any existing source files.

### Module Interface

Every processing module subclasses `BaseModule` and declares four class-level attributes:

```{code-block} python
:caption: Minimal module definition.
class MyModule(BaseModule):
    name = "my_module"
    inputs = ["grid_ds_2d", "config"]
    outputs = ["my_product"]

    def run(self, context: dict) -> dict:
        data = context["grid_ds_2d"]
        result = compute(data)
        return {"my_product": result}
```

Optionally, a module may declare `input_contracts` and `output_contracts` as dictionaries mapping context keys to validator functions and also their version. The graph executor enforces these contracts automatically. The module author does not call input/output validation logic.

### Registration

Modules register with a global `ModuleRegistry` when their package is imported. Registration records the module class by its `name` attribute. The pipeline configuration specifies which module packages to import, the registry instantiates them, and the graph builder wires them automatically. Adding a new module to a pipeline requires:

1. Implement `BaseModule` in `modules/<domain>/module.py`
2. Currently requires adding the package path to the `pipeline.modules` list in the pipeline's configuration file
3. Optionally declare contracts for boundary validation. These implementations will be made mandatory in the stable release.

No other source files should require modification for adding a module. Replacing an existing module. For example, substituting a deep-learning segmenter for the threshold-based detector, requires only that the replacement produce the same output context key (`segmented_ds`) satisfying the same output contract.

### Scaling to Many Modules

Because modules communicate only through context keys and are wired by the graph builder from their declarations, a large number of modules can coexist in the registry without interfering. Pipeline configurations select which registered modules to activate, and inactive modules have zero runtime cost.

@tbl-modules lists the modules in the current NEXRAD reference pipeline. New module categories currently under consideration include: wavelet-based cell detection, lightning data ingestion, GOES satellite ingest, nowcasting tendency modules, and alert-generation modules for automated scan-control decisions.

(sec-graph)=
## Directed Acyclic Graph Execution

### Graph Construction

`GraphBuilder` constructs the execution DAG during pipeline initialisation. It creates a node for each registered module, builds an output map from product names to producing nodes, and wires dependencies by matching each module's declared inputs to entries in the output map. Two checks are enforced at construction time: (1) all declared inputs must have a corresponding producer, and (2) no two modules may declare the same output key. Violations cause startup failure before any processing begins.

### Execution and Context Dictionary

`GraphExecutor` traverses the graph in topological order within a single thread. At each step it selects nodes whose upstream dependencies have completed, invokes the module, and merges returned outputs into the shared execution context dictionary. The context dictionary holds all in-memory data products for one file's execution and is discarded after the file completes.

Input contracts are enforced before each module invocation and output contracts after the module finishes processing. A `ContractViolation` halts processing for the current file and records a diagnostic failure state in the catalog; processing continues with subsequent files.

### Multiple Pipelines

A pipeline is a named configuration that specifies which modules to activate and in what parameter regime. Because module wiring is derived automatically from declarations, a new pipeline definition currently requires only a configuration file listing the desired modules. The same module can participate in multiple pipelines. Currently, the NEXRAD pipeline is the only implementation and we aim to provide a framework for additional pipelines (e.g., for satellite data, multi-sensor fusion, or model data).

This design can scale naturally to tens of pipelines. Users select a pipeline by name via CLI flag or configuration file.

(sec-config)=
## Three-Tier Configuration

Pipeline behaviour is controlled through a layered configuration hierarchy. The lowest tier is `ParamConfig`, encoding defaults for every tunable parameter, maintained in a `defaults.yaml`. Users must not edit this file. The middle tier is `UserConfig`, a config file containing only what they wish to override in `ParamConfig`. The highest tier is `CLIConfig`, populated from command-line flags.

At startup, `resolve_config()` merges the three tiers in precedence order (CLI overrides user, user overrides param) and validates the result as a frozen `InternalConfig` object. Unknown fields in user files raise a `ValidationError` immediately. The resolved configuration is stored alongside pipeline outputs, creating a permanent record sufficient to reproduce any run.

This design separates scientific expertise (defaults), deployment preferences (user config), and operational flags (CLI) into independent, composable concerns. A shared configuration file captures pipeline settings and algorithm parameters for reproducibility.

A user config lists the modules to run and overrides only the defaults it cares about; every unspecified parameter falls back to `defaults.yaml`.

```{code-block} yaml
:caption: Excerpt of a user `config.yaml`. Only overrides are listed; all other parameters inherit from `defaults.yaml`.
base_dir: /path/to/repository
mode: historical
source: aws_nexrad
modules:            # dependency order; subset at runtime with --only / --not
  - ingest
  - detection
  - projection
  - analysis
  - tracking
  - cell_volume_stats
regridder:
  grid_shape: [41, 301, 301]
  grid_limits: [[0, 20000], [-150000, 150000], [-150000, 150000]]
  weighting_function: cressman
  save_netcdf: true
```

The pipeline is generated and run from the command line. `adapt config` writes a template, and `run-nexrad` processes a small historical window for one radar:

```{code-block} bash
:caption: Generating a config template and running the NEXRAD pipeline on a historical case.
adapt config                                # write a config.yaml template
adapt run-nexrad config.yaml --radar KHTX \
    --mode historical \
    --start-time 2026-07-24T12:00 --end-time 2026-07-24T13:00
```

CLI flags override the config file, so a single shared config can be reused across runs while `--radar`, the time window, `--only`/`--not` (to activate a subset of modules), and `--max-runtime` (realtime mode) vary per invocation.

(sec-contracts)=
## Contract-Based Validation

Scientific pipelines are susceptible to silent failures when intermediate data products change structure. Adapt addresses this through a contract system that validates data products at module boundaries. Contracts are pure functions in module `contracts.py` files that verify invariants such as: variable presence, array dimensionality, integer label integrity, coordinate consistency, and DataFrame schema completeness.

Each module may declare optional `input_contracts` and `output_contracts` dictionaries that map a context key to a `(validator, version)` pair. The graph executor calls these automatically and the module author does not invoke validation logic. For example, the detection module guarantees that its `segmented_ds` output satisfies the segmentation contract:

```{code-block} python
:caption: A module declares contracts by name; the executor enforces them at the boundary.
from adapt.modules.detection.contracts import check_segmentation

class Detection(BaseModule):
    name = "detection"
    inputs = ["grid_ds_2d", "config"]
    outputs = ["segmented_ds"]
    output_contracts = {"segmented_ds": (check_segmentation, "v1")}
```

Here `check_segmentation` is a pure function in `detection/contracts.py` that verifies the segmentation invariants (label variable present, integer dtype, non-negative labels, 2-D shape). Because the guarantee is attached to the `segmented_ds` key rather than to the detector implementation, a deep-learning segmenter can replace the threshold-based detector as long as it satisfies the same output contract. Failures raise `ContractViolation` with a message identifying which contract failed, what was wrong, and which module produced the data. Contract coverage can be introduced incrementally as module interfaces stabilise.

Contracts currently do not validate scientific correctness or statistical properties. As these are monitoring concerns, not interface contracts, checking them can be added as separate layers around the science modules, considering legitimate edge cases such as unusual storm morphologies or degraded radar modes.

(sec-repo)=
## Data Repository and Consumer API

### Repository Structure

The repository is a structured directory tree combining SQLite catalog tables with per-scan NetCDF and Parquet files, organised under a fixed layout. All artifacts are written through `DataRepository`, the single component that owns this layout, rather than by modules opening files or constructing paths directly. This indirection keeps producers agnostic to storage details and lets the consumer API resolve and read only the fields a query needs. Migrating the underlying store (for example, to a Zarr or Icechunk backend) therefore requires no changes to the science modules. Gridded outputs (radar volumes and segmentation) are stored as NetCDF [@rew1990netcdf] files per scan with zlib compression. Per-cell statistics are stored as Parquet files for efficient columnar queries. Cell tracking data is stored in per-radar SQLite tables to support row-oriented graph traversal queries (single storm history, recent events, lineage enumeration). We are currently exploring cloud-native data-store strategies such as Zarr, a chunked format for large N-dimensional arrays [@moore2023zarr], and Icechunk, a transactional storage engine layered on Zarr, for flexible concurrent read-write and long-term cloud storage capabilities. The repository layout and its access APIs are still stabilising ahead of the v1.0 release, and consumers should expect breaking changes until then.

### Reproducibility and Run Management

Each processing run receives a unique identifier of the form `YYYYMONDD-HHMM-RADAR`. Cell identifiers (`cell_uid`) are deterministic hashes derived from observable properties at initiation time. Hence the same input always produces the same identifiers regardless of when or how many times processing is performed. Reprocessing historical datasets is identical to real-time operation.

### Consumer API

`DataClient` is the sole interface through which downstream systems access pipeline outputs. No consumer has access to `DataRepository`, `RadarCatalog`, or any pipeline internal. The interface provides:

```{code-block} python
:caption: Consumer API usage.
client = DataClient("/path/to/output")

# Batch queries via DuckDB over Parquet
df = client.query(
    "SELECT * FROM analysis "
    "WHERE cell_area_sqkm > 50"
)

# Streaming for live dashboards
for df in client.stream(
    "SELECT * FROM cells_by_scan "
    "WHERE time > :last_seen",
    poll_interval=5
):
    update_dashboard(df)
```

DuckDB executes SQL directly against Parquet files with predicate and projection pushdown, making queries over multi-month archives feasible without loading data into memory. The streaming interface yields new rows as scans complete, enabling live dashboards to update incrementally without filesystem access or pipeline coupling. This producer/consumer separation is fundamental to the architecture.

(sec-nexrad)=
## NEXRAD Pipeline

The NEXRAD pipeline demonstrates the full architecture end-to-end. It processes NEXRAD Level-II data containing reflectivity, radial velocity, and spectrum width for each radar volume scan [@noaa1991nexrad], through six registered modules (shown in @tbl-modules) and exposes results through `DataClient`. The Level-II archive is publicly hosted on AWS S3 as an Open Data registry dataset (`s3://noaa-nexrad-level2`)[^nexrad-s3].

[^nexrad-s3]: <https://registry.opendata.aws/noaa-nexrad/>

### Two-Thread Runtime

The pipeline operates as two cooperating threads. `AwsNexradDownloader` continuously polls S3 for new Level-II files (realtime mode) or enumerates a specified time range and exits (historical mode). Files are placed in a bounded queue providing back-pressure when the processor is slower than the downloader. `RadarProcessor` dequeues files, checks for duplicates, maintains a two-frame rolling buffer for the projection module, executes the DAG via `NexradPipeline`, and writes results through `RepositoryWriter`. No shared mutable state exists between threads beyond the queue.

```{list-table} Modules in the NEXRAD reference pipeline, in dependency order. Each module reads and writes named products in the shared context; the graph builder derives the execution order from these declarations.
:label: tbl-modules
:header-rows: 1
* - Module
  - Inputs → Outputs
  - Function and Data Flow
* - `ingest`
  - Level-II file → gridded volume
  - Downloads Level-II files from AWS S3 to a background queue, decodes them with PyART, and regrids to a Cartesian volume.
* - `detection`
  - gridded volume → labeled cells
  - Segments 2D grids into labeled storm-cell datasets (`segmented_ds`) via thresholding and connected-component labeling.
* - `projection`
  - current + previous labeled cells → projected boundaries
  - Uses Farnebäck optical flow on current and past segments to project cell boundaries forward.
* - `analysis`
  - labeled cells + projected boundaries → per-cell statistics
  - Combines grids and projected boundaries into 2D per-cell statistics.
* - `tracking`
  - per-cell statistics → tracking graph
  - Links cells across scans, generating tracking graphs and lifecycle (split/merge) events.
* - `cell_volume_stats`
  - gridded volume + labeled cells → 3D cell statistics
  - Computes 3D volume statistics (e.g. cloud-top height) per cell; requires `regridder.save_netcdf`.
```

### Runtime Performance

@tbl-benchmark reports per-module timings from a sample realtime run over the KHTX radar (`adapt run-nexrad config.yaml --radar KHTX`), processing six volume scans (about 10 MB per Level-II file) and detecting 343 cell objects on an Apple M3 Mac with 36 GB RAM. End-to-end processing averaged 19.0 s per scan (max 20.04 s), dominated by ingestion (download, decode, and regridding). NEXRAD volume scans complete roughly every four to six minutes depending on the operational scan strategy, so a per-scan cost of about 19 s leaves ample headroom for real-time operation.

```{list-table} Per-module timings for a six-scan KHTX sample run (343 objects detected).
:label: tbl-benchmark
:header-rows: 1
* - Module
  - Calls
  - Total (s)
  - Avg (s)
* - `ingest`
  - 6
  - 79.0
  - 13.2
* - `detection`
  - 6
  - 14.2
  - 2.4
* - `projection`
  - 5
  - 18.8
  - 3.8
* - `analysis`
  - 5
  - 0.71
  - 0.14
* - `tracking`
  - 5
  - 1.1
  - 0.22
* - `cell_volume_stats`
  - 5
  - 21.0
  - 4.2
```

(sec-stack)=
## Scientific Python Stack

**Xarray.** All gridded data products are `xr.Dataset` objects. Spatial dimensions $(z, y, x)$, coordinate information, physical units, and timestamps travel with the data through every module. No function in the pipeline accepts grid spacing, CRS, or coordinate names as arguments; this information must be embedded in the dataset [@hoyer2017xarray].

**PyART.** NEXRAD Level-II decoding and Cartesian gridding are currently performed through PyART [@helmus2016python]. PyART is the core radar library because it supplies a broad range of radar algorithms except tracking (e.g. I/O, gridding, and classification). Adapt's own detection uses a tobac-like segmentation, and its tracking is an improved variant of TINT [@raut2021adaptive] adapted for real-time operation. At present, PyART usage is largely isolated to the ingestion and gridding stages. However, PyART is expected to become a foundational dependency within Adapt, with future modules exposing PyART-based radar analysis algorithms directly through configuration. This approach will allow users to construct processing pipelines from a broad range of PyART capabilities without modifying framework code.


**Pydantic.** Configuration validation and schema enforcement. Expert defaults, user overrides, and CLI flags are resolved and frozen into a single `InternalConfig` at startup. Unknown fields in user configurations raise at startup; type and range violations produce descriptive errors before any processing begins [@narayanan2024getting].

**OpenCV and SciPy.** Dense optical flow via the Farnebäck algorithm (OpenCV) for motion estimation. Connected component labeling and morphological operations via `scipy.ndimage`. Linear sum assignment for optimal cell-to-cell matching via `scipy.optimize` [@virtanen2020scipy; @farneback2003two].

**NetworkX.** The tracking module maintains a `networkx.DiGraph` of all observed cells and lineage edges across the processing session. This graph computes cell age and dominant lineage edges in split/merge complexes [@hagberg2008exploring].

**DuckDB and PyArrow.** Per-cell statistics are persisted as Parquet, a columnar on-disk file format for analytical data [@vohra2016parquet], written via PyArrow, the Python bindings to Apache Arrow's in-memory columnar layout [@lentner2019arrow]. Downstream queries are executed by an embedded DuckDB instance within `DataClient`. DuckDB is an in-process analytical SQL engine that runs directly against Parquet files [@raasveldt2019duckdb] which apply predicate and projection pushdown without loading full datasets into memory.

(sec-roadmap)=
## Roadmap and Contribution Opportunities

Adapt is an actively developed prototype. @tbl-status summarizes what is implemented today, what is experimental, and what is planned, to distinguish the working system from its intended trajectory.

```{list-table} Implementation status of Adapt components.
:label: tbl-status
:header-rows: 1
* - Status
  - Components
* - Implemented
  - NEXRAD pipeline (six modules), registry-based module system, dependency-driven DAG executor, three-tier configuration, read-only `DataClient` access layer, opt-in boundary contracts, `postprocess` command for repository enrichment, and `cell_volume_stats` 3D statistics (cloud-top height).
* - Experimental
  - Migration of the data repository to a Zarr/Icechunk cloud-native store.
* - In progress
  - Tracking of connected groups of cells (storm complexes).
* - Planned
  - Offline post-processing engine for after-run analysis; parallel satellite and radar pipelines with product merging; ML-based nowcasting trained on archived Adapt output; 3D modules for segmentation and storm analysis; mandatory contracts at the stable release.
```

The first stable release (v1.0) will establish the core architectural foundations of Adapt including a registry-based module system, dependency-driven DAG execution, configuration-defined pipelines, and a stable data-access layer. Current development focuses on simplicity, composability, and maintainability so that future capabilities can be added without changes to the framework core.

At this stage, our highest priority is architectural feedback. We welcome design reviews, validation strategies, testing frameworks, and software engineering practices that improve scalability, correctness, extensibility, and long-term sustainability.

After v1.0, community contributions will expand toward new modules, algorithms, pipeline definitions, data products, and user-facing applications.

### Architecture and Agentic Development

Adapt is being developed at a time when scientific software is increasingly created and maintained using AI-assisted and agentic workflows. To preserve architectural integrity, the project already employs automated checks including `pytest`, `import-linter`, static type checking with `mypy`, and code-quality tools. These tools provide a first line of defense against architectural drift and unintended design violations introduced during rapid development.

We seek contributions on additional mechanisms for architecture validation and governance, including interface contracts, architecture-aware testing, machine-readable design specifications, dependency analysis, automated design-rule enforcement, and other approaches that make architectural constraints explicit and verifiable.

### Pipeline Composition and Orchestration

Adapt constructs execution graphs from declared module dependencies, but pipeline definitions remain largely static. Future work will explore more flexible pipeline composition from available modules, data products, and instrument platforms. Contributions are encouraged in dependency resolution, workflow planning, execution scheduling, and multi-instrument workflows.

### Cloud and Interactive Applications

Future deployments include cloud-hosted nowcasting services, web-based dashboards, and JupyterHub environments for scientific analysis. Contributions in deployment architecture, distributed execution, cloud storage, and data-serving infrastructure are welcome.

## Summary

Adapt establishes the architectural foundations for a modular framework supporting adaptive radar-scanning operations. Independently developed modules are composed automatically through dependency-driven DAG execution, enabling extensible and reproducible scientific workflows.

The reference NEXRAD pipeline demonstrates the architecture from data acquisition to cell tracking, while the consumer API provides a stable interface for downstream analysis and applications. Future development will focus on strengthening the architecture, expanding the module ecosystem, and supporting new scientific workflows and instrument platforms.

Code and documentation are available at <https://github.com/ARM-DOE/adapt>.

## Acknowledgments

This research was supported by the Atmospheric Radiation Measurement (ARM) User Facility, a U.S. Department of Energy Office of Science user facility managed by the Biological and Environmental Research Program. Argonne National Laboratory's work was supported by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research, under Contract DE-AC02-06CH11357.

The work benefited from interactions with Sean Shahkarami, Sean Freeman, Brenda Dolan, Michael Giansiracusa, and Scott Giangrande. The authors also thank Jim Mather, Jennifer Comstock, Giri Prakash, and the DOE-ARM radar community for their support and feedback. Valuable discussions with the tobac development team and the Open Radar community also contributed to the direction of this work.
