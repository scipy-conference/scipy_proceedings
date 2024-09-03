---
title: "Echodataflow: Recipe-based Fisheries Acoustics Workflow Orchestration"
abstract: |
  With the influx of large data from multiple instruments and experiments, scientists are wrangling complex data pipelines that are context-dependent and non-reproducible. We demonstrate how we leverage Prefect [@prefect], a modern orchestration framework, to facilitate fisheries acoustics data processing. We built a Python package Echodataflow [@echodataflow] which 1) allows users to specify workflows and their parameters through editing text “recipes” which provide transparency and reproducibility of the pipelines; 2) supports scaling of the workflows while abstracting the computational infrastructure; 3) provides monitoring and logging of the workflow progress. Under the hood, Echodataflow uses Prefect to execute the workflows while providing a domain-friendly interface to facilitate diverse fisheries acoustics use cases. We demonstrate the features through a typical ship survey data processing pipeline. 
---


## Motivation
Acoustic fisheries surveys and ocean observing systems collect terabytes of echosounder (water column sonar) data that require custom processing pipelines to obtain the distributions and abundance of fish and zooplankton in the ocean [@ncei_story_map]. The data are collected by sending an acoustic signal into the ocean which scatters from objects in the water column and the returning “echo” is recorded. Although data usually have similar dimensions: range, time, location, and frequency, and can be stored into multi-dimensional arrays, the exact format varies based on the data collection scheme and the exact instrument used. Fisheries ship surveys, for example, follow pre-defined paths (transects) and can span several months ([Figure %s ](#fig:data_collection) left). Ocean moorings, on the other hand, have instruments at fixed locations and can collect data continuously at specified intervals for months ([Figure %s ](#fig:data_collection) right). Uncrewed Surface Vessels (USVs) (e.g. Saildrone [@saildrone], DriX [@drix], [Figure %s ](#fig:data_collection) middle) can autonomously collect echosounder data over large spatial regions. In all these scenarios, data are usually collected with similar instruments, and there is an overlap between the initial processing procedures. However, there are always variations associated with the specific data collection format, end research needs, data volume, and available computational infrastructure. For example, ship surveys may require grouping data along individual transects and exluding other data; they may also have varying range/depth resulting into data arrays of different dimensions. Mooring data are more regular, but their volume is large, and studies may require organizing data into daily patterns to analyze long term trends. USVs collect data at varying speeds thus requiring converting the time dimension to distance in order to have consistent echo patterns. The time when the data needs to be processed also affects the workflows: on premise/realtime applications usually require processing small data subsets at a time with limited computing resources; historical analyses require processing large datasets, and can benefit from cluster/cloud computing. The various scenarios demand different data workflows, and adapting from one setting to another is not trivial.



:::{figure} data_collection.png
:label: fig:data_collection
**Data Collection Schemes:** left, ship survey transect map for the Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey [@NWFSC_FRAM_2022]; middle, USV path map for Saildrone west coast survey [@saildrone_map]; right, map and instrument diagram for a stationary ocean observing system (Ocean Observatories Initiative Cabled and Endurance Arrays [@trowbridge_ooi_2019], Image Credit: Center for Environmental Visualization, University of Washington)
:::

## Fisheries Acoustics Workflows
Fisheries acoustics scientists traditionally have had go-to tools and procedures for their data processing and analysis, mostly relying on computation on a local computer. However, as the diversity of computing and data storage resources grows and the field becomes more interdisciplinary (it involves scientists with backgrounds in physics, biology, oceanography, acoustics, signal processing, machine learning, software engineering, etc.), it is becoming more challenging to make decisions on the best arrangement to accomplish the work. For example, [Figure %s ](#fig:workflow_variations) shows the many variations of workflows that can be defined based on the use cases and the options for data storage and computing infrastructure. 

:::{figure} workflow_variations.png
:label: fig:workflow_variations
**Fisheries Acoustics Workflow Variations:** Various use cases (fisheries, data management, machine learning, education) drive different needs for data storage and computing infrastructure. Options are abundant but adopting new technology and adapting workflows across use cases is not trivial.
:::


### User Stories
To demonstrate the software requirements of the fisheries acoustics community, below we describe several example user stories.

A **fisheries scientist** needs to process all data after a 2-month ship survey to obtain fish biomass estimates. They have previously used a commercial software package and are open to exploring open-source tools to achieve the same goal. They are familiar with basic scripting in Python.

A **machine learning engineer** is developing an ML algorithm to automatically detect fish on a USV. They need to prepare a large dataset for training but do not know all the necessary preprocessing steps. They are very familiar with ML libraries but do not have the domain-specific knowledge for acoustic data processing. They are also not familiar with distributed computing libraries.

A **data manager** wants to process several terabytes of mooring data and serve them to the scientific community. They have a few Python scripts to do this for a small set of files at a time, but want to scale the processing for many deployments using a cloud infrastructure.

An **acoustics graduate student** obtained echosounder data analysis scripts from a retired scientist but does not have all the parameters needed to reproduce the results in order to proceed with their dissertation research.

We draw attention to the different levels of experience of these users: each user has expertise in a subdomain, however, to accomplish their specific goal(s), they need to learn new tools or obtain knowledge from others. We outline several requirements that stem from these stories:

* The system should run both on a local computer and within a cloud environment. 
* The system should allow processing to be scaled to large datasets, but should not be overly complicated. For example, users with Python scripting experience can run it locally with pre-defined stages and parameters.
* The system should provide visibility into the operations that are applied to the data, and the procedures should be interpretable to users without acoustics expertise.
* The system should preferably be free and open source so that it is accessible to members of different institutions.
* The system should adapt to rapid changes of cloud and distributed computing libraries, and preferably should leverage existing developments within the technical communities.


### Software Landscape
Traditionally echosounder data processing pipelines are executed within a GUI-based software (e.g. Echoview [@echoview_software_pty_ltd_echoview_nodate], LSSS [@korneliussen_lsss_2006], ESP3 [@ladroit_esp3_2020], Matecho [@perrot_matecho_2018]). These software packages have been invaluable for onboard real-time visualization, as well as post-survey data screening and annotation. Some of them also support integration with scripting tools which facilitates the reproducible execution of the pipelines. For example, the Echoview software provides the option to automate pipelines through an Automation Module and to visualize the processing stages in a Dataflow Toolbox. Further, one can script operations through the `echoviewR` package [@harrison_echoviewR_2015]. However, since Echoview is neither free nor open source, these pipelines cannot be shared with researchers who do not have a license. In general, the GUI tools are usually designed to be used on a desktop computer and require downloading the data first, which is becoming challenging with the growing volume of the datasets. There has been also growth in development of new methods to detect the species of interest from the echosounder data, with the goal of substituting for the manual annotation procedures and making analysis of large datasets more efficient and objective. However, the new methods are typically developed independently from the existing software packages.
 Over the last several years there has been substantial development of open source Python packages (PyEchoLab [@wall_pyecholab_2018], `echopype` [@lee_echopype_2021], `echopy` [@echopy_2024]), each providing common echosounder processing functionalities, but differing in the data structure organization and processing. Since echosounder instruments store the data in binary, instrument-specific formats, the first stage requires parsing the raw data into a more common format. PyEcholab converts the data into `numpy` [@numpy] arrays. `echopy` expects data are already parsed into `numpy` arrays and all methods operate on them. Echopype converts raw data files into a standardized Python `EchoData` object, which can be stored in a `zarr` [@zarr] format and supports distributed computing by utilizing `dask` [@dask] and `xarray` [@xarray]. The use of open source packages and well-established formats allow further integration with other open source libraries such as those for machine learning (e.g. classification, clustering) or visualization. In addition, if custom modification is required for a specific application scenario, researchers can adapt the code and contribute the modification back to the packages, which is likely to benefit other researchers.

#### Challenges

Despite the availability of methods and tools to process echosounder data, it is not trivial to orchestrate all function calls in an end-to-end pipeline. While a well-documented Jupyter [@jupyter] notebook can show the sequence of processing stages, a considerable amount of path and parameter configuration is required to execute these stages on a large dataset, store the intermediate data products, and log the process comprehensively. Although automation can be achieved through a combination of Python and bash scripts that provision the environment, execute the stages, and manage inputs/outputs, the configuration process can be tedious, prone to error, and specific to the use case and the computing platform. Adapting an existing procedure to a new setting is usually not straightforward, and sometimes even reproducing previous results can pose a challenge. Below we discuss in more detail the different choices of data storage and computational infrastructure and the associated challenges of building workflows across them.

##### Data Storage

Researchers are faced with decisions of where to store the data from experiments, intermediate products, and final results. Initially, data are usually stored on local hard drive storage associated with the instrument (which on some platforms may have limited capacity), but eventually, these data may be transferred to a data archive if one is maintained within the community. Some agencies (e.g. NOAA National Centers for Environmental Information (NCEI) [@wall_2016]) have adopted cloud storage, and have publicly shared their data, which greatly facilitates data access and reuse. However, those repositories are usually not where researchers can store processed products. Funding models and organizational structure can result in short-term availability of resources and the need to change providers. Certain applications may need to access the data before they are archived and unreliable internet connection may require storing the data on-premise or at temporary locations. *To be agile to those frequent changes and allow to easily switch between different platforms, workflows will benefit from a level of abstraction from storage systems.*

##### Computing Infrastructure

With the growth of the echosounder datasets, researchers face challenges processing the data on their personal machines: both in terms of memory usage and computational time. A typical first attempt for resolution would be to amend the workflow to process smaller chunks of the data and parallelize operations across multiple cores if available. However, today researchers are also presented with a multitude of options for distributed computing: high-performance computing clusters at local or national institutions, cloud provider services: batch computing (e.g. Azure Batch, AWS Batch, Google Cloud Batch), container services (e.g. Amazon Elastic Container Services, Azure Container Apps, Google Kubernetes Engine), serverless functions (e.g. AWS Lamdba Functions, Google Cloud Functions, Microsoft Azure Functions). The choice may be driven by the storage system: its usage fees and retrieval speeds. Data, code and workflow organization usually has to be adapted based on the computing infrastructure. The knowledge required to configure these systems to achieve efficient processing is quite in-depth, and distributed libraries can be hard to debug and can have unexpected performance bottlenecks. *Abstracting the computing infrastructure and the execution of the tasks can allow researchers to focus on the scientific analysis of these large and rich datasets.*

## Echodataflow Overview
At the center of `echodataflow`'s design is the notion that a workflow can be configured through a set of recipes (`.yaml` files) that specify the pipeline, data storage, and logging structure. The idea draws inspiration from the Pangeo-Forge Project which facilitates the Extraction, Transformation, Loading (ETL) of earth science geospatial datasets from traditional repositories to analysis-ready, cloud-optimized (ARCO) data stores [@pangeo-forge]. The pangeo-forge recipes (which themselves are inspired by the conda-forge recipes [@conda_forge_community_2015_4774216]) provide a model of how the data should be accessed and transformed, and the project has garnered numerous recipes from the community. 

While Pangeo-Forge’s focus is on transformation from `netcdf` [@netcdf] and `hdf5` [@hdf5] formats to `zarr`, `echodataflow`’s aim is to support full echosounder data processing and analysis pipelines: from instrument-generated raw data files to data products which contain acoustically-derived biological estimates, such as abundance and biomass. `echodataflow` leverages Prefect [@prefect] to abstract data and computation management. In [Figure %s ](#fig:echodataflow_overview) we provide an overview of `echodataflow`’s framework. At the center we see several steps of an echosounder data processing pipeline: `open_raw`, `combine_echodata`, `compute_Sv`, `compute_MVBS`, `frequency_differencing`, which produce echo classificaton results using a simple threshold-based criterion. All these functions exist in the `echopype` package, and are wrapped by `echodataflow` into pre-defined stages. Prefect executes the stages on a Dask cluster which can be started locally or can be externally set up. These `echopype` functions already support distributed operations with Dask, and thus the integration with Prefect within `echodataflow` is natural. Dask clusters can be set up on a variety of platforms: local computers, cloud virtual machines, kubernetes [@kubernetes] clusters, or HPC clusters (via `dask-jobqueue` [@dask-jobqueue]), etc. and allow abstraction from the computing infrastructure. The input datasets, intermediate data products, and final data products can live in different storage systems (local/cloud) and Prefect’s block feature provides seamless, provider-agnostic, and secure integration with them. Workflows can be executed and monitored through Prefect’s dashboard, while logging of each function is handled by `echodataflow`.

:::{figure} echodataflow.png
:label: fig:echodataflow_overview
**Echodataflow Framework:** The above diagram provides an overview of the `echodataflow` framework: the objective is to fetch raw files from a local filesystem/cloud archive, process them through several stages of an echosounder data workflow using a cluster infrastructure, and store both intermediate and final data products. In `echodataflow` the workflow is executed based on text configurations, and logs are generated for the individual processing stages. Prefect handles the execution of the tasks on the cluster and provides tools for monitoring the workflow runs. 
:::

### Why Prefect?
We chose Prefect among other Python workflow orchestration frameworks such as Apache Airflow [@airflow], Dagster [@dagster], Argo [@argo], Luigi [@luigi]. While most of these tools provide flexibily and level of abstraction suitable for executing fisheries acoustics pipelines, we selected Prefect for the following reasons:
* Prefect accepts dynamic workflows which are specified at runtime and do not require to follow a Directed Acyclic Graph, which can be restricting and difficult to implement.
* In Prefect, Python functions are first class citizens, thus building a Prefect workflow does not deviate substantially from traditional science workflows composed of functions.
* Prefect integrates with a `dask` cluster, and `echopype` processing functions are already using `dask` to scale operations.
* Prefect’s code runs similarly locally as well as on cloud services. 
* Prefect’s monitoring dashboard is open source, can be run locally, and is intuitive to use.

We next describe in more detail the components of the workflow lifecycle.

## Workflow Configuration
The main goal of `echodataflow` is to allow users to configure an echosounder data processing pipeline through editing configuration “recipe” templates. `echodataflow` can be configured through three templates: `datastore.yaml` which handles the data storage locations, `pipeline.yml` which specifies the processing stages, and `logging.yaml` which sets the logging format. 

### Data Storage Configuration
Below we show an example file `datastore.yaml` with a data storage configuration for a ship survey. In this scenario the goal is to process data from the Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey [@NWFSC_FRAM_2022] which are publicly available on an AWS S3 bucket hosted by NOAA National Centers for Environmental Information Acoustics (NCEA) Archive [@wall_2016]. The archive contains data from many surveys dating back to 1991 (~280TB). The configuration allows to pass parameters specifying the ship, survey, and sonar model names and select the subset of files belonging only to the survey of interest. The output destination is set to a private S3 bucket belonging to the user (within an AWS account different from the input one), and the credentials are passed through a `block_name`. The survey contains ~4000 files, and one can set the group option to combine the files into survey-specific groups: based on the transect information provided in the `transect_group.txt` file. One can further use regular expressions to subselect other subgroups based on needs. 

<!-- :::{figure} datastore_config.png
:label: fig:datastore_config
**Data Storage Configuration:** The input is data from a ship survey data on and S3 data on the NCEA archive. The output goes to an S3 bucket on another AWS account. Credentials are stored in blocks.
:::
-->


<!--:name: datastore-config
Data Storage Configuration 
-->


```yaml
# datastore.yaml

name: Bell_M._Shimada-SH1707-EK60
sonar_model: EK60 
raw_regex: (.*)-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6}) 
args:
  urlpath: s3://ncei-wcsd-archive/data/raw/{{ ship_name }}/{{ survey_name }}/{{ sonar_model }}/*.raw 
  parameters: 
    ship_name: Bell_M._Shimada
    survey_name: SH1707
    sonar_model: EK60
  storage_options: 
    anon: true
  group: 
    file: ./transect_group.txt 
    storage_options: 
      block_name: echodataflow-aws-credentials
      type: AWS 
  group_name: default_group 
  json_export: true 
  raw_json_path: s3://echodataflow-workground/combined_files/raw_json 
output: 
  urlpath: <YOUR-S3-BUCKET>
  overwrite: true 
  retention: false 
  storage_options: 
    block_name: echodataflow-aws-credentials
    type: AWS
```


### Pipeline Configuration
The pipeline configuration file’s purpose is to list the stages of the processing pipeline and the computational set-up for their execution. Below we show an example `pipeline.yaml` file which cofigures a pipeline with several stages: `open_raw`, `combine_echodata`, `compute_Sv`, `compute_MVBS`. Each stage is executed as a separate Prefect subflow (a component of a Prefect workflow), and one can specify additional options on whether to store the raw files. `echodataflow` requires access to a Dask cluster: it can be either created on the fly by setting the `use_local_dask` to `true`, or an IP address of an already running cluster can be provided. Individual stages may require different cluster configurations to efficiently execute the tasks. Those can be specified with the additional `prefect_config` option through which the user can set a specific Dask task runner or the number of retries. Managing retries is essential for handling transient failures, such as connectivity issues, ensuring the stages can be re-executed without any manual interference if a failure occurs.

<!--
:::{figure} pipeline_config.png
:label: fig:pipeline_config
**Pipeline Configuration:** The pipeline consists of two stages: `echodataflow_open_raw` and `echodataflow_compute_TS` and will run on a local cluster.
:::
-->

```yaml
# pipeline.yaml

active_recipe: standard 
use_local_dask: true 
n_workers: 4 
scheduler_address: tcp://127.0.0.1:61918 
pipeline: 
- recipe_name: standard 
  stages: 
  - name: echodataflow_open_raw 
    module: echodataflow.stages.subflows.open_raw 
    options: 
      save_raw_file: true 
      use_raw_offline: true 
      use_offline: true 
    prefect_config:
      retries: 3
  - name: echodataflow_combine_echodata
    module: echodataflow.stages.subflows.combine_echodata
    options:
      use_offline: true
  - name: echodataflow_compute_Sv
    module: echodataflow.stages.subflows.compute_Sv
    options:
      use_offline: true
  - name: echodataflow_compute_MVBS
    module: echodataflow.stages.subflows.compute_MVBS
    options:
      use_offline: true
    external_params:
      range_meter_bin: 20
      ping_time_bin: 20S
```

### Logging Configuration

By default, the outcomes of each stage are logged. The logs can be stored in `.json` or plain text files, and the format of the entries can be specified in the configuration file as displayed below. The `json` format allows searching through the logs for a specific key. 

<!-- :::{figure} logging_config.png
:label: fig:logging_config
**Logging Configuration:** The logs are written to a local `echodataflow.log` file in a plain text form based on the format specification
:::
-->

```yaml
# logging.yaml

version: 1
disable_existing_loggers: False
formatters:
  json:
    format: "[%(asctime)s] %(process)d %(levelname)s %(mod_name)s:%(func_name)s:%(lineno)s - %(message)s"
  plaintext:
    format: "[%(asctime)s] %(process)d %(levelname)s %(mod_name)s:%(func_name)s:%(lineno)s - %(message)s"
handlers:
  logfile:
    class: logging.handlers.RotatingFileHandler
    formatter: plaintext
    level: DEBUG
    filename: echodataflow.log
    maxBytes: 1000000
    backupCount: 3

loggers:
  echodataflow:
    level: DEBUG
    propagate: False
    handlers: [logfile]
```

In this case the logs are stored in the plain text file `echodataflow.log`. Below we show an example of output logs.

<!--:::{figure} log_output.png
:label: fig:log_output
**Log Output:** An example of an output log of file processing failure. The leading indicators for each entry are: date, process id, log level, followed by the module, function, line number, followed by the message.
:::
-->

```log
[2024-06-06 17:32:08,945] 51493 ERROR apply_mask.py:EK60_SH1707_Shimada2_applymask.zarr:147 - Computing apply_mask
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - Encountered Some Error in EK60_SH1707_Shimada0
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - 'source_ds' must have coordinates 'ping_time' and 'range_sample'!
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - Encountered Some Error in EK60_SH1707_Shimada1
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - 'source_ds' must have coordinates 'ping_time' and 'range_sample'!
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - Encountered Some Error in EK60_SH1707_Shimada2
[2024-06-06 17:32:08,946] 51493 ERROR file_utils.py:file_utils:147 - 'source_ds' must have coordinates 'ping_time' and 'range_sample'!
```

In Section [Workflow Logging](#Workflow-Logging), we provide more information on logging options.


## Workflow Execution
To convert a scientific pipeline into an executable Prefect workflow, one needs to organize its components into flows, sublfows, and tasks (the key objects of Prefect’s execution logic). Usually, the stages of a pipeline are organized into flows and subflows, while the individual pieces of work within the stage are organized into tasks. In practice, flows, subflows, and tasks are all Python functions, and they differ in how we want to execute them (e.g. concurrently/sequentially, w/o retries), and what we want to track during execution (e.g. input/outputs, state logging, etc.). In `echodataflow` we organize the typical echosounder processing stages into subflows (flows within the main workflow), while the operations on different files (or groups of them) are individual tasks. We describe how functions are organized in the `open_raw` stage, which reads the files from raw format, parses the data, and writes them into a `zarr` format. The `echodataflow_open_raw` function is decorated as a flow, and is one of many subflows of the full workflow. This function processes all files. 
<!--
:::{figure} flow_task.png
:label: fig:flow_task
**Decorating Functions into Flows \& Tasks:** The flow `echdataflow_open_raw` calls the task `process_raw` (processing a single file) within a loop to process all files
:::
-->

```python
@flow
@echodataflow(processing_stage="Open-Raw", type="FLOW")
def echodataflow_open_raw(
    groups: Dict[str, Group], config: Dataset, stage: Stage, prev_stage: Optional[Stage]
):
    """
    Process raw sonar data files and convert them to zarr format.

    Args:
        config (Dataset): Configuration for the dataset being processed.
        stage (Stage): Configuration for the current processing stage.
        prev_stage (Stage): Configuration for the previous processing stage.

    Returns:
        List[Output]: List of processed outputs organized based on transects.
```

`echodataflow_open_raw` contains a loop which iterates through all file groups and applies the `process_raw` function which operates on a single group and is decorated as a task. All tasks will be executed on the Dask cluster. 
```python
for name, gr in groups.items():
        for raw in gr.data:
            new_processed_raw = process_raw.with_options(
                task_run_name=raw.file_path, name=raw.file_path, retries=3
            )
            future = new_processed_raw.submit(raw, gr, working_dir, config, stage)
            futures[name].append(future)

```

```python
@task()
@echodataflow()
def process_raw(
    raw: EchodataflowObject, group: Group, working_dir: str, config: Dataset, stage: Stage
):
    """
    Process a single group of raw sonar data files.
```

## Workflow Monitoring
One of the main advantages of using orchestration frameworks is that they usually provide tools to monitor the workflow execution. The integration with Prefect allows leveraging Prefect’s dashboard (Prefect UI) for monitoring the execution of the flows. The dashboard can be run locally and within Prefect’s online managed system (Prefect Cloud). The local version provides an entirely open source framework for running and monitoring workflows. [Figure %s ](#fig:flow_sequence_expanded) shows the view of completed runs within the dashboard. The progress can be monitored while the flows are in progress. 

:::{figure} flow_sequence_expanded.png
:label: fig:flow_sequence_expanded
**Flow Runs:** Log of completed runs in Prefect UI. The stages (subflows) are executed sequentially. One can expand the view of an individual flow and see the tasks computed (asynchronously) within it.
:::

Further, one can also view the progress of the execution of the tasks on the Dask cluster.

:::{figure} dask_progress.png
:label: fig: dask_progress
**Dask Dashboard:** The execution of the tasks on the Dask cluster can also be monitored through the Dask dashboard.
:::



## Workflow Logging
Processing large data archives requires a robust logging system to identify at which step and for which files the processing has failed. Locating the issues allows to set a path forward to resolve them: either through improving the robustness of the individual libraries performing the processing steps, or through identifying the artifacts of the data which are incompatible with the existing pipeline. To address this, we provide several approaches:
* Basic Logging with Dask Worker Streams: this approach configures Dask worker streams to handle `echodataflow` logs, which is straightforward if exact log order is not crucial.
* Centralized Logging with Amazon CloudWatch [@cloudwatch]: this approach centralizes all logs for easy access and analysis. It can be useful when users are already utilizing AWS.
* Advanced Logging with Apache Kafka [@kafka] and Elastic Stack [@elastic_stack] (Elasticsearch, Kibana, Beats, Logstash): this approach leverages Kafka for log aggregation and Elastic Stack for log analysis and visualization, offering a robust solution for those who can maintain the infrastructure, for example data center managers.

By default if logging is not configured, all the worker messages are directed to the application console. The order of logs may not be preserved since logs are written once control returns from the Dask workers to the main application.

## Workflow Deployment 

### Notebook
`echodataflow` can be directly initiated within a Jupyter notebook, which makes development interactive and provides a work environment familiar to researchers. One can see how the workflow is initiated within the Jupyter cell in [Figure %s ](#fig:notebook_start).

:::{figure} notebook_start.png
:label: fig:notebook_start
**Initiating `echodataflow` in a Jupyter Notebook:** Once one has a set of "recipe" configuration files, they can initiate the workflow in a notebook cell with the `echodataflow_start` command.
:::


We provide two demo notebooks: one for execution on a [local machine](https://github.com/OSOceanAcoustics/echodataflow/blob/1ac65fa0bfcdd01b151b98134b842364311059fd/docs/source/local/notebook.ipynb) and another one for execution on [AWS](https://github.com/OSOceanAcoustics/echodataflow/blob/1ac65fa0bfcdd01b151b98134b842364311059fd/docs/source/local/notebook.ipynb). 

### Docker
We facilitate the deployment of `echodataflow` on various platforms by building a Docker image from which one can launch a container with all required components and the user can access the workflow dashboard on the corresponding port.

```
docker pull blackdranzer/echodataflow 

prefect server start

docker run --network="host" -e PREFECT_API_URL=http://host.docker.internal:4200/api blackdranzer/echodataflow
```

Upon execution, the user can readily access the Prefect UI dashboard and run workflows from there.

We also provide a Docker image for initiating logging with Kafka and Elastic Stack, thus streamlining the configuration of several tools.



## Command Line Interface
We provide a command line interface which supports credential handling, and several additional features for managing workflows: stage addition and rule validation.

### Adding Stages
Currently, most major functionalities in the echopype package are wrapped into stages: `open_raw`, `add_depth`, `add_location`, `compute_Sv`, `compute_TS`, `compute_MVBS`, `combine_echodata`, `frequency_differencing`, `apply_mask`. 

We provide tools to generate boilerplate template configuration based on the existing stages. Here is an example to add a stage:

```
echodataflow gs <stage_name>
```

For instance, to generate a boilerplate configuration for the `compute_Sv` stage, one would use:

```
echodataflow gs compute_Sv
```

This command creates a template configuration file for the specified stage, allowing to customize and integrate it into a workflow. The generated file includes:
* a flow: it orchestrates the execution of all files that need to be processed, either concurrently or in parallel, based on the configuration.
* a task (helper function): it assists the flow by processing individual files.

### Rule Validation
Scientific workflows often have stages that cannot be executed until other stages have completed. Those conditions can be set through `echodataflow` client during the initialization process and are stored in a `echodataflow_rules.txt` file:

```
echodataflow_open_raw:echodataflow_compute_Sv
echodataflow_open_raw:echodataflow_combine_echodata
echodataflow_open_raw:echodataflow_compute_TS
echodataflow_combine_echodata:echodataflow_compute_Sv
echodataflow_compute_Sv:echodataflow_compute_MVBS
```

These rules dictate the sequence in which stages should be executed, ensuring that each stage waits for its dependencies to complete. They can be set through the `echodataflow rules -add ...` command.

#### Aspect-Oriented Programming (AOP) for Rule Validation

In `echodataflow`, we adopt an aspect-oriented programming [@aop] approach for rule validation. This is achieved using a decorator that can be applied to functions to enforce rules and log function execution details. The echodataflow decorator logs the entry and exit of a decorated function and modifies the function's arguments based on the execution context. This supports two types of execution: "TASK" and "FLOW".

Example Usage:

```python
@echodataflow(processing_stage="StageA", type="FLOW")
def my_function(arg1, arg2):
    # Function code here
    pass
```

In the example, the echodataflow decorator ensures that the function `my_function` is executed within the context of "StageA" as a "FLOW", checking for dependencies and logging relevant information.

## Example Use Case: Processing Ship Survey Data from an Archive

We demonstrate a workflow processing all acoustic data for the 2017 Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey through a few routine processing stages. The survey spans a period of 06/15/2017 - 09/13/2017, covering the entire west coast of the US and Canada. Figure 1(a) shows a map of a typical transect schedule of the survey. Raw acoustic data are collected continuously while the ship is in motion, resulting in a total of 3873 files collected with a total size of 93 GB. The raw files are archived by the NOAA NCEI Water Column Sonar Data Archive and are publicly accessible on their Amazon Web Services S3 bucket ([https://registry.opendata.aws/ncei-wcsd-archive/](https://registry.opendata.aws/ncei-wcsd-archive/)). The processing pipeline involves several steps:

* Convert raw files to cloud-native `zarr` format following closely a community convention [@lee_echopype_2021], [@convention]
* Combine multiple individual `zarr` files within a continuous transect segment into a single `zarr` file
* Compute Sv: calibrate the measured acoustic backscatter data to volume backscattering strength (Sv, unit: dB re 1 m$^{-1}$)

Once data are converted to Sv, they are easy to manipulate, as the data are stored in an `xarray` data array and are smaller than that of the original data. The final dataset can be served as an analysis-ready data product to the community. It can be beneficial to store also intermediate datasets at different processing stages: for example, preserving the converted raw files in the standardized `zarr` format allows users to regenerate any of the following stages with different groupings or resolution, without having to fetch and convert raw data again.

The execution of the workflow with `echodataflow` allowed us to monitor the progress of all files [Figure %s ](#fig:one_failed): 3872 files were successfully processed, and 1 failed. Most importantly, the failure did not block the execution of the other files, and a log was generated for the stage and the filename for which the error occurred. This experiment serves as a confirmation that the transition from local development to a full production pipeline with `echodataflow` can indeed be smooth.


:::{figure} one_failed.png
:label: fig:one_failed
**Processing full 2017 Survey Data:** 1/3873 files failed at the `open_raw` stage, but this did not impact the entire pipeline. As shown, other files were processed successfully through all stages.
::: 

## Future Development
Our immediate goal is to provide more example workflow recipes integrating other stages of echosounder data processing, such as machine learning prediction, training dataset generation, biomass estimation, interactive visualization, etc. We will demonstrate utilizing functionalities from a suite of open source Python packages (`echoregions` [@echoregions] for reading region annotations and creating corresponding masks, `echopop` [@echopop] for combining acoustic data with biological "ground truth" into biomass estimation, `echoshader` [@echoshader] for echogram and map dashboard visualization) in building workflows for the Pacific Hake Survey: both in a historical and near-realtime on-ship data processing context. We aim to streamline the stage addition process. We will further investigate how to improve memory management and caching between and within stages to optimize for different scenarios. There is growing interest in the fisheries acoustics community to share global, accessible, and interoperable datasets [@gain_repo], and to agree on community data standards and definitions of processing levels [@convention], [@levels]. As those mature we will align them with existing stages in `echodataflow`, which will support building interoperable datasets whose integration will push us to study bigger and more challenging questions in fisheries acoustics.



## Beyond Fisheries Acoustics

Echodataflow was designed to facilitate fisheries acoustics workflows, but the structure can be adapted to data processing pipelines in other scientific communities. The key aspects are to identify the potential stages of the workflows and associated Python packages/functions that implement them, and to design the structure of the configuration files. The other aspects such as logging, deployment, monitoring, new-stage integration are domain-agnostic. Processing pipelines that require manipulation of large labeled arrays can directly benefit from the Dask cluster integration and are prevalent in the research community. Our use case of regrouping data based on time segments is a common need within scientific settings in which the file unit level of the instrument is not aligned with the unit level of analysis, and requires further reorganization and potential resampling and regridding along certain coordinates. We hope it can serve as a guide on how to build configurable, reproducible, and scalable workflows in new scientific areas.

## Acknowledgements:
We thank the Fisheries Engineering and Acoustic Technologies team at the NOAA Northwest Fisheries Science Center: Julia Clemons, Alicia Billings, Rebecca Thomas, Elizabeth Phillips for introducing us to the Pacific Hake Survey operations and the hake biomass estimation workflow.

This work used cpu compute and storage resources at Jetstream2 through allocation AGR230002 from the Advanced cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program [@10.1145/3437359.3465565], [@10.1145/3569951.3597559], which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## Funding:
NOAA Award No. NA21OAR0110201, NOAA Award No. NA20OAR4320271 AM43, eScience Institute
