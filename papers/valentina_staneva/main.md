---
# Ensure that this title is the same as the one in `myst.yml`
title: "echodataflow: Recipe-based Fisheries Acoustics Workflow Orchestration"
abstract: |
  With the influx of large data from multiple instruments and experiments, scientists are wrangling complex data pipelines that are context-dependent and non-reproducible. We demonstrate how we leverage Prefect [@prefect], a modern orchestration framework, to facilitate fisheries acoustics data processing. We built a Python package Echodataflow which 1) allows users to specify workflows and their parameters through editing text “recipes” which provide transparency and reproducibility of the pipelines; 2) supports scaling of the workflows while abstracting the computational infrastructure; 3) provides monitoring and logging of the workflow progress. Under the hood, echodataflow uses Prefect to execute the workflows while providing a domain-friendly interface to facilitate diverse fisheries acoustics use cases. We demonstrate the features through a typical ship survey data processing pipeline. 
---


## Motivation
Acoustic fisheries surveys and ocean observing systems collect terabytes of echosounder (water column sonar) data that require custom processing pipelines to obtain the distributions and abundance of fish and zooplankton in the ocean [@ncei_story_map]. The data are collected by sending an acoustic signal into the ocean which scatters from objects in the water column and the returning “echo” is recorded. Although data usually have similar dimensions: range, time, location, and frequency, and can be stored into multi-dimensional arrays, the exact format varies based on the data collection scheme and the exact instrument used. Fisheries ship surveys, for example, follow pre-defined paths and can span several months ([Figure %s ](fig:data_collection) top-left). Ocean moorings, on the other hand, have instruments at fixed locations and can collect data 24/7 for several years (when wired) ([Figure %s ](#fig:data_collection) bottom). Unmanned Surface Vehicles (USVs) (e.g. Saildrone[@saildrone], DriX[@drix], [Figure %s ](fig:data_collection) top-right) can autonomously collect echosounder data over large spatial regions. Despite that in all these scenarios data are usually collected with the same type of instruments, and there is an overlap between the intial processing procedures, the combination of research needs, data volume, and available computational infrastructure demand different data workflows, which are not trivial to adapt to new scenarios.



:::{figure} data_collection.png
:label: fig:data_collection
**Data Collection Schemes:** top left, ship survey transect map for the Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey[@NWFSC_FRAM_2022]; top right, USV path map for Saildrone west coast survey [ref][image_source]; bottom, map and instrument diagram for a stationary ocean observing system (Ocean Observatories Initiative Cabled and Endurance Arrays [trowbridge_ooi_2019] [image_source])


## Fisheries Acoustics Workflows
While traditionally fisheries acoustics scientists have had go-to tools and procedures for their data processing and analysis, now they have more options on how to accomplish their work. While this opens more opportunities for research, it also posed challenges, as  field has also become very interdisciplinary and includes researchers from different backgrounds (physics, biology, oceanography, acoustics, signal processing, machine learning, software engineering, etc.) at different levels of experience. [Figure %s ](#fig:workflow_variations) shows the many variations of workflows that can be defined based on the use cases, the options for data storage and computing infrastructure. We discuss these in more detail in the next sections.

:::{figure} workflow_variations.png
:label: fig:workflow_variations
**`echodataflow` Workflow Variations:** Various use cases (fisheries, data management, machine learning, education) drive different needs for data storage and computing infrastructure. Options are abundant but adopting new technology and adapting workflows across use cases is not trivial.
:::


### User Stories
To understand software requirements of the fisheries acoustics community, below we describe several example user stories.

A **fisheries scientist** needs to process all data after a 2-month ship survey to obtain fish biomass estimates. They have previously used a commercial software package and are open to exploring open-source tools to achieve the same. They are familiar with basic scripting in Python.

A **data manager**'s goal is to process several terabytes of mooring data and serve them to the community. They have a Python procedure on how to do this for a small set of files at a time but want to scale the processing for many deployments and want to be able to achieve this using a cloud infrastructure.

A **machine learning engineer** is developing an ML algorithm to automatically detect fish on a USV. They need to prepare a large dataset for training but do not know all the necessary preprocessing steps. They are very familiar with ML libraries but do not have the domain-specific knowledge for acoustics data processing. They are also not familiar with distributed computing libraries.

An **acoustics graduate student** obtained echosounder data analysis scripts from a retired scientist but is missing several parameters to reproduce the results needed to proceed with their dissertation research.

We draw attention to the different levels of experience of these users: each one has expertise in a subdomain but to accomplish their goal they needs to learn new tools or obtain knowledge from others. We outline several requirements that stem from these stories:

* The system should run both on a local computer and within a cloud environment. 
* The system should allow processing to be scaled to large datasets, but should not be too complicated so that users with Python scripting experience can run it locally with pre-defined stages and parameters.
* The system should provide visibility into the operations that are executed to the data, and the procedures should be interpretable to users without acoustics expertise.
* The system should preferably be free and open source so that it is accessible to members of different institutions.
* The system should adapt to the rapid changes of cloud and distributed computing libraries,and preferably should leverage the existing development within technical communities.


### Software Landscape
Traditionally echosounder data processing pipelines are executed within a GUI-based software (e.g. Echoview[@harrison_echoviewR_2015], LSSS[@korneliussen_lsss_2006], ESP3[@ladroit_esp3_2020], Matecho[perrot_matecho_2018]). These software applications have been invaluable for onboard real-time visualization, and post-survey data screening and annotation. Some of them also support integration with scripting tools which facilitates the reproducible execution of the pipelines. For example, the Echoview software provides the option to automate pipelines through the Automation Module and to visualize the processing stages in the Dataflow Toolbox. Further, one can script operations through the echoviewR package [ref]. However, since Echoview is neither free nor open source, these pipelines cannot be shared with researchers who do not have the license. In general, the graphics software tools are usually designed for work on a desktop and require downloading the data first, which is becoming challenging with the growth of the datasets. The manual annotation for long-term surveys has also become more demanding, and researchers are investigating new methods to detect the species of interest, and those often are developed independently of the software applications. Over the last several years there has been a substantial development of open source Python packages (PyEchoLab [@wall_pyecholab_2018], `echopype`[@lee_echopype_2021], `echopy`[@echopy_2024]), each providing common echosounder processing functionalities. Since echosounder instruments store the data in a binary instrument-specific formats, the first stage requires parsing the raw data into a more common Python data structure. PyEcholab converts the data into numpy [@numpy] arrays, Echopy assumes the data are already parsed in `numpy`[@numpy] format. Echopype converts raw data files into zarr datasets [@zarr] and supports distributed computing by utilizing dask and xarray [@xarray]. The use of open source packages and well-established formats allows to further integrate with other open source libraries such as those for machine learning (e.g. classification, clustering) or visualization. Further, if a scenario requires a custom modification, they can adapt the software and can contribute the modification back to the community, as it is highly possible another team could benefit from it.

#### Challenges

To orchestrate all function calls in an end-to-end pipeline, however, is not trivial. While a well-documented Python notebook can show the processing stages, executing those on a large dataset, storing intermediate products, and logging the process require a considerable amount of path and parameter configuration, and often becomes use case and platform specific. 
Python scripts executing the stages can require passing many arguments, thus making the letter flags not interpretable. Bash scripts can wrap the Python scripts but they require their own set of parameters and arguments, and bash syntax can be platform-dependent. Although scripts can eventually be stitched to produce the desired results, the process is tedious and prone to error. Adapting an existing procedure to a new setting is usually not straightforward, and sometimes even reproducing previous results can pose a challenge. Below we discuss in more detail the selection of data storage and computational infrasture and the associated challenges.

##### Data Storage

Researchers are faced with decisions of where to store the data from experiments, intermediate products, and final results. Initially, data are usually stored on local hard drive storage associated with the instrument (which on some platforms may have limited capacity), but eventually, these data may be transferred to a data archive if one is maintained within the community. Some agencies (e.g. NOAA NCEI [ref]) have adopted cloud storage, and have publicly shared their data, which greatly facilitates data access and reuse. However, those repositories are usually not where researchers can store processed products. Funding models and organizational structure can result in short-term availability of resources and the need to change providers. Certain applications may need to access the data before it is archived, or have unreliable internet connection thus requiring to store the data on-premise or at temporary locations. *To be agile to those frequent changes and allow to easily switch between different platforms workflows will benefit from a level of abstraction from storage systems.*

##### Computing Infrastructure

With the growth of the echosounder datasets, researchers face challenges processing the data on their personal machines: both in terms of memory usage and computational time. A typical first attempt for resolution would be amend the workflow to process smaller chunks of the data and parallelize operations across multiple cores if available. However, today researchers are also presented with a multitude of options for distributed computing: high-performance computing clusters at local or national institutions, cloud provider services: batch computing (e.g. Azure Batch[ref], AWS Batch[ref]), elastic container services (e.g. , serverless options (e.g. Amazon Web Services Lamdba Functions [ref], Google Cloud Functions, Microsoft Azure Functions). The choice may be driven by the storage system: its access fees and retrieval speeds. Data, code and workflow organization usually has to be adapted based on the computing infrastructure. The knowledge required to configure these systems to achieve efficient processing is quite in-depth, and distributing libraries can be hard to debug and can have unexpected performance bottlenecks. *Abstracting the computing infrastructure and distribution of the tasks can allow researchers to focus on the scientific analysis of these large and rich datasets.*

## Echodataflow Overview
At the center of `echodataflow`'s design is the notion that a workflow can be configured through a set of recipes (`.yaml` files) that specify the pipeline, data storage, and logging structure. The idea draws inspiration from the Pangeo-Forge Project [ref] which facilitates the Extraction, Transformation, Loading (ETL) of earth science geospatial datasets from traditional repositories to analysis-ready, cloud-optimized (ARCO) data stores [@pangeo-forge].  The pangeo-forge recipes provide a model of how the data should be accessed and transformed, and the project has garnered numerous recipes from the community. While Pangeo-Forge’s focus is on transformation from `.netcdf` [ref] and `hdf5` [ref] formats to `.zarr`, `echodataflow`’s aim is to support full echosounder data processing and analysis pipelines: from instrument raw data formats to biological products. `echodataflow` leverages Prefect [@prefect] to abstract data and computation management. In [Figure %s ](@echodataflow) we provide an overview of `echodataflow`’s framework. At the center we see several steps from an echosounder data processing pipeline: `open_raw`, `combine_echodata`, `compute_Sv`, `compute_MVBS`. All these functions exist in the `echopype` package, and are wrapped by `echodataflow` into pre-defined stages. Prefect executes the stages on a dask cluster which can be started locally or can be externally set up. These `echopype` functions already support distributed operations with dask thus the integration with Prefect within echodataflow is natural. Dask clusters can be set up on a variety of platforms: local, cloud, kubernetes [@kubernetes], HPC cluster via `dask-jobqueue` [@dask-jobqueue], etc. and allow abstraction from the computing infrastructure. Input, intermediate, and final data sets can live in different storage systems (local/cloud, public/private) and Prefect’s block feature provides seamless, provider-agnostic, and secure integration. Workflows can be executed and monitored through Prefect’s dashboard, while logging of each function is handled by echodataflow.

:::{figure} echodataflow.png
:label: fig:echodataflow
**`echodataflow` Framework:** The above diagram provides an overview of the `echodataflow` framework: the task is to fetch raw files from a local filesystem/cloud archive, process them through several stages of an echosounder data workflow using a cluster infrastructure, and store intermediate and final products. `echodataflow` allows the workflow to be executed based on text configurations, and logs are generated for the individual processing stages. Prefect handles the distribution of the tasks on the cluster, and provides tools for monitoring the workflow runs. 
:::

### Why Prefect?
We chose Prefect among other Python workflow orchestration frameworks such as Apache Airflow[@airflow], Dagster[@dagster], Argo[@argo], Luigi[@luigi]. While most of these tools provide flexibily and level of abstraction to execute fisheries acoustics pipelines, we selected Prefect for the following reasons:
* Prefect accepts dynamic workflows which are specified at runtime and do not require to follow a Directed Acyclic Graph, which can be restricting and difficult to implement.
* In Prefect, Python functions are first class citizens, thus building a Prefect workflow does not deviate a lot from traditional science workflows built out of functions.
* Prefect integrates with a `dask` cluster, and `echopype` processing functions are already using `dask` to scale operations.
* Prefect’s code runs similarly locally as well as on cloud services. 
* Prefect’s monitoring dashboard is open source, can be run locally, and is intuitive to use.

We next describe in more detail the components of the workflow lifecycle.

## Workflow Configuration
The main goal of echodataflow is to allow users to configure an echosounder data processing pipeline through editing configuration “recipe” templates. Echodataflow can be configured through three templates: `datastore.yaml` which handles the data storage decisions, `pipeline.yml` which specifies the processing stages, and `logging.yaml` which sets the logging format. 

### Data Storage Configuration
In [Figure %s ](#fig:datastore_config): `datstore.yaml` we provide an example of a data store configuration for a ship survey. In this scenario we want to process data from the Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey [@NWFSC_FRAM_2022] which is being publicly shared on an AWS S3 bucket by NOAA National Center for Environmental Information Acoustics Archive (NCEA)[@wall_2016]. The archive contains data from many surveys dating back to 1991 and contains ~280TB of data. The additional parameters referring to ship, survey, and sonar model names allow to select the subset of files belonging only to the survey of interest. The output destination is set to a private S3 bucket belonging to the user (i.e. an AWS account different from the input one), and the credentials are passed through a `block_name`. The survey contains ~4000 files, and one can set the group option to combine the files into survey-specific groups: based on the transect information provided in the `transect_group.txt` file. One can further use regular expressions to subselect other subgroups based on needs. 

:::{figure} datastore_config.png
:label: fig:datastore_config
**Data Storage Configuration:** The input is data from a ship survey data on and S3 data on the NCEA archive. The output goes to an S3 bucket on another AWS account. Credentials are stored in blocks.
:::


### Pipeline Configuration
The pipeline configuration file’s purpose is to list the stages of the processing pipeline and the computational set-up for their execution. In [Figure %s ](#fig:pipeline_config) we provide a short example with two stages: `open_raw` (which fetches data from the cloud, converts it from a raw format to a cloud native standardized output) and `compute_TS` (computes target strength: a popular quantity in fisheries acoustics). Each stage is executed as a separate Prefect subflow (a component of a Prefect workflow), and one can specify additional options on whether to store the raw files. `echodataflow` requires access to a dask cluster: one can either create one on the fly by setting the `use_local_dask` to `true`, or one can provide a link to an already running cluster. Individual stages may require different cluster configurations to efficiently execute the tasks. Those can be specified with the additional `prefect_config` option through which the user can set a specific dask task runner or the number of retries. Managing retries is essential for handling transient failures, such as connectivity issues, ensuring the stages can be re-executed without any manual interference if a failure occurs.

:::{figure} pipeline_config.png
:label: fig:pipeline_config
**Pipeline Configuration:** The pipeline consists of two stages: `echodataflow_open_raw` and `echodataflow_compute_TS` and will run on a local cluster.
:::

### Logging Configuration

By default, the outcomes of each stage are logged. The logs can be stored in `.json` or plain text files, and the format of the entries can be specified in the configuration file [Figure %s ](#fig:logging_config). The `json` allows searching through the logs for a specific key.

:::{figure} logging_config.png
:label: fig:logging_config
**Logging Configuration:** The logs are written to a local `echodataflow.log` file in a plain text form based on the format specification
:::



In [Figure %s ](#fig:log_output) we show an example of output logs for different processing stages.

:::{figure} log_output.png
:label: fig:log_output
**Log Output:** An example of an output log of file processing failure. The leading indicators for each entry are: date, process id, log level, followed by the module, function, line number, followed by the message.
:::

In Section [Workflow Logging](#Workflow-Logging), we provide more information on logging options.


## Workflow Execution
To convert a scientific pipeline into an executable Prefect workflow, one needs to organize its components into flows, sublfows, and tasks (the key objects of Prefect’s execution logic). Usually, the stages of a pipeline will be organized into flows and subflows, while the individual pieces of work within the stage will be organized into tasks. In practice, flows, subflows, and tasks are all Python functions, and they differ in how we want to execute them (e.g. concurrently/sequentially, w/o retries), and what we want to track during execution (e.g. input/outputs, state logging, etc.). In `echodataflow` we organize the typical echosounder processing steps into subflows (flows within the main workflow), while the operations on different files are individual tasks. We describe how functions are organized in the `open_raw` stage, which reads the files from raw format, parses the data, and writes them into a `.zarr` format. In [Figure %s ](#fig:flow_task) we show how the `echodataflow_open_raw` function is decorated into a flow, and is one of many subflows of the full workflow. This function processes all files. The contains a loop which iterates through all files and applies the `process_raw` function which operates on a single file and is decorated as a task. All tasks will be executed on the dask cluster. 

:::{figure} flow_task.png
:label: fig:flow_task
**Decorating Functions into Flows \& Tasks:** The flow `echdataflow_open_raw` calls the task `process_raw` (processing a single file) within a loop to process all files
:::

`
## Workflow Monitoring
One of the main advantages of using an orchestration framework is the features it provides to monitor the workflow execution. The integration with Prefect allows leveraging Prefect’s dashboard for monitoring the execution of the flows: Prefect UI. The dashboard can be run locally and within Prefect’s online managed system (Prefect Cloud). The local version provides an entirely open source framework for running and monitoring workflows. [Figure %s ](@fig:flow_sequence_expanded) shows the view of completed runs within the dashboard. The progress can be monitored while the flows are in progress. 

:::{figure} flow_sequence_expanded.png
:label: fig:flow_sequence_expanded
**Flow Runs:** Log of completed runs in Prefect UI. The stages (subflows) are executed sequentially. One can expand the view of an individual flow and see the tasks computed (asynchronously) within it.
:::

Further, one can also view the progress of the execution of the tasks on the dask cluster.

:::{figure} dask_progress.png
:label: fig: dask_progress
**Dask Dashboard:** The execution of the tasks on the dask cluster can also be monitored through the dask dashboard.
:::



## Workflow Logging
Processing large data archives requires a robust logging system to identify at which step and for which files the processing has failed and thus can set a path forward to resolve those issues: either through improving the robustness of the individual libraries performing the processing steps, or through identifying the artifacts of the data are incompatible with the existing pipeline. To address this, we are considering several approaches:
* Utilizing Dask Worker Streams for `echodataflow` logs: this approach configures Dask worker streams to handle echodataflow logs, which can be straightforward if exact log order is not crucial.
* Centralized Logging with AWS CloudWatch [ref]: this approach centralizes all logs for easy access and analysis. It can be useful when users are already utilizing AWS.
* Advanced Logging with Kafka and Elastic Stack[ref]: this approach leverages Kafka for log aggregation and Elastic Stack for log analysis and visualization, offering a robust solution for those with the necessary infrastructure, for example data center managers.

By default if logging is not configured, all the worker messages are directed to the application console. The order of logs may not be preserved since logs are written once control returns from the Dask workers to the main application.

## Workflow Deployment 

### Notebook
Echodataflow can be directly initiated within a Jupyter[@jupyter] notebook, which makes development interactive and provides a work environment familiar to researchers. One can see how the workflow is initiated within the Jupyter cell in [Figure %s ](#fig:notebook_start).

:::{figure} notebook_start.png
:label: fig:notebook_start
**Initiating `echodataflow` in a Jupyter Notebook:** Once one has a set of "recipe" configuration files, they can initiate the workflow in a notebook cell with the `echodataflow_start` command.
:::


We provide two demo notebooks: one for execution on a [local machine](https://github.com/OSOceanAcoustics/echodataflow/blob/1ac65fa0bfcdd01b151b98134b842364311059fd/docs/source/local/notebook.ipynb) and another one for execution on [AWS](https://github.com/OSOceanAcoustics/echodataflow/blob/1ac65fa0bfcdd01b151b98134b842364311059fd/docs/source/local/notebook.ipynb). 

### Docker
We facilitate the deployment of echodataflow on various platforms by building a docker image that can be spun up with all required components and the user can access the workflow dashboard on the corresponding port.

```
docker pull blackdranzer/echodataflow 

prefect server start

docker run --network="host" -e PREFECT_API_URL=http://host.docker.internal:4200/api blackdranzer/echodataflow
```

Upon execution, the user can readily access the Prefect UI dashboard and run workflows from there.
## Command Line Interface
We also provide a command-line interface which supports credential handling, and some additional useful features for managing the workflows.

### Adding Stages
Currently, most major functionalities in the echopype package are wrapped into stages: `open_raw`, `add_depth`, `add_location`, `compute_Sv`, `compute_TS`, `compute_MVBS`, `combine_echodata`, `frequency_differencing`, `apply_mask`. 

We provide tools to generate boilerplate template configuration based on the existing stages. Here is an example to add a stage:

```
echodataflow gs <stage_name>
```

For instance, to generate a boilerplate configuration for the `compute_Sv` stage, you would use:

```
echodataflow gs compute_Sv
```

This command creates a template configuration file for the specified stage, allowing you to customize and integrate it into your workflow. The generated file includes:
* a flow: this orchestrates the execution of all files that need to be processed, either concurrently or in parallel, based on the configuration.
* a task (helper function): this assists the flow by processing individual files.
### Rule Validation
Scientific workflows often have stages that cannot be executed until other stages have completed. Those conditions can be set through `echodataflow` client during the initialization process and are stored in a `echodataflow_rules.txt` file:

```
echodataflow_open_raw:echodataflow_compute_Sv
echodataflow_open_raw:echodataflow_combine_echodata
echodataflow_open_raw:echodataflow_compute_TS
echodataflow_combine_echodata:echodataflow_compute_Sv
echodataflow_compute_Sv:echodataflow_compute_MVBS
```

These rules dictate the sequence in which stages should be executed, ensuring that each stage waits for its dependencies to complete. They can be set through the `echodataflow rules -add...` command.

#### Aspect-Oriented Programming (AOP) for Rule Validation

In `echodataflow`, we adopt an aspect-oriented programming [ref] approach for rule validation. This is achieved using a decorator that can be applied to functions to enforce rules and log function execution details. The echodataflow decorator logs the entry and exit of a decorated function and modifies the function's arguments based on the execution context. This supports two types of execution: "TASK" and "FLOW".

Example Usage:

```
@echodataflow(processing_stage="StageA", type="FLOW")
def my_function(arg1, arg2):
    # Function code here
    pass
```

In the example, the echodataflow decorator ensures that the function `my_function` is executed within the context of "StageA" as a "FLOW", checking for dependencies and logging relevant information.

## Example Use Case: Processing Ship Survey Data from an Archive

We demonstrate a workflow of processing all acoustic data for the 2017 Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey through a few routine processing stages. The survey spans a period of 06/15/2017 - 09/13/2017, covering the entire west coast of the US and Canada. Figure 1(a) shows a typical map of the transect schedule of the survey. Raw acoustic data are collected continuously while the ship is in motion, resulting in a total of 3873 files collected with a total size of 93 GB. The raw files are archived by the NOAA NCEI Water Column Sonar Data Archive and are publicly accessible on their Amazon Web Services S3 bucket ([https://registry.opendata.aws/ncei-wcsd-archive/](https://registry.opendata.aws/ncei-wcsd-archive/)). The processing pipeline involves several steps:

* Convert raw files to cloud-native `.zarr` format following closely a community convention [@echopype, @convention]
* Combine multiple individual `.zarr` files within a continuous transect segment into a single zarr file
* Compute Sv: calibrate the measured acoustic backscatter data to volume backscattering strength (Sv, unit: dB re 1 m$^{-1}$)

Once data is converted to Sv, they are easy to manipulate, as the data are stored in an `xarray` data array and are smaller than that of the original data. The final dataset can be served as an analysis-ready data product to the community. It can be beneficial to store also intermediate datasets at different processing stages: for example, preserving the converted raw files in the standardized `.zarr` format allows users to regenerate any of the following stages with different groupings or resolution, without having to fetch and convert raw data again.

The execution of the workflow with echodataflow allowed us to monitor the progress of all files [Figure %s ](#fig:one_failed): 3872 files were successfully processed, and 1 failed. Most importantly, that failure did not block the execution of the other files, and a log was generated for the stage and the filename for which the error occurred. This experience serves as a confirmation that the transition from local development to a full production pipeline with `echodataflow` can indeed be smooth.


:::{figure} one_failed.png
:label: fig:one_failed
** Processing full 2017 Survey Data: 1/3873 files failed at the `open_raw` stage, but this did not impact the entire pipeline. As shown, other files were processed until the last stage.
::: 

## Future Development
Our immediate goal is to provide more example workflow recipes integrating other stages of echosounder data processing, such as machine learning prediction, training dataset generation, biomass estimation, interactive visualization, etc. We will demonstrate utilizing functionalities from a suite of open source Python packages (echoregions[@echoregions] for reading region annotations and creating corresponding masks, echopop[@echopop] for combining acoustic data with biological "ground truth" into biomass estimation, echoshader[@echoshader] for echogram and map dashboard visualization) in building workflows for the Pacific Hake Survey: both in a historical and near-realtime on-ship data processing context. We aim to streamline the stage addition process. We will further investigate how to improve memory management and caching between and within stages to optimize for different scenarios. There is growing interest in the fisheries acoustics community to share global, accessible, and interoperable (GAIN) datasets[ref], and to agree on community data standards and definitions of processing levels [@levels, @NASA_levels, ref]. As those mature we will align them with existing stages in echodataflow, which will support building interoperable datasets whose integration will push us to study bigger and more challenging questions in fisheries acoustics.



## Beyond Fisheries Acoustics

Echodataflow was designed to facilitate fisheries acoustics workflows, but the structure can be adapted to data processing pipelines in other scientific communities. The key aspects are to identify the potential stages of the workflows and associated Python packages/functions that implement them, and to design the structure of the configuration files. The other aspects such as logging, deployment, monitoring, new-stage integration are domain-agnostic. Processing pipelines that require manipulation of large labeled arrays can directly benefit from the dask cluster integration and are prevalent in the research community. Our use case of regrouping data based on time segments is a common need within scientific pipelines in which the file unit level of the instrument is not aligned with the unit level of analysis, and requires further reorganization and potential resampling and regridding along certain coordinates. We hope it can serve as a guide on how to build configurable, reproducible, and scalable workflows in new scientific areas.

## Acknowledgements:
We thank NOAA Fisheries Engineering and Acoustic Technologies team: Julia Clemons, Alicia Billings, Rebecca Thomas, Elizabeth Phillips for introducing us to the Pacific Hake Survey operations and biomass estimation workflow.

## Funding:
NOAA Fisheries, eScience Institute
