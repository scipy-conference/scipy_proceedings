---
# Ensure that this title is the same as the one in `myst.yml`
title: "echodataflow: Recipe-based Fisheries Acoustics Workflow Orchestration"
abstract: |
  With the influx of large data from multiple instruments and experiments, scientists are wrangling complex data pipelines that are context-dependent and non-reproducible. We demonstrate how we leverage Prefect, a modern orchestration framework, to facilitate fisheries acoustics data processing. We built a Python package `echodataflow` which 1) wraps common echosounder data processing steps in a few lines of code; 2) allows users to specify workflows and their parameters through editing text “recipes” which provide transparency and reproducibility of the pipelines; 3) supports scaling of the workflows while abstracting the computational infrastructure; 4) provides monitoring and logging of the workflow progress. Under the hood, echodataflow uses Prefect to execute the workflows while providing a domain-friendly interface to facilitate diverse fisheries acoustics use cases. We demonstrate the features through a typical ship survey data processing pipeline. 
---

## Fisheries Acoustics Workflows

While traditionally fisheries acoustics scientists have had a go-to tool and procedures for their data processing and analysis, now they are facing a lot of choices in designing their workflows. The field has also become very interdisciplinary and it involves people from different backgrounds (physics, biology, oceanography, acoustics, signal processing, machine learning, software engineering, etc.) and with different levels of experience. [Figure%s](#fig:workflow_variations) shows the many variations of workflows that can be defined based on the data collection scheme, the use case, the data storage and computing infrastructure options. We discuss these in more detail in the next sections.

:::{figure} workflow_variations.png
:label: fig:workflow_variations
**`echodataflow` Workflow_Variations:** Various use cases drive different needs for data storage and computing infrastructure. Options are abundant but adapting workflows across them is not trivial.
:::


## Echodataflow Overview
At the center of `echodataflow` design is the notion that a workflow can be configured through a set of recipes (.yaml files) that specify the pipeline, data storage, and logging details. The idea draws inspiration from the Pangeo-Forge Project [@pangeo-forge] which facilitates the Extraction, Transformation, Loading (ETL) of earth science geospatial datasets from traditional repositories to analysis-ready, cloud-optimized (ARCO) data stores [ref]. The pangeo-forge recipes provide a model of how the data should be accessed and transformed, and the project has garnered numerous recipes from the community. While Pangeo-Forge’s focus is on transformation from `.netcdf` [ref] and `hdf5` [ref] formats to `zarr`, echodataflow’s aim is to support full echosounder data processing and analysis pipelines: from instrument raw data formats to biological products. Echodataflow leverages Prefect to abstract data and computation management. In  we provide an overview of echodataflow’s framework. At the center we see several steps from an echosounder data processing pipeline: `open_raw`, `combine_echodata`, `compute_Sv`, `compute_MVBS`. All these functions exist in the echopype package, and are wrapped by echodataflow into predefined stages. Prefect executes the stages on a dask cluster which can be started locally or can be externally set up. These echopype functions already support distributed operations with dask thus the integration with Prefect within echodataflow is natural. Dask clusters can be set up on a variety of platforms: local, cloud, kubernetes [ref], HPC cluster via `dask-jobqueue` [ref], etc. and allow abstraction from the computing infrastructure. Input, intermediate, and final data sets can live in different storage systems (local/cloud, public/private) and Prefect’s block feature provides seamless, provider-agnostic, and secure integration. Workflows can be executed and monitored through Prefect’s dashboard, while logging of each function is handled by echodataflow.

:::{figure} echodataflow.png
:label: fig:echodataflow
**`echodataflow` Framework:** The above diagram provides an overview of the echodataflow framework: the task is to fetch raw files from a local filesystem/cloud archive, process them through several stages of an echosounder data workflow using a cluster infrastructure, and store intermediate and final products. Echodataflow allows the workflow to be executed based on text configurations, and logs are generated for the individual processing stages. Prefect handles the distribution of the tasks on the cluster, and provides tools for monitoring the workflow runs. 
:::

### Why Prefect?
We chose Prefect among other Python workflow orchestration tools such as Apache Airflow[ref], Dagster[ref], Argo[ref], Luigi[ref]. We provide a few reasons for our decision:
Prefect accepts dynamic workflows which are specified at runtime and do not require to follow a Directed Acyclic Graph, which can be restricting and difficult to implement.
In Prefect, Python functions are first class citizens, thus building a Prefect workflow does not deviate a lot from traditional science workflows built out of functions.
Prefect integrates with a dask cluster, and echopype processing functions are already using dask to scale operations
Prefect’s code runs similarly locally as well as on cloud services. 
Prefect’s monitoring dashboard is open source, can be run locally, and is intuitive to use.

We next describe in more detail the components of the workflow lifecycle.

## Workflow Configuration
The main goal of echodataflow is to allow users to configure an echosounder data processing pipeline through editing configuration “recipe” templates. Echodataflow can be configured through three templates: datastore.yaml which handles the data storage decisions, pipeline.yml which specifies the processing stages, and logging.yaml which sets the logging format. 

### Data Storage Configuration
In [Figure%s](#fig:datastore_config): datstore.yaml we provide an example of a data store configuration for a ship survey. In this scenario we want to process data from the Joint U.S.-Canada Integrated Ecosystem and Pacific Hake Acoustic Trawl Survey which is being publicly shared on an AWS S3 bucket by NOAA National Center for Environmental Information Acoustics Archive (NCEA)[ref]. The archive contains data from many surveys dating back to 1991 and contains ~280TB of data. The additional parameters referring to ship, survey, and sonar model names allow to parse the files to those belonging only to the survey of interest. The output is set to a private S3 bucket belonging to the user (i.e. an AWS account different from the input one), and the credentials are passed through a block_name. The survey contains ~4000 files, and one can set the group option to combine the files into survey-specific groups: based on transect information provided in the transect_group.txt file. One can further use regular expressions to subselect new groups based on needs. 

:::{figure} datastore_config.png
:label: fig:datastore_config
**Data Storage Configuration:** The input is data from a ship survey data on and S3 data on the NCEA archive. The output goes to an S3 bucket on another AWS account. Credentials are stored in blocks.
:::


### Pipeline Configuration
The pipeline configuration file’s purpose is to list the stages of the processing pipeline and the computational set up for their execution. In [Figure%s](#fig:pipeline_config) we provide a short example with two stages: `open_raw` (which fetches data from the cloud, converts it from proprietary raw format to a cloud native standardized output) and `compute_TS` (computes target strength: a popular quantity in fisheries acoustics). Each stage is executed as a separate Prefect subflow (a component of a Prefect workflow), and one can specify additional options on whether to store input and output files.  Echodataflow requires access to a dask cluster: one can either create one on the fly by setting the `use_local_dask` to `true`, or one can provide a link to an already running cluster. Individual stages may require different cluster configurations to efficiently execute the tasks. Those can be specified with the additional `prefect_config` option through which the user can set a specific dask task runner or the number of retries. Configuring retries is beneficial for handling transient failures, such as connectivity issues, ensuring the stages can be re-executed without any manual interference if a failure occurs.

:::{figure} pipeline_config.png
:label: fig:pipeline_config
**Pipeline Configuration:** The pipeline consists of two stages: `echodataflow_open_raw` and `echodataflow_compute_TS` and will run on a local cluster
:::

### Logging Configuration

By default, the outcomes of each stage get logged. The logs can be stored in `.json` or plain text files, and the format of the entries can be specified in the configuration file [Figure%s](#fig:logging_config). The `json` allows searching through the logs for a specific key.

:::{figure} logging_config.png
:label: fig:logging_config
**Logging Configuration:** The logs are written to a local `echodataflow.log` file in a plain text form based on the format specification
:::



In [Figure%s](#fig:log_output) we show an example of output logs for different processing stages.

:::{figure} log_output.png
:label: fig:log_output
**Log Output:** The leading indicators are: date, process id, log level, followed by the module, function, line number, and message.
:::

## Workflow Execution
To convert a scientific pipeline into an executable Prefect workflow, one needs to organize its components into flows, sublfows, and tasks (the key objects of Prefect’s execution logic). Usually, the stages of a pipeline will be organized into flows and subflows, while the individual pieces of work within the stage will be organized into tasks. In practice, flows, subflows, and tasks are all Python functions, and they differ in how we want to execute them (e.g. concurrently/sequentially, w/o retries), and what we want to track during execution (e.g. input/outputs, state logging, etc.). In echodataflow we organize the typical echosounder processing steps into subflows (flows within the main workflow), while the operations on different files are individual tasks. We describe how functions are organized in the `open_raw` stage, which reads the files from raw format, parses the data, and writes them into a `.zarr` format. In [Figure%s](#fig:flow_task) we show how the `echodataflow_open_raw` function is decorated into a flow, and is one of many subflows of the full workflow. This function processed all files. Within the function we have a loop which goes through all the files and applies the process_raw function which operates on a single file and is decorated as a task. All tasks will be executed on the dask cluster. 

:::{figure} flow_task.png
:label: fig:flow_task
**Decorating Functions into Flows \& Tasks:** The flow `echdataflow_open_raw` calls the task `process_raw` (processing a single file) within a loop to process all files
:::

`
## Workflow Monitoring
One of the main advantages of using an orchestration framework is the features it provides to monitor the workflow execution. The integration with Prefect allows leveraging Prefect’s dashboard for monitoring the execution of the flows: Prefect UI. The dashboard can be run locally and within Prefect’s online managed system (Prefect Cloud). The local version provides an entirely open source framework for running and monitoring workflows. [Figure%s](#fig:echodataflow_flow_runs) shows the view of completed runs within the dashboard. 

:::{figure} echodataflow_flow_runs.png
:label: fig:echodataflow_flow_runs
**Flow Runs:** Log of completed runs in Prefect UI
:::


:::{figure} task_progress.png
:label: fig:task_progress
**Task Progress** 
:::


## Workflow Logging
Processing large data archives requires a robust logging system to identify at which step and for which files the processing has failed and thus can set a path forward to resolve those issues: either through improving the robustness of the individual libraries performing the processing steps, or through identifying the artifacts of the data are incompatible with the existing pipeline. To address this, we are considering several approaches:


Utilizing Dask Worker Streams for Echodataflow Logs: This approach configures Dask worker streams to handle echodataflow logs, which can be straightforward if exact log order is not crucial.
Centralized Logging with AWS CloudWatch [ref]: This approach centralizes all logs for easy access and analysis.
Advanced Logging with Kafka and Elastic Stack: This approach leverages Kafka for log aggregation and Elastic Stack for log analysis and visualization, offering a robust solution for those with the necessary infrastructure.
By default if logging is not configured, all the worker messages are directed to application console. The order of logs may not be preserved since logs are written once control returns from the Dask workers to the main application.

## Workflow Deployment 

### Notebook
Echodataflow can be directly initiated within a Jupyter notebook, which makes development interactive and provides a work environment familiar to researchers. One can see how the workflow is initiated within the Jupyter cell in 

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

These rules dictate the sequence in which stages should be executed, ensuring that each stage waits for its dependencies to complete.

There are two options:
* add a rule interactively
  ```
  echodataflow rules --add
  ```
  This command will prompt the user to input a new rule in the `parent_flow:child_flow format`, for example `echodataflow_compute_MVBS:echodataflow_frequency_differencing`. 

* import rules from a file:
  ```
   echodataflow rules --add-from-file path/to/rules.txt
  ```

#### Aspect-Oriented Programming (AOP) in echodataflow

In echodataflow, we adopt an aspect-oriented programming approach for rule validation. This is achieved using a decorator that can be applied to functions to enforce rules and log function execution details. The echodataflow decorator logs the entry and exit of a decorated function and modifies the function's arguments based on the execution context. This supports two types of execution: "TASK" and "FLOW".

Example Usage:

```
@echodataflow(processing_stage="StageA", type="FLOW")
def my_function(arg1, arg2):
    # Function code here
    pass
```

In the example, the echodataflow decorator ensures that the function `my_function` is executed within the context of "StageA" as a "FLOW", checking for dependencies and logging relevant information.


## Future Development
Our immediate goal is to provide more example workflow recipes integrating other stages of echosounder data processing such as machine learning prediction, label dataset generation (echoregions[ref]), biomass estimation (echopop[ref]), interactive visualization integration (echoshader), etc. We plan to explore more use case scenarios such as near-realtime on-ship processing. We will investigate how to improve memory management and caching between flows. We further aim to streamline the stage addition process. We hope that as the community agrees one data processing levels [ref], we can align them with existing stages in echodataflow, which will support building interoperable data sets whose integration will push us to study bigger and more challenging questions in fisheries acoustics.


## Beyond Fisheries Acoustics

Echodataflow was designed to facilitate echosounder data processing workflows, but the structure can be adapted to data processing pipelines in other scientific communities. The key aspects are to identify the potential stages of the workflow and associated Python packages/steps that implement them, and to design the structure of the configuration files. The other aspects such as logging, deployment, monitoring, new-stage integration are domain-agnostic. Processing pipelines that require manipulation of large labeled arrays can directly benefit from the dask cluster integration and are prevalent in the researrch community. Our use case of regrouping data based on time segments is a common need within scientific pipelines in which the file unit level of the instrument is not aligned with the unit level of analysis, and requires further reorganization and potential resampling and regridding along certain coordinates. We hope it can serve as a guide on how to build configurable, reproducible, and scalable workflows in new scientific areas.

## Acknowledgements:
We thank NOAA Fisheries Engineering and Acoustic Technologies team: Julia Clemons, Alicia Billings, Rebecca Thomas, Elizabeth Phillips for introducing us to the Pacific Hake Survey operations and collaborating with us to improve fisheries acoustics workflows.

## Funding:
NOAA Fisheries, eScience Institute



## Figures



















:::{figure} case_study.png
:label: fig:case_study
**Hake Survey Processing:** We provide executation times and data product sizes for processing 2017 survey data on an Jetstream machine.
:::



## Bibliographies, citations and block quotes

Bibliography files and DOIs are automatically included and picked up by `mystmd`.
These can be added using pandoc-style citations `[@doi:10.1109/MCSE.2007.55]`
which fetches the citation information automatically and creates: [@doi:10.1109/MCSE.2007.55].
Additionally, you can use any key in the BibTeX file using `[@citation-key]`,
as in [@hume48] (which literally is `[@hume48]` in accordance with
the `hume48` cite-key in the associated `mybib.bib` file).
Read more about [citations in the MyST documentation](https://mystmd.org/guide/citations).

If you wish to have a block quote, you can just indent the text, as in:

> When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication.
>
> -- @hume48

Other typography information can be found in the [MyST documentation](https://mystmd.org/guide/typography).

### DOIs in bibliographies

In order to include a DOI in your bibliography, add the DOI to your bibliography
entry as a string. For example:

```{code-block} bibtex
:emphasize-lines: 7
:linenos:
@book{hume48,
  author    =  "David Hume",
  year      = {1748},
  title     = "An enquiry concerning human understanding",
  address   = "Indianapolis, IN",
  publisher = "Hackett",
  doi       = "10.1017/CBO9780511808432",
}
```

### Citing software and websites

Any paper relying on open-source software would surely want to include citations.
Often you can find a citation in BibTeX format via a web search.
Authors of software packages may even publish guidelines on how to cite their work.

For convenience, citations to common packages such as
Jupyter [@jupyter],
Matplotlib [@matplotlib],
NumPy [@numpy],
pandas [@pandas1; @pandas2],
scikit-learn [@sklearn1; @sklearn2], and
SciPy [@scipy]
are included in this paper's `.bib` file.

In this paper we not only terraform a desert using the package terradesert [@terradesert], we also catch a sandworm with it.
To cite a website, the following BibTeX format plus any additional tags necessary for specifying the referenced content is recommended.
If you are citing a team, ensure that the author name is wrapped in additional braces `{Team Name}`, so it is not treated as an author's first and last names.

```{code-block} bibtex
:emphasize-lines: 2
:linenos:
@misc{terradesert,
  author = {{TerraDesert Team}},
  title  = {Code for terraforming a desert},
  year   = {2000},
  url    = {https://terradesert.com/code/},
  note   = {Accessed 1 Jan. 2000}
}
```

## Source code examples

No paper would be complete without some source code.
Code highlighting is completed if the name is given:

```python
def sum(a, b):
    """Sum two numbers."""

    return a + b
```

Use the `{code-block}` directive if you are getting fancy with line numbers or emphasis. For example, line-numbers in `C` looks like:

```{code-block} c
:linenos: true

int main() {
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
    return 0;
}
```

Or a snippet from the above code, starting at the correct line number, and emphasizing a line:

```{code-block} c
:linenos: true
:lineno-start: 2
:emphasize-lines: 3
    for (int i = 0; i < 10; i++) {
        /* do something */
    }
```

You can read more about code formatting in the [MyST documentation](https://mystmd.org/guide/code).

## Figures, Equations and Tables

It is well known that Spice grows on the planet Dune [@Atr03].
Test some maths, for example $e^{\pi i} + 3 \delta$.
Or maybe an equation on a separate line:

```{math}
g(x) = \int_0^\infty f(x) dx
```

or on multiple, aligned lines:

```{math}
\begin{aligned}
g(x) &= \int_0^\infty f(x) dx \\
     &= \ldots
\end{aligned}
```

The area of a circle and volume of a sphere are given as

```{math}
:label: circarea

A(r) = \pi r^2.
```

```{math}
:label: spherevol

V(r) = \frac{4}{3} \pi r^3
```

We can then refer back to Equation {ref}`circarea` or
{ref}`spherevol` later.
The `{ref}` role is another way to cross-reference in your document, which may be familiar to users of Sphinx.
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas[^footnote-1] diam turpis, placerat[^footnote-2] at adipiscing ac,
pulvinar id metus.

[^footnote-1]: On the one hand, a footnote.
[^footnote-2]: On the other hand, another footnote.

:::{figure} figure1.png
:label: fig:stream
This is the caption, sandworm vorticity based on storm location in a pleasing stream plot. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/arrays/streamplot.html).
:::

:::{figure} figure2.png
:label: fig:em
This is the caption, electromagnetic signature of the sandworm based on remote sensing techniques. Based on example in [matplotlib](https://matplotlib.org/stable/plot_types/stats/hist2d.html).
:::

As you can see in @fig:stream and @fig:em, this is how you reference auto-numbered figures.
To refer to a sub figure use the syntax `@label [a]` in text or `[@label a]` for a parenhetical citation (i.e. @fig:stream [a] vs [@fig:stream a]).
For even more control, you can simply link to figures using `[Figure %s](#label)`, the `%s` will get filled in with the number, for example [Figure %s](#fig:stream).
See complete documentation on [cross-references](https://mystmd.org/guide/cross-references).

```{list-table} This is the caption for the materials table.
:label: tbl:materials
:header-rows: 1
* - Material
  - Units
* - Stone
  - 3
* - Water
  - 12
* - Cement
  - {math}`\alpha`
```

We show the different quantities of materials required in
@tbl:materials.

Unfortunately, markdown can be difficult for defining tables, so if your table is more complex you can try embedding HTML:

:::{table} Area Comparisons (written in html)
:label: tbl:areas-html

<table>
<tr><th rowspan="2">Projection</th><th colspan="3" align="center">Area in square miles</th></tr>
<tr><th align="right">Large Horizontal Area</th><th align="right">Large Vertical Area</th><th align="right">Smaller Square Area<th></tr>
<tr><td>Albers Equal Area   </td><td align="right"> 7,498.7   </td><td align="right"> 10,847.3  </td><td align="right">35.8</td></tr>
<tr><td>Web Mercator        </td><td align="right"> 13,410.0  </td><td align="right"> 18,271.4  </td><td align="right">63.0</td></tr>
<tr><td>Difference          </td><td align="right"> 5,911.3   </td><td align="right"> 7,424.1   </td><td align="right">27.2</td></tr>
<tr><td>Percent Difference  </td><td align="right"> 44%       </td><td align="right"> 41%       </td><td align="right">43%</td></tr>
</table>
:::

or if you prefer LaTeX you can try `tabular` or `longtable` environments:

```{raw} latex
\begin{table*}
  \begin{longtable*}{|l|r|r|r|}
  \hline
  \multirow{2}{*}{\bf Projection} & \multicolumn{3}{c|}{\bf Area in square miles} \\
  \cline{2-4}
   & \textbf{Large Horizontal Area} & \textbf{Large Vertical Area} & \textbf{Smaller Square Area} \\
  \hline
  Albers Equal Area   & 7,498.7   & 10,847.3  & 35.8  \\
  Web Mercator        & 13,410.0  & 18,271.4  & 63.0  \\
  Difference          & 5,911.3   & 7,424.1   & 27.2  \\
  Percent Difference  & 44\%      & 41\%      & 43\%  \\
  \hline
  \end{longtable*}

  \caption{Area Comparisons (written in LaTeX) \label{tbl:areas-tex}}
\end{table*}
```

Perhaps we want to end off with a quote by Lao Tse[^footnote-3]:

> Muddy water, let stand, becomes clear.

[^footnote-3]: $\mathrm{e^{-i\pi}}$
