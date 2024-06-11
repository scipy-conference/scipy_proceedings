---
# Ensure that this title is the same as the one in `myst.yml`
title: Orchestrating Bioinformatics Workflows Across a Heterogeneous Toolset with Flyte
abstract: |
  While Python excels at prototyping and iterating quickly, it’s not always performant enough for whole-genome scale data processing. Flyte, an open-source Python-based workflow orchestrator, presents an excellent way to tie together the myriad tools required to run bioinformatics workflows. Flyte is a Kubernetes native orchestrator, meaning all dependencies are captured and versioned in container images. It also allows you to define custom types in Python representing genomic datasets, enabling a powerful way to enforce compatibility across tools. Computational biologists, or any scientists processing data with a heterogeneous toolset, stand to benefit from a common orchestration layer that is opinionated yet flexible.
---

## Introduction

Since the sequencing of the human genome [@doi:10.1126/science.1058040], and as other wet lab processes have scaled in the last couple decades, computational approaches to understanding the living world have exploded. The firehose of data generated from all these experiments led to algorithms and heuristics developed in low-level high-performance languages such as C and C++. Later on, industry standard collections of tools like GATK [@doi:10.1002/0471250953.bi1110s43] were written in Java. A number of less performance intensive offerings such as MultiQC [@doi:10.1093/bioinformatics/btw354] are written in Python; and R is used extensively where it excels: visualization and statistical modeling. Finally, newer deep-learning models and Rust based components are entering the fray.

Different languages also come with different dependencies and approaches to dependency management, interpreted versus compiled languages for example handle this very differently. They also need to be installed correctly and available in the user’s PATH for execution. Moreover, compatibility between different tools in bioinformatics often falls back on standard file types expected in specific locations on a traditional filesystem. In practice this means searching through datafiles or indices available at a particular directory and expecting a specific naming convention or filetype.

In short, bioinformatics suffers from the same reproducibility crisis [@doi:10.1038/533452a] as the broader scientific landscape. Standardizing the interfaces, orchestration and encapsulation of these different tools in a flexible and future-proof way is of paramount importance on this unrelenting march towards larger and larger datasets.

## Methods

Solving these problems using Flyte is accomplished by capturing dependencies flexibly with dynamically generated container images, defining custom types to enforce at the task boundary, and wrapping tools in Flyte tasks. Before diving into the finer points, a brief primer on Flyte is required. While the [introduction](https://docs.flyte.org/en/latest/introduction.html) in the docs is a worthwhile read before continuing, here is a more salient *hello world* example:

```python
from flytekit import task, workflow

@task
def greet() -> str:
  return "Hello"

@task
def say(greeting: str, name: str) -> str:
    return f"{greeting}, {name}!"

@workflow
def hello_world_wf(name: str = 'world') -> str:
    greeting = greet()
    res = say(greeting=greeting, name=name)
    return res
```

Tasks are the most basic unit of work in Flyte. They are pure-python functions and their interface is strongly typed in order to compose the workflow. The workflow itself is actually a DSL that statically compiles a directed-acyclic graph (DAG) based on the dependencies between the different tasks. There are different flavors of tasks and workflows as we'll see later, but this is the core concept.

### Images

While it's possible to run Flyte tasks and workflows locally in a Python virtual environment, production executions in Flyte run on a Kubernetes cluster. As a k8s native orchestrator, all tasks run in their own (typically single-container) pods using whatever image is specified in the `@task` decorator. Capturing dependencies in container images has been a gold-standard for some time now, but this is taken a step further with [ImageSpec](https://docs.flyte.org/en/latest/user_guide/customizing_dependencies/imagespec.html#imagespec). ImageSpec lets you easily define a base image and additional dependencies right alongside your task and workflow code. These additional dependencies can include apt, python or conda packages. While [envd](https://github.com/tensorchord/envd) is the default builder, other backends like Docker are available should the need arise. 

These ImageSpec definitions are loosely coupled to your workflow code and are built automatically when tasks are registered or run on a Flyte cluster. ImageSpec reduces the complexity inherent in manually authoring a Dockerfile and enables a more streamlined approach to building images without the need for an additional build step and configuration update to reference the latest image. This coupling and reduced complexity makes it easier to build single-purpose images instead of throwing everything into one monolithic image - here are a couple:

```python
main_img = ImageSpec(
    name="main",
    packages=["flytekit"],
    python_version="3.11",
    conda_channels=["bioconda"],
    conda_packages=[
        "samtools",
        "bcftools",
        "bwa",
        "fastp",
        "hisat2",
        "bowtie2",
        "gatk4",
        "fastqc",
        "htslib",
    ],
    registry="ghcr.io/pryce-turner",
)

folding_img = ImageSpec(
    name="protein",
    platform="linux/amd64",
    python_version="3.11",
    packages=["flytekit", "transformers", "torch"],
    conda_channels=["bioconda", "conda-forge"],
    conda_packages=[
        "prodigal",
        "biotite",
        "biopython",
        "py3Dmol",
        "matplotlib",
    ],
    registry="ghcr.io/pryce-turner",
)
```

The `main` image has a lot of functionality and could arguably be pared down. It contains a number of very common low-level tools, along with GATK and a couple aligners. The `protein` image on the other hand, only contains a handful of tools related to a very specific protein folding and visualization workflow. They can be specified in the task decorator:

```python
@task(container_image=folding_img)
def predict_structure(seq: str):
  fold_protein(seq)
```

This image will be built and uploaded when your tasks and workflows are registered to a Flyte cluster. 

### Datatypes

Having rich data types to enforce compatibility at the task boundary is essential to these wrapped tools working together. Flyte supports arbitrary data types through Python’s dataclasses library. Data types representing raw reads and alignment files allow us to reason about these files and their metadata more easily across tasks, as well as enforce naming conventions. Importantly, Flyte abstracts the object store, allowing you to load these assets into pods wherever is most convenient for your tool. This not only makes it easier to work with these files, but also safer as you’re working with ephemeral storage during execution instead of a full production filesystem. Here is the Reads dataclass:

```python
@dataclass
class Reads(DataClassJSONMixin):

    sample: str
    filtered: bool | None = None
    filt_report: FlyteFile | None = None
    uread: FlyteFile | None = None
    read1: FlyteFile | None = None
    read2: FlyteFile | None = None

    def get_read_fnames(self):
        filt = "filt." if self.filtered else ""
        return (
            f"{self.sample}_1.{filt}fastq.gz",
            f"{self.sample}_2.{filt}fastq.gz",
        )

    def get_report_fname(self):
        return f"{self.sample}_fastq-filter-report.json"

    @classmethod
    def make_all(cls, dir: Path):
      ...
```

We're capturing a few important aspects: whether the reads have been filtered and the results of that operation, as well as if they're [paired-end](https://thesequencingcenter.com/knowledge-base/what-are-paired-end-reads/) reads or not. The `make_all` function body has been omitted for brevity, but it accepts a directory and returns a list of these objects based on it's contents. In the other direction, a `get_read_fnames` method is defined to standardize naming conventions based on The 1000 Genomes Project [guidelines](https://www.internationalgenome.org/faq/what-are-your-filename-conventions). FlyteFile, along with FlyteDirectory, represent a file or directory in a Flyte aware context. These types handle serialization and deserialization into and out of the object store. They re-implement a number of common filesystem operations like `open()`, which returns a streaming handle, for example. Simply returning a FlyteFile from a task will automatically upload it to whatever object store is defined. This unassuming piece of functionality is one of Flyte's key strengths: abstracting data management so researchers can focus on their task code.

Since dataflow in Flyte is a first-class construct, having well defined inputs and outputs at the task boundary makes authoring workflows that much more reliable. Here is another dataclass representing a downstream alignment from the reads above:

```python
@dataclass
class Alignment(DataClassJSONMixin):

    sample: str
    aligner: str
    format: str | None = None
    alignment: FlyteFile | None = None
    alignment_idx: FlyteFile | None = None
    alignment_report: FlyteFile | None = None
    sorted: bool | None = None
    deduped: bool | None = None
    bqsr_report: FlyteFile | None = None

    def _get_state_str(self):
        state = f"{self.sample}_{self.aligner}"
        if self.sorted:
            state += "_sorted"
        if self.deduped:
            state += "_deduped"
        return state

    def get_alignment_fname(self):
        return f"{self._get_state_str()}_aligned.{self.format}"

    @classmethod
    def make_all(cls, dir: Path):
      ...
```

More information related to alignments is being captured here. However, the standard methods for interacting with a local filesystem remain. In the next section we'll look at tasks that actually carry out this alignment.


### Tasks

While Flyte tasks are written in Python, there are a couple of ways to wrap arbitrary tools. ShellTasks are one such way, allowing you to define scripts as multi-line strings in Python. For added flexibility around packing and unpacking data types before and after execution, Flyte also ships with a `subproc_execute` function which can be used in vanilla Python tasks. Finally, arbitrary images can be used via a [ContainerTask](https://docs.flyte.org/en/latest/user_guide/customizing_dependencies/raw_containers.html#raw-containers) and avoid any `flytekit` dependency altogether.

Here is a ShellTask creating a `bowtie2` index directory from a genome reference file.

```python
bowtie2_index = ShellTask(
    name="bowtie2-index",
    debug=True,
    requests=Resources(cpu="4", mem="10Gi"),
    metadata=TaskMetadata(retries=3, cache=True, cache_version=ref_hash),
    container_image=main_img,
    script="""
    mkdir {outputs.idx}
    bowtie2-build {inputs.ref} {outputs.idx}/bt2_idx
    """,
    inputs=kwtypes(ref=FlyteFile),
    output_locs=[
        OutputLocation(var="idx", var_type=FlyteDirectory, location="/tmp/bt2_idx")
    ],
)
```

This task uses the `main_img` defined above; it also accepts a FlyteFile and outputs a FlyteDirectory. Another important feature to highlight here is [caching](https://docs.flyte.org/en/latest/user_guide/development_lifecycle/caching.html#caching), which saves us valuable compute for inputs that rarely change. Since the alignment index for a particular aligner only needs to be generated once for a given reference, we've set the `cache_version` to a hash of the reference's URI. As long as the reference exists at that URI, this bowtie indexing task will complete immediately and return that index. 

To perform the actual alignment, a regular python task is used with a Flyte-aware subprocess function to call the bowtie CLI. 

```python
@task(container_image=main_img, requests=Resources(cpu="4", mem="10Gi"))
def bowtie2_align_paired_reads(idx: FlyteDirectory, fs: Reads) -> Alignment:
    idx.download()
    logger.debug(f"Index downloaded to {idx.path}")
    ldir = Path(current_context().working_directory)

    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir.joinpath(alignment.get_alignment_fname())
    rep = ldir.joinpath(alignment.get_report_fname())
    logger.debug(f"Writing alignment to {al} and report to {rep}")

    cmd = [
        "bowtie2",
        "-x",
        f"{idx.path}/bt2_idx",
        "-1",
        fs.read1,
        "-2",
        fs.read2,
        "-S",
        al,
    ]
    logger.debug(f"Running command: {cmd}")

    result = subproc_execute(cmd)

    # Bowtie2 indexing writes stats to stderr
    with open(rep, "w") as f:
        f.write(result.error)

    setattr(alignment, "alignment", FlyteFile(path=str(al)))
    setattr(alignment, "alignment_report", FlyteFile(path=str(rep)))
    setattr(alignment, "sorted", False)
    setattr(alignment, "deduped", False)

    return alignment
```

Since Python tasks are the default task type, they're the most feature rich and stable. The main advantage to using one here is to unpack the inputs and construct the output type. 

## Results

A real world alignment workflow exemplifies how to tie all these disparate parts together. Starting with a directory containing raw FastQ files, we'll perform QC, filtering, index generation, alignment and conclude with a final report of all the steps. Here's the code:

```python
@workflow
def simple_alignment_wf(seq_dir: FlyteDirectory = seq_dir_pth) -> FlyteFile:
    
    # Generate FastQC reports and check for failures
    fqc_out = fastqc(seq_dir=seq_dir)
    samples = prepare_raw_samples(seq_dir=seq_dir)

    # Map out filtering across all samples and generate indices
    filtered_samples = map_task(pyfastp)(rs=samples)

    fqc_out >> filtered_samples

    bowtie2_idx = bowtie2_index(ref=ref_loc)

    # Compare alignment results using two different aligners in a dynamic task
    sams = bowtie2_align_samples(idx=bowtie2_idx, samples=filtered_samples)

    # Generate final multiqc report with stats from all steps
    return render_multiqc(fqc=fqc_out, filt_reps=filtered_samples, sams=sams)
```

To help make sense of the flow of tasks, here is a screenshot from the Flyte UI that offers a visual representation of the different steps:

:::{figure} nodes.png
:label: fig:nodes
List of workflow nodes and their status
:::

FastQC, an extremely common QC tool written in **Java**, starts off the workflow by generating a report for all FastQ files in a given directory. That directory is then turned into Reads objects via the `prepare_raw_samples` task. Those samples are then passed to `fastp` for adapter removal and filtering of duplicate or low quality reads. Fastp is a **C/C++** tool that is wrapped in a python task which accepts a single Reads object. This task is then used in a [map task](https://docs.flyte.org/en/latest/user_guide/advanced_composition/map_tasks.html#map-task) to parallelize the processing of however many discrete samples were present in the input directory. Since there is no implicit dependency between filtering and QC, we make this relationship explicit with the `>>` operator. Next, `bowtie2_index` generates an index if one is not already cached. Bowtie2 is primarily written in **C++**, with some parts implemented in **Perl**. Since the bowtie2 alignment task processes samples one at a time, it was wrapped in a [dynamic workflow](https://docs.flyte.org/en/latest/user_guide/advanced_composition/dynamic_workflows.html#dynamic-workflows) to process a list of inputs. Dynamics are another parallelism construct, similar to map tasks with some key [differences](https://flyte.org/blog/map-tasks-in-flyte). Lastly, MultiQC, a common QC report aggregator written in **Python**, produces a final report of all the different steps in the workflow. Here is the workflow visualized as a DAG:

:::{figure} dag.png
:label: fig:dag
Workflow DAG
:::

Finally, it's helpful to inspect a timeline of the execution which highlights a few things. Since this workflow was run several times over the course of capturing these figures, the `fastqc` and `bowtie2_index` tasks were cached in previous runs. It's also clear from this figure which tasks were run in parallel in contrast to those which had dependencies on upstream outputs. Finally, the overall runtime is broken down into it's separate parts. Since this was run on all test data, everything executed fairly quickly. 

:::{figure} timeline.png
:label: fig:tl
Execution timeline
:::


## Conclusion

Different steps in a bioinformatics pipeline often require tools with significantly different characteristics. As such, different languages are employed where their strengths are best leveraged. Regardless of which language or framework is used, ImageSpec captures those dependencies in an OCI-compliant image for use in Flyte workflows and beyond in a very ergonomic way. Defining dataclasses with FlyteFiles and additional metadata frees the data flow from the trappings of a traditional filesystem while Flyte handles serialization so we can easily operate in a cloud native paradigm. With dependencies handled in a robust way and the data interface standardized, wrapping arbitrary tools in Flyte tasks produces reusable and composable components that behave predictably. Tying all of this into a common orchestration layer presents an enormous benefit to the developer experience and consequently the reproducibility and extensibility of the research project as a whole.
