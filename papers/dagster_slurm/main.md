---
title: 'dagster-slurm: Dagster Assets on Slurm-Managed Scientific Workflows'
abstract: |
  Scientific pipelines increasingly span laptops, cloud services, institutional servers, and batch-scheduled high-performance computing clusters.
  Modern data orchestrators such as Dagster provide lineage, monitoring, and asset-centric scheduling for heterogeneous dataflows, but they do not natively submit work to Slurm, the scheduler used by many academic supercomputing centers.
  Researchers therefore often maintain a second `sbatch`-based toolchain for the HPC stages of otherwise orchestrated pipelines, fragmenting observability at the most resource-intensive steps.

  We present dagster-slurm, an open-source package that extends Dagster's control plane to Slurm-managed clusters.
  dagster-slurm lets the same asset source code run on a laptop, in continuous integration, against a containerized Slurm cluster, and on a production HPC cluster reached over SSH, while target-specific queues, credentials, filesystem paths, and site defaults remain in deployment configuration.
  A `ComputeResource` abstraction packages environments, submits jobs through Slurm, streams logs through Dagster Pipes, and records scheduler metadata such as job state, CPU efficiency, memory use, and node-hours.

  We evaluate the package through a production pipeline that discovers and characterizes family-owned firms across European countries from Common Crawl and a commercial company registry.
  The pipeline spans four compute tiers, from a DuckDB index scan over roughly 170 million URLs per crawl to fine-tuned language-model extraction on A100 GPUs.
  On the TU Wien DataLAB cluster, the pipeline has run more than 9,600 Slurm jobs while preserving the same Dagster asset graph used on a developer laptop and in CI.
  We also report portability checks using site configurations for the VSC-5 and Leonardo supercomputers.
---

## Introduction

Scientific pipelines increasingly cross hardware and administrative boundaries.
A representative workload filters a large dataset on commodity CPU nodes, runs an expensive model on scarce accelerators, and aggregates results on a different machine again.
The cost-effective implementation places each stage on the hardware that suits it, but that placement often moves the pipeline between a data team's normal orchestration layer and a batch-scheduled high-performance computing (HPC) system.
HPC has become an important part of research software engineering practice [@hettrick_2022_7015772], yet reproducibility in these environments remains difficult [@ANTUNES2024100655; @Courtes9882991; @keahey_2025_15306610].

Dagster models a pipeline as a graph of persistent assets, datasets, models, and tables rather than only as a graph of tasks [@dagster].
This model gives users a control plane for lineage, monitoring, schedules, asset checks, and metadata across heterogeneous infrastructure.
Dagster does not, however, natively support Slurm [@yoo2003slurm], the resource manager behind a large share of academic HPC centers.
Teams that use Dagster for cloud-facing or database-facing stages often bridge this gap with custom submission scripts or a second workflow system for HPC stages.
That split can fragment lineage and observability exactly where the pipeline consumes the most specialized resources.

We present **dagster-slurm**, an open-source package that extends Dagster's control plane to Slurm-managed hardware.
The package is designed around a precise portability boundary.
The same asset source code can run directly on a laptop, in continuous integration (CI), against a containerized Slurm cluster, and on a production HPC cluster reached over SSH.
Deployment-specific queues, credentials, filesystem paths, QoS, reservations, and site defaults remain in configuration; per-asset resource requests may remain in the asset source.
A `ComputeResource` abstraction hides SSH transport, environment packaging through Pixi and pixi-pack [@pixi], and queue configuration, while Dagster Pipes streams logs and scheduler metadata back into the Dagster UI [@dagsterpipes].

dagster-slurm occupies a complementary role to established Python HPC workflow tools.
Parsl [@parsl], executorlib [@executorlib], jobflow [@jobflow], and PSI/J [@psij] provide mature APIs for task execution and scheduler submission.
The contribution here is different: Slurm becomes one execution target for a Dagster asset graph that also touches object stores, databases, local development, and CI.
Teams that already use task workflow managers for compute-heavy stages can still wrap those invocations inside Dagster assets to recover lineage over the full dataflow.

We evaluate the integration with a production pipeline that discovers family-owned firms across European countries from web-scale text and matches them to a commercial company registry.
Throughout the paper, **tier** means a class of hardware matched to a pipeline stage, such as CPU filtering, CPU-plus-network fetching, GPU training, or GPU inference.
An **execution target** is a place the same asset source can run, such as a laptop, CI, containerized Slurm, or a production cluster.
The case-study pipeline exercises four compute tiers and scans roughly 170 million URLs per crawl across dozens of crawls.
We use it to make the system design concrete; the domain results are descriptive artifacts of the case study rather than validated empirical claims.

## System design

The integration is organized in three layers, shown below, and mapped onto execution targets in @fig:arch.
@fig:graph combines the asset graph, compute tiers, scheduler metadata, execution targets, and measured Slurm totals for the case-study deployment.

:::{figure} software_layers.png
The three software layers.
User asset code calls `ComputeResource`, which selects a local or Slurm Pipes client.
The client drives a launcher (Bash or Ray) and, for Slurm targets, an SSH pool that submits and monitors jobs.
:::

**Resource definitions.** `ComputeResource`, `SlurmResource`, and `SSHConnectionResource` are Dagster `ConfigurableResource` objects.
They encapsulate queue defaults, SSH authentication, and the execution mode.
The asset author interacts only with `ComputeResource`; the rest is configuration that lives outside the asset body.

**Launchers and Pipes clients.** A launcher translates a payload into an execution plan.
The Bash launcher runs a script; the Ray launcher [@moritz2018ray] starts and terminates a multi-node Ray cluster around the script.
The Slurm Pipes client handles environment packaging, transfers the payload, submits the job with `sbatch`, and streams structured messages, metadata, and logs back through Dagster Pipes [@dagsterpipes].
New launchers extend a `ComputeLauncher` base class.

**Operational helpers.** Environment deployment, metrics collection, and SSH connection pooling with ControlMaster address HPC constraints such as login-node sandboxes and intermittent connections.

:::{figure} arch-overview.png
:label: fig:arch
The same scalable job follows multiple paths from one asset definition: direct local execution for development, automated testing through CI, or production deployment to a Slurm cluster over an SSH-accessible edge node.
The Dagster control plane, `ComputeResource`, Pipes client, launcher, and the return path for logs and scheduler metadata are shared across all paths.
:::

:::{figure} asset_graph_tiers.png
:label: fig:graph
Asset graph, compute tiers, execution targets, and Slurm observability for the case-study pipeline.
The figure links representative Dagster assets to their CPU, CPU-plus-network, and GPU tiers; shows the scheduler metadata streamed back into Dagster through Pipes; lists the local, CI, and DataLAB execution targets; and summarizes the measured Slurm totals reported in @tbl:ops.
:::

dagster-slurm builds on Dagster's `ConfigurableResource` and Pipes protocols rather than on I/O managers, and it deliberately leaves data management to the user.
I/O strategies vary widely across HPC sites, so the package transfers payload scripts and environments but not datasets.
The recommended pattern is a deployment-mode-aware path that resolves to a local directory during development and to a shared parallel filesystem or object store in production, so that asset code stays unchanged across targets.
This matters because data-intensive HPC workloads such as ours operate on data large enough that automatic serialization across network boundaries would be impractical.

### Environment packaging and reproducibility boundaries

Remote execution requires the cluster to have the same software the developer used.
dagster-slurm packages environments with Pixi and pixi-pack [@pixi]: a Pixi lockfile pins every dependency, and pixi-pack produces a relocatable bundle that is uploaded and unpacked on the target, or, on sites where a shared environment is pre-deployed, reused so that only the per-run payload script is transferred.
This addresses repeatability of the execution environment, but full scientific reproducibility of a production run additionally depends on inputs and state the package does not control: the specific Common Crawl snapshots, the licensed registry data, the model checkpoints, site-specific modules and CUDA drivers, and credentials.
We therefore frame the reproducibility contribution as repeatable, version-pinned execution paths exercised in CI, not as turnkey reproduction of the case-study science.

### Failure and restart semantics

Slurm jobs fail in characteristic ways, and the integration is explicit about each.
A submission error or SSH timeout surfaces as a failed Dagster step with the captured stderr.
A job killed by walltime or preempted is reported through the scheduler state and marked failed, so Dagster's retry policy can resubmit.
Clean resumption after a partial run depends on asset-level idempotency: the case-study payloads write outputs keyed by input URL and, on restart, read the URLs already written and skip them, so a resubmitted partition continues rather than recomputing.
Assets without such idempotent output conventions recompute from the start on retry.
Recovering scheduler accounting metadata (`node-hours`, CPU efficiency) requires Slurm accounting (`sacct`) to be available on the site; where it is not, the job still runs and logs stream, but those metadata fields are omitted.

## The same code across targets

The integration is designed so that one asset definition runs in every target.
An asset calls `compute.run`, passes a payload path and its resource requirements, and yields the results.
The example below is the GPU extraction stage of the firm-discovery case study, abridged: imports and the partition and resource definitions are omitted, and `_payload`, `model_path`, and `country_crawl_partitions` are defined elsewhere in the project.
The asset states hardware requirements in deployment-neutral terms; each target maps those requirements to local execution, CI resources, or site-specific Slurm options.

```python
@dg.asset(partitions_def=country_crawl_partitions)
def cc_family_nuextract_results(
    context: dg.AssetExecutionContext,
    compute: ComputeResource,
):
    completed = compute.run(
        context=context,
        payload_path=_payload("extract_nuextract.py"),
        extra_env={
            "NUEXTRACT_MODEL_PATH": model_path,
            # vLLM gpu_memory_utilization
            "NUEXTRACT_GPU_MEM_UTIL": "0.85",
        },
        resource_requirements={
            "gpus": 1,
            "cpus": 8,
            "memory_gb": 64,
            "walltime": "23:00:00",
        },
    )
    yield from completed.get_results()
```

Nothing in this asset names a host, a queue system, or a transport.
Those are supplied by the `ComputeResource` that Dagster injects, and that resource is selected by deployment.
A small factory reads an environment variable and returns the resource set for the current target (again abridged; `Environment`, `ExecutionMode`, the launchers, and the SSH and Slurm resources are imported from the package):

```python
def get_resources(deployment: Environment):
    if deployment == Environment.DEVELOPMENT:
        return {
            "compute": ComputeResource(
                mode=ExecutionMode.LOCAL,
                default_launcher=BashLauncher(),
            ),
        }
    # staging_docker_slurm, or a named supercomputer site
    ssh = SSHConnectionResource.from_env(prefix="SLURM_EDGE_NODE")
    slurm = SlurmResource(ssh=ssh, queue=site_queue_defaults(deployment))
    return {
        "compute": ComputeResource(
            mode=ExecutionMode.SLURM,
            slurm=slurm,
            default_launcher=BashLauncher(),
        ),
    }
```

Setting `DAGSTER_DEPLOYMENT=development` runs the extraction on the local machine with no queue and no SSH, which is how a developer iterates.
Setting it to a containerized Slurm target runs the identical asset against a Dockerized cluster, which is what CI exercises [@docker].
Setting it to a named Slurm site, the TU Wien DataLAB in our production runs, with VSC-5 and Leonardo as additional configured sites, maps the asset's generic GPU, CPU, memory, and walltime requirements onto that cluster's partition, QoS, reservation, and account defaults, and reuses a pre-deployed environment.
Host names, transport, credentials, partition names, and site policy stay in deployment configuration, so the same asset source remains portable across targets.

For distributed workloads the Ray launcher [@moritz2018ray] replaces the Bash launcher in the same `ComputeResource`.
It allocates a multi-node Slurm job, starts a Ray head and workers across the allocation, runs the payload against that cluster, and terminates it.
The Bash and Ray launchers are the two production-stable launchers; a Spark launcher, session-based allocation reuse, and heterogeneous jobs are experimental and not covered by the operational experience reported here.

## Case study: web-scale firm discovery

We operate dagster-slurm on a pipeline that builds a structured corpus of European firms from public web text.
The scientific question is which firms present themselves as family-owned.
The engineering question, and the one this paper evaluates, is how to run a pipeline whose stages have very different hardware needs without maintaining two toolchains.

### What the pipeline computes

The pipeline runs once for each region and Common Crawl snapshot [@commoncrawl].
The diagram below shows the stages.

:::{figure} pipeline.png
Stages of one (region, crawl) run.
The first three stages are CPU and network work; only the extraction stage (NX) uses a GPU.
Consolidation and registry matching return to CPU.
:::

The index scan filters a crawl, roughly 170 million URLs, down to a few million candidate pages using DuckDB [@raasveldt2019duckdb], keeping pages whose registered domain or detected content language ties them to a target region and whose page type is one of landing, imprint, about, or team.
Matching records are fetched from `s3://commoncrawl` with authenticated byte-range requests, cleaned with resiliparse [@resiliparse], and deduplicated to one page per domain and page type.
The extraction stage serves a fine-tuned NuExtract-2.0-8B model with vLLM [@nuextract; @kwon2023vllm] and returns a 30-field JSON record per page covering the company name, registration and VAT identifiers, legal form, address, described products, and an explicit family-ownership flag with a supporting sentence.
Records are consolidated to one row per firm and matched to a commercial company registry by identifier, domain, or name.

A single multilingual base model serves all target regions, reducing the need to train separate country-specific extractors.
Rather than train a separate extractor per country, we adapt the one 8.33-billion-parameter model per language group with a LoRA adapter [@hu2021lora] that trains 40.4 million parameters, 0.48% of the weights.
The adapters are trained in a chain, each starting from the previous one, with a fraction of earlier-region data replayed; on held-out test domains the adapter is intended to retain coverage of earlier language groups while adding the new one, and our internal evaluation reports eval loss between 0.047 and 0.055 down the chain.
Adapter training for the regions we have run took from 15 hours to several days on one A100 (80 GB), depending on the size of the new language set.

### Why it needs multiple tiers

The stages have different hardware requirements.
@tbl:tiers gives the measured resource profile of each stage.
The index scan and the fetch-and-clean stages are CPU and network bound and would waste a GPU allocation.
Fine-tuning needs an A100 with 80 GB of GPU memory for activations and optimizer state at sequence length 4096.
Inference fits on a 40 GB card.
Consolidation and registry matching are again CPU work.
A homogeneous allocation would either underprovision the GPU stages or waste accelerators on CPU-bound scans.

@tbl:tiers reports production resource profiles.
"Allocation" is the Slurm request; "Runtime / throughput" reports the configuration and observed rate.
RAM values are host memory; GPU memory is stated separately.
Figures are per job unless noted.

```{list-table} Measured resource profile by stage.
:label: tbl:tiers
:header-rows: 1
* - Stage
  - Hardware
  - Allocation
  - Runtime / throughput
* - classify / filter
  - CPU
  - 8-32 cores, 32-128 GB RAM
  - DuckDB; input scale ~170 M URLs/crawl
* - fetch + clean
  - CPU + network
  - 8-96 worker threads
  - ~1 ms/page clean (per thread); 50-95 MB/s S3 (per job)
* - fine-tune
  - 1 GPU
  - 16 cores, 128 GB RAM, A100 80 GB
  - seq len 4096, bf16; 26-37 s/step
* - LLM inference
  - 1 GPU
  - 8 cores, 64 GB RAM, A100 40-80 GB
  - vLLM, gpu mem util 0.85; 15,000-22,000 pages/hour per A100
```

The pipeline is partitioned so that one (region, crawl) pair is one Slurm job.
The case-study results shown below cover fifteen regional groupings (DACH, plus the Nordic, Baltic, and Slavic countries); the pipeline's full partition space spans two axes, the registered country-code top-level domain (23 values) and the detected content language (19 values), each crossed with the configured set of 69 crawl snapshots, giving partition spaces of 1,587 and 1,311 (region, crawl) jobs respectively that are then merged and aggregated into the regional groupings.
Materializing a partition is independent and resumable: a relaunched partition reads the URLs already written and skips them, and a large region shards across GPUs by a hash of the URL so each worker takes a disjoint slice.

The filtering tier is a cost control, not only a convenience.
For the Romance languages, language does not equal country: Spanish and French reach across Latin America and Africa, so the index lists 1.25 billion Spanish and 1.11 billion French URLs, most of them out of scope.
These collapse to 6.9 million distinct domains.
Even after deduplicating to one page per domain and page type, running extraction over the resulting candidate pages without first applying the registry country filter would exceed 1,100 A100-hours, with most of it spent on out-of-scope domains.
The filter that reduces this cost is the registry-based country match applied at the index, on CPU, before any page is fetched or any GPU time is spent.
Placing that filter on the right tier is what keeps the run within a realistic GPU allocation.

### Descriptive results

The numbers below are descriptive artifacts of the case study.
They characterize firms' web self-description as recovered by this pipeline, and they should not be read as population prevalence of family ownership: they reflect registry coverage, page availability, extraction quality, and matching rules, and a "firm" here is a web domain, not a legal entity.

The corpus covers 5,312,872 firms, of which 2,690,433 (51%) match at least one registry company.
The family-ownership flag is resolved for 3.3 million firms, of which 120,446 (3.6%) describe themselves as family-owned.
@fig:family shows how this self-description rate varies across the fifteen regional groupings.

:::{figure} family_by_region.png
:label: fig:family
Share of firms describing themselves as family-owned, by regional grouping.
Unit of observation: one web domain with a resolved family-ownership flag.
Numerator: domains whose extracted record carries the family-owned flag; denominator: domains in the region with a resolved (true or false) flag.
Source: the case-study corpus over the fifteen regional groupings.
Cross-region differences co-occur with national disclosure conventions and with the density of the relevant page types, which this descriptive plot does not disentangle.
:::

As an internal consistency check we measure eponymy, whether a firm's name contains a disclosed manager's or owner's surname, on the subset of firms that both name a person and have a company name.
Among that subset, 32% are eponymous overall; the rate is 52% for self-described family firms against 32% for the rest (@fig:eponymy).
This is a descriptive consistency check rather than ground truth: it can be confounded by region, sector, firm size, language, and extraction error, and we do not interpret it as validation of the family-ownership measure.

:::{figure} eponymy.png
:label: fig:eponymy
Eponymy rate, split by family-ownership self-description.
Sample restriction: firms that disclose at least one person and have a non-empty company name.
Numerator: firms whose company name contains a disclosed surname; denominator: firms in each group within that restricted sample.
The family vs non-family gap (52% vs 32%) is shown as a within-sample comparison, not a population estimate.
:::

These results are a by-product here.
The contribution this paper evaluates is the orchestration that produced them with one asset graph rather than two.

## Evaluation

We assess the integration as a piece of research software along three dimensions: that it is tested, that it has been operated at production scale, and that it surfaces useful observability.
@tbl:ops summarizes the operational profile.
We report scheduler-derived values from a fixed accounting window rather than estimating lifetime totals.

@tbl:ops reports the operational profile of the case-study deployment on the TU Wien DataLAB cluster.
Compute totals are from Slurm's accounting database (`sreport`); job counts and outcomes are from `sacct`, with the two sources agreeing to within 1% on core-hours.
These figures are taken directly from Slurm rather than from Dagster's run log, and so are independent of any drift in the orchestrator's own records.
They cover one production cluster and one accounting window, not the project's full lifetime.

```{list-table} Operational profile of the case-study deployment.
:label: tbl:ops
:header-rows: 1
* - Property
  - Value
* - Production cluster
  - TU Wien DataLAB, GPU-only partitions (A100 80 GB, A100s 40 GB, L40s, A40). The package also ships VSC-5 and Leonardo site configs used for development-time portability checks; no production workloads ran there
* - Accounting window (measured)
  - 2026-04-07 to 2026-06-19 (~10 weeks)
* - Execution modes shipped
  - local; one Slurm job per asset partition
* - Stable launchers
  - Bash, Ray
* - Slurm-backed partition space
  - 1,587 (country, crawl) + 1,311 (language, crawl)
* - Slurm jobs submitted (measured)
  - 9,626 (9,447 Dagster-named + 179 named worker jobs)
* - Job outcomes (measured)
  - 7,654 completed (79.5%); 954 failed for other reasons (9.9%); 64 out-of-memory; 57 timeout; 897 cancelled (9.3%)
* - Compute consumed (measured, `sreport`)
  - 118,914 CPU core-hours; 5,746 node-hours; 2,509 GPU-hours
* - Per-stage throughput
  - see @tbl:tiers (15,000-22,000 pages/hour per A100)
* - CI coverage
  - local and Dockerized Slurm execution modes (GitHub Actions); build, unit, and Slurm-on-Docker integration job runs in ~16 min wall time
* - Package
  - dagster-slurm, Apache-2.0, github.com/ascii-supply-networks/dagster-slurm
```

**Reproducibility (testing).** The integration tests run in GitHub Actions against a containerized Slurm cluster.
CI provisions the environment with Pixi, deploys it once, starts a Slurm cluster in Docker, and runs Dagster assets [@docker] through the local and Slurm execution modes, exercising the same code path users run in production.
The CI fixture materializes the same test assets in local and Dockerized Slurm modes and compares their emitted Dagster metadata and output files.
The full build, unit, and Slurm-on-Docker integration job completes in about 16 minutes of wall time per commit, which keeps the cross-environment check cheap enough to run on every change.
This improves the repeatability of the selected execution paths, in line with current recommendations for CI-based HPC reproducibility [@hayotsasson2025address]; as noted above, it does not by itself reproduce a production scientific run.

**Operational experience.** The system has run the case-study pipeline in production on the TU Wien DataLAB cluster.
Over the ten-week accounting window in @tbl:ops it submitted 9,626 Slurm jobs, of which 79.5% completed and 9.9% failed for reasons other than memory or time limits (with 64 out-of-memory and 57 timeout terminations counted separately; a further 9.3% were cancelled, which on this cluster includes deliberate reruns for stack consistency and retry supersessions rather than only errors), and it consumed 2,509 GPU-hours and 118,914 CPU core-hours.
These totals are read directly from Slurm's accounting database (`sreport` and `sacct`), independent of Dagster's own run records; we use the scheduler as the source of truth precisely because the orchestrator's event log can drift when a connection is lost mid-job.
SSH ControlMaster fallbacks, password-based jump hosts, login-node hygiene, and queue, QoS, and reservation overrides are documented per site, with verification snippets using `squeue` and `scontrol`.
During development we also ran the integration against the VSC-5 and Leonardo supercomputers to validate portability; no production workloads were submitted to those sites, and their development-time accounting records are no longer available to us, so every measured figure in this paper is from DataLAB.
We report this as production operational experience rather than as a scheduler-scalability benchmark, which would require a controlled study of submission throughput and concurrent-job limits that is outside this paper's scope.

**Observability.** Slurm job identifiers, CPU efficiency, memory use, and node-hours surface as Dagster metadata where Slurm accounting is available, and Ray clusters stream their output back through Pipes.
Conventional Dagster asset checks and alerting therefore operate over those metadata and failure signals without modification, a property that separate script-based workflows often lose or must rebuild manually.

### Positioning against existing tools

The practical distinction is the object being moved across the HPC boundary.
Direct `sbatch` submits shell jobs and leaves lineage, retries, and metadata to user conventions.
Parsl, PSI/J, executorlib, Snakemake, and Nextflow provide mature scheduler-facing abstractions for functions, jobs, rules, or processes.
Flyte and Airflow provide broader workflow control planes, usually with a container-first deployment model.
dagster-slurm targets a narrower case: teams that already model their work as Dagster assets and want Slurm to be one execution target for those assets rather than a separate workflow boundary.

## Related work

The Python HPC workflow ecosystem is rich.
Parsl provides parallel scripting with multi-site execution and SSH submission [@parsl]; executorlib extends the standard `concurrent.futures` interface to schedulers with per-function resource control [@executorlib]; jobflow defines workflows with decorators [@jobflow]; and PSI/J offers a portable submission interface across schedulers [@psij].
In the bioinformatics tradition, Snakemake [@koster2012snakemake] and Nextflow [@ditommaso2017nextflow] define file-driven rule and process graphs that submit to Slurm through executors, and Dask [@rocklin2015dask] parallelizes array and dataframe computation, including onto Slurm via dask-jobqueue [@daskjobqueue].
General-purpose orchestrators such as Apache Airflow [@airflow] and Flyte schedule task graphs but target containerized cloud backends first.
The Common Workflow Language community catalogs many more systems [@cwl_wiki].
These tools orchestrate compute tasks, typically file- or task-centric and HPC- or cloud-centric.
dagster-slurm differs in level and in unit of abstraction: it brings Slurm execution under a data orchestrator whose unit is a persistent asset and whose control plane already manages the cloud and database stages of a pipeline, so the supercomputer is one target in a larger asset graph rather than the entire workflow environment.

Within the SciPy community, recent proceedings show the adjacent niches.
Feickert et al. [@feickert2025pixi] use Pixi to build reproducible environments for scientific machine learning across heterogeneous platforms, the same packaging foundation we rely on for remote deployment.
Turner [@turner2024flyte] orchestrates heterogeneous bioinformatics tools with Flyte, and Staneva et al. [@staneva2024echodataflow] build recipe-based, reproducible fisheries-acoustics pipelines; both, like this work, motivate an orchestration layer through a concrete scientific workload.
Bednar and Durant [@bednar2023pandata] assemble a scalable, domain-independent analysis stack.
dagster-slurm complements these by connecting the orchestration layer to Slurm-managed public HPC systems.

## Conclusion and future work

dagster-slurm lets a single Dagster asset graph span a laptop, CI, and an HPC cluster without rewriting the asset source code for the HPC stages, while deployment-specific queues, credentials, paths, and site defaults remain external to that code.
By keeping the dataflow visible end to end and surfacing scheduler metadata in the orchestrator, it reduces duplicated submission scripting and brings more of the HPC execution path under Dagster's observability model, while supporting the testing, documentation, and observability practices expected of high-quality research software [@eisty2025].
The firm-discovery case study demonstrates the practical value: a four-tier workload, with a CPU index scan over hundreds of millions of URLs and GPU language-model inference, run from the same code on a laptop and on a GPU-partitioned production HPC cluster.

The package ships two production execution modes, local and one Slurm job per asset partition, and two stable launchers, Bash and Ray.
Work in progress includes a Spark launcher, session-based allocation reuse and heterogeneous jobs for finer-grained scheduling inside an allocation, and non-interactive integration with strict multi-factor environments.
The integration delegates credential handling to SSH and site-approved authentication; it does not bypass multi-factor authentication or store one-time passwords.
Contributions, including cluster-specific recipes and new launchers, are welcome at <https://github.com/ascii-supply-networks/dagster-slurm>.

## Disclosure of generative AI use

Portions of this work were assisted using a generative AI tool (Claude).
The tool was used for drafting and refining prose and for summarizing the authors' existing code and experiment logs.
All outputs were reviewed, verified, and revised by the authors, who take full responsibility for the accuracy and integrity of the final content.

## Acknowledgements

We thank the operations team at the TU Wien DataLAB, where the case-study pipeline runs in production, and the operations teams at Austrian Scientific Computing (VSC-5) and CINECA's Leonardo supercomputer for early feedback.
We also thank the Dagster community for discussions on orchestrating HPC workloads.
Funding and in-kind support were provided by the Complexity Science Hub Vienna and the Austrian Supply Chain Intelligence Institute.
This work was completed in part at the EUROCC AI Hackathon 2025, part of the Open Hackathons program.
We acknowledge OpenACC-Standard.org for their support.
