---
title: "BCDNM-HPO: Balancing Coherence, Diversity, and Target Noise in GPU-Accelerated BERTopic"
abstract: |
  Modern topic modeling pipelines based on transformer embeddings, dimensionality reduction, and density-based clustering can discover useful semantic structure in unstructured text. In noisy real-world corpora, however, a large fraction of documents may be assigned to an outlier or noise cluster, reducing the amount of data available for interpretation and downstream analysis. We present Balanced Coherence, Diversity, and Noise-Suppression Metric-guided Hyperparameter Optimization (BCDNM-HPO), a practical workflow for BERTopic-style topic modeling that maximizes a scalar objective combining topic coherence, topic diversity, and a noise-cluster penalty.

  The proposed objective lets practitioners explicitly encode the desired trade-off among interpretable topics, non-overlapping topic vocabularies, and closeness to a target noise percentage. We implement the workflow in a Jupyter notebook using BERTopic, Optuna, and Gensim coherence scoring. Within the BERTopic pipeline, the UMAP dimensionality-reduction and HDBSCAN clustering stages are GPU-accelerated through NVIDIA cuML's zero-code-change `cuml.accel` interface. On the IMDb movie review dataset with a 20% target noise percentage, source experiments report a reduction in noise assignments from 62.39% to 39.19%, reducing target deviation from 42.39 to 19.19 percentage points and increasing clustered document utilization from 37.61% to 60.81%, while coherence decreased from 0.69 to 0.66 and diversity decreased from 0.80 to 0.75. These results suggest that scalarized hyperparameter optimization can recover substantially more usable topic assignments with modest trade-offs in conventional topic-quality metrics.
---

## Introduction

Topic modeling is often used when a text corpus is too large or varied for manual inspection. Classical methods such as Latent Dirichlet Allocation (LDA) model documents as mixtures of latent topics [@lda]. More recent approaches, including BERTopic, treat topic discovery as a clustering problem over dense semantic representations: documents are embedded with transformer models, reduced to a lower-dimensional manifold, clustered, and then summarized with topic keywords [@bertopic].

This modern pipeline is attractive because it is modular and works well with general-purpose text embeddings. It also exposes a practical failure mode. Density-based clustering algorithms such as HDBSCAN can assign low-confidence or low-density documents to an outlier topic, commonly labeled `-1` [@hdbscan]. In small amounts, this behavior is useful: not every document should be forced into a topic. In real corpora, however, the noise cluster can become large enough to undermine the analysis. In the motivating IMDb experiment, a baseline BERTopic configuration assigned 62.39% of documents to noise, leaving 37.61% of the corpus represented by explicit topics.

This paper introduces Balanced Coherence, Diversity, and Noise-Suppression Metric-guided Hyperparameter Optimization (BCDNM-HPO), a workflow for reducing such noise assignments without treating noise reduction as the only objective. The central idea is simple: define a scalar objective that rewards coherence and diversity while penalizing deviation from a target noise percentage, then use hyperparameter optimization (HPO) to search over the UMAP and HDBSCAN configuration space. The method does not propose a new topic model. Instead, it supplies an optimization layer around existing BERTopic components and can be adapted to other embedding, dimensionality-reduction, and clustering choices.

The implementation is provided in a single Jupyter notebook. After activating `cuml.accel` with `%load_ext cuml.accel`, the notebook constructs the BERTopic pipeline, defines the BCDNM objective and component metrics, and runs the Optuna study. Within the pipeline, cuML accelerates the UMAP and HDBSCAN stages while preserving their familiar Python APIs, so the existing BERTopic workflow does not need to be rewritten with GPU-specific estimator classes.

The contributions of this paper are:

- A scalarized objective for topic-model selection that combines coherence, diversity, and target noise-percentage deviation.
- A BERTopic implementation that tunes UMAP and HDBSCAN hyperparameters with Optuna.
- A zero-code-change acceleration path using NVIDIA cuML for iterative topic-model evaluation.
- An empirical IMDb case study showing substantially lower target noise deviation with modest changes to coherence and diversity.

The weights control the relative emphasis on coherence, diversity, and target-noise deviation, while the target specifies the desired noise percentage. This makes the model-selection preferences explicit rather than leaving document coverage as an incidental consequence of the default clustering configuration.

## Background and Related Work

LDA remains a foundational topic-modeling method and is still useful when bag-of-words assumptions and a fixed number of topics are appropriate [@lda]. In noisy corpora, however, LDA requires practitioners to choose the number of topics and tune priors indirectly. It does not expose a native noise-cluster mechanism and does not directly optimize for the fraction of documents that receive interpretable topic assignments.

Neural topic models, including the Neural Variational Document Model (NVDM), represent documents with learned latent variables and can capture richer semantics than classical count-based approaches [@nvdm]. These models add flexibility, but they also introduce more training complexity and can be sensitive to the quality of the latent space. Rather than replacing an existing topic-modeling pipeline with a new neural architecture, our work addresses a more operational problem: how to tune a BERTopic-style workflow so that excessive noise assignments move toward a chosen target without sacrificing topic coherence or diversity.

BERTopic is a modular, embedding-based topic-modeling framework that represents documents with pretrained transformer embeddings, clusters those embeddings, and generates topic representations using class-based TF-IDF (c-TF-IDF) [@bertopic]. A common BERTopic configuration uses UMAP for dimensionality reduction [@umap] and HDBSCAN for density-based clustering [@hdbscan]. UMAP parameters determine the structure of the reduced embedding space, while HDBSCAN parameters determine how that space is divided into clusters and outliers. Because HDBSCAN operates on the representation produced by UMAP, the two parameter sets must be considered together. This dependency motivates joint hyperparameter optimization rather than tuning each stage independently.

Multi-objective optimization is a natural alternative: coherence, diversity, and noise suppression can be treated as separate objectives, and a Pareto front can expose non-dominated trade-offs. That representation is useful when the goal is exploration, but less convenient when a practitioner needs a specific configuration and has a specific operational preference, such as maintaining topic quality while moving the noise rate toward a chosen target.

To address this need, BCDNM combines coherence, diversity, and deviation from a practitioner-defined noise target into a single scalar objective. HPO then searches the coupled UMAP and HDBSCAN parameter space for the configuration that maximizes this objective. In this way, BCDNM-HPO converts the trade-offs represented by a Pareto front into one preference-aware optimization criterion.

We use Optuna to optimize the BCDNM objective, with the Tree-structured Parzen Estimator (TPE) guiding the search [@optuna]. Gensim's topic-coherence implementation computes $c_v$ coherence [@roeder_topic_coherence], while a topic-word uniqueness score measures diversity. The GPU path builds on NVIDIA cuML. Prior work introduced an end-to-end GPU implementation of UMAP and reported speedups of up to 100x while preserving embedding quality [@nolet_gpu_umap]. More recent work extends GPU-accelerated UMAP to massive-scale, out-of-core processing with optional multi-GPU execution [@park_out_of_core_umap]. In our notebook, the UMAP and HDBSCAN stages access GPU acceleration through the zero-code-change `cuml.accel` interface [@cuml_accel].

## BCDNM Objective

### Objective Function

The BCDNM objective combines topic coherence, topic diversity, and target-noise control. Let $C$ denote topic coherence, $D$ topic diversity, and $N$ the percentage of documents assigned to BERTopic's noise cluster, labeled `-1`. Let $\tau$ be the target noise percentage, and let $w_c$, $w_d$, and $w_n$ be nonnegative weights controlling the contribution of each component. The objective is defined as:

```{math}
:label: eq:bcdnm
J_\mathrm{target} = w_c C + w_d D - w_n \left|\frac{N - \tau}{100}\right|
```

This objective rewards configurations whose topics are internally coherent and mutually diverse, while penalizing configurations whose noise-cluster percentage deviates from the target noise percentage. In the reference notebook, $\tau$ is set to 20, so the penalty is proportional to the absolute distance between the observed noise percentage and a 20% target. Throughout this paper, $C$ and $D$ are unitless scores, $N$ and $\tau$ are percentages, and $|N - \tau|$ is reported in percentage points.

### Metric Definitions

In the current notebook implementation, coherence is computed with Gensim's `CoherenceModel` using the $c_v$ score. For each non-noise topic, the top topic words returned by BERTopic are passed to the coherence model along with preprocessed documents and a Gensim dictionary. The noise topic `-1` is excluded.

Diversity is computed as the ratio of unique topic words to total topic words across all non-noise topics:

```{math}
:label: eq:diversity
D = \frac{\left|\cup_{k \in \mathcal{K}} W_k\right|}{\sum_{k \in \mathcal{K}} |W_k|}
```

where $\mathcal{K}$ is the set of non-noise topics and $W_k$ is the top-word set for topic $k$. This lexical-diversity score is one when the top-word sets do not overlap and decreases as different topics reuse the same words.

Noise percentage is computed directly from `topic_model.get_topic_info()`:

```{math}
:label: eq:noise
N = \frac{100}{n}\sum_{i=1}^{n}\mathbf{1}(z_i = -1)
```

where $z_i$ is the topic label assigned to document $i$, $n$ is the number of documents, and $\mathbf{1}(z_i=-1)$ equals one when the document is assigned to the noise topic and zero otherwise.

The target-deviation term used by the objective is:

```{math}
:label: eq:noise_deviation
\Delta_N = |N - \tau|
```

The objective normalizes this deviation by dividing by 100. For example, with $\tau = 20$, the baseline noise percentage $N = 62.39$ corresponds to $\Delta_N = 42.39$ percentage points and a normalized penalty input of 0.4239.

Together, these metrics convert the qualitative goal of finding more usable, distinct, and interpretable topics into an optimization signal.

:::{figure} figures/bcdnm_pipeline_clean.png
:label: fig:pipeline
BCDNM-HPO wraps a BERTopic-style pipeline with metric evaluation and hyperparameter optimization. Each trial builds a topic model, evaluates coherence, diversity, noise percentage, and target noise deviation, and returns a scalar objective value to the optimizer.
:::

## Implementation

The implementation follows the structure of the public notebook `video_notebook_for_Minimizing_Noise_Cluster_for_Topic_Modeling.ipynb` [@ding_notebook]. It begins with zero-code-change GPU acceleration:

```python
%load_ext cuml.accel
```

The baseline topic model then uses BERTopic with UMAP and HDBSCAN:

```python
from bertopic import BERTopic
import hdbscan
import umap
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
docs = dataset["text"]

umap_model = umap.UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.0,
    random_state=42,
)
hdbscan_model = hdbscan.HDBSCAN(
    min_samples=10,
    gen_min_span_tree=True,
    prediction_data=True,
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
)
topics, probs = topic_model.fit_transform(docs)
```

The HPO search space tunes six parameters:

```{list-table} Tuned hyperparameters in the reference implementation.
:label: tbl:hyperparameters
:header-rows: 1
* - Component
  - Hyperparameter
  - Search range or choices
* - UMAP
  - `n_components`
  - integer, 5 to 20
* - UMAP
  - `n_neighbors`
  - integer, 5 to 20
* - UMAP
  - `min_dist`
  - float, 0.0 to 1.0
* - HDBSCAN
  - `min_samples`
  - integer, 5 to 25
* - HDBSCAN
  - `gen_min_span_tree`
  - `True` or `False`
* - HDBSCAN
  - `prediction_data`
  - `True` or `False`
```

Optuna evaluates candidate parameter sets by constructing the UMAP and HDBSCAN models, fitting BERTopic, computing $C$, $D$, and $N$, and returning the scalar score. The source notebook uses `TPESampler(seed=142)` and a maximization study for the weighted target-deviation score.

The loop is intentionally modular. The embedding model, dimensionality-reduction algorithm, clustering algorithm, optimizer, and weights can all be changed. For example, practitioners could replace UMAP with PCA for a faster baseline or replace HDBSCAN with another clustering algorithm. The objective weights can also be adjusted to reflect the relative importance of coherence, diversity, and target-noise deviation.

## Experimental Setup

The source experiment uses the IMDb movie review corpus, loaded through the Hugging Face `datasets` package as the `train` split of 25,000 reviews. IMDb is useful for this case study because long-form user reviews contain mixed sentiment, plot details, actor names, genre terms, HTML artifacts, and highly variable writing styles [@maas_imdb]. These characteristics produce a realistic setting in which many documents may sit between dense topic regions.

The baseline is the default notebook BERTopic pipeline with UMAP parameters `n_components=5`, `n_neighbors=15`, and `min_dist=0.0`, and HDBSCAN parameter `min_samples=10`. The optimized model uses the best parameters returned by Optuna under the BCDNM objective.

The reported evaluation focuses on four quantities:

- $c_v$ coherence, where higher values indicate more semantically interpretable topic-word groups.
- Topic diversity, where higher values indicate less overlap among topic-word lists.
- Noise percentage $N$, the percentage of documents assigned to topic `-1`.
- Target noise deviation $\Delta_N = |N - \tau|$, where lower values indicate closer agreement with the chosen target noise percentage.

## Results

The source experiment uses $\tau = 20$. The baseline assigns 62.39% of documents to the noise topic, corresponding to a target deviation of 42.39 percentage points. The optimized BCDNM-HPO configuration assigns 39.19% to noise, corresponding to a target deviation of 19.19 percentage points. This is a 23.20 percentage-point absolute reduction in both observed noise percentage and target deviation, increasing the share of clustered documents from 37.61% to 60.81%.

```{list-table} Source-reported IMDb metrics for baseline BERTopic and BCDNM-HPO.
:label: tbl:results
:header-rows: 1
* - Metric
  - Baseline BERTopic
  - BCDNM-HPO
  - Direction
* - Coherence
  - 0.69
  - 0.66
  - Higher is better
* - Diversity
  - 0.80
  - 0.75
  - Higher is better
* - Noise percentage $N$
  - 62.39%
  - 39.19%
  - Closer to the 20% target is better
* - Target noise deviation
  - 42.39 pp
  - 19.19 pp
  - Lower is better
* - Clustered document utilization
  - 37.61%
  - 60.81%
  - Derived as $100\% - N$
```

:::{figure} figures/metric_comparison.png
:label: fig:metrics
Metric comparison for the IMDb source experiment. BCDNM-HPO reduces target noise deviation substantially while retaining similar coherence and diversity.
:::

The coherence and diversity scores both decrease slightly. This is expected: moving the noise percentage closer to a 20% target can introduce harder boundary cases into explicit topics, and increasing clustered document utilization can trade off against tightly separated topic vocabularies. The BCDNM objective makes that trade-off visible and controllable. In this use case, the decrease from 0.69 to 0.66 in coherence and from 0.80 to 0.75 in diversity is modest relative to the gain in clustered documents.

The intertopic distance maps provide a qualitative view of the change. The optimized result contains more visible topic bubbles and a broader topic index range, consistent with more documents being represented by explicit topics. The maps should be interpreted as exploratory diagnostics rather than definitive proof of topic quality.

:::{figure} figures/baseline_intertopic.png
:label: fig:baseline_intertopic
Baseline BERTopic intertopic distance map from the source materials.
:::

:::{figure} figures/bcdnm_intertopic.png
:label: fig:bcdnm_intertopic
BCDNM-HPO intertopic distance map from the source materials. The optimized model assigns more documents to explicit topics and reports a noise percentage closer to the 20% target.
:::

## Discussion

BCDNM-HPO is an optimization layer built on top of BERTopic, not a replacement topic model. The underlying embedding, UMAP, HDBSCAN, and topic-representation pipeline remains unchanged. The method adds an inspectable objective that treats document coverage as an optimization criterion alongside coherence and diversity.

The IMDb experiment illustrates why this matters. The default configuration assigns 62.39% of documents to the noise cluster, whereas BCDNM-HPO reduces that percentage to 39.19%. This broader coverage is accompanied by modest decreases in coherence and diversity, reflecting a real trade-off: assigning more ambiguous documents to topics can reduce topic distinctness. The aggregate metrics indicate that topic quality remained reasonably stable, but they do not prove that every reassigned document received a meaningful topic. Human inspection is still necessary.

Whether this trade-off is desirable depends on the application. Exploratory analysis may favor broader corpus coverage, while applications requiring conservative, high-confidence assignments may permit a higher noise target or place greater weight on coherence and diversity. The target and weights are therefore design choices rather than universal constants.

More broadly, BCDNM-HPO makes noise assignment visible as part of model selection. A high noise percentage indicates that many documents fall outside the clusters identified under the current configuration. Instead of accepting that outcome as an incidental consequence of HDBSCAN defaults, practitioners can measure it, specify a target, and search for a configuration that better reflects their analysis goals.

## Conclusion

BCDNM-HPO provides a lightweight way to tune BERTopic pipelines for improved document utilization. In the IMDb experiment, it reduced the noise percentage from 62.39% to 39.19% and the target deviation from 42.39 to 19.19 percentage points, with modest changes in coherence and diversity. Its main contribution is to make the trade-off among topic quality, coverage, and target noise explicit and searchable. The method does not replace human topic review, but it provides a practical mechanism for selecting BERTopic configurations that leave less of the corpus unexamined.
