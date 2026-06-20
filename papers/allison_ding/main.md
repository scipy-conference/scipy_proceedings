---
title: "BCDNM-HPO: Noise-Suppressed Topic Modeling with Zero-Code-Change GPU Acceleration"
abstract: |
  Modern topic modeling pipelines based on transformer embeddings, dimensionality reduction, and density-based clustering can discover useful semantic structure in unstructured text. In noisy real-world corpora, however, a large fraction of documents may be assigned to an outlier or noise cluster, reducing the amount of data available for interpretation and downstream analysis. We present BCDNM-HPO, a practical hyperparameter optimization workflow for BERTopic-style topic modeling that maximizes a scalar objective combining topic coherence, topic diversity, and a noise-cluster penalty.

  The proposed objective lets practitioners explicitly encode the desired trade-off among interpretable topics, non-overlapping topic vocabularies, and closeness to a target noise percentage. We implement the workflow in a Jupyter notebook using BERTopic, UMAP, HDBSCAN, Gensim coherence scoring, Optuna, and RAPIDS cuML zero-code-change acceleration. On the IMDb movie review dataset with a 20% target noise percentage, source experiments report a reduction in noise assignments from 62.39% to 39.19%, reducing target deviation from 42.39 to 19.19 percentage points and increasing clustered document utilization from 37.61% to 60.81%, while coherence decreased from 0.69 to 0.66 and diversity decreased from 0.80 to 0.75. These results suggest that scalarized hyperparameter optimization can recover substantially more usable topic assignments with modest trade-offs in conventional topic-quality metrics.
---

## Introduction

Topic modeling is often used when a text corpus is too large or varied for manual inspection. Classical methods such as Latent Dirichlet Allocation (LDA) model documents as mixtures of latent topics [@lda]. More recent approaches, including BERTopic, treat topic discovery as a clustering problem over dense semantic representations: documents are embedded with transformer models, reduced to a lower-dimensional manifold, clustered, and then summarized with topic keywords [@bertopic].

This modern pipeline is attractive because it is modular and works well with general-purpose text embeddings. It also exposes a practical failure mode. Density-based clustering algorithms such as HDBSCAN can assign low-confidence or low-density documents to an outlier topic, commonly labeled `-1` [@hdbscan]. In small amounts, this behavior is useful: not every document should be forced into a topic. In real corpora, however, the noise cluster can become large enough to undermine the analysis. In the motivating IMDb experiment, a baseline BERTopic configuration assigned 62.39% of documents to noise, leaving 37.61% of the corpus represented by explicit topics.

This paper introduces BCDNM-HPO, a workflow for reducing such noise assignments without treating noise reduction as the only objective. The central idea is simple: define a scalar objective that rewards coherence and diversity while penalizing deviation from a target noise percentage, then use hyperparameter optimization (HPO) to search over the UMAP and HDBSCAN configuration space. The method does not propose a new topic model. Instead, it supplies an optimization layer around existing BERTopic components and can be adapted to other embedding, dimensionality reduction, and clustering choices.

The implementation is designed for scientific Python users. A single notebook activates RAPIDS cuML acceleration with `%load_ext cuml.accel`, constructs the BERTopic pipeline, defines metric functions, and runs Optuna trials. The accelerator targets common Python machine learning libraries including `sklearn`, `umap`, and `hdbscan`, falling back to CPU execution for unsupported cases [@cuml_accel; @cuml_limitations]. This keeps the workflow close to the code a data scientist would already write.

The contributions of this paper are:

- A scalarized objective for topic model selection that combines coherence, diversity, and target noise-percentage deviation.
- A BERTopic implementation that tunes UMAP and HDBSCAN hyperparameters with Optuna.
- A zero-code-change acceleration path using RAPIDS cuML for iterative topic-model evaluation.
- An empirical IMDb case study showing substantially lower target noise deviation with modest changes to coherence and diversity.

## Background and Related Work

LDA remains a foundational topic modeling method and is still useful when bag-of-words assumptions and a fixed number of topics are appropriate [@lda]. In noisy corpora, however, LDA requires practitioners to choose the number of topics and tune priors indirectly. It does not expose a native noise-cluster mechanism and does not directly optimize for the fraction of documents that receive interpretable topic assignments.

Neural topic models, including the Neural Variational Document Model (NVDM), represent documents with learned latent variables and can capture richer semantics than classical count-based approaches [@nvdm]. These models add flexibility, but they also introduce more training complexity and can be sensitive to the quality of the latent space. The motivating use case here is more operational: a practitioner already has a BERTopic-style pipeline and wants to improve how many documents become useful topic assignments.

BERTopic formalizes a modular neural topic modeling approach based on embeddings, clustering, and class-based TF-IDF topic representation [@bertopic]. A common BERTopic configuration uses UMAP for dimensionality reduction [@umap] and HDBSCAN for density-based clustering [@hdbscan]. UMAP parameters influence the local and global structure of the reduced embedding space. HDBSCAN parameters influence minimum cluster size, density thresholds, and outlier assignment. These interacting parameters make HPO attractive.

Multi-objective optimization is a natural alternative: coherence, diversity, and noise suppression can be treated as separate objectives, and a Pareto front can show non-dominated trade-offs. That representation is useful when the goal is exploration. It is less convenient when a practitioner needs one deployed model and has a clear preference, for example "keep coherence reasonable, but recover many documents from the noise cluster." BCDNM-HPO uses a scalar objective to encode those preferences directly.

The workflow uses Optuna, a define-by-run hyperparameter optimization framework that supports samplers such as the Tree-structured Parzen Estimator (TPE) [@optuna]. The paper's implementation also uses Gensim's topic coherence machinery to compute `c_v` coherence [@roeder_topic_coherence], along with a simple topic-word diversity score. The GPU acceleration path builds on RAPIDS cuML. The current cuML accelerator documentation describes a zero-code-change mechanism for accelerating existing Python machine learning code and notes that it targets `sklearn`, `umap`, and `hdbscan` [@cuml_accel].

## BCDNM Objective

BCDNM stands for Balanced Coherence, Diversity, and Noise-Suppression Metric. Let `C` denote topic coherence, `D` denote topic diversity, and `N` denote the percentage of documents assigned to the noise topic. Let `tau` be the target noise percentage, and let `w_c`, `w_d`, and `w_n` be nonnegative weights. The BCDNM objective used in this paper is:

```{math}
:label: eq:bcdnm
J_\mathrm{target} = w_c C + w_d D - w_n \left|\frac{N - \tau}{100}\right|
```

This objective rewards configurations whose topics are internally coherent and mutually diverse, while penalizing configurations whose noise-cluster percentage deviates from the target noise percentage. In the reference notebook, `tau` is set to 20, so the penalty is proportional to the absolute distance between the observed noise percentage and a 20% target. Throughout this paper, `C` and `D` are unitless scores, `N` and `tau` are percentages, and `|N - tau|` is reported in percentage points.

### Metric Definitions

In the current notebook implementation, coherence is computed with Gensim's `CoherenceModel` using the `c_v` score. For each non-noise topic, the top topic words returned by BERTopic are passed to the coherence model along with preprocessed documents and a Gensim dictionary. The noise topic `-1` is excluded.

Diversity is computed as the ratio of unique topic words to total topic words across all non-noise topics:

```{math}
:label: eq:diversity
D = \frac{|\mathrm{unique}(\cup_k W_k)|}{\sum_k |W_k|}
```

where `W_k` is the top-word list for topic `k`. This score is one when top-word lists do not overlap and decreases as different topics reuse the same words.

Noise percentage is computed directly from `topic_model.get_topic_info()`:

```{math}
:label: eq:noise
N = 100 \times \frac{\# \{d_i : topic(d_i) = -1\}}{\# \{d_i\}}
```

The target-deviation term used by the objective is:

```{math}
:label: eq:noise_deviation
\Delta_N = |N - \tau|
```

The formula normalizes this deviation by dividing by 100. For example, with `tau = 20`, the baseline noise percentage `N = 62.39` corresponds to `Delta_N = 42.39` percentage points and a normalized penalty input of 0.4239.

Together, these metrics convert the qualitative goal "find more usable, distinct, interpretable topics" into an optimization signal.

:::{figure} figures/bcdnm_pipeline_clean.png
:label: fig:pipeline
BCDNM-HPO wraps a BERTopic-style pipeline with metric evaluation and hyperparameter optimization. Each trial builds a topic model, evaluates coherence, diversity, noise percentage, and target noise deviation, and returns a scalar objective value to the optimizer.
:::

## Implementation

The implementation follows the structure of the public notebook `HPO_BERTopic_zero_code_change.ipynb` [@ding_notebook]. It begins with zero-code-change GPU acceleration:

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

Optuna evaluates candidate parameter sets by constructing the UMAP and HDBSCAN models, fitting BERTopic, computing `C`, `D`, and `N`, and returning the scalar score. The source notebook uses `TPESampler(seed=142)` and a maximization study for the weighted target-deviation score.

The loop is intentionally modular. The embedding model, dimensionality reduction algorithm, clustering algorithm, optimizer, and weights can all be changed. For example, practitioners could replace UMAP with PCA for a faster baseline, replace HDBSCAN with another clustering algorithm, or increase `w_n` for user-generated content where noise suppression is more important than preserving every high-coherence microtopic.

## Experimental Setup

The source experiment uses the IMDb movie review corpus, loaded through the Hugging Face `datasets` package as the `train` split of 25,000 reviews. IMDb is useful for this case study because long-form user reviews contain mixed sentiment, plot details, actor names, genre terms, HTML artifacts, and highly variable writing styles [@maas_imdb]. These characteristics produce a realistic setting in which many documents may sit between dense topic regions.

The baseline is the default notebook BERTopic pipeline with UMAP parameters `n_components=5`, `n_neighbors=15`, `min_dist=0.0`, and HDBSCAN parameter `min_samples=10`. The optimized model uses the best parameters returned by Optuna under the BCDNM objective.

The reported evaluation focuses on four quantities:

- `c_v` coherence, where higher values indicate more semantically interpretable topic word groups.
- Topic diversity, where higher values indicate less overlap among topic word lists.
- Noise percentage `N`, the percentage of documents assigned to topic `-1`.
- Target noise deviation `Delta_N = |N - tau|`, where lower values indicate closer agreement with the chosen target noise percentage.

The current source materials report aggregate metric values but not a full statistical replication across random seeds or hardware environments. Before final submission, the experiment should be rerun with recorded package versions, GPU model, CPU fallback logs, random seeds, number of Optuna trials, and wall-clock timings.

## Results

The source experiment uses `tau = 20`. The baseline assigns 62.39% of documents to the noise topic, corresponding to a target deviation of 42.39 percentage points. The optimized BCDNM-HPO configuration assigns 39.19% to noise, corresponding to a target deviation of 19.19 percentage points. This is a 23.20 percentage point absolute reduction in both observed noise percentage and target deviation, increasing the share of clustered, usable documents from 37.61% to 60.81%.

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
* - Noise percentage `N`
  - 62.39%
  - 39.19%
  - Closer to 20% target is better
* - Target noise deviation
  - 42.39 pp
  - 19.19 pp
  - Lower is better
* - Clustered document utilization
  - 37.61%
  - 60.81%
  - Derived as 100% - `N`
```

:::{figure} figures/metric_comparison.png
:label: fig:metrics
Metric comparison for the IMDb source experiment. BCDNM-HPO reduces target noise deviation substantially while retaining similar coherence and diversity.
:::

The coherence and diversity scores both decrease slightly. This is expected: moving the noise percentage closer to a 20% target can introduce harder boundary cases into explicit topics, and increasing clustered document utilization can trade off against tightly separated topic vocabularies. The BCDNM objective makes that trade-off visible and controllable. In this use case, the decrease from 0.69 to 0.66 in coherence and from 0.80 to 0.75 in diversity is modest relative to the gain in clustered documents.

The intertopic distance maps provide a qualitative view of the change. The optimized result contains more visible topic bubbles and a broader topic index range, consistent with the claim that documents previously assigned to noise are being represented by explicit topics. The maps should be interpreted as exploratory diagnostics rather than definitive proof of topic quality.

:::{figure} figures/baseline_intertopic.png
:label: fig:baseline_map
Baseline BERTopic intertopic distance map from the source materials.
:::

:::{figure} figures/bcdnm_intertopic.png
:label: fig:bcdnm_map
BCDNM-HPO intertopic distance map from the source materials. The optimized model assigns more documents to explicit topics and reports a noise percentage closer to the 20% target.
:::

## Discussion

The central advantage of BCDNM-HPO is not that it discovers a universally optimal topic model. Instead, it gives practitioners a small, inspectable objective that reflects the operational quality they care about. A model that maximizes coherence alone can produce a small number of clean topics while discarding many documents as noise. A model that minimizes noise alone can force ambiguous documents into low-quality clusters. BCDNM-HPO occupies the middle: it encourages noise assignment to move toward a chosen target while retaining coherence and diversity checks.

The method is also practical because it works with existing Python topic modeling components. BERTopic users already configure UMAP and HDBSCAN. Optuna adds a search loop around those parameters, while cuML acceleration can reduce the cost of repeated UMAP and HDBSCAN execution when the selected operations are supported on GPU. The current cuML documentation is explicit that unsupported cases may fall back to CPU and that accelerated implementations can differ numerically from CPU implementations [@cuml_limitations]. For a final proceedings paper, logging GPU and CPU fallback behavior should therefore be part of the reproducibility record.

The scalar objective also makes domain-specific priorities explicit. In legal or medical text mining, a user might increase `w_c` to keep topics semantically conservative. In exploratory customer feedback analysis, a user might increase `w_d` to broaden thematic coverage. In noisy social media or product review corpora, a user might increase `w_n` or choose a lower target noise percentage `tau` to make noise suppression more aggressive.

There are limitations. First, the reported experiment uses a single dataset and a small set of aggregate metrics. Second, coherence and diversity are proxy metrics; they do not guarantee that topics are useful to a human analyst. Third, the target noise percentage is a user-defined preference rather than a statistical optimum. Fourth, the HPO process can overfit to the objective if the search budget is large and no held-out validation or human review is used. Finally, reducing noise assignment is not always desirable: some documents truly may not belong to a coherent topic.

Future work should evaluate BCDNM-HPO across additional corpora, including product reviews, technical support tickets, social media posts, and domain-specific documents. It should also measure sensitivity to `w_c`, `w_d`, `w_n`, and `tau`, and report runtime speedups with and without cuML acceleration. Human topic evaluation would be especially valuable because the method is intended to improve practical interpretability, not only automatic metrics.

## Reproducibility Notes

The notebook artifact is public and gives a compact starting point for reproducing the workflow [@ding_notebook]. Before camera-ready submission, the following details should be added:

- Exact package versions for BERTopic, UMAP, HDBSCAN, Optuna, Gensim, RAPIDS cuML, CUDA, and `datasets`.
- GPU and CPU hardware, including whether each HPO trial used GPU execution or fell back to CPU.
- Number of Optuna trials, sampler configuration, random seeds, and best parameter values.
- Runtime for the baseline fit, each HPO study, and the final optimized fit.
- Topic quality validation beyond automatic metrics, such as manual review of a stratified sample of topics and documents recovered from noise.

## Conclusion

BCDNM-HPO provides a lightweight, reproducible way to tune BERTopic-style topic modeling pipelines for better document utilization. By combining coherence, diversity, and target noise-percentage deviation into a single scalar objective, the workflow allows practitioners to express the trade-off they want and then search the UMAP/HDBSCAN parameter space directly. In the IMDb source experiment with `tau = 20`, the method reduces observed noise percentage from 62.39% to 39.19% and target deviation from 42.39 to 19.19 percentage points while maintaining similar coherence and diversity. The result is not a replacement for human topic review, but it is a practical step toward topic models that waste less real-world data.
