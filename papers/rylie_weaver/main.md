---
# Ensure that this title is the same as the one in `myst.yml`
title: "alphagenome-pt: Flexible Training of AlphaGenome Models in PyTorch"
abstract: |
  We present `alphagenome-pt`, an open-source PyTorch implementation of Google DeepMind’s AlphaGenome model for training and fine-tuning. AlphaGenome predicts functional genomic outputs directly from DNA sequence, but the released models are trained on human and mouse genomes, motivating dataset-specific training for other species and biological contexts. Because the released implementation is written in JAX, `alphagenome-pt` lowers the software barrier to adapting AlphaGenome by porting the model to PyTorch and exposing relevant model hyperparameters to custom configuration.
---

## Introduction

We present `alphagenome-pt` ([PyPI](https://pypi.org/project/alphagenome-pt/), [GitHub](https://github.com/RylieWeaver/AlphaGenome_PyTorch)), an open-source PyTorch implementation of Google DeepMind’s AlphaGenome model architecture for training and fine-tuning. DNA language models such as AlphaGenome learn representations of DNA sequence and can be broadly grouped into two categories based on their prediction objectives: (1) masked language modeling (MLM), which tasks the model with recovering the original DNA sequence given a corrupted input [@EVO2-Nature-2026; @EVO-nguyen-2024; @HyenaDNA-NeurIPS-2023; @DNABERT-2-arxiv-2024; @DNABert-Bioinformatics-2021; @NucleotideTransformer-Nature-2025; @NucleotideTransformer-biorxiv-2023; @PlantCAD2-2025], and (2) sequence-to-function modeling, which tasks the model with predicting experimental measurements of biological quantities [@AlphaGenome-Nature-2026; @Enformer-Nature-2021; @DeepSEA-Nature-2015; @Basenji-Nature-2018; @Borzoi-Nature-2025; @SpliceAI-Nature-2019; @Pangolin-PubMed-2022].[^model-categories] Example functional prediction tasks include gene expression, chromatin accessibility, splicing, and chromatin contact maps, each of which captures a different aspect of genome function. By serving as computational surrogates for physical experiments, these models provide fast estimates for quantities that would otherwise be relatively expensive and time-consuming to measure. Moreover, DNA sequence inputs can be precisely specified and systematically modified in silico, enabling controlled perturbations that are difficult or impossible to perform experimentally.

[^model-categories]: The two DNA language model categories are not mutually exclusive since prediction tasks can be swapped while keeping the majority of the model the same. In particular, it's common and often useful to pretrain DNA language models with the MLM task before fine-tuning on sequence-to-function tasks. However, the separation is a useful framework for understanding the differences between DNA language models at a high level.

Functional measurements are useful in downstream biological workflows. Consequently, sufficiently inexpensive and accurate predictions of those measurements are similarly useful, but sequence-to-function prediction remains a central challenge in genomics because of the complexity of the relationships that must be modeled. AlphaGenome is a DNA language model that advances sequence-to-function prediction by modeling a wide range of functional genomic outputs directly from DNA sequence for human and mouse genomes [@AlphaGenome-Nature-2026]. However, Google DeepMind’s released AlphaGenome models are trained on human and mouse genomes, limiting their direct application to other species. Practitioners who want to apply AlphaGenome to other species may therefore need to train or fine-tune models with the same core architecture on species-specific data. At the same time, the released implementation is written in JAX [@JAX-GitHub-2018], while many deep learning workflows in the scientific Python community are built in PyTorch [@PyTorch-NeurIPS-2019]. `alphagenome-pt` addresses this gap by allowing researchers to instantiate, modify, train, and checkpoint AlphaGenome models in PyTorch without reimplementing the model themselves. By lowering the software barrier to training and fine-tuning AlphaGenome, `alphagenome-pt` aims to make the model easier to adapt to new biological settings.


## Goals and Motivations

The goal of `alphagenome-pt` is to make the AlphaGenome model easily trainable in PyTorch, with users only needing to know minimal details about the architecture.

Fine-tuning is a well-established and common method to adapt and/or specialize deep learning models to specific domains or tasks. Fine-tuning is useful for language models generally [@TransferLearning-IEEE-2010; @TransferLearning-arxiv-2014; @ULM-Finetuning-arxiv-2018; @BERT-arxiv-2018; @DNABert-Bioinformatics-2021; @HyenaDNA-NeurIPS-2023; @DNABERT-2-arxiv-2024], but is especially relevant for the AlphaGenome architecture because the model allocates a significant amount of parameters that are specific to organisms and output tracks. These include organism embedding parameters of shape `[num_organisms, num_channels]` and multiorganism linear layers of shape `[num_organisms, num_in_channels, num_out_channels]` used in the output heads [^footnote-1]. In the case of the multiorganism linear layers for some output heads (e.g. RNA-Seq gene expression), different output channels may correspond to different biological contexts. This architectural choice is grounded in the biological reality, where the same DNA sequence can produce different measured values (e.g. expression, chromatin accessibility, splicing, contact-maps) depending on biological context (e.g. tissue, cell type, developmental stage, environment, and organism).

[^footnote-1]: Parameter shapes are expressed up to permutation.

However, this may present a problem when applying the model to other species or biological contexts. In such cases, a user has three options:
(1) Initialize the new parameters randomly, which gives no useful prior and, in the case of the multiorganism linear layers, random output predictions.
(2) Reuse the closest existing organism or track, which may be reasonable for closely related settings but becomes weaker as the biological context diverges.
(3) Finetune the model on the new species/contexts.

Although future AlphaGenome releases may incorporate more species and biological contexts, it is difficult for any released model to cover every species and context of interest. In many cases, the best solution may be to train or fine-tune AlphaGenome on user-specific data. `alphagenome-pt` is designed to make that option practical.


## AlphaGenome

AlphaGenome is a hybrid convolutional-transformer architecture that balances long-context with computational feasibility while maintaining single base-pair resolution in its predictions. At a high level, AlphaGenome can be viewed as a U-Net-like encoder-decoder model [@UNet-MICCAI-2015] with transformer layers [@Transformers-NeurIPS-2017] in the middle.

:::{figure} images/AG_Total.png
:label: fig:alphagenome-architecture
Architecture of the AlphaGenome model, including its overall encoder-transformer-decoder structure and its row attention mechanism.
:::

The model begins with an encoder that consists of seven convolutional downsampling blocks, which capture short-range dependencies and coarsen the sequence from 1bp to $2^7=128$bp resolution. The resulting 128bp embeddings are then input to transformer layers [@Transformers-NeurIPS-2017], which capture long-range dependencies over the full context at reduced sequence length. The transformer attention logits are adjusted by an attention bias computed from pairwise sequence representations at a coarser $(2048 \times 2048)$ bp resolution. These pairwise representations are updated with a pair-to-pair attention mechanism (called row attention in AlphaGenome) where each pair representation attends across its row, which improves the model's capability to capture higher-order relationships. Finally, AlphaGenome applies a decoder that consists of seven convolutional upsampling blocks, which capture short-range dependencies and restores the sequence 1 bp resolution. U-Net-style skip connections [@UNet-MICCAI-2015] with the fine-grained encoder representations are used to prevent information loss.

The resulting 1bp, 128bp, and $(2048 \times 2048)$ bp representations are then passed to task-specific output heads. In total, the encoder-transformer-decoder structure allows AlphaGenome to retain fine-grained information while also incorporating long-range context.


## Use Cases and Design Philosophy

The core use case of `alphagenome-pt` is to allow users to train the AlphaGenome model on their own datasets to predict functional genomic tracks of interest, including RNA-seq, CAGE, PRO-cap, ATAC-seq, DNase-seq, transcription factor ChIP-seq, histone-mark ChIP-seq, splice-site classification, splice-site usage, splice-junction prediction, and chromatin contact maps.

Instantiate the model and get embeddings
```python
import random, torch
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, DataBatch
S = 2048
metadata = {'organisms': ['human', 'mouse']}
model_cfg = AlphaGenomeConfig(max_seq_len=S, num_channels=96, metadata=metadata)
model = AlphaGenome(model_cfg)
sequence = "".join(random.choices("ACGTN", k=S))
dna_sequence = torch.tensor(
    [[[base == nucleotide for nucleotide in "ACGT"] for base in sequence]],
    dtype=torch.float32,
)
data = DataBatch(dna_sequence=dna_sequence, organism_index=torch.tensor([0]))
predictions, embeddings = model(data)  # NOTE: predictions are empty here because we haven't defined any output heads
print(embeddings.embeddings_1bp.shape, embeddings.embeddings_128bp.shape, embeddings.embeddings_pair.shape)
```

Metadata specifies the organism and output heads for the model architecture. Real workflows will need to calculate the true means of nonzero target values, but dummy values are made here for simplicity.
```python
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, Metadata, synthetic_batch

# Dummy Example: replace with actual nonzero means
def make_means(num_tracks):
    max_t = max(num_tracks)
    return [[1.0] * max_t for _ in num_tracks]

metadata = Metadata({
    "organisms": ["human", "mouse"],
    "heads": {
        # Gene expression / transcriptional activity
        "rna_seq": {
            "num_tracks": [10, 8],
            "means": make_means([10, 8]),
        },
        "cage": {
            "num_tracks": [6, 4],
            "means": make_means([6, 4]),
        },
        "procap": {
            "num_tracks": [3, 2],
            "means": make_means([3, 2]),
        },

        # Chromatin accessibility / regulatory binding
        "atac": {
            "num_tracks": [5, 4],
            "means": make_means([5, 4]),
        },
        "dnase": {
            "num_tracks": [5, 5],
            "means": make_means([5, 5]),
        },
        "chip_tf": {
            "num_tracks": [7, 6],
            "means": make_means([7, 6]),
        },
        "chip_histone": {
            "num_tracks": [4, 3],
            "means": make_means([4, 3]),
        },

        # Splicing
        "splice_sites_classification": {
            "num_tracks": [5, 5],
        },
        "splice_sites_usage": {
            "num_tracks": [4, 3],
        },
        "splice_sites_junction": {
            "num_tissues": [4, 3],
        },

        # Contact maps
        "contact_maps": {
            "num_tracks": [2, 2],
        },

        # Masked Language Modeling
        "masked_language_modeling": {},
    },
})

# Build a small model and a synthetic batch containing targets for every head.
S = 2048
model_cfg = AlphaGenomeConfig(max_seq_len=S, num_channels=96, metadata=metadata)
model = AlphaGenome(model_cfg)
data = synthetic_batch(
    metadata,
    batch_size=1,
    seq_len=S,
    num_splice_sites=model_cfg.num_splice_sites,
)
```

Access predictions and loss values once a model has been instantiated and data has been fed to it:
```python
predictions, embeddings = model(data)
total_loss, scalars, predictions = model.loss(data)

# Genome-track heads:
# rna_seq, cage, procap, atac, dnase use 1 bp and 128 bp outputs.
predictions["rna_seq"]["predictions_1bp"]
predictions["rna_seq"]["predictions_128bp"]
scalars["rna_seq_loss"]

# chip_tf and chip_histone use 128 bp outputs.
predictions["chip_tf"]["predictions_128bp"]
scalars["chip_tf_loss"]

# Contact maps.
predictions["contact_maps"]["predictions"]
scalars["contact_maps_loss"]

# Splicing heads.
predictions["splice_sites_classification"]["predictions"]
predictions["splice_sites_usage"]["predictions"]
predictions["splice_sites_junction"]["predictions"]
scalars["splice_sites_classification_loss"]
scalars["splice_sites_usage_loss"]
scalars["splice_sites_junction_loss"]

# Masked language modeling.
predictions["masked_language_modeling"]["predictions"]
scalars["masked_language_modeling_loss"]

# Intermediate representations.
embeddings.embeddings_1bp
embeddings.embeddings_128bp
embeddings.embeddings_pair
```

When training from scratch, our design philosophy is to let users control the most important model parameters while still defaulting toward the original AlphaGenome architecture. For example, users can directly set parameters such as `num_channels` and `max_seq_len`, which control the model width and input sequence length. Other hyperparameters, including `channel_increment`, `qk_head_dim`, `v_head_dim`, `pair_channels`, `num_splice_sites`, and `splice_site_channels`, can also be specified manually. However, when these values are not provided, `alphagenome-pt` infers defaults that keep their proportions consistent with the published model. This allows users to scale the model up or down via `num_channels` and `max_seq_len` without needing to manually adjust every dependent hyperparameter.

```python
from alphagenome_pt import AlphaGenomeConfig, official_alphagenome_config

official_cfg = official_alphagenome_config()
small_cfg = AlphaGenomeConfig(num_channels=96, max_seq_len=2048)

# Same proportions
assert small_cfg.channel_increment / small_cfg.num_channels == official_cfg.channel_increment / official_cfg.num_channels == 1 / 6
assert small_cfg.qk_head_dim / small_cfg.num_channels == official_cfg.qk_head_dim / official_cfg.num_channels == 1 / 6
assert small_cfg.v_head_dim / small_cfg.num_channels == official_cfg.v_head_dim / official_cfg.num_channels == 1 / 4
assert small_cfg.pair_channels / small_cfg.num_channels == official_cfg.pair_channels / official_cfg.num_channels == 1 / 6
assert small_cfg.num_splice_sites / small_cfg.max_seq_len == official_cfg.num_splice_sites / official_cfg.max_seq_len == 1 / 2048
assert small_cfg.splice_site_channels / small_cfg.num_channels == official_cfg.splice_site_channels / official_cfg.num_channels == 1
```

In addition to training from scratch, `alphagenome-pt` supports training from converted AlphaGenome checkpoints. The package provides utilities for instantiating a checkpoint-compatible public AlphaGenome configuration, converting the released JAX parameters to PyTorch, and loading those parameters into the PyTorch model. This is useful for users who want to fine-tune from released AlphaGenome weights rather than begin from random initialization. At the same time, checkpoint loading is still a current limitation of the package. At present, `alphagenome-pt` supports loading the all-folds model and folds 0, 1, 2, and 3, but we are still validating numerical equivalence with the original JAX implementation.

To get the full public AG config
```python
from pathlib import Path
from alphagenome_pt import (
    AlphaGenome,
    load_alphagenome_checkpoint,
    official_alphagenome_config,
)

# Instantiate full official model
cfg = official_alphagenome_config()
model = AlphaGenome(cfg)

# Downloads from Hugging Face if the checkpoint is not already present
checkpoint_path = Path("checkpoints/alphagenome_all_folds.pt")
load_result = load_alphagenome_checkpoint(
    model,
    checkpoint_path,
    fold="all_folds",
    repo_id="RylieWeaver/alphagenome-pytorch",
    repo_dir="v0.3.0",
    heads=True,         # keep released output heads
    organisms=True,     # keep released human/mouse organism parameters
    map_location="cpu",
)

# Check for any missed mappings
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)
```

For fine-tuning with custom heads, skip the released heads:
```python
# Instantiate official model except for custom heads
custom_metadata = {
    "organisms": ["human", "mouse"],
    "heads": {
        "rna_seq": {
            "num_tracks": [10, 8],
            "means": [[1.0] * 10, [1.0] * 10],
        },
    },
}
cfg = official_alphagenome_config(metadata=custom_metadata)
model = AlphaGenome(cfg)

# Downloads from Hugging Face if the checkpoint is not already present
checkpoint_path = Path("checkpoints/alphagenome_fold_1.pt")
load_result = load_alphagenome_checkpoint(
    model,
    checkpoint_path,
    fold="fold_1",
    repo_id="RylieWeaver/alphagenome-pytorch",
    repo_dir="v0.3.0",
    heads=False,       	# skip released heads; keep your custom heads
    organisms=True,     # keep human/mouse organism embeddings
    map_location="cpu",
)
```

For new organisms, also skip organism-specific parameters
```python
# Instantiate official model except for custom heads
custom_metadata = {
    "organisms": ["poplar", "arabidopsis"],
    "heads": {
        "rna_seq": {
            "num_tracks": [10, 8],
            "means": [[1.0] * 10, [1.0] * 10],
        },
    },
}
cfg = official_alphagenome_config(metadata=custom_metadata)
model = AlphaGenome(cfg)

# Downloads from Hugging Face if the checkpoint is not already present
checkpoint_path = Path("checkpoints/alphagenome_fold_2.pt")
load_result = load_alphagenome_checkpoint(
    model,
    checkpoint_path,
    fold="fold_2",
    repo_id="RylieWeaver/alphagenome-pytorch",
    repo_dir="v0.3.0",
    heads=False,       	    # skip released heads; keep your custom heads
    organisms=False,        # skip human/mouse organism embeddings
    map_location="cpu",
)
```

Overall, `alphagenome-pt` is designed to keep the model customizable, but faithful to the original AlphaGenome while making it easier to adapt for new datasets in PyTorch workflows.


## Limitations and Related Work

The first main area of limitation in `alphagenome-pt` is checkpoint loading. The package can load converted AlphaGenome parameters for the all-folds model and folds 0, 1, 2, and 3 into the PyTorch implementation, and provides a checkpoint-compatible public AlphaGenome configuration for this purpose. However, we are still validating that loading from checkpoint produces numerically equivalent outputs to the original JAX implementation. As a result, the most reliable use case of the package at present is training and experimentation in PyTorch, rather than exact reproduction of every released AlphaGenome checkpoint.

The second main area of limitation in `alphagenome-pt` is that it does not provide any preprocessing pipeline for functional genomics data. However, this is an intentional scope choice. AlphaGenome supports many target types, each of which may be stored in multiple common file formats, and large-scale HPC training may require different storage formats than smaller experiments. Rather than assuming a particular file format or data-loading strategy, `alphagenome-pt` assumes that users yield tensorized data to the model, however that is done. The package focuses on the model rather than prescribing a universal data-processing workflow.

The most closely related software is the official AlphaGenome release from Google DeepMind ([Link1](https://github.com/google-deepmind/alphagenome_research), [Link2](https://github.com/google-deepmind/alphagenome)). That implementation remains the reference implementation for the model and released checkpoints, but it is written in JAX, whereas `alphagenome-pt` is motivated by the complementary goal of making the model trainable in PyTorch. There are also other unofficial PyTorch implementations of AlphaGenome. The [`genomicsxai/alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch) package is a strong and polished PyTorch port, with pretrained checkpoint loading, named outputs, example notebooks, fine-tuning utilities, and reported numerical parity with the original JAX model. In contrast, `alphagenome-pt` is focused on flexible model training. In particular, `alphagenome-pt` includes a masked-language-modeling pretraining head, exposes more architecture hyperparameters for modification, and implements RMS batch normalization using current-batch statistics during training. This differs from the `genomicsxai` implementation, where RMS batch normalization uses stored running variance in the training forward passes. As a result, `alphagenome-pt` allows training gradients to flow through the normalization statistics induced by the current batch, which may be important when training AlphaGenome models from scratch or substantially changing their training distribution. A previous PyTorch implementation by Phil Wang (also known as lucidrains on GitHub), Miquel Anglada-Girotto, and Xinming Tu is also available at [lucidrains/alphagenome](https://github.com/lucidrains/alphagenome).

Relative to these related projects, the goal of `alphagenome-pt` is not to replace the official implementation or provide the most complete inference interface. Instead, the goal is to provide a clear and maximally flexible PyTorch training implementation for users who want to adapt AlphaGenome to new datasets, species, and biological contexts while still remaining faithful to the original architecture.


## Conclusions and Future Work

We presented `alphagenome-pt`, an open-source PyTorch implementation of AlphaGenome for training and fine-tuning. The goal of the package is to make it easier for researchers to train AlphaGenome models on their own datasets, species, and biological contexts without reimplementing the model themselves.

By lowering the software barrier to AlphaGenome training, `alphagenome-pt` aims to make sequence-to-function modeling more accessible beyond the released human and mouse models. More broadly, we hope this helps researchers use computational models to generate and test biological hypotheses faster.

Future work will focus on validating checkpoint equivalence with the original JAX implementation. We also plan to improve documentation and add distributed sequence-parallel training to support longer sequences and larger-scale HPC training.


## Generative AI Disclosure

Portions of this work were assisted using ChatGPT and Codex. The tools were used for drafting and revising code in the package and text for this proceedings submission. All outputs were reviewed, verified, and revised by the author(s), who take full responsibility for the accuracy and integrity of the final content.
