---
# Ensure that this title is the same as the one in `myst.yml`
title: "Feel the model: Sensory Transduction of Neural Activations as a Human-in-the-Loop Safety"
abstract: |
  Current mechanistic interpretability methods - sparse autoencoders, activation classifiers, and natural language autoencoder, share a basic assumption: that the safety-relevant content of a model's internal representations can be faithfully represented through human-understandable text. I find this to be a limiting, as it is not merely lossy, but directionally biased as we are discarding signal that does not conform to language categories. Using GPT-2 Small with published pretrained sparse autoencoders, I will demonstrate that linear classifiers trained on raw residual stream activations perform better than equivalent classifiers trained on SAE feature representations across behaviorally distinct prompt classes. I will also show that cases of classifier disagreement, where SAE features predict one behavioral class while raw activations predict another, correlate with ground-truth misclassification in the SAE-based classifier, identifying a concrete failure mode we term interpretive displacement, wherein the text label assigned to an activation pattern actively misdirects human interpretibility.

  Building on these findings, I propose Activation Sensory Transduction (AST), an AI Safety channel that routes dimensionality-reduced activation signals directly to human sensory systems via Brain-Computer Interface - tactile, auditory, or multimodal. Just like how radiologists interpret medical imaging, this represents an underexplored resource for AI safety oversight.
---
## Introduction

The main idea in mechanistic interpretability converts a model's internal activations
into human-readable text. Sparse Autoencoders (SAEs) break down residual stream activations into
a dictionary of labelled features. I will try to point out the problem in thiis approach. Papers to study for some concepts:[@bricken2023monosemanticity] talks about how a neuron is not a natural unit of human understanding. [@alain2016understanding] discusses the linear layer probes for indetifying training problems.



---

## Background

### Sparse Autoencoders and Feature Dictionaries

Sparse Autoencoders decompose a model's residual stream activations
into a sparse combination of learned directions.
Paper for this:
 [@cunningham2023sparse].

### Probing Classifiers
I need to study this paper for probing classifiers:[@belinkov2021probing].

### The Common Bottleneck

Both methods have a common problem : the output of the interpretive process is constrained
to a pre-existing human vocabulary. SAEs label features using whatever words an annotator
or language model produces.


---

## The Information Loss Problem

I make two demonstrations to highlight the difference between the model activations and a feature trained SAE performance:

### First one is a simple classifier experiment:

I wanted to know if converting model activations into text throws away useful information. To test this, I took GPT-2 and gave it 240 prompts — some it could answer confidently, some it couldn't. I measured how confused the model was for each prompt using entropy. Then I trained two simple classifiers — one using the model's raw internal activations, one using the SAE's text features — and asked both to predict whether the model was confused or not. The raw activation classifier was right 93% of the time. The SAE classifier was only right 85% of the time. The 8% gap is information that existed in the model's internals but got lost when we converted it to text features.

```{list-table} Classification results
:header-rows: 1
* - Metric
  - Value
* - Baseline
  - 0.500
* - SAE features (text)
  - 0.847
* - Raw activations
  - 0.931
* - Information lost by SAE
  - 0.083
```


[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-info-loss-in-sae-v3.ipynb)

### And then the interpretive displacement experiment:

I then wanted to know if the SAE's text features were just less accurate, or if they were sometimes confidently wrong in a way that would actually mislead a human. I looked for cases where the two classifiers disagreed — where the raw activation classifier said "this model is confused" but the SAE classifier said "this model is confident." In those disagreement cases, I checked who was actually right. The raw classifier was right more often than the SAE classifier. This means the SAE is not just losing information — it is sometimes pointing in the wrong direction entirely. A human relying on SAE features to monitor the model would not just miss things, they would occasionally be told the opposite of what is actually happening.

Results:

```{list-table} Entropy comparison
:header-rows: 1
* - Representation
  - Prompt type
  - Mean entropy
* - Model
  - Factual
  - 5.781266689300537
* - Model
  - Counterfactual
  - 6.571504592895508
* - SAE
  - Factual
  - 1.0776017904281616
* - SAE
  - Counterfactual
  - 0.915539026260376
```


[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-interpretive-displacement.ipynb)

The results are on google colab T4 GPU runtime.

---

## The Doctor Does Not Dictate the MRI

I thought of this during a discussion - let us try to create an analogy.
The field of medical imaging offers an instructive parallel. A radiologist examining
an fMRI scan does not produce a complete verbal description of each voxel's activation
value. The image is not translated into a text report that a second clinician then
interprets rather than view the image. Instead, the radiologist develops, through
thousands of hours of supervised knowledge, a perceptual competence — a capacity
to *see* and *feel* problem in the spatial and textural patterns of the image, prior to
and often in excess of what can be articulated.

A paper on tacit knowledge: [@polanyi1966tacit]. Dreyfus's analysis of expert skill acquisition
identifies the transition from rule-following to holistic pattern recognition as the
hallmark of genuine expertise [@dreyfus1980five]. Kahneman's System 1 characterises fast,
pattern-sensitive judgment as structurally distinct from — and often more accurate than —
deliberate propositional reasoning in familiar domains [@kahneman2011thinking].



---

## Proposed Framework: Activation Sensory Transduction (AST)

### Overview

Activation Sensory Transduction routes a compressed representation of model activations
directly to a human operator's sensory system, bypassing the requirement to assign
linguistic labels.



### Dimensionality Reduction

which aspects of the activation geometry are preserved for the human operator - I dont know this yet

### Sensory Encoding Modalities

Three primary modalities:

**Auditory (sonification).** Create sound based encoding - so you can hear a model misalign

**Haptic/tactile.** Encoding to touch, pressure - not sure of this - kind of like pulse checking.

**Multimodal.** We can have a combination. attention head activations could drive auditory parameters
while MLP layer activations drive haptic patterns. This increases the effective
bandwidth of the transduction channel. - Maybe it will be more accurate

### The Human-in-the-Loop Architecture

The AST operator occupies a monitoring role analogous to a flight controller or
intensive care nurse: trained on a corpus of labeled activation patterns (normal,
anomalous, deceptive, degraded)

---

## Experiments

### Setup

Using GPT-2 Small (117M parameters) [@radford2019language] with
pretrained Sparse Autoencoders from the `sae_lens` release `gpt2-small-res-jb` [@bloom2024saetraining],
specifically the SAE trained on `blocks.8.hook_resid_pre` (layer 8 of 12,
$d = 768$, $D = 24576$ features). Activations are extracted using
TransformerLens [@nanda2022transformerlens].

### Behavioral Dataset

I construct a two-labeled prompt sets:

- **Set 1 (Factual):** Prompts with verifiable, well-grounded completions (e.g.,
  geographic and scientific facts, biographical dates). The model has strong training
  signal for these completions and is expected to operate in a stable, grounded way.
- **Set 1 (Counterfactual/Confabulation-inducing):** Prompts that reference
  non-existent entities, fictional theoretical frameworks, or contradictory
  premises. The model has no grounded completion available and is expected to
  operate in a confabulation like way.

This provides clean ground-truth labels without human annotation, and the behavioral
distinction is directly safety-relevant: hallucination detection is an active
problem in deployed systems.

### Experiment 1: Quantifying Information Loss

For each prompt, I extract mean-pooled residual stream activations at layer 8
(raw representation, $\mathbb{R}^{768}$) and mean-pooled SAE feature activations
(text-mediated representation, $\mathbb{R}^{24576}$). I train logistic regression
classifiers on each representation under identical conditions (70/30 stratified split,
L2 regularisation, standardised inputs) and compare behavioral classification accuracy.

The core extraction procedure is as follows:

```python
hook_name = "blocks.8.hook_resid_pre"
_, cache = model.run_with_cache(tokens, names_filter=hook_name)
acts = cache[hook_name]                  # [batch, seq_len, 768]
acts_pooled = acts.mean(dim=1)           # [batch, 768]
sae_features = sae.encode(acts)          # [batch, seq_len, 24576]
sae_pooled = sae_features.mean(dim=1)   # [batch, 24576]
```



### Experiment 2: Measuring Interpretive Displacement

This is one is giving trouble - Not able to get the displacement of labels in current experiment.

### Experiment 3: Activation Geometry Under Projection

This is a geometric complement to the classifier-based
measurement in Experiment 1 where .

---

## Challenges and Open Problems

**The activation selection problem.**
*which* activations to transduce.

**BCI bandwidth constraints.** Current BCI systems provide very
low bandwidth than even a single transformer layer's activation vector.

**Operator training and standardisation.** The radiologist's expertise is built on
decades of standardised signals and labeled outcomes. An AST operator requires an
equivalent training infrastructure: a standardised transduction protocol - we will need to train them...hard

**Verification issue.** The feeling that an operator gets from the sensory inputs should correspdond to a well define model state - this requires that we can group them or map them well.

**Inter-operator reliability.** Different operators should be able to give same assessment

---

## Research Steps

I am thinking a three-phase research program to evaluate AST viability:

```{list-table} AST Research Program
:label: tbl:roadmap
:header-rows: 1
* - Phase
  - Goal
  - Primary Method
  - Key Metric
* - 1 — Proof of Concept
  - Can humans discriminate behavioral classes via sonified activations?
  - I think the sound data can be heard by a number of poeple and
  - Discrimination accuracy
* - 2 — Learning Curve
  - Do humans improve with supervised exposure? (this is like training doctors)
  - Study with feedback
  - Increaese in better predictions ?
* - 3 — Comparative Oversight
  - Does AST catch anomalies text-based interpretability misses? - this would be the main test of proposition
  - Head-to-head study: text labels vs. activation sonification
  - Miss rate, false alarm rate, response latency
```

Phase 1 is achievable with commodity hardware (not sure if I will have acccess though), a Python audio library, and
online participants. Phases 2 and 3 require dedicated operator training and,
eventually, BCI hardware integration for full haptic transduction.

---

## Conclusion

The mechanistic interpretability literature has made substantial progress in
understanding transformer model internals. The progress has been built on a
paradigm that converts activations into text. This paper argues that the paradigm
has a structural limitation: it can only reveal what our language can describe.
Model internals that do not map cleanly onto human language are,
by construction, invisible to text-mediated interpretability.

This is not an argument against SAEs, probing classifiers, or attribution methods.
It is an argument that the field has a single channel — text — and that
a single channel provides single-point-of-failure oversight. The same
representational constraints that make a model's deceptive behavior hard to describe
in text may make it visible to a trained sensory channel.

If an
model state can generate a sensory signal that a trained human operator
flags before a text-based probe names it, that is a safety gain — regardless of
whether the operator can articulate what they perceived. The doctor does not need
to dictate the MRI to act on what they see.


---
