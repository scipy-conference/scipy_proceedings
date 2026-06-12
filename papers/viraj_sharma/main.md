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

Raw activations predicted model behavioral state with 93.1% accuracy, while SAE text features achieved only 84.7% — an 8.3 percentage point gap above a 50% baseline. This gap represents a direct measurement of behavioral signal destroyed by the SAE's text-mediated encoding. Information that existed in the model's internal geometry did not survive the compression into human-readable features.

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

When entropy was computed over SAE feature activations rather than the model's output distribution, the ordering reversed — factual prompts produced higher SAE entropy (1.077) than counterfactual prompts (0.916), the opposite of what the model itself showed (5.78 vs 6.57). The SAE does not merely lose the model's uncertainty signal — it inverts it. This is clearly interpretive displacement.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-interpretive-displacement.ipynb)

The results are on google colab T4 GPU runtime.

---

## The Doctor Does Not Dictate the MRI

I thought of this during a discussion - let us try to create an analogy.
The field of medical imaging offers a good parallel to our case. A radiologist examining
an fMRI scan does not produce a complete verbal description of each voxel's activation
value. The image is not translated into a text report that a second clinician then
interprets rather than view the image. Instead, the radiologist develops, through
thousands of hours of supervised knowledge, a perception based competence — a capacity
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
