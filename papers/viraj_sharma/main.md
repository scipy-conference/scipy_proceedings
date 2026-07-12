---
# Ensure that this title is the same as the one in `myst.yml`
title: "Feel the model: Sensory Transduction of Neural Activations as a Human-in-the-Loop Safety"
abstract: |
  Current mechanistic interpretability methods, such as sparse autoencoders (SAEs), activation classifiers, and natural language autoencoders share a core assumption: that safety-relevant information in a model’s internal representations can be faithfully captured through human-understandable text. I find this to be limiting, as it is not only lossy but also directionally biased, discarding signals that do not align with linguistic categories.

  Using GPT-2 and published sparse autoencoders (SAEs), I compare classifiers trained directly on raw model activations with those trained on SAE-derived features. I show that cases of disagreement between these classifiers where SAE features predict one behavioral class while raw activations predict another—correlate with ground-truth misclassifications in the SAE-based classifier. This reveals a concrete failure mode, which I term interpretive displacement, wherein the text label assigned to an activation pattern actively misdirects human interpretability.
---
## Introduction

The main idea in mechanistic interpretability converts a model's internal activations
into human-readable text. Sparse Autoencoders (SAEs) break down residual stream activations into
a dictionary of labelled features. This work identifies a limitation in this approach. Prior work has argued that individual neurons are not natural units of human-understandable concepts [@bricken2023monosemanticity]. Probing methods have also demonstrated that internal representations can be analyzed through learned classifiers [@alain2016understanding].



## Background

### Sparse Autoencoders and Feature Dictionaries

Sparse Autoencoders decompose a model's residual stream activations
into a sparse combination of learned directions. Sparse autoencoders have emerged as a widely used method for decomposing model activations into sparse, interpretable feature representations [@cunningham2023sparse].

### Probing Classifiers

Probing classifiers are surveyed in:[@belinkov2021probing].

### The Common Bottleneck

Both methods have a common problem: the output of the interpretive process is constrained
to a pre-existing human vocabulary. SAEs label features using whatever words an annotator
or language model produces.



## The Information Loss Problem

The following two experiments examine differences between raw model activations and SAE-derived representations.

### Experiment 1: Classification Performance

To evaluate whether SAE-derived representations discard behaviorally relevant information, a comparative classification experiment was conducted.

To test this, I took GPT-2 and gave it 240 prompts - some it could answer confidently, some it couldn't. I measured how confused the model was for each prompt using entropy. Then I trained two simple classifiers - one using the model's raw internal activations, one using the SAE's text features - and asked both to predict whether the model was confused or not. The raw activation classifier was right 93% of the time. The SAE classifier was only right 85% of the time. The 8% gap is information that existed in the model's internals but got lost when I converted it to text features.

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

Raw activations predicted model behavioral state with 93.1% accuracy, while SAE text features achieved only 84.7%, an 8.3 percentage point gap above a 50% baseline. This gap represents a direct measurement of behavioral signal destroyed by the SAE's text-mediated encoding. Information that existed in the model's internal geometry did not survive the compression into human-readable features.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-info-loss-in-sae-v3.ipynb)

### Experiment 2: Interpretive Displacement

A second experiment examined whether SAE-derived features merely reduced predictive accuracy or could actively produce misleading interpretations.

I looked for cases where the two classifiers disagreed - where the raw activation classifier said "this model is confused" but the SAE classifier said "this model is confident." In those disagreement cases, I checked which classifier matched ground truth. The raw classifier was correct more often than the SAE classifier. This means the SAE is not just losing information - it is sometimes pointing in the wrong direction entirely. A human relying on SAE features to monitor the model would not just miss things; they might be actively misled.

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

When entropy was computed over SAE feature activations rather than the model's output distribution, the ordering reversed - factual prompts produced higher SAE entropy (1.077) than counterfactual prompts (0.916), the opposite of what the model itself showed (5.78 vs 6.57). This result suggests a form of interpretive displacement, where the SAE-derived representation appears to encode uncertainty differently from the underlying model.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-interpretive-displacement.ipynb)



## The Doctor Does Not Dictate the MRI

A useful analogy can be drawn from medical imaging.

The concept of tacit knowledge is relevant in this context [@polanyi1966tacit]. Dreyfus's analysis of expert skill acquisition identifies the transition from rule-following to holistic pattern recognition as the hallmark of genuine expertise [@dreyfus1980five]. Kahneman's System 1 characterises fast, pattern-sensitive judgment as structurally distinct from - and often more accurate than - deliberate propositional reasoning in familiar domains [@kahneman2011thinking].



## Proposed Framework: Activation Sensory Transduction (AST)

### Overview

Activation Sensory Transduction routes a compressed representation of model activations
directly to a human operator's sensory system, bypassing the requirement to assign
linguistic labels. One interpretation of AST is that it transfers part of the interpretive burden from automated feature-labeling systems to trained human operators. Through exposure to aligned, misaligned, and anomalous activation patterns, operators may develop the ability to generate safety-relevant assessments without requiring every activation pattern to be translated into language.

### Sensory Encoding Modalities

Four primary modalities:

**Visual** Create visualizations that can reveal information about activations through heatmaps, graphs, and perturbations rendered on two-dimensional representations. Operators trained on large collections of activation visualizations may develop the ability to identify patterns that are difficult to express through predefined textual labels.

**Auditory (sonification).** Create sound-based encodings - enabling operators to hear changes in model state and potential misalignment.

**Haptic/tactile.** Activation signals may also be encoded through touch, pressure, or vibration-based interfaces.

**BCI: The Brain–computer interface.** Multiple modalities may be combined. Attention-head activations could drive auditory parameters while MLP layer activations drive haptic patterns. This may increase the effective bandwidth of the transduction channel and potentially improve operator performance.

### The Human-in-the-Loop Architecture

The AST operator occupies a monitoring role analogous to a flight controller or intensive care nurse: trained on a corpus of labeled activation patterns (normal, anomalous, deceptive, degraded), the operator provides a complementary judgment channel that does not rely solely on text labels.



## Research Steps

I propose a three-phase research program to evaluate the viability of AST:

```{list-table} AST Research Program
:label: tbl:roadmap
:header-rows: 1
* - Phase
  - Goal
  - Primary Method
  - Key Metric
* - 1  -  Proof of Concept
  - Can humans discriminate behavioral classes via sonified, visual activations?
  - Controlled human-subject evaluation of sonified activation signals
  - Discrimination accuracy
* - 2  -  Learning Curve
  - Do humans improve with supervised exposure and feedback?
  - Assessment of operator learning under supervised training
  - Improvement in classification accuracy
* - 3  -  Comparative Oversight
  - Does AST detect anomalies missed by text-based interpretibility methods?
  - Head-to-head study: text labels vs. activation sonification
  - Miss rate, false alarm rate, response latency
```

Phase 1 is achievable with commodity hardware, standard audio-processing libraries, and online participants. Phases 2 and 3 require dedicated operator training and, eventually, BCI hardware integration for full haptic transduction.



## Experimentation

As a part of testing the proposition, a set of demonstrations which target different modalities were performed.

### Main setup

#### Model Activations API

As a part of testing the client modalities of a typical activation data, an API is created to generate activation vectors and activation sequences for clients.

It has two API endpoints:

/activate accepts a prompt and returns the model's internal state at the last token position only - a single vector representing the model's state at prediction time, along with the predicted output token and output entropy.

/sequence accepts the same prompt but returns the model's internal state at every token position - one vector per token, giving the full trajectory as the model reads the prompt.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-ast-backend.ipynb)

#### Visual Modalities

##### Phase Portrait

```{figure} phaseportrait.png
:alt: Phase portrait
:align: left
:width: 600px

Figure: Phase portrait of activation trajectories.
```

Interpretation of the Phase Portrait

The points crowd near (0, -5) to (0, -10) where many prompts' final tokens land. This region appears to correspond to a common pre-generation activation state.

The counterfactual paths (red) diverge most. "The capital of Valdoria is" goes down to (-30, -20) before returning. "The Zorblax protocol" starts far right at (30, +14). These trajectories exhibit attractor-like behavior, with simple prompts converging rapidly toward a stable region.

The open-ended trajectories (purple) exhibit distinctive dynamics. "Once upon a time" starts at (20, +6) and takes a long curved path. "The meaning of life is" starts at the top right (+32, +14) - the most distant starting point of any prompt. Open-ended prompts push the model into unfamiliar territory.

Phase portraits may serve as useful training examples for operators learning to recognize activation-space dynamics.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-phase-portrait.ipynb)

### Audio Modalities

A sonifier dashboard was created to convert activations to audible sound with several synthesis layers and mapping controls. Sonification parameters (pitch, timbre, spatialization, rhythm) are driven by reduced activation dimensions and entropy measures. Sonification outputs require testing with human subjects to determine perceptually useful mappings.

A demonstration of the AST sonifier is available at:
[https://www.youtube.com/watch?v=8D0n7ruvTxk](https://www.youtube.com/watch?v=8D0n7ruvTxk)

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-paper-2026-ast-sonification.ipynb)

### BCI modalities

The BCI modality explores whether model activations can be represented using formats familiar to neuroscience tools. Instead of converting activations into text labels, AST converts a compressed activation vector into a multi-channel signal that can be visualised and analysed using existing brain-signal software. The goal is not to claim that model activations are brain activity, but to investigate whether signal-analysis techniques developed for neuroscience can provide another way for humans to inspect model state.

MNE-Python is a widely used neuroscience visualisation tool. In AST it is used as a display layer that converts activation data into familiar signal plots, heatmaps, and channel views. The resulting visualisations provide another way to inspect model state without first translating activations into language.

The following example was generated from the prompt "France".

```{figure} ast_edf_mne.png
:alt: MNE generated EDF view
:align: left
:width: 600px

Figure: EDF view.
```

The EDF view presents the same activation data through several complementary visualisations. The channel traces show how activation strength varies across channels. The heatmap reveals which activation buckets are most active. Spectral views provide a summary of how activity is distributed across different signal components. Together, these views present the activation state as a visual pattern rather than a collection of text labels.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-edf-view.ipynb)



## Conclusion

The mechanistic interpretability literature has made substantial progress in understanding transformer model internals. The progress has been built on a paradigm that converts activations into text. This paper argues that the paradigm has a structural limitation: it can only reveal what our language can describe. Model internals that do not map cleanly onto human language are, by construction, invisible to text-mediated interpretability.

This is not an argument against SAEs, probing classifiers, or attribution methods. It is an argument that the field has a single channel - text - and that a single channel provides single-point-of-failure oversight. The same representational constraints that make a model's deceptive behavior hard to describe in text may make it visible to a trained sensory channel.

If a model state can generate a sensory signal that a trained human operator flags before a text-based probe names it, that is a safety gain - regardless of whether the operator can articulate what they perceived. The doctor does not need to dictate the MRI to act on what they see.
