---
# Ensure that this title is the same as the one in `myst.yml`
title: "Feel the model: Sensory Transduction of Neural Activations as a Human-in-the-Loop Safety"
abstract: |
  Current mechanistic interpretability methods, such as sparse autoencoders (SAEs), activation classifiers, and natural language autoencoders share a core assumption: that safety-relevant information in a model's internal representations can be faithfully captured through human-understandable text. I find some evidence that this assumption may not hold well for unusual or out-of-distribution internal states.

  Using GPT-2 and a published sparse autoencoder, I find that SAE reconstruction quality tends to drop on unusual inputs (source code, repeated-token sequences, keyboard mashing, non-English text) compared to ordinary text, and that in several cases, the existing text labels for the most active SAE features on these inputs describe unrelated, ordinary concepts instead of flagging the input as unusual. This points to a possible failure mode: a human operator monitoring only text-based feature labels might get no warning exactly when the underlying activation looks least like anything the labeling process was built to describe. This motivates Activation Sensory Transduction (AST), a framework proposed in this paper that tries to give a human operator a way to inspect activation information directly, through non-linguistic senses, instead of only through text.
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

The following experiment looks at whether SAE-derived representations still signal that something is unusual when the underlying model activation is unusual, rather than just testing which representation makes a "better" classifier feature.

### Anomaly Blind-Spot Test

To test this, I built two sets of 20 prompts for GPT-2 (the 124M-parameter model, `d_model` 768): a "normal" set of ordinary English sentences, and an "anomalous" set of inputs GPT-2 can still process but that do not look like typical training text - source code, repeated-token sequences, keyboard mashing, and non-English text. For each prompt, I took the last-token residual-stream activation at `blocks.8.hook_resid_pre` and ran it through a published sparse autoencoder (`gpt2-small-res-jb`), then measured how much of the original activation survived being compressed and rebuilt by the SAE (fraction of variance explained, or FVE).

On average, normal prompts kept 94.8% of their variance (SD 1.1%), while anomalous prompts kept only 82.8% (SD 4.5%), a gap that looks unlikely to be chance (Mann-Whitney U test, p < 0.0001). In this sample, the SAE reconstructed anomalous activations noticeably worse than ordinary ones.

```{list-table} Reconstruction fidelity, normal vs. anomalous prompts
:header-rows: 1
* - Prompt type
  - Mean FVE
  - SD
* - Normal
  - 0.948
  - 0.011
* - Anomalous
  - 0.828
  - 0.045
```

I then looked, for each anomalous prompt, at which SAE feature fired most strongly and what text description that feature has in Neuronpedia's existing public explanations for this SAE. A few of the resulting labels seemed to have little to do with the actual input: repeated-token spam ("the the the...") was labeled "phrases related to physical actions or confrontations," a string of hex byte values was labeled as referring to "the 21st century," and keyboard-mashed text was labeled as referring to a made-up named entity. In these cases, a human operator relying only on the text label would likely get no indication that anything unusual was happening, even though the underlying activation was measurably unusual by reconstruction error. A similar pattern showed up across a few different kinds of anomalous input (code, repeated tokens, keyboard mashing, hex values, and foreign-language text), though this was based on reading through examples rather than a formal count; scoring label accuracy systematically over a larger set is left for future work.

This looks like a concrete example of the paper's central concern: the text layer may not just lose information overall, it can sometimes fail quietly - giving a plausible-sounding but unrelated label right when the underlying signal looks least like anything the labeling process was built to describe.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-anomaly-blind-spot.ipynb)



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

As a part of testing the proposition, a set of demonstrations which target different modalities were performed. The phase portrait, sonification, and EDF clients below are test/development scripts rather than standalone reproducible artifacts: each calls the Model Activations API over a live backend session (Section 7.1.1) and requires the reader to run that backend themselves and supply the resulting URL. This is a known limitation of the current demonstration code.

### Main setup

#### Model Activations API

As a part of testing the client modalities of a typical activation data, an API is created to generate activation vectors and activation sequences for clients. The backend loads GPT-2 Large (`d_model` 1280) and exposes activations from layer 8's residual stream.

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

This figure plots GPT-2 Large's layer-8 residual-stream activations, projected to two dimensions, as the model reads each prompt token by token.

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

The BCI modality explores whether model activations can be represented using formats familiar to neuroscience tools. Instead of converting activations into text labels, AST converts a compressed activation vector into a multi-channel signal that can be visualised and analysed using existing brain-signal software. The goal is not to claim that model activations are brain activity, but to investigate whether signal-analysis techniques developed for neuroscience can provide another way for humans to inspect model state. As with the other modalities in this section, activations come from GPT-2 Large's layer-8 residual stream (`d_model` 1280), bucketed to 32 channels for display.

MNE-Python is a widely used neuroscience visualisation tool [@mne-python]. In AST it is used as a display layer that converts activation data into familiar signal plots, heatmaps, and channel views. The resulting visualisations provide another way to inspect model state without first translating activations into language.

The following example was generated from the prompt "France".

```{figure} ast_edf_mne.png
:alt: MNE generated EDF view
:align: left
:width: 600px

Figure: EDF view.
```

The EDF view presents the same activation data through several complementary visualisations. The channel traces show how activation strength varies across channels. The heatmap reveals which activation buckets are most active. Spectral views provide a summary of how activity is distributed across different signal components. Together, these views present the activation state as a visual pattern rather than a collection of text labels.

[View the Source Code](https://raw.githubusercontent.com/virajsharma2000/scipy-26-paper/refs/heads/main/scipy-2026-paper-edf-view.ipynb)



## Software

This work relies on the following open-source software: TransformerLens [@transformerlens], SAE-Lens [@saelens], scikit-learn [@scikit-learn], MNE-Python [@mne-python], pyEDFlib [@pyedflib], NumPy [@numpy], and Matplotlib [@matplotlib]. The sparse autoencoder used throughout this paper, `gpt2-small-res-jb`, is due to [@bloom2024gpt2sae].



## Conclusion

The mechanistic interpretability literature has made substantial progress in understanding transformer model internals. Much of that progress has been built on a paradigm that converts activations into text. This paper argues that this paradigm may have a structural limitation: it can mostly only reveal what our language is able to describe. Model internals that do not map cleanly onto human language may be difficult, or impossible, for text-mediated interpretability to capture.

This is not an argument against SAEs, probing classifiers, or attribution methods. It is an argument that the field currently leans on a single channel - text - and that relying on a single channel could mean single-point-of-failure oversight. The same representational limits that make a model's deceptive behavior hard to describe in text might, in principle, still be noticeable through a trained sensory channel.

If a model state can generate a sensory signal that a trained human operator notices before a text-based probe names it, that could be a meaningful safety gain - regardless of whether the operator can put into words what they perceived. The doctor does not need to dictate the MRI to act on what they see.
