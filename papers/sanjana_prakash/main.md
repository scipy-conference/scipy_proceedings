---
title: A Python Framework for Molecular Dynamics-Informed Machine Learning in Enzyme Engineering
---

# Abstract

Molecular dynamics simulations provide rich structural and dynamical information relevant to protein function, but integrating simulation-derived descriptors into machine learning workflows remains challenging. Existing protein machine learning pipelines often rely primarily on sequence representations or static structural features and do not provide a standardized framework for incorporating mechanistically meaningful dynamical information.

 We present a Python framework for molecular dynamics-informed machine learning in enzyme engineering. The framework supports automated construction of enzyme registries, structure retrieval and management, extraction of structural and dynamical descriptors, generation of catalytic-site features, and creation of machine-learning-ready datasets. By combining sequence embeddings, structural geometry, molecular dynamics-derived features, and mechanistically motivated catalytic descriptors, the framework enables multimodal modeling of enzyme function. We demonstrate the workflow using PET-degrading enzymes as a case study and show how simulation-derived features can be integrated into predictive enzyme engineering pipelines.

# Introduction

Machine learning has become an increasingly important tool in protein engineering and enzyme design. Recent advances in protein language models and structure prediction methods have significantly improved the ability to generate informative sequence and structural representations. However, many machine learning workflows remain limited to static descriptions of proteins and do not incorporate information describing conformational dynamics and catalytic behavior.

Molecular dynamics simulations provide atomistic information regarding structural flexibility, active-site stability, residue interactions, and conformational fluctuations. These properties are often directly relevant to enzyme function but remain underutilized in machine learning pipelines due to the complexity of extracting, standardizing, and integrating simulation-derived features.

This work presents a Python framework designed to bridge molecular simulation and machine learning for enzyme engineering. The framework enables researchers to transform sequence, structural, and simulation data into machine-learning-ready representations while maintaining interpretability through biologically meaningful descriptors.

# System Architecture

The framework consists of five stages:

1. Registry Construction
2. Structure Management
3. Structural Feature Extraction
4. Molecular Dynamics Feature Extraction
5. Machine Learning Dataset Generation

## Registry Construction

The workflow begins with a curated registry of enzymes and variants collected from literature, databases, or experimental studies.

Each registry entry contains:

* Enzyme identifier
* Scaffold
* Mutation information
* Structural metadata
* Experimental measurements

## Structure Management

Structures may be obtained from experimental PDB entries or generated using modern structure prediction methods.

The framework provides utilities for:

* Structure retrieval
* Metadata management
* Local structure repositories
* Structure standardization

# Structural Feature Extraction

Static structural descriptors are extracted from protein structures.

Examples include:

* Catalytic triad geometry
* Oxyanion-hole geometry
* Pocket volume
* Pocket depth
* Solvent accessible surface area
* Secondary structure composition

These descriptors provide interpretable representations of active-site architecture.

# Molecular Dynamics Feature Extraction

The framework supports extraction of simulation-derived descriptors from molecular dynamics trajectories.

Examples include:

* RMSD
* RMSF
* Hydrogen-bond occupancy
* Radius of gyration
* Residue-residue distance distributions
* Active-site flexibility metrics

These descriptors capture conformational properties not available from static structures alone.

# Catalytic Feature Engineering

A central contribution of the framework is the generation of mechanistically motivated catalytic descriptors.

Examples include:

* Oxyanion-hole stability
* Catalytic triad stability
* Active-site competence metrics
* Transition-state stabilization proxies

These features combine structural and dynamical information into biologically interpretable representations suitable for machine learning.

# Machine Learning Integration

The framework produces machine-learning-ready datasets containing:

* Sequence representations
* Structural descriptors
* Dynamical descriptors
* Catalytic descriptors
* Experimental labels

The resulting feature matrices may be used with a variety of machine learning models including tree-based methods, neural networks, and multimodal architectures.

# PETase Case Study

To demonstrate the framework, we assembled a curated registry of PET hydrolases and engineered PETase variants reported in the literature.

For each enzyme, the workflow generated:

* Structural descriptors
* Dynamic descriptors
* Catalytic-site features

These features were combined into a unified multimodal representation suitable for downstream predictive modeling and variant prioritization.

# Discussion

Simulation-derived descriptors provide complementary information to sequence and static structural representations. By exposing mechanistically meaningful features through a reproducible software framework, the approach enables interpretable machine learning workflows for enzyme engineering.

# Conclusion

This work introduces a Python framework for molecular dynamics-informed machine learning in enzyme engineering. The framework standardizes the generation of structural, dynamical, and catalytic descriptors and provides a foundation for multimodal predictive modeling and rational enzyme design.
