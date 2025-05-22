---
# Ensure that this title is the same as the one in `myst.yml`
title: An Active Learning Plugin In Napari To Fine Tune Models For Large-scale Bioimage Analysis
abstract: |
  The “napari-activelearning” plugin provides a framework to fine tune deep learning models for large-scale bioimage analysis, such as digital pathology Whole Slide Images (WSI). This plugin was developed with the motivation of easing the integration of deep learning tools into bioimage analysis workflows. This plugin implements the concept of Active Learning for reducing the time spent on labeling samples when fine tuning models. Because this plugin is integrated into Napari and leverages the use of Next Generation File Formats (Zarr), it is suitable for fine tuning deep learning models on large-scale images with little image preparation.
---

## Introduction

Adoption of deep learning methods for bioimage analysis has grown exponentially in recent years. Part of such success is thanks to transfer learning that enables using models that were trained on large volumes of data from diverse domains, such as the ImageNet and SA-1B datasets, into tasks where annotated data is scarce.
An example is the cell segmentation task in biological microscopy images, which requires human annotation of cell structures in images of different modalities and scales.
Such images tend to be tens and even hundreds of thousands of pixels per side, according to the acquisition magnification and imaging modality.
During the last years, the research community has made considerable efforts to curate databases for de-novo training of models relevant to the bioimage analysis field.
These databases include the LIVECell, TissueNet, and CellSeg, among several others.

Multiple deep learning models have been trained using these datasets to learn the cell segmentation task. The most relevant include Cellpose, Stardist, and Micro-SAM, which are used for a wide variety of applications.
These methods offer pre-trained models for carry out cell segmentation and also means to fine-tune these to new data using user's annotations.
However, each of the existing fine tuning procedures requires its own structure for training data organization.
Being the standard for training deep learning models to extract multiple image tiles from the original image files, and store them along with their respective annotations.
Some clear drawbacks of this practice are difficulties related to sharing training data between researchers for reproducibility purposes, in addition to usage of storage space for duplicated data already present in the original images.

Recently, the bioimage analysis community has started moving towards FAIR standards for data management, which combined with NGFFs (i.e. Zarr format) allows for efficient image data storage and sharing.
This approach has been leveraged by multiple projects, such as OME with OMERO for image data management, and Napari for n-dimensional data visualization, such as 2D and 3D imaging modalities and even time-lapse data.
In this context, image data stored under FAIR standards as NGFFs could be used for training deep learning models without needing to duplicate data while maintaining reproducibility of the training process.

In this work, the “napari-activelearning” plugin for Napari is introduced as a tool for easing fine tuning existing models on large-scale images.

## Methods

### NGFFs
Next Generation File Formats, such as Zarr, have been increasingly adopted by the bioimage analysis community. Zarr format stores large-scale image data as independent n-dimensional tiles, also called chunks, either on local disk or cloud storage. By using chunks as units of storage the amount of data required to be loaded into memory when accessing specific regions of the image is reduced. This is useful when applying a model for inference in larger-than-memory image data.

Mention ZarrDataset ...

### Napari ecosystem
Napari is a user-friendly visualizer for n-dimensional data which capabilities are extensible through plugins. This visualizer already offers tools for data annotation and is compatible with Next Generation File Formats such as Zarr.

### Active Learning framework
On the other hand, to reduce the amount of data presented to the human annotator, concepts from the Active Learning framework are used. This field studies methods for human-in-the-loop learning workflows that prevent overwhelming the annotator. This is achieved through computation of Acquisition Functions that assist the selection of samples predicted with low confidence, and when annotated by a human, these could improve the model’s performance after fine-tuning.


## Results

### Fine tuning Cellpose with napari-activelearning

Link to Documentation ...

A brief overview of the “napari-activelearning” plugin’s graphical interface shows the tool integrated in Napari and general usage of the plugin controls.

## Conclusion

While this plugin was developed with the goal of easing adoption of deep learning models in bioimage analysis projects, it is not restricted to these imaging modalities. Moreover, it can be applied as a transfer learning tool for methods that lack from an existing user interface or that are not adapted to work with large-scale image data.