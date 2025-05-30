---
title: An Active Learning Plugin in napari to Fine-Tune Models for Large-scale Bioimage Analysis
abstract: |
  The “napari-activelearning” plugin provides a framework to fine-tune deep learning models for large-scale bioimage analysis, such as digital pathology Whole Slide Images.
  The development of this plugin was motivated by easing the integration of deep learning tools into bioimage analysis workflows.
  This plugin implements the concept of Active Learning for reducing the time spent on labeling samples when fine-tuning models.
  Because this plugin is integrated into napari and leverages the use of Next Generation File Formats (e.g. Zarr), it is suitable for fine-tuning deep learning models on large-scale images with little image preparation.
---

## Introduction

Adoption of deep learning methods for bioimage analysis has grown exponentially in recent years.
Part of such success is thanks to transfer learning that enables using models that were trained on large volumes of data from diverse domains, such as the ImageNet[@imagenet] and Segment Anything 1 Billion (SA-1B)[@kirillov2023segment] datasets, into tasks where annotated data is scarce.
An example is cell segmentation in biological microscopy images [@Greenwald2021-hj], which requires human annotation of cell structures in images of different modalities and scales.
Such images tend to be tens or hundreds of thousands of pixels per side, depending on the acquisition magnification and imaging modality.
Moreover, the research community has made considerable efforts during the last years to curate databases for training models relevant to the bioimage analysis field.
These databases include LIVECell[@Edlund2021-bi], TissueNet[@Greenwald2021-hj], and CellSeg[@Lee2022-ln], among many others.
Such databases can be used for de-novo training or be used for fine-tuning models that were trained with image datasets from general domains.

### Related work
Segmentation of structures in biological image data is a recurrent task in bioimage analysis that serves as an intermediate step for downstream applications.
The most relevant deep learning segmentation methods include Cellpose[@Stringer2021-od], Stardist[@weigert2022], and Micro-SAM[@Archit2025-wa].
These methods offer pre-trained models for carrying out segmentation of biological structures in multiple imaging modalities, and tools for fine-tuning those same models to new data with user-defined annotations.
Training deep learning models for bioimage analysis involves: 1) extracting several image tiles from the original image files, 2) annotating each of those tiles according to the tasks being learned, and 3) storing the pairs of tiles and annotations in separate containers or folders as training and testing datasets.
However, this approach involves some technical difficulties such as the costs and time associated with transferring training data between researchers for sharing and reproducibility purposes, costs for storing duplicated data from the image tiles already present in the original images, and lack of context and coordinates from where each tile was originally extracted. 

The field of Active Learning studies human-in-the-loop strategies in deep learning that can reduce the time and effort required for de-novo training or fine-tuning models.
Applications from active learning have also been developed to address computer vision tasks[@Gal2017DeepBA].
This is relevant to biological image analysis where data annotation is one of the most time-consuming tasks[@BUDD2021102062].
Due to the scale of the images, the number of samples that can be extracted for labeling can overwhelm the annotator.
An active learning sampling strategy, based on acquisition functions, can be used to prevent this by presenting a limited number of samples at a time.
Moreover, there are acquisition functions such as Bayesian Active Learning by Disagreement (BALD)[@Houlsby2011BayesianAL] that focus on obtaining only cases that could improve a model’s performance when correctly labeled.

The bioimage analysis community has been steering towards efficient creation and sharing of repositories containing large volumes of data.
NGFFs[@Moore2021-we] are one of the most preferred options for large-scale image storage and management, such as the OME-Zarr data format[@Moore2023-nq].
The Zarr data format establishes a standard structure for compressed, chunked-based, n-dimensional data storage that can be efficiently stored either in local or cloud repositories[@Moore2021-we,@Moore2023-nq].
Multiple projects have been created on top of Zarr, including Application Programming Interfaces (API) such as the zarr-python library[@zarrpython]. Some image processing softwares have adopted this data format, like Fiji[@Fiji] with the MobIE plugin[@MoBIE], and QuPath[@QuPath] with its own Zarr data loader. Visualization tools such as napari[@napari], neuroglancer[@neuroglancer], viv[@viv], and webKnossos[@webKnossos] have also added support to this data format.
Additionally, image data stored as Zarr can be used for training deep learning models without duplicating data.
That can be achieved by accessing chunked image data instead of extracting image tiles and storing them separately as is commonly needed in standard deep learning training.

In this work, the “napari-activelearning” plugin for napari is introduced as a tool for easing fine-tuning existing models on large-scale images.
This plugin leverages napari’s user-friendly features for annotating new samples following an active learning workflow.
Additionally, the “napari-activelearning” plugin uses NGFF to store annotations in a storage-efficient manner, ready for deep learning training.

## Methods
The “napari-activelearning” plugin relies on three components to provide a user-friendly framework to train and fine-tune deep learning models for bioimage analysis: 1) NGFF, for efficient storage of the data generated during the active learning workflow, 2) napari, for visualization and graphical interface with the user, and 3) Active Learning, for assisting users with selection of samples of interest that require annotation for improving the deep learning model performance.

### NGFFs

NGFFs [@Moore2021-we], such as OME-Zarr[@Moore2023-nq], have been increasingly adopted by the bioimage analysis community thanks to its computation and storage advantages. Zarr format stores large-scale image data as independent n-dimensional tiles, also called chunks, either on local disk or cloud storage. By using chunks as units of storage, the amount of data required to be loaded into memory when accessing specific regions of the image is reduced. Accessing image chunks is parallel-safe, which enables acceleration of image processing through parallel computing. This is useful when applying a model for inference in larger-than-memory image data, where deep learning inference can be applied to regions of the image separately, and applying a subsequent reduction operation on the results. This reduction operation could be an accumulation function for whole image classification, or a stitching algorithm for segmentation tasks.

### napari Visualization Tool

napari[@napari] is a user-friendly n-dimensional data viewer with extensible capabilities via plugins. This visualizer already offers tools for data annotation, and it is compatible with Zarr, enabling visualization of large-scale image data efficiently. napari has been used to develop deep learning applications for bioimaging analysis such as napari-cellpose[@Stringer2021-od], and Micro-SAM[@Archit2025-wa] plugins. These plugins can be used to segment biological structures in data acquired with multiple imaging modalities. However, these plugins are intended to be used with tiles of images that have already been extracted and stored, in contrast to the plugin presented in this work, which can be applied on regions defined by the user directly on original images without storing duplicate data from image tiles.

### Active Learning framework

To reduce the amount of data presented to the user for annotation, concepts from the Active Learning framework are implemented in this plugin. This field studies methods for human-in-the-loop learning workflows that avoid overwhelming the annotator with samples to labeling for training. This is achieved through a computation of Acquisition Functions that determine what samples require labeling to subsequently improve the performance of a specific model. In this plugin, the BALD[@Houlsby2011BayesianAL] acquisition function is implemented to score and sort a set of image patches sampled from an image. 
In this plugin, the score assigned to a sample extracted at random from the input image is computed following Houlsby et al.[@Houlsby2011BayesianAL]:
```{math}
a(x, M) = \mathbb{I}[y, \theta| x, D] ,
```
where $a(x, M)$ is the acquisition function score for sample $x$ for model $M$, the mutual information $\mathbb{I}$ is computed from the prediction $y$ made by model $M$ for sample $x$ in the dataset $D$ with the current state of parameters $\theta$. Following Gal et al.[@Gal2017DeepBA], the mutual information can be computed through Monte Carlo integration to compute the BALD function as follows:
```{math}
\mathbb{I}[y, \theta| x, D] = -\sum_{c}\left(\frac{1}{T} \sum_{t} \hat{p_c}^t\right) log \left(\frac{1}{T}\sum_{t}\hat{p_c}^t\right) + \frac{1}{T}\sum_{c,t}\hat{p_c}^t log~\hat{p_c}^t ,
```
where $T$ is the total number of steps in the Monte Carlo integration, $\hat{p_c}^t$ is the component of the prediction made by model $M$ for sample $x$ at class $c$. To apply the Monte Carlo integration, the prediction made with model $M$ needs to be converted into a gaussian process. In this plugin, this is done by introducing a Dropout operation after each activation layer found in the deep learning model $M$ [@Gal2015DropoutAA].

### Implementation details

The “napari-activelearning” plugin is an open-source project implemented in the Python programming language that relies on the napari plugin architecture to offer a user-friendly interface. This plugin consists of three main component widgets that allow the execution of an end-to-end fine-tuning process from within a napari window. Such components are 1) an Image groups manager widget, 2) an acquisition function configuration widget, and 3) a labels manager widget.

#### Image Groups Manager

This component is used to gather the metadata of the images used for training and fine-tuning into a single data structure called “image group”. The purpose of this structure, shown in @fig: images_mgr, is to define how each layer shown on the napari window will be used in the active learning workflow, such as input data, labels or annotations, and active masks from where the plugin can sample patches to be processed through the fine-tuning process.
:::{figure} image_groups_mgr.png
:label: fig:images_mgr
Image groups manager widget used to manage metadata of napari's layers to be used for inference and fine-tuning along with the Acquisition Function Manager component.
:::
#### Acquisition Function Manager

In this component, the model used for active learning can be selected from a list of registered models, and their hyper-parameters can be configured according to the user’s needs. This component uses NumPy[@numpy] and PyTorch[@pytorch] libraries to implement the BALD[@Houlsby2011BayesianAL] acquisition function and the Dropout operation insertion[@Gal2015DropoutAA]. The outputs generated in the intermediate steps of the active learning are generated with transforms from the Sci-kit Image library[@scikit-image] and stored following the OME-Zarr[@Moore2023-nq] specification to make them shareable and reproducible under FAIR[@Wilkinson2016-bv] guidelines.
This component computes the acquisition function score for a set of image patches sampled from the input image defined in the Image Groups Manager component. The number of samples extracted and the total steps for the Monte Carlo integration process can be defined by the user in this widget, as shown in @fig:acquisition_mgr. For each sampled patch, the inference and acquisition function are computed and presented to the user in a descending list on the Labels Manager widget.
The “napari-activelearning” uses Cellpose[@Stringer2021-od] as default deep learning framework for inference and fine-tuning with its pre-trained models. Moreover, the code that implements this component is intended for its extension to other deep learning models with PyTorch backend[@pytorch]. This can be achieved through the inheritance of a segmentation method class that is used to execute pure inference, probability computations, and even fine-tuning of model weights.
:::{figure} acquisition_fun_mgr.png
:label: fig:acquisition_mgr
Acquisition function manager widget used mainly to execute model inference and fine-tuning, along with configuration of model parameters and the active learning sampling parameters.
:::

#### Labels Manager

After the samples are extracted and their acquisition function scores and inferences are computed by the Acquisition Function Manager, these can be reviewed by the user. The labels predicted by the selected model can be corrected using built-in napari annotation tools. These corrected annotations can be used as new targets for fine-tuning the selected model. Additionally, this component provides a simple navigation system to move between inferred labels in space as presented in @fig:label_mgr. This system is ideal for reviewing the output generated by the selected model across the image, which can be time consuming in large-scale images, such as Whole Slide Images (WSI) or three-dimensional image data.
:::{figure} labels_groups_mgr.png
:label: fig:label_mgr
Label groups manager widget used for handling sampled patches for labels correction following an active learning workflow.
:::

## Results
The “napari-activelearning” source code is open source and available to the research community at the http://github.com/thejacksonlaboratory/activelearning repository. This package is distributed through PyPI (https://pypi.org/project/napari-activelearning) and can also be installed using the napari plugin manager.

### Fine-tuning a Cellpose model
As a proof-of-concept of the “napari-activelearning” plugin for fine-tuning an existing model to new data, a Cellpose model[@Stringer2021-od] was fine-tuned to carry out nuclei segmentation on the Cells 3D+2 Channels image from Sci-kit Image[@scikit-image]. This image is included as a sample image in napari[@napari].  The proof-of-concept can be found as a tutorial in the supporting documents section of this paper, or in the documentation website of this plugin at https://thejacksonlaboratory.github.io/activelearning/tutorials.html.

## Conclusion

In this work, the “napari-activelearning” plugin has been introduced for carrying out fine-tuning of deep learning models for large-scale bioimage data. The napari visualization tool was used to host the plugin and serve as a graphical user interface. By leveraging NGFF, the annotated data is stored efficiently by creating only image chunks that contain any labels information and avoiding writing empty chunks. Moreover, management of the data created through the active learning workflow is annotated following the OME-Zarr v2 specification making it ready for sharing and reproduction. Whereas this plugin was developed to facilitate adoption of deep learning models in bioimage analysis, it is not restricted to these imaging modalities and can be extended to be used with any image stored in the Zarr format. Finally, the plugin can be used for transfer learning or simply as an interface for deep learning methods that lack graphical user interfaces for inference.
