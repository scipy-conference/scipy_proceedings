---
# Ensure that this title is the same as the one in `myst.yml`
title: Training a Supervised Cilia Segmentation Model from Self-Supervision
exports:
  - format: pdf
    template: arxiv_two_column
    output: exports/my-document.pdf

abstract: |
  Cilia are organelles found on the surface of some cells in the human body that sweep rhythmically to transport substances. Dysfunctional cilia are indicative of diseases that can disrupt organs such as the lungs and kidneys. Understanding cilia behavior is essential in diagnosing and treating such diseases. But, the tasks of automatically analysing cilia are often a labor and time-intensive since there is a lack of automated segmentation. In this work we overcome this bottleneck by developing a robust, self-supervised framework exploiting the visual similarity of normal and dysfunctional cilia. This framework generates pseudolabels from optical flow motion vectors, which serve as training data for a semi-supervised neural network. Our approach eliminates the need for manual annotations, enabling accurate and efficient segmentation of both motile and immotile cilia.
---

## Introduction

Cilia are hair-like membranes that extend out from the surface of the cells and are present on a variety of cell types such as lungs and brain ventricles and can be found in the majority of vertebrate cells. Categorized into motile and primary, motile cilia can help the cell to propel, move the flow of fluid, or fulfill sensory functions, while primary cilia act as signal receivers, translating extracellular signals into cellular responses [@doi:10.1007/978-94-007-5808-7_1]. Ciliopathies is the term commonly used to describe diseases caused by ciliary dysfunction. These disorders can result in serious issues such as blindness, neurodevelopmental defects, or obesity [@Hansen2021-fd]. Motile cilia beat in a coordinated manner with a specific frequency and pattern [@doi:10.1016/j.compfluid.2011.05.016]. Stationary, dyskinetic, or slow ciliary beating indicates ciliary defects. Ciliary beating is a fundamental biological process that is essential for the proper functioning of various organs, which makes understanding the ciliary phenotypes a crucial step towards understanding ciliopathies and the conditions stemming from it [@zain2022low].

Identifying and categorizing the motion of cilia is an essential step towards understanding ciliopathies. However, this is generally an expert-intensive process. Studies have proposed methods that automate the ciliary motion assessment [@zain2020towards]. These methods rely on large amounts of labeled data that are annotated manually which is a costly, time-consuming, and error-prone task. Consequently, a significant bottleneck to automating cilia analysis is a lack of automated segmentation. Segmentation has remained a bottleneck of the pipeline due to the poor performance of even state-of-the-art models on some datasets. These datasets tend to exhibit significant spatial artifacts (light diffraction, out-of-focus cells, etc.) which confuse traditional image segmentation models [@doi:10.48550/arXiv.1803.07534].

Video segmentation techniques tend to be more robust to such noise, but still struggle due to the wild inconsistencies in cilia behavior: while healthy cilia have regular and predictable movements, unhealthy cilia display a wide range of motion, including a lack of motion altogether [@doi:10.1002/ppul.24078]. This lack of motion especially confounds movement-based methods which otherwise have no way of discerning the cilia from other non-cilia parts of the video. Both image and video segmentation techniques tend to require expert-labeled ground truth segmentation masks. Image segmentation requires the masks in order to effectively train neural segmentation models to recognize cilia, rather than other spurious textures. Video segmentation, by contrast, requires these masks in order to properly recognize both healthy and diseased cilia as a single cilia category, especially when the cilia show no movement.

To address this challenge, we propose a two-stage image segmentation model designed to obviate the need for expert-drawn masks. We first build a corpus of segmentation masks based on optical flow (OF) thresholding over a subset of healthy training data with guaranteed motility. We then train a semi-supervised neural segmentation model to identify both motile and immotile data as a single segmentation category, using the flow-generated masks as “pseudolabels”. These pseudolabels operate as “ground truth” for the model while acknowledging the intrinsic uncertainty of the labels. The fact that motile and immotile cilia tend to be visually similar in snapshot allows us to generalize the domain of the model from motile cilia to all cilia. Combining these stages results in a semi-supervised framework that does not rely on any expert-drawn ground-truth segmentation masks, paving the way for full automation of a general cilia analysis pipeline.

The rest of this article is structured as follows: The Background section enumerates the studies relevant to our methodology, followed by a detailed description of our approach in the Methodology section. Finally, the next section delineates our experiment and provides a discussion of the results obtained.

## Background

Dysfunction in ciliary motion indicates diseases known as ciliopathies, which can disrupt the functionality of critical organs like the lungs and kidneys. Understanding ciliary motion is crucial for diagnosing and understanding these conditions. The development of diagnosis and treatment requires the measurement of different cell properties including size, shape, and motility [@vaezi2022novel].

Accurate analysis of ciliary motion is essential but challenging due to the limitations of manual analysis, which is labor-intensive, subjective, and prone to error. [@zain2020towards] proposed a modular generative pipeline that automates ciliary motion analysis by segmenting, representing, and modeling the dynamic behavior of cilia, thereby reducing the need for expert intervention and improving diagnostic consistency. [@quinn2015automated] developed a computational pipeline using dynamic texture analysis and machine learning to objectively and quantitatively assess ciliary motion, achieving over 90% classification accuracy in identifying abnormal ciliary motion associated with diseases like primary ciliary dyskinesia (PCD). Additionally, [@zain2022low] explored advanced feature extraction techniques like Zero-phase PCA Sphering (ZCA) and Sparse Autoencoders (SAE) to enhance cilia segmentation accuracy. These methods address challenges posed by noisy, partially occluded, and out-of-phase imagery, ultimately improving the overall performance of ciliary motion analysis pipelines. Collectively, these approaches aim to enhance diagnostic accuracy and efficiency, making ciliary motion analysis more accessible and reliable, thereby improving patient outcomes through early and accurate detection of ciliopathies. However, these studies rely on manually labeled data. The segmentation masks and ground-truth annotations, which are essential for training the models and validating their performance, are generated by expert reviewers. This dependence on manually labeled data is a significant limitation making automated cilia segmentation the bottleneck to automating cilia analysis.

In the biomedical field, where labeled data is often scarce and costly to obtain, several solutions have been proposed to augment and utilize available data effectively. These include semi-supervised learning [@YAKIMOVICH2021100383,@van2020survey], which utilizes both labeled and unlabeled data to enhance learning accuracy by leveraging the data's underlying distribution. Active learning [@settles2009active] focuses on selectively querying the most informative data points for expert labeling, optimizing the training process by using the most valuable examples. Data augmentation techniques [@10.3389/fcvm.2020.00105], [@Krois2021], [@10.1148/ryai.2020190195], [@Sandfort2019], [@YAKIMOVICH2021100383], [@van2001art], [@krizhevsky2012imagenet], [@ronneberger2015u], such as image transformations and synthetic data generation through Generative Adversarial Networks [@goodfellow2014generative], [@yi2019generative], increase the diversity and volume of training data, enhancing model robustness and reducing overfitting. Transfer learning [@YAKIMOVICH2021100383], [@Sanford2020-yg], [@NEURIPS2019_eb1e7832], [@hutchinson2017overcoming] transfers knowledge from one task to another, minimizing the need for extensive labeled data in new tasks. Self-supervised learning [@kim2019self], [@kolesnikov2019revisiting], [@mahendran2019cross] creates its labels by defining a pretext task, like predicting the position of a randomly cropped image patch, aiding in the learning of useful data representations. Additionally, few-shot, one-shot, and zero-shot learning techniques [@li2006one], [@miller2000learning] are designed to operate with minimal or no labeled examples, relying on generalization capabilities or metadata for making predictions about unseen classes.

A promising approach to overcome the dependency on manually labeled data is the use of unsupervised methods to generate ground truth masks. Unsupervised methods do not require prior knowledge of the data [@khatibi2021proposing]. Using domain-specific cues unsupervised learning techniques can automatically discover patterns and structures in the data without the need for labeled examples, potentially simplifying the process of generating accurate segmentation masks for cilia. Inspired by advances in unsupervised methods for image segmentation, in this work, we firstly compute the motion vectors using optical flow of the ciliary regions and then apply autoregressive modelling to capture their temporal dynamics. Autoregressive modelling is advantageous since the labels are features themselves. By analyzing the OF vectors, we can identify the characteristic motion of cilia, which allows us to generate pseudolabels as ground truth segmentation masks. These pseudolabels are then used to train a robust semi-supervised neural network, enabling accurate and automated segmentation of both motile and immotile cilia.

## Methodology

Dynamic textures, such as sea waves, smoke, and foliage, are sequences of images of moving scenes that exhibit certain stationarity properties in time [@doretto2003dynamic]. Similarly, ciliary motion can be considered as dynamic textures for their orderly rhythmic beating. Taking advantage of this temporal regularity in ciliary motion, OF can be used to compute the flow vectors of each pixel of high-speed videos of cilia. In conjunction with OF, autoregressive (AR) parameterization of the OF property of the video yields a manifold that quantifies the characteristic motion in the cilia. The low dimension of this manifold contains the majority of variations within the data, which can then be used to segment the motile ciliary regions.

### Optical Flow Properties

Taking advantage of this temporal regularity in ciliary motion, we use OF to capture the motion vectors of ciliary regions in high-speed videos. OF provides the horizontal $(u)$ and vertical $(v)$ components of the motion for each pixel. From these motion vectors, several components can be derived such as the magnitude, direction, divergence, and importantly, the curl (rotation). The curl, in this context, represents the rotational motion of the cilia, which is indicative of their rhythmic beating patterns. We extract flow vectors of the video recording of cilia, under the assumption that pixel intensity remains constant throughout the video.

```{math}
:label: of_formula
I(x,y,t)=I(x+u\delta t,y+v\delta t,t+\delta t)
```

Where $I(x,y,t)$ is the pixel intensity at position $(x,y)$ a time $t$. Here, $(u\delta t, v\delta t)$ are small changes in the next frame taken after $\delta t$ time, and $(u,v)$, respectively, are the OF components that represent the displacement in pixel positions between consecutive frames in the horizontal and vertical directions at pixel location $(x, y)$.

:::{figure} sample_vids_with_gt_mask.png
:label: fig:sample_vids_with_gt_mask
A sample of three videos in our cilia dataset with their manually annotated ground truth masks.
:::
<!-- :::{figure} ground_truth.png
:label: fig:ground_truth
Manually labeled ground truth
::: -->
:::{figure} sample_OF.png
:label: fig:sample_OF
Representation of rotation (curl) component of OF at a random time
:::

### Autoregressive Modeling

@fig:sample_OF shows a sample of the OF component at a random time. From OF vectors, elemental components such as rotation are derived, which highlights the ciliary motion by capturing twisting and turning movements. To model the temporal evolution of these motion vectors, we employ an autoregressive (AR) model [@doi:10.5244/C.21.76]. This model captures the dynamics of the flow vectors over time, allowing us to understand how the motion evolves frame by frame. The AR model helps in decomposing the motion into a low-dimensional subspace, which simplifies the complex ciliary motion into more manageable analyses.

```{math}
:label: AR
y_t =C\vec{x_t} + \vec{u} 
```

```{math}
:label: AR_state
\vec{x}_t = A_1\vec{x}_{t-1} + A_2\vec{x}_{t-2} + ... + A_d\vec{x}_{t-d} + \vec{v_t}
```

In equation {ref}`AR`, $\vec{y}_t$ represents the appearance of cilia at time $t$ influenced by noise $\vec{u}$. Equation {ref}`AR_state` represents the state $\vec{x}$ of the ciliary motion in a low-dimensional subspace defined by an orthogonal basis $C$ at time $t$, plus a noise term $\vec{v}_t$
and how the state changes from $t$ to $t + 1$.

Equation {ref}`AR_state` is a decomposition of each frame of a ciliary motion video $\vec{y}_t$ into a low-dimensional state vector $\vec{x}_t$
using an orthogonal basis $C$. This equation at position ${x}_t$ is a function of the sum of $d$ of its previous positions $\vec{x}_{t−1}, \vec{x}_{t−2}, \vec{x}_{t−d}$, each multiplied by its corresponding coefficients $A = {A_1, A_2, ..., A_d}$. The noise terms $\vec{u}$ and $\vec{v}$ are used to represent the residual difference between the observed data and the solutions to the linear equations. The variance in the data is predominantly captured by a few dimensions of $C$, simplifying the complex motion into manageable analyses.

Each order of the autoregressive model roughly aligns with different frequencies within the data, therefore, in our experiments, we chose $d = 5$ as the order of our autoregressive model. This choice allows us to capture a broader temporal context, providing a more comprehensive understanding of the system's dynamics. We then created raw masks from this lower-dimensional subspace, and further enhanced them with adaptive thresholding to remove the remaining noise.

:::{figure} AR matrices.png
:label: fig:sample_AR
The pixel representation of the 5-order AR model of the OF component of a sample video. The $x$ and $y$ axes correspond to the width and height of the video.
:::
In @fig:sample_AR, the first-order AR parameter is showing the most variance in the video, which corresponds to the frequency of motion that cilia exhibit. The remaining orders have correspondence with other different frequencies in the data caused by, for instance, camera shaking. Evidently, simply thresholding the first-order AR parameter is adequate to produce an accurate mask, however, in order to get a more refined result we subtracted the second order from the first one, followed by a Min-Max normalization of pixel intensities and scaling to an 8-bit unsigned integer range. We used adaptive thresholding to extract the mask on all videos of our dataset. The generated masks exhibited under-segmentation in the ciliary region, and sparse over-segmentation in other regions of the image. To overcome this, we adapted a Gaussian blur filter followed by an Otsu thresholding to restore the under-segmentation and remove the sparse over-segmentation. @fig:thresholding illustrates the steps of the process.

:::{figure} thresholding.png
:label: fig:thresholding
The process of computing the masks. a) Subtracting the second-order AR parameter from the first-order, followed by b) Adaptive thresholding, which suffers from under/over-segmentation. c) A Gaussian blur filter, followed by d) An Otsu thresholding eliminates the under/over-segmentation.
:::

### Training the model

Our dataset includes 512 videos, with 437 videos of dyskinetic cilia and 75 videos of healthy motile cilia, referred to as the control group. The control group is split into %85 and %15 for training and validation respectively. 108 videos in the dyskinetic group are manually annotated which are used in the testing step. @fig:sample_vids_with_gt_mask shows annotated samples of our dataset.

In our study, we employed a Feature Pyramid Network (FPN) [@kirillov2017unified] architecture with a ResNet-34 encoder. The model was configured to handle grayscale images with a single input channel and produce binary segmentation masks. For the training input, one mask is generated per video using our methodology, and we use the first 250 frames from each video in the control group making a total of 18,750 input images. We utilized Binary Cross-Entropy Loss for training and the Adam optimizer with a learning rate of $10^-3$. To evaluate the model's performance, we calculated the Dice score during training and validation. Data augmentation techniques, including resizing, random cropping, and rotation, were applied to enhance the model's generalization capability. The implementation was done using a library [@Iakubovskii:2019] based on PyTorch Lightning to facilitate efficient training and evaluation. @tbl:model_specs contains a summary of the model parameters and specifications.

The next section discusses the results of the experiment and the performance of the model in detail.

:::{table} Summary of model architecture, training setup, and dataset distribution
:label: tbl:model_specs
| **Aspect**                      | **Details**                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **Architecture**                | FPN with ResNet-34 encoder                                                                                                               |
| **Input**                       | Grayscale images with a single input channel                                                                                             |
| **Number of Epochs**            | 20                                                                                                                                       |
| **Batch Size**                  | 4                                                                                                                                        |
| **Training Samples**            | 15,662                                                                                                                                   |
| **Validation Samples**          | 2,763                                                                                                                                    |
| **Test Samples**                | 108                                                                                                                                      |
| **Loss Function**               | Binary Cross-Entropy Loss                                                                                                                |
| **Optimizer**                   | Adam optimizer with a learning rate of $10^{-3}$                                                                                         |
| **Evaluation Metric**           | Dice score during training and validation                                                                                                |
| **Data Augmentation Techniques**| Resizing, random cropping, and rotation                                                                                                  |
| **Implementation**              | Using a Python library with Neural Networks for Image Segmentation based on PyTorch [@Iakubovskii:2019]                                  |

:::

## Results and Discussion

The model's performance metrics, including IoU, Dice score, sensitivity, and specificity, are summarized in @tbl:metrics. The validation phase achieved an IoU of 0.312 and a Dice score of 0.476, which indicates a moderate overlap between the predicted and ground truth masks. The high sensitivity (0.999) observed during validation suggests that the model is proficient in identifying ciliary regions, albeit with a specificity of 0.813, indicating some degree of false positives. In the testing phase, the IoU and Dice scores decreased to 0.230 and 0.374, respectively, reflecting the challenges posed by the dyskinetic cilia data, which were not included in the training or validation sets. Despite this, the model maintained a reasonable sensitivity of 0.631 and specificity of 0.787.

:::{figure} out_sample.png
:label: fig:out_sample
The model predictions on 5 dyskinetic cilia samples. The first column shows a frame of the video, the second column shows the manually labeled ground truth, the third column is the model's prediction, and the last column is a thresholded version of the prediction.
:::

@fig:out_sample provides visual examples of the model's predictions on dyskinetic cilia samples, alongside the manually labeled ground truth and thresholded predictions. The dyskinetic samples were not used in the training or validation phases. These predictions were generated after only 20 epochs of training with a small training data.  The visual comparison reveals that, while the model captures the general structure of ciliary regions, there are instances of under-segmentation and over-segmentation, which are more pronounced in the dyskinetic samples. This observation is consistent with the quantitative metrics, suggesting that further refinement of the pseudolabel generation process or model architecture could enhance segmentation accuracy.

:::{table} The performance of the model in validation and testing phases.
:label: tbl:metrics
| Phases     | Metrics       |             |            |            |
|------------|---------------|-------------|------------|------------|
|            | IoU over dataset | Dice Score  | Sensitivity| Specificity|
| Validation | 0.312         | 0.476       | 0.999      | 0.813      |
| Testing    | 0.230         | 0.374       | 0.631      | 0.787      |

:::

These results show the potential of our approach to reduce the reliance on manually labeled data for cilia segmentation. The use of this unsupervised learning framework allows the model to generalize from the motile cilia domain to the more variable dyskinetic cilia, although with some limitations in accuracy. Future work could focus on expanding the dataset and improving the process of generating pseudolabels to enhance the model's accuracy.

## Conclusions

In this paper, we introduced a self-supervised framework for cilia segmentation that eliminates the need for expert-labeled ground truth segmentation masks. Our approach takes advantage of the inherent visual similarities between healthy and unhealthy cilia to generate pseudolabels from optical flow-based motion segmentation of motile cilia. These pseudolabels are then used as ground truth for training a semi-supervised neural network capable of identifying regions containing dyskinetic cilia. Our results indicate that the self-supervised framework is a promising step towards automated cilia analysis. The model's ability to generalize from motile to dyskinetic cilia demonstrates its potential applicability in clinical settings. Although there are areas for improvement, such as enhancing segmentation accuracy and expanding the dataset, the framework sets the foundation for more efficient and reliable cilia analysis pipelines.
