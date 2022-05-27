:author: Seyed Alireza Vaezi
:email: SeyedAlireza.Vaezi@uga.edu
:institution: University of Georgia

:author: Gianni Orlando
:email: Gianni.Orlando22@UGA.EDU
:institution: University of Georgia

:author: Mojtaba Fazli
:email: mfazli@meei.harvard.edu
:institution: harvard University

:author: Gary Ward
:email: Gary.Ward@uvm.edu
:institution: University of Vermont

:author: Silvia Moreno
:email: smoreno@uga.edu
:institution: University of Georgia

:author: Shannon Quinn
:email: SPQ@uga.edu
:institution: University of Georgia

:bibliography: paper

----------------------------------------------------------------------------------------------------------------------
A Novel Pipeline for Cell Instance Segmentation, Tracking and Motility Classification of Toxoplasma Gondii in 3D Space
----------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   Toxoplasma gondii is the parasitic protozoan that causes disseminated toxoplasmosis, a disease that is estimated to infect around one-third of the world's population. While the disease is commonly asymptomatic, the success of the parasite is in large part due to its ability to easily spread through nucleated cells. The virulence of T. gondii is predicated on the parasite's motility. Thus the inspection of motility patterns during its lytic cycle has become a topic of keen interest. Current cell tracking projects usually focus on cell images captured in 2D which are not a true representative of the actual motion of a cell. Current 3D tracking projects lack a comprehensive pipeline covering all phases of preprocessing, cell detection, cell instance segmentation, tracking, and motion classification, and merely implement a subset of the phases. Moreover, current 3D segmentation and tracking pipelines are not targeted for users with less experience in deep learning packages. Our pipeline, on the other hand, is developed for segmenting, tracking, and classification of motility phenotypes of T. gondii in 3D microscopic images. Although the pipeline is built initially focusing on T. gondii, it provides generic functions to allow users with similar but distinct applications to use it off-the-shelf. Interacting with all of our pipeline's modules is possible through our Napari plugin which is developed mainly off the familiar SciPy scientific stack. Additionally, our plugin is designed with a user-friendly GUI in Napari which adds several benefits to each step of the pipeline such as visualization and representation in 3D.  Our pipeline proves to be able to cover a broad spectrum of applications with considerable results.

.. class:: keywords
   Toxoplasma, Segmentation, Napari

Introduction
------------
Quantitative cell research often requires the measurement of different cell properties including the size, shape, and motility. This step is facilitated using the segmentation of imaged cells. Using fluorescent markers, computational tools can be used to complete segmentation and identify cell features and positions over time. 2D measurements of cells can be useful, but the more difficult task of deriving 3D information from cell images is vital for metrics such as motility and volumetric qualities. Toxoplasmosis is an infection caused by the intracellular parasite Toxoplasma gondii. T. gondii is one of the most successful parasites and at least one-third of the world's population is infected with the parasite. Although Toxoplasmosis is generally benign in healthy individuals, the infection has fatal implications in fetuses and immunocompromised individuals :cite:`saadatnia2012review` . T. gondii's virulence is directly linked to its lytic cycle which is comprised of invasion, replication, egress, and motility. Studying the motility of T. gondii is crucial in understanding its lytic cycle to develop potential treatments. 

For this reason, we present a novel pipeline to detect, segment, track, and classify the motility pattern of T. gondii in 3D space. Our pipeline is easy to use for users with limited knowledge in programming, Machine Learning, or Computer Vision technicalities. We also aimed to make it applicable in a broader scope of applications right off the shelf. 

Most of the state-of-the-art pipelines are restricted to 2D space which is not a true representative of the actual motion of the organism. Many require knowledge and expertise in programming or Machine Learning and Deep Learning models and frameworks, limiting the demographic of users that can use them. All of the pipelines solely include a subset of the aforementioned modules (i.e. detection, segmentation, tracking, and classification). And many of them rely on the user to train their own model hand-tailored for their specific application. This demands high levels of experience and skill in ML/DL and consequently undermines the possibility and feasibility of quickly utilizing an off-the-shelf pipeline and still getting good results.

To address these we present our pipeline that segments T. gondii cells in 3D microscopic images, tracks their trajectories, and classifies the motion patterns observed throughout the 3D frames. Our pipeline is comprised of four modules: pre-processing, segmentation, tracking, and classification. We developed our pipeline as a plugin for Napari - an open-source fast and interactive image viewer for Python designed for browsing, annotating, and analyzing large multi-dimensional images. Having our pipeline implemented as a part of Napari not only provides a user-friendly design but also gives more advanced users the possibility to attach and execute their custom code and even interact with the steps of the pipeline if needed. The preprocessing module is equipped with basic and extra filters and functionalities to aid in the preparation of the input data. Our pipeline gives its users the advantage of utilizing the functionalities that PlantSeg, VollSeg, and CellPose provide. These functionalities can be chosen in the pre-processing, detection, and segmentation steps. This brings forth a huge variety of algorithms and pre-build models to select from, making our pipeline a great fit for a greater scope of applications other than Toxoplasma gondii.

PlantSeg uses a variant of 3D U-Net, called Residual 3D U-Net, for preprocessing and segmentation of multiple cell types :cite:`plantseg`. PlantSeg performs best among Deep Learning algorithms for 3D Instance Segmentation and is very robust against image noise :cite:`Kar2021.06.09.447748`. The segmentation module also includes the optional use of CellPose :cite:`stringer2021cellpose`. CellPose is a generalized segmentation algorithm trained on a wide range of cell types and is the first step toward increased optionality in our pipeline. The Cell Tracking module consolidates the cell particles across the z-axis to materialize cells in 3D space and estimates centroids for each cell. The tracking module is also responsible for extracting the trajectories of cells based on the movements of centroids throughout consecutive video frames, which is eventually the input of the motion classifier module.


The rest of this paper is structured as follows: After briefly reviewing the literature in Related Work, we move on to thoroughly describe the details of our work in the Method section. Following that, the Results section depicts the results of comprehensive tests of our plugin on T. gondii cells.



Related Work
------------

The recent solutions in generalized and automated segmentation tools are focused on 2D cell images. Segmentation of cellular structures in 2D is important but not representative of realistic environments. Microbiological organisms are free to move on the Z-axis and tracking without that factor is a disservice to the research integrity. The focus on 2D research is understandable due to several factors. 3D data is difficult to capture as tools for capturing 3D slices and the computational requirements for analyzing this data are not available in most research labs. Most segmentation tools are unable to track objects in 3D space as the assignment of related centroids is more difficult. The additional noise from capture and focus increases the probability of incorrect assignment. 3D data also has issues with overlapping features and increased computation required per frame of time.

Fazli et al. :cite:`fazli2018unsupervised` studies the motility patterns of T. gondii and provides a computational pipeline for identifying motility phenotypes of T. gondii in an unsupervised, data-driven way. In that work Ca2+ is added to T. gondii cells inside a Fetal Bovine Serum. T. gondii cells react to Ca2+ and become motile and fluorescent. The images of motile T. gondii cells were captured using an LSM 710 confocal microscope. They use Python 3 and associated scientific computing libraries (NummPy, SciPy, scikit-learn, matplotlib) in their pipeline to track and cluster the trajectories of T. gondii. Based on this work Fazli et al. :cite:`fazli2018toward` work on another pipeline consisting of preprocessing, sparsification, cell detection, and cell tracking modules to track T. gondii in 3D video microscopy where each frame of the video consists of image slices taken 1 micro-meters of focal depth apart along the z-axis direction. In their last work Fazli et al. :cite:`fazli2019lightweight` developed a lightweight and scalable pipeline using task distribution and parallelism. Their pipeline consists of multiple modules: reprocessing, sparsification, cell detection, cell tracking, trajectories extraction, parametrization of the trajectories, and clustering. They could classify three distinct motion patterns in T. gondii using the same data from their previous work. 

While combining open source tools is not a novel architecture, little has been done to integrate 3D cell tracking tools. Fazeli et al. :cite:`fazeli2020automated` motivated by the same interest in providing better tools to non-software professionals created a 2D cell tracking pipeline. This pipeline combines Stardist and TrackMate for automated cell tracking. This pipeline begins with the user loading cell images and centroid approximations to the ZeroCostDL4Mic platform. ZeroCostDL4Mic is a deep learning training tool for those with no coding expertise. Once the platform is trained and masks for the training set are made for hand-drawn annotations, the training set can be input to Stardist. Stardist performs automated object detection using Euclidean distance to probabilistically determine cell pixels versus background pixels. Lastly, Trackmate uses segmentation images to track labels between timeframes and display analytics. 

This Stardist pipeline is similar in concept to our pipeline. Both create an automated segmentation and tracking pipeline but our pipeline is oriented to 3D data. Cells move in 3-dimensional space that is not represented in a flat plane. Our pipeline also does not require the manual training necessary for the other pipeline. Individuals with low technical expertise should not be expected to create masks for training or even understand the training of deep neural networks. Lastly, this pipeline does not account for imperfect datasets without the need for preprocessing. All implemented algorithms in our pipeline account for microscopy images with some amount of noise.  

Wen et al. :cite:`Wen2021-bn` combines multiple existing new technologies including deep learning and presents 3DeeCellTracker. 3DeeCellTracker segments and tracks cells on 3D time-lapse images. Using a small subset of their dataset they train the deep learning architecture, 3D U-Net, for segmentation. For tracking, a combination of two strategies was used to increase accuracy; local cell region strategies, and, spatial pattern strategy. Kapoor et al. :cite:`kapoor2021cell` presents VollSeg that uses deep learning methods to segment, track, and track analysis of cells in 3D with irregular shape and intensity distribution. It is a Jupyter Notebook-based python package and also has a UI in Napari. For tracking, a custom tracking code is developed based on Trackmate.

Many segmentation tools require some amount of knowledge in Machine or Deep Learning concepts. Training the neural network in creating masks is a common step for open-source segmentation tools. Automating this process makes the pipeline more accessible to microbiology researchers. 


Conclusion and Discussion
-------------------------

Future work on our pipeline will include the expanding of implemented algorithms and completely automated setup scripting. In order to currently run the pipeline, conda must be installed with Plantseg and cellpose environments installed if needed. Ideally, a script should be created to spawn these resources without any additional actions from the end user. Our pipeline aims to be easy-to-use by any party of microscopy researcher thus templates for scripting new segmentation algorithms into the pipeline would make the tool easier to customize. Even segmentation tools that perform suboptimally in comparison to cellpose and Plantseg should be "plug-in" options for research use. Stardust and computationally non-intensive segmentation would create a flexible tool for researchers without access to GPU machines. 


References
----------