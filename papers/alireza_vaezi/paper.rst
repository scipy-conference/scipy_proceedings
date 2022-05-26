:author: Seyed Alireza Vaezi
:email: SeyedAlireza.Vaezi@uga.edu
:institution: University of Georgia

:author: Gianni Orlando
:email: mark37@rome.it
:institution: University of Georgia

:author: Shannon Quinn
:email: SPQ@uga.edu
:institution: University of Georgia

:bibliography: paperbib

----------------------------------------------------------------------------------------------------------------------
A Novel Pipeline for Cell Instance Segmentation, Tracking and Motility Classification of Toxoplasma Gondii in 3D Space
----------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   Toxoplasma gondii is the parasitic protozoan that causes disseminated toxoplasmosis, a disease that is estimated to infect around one-third of the world's population. While the disease is commonly asymptomatic, the success of the parasite is in large part due to its ability to easily spread through nucleated cells. The virulence of T. gondii is predicated on the parasite's motility. Thus the inspection of motility patterns during its lytic cycle has become a topic of keen interest. Current cell tracking projects usually focus on cell images captured in 2D which are not a true representative of the actual motion of a cell. Current 3D tracking projects lack a comprehensive pipeline covering all phases of preprocessing, cell detection, cell instance segmentation, tracking, and motion classification, and merely implement a subset of the phases. Moreover, current 3D segmentation and tracking pipelines are not targeted for users with less experience in deep learning packages. Our pipeline, on the other hand, is developed for segmenting, tracking, and classification of motility phenotypes of T. gondii in 3D microscopic images. Although the pipeline is built initially focusing on T. gondii, it provides generic functions to allow users with similar but distinct applications to use it off-the-shelf. Additionally, we aim to incorporate a user-friendly GUI in napari to add several benefits to each step of the pipeline such as visualization and representation in 3D. Interacting with all of our pipeline's modules is possible through our napari plugin which is developed mainly off the familiar SciPy scientific stack.

.. class:: keywords
   Toxoplasma, Segmentation, Napari

Introduction
------------
Quantitative cell research often requires the measurement of different cell properties including the size, shape and motility. This step is facilitated using the segmentation of imaged cells. Using fluorescent markers, computational tools can be used to complete segmentation and identify cell features and positions over time. 2D measurements of cells can be useful, but the more difficult task of deriving 3D information from cell images is vital for metrics such as motility and volumetric qualities. Toxoplasmosis is an infection caused by the intracellular parasite Toxoplasma gondii. T. gondii is one of the most successful parasites and at least one-third of the world's population is infected with the parasite. Although Toxoplasmosis is generally benign in healthy individuals, the infection has fatal implications in fetuses and immunocompromised individuals :cite:`saadatnia2012review` . T. gondii's virulence is directly linked to its lytic cycle which is comprised of invasion, replication, egress, and motility. Studying the motility of T. gondii is crucial in understanding its lytic cycle to develop potential treatments. 

For this reason, we present a novel pipeline to detect, segment, track, and classify the motility pattern of T. gondii in 3D space. Our pipeline is easy to use for users with limited knowledge in programming, Machine Learning, or Computer Vision technicalities. We also aimed to make it applicable in a broader scope of applications right off the shelf. 

Most of the state-of-the-art pipelines are restricted to 2D space which is not a true representative of the actual motion of the organism. Many require knowledge and expertise in programming or Machine Learning and Deep Learning models and frameworks, limiting the demographic of users that can use them. All of the pipelines solely include a subset of the aforementioned modules (i.e. detection, segmentation, tracking, and classification). And many of them rely on the user to train their own model hand-tailored for their specific application. This demands high levels of experience and skill in ML/DL and consequently undermines the possibility and feasibility to quickly utilize an off-the-shelf pipeline and still get good results.

To address these we present our pipeline that segments T. gondii cells in 3D microscopic images, tracks their trajectories, and classifies the motion patterns observed throughout the 3D frames. Our pipeline is comprised of four modules: pre-processing, segmentation, tracking, and classification. We developed our pipeline as a plugin for Napari - an open-source fast and interactive image viewer for Python designed for browsing, annotating, and analyzing large multi-dimensional images. Having our pipeline implemented as a part of Napari not only provides a user-friendly design, but also gives more advanced users the possibility to attach and execute their custom code and even interact with the steps of the pipeline if needed. Our pipeline gives its users the advantage of utilizing the functionalities that PlantSeg, VollSeg, and CellPose provide. These functionalities can be chosen in the pre-processing, detection, and segmentation steps. This brings forth a huge variety of algorithms and pre-build models to select from, making our pipeline a great fit for a greater scope of application other than Toxoplasma gondii.


Related Work
------------

While combining open source tools is not a novel architecture, little has been done to integrate 3D cell tracking tools. Elnaz Fazeli et all :cite:`fazeli2020automated` motivated by the same interest in providing better tools to non-software professionals created a 2D cell tracking pipeline. This pipeline combines Stardist and TrackMate for automated cell tracking. This pipeline begins with the user loading cell images and centroid approximations to the ZeroCostDL4Mic platform. ZeroCostDL4Mic is a deep learning training tool for those with no coding expertise. Once the platform is trained and masks for the training set are made for hand drawn annotations, the training set can be input to Stardist. Stardist performs automated object detection using Euclidean distance to probabilistically determine cell pixels versus background pixels. Lastly, Trackmate uses the segmentation images to track labels between timeframes and display analytics. 

This Stardist pipeline is similar in concept to our pipeline. Both create an automated segmentation and tracking pipeline but our pipeline is oriented to 3D data. Cells move in 3 dimensional space that is not represented in a flat plane. Our pipeline also does not require the manual training necessary for the other pipeline. Individuals with low technical expertise should not be expected to create masks for training or even understand training of deep neural networks. Lastly, this pipeline does not account for imperfect datasets without need for preprocessing. All implemented algorithms in our pipeline account for microscopy images with some amount of noise.  

Conclusion and Discussion
-------------------------

Future work on our pipeline will include the expanding of implemented algorithms and completely automated setup scripting. In order to currently run the pipeline, conda must be installed with Plantseg and cellpose environments installed if needed. Ideally, a script should be created to spawn these resources without any additional actions from the end user. Our pipeline aims to be easy-to-use by any party of microscopy researcher thus templates for scripting new segmentation algorithms into the pipeline would make the tool easier to customize. Even segmentation tools that perform suboptimally in comparison to cellpose and Plantseg should be “plug-in” options for research use. Stardust and computationally non-intensive segmentation would create a flexible tool for researchers without access to GPU machines. 


References
----------