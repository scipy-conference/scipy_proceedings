:author: Seyed Alireza Vaezi
:email: SeyedAlireza.Vaezi@uga.edu
:institution: University of Georgia

:author: Gianni Orlando
:email: mark37@rome.it
:institution: َUniversity of Georgia

:author: Shannon Quinn
:email: SPQ@uga.edu
:institution: University of Georgia

:bibliography: mybib


.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

----------------------------------------------------------------------------------------------------------------------
A Novel Pipeline for Cell Instance Segmentation, Tracking and Motility Classification of Toxoplasma Gondii in 3D Space
----------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   Toxoplasma gondii is the parasitic protozoan that causes disseminated toxoplasmosis, a disease that is estimated to infect around one-third of the world’s population. While the disease is commonly asymptomatic, the success of the parasite is in large part due to its ability to easily spread through nucleated cells. The virulence of T. gondii is predicated on the parasite’s motility. Thus the inspection of motility patterns during its lytic cycle has become a topic of keen interest. Current cell tracking projects usually focus on cell images captured in 2D which are not a true representative of the actual motion of a cell. Current 3D tracking projects lack a comprehensive pipeline covering all phases of preprocessing, cell detection, cell instance segmentation, tracking, and motion classification, and merely implement a subset of the phases. Moreover, current 3D segmentation and tracking pipelines are not targeted for users with less experience in deep learning packages. Our pipeline, on the other hand, is developed for segmenting, tracking, and classification of motility phenotypes of T. gondii in 3D microscopic images. Although the pipeline is built initially focusing on T. gondii, it provides generic functions to allow users with similar but distinct applications to use it off-the-shelf. Additionally, we aim to incorporate a user-friendly GUI in napari to add several benefits to each step of the pipeline such as visualization and representation in 3D. Interacting with all of our pipeline’s modules is possible through our napari plugin which is developed mainly off the familiar SciPy scientific stack.

.. class:: keywords
   Toxoplasma, Segmentation, Napari

Introduction
------------
.. Twelve hundred years ago  |---| in a galaxy just across the hill...
Quantitative cell research often requires the measurement of different cell properties including the size, shape and motility. This step is facilitated using the segmentation of imaged cells. Using fluorescent markers, computational tools can be used to complete segmentation and identify cell features and positions over time. 2D measurements of cells can be useful, but the more difficult task of deriving 3D information from cell images is vital for metrics such as motility and volumetric qualities. Toxoplasmosis is an infection caused by the intracellular parasite Toxoplasma gondii. T. gondii is one of the most successful parasites and at least one-third of the world’s population is infected with the parasite. Although Toxoplasmosis is generally benign in healthy individuals, the infection has fatal implications in fetuses and immunocompromised individuals [1]. T. gondii’s virulence is directly linked to its lytic cycle which is comprised of invasion, replication, egress, and motility. Studying the motility of T. gondii is crucial in understanding its lytic cycle to develop potential treatments. 

For this reason, we present a novel pipeline to detect, segment, track, and classify the motility pattern of T. gondii in 3D space. Our pipeline is easy to use for users with limited knowledge in programming, Machine Learning, or Computer Vision technicalities. We also aimed to make it applicable in a broader scope of applications right off the shelf. 

Most of the state-of-the-art pipelines are restricted to 2D space which is not a true representative of the actual motion of the organism. Many require knowledge and expertise in programming or Machine Learning and Deep Learning models and frameworks, limiting the demographic of users that can use them. All of the pipelines solely include a subset of the aforementioned modules (i.e. detection, segmentation, tracking, and classification). And many of them rely on the user to train their own model hand-tailored for their specific application. This demands high levels of experience and skill in ML/DL and consequently undermines the possibility and feasibility to quickly utilize an off-the-shelf pipeline and still get good results.

To address these we present [our pipeline] that segments T. gondii cells in 3D microscopic images, tracks their trajectories, and classifies the motion patterns observed throughout the video. Our pipeline is comprised of four modules: pre-processing, segmentation, tracking, and classification. We developed our pipeline as a plugin for Napari - a fast and interactive image viewer for Python designed for browsing, annotating, and analyzing large multi-dimensional images. Our user friendly design provides the condition for users to easily and effectively perform the tasks regardless of their skill level. We put a variety of detection and segmentation tools such as PlantSeg, CellPose and VollSeg in the detection and segmentation modules to choose from. Each of 


Related Work
------------
