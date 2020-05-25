:author: Gaius Caesar
:email: jj@rome.it
:institution: Senate House, S.P.Q.R.
:institution: Egyptian Embassy, S.P.Q.R.

:author: Mark Anthony
:email: mark37@rome.it
:institution: Egyptian Embassy, S.P.Q.R.

:author: Jarrod Millman
:email: millman@rome.it
:institution: Egyptian Embassy, S.P.Q.R.
:institution: Yet another place, S.P.Q.R.

:author: Brutus
:email: brutus@rome.it
:institution: Unaffiliated
:bibliography: mybib


------------------------------------------------------------------------------------------------
Unsupervised Spatiotemporal Representation of Cilia Video Using A Modular Generative Pipeline
------------------------------------------------------------------------------------------------

.. class:: abstract

Motile cilia are organelles found on the surface of some cells in the human body that sweep 
rhythmically and simultaneously to transport substances. Dysfunction of ciliary motion is 
often indicative of diseases known as ciliopathies, which disrupt the functionality of 
macroscopic structures within the lungs, kidneys and other organs. Thus, phenotyping 
ciliary motion is an essential step towards diagnosing potentially harmful ciliopathies. 
We propose a modular generative pipeline for the analysis of cilia video data so that expert 
labor may be supplemented or replaced for this task. Our proposed model is divided into 
three notable modules: preprocessing, appearance, and dynamics. The preprocessing module 
augments initial video data with derivative quantities such as a dense optical flow field. 
The augmented data is fed frame-by-frame into a generative appearance model which segments 
the cilia in each frame, and consequently learns a compressed latent representation of the 
cilia. The frames are then embedded into the latent space and recorded as a path. The 
low-dimensional path is fed into the generative dynamics module. The low-dimensional 
representation allows the dynamics module to only focus on the motion of the sparse cilia, 
with otherwise useless information being removed during the compression in the appearance 
module. Since both the appearance and dynamics modules are generative, the pipeline itself 
serves as a generative model which creates a factored spatiotemporal latent space.

.. class:: keywords

Machine Learning, Data Science, Video Analysis, Generative Modeling, Variational Autoencoder, Modular, Pipeline

Introduction
------------

Motile cilia are organelles that line human bronchial and nasal passages. The cilia sweep 
synchronously to expel foreign matter and mucus. This ciliary motion acts as a defense 
mechanism and is vital for pulmonary health. Conversely, the abnormal motion of cilia can 
indicate the presence of ciliopathies, which present themselves in more visible phenotypes. 
Disorders resulting from the disruption of ciliary motion range from sinopulmonary disease 
such as in primary ciliary dyskinesia (PCD) [O’Callaghan, Diagnosing primary ciliary dyskinesia.], 
to mirror symmetric organ placement or randomized left-right organ placement as in heterotaxy 
[Garrod, Airway ciliary dysfunction]. Diagnosing patients exhibiting abnormal ciliary motion 
prior to surgery may provide the clinician with opportunities to institute prophylactic 
respiratory therapies to prevent these complications. Therefore, the study of ciliary motion 
may have a broad clinical impact.

The visual examination of ciliary motion by medical professionals is critical in diagnosing 
ciliary motion defects, but manual analysis is highly subjective and prone to error. We 
aim to develop an unsupervised, computational approach to analyze ciliary motion. 
Clustering and classification of motion and motion patterns is not an established problem 
in machine learning. However, the motion of cilia in particular is highly unique: 
cilia exhibit subtle, rotational, non-linear motion. Addressing this problem through 
machine learning techniques, therefore, is not only a unique problem in terms of its 
application, but on a theoretical bases, as many popular models fail to classify and 
cluster this type of motion accurately or meaningfully. Likewise, we also avoid using 
labeled data, as it would be costly in terms of both money and time for medical professionals 
to create an accurately labelled data set. We are thus presented with a novel, 
unsupervised clustering problem. 

Our approach is to create a pipeline that learns a low-dimensional representation of 
ciliary motion on unlabeled data. The model we propose considers the spatial and 
temporal dimensions of ciliary motion separately. The pipeline encodes each frame 
of the input video and then encodes the paths between frames in the latent space. 

The low dimensional latent space in this pipeline will have semantic significance, 
and thus it's interpolation and clustering should be meaningful for those studying 
ciliary motion and its connection to ciliapathies. 




Related Works
-------------
A computational method for identifying abnormal ciliary motion patterns was proposed 
by Shannon Quinn in 2015 []. In this paper, ciliary motion is treated as a dynamic 
texture. Dynamic textures are modeled as rhythmic motions of particles subjected to 
stochastic noise [Dorretto, Dynamic Textures]. Dynamic textures include familiar 
motion patters such as flickering flames and rippling water, and grass in the wind, 
and motion patterns common in biomedical contexts. Each instance of dynamic texture 
contains a small amount of stochastic behavior altering an otherwise regular visual 
pattern. Likewise, ciliary motion can be considered a dynamic texture as it consists 
of rhythmic behavior subject to stochastic noise that collectively determine the beat 
pattern. The paper uses autoregressive representations of optical flow features that 
were fed into a support vector machine classifier to decompose high-speed digital 
videos of ciliary motion into elemental components and classify them as normal or abnormal.

While this study proves there is merit in treating ciliary motion as a dynamic texture, 
the model used imposes some limitations. Specifically, the use of an autoregressive (AR) 
model for the classification task imposes some critical limitations. One limitation is 
that while AR models are often used in representing dynamic textures, they are primarily 
used in distinguishing between different types of dynamic textures, rather than between 
different instances of the same primary texture. Additionally, AR models impose strong 
parametric assumptions on the underlying structure of the data. This limits the power 
of AR models, as they are incapable of capturing nonlinear interactions. Lastly, even 
though the majority of the pipeline is automated, the previous work relied on clinical 
experts to manually draw regions of influence (ROI) from raw video data in order to 
focus the computational pipeline on the areas in the video containing cilia. This is 
potentially problematic in that expert drawn ROI introduces subjective bias into the 
study and makes the model rely on specialized labor, increasing the monetary cast 
and time of clinical operations.

Quinn's model was built upon by Lu in 2017 and 2018, the more recent of which which 
attempts to classify ciliary motion using stacked neural networks [LU and Quinn, 2018]. 
Lu's paper uses a deep learning approach to address the constraints of Quinn's 
previous paper. Deep learning models (convolutional neural networks in Lu's case) 
do not make strong parametric or linear assumptions on the underlying data, compared 
to autoregressive models, allowing more complex behavior to be captured. Additionally, 
the model in Lu's paper does not use hand-drawn ROI maps. Lu's 2018 paper uses DenseNet 
architecture [Huang 2015] to create segmentation maps for cilia video data; then, 
ciliary motion is treated as a time series using long short-term memory (LSTM) networks, 
a specific type of recurrent neural network (RNN), to model the long-term temporal 
dependencies in the data. This pipeline achieves an accuracy of 88\% when trained for 
200 epochs on the given data, rivalling Quinn's 2015 paper without the use of hand-drawn ROI maps.

We aim to build upon this research by developing a fully unsupervised pipeline to 
phenotype ciliary motion. This pipeline is advantageous in that it does not need 
hand-drawn ROI maps, nor a labelled dataset for training data. We also shift away 
from a classification style task (classifying abnormal versus normal ciliary motion) 
to a representational learning task (generating meaningful, low dimensional representations 
of ciliary motion). The pipeline could be adapted into a "black box" classification 
pipeline to be used by clinicians, or it can be used as a clinical tool to better 
understand which specific characteristics indicate normal or abnormal ciliary motion. 
We propose a modular, generative pipeline to generate low-dimensional representations 
of ciliary motion from video data. The pipeline consists of a prepossessing module, 
an appearance module, and a dynamics module. 

Methods
-------

Our proposed model is divided into three modules: preprocessing, appearance, and dynamics. 
The preprocessing module applies dense optical flow to the video, generating a velocity 
vector field. We then calculate quantities such as the divergence and curl of the vector field. 
This information is concatenated with the original video as additional channels to each 
frame to form an augmented video. This augmented video is fed frame-by-frame to the 
appearance module which learns a compressed spatial representation for images of cilia. 
Next, each frame is then fed through a variational autoencoder (VAE), generating a latent 
representation of the spatial aspects of cilia in that frame. Thus the video is transformed 
into a sequence of points, or path, in the compressed latent space. We carry out motion-based 
representation on this compressed sequence to reduce the amount of irrelevant information 
considered during the process, focusing only on the motion of the cilia. The dynamics module 
feeds the path through another VAE, generating a latent representation of the dynamics patterns 
of the cilia. Thus we factor the representation of the cilia into disentangled spatial and 
temporal components. We measure the performance of our model quantitatively by considering its 
reconstruction ability under usual VAE metrics such as MSE, as well as qualitatively by 
considering the semantic cohesiveness of the cilia data embeddings in the spatial and temporal latent spaces.


Appearance
==========

After the source data is augmented in the preprocessing module, it is fed into the appearance 
module frame-by-frame. We empirically observe that cilia don't exhibit a large degree of 
spatial differences over time, thus rather than processing every frame of the dataset, 
we opt to sample a fixed number of frames from each video. For testing purposes, we set the 
number of sampled frames to :math:`40`. Each augmented frame is input to a variational autoencoder (VAE) 
[Kingma and Welling, 2014], a class of generative model, based on the traditional autoencoder (AE), 
which generates a low-dimensional representation of the data parameterized as a probability distribution.

A general autoencoder (AE) attempts to learn a low-dimensional representation of the data by 
enforcing a so-called *bottleneck* in the network. This bottleneck is usually in the form 
of a hidden layer whose number of nodes is significantly smaller than the dimensionality of 
the input. The AE then attempts to reconstruct the original input using only this bottleneck 
representation. The idea behind this approach is that to optimize the reconstruction, only 
the most essential information will be maintained in the bottleneck, effectively creating 
a compressed, critical information based representation of the input data. The size of 
the bottleneck is a hyperparameter which attenuates how much the data is compressed.

With this task in mind, an AE can be considered as the composition of two constituent 
neural networks: the encoder, and the decoder. Suppose that the input to the AE has a 
dimensionality of :math:`n`, and we want the bottleneck to be of size :math:`l`, 
then we can write the encoder and decoder as fucntions mapping between :math:`\mathbb{R}^n` and :math:`\mathbb{R}^l`: 

.. math::

	e:\mathbb{R}^n\rightarrow\mathbb{R}^l 

.. math::

	d:\mathbb{R}^l\rightarrow\mathbb{R}^n

The encoder is tasked with taking the original data input and sending it to a 
compressed or *encoded* representation. The output of the encoder serves as the 
bottleneck layer. Then the decoder is tasked with taking this encoded representation and 
reconstructing a plausible input which could have been encoded to generate this representation. 
The loss target of a AE is generally some distance measure between items in the data space

Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension. 

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work. 

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references. 
