:author: Rajeswari Sivakumar
:email: rajeswari.a.sivakumar@gmail.com
:institution: University of Georgia

:author: Shannon Quinn
:email: spq@cs.uga.edu
:institution: University of Georgia
:bibliography: bibliography



------------------------------------------------
Parkinson's Classification and Feature Extraction from Diffusion Tensor Images
------------------------------------------------

.. class:: abstract

    Parkinson’s disease (PD) affects over 6.2 million people around the world. Despite its prevalence, there is still no cure, and diagnostic methods are extremely subjective,  relying on observation of physical motor symptoms and response to treatment protocols. Other neurodegenerative diseases can manifest similar motor symptoms and often too much neuronal damage has occurred before motor symptoms can be observed. The goal of our study is to examine  diffusion tensor images (DTI) from Parkinson’s and control patients through linear dynamical systems and tensor decomposition methods to generate features for training classification models. Diffusion tensor imaging emphasizes the spread and density of white matter in the brain. We will reduce the dimensionality of these images to allow us to focus on the key features that differentiate PD and control patients. We show through our experiments that these approaches can result in good classification accuracy (90%), and indicate this avenue of research has a promising future.


.. class:: keywords

    tensor decomposition, brain imaging, diffusion tensor image, Parkinsons disease

Introduction
------------
Parkinson’s Disease
+++++++++++++++++++
  Parkinson’s disease (PD) is one of the most common neurodegenerative disorders.
  The disease mainly affects the motor systems and its symptoms can include shaking,
  slowness of movement, and reduced fine motor skills. As of 2015 an estimated
  6.2 million globally were afflicted with the disease. Its cause is largely unknown
  and there are some treatments available, but no cure has yet been found.
  Early diagnosis of PD is a topic of keen interest to diagnosticians and
  researchers alike. Currently Parkinson’s is diagnosed based on the presence of
  observable motor symptoms and change in symptoms in response to medications that
  target dopaminergic receptors such as Levdopa.
  The problem with this approach is that it relies on treating symptoms instead of
  preventing them. Once motor symptoms present, at least 60% of neurons have been
  affected and there is little likelihood of healing them fully. Additionally
  early diagnosis will help reduce likelihood of misdiagnosis
  with other motor neuron diseases.

Parkinson’s Progression Markers Initiative Datasets
+++++++++++++++++++++++++++++++++++++++++++++++++++
  The Parkinson’s Progression Markers Initiative (PPMI) is an observational
  clinical study designed to identify PD biomarkers [4] and contribute towards new
  and better treatments for the disease. The cohort consists of approximately 400
  de novo, untreated PD subjects and 200 healthy subjects followed longitudinally
  for clinical, imaging and biospecimen biomarker assessment. The PPMI data set is
  a collection of biomarker data collected from a longitudinal study of Parkinson’s
  and control subjects. They have thus far collected DaT scan, MRI, fMRI, and CT
  scan data from several hundred subjects in 6 month intervals. They first began
  collecting data in 2010, funded by the Michael J.Fox Foundation.
  The dataset chosen for this paper was PPMI’s Diffusion Tensor Imaging (DTI) records.
  DTI has been shown to be a promising biomarker in Parkinsonian symptoms [5] and can
  provide unique insights into brain network connectivity. Moreover, the DTI data was
  one of PPMI’s cleanest and largest datasets and thus expected to be one of the most
  useful for further analysis. A DTI record is a four-dimensional dataset comprised of
  a time-series of a three-dimensional imaging sequence of the brain. PPMI’s DTIs
  generally consisted of 65 time slices, each taken approximately five seconds apart.

Existing Work
-------------
Parkinson’s Disease
+++++++++++++++++++
  A variety of tools currently exist for diagnosis of Parkinson’s through
  pre-motor symptoms. For example Parkinson’s seems to measurably affect olfactory
  sensitivity prior to presenting motor symptoms more than other motor neuron diseases,
  as illustrated by the University of Pennsylvania Smell Identification Test (UPSIT).
  While there is still more work needed to refine tests like these, it is one example
  that proves the feasibility of earlier diagnosis of Parkinson’s disease.
  The PPMI holds that discovery of one or more biomarkers for PD is a critical step
  for developing treatments for the disease. Chahine & Stern conducted a search
  of existing PD articles relating to objective biomarkers for PD and found that
  there are several potential candidates, including biofluids, peripheral tissue,
  imaging, genetics, and technology based objective motor testing.
  Dinov et al. explored both model-based and model-free approaches for PD
  classification and prediction, jointly processing imaging, genetic, clinical,
  and demographic data. They were able to develop and full data-processing
  pipeline enabling modeling of all the data available from PPMI, and found that
  model-free approaches such as support vector machines (SVM) and K-nearest-neighbor
  (KNN) outperformed model-based techniques like logistic regression in terms of
  predicted accuracy. Several of these classifiers generated specificity exceeding
  96% when all data available from the dataset was aggregated and used. One
  interesting finding was a notable increase in accuracy when using group size
  rebalancing techniques to counteract the effect of cohort sample-size disparities
  (there are many more patients than control subjects), increasing accuracy in one
  SVM classifier from 75.9% to 96.3%.
  Baytas et al. recognized the inherent difficulty of using time-series analysis
  techniques on longitudinal data collected at irregularly-spaced intervals and proposed a new Long-Short Term Memory (LSTM) technique: Time-Aware LSTM (T-LSTM). Simuni et al. (2016) found that the subgroup PD classification of tremor dominant (TD) versus postural instability gait disorder dominant (PIGD) has substantial variability, especially in the early stages of diagnosis. For this reason no attempt was made in this paper to include subtype assignment, but only to learn a binary Yes/No PD classification prediction.
  State-of-the art Parkinson’s classification results were reported by
  Adeli et al. in early 2017 through use of a joint kernel-based feature
  selection and classification framework. Unlike conventional feature selection
  techniques, this allowed them to select features that best benefit the classification
  scheme in the kernel space as opposed to the original input feature space.
  They analyzed MRI and SPECT data of 538 subjects from the PPMI database and
  obtained a diagnosis accuracy of 70.5% in MRI generated features and 95.6% in
  SPECT image generated features. The authors speculated that their non-linear
  feature selection was the reason for their outperformance of other methods on
  this non-linear classification problem. Other researchers, Banerjee, et al. were
  able to achieve 98.53% using ensemble learning methods trained on
  T1 weighted MRI data. However Banerjee used several domain knowledge based feature
  extraction methods to preprocess their data including image registration,
  segmentation, and volumetric analysis.
  Our present research strikes a balance between the two. While our
  autoregressive model does utilize a basic understanding of relevance of time
  in diffusion tensor imaging, we do not utilize any other domain specific
  knowledge to inform our feature extraction. Our hope is to build a
  generalizable approach that can be applied to other data structured similarly
  both within and outside the domain of biomedical image analysis. Additionally
  we want to improve the models being trained without domain specific knowledge
  on MRI data. This is because MRI is a far less invasive brain imaging method
  than SPECT imaging which is an X-ray based technique and must be used at a
  limited frequency. Additionally the multiple MRI modalities offer versatility
  in examining biological structures.
Tensor and Matrix Decomposition
+++++++++++++++++++++++++++++++
  Matrix decomposition has been used in a variety of computer vision applications
  in recent years including analysis of facial features. It offers a another
  means of quantifying the features that describe the relationships between
  values in a 2D space and can be generalized to a variety of applications.
  The key being that decomposition offers a powerful means of simultaneously
  evaluating the relationships of values in a 2 or higher dimensional space.
  In higher dimensional spaces, tensor decomposition is used, where tensors are
  a generalization of matrices.
  Matrix decomposition can be described as a means of separating a matrix into
  several component matrices whose product would result in the original matrix.
  For example when solving a system of equations you might approach formulate
  the problem as:

  .. math::

      Ax = b

  where A is a matrix and x and b are vectors. When
  trying to solve this equation, we could apply a matrix decompositions
  operations to the matrix A, to more efficiently solve the system. By
  finding the products of the of x and b with the the one matrix resulting from
  the decomposition and the inverse of the other, we can solve the system of
  equations with significantly fewer operations. We can generalize this premise
  to machine learning, when model complexity of models, often result in
  exponential increases in number of computations. This also affects the
  applications of new algorithms and pipelines can be used in because of their
  complexity.
  We can choose specific types of decompositions that also allow us to preserve
  unique information about original matrix while also reducing the the size of
  the matrix. For example, in the case of singular value decomposition we are
  trying to solve:

  .. math::

      A = USV^T

  Where A is the original matrix, of size m ✕ n, U is an orthogonal matrix of
  size m ✕ n, S is a diagonal matrix of size n ✕ n, and VT is an orthogonal
  matrix of size n ✕ n. This generalization of the eigendecomposition is useful
  in compressing matrices without losing information. It will come into play
  with our final experiment using linear dynamical systems to extract features
  from the DTIs.
  Extending the premise of singular value decomposition (SVD) to higher order
  matrices, or tensors, we come to Tucker decomposition.
  Similarly to SVD, it is used to compress tensors. We are thus able to use it
  as means to describe brain images without breaking down specific regions of
  interest or or focusing on specific brain images.

Methods
-------
There are two main experiments conducted. We examine both Tucker tensor
decomposition and a linear dynamical systems approach to reduce number of
dimensions and scale down diffusion tensor images. The goal is to evaluate
the two approaches for the quality of features extracted. To this end, the
final feature vectors produced by each method is then passed on to a random
forest classifier, where the accuracy of the final trained model is measured
on a classification task to predict control or Parkinson’s (PD) group.
