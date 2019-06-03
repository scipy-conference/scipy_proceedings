:author: Christian McDaniel
:email: clm121@uga.edu
:institution: University of Georgia

:author: Shannon Quinn, PhD
:email: spq@uga.edu
:institution: University of Georgia
:bibliography: bibli

***********************************************************************************************************************************************
Developing a Graph Convolution-Based Analysis Pipeline for Multi-Modal Neuroimage Data: An Application to Parkinson's Disease
***********************************************************************************************************************************************

.. class:: abstract

Parkinson's disease (PD) is a highly prevalent neurodegenerative disease originating in subcortical areas of the brain and resulting in progressively worsening motor, cognitive, and psychological (e.g., depression) symptoms. Neuroimage data is an attractive research tool given the neurophysiological origins of the disease. Despite the insights potentially available in magnetic resonance imaging (MRI) data, and the noninvasive and low-risk nature thereof, developing sound analytical techniques for this data has proven difficult. Principally, multiple image modalities are needed to provide clinicians with the most accurate view possible; the process of incorporating multiple image modalities into a single holistic model is both poorly defined and extremely challenging. Previous work has demonstrated that graph-based signal processing techniques can address this issue, while also drastically reducing the size and complexity of the data. In this paper, we propose a novel graph-based convolutional neural network architecture and present an end-to-end pipeline for preprocessing, formatting, and analyzing this highly complex data. On data downloaded from the Parkinson's Progression Markers Initiative (PPMI) database, we are able to exceed the performance of related models.

Introduction
============
Affecting more than 1% of the United States population over the age of 60, Parkinson's disease (PD) is the second-most prevalent age-related neurodegenerative disease following Alzheimer's disease :cite:`RST2014`. PD diagnosis has traditionally relied on clinical assessments with some degree of subjectivity :cite:`GGLVVZ2018`, often missing early-stage PD altogether :cite:`BDH2016`. Benchmarks for delineating PD progression or differentiating between similar conditions are lacking [:cite:`LMSACRMW2018,LWXGXKZ2012`]. As such, many efforts have emerged to identify quantitatively rigorous methods through which to distinguish PD.

Neuroimage data is an attractive tool for PD research. Magnetic resonance imaging (MRI) in particular is safe for patients, highly diverse in what it can capture, and decreasing in cost to acquire. Recent work shows that multiple MRI modalities are required to provide researchers and clinicians with the most accurate view of a patient's physiological state [:cite:`LCLZFFPK2015,BDH2016,LWXGXKZ2012`]. For example, anatomical MRI (aMRI [1]_) data is useful for identifying specific brain regions, but the Euclidean distance between regions does not well-approximate the functional or structural connectivity between them. Diffusion-weighted MRI (dMRI) measures the flow of water through the brain in order to track the tube-like connections (i.e., *nerve fiber bundles* a.k.a. *tracts*) between regions (i.e., *tractography*, see Appendix A below for more information), and functional MRI (fMRI) measures changes in blood oxygenation to approximate which regions of the brain function together. As such, it is useful to analyze a combination of these modalities to gain insights from multiple measures of brain physiology. Processing and analyzing multi-modal data together is both poorly defined and extremely challenging, requiring combined expertise from neuroscience and data analytics.

MRI data is inherently noisy data and requires extensive preprocessing before analysis can be performed. This is often left to the researcher to carry out; many techniques exist, and the technical implementation decisions made along the way can affect the outcome of later analysis. This is a major barrier to reproducibility and prevents data analysts from applying their skills in this domain. More work is needed to automate the procedure and provide better documentation for steps which require case-specific input. To that end, we discuss our findings and methods below, and our code is available on GitHub [2]_.

Following preprocessing, we address the issue of analyzing multimodal MRI data together. Previous work has shown that graph-based signal processing techniques allow multimodal analysis in a common data space [:cite:`KPFRLGR2017,KPFRLGR2018,ZHCLZW2018`]. It has been shown that graph-based signal processing classifiers can be incorporated in neural network-like architectures and applied to neuroimage data. Similar to convolutional neural networks, Graph Convolutional Networks (GCNs) learn *filters* over a graph so as to identify patterns in the graph structure, and ultimately perform classification on the nodes of the graph.  In this paper, following the discussion of our preprocessing pipeline, we propose a novel GCN architecture which uses graph attention network (GAT) layers to perform whole-graph classification (e.g., PD or healthy control) on graphs formed from multimodal neuroimage data.

On data downloaded from the Parkinson's Progression Markers Initiative (PPMI), we _____. As a note going forward, we have tried to make this paper accassible to the general SciPy audience, and have included more field-specific information in the Appendices at the end and as parenthetical addendums throghout.

Related Works
=====================
While genetic and molecular biomarkers have exhibited some efficacy in developing a PD blueprint [:cite:`GGLVVZ2018,MLLAMWGSECES2018,BP2014`], many research efforts have turned to neuroimaging due to its noninvasive nature and alignment with existing knowledge of the disease. Namely, PD affects a major dopamine-producing pathway (i.e., the nigrostriatal dopaminergic pathway) of the brain :cite:`Brodal2016`, and results in various structural and functional brain abnormalities that can be captured by existing imaging modalities [:cite:`ZYHJZLWWZZLLH2018,MLLAMWGSECES2018,GLHSA2014,TBEDTEEE2015,LSCZCCYLHGS2014,GRSKMFZJHM2016`]. Subsequent whole-brain neuroimage analysis has identified PD-related regions of interest (ROIs) throughout the brain, from cortical and limbic regions to the brainstem and cerebellum [:cite:`BWSWBKSDRH2011,TBEDTEEE2015,GRSKMFZJHM2016`].

As neuroimage data has accumulated, researchers have worked to develop sound analytical techniques for the complex images. Powerful machine learning techniques have been employed for analyzing neuroimage data [:cite:`MLLAMWGSECES2018,TBEDTEEE2015,BWSWBKSDRH2011,LSCZCCYLHGS2014`], but algorithmic differences can result in vastly different results [:cite:`GLHSA2014,K2018,ZYHJZLWWZZLLH2018`]. :cite:`CJMRCMBD2017` and :cite:`GRSKMFZJHM2016` found that implementation choices made during the processing pipeline can affect analysis results as much as anatomical differences themselves (e.g., when performing white matter (WM) tractography on diffusion-weighted MRI (dMRI) data and in group analysis of resting-state functional MRI (rfMRI) data, respectively). To overcome the effect of assumptions made by a given analysis algorithm, many researchers have turned to applications of deep machine learning (DL) for neuroimage data analysis. Considered “universal function approximators,” DL algorithms are highly flexible and therefore have low bias. Examples of DL applications to neuroimage analysis are widespread. :cite:`KUHSMHBB2016` proposes a 3D convolutional neural network (CNN) for skull stripping 3D brain images, :cite:`HDCLPC2018` proposes a novel recurrent neural network plus independent component analysis (RNN-ICA) model for fMRI analysis, and :cite:`HCSAAP2014` demonstrate the efficacy of the restricted Boltzmann machine (RBM) for network identification. :cite:`LZCY2017` offer a comprehensive review of deep learning-based methods for medical image computing.

Multi-modal neuroimage analysis is increasing in prevalence [:cite:`BSSNSOV2018,LCLZFFPK2015,BDH2016,LMSACRMW2018,LWXGXKZ2012`] due to limitations of single modalities, resulting in larger and increasingly complex data sets. Recently, researchers have utilized advances in graph convolutional networks to address these concerns. We discuss the mathematical background of graph convolutional networks (GCNs) and graph attention networks (GATs, a variant of GCNs with added attention mechanisms) in the Methods Section below and Appedix B. Principally, this paper is based on the advancements made by :cite:`KW2017` and :cite:`VCCRLB2018` on GCNs and GATs respectively.

This work follows from previous efforts applying GCNs to similar classification tasks. :cite:`SNFOV2013`, in addition to providing in-depth intuition behind spectral graph processing (i.e., processing a signal defined on a graph structure), demonstrate graph spectral filtering on diffusion signals defined on a cerebral cortex graph. Their paper laid the groundwork for incorporating spectral graph processing into neural network architectures. To classify image objects based on multiple “views” or angles, [:cite:`KZS2015,KCR2016,ZHCLZW2018`] developed “siamese” and “multi-view” neural networks. These architectures share weights across parallel neural networks and group examples into pairs, aiming to classify the pairs as being from the same class or different classes. Efforts to utilize GCNs for multimodal neuroimage data have used similar pairwise grouping as a way to increase the size of their data set. [:cite:`KPFRLGR2017,KPFRLGR2018`] train GCN models to learn similarity metrics between subjects with Autism Spectrum Disorder (ASD) and healthy controls (HC), using fMRI data from the Autism Brain Imaging Data Exchange (ABIDE) database. :cite:`ZHCLZW2018` apply a similar architecture to learn similarity metrics between subjects with Parkinson's disease (PD) and HC, using dMRI data from the PPMI data set. Their work inspired our paper; to our knowledge, we are the first publication that uses GCNs to predict the class of neuroimage data directly, instead of making predictions on pairwise examples.

Discussion of the End-to-End Pipeline
=======================================

This section walks through our pipeline, which handles the formatting and preprocessing of multimodal neuroimage data and readies it for analysis via our GCN architecture. We reference the specific python files that handle each task, and we provide some background information. More information can be found in the Appendices below.

Data Formatting
------------------

MRI data requires extensive artifact correction and removal before it can be used. MRI signals are acquired through the application of precisely coordinated magnetic fields and radiofrequency (RF) pulses. Each image is reconstructed from a series of recordings averaged over many individual signals. This inherently results in noisy measurements, magnetic-based artifacts, and artifacts from human error such as motion artifacts [:cite:`Wang2015,HBL2010`]. As such, extensive preprocessing must be performed to clean the data before analysis. Appendix A provides more details on the main MRI modalities.

Our pipeline assumes that a "multi-zip" download is used to get data from the PPMI database. When using the "Advanced Download" option on the PPMI database, the data is split into multiple zip files, often splitting up the data of a single subject. The file :code:`neuro_format.py` combines the data from multiple download folders into a single folder, consolidating the multiple zip files and recombining data from the same subject.

Next, before preprocessing, images should be converted to the Neuroimaging Informatics Technology Initiative (NIfTI) [3]_ file format. Whereas many MRI data are initially in the Digital Information and Communications in Medicine (DICOM) [4]_ format for standardized transfer of medical data and metadata, the NIfTI format is structured for ease of use when conducting computational analysis and processing on these files. The size, orientation, and location in space of the voxel data is dependent on settings used during image acquisition and requires an *affine matrix* to relate two images in a standard coordinate space. The NIfTI file format automatically associates each image with an affine matrix as well as a *header file*, which contains other helpful metadata. The software :code:`dcm2niix` [5]_ is helpful for converting the data from DICOM format to NIfTI format.

Next, it is common practice to convert your data file structure to the Brain Imaging Data Structure (BIDS) [6]_ format. Converting data to the BIDS format is required by certain softwares, and ensures a standardized and intuitive file structure. There exist some readily available programs for doing this, but we wrote our own function specifically for PPMI data in :code:`make_bids.py`, as the PPMI data structure is quite nuanced. This file also calls :code:`dcm2niix` to convert the image files to NIfTI format.

Data Preprocessing
-------------------

This subsection discusses the various softwares and commands used to preprocess the multimodal MRI data. The bash script :code:`setup` should help with getting the necessary dependencies installed [7]_. The script was written for setting up a Google cloud virtual machine, and assumes the data and pipeline files are already stored in a Google cloud bucket.

The standard software for preprocessing anatomical MRI (aMRI [1]_) data is Freesurfer [8]_. Freesurfer is an actively developed software with responsive technical support and rich forums. The software is dense and the documentation is lacking in some areas, so training may still be helpful, although not available in our case. The :code:`recon-all` command performs all the steps needed for standard aMRI preprocessing, including motion correction, registration to a common coordinate space using the Talairach atlas by default, intensity correction and thresholding, skull-stripping, region segmentation, surface tessellation and reconstruction, statistical compilation, etc.

The entire process takes around 15 or more hours per image. Support for GPU-enabled processing was stopped years ago, and the :code:`-openmp <num_cores>` command, which allows parallel processing across the designated number of cores, may only reduce the processing time to around 8-10 hours per image [9]_. We employed many CPU cores using Google Cloud Platform virtual machines and utilized the python module :code:`joblib.Parallel` to run many single-core processes in parallel. For segmentation, the Deskian/Killiany atlas is used, resulting in around 115 volume segmentations per image, to be used as the nodes for the graph.

The Functional Magnetic Resonance Imaging of the Brain (FMRIB) Software Library (FSL) [10]_ is often used to preprocess diffusion data (dMRI). The *b0* volume is taken at the beginning of dMRI acquisition and is used to align dMRI images to aMRI images of the same subject. This volume is isolated (:code:`fslroi`) and merged with b0's of other clinic visits (CVs) [11]_ for the same subject (:code:`fslmerge`). :code:`fslmerge` requires that all dMRI acquisitions for a given subject have the same number of coordinates (e.g., (116,116,**78**,65) vs. the standard (116,116,**72**,65)). Since some acquisitions had excess coordinates, we manually examined these images and, if possible, removed empty space above or below the brain. Otherwise, these acquisitions were discarded. Next, the brain is isolated from the skull (*skull stripped*, :code:`bet` with the help of :code:`fslmaths -Tmean`), susceptibility correction is performed *for specific cases* (see below) using :code:`topup`, and eddy correction is performed using :code:`eddy_openmp`. (Magnetic) susceptibility and eddy correction refer to specific noise artifacts that significantly affect dMRI data.

The :code:`topup` tool requires two or more dMRI acquisitions for a given subject, where the image acquisition parameters :code:`TotalReadoutTime` and/or :code:`PhaseEncodingDirection` (found in the image's header file) differ from one another. Since the multiple acquisitions for a given subject typically span different visits to the clinic (CVs), the same parameters are often used and :code:`topup` cannot be utilized. We found another software, BrainSuite [12]_, which can perform susceptibility correction using a single acquisition. Although we still include FSL in our pipeline since it is the standard software used in many other papers, we employ the BrainSuite software's Brain Diffusion Pipeline to perform susceptibility correction and to register the corrected dMRI data to the aMRI data for a given subject (i.e., *coregistration*).

First, a BrainSuite compatible brain mask is obtained using :code:`bse`. Next, :code:`bfc` is used for bias field (magnetic susceptibility) correction, and finally :code:`bdp` performs co-registration of the diffusion data to the aMRI image of the same subject. The calls to the Freesurfer, FSL, and BrainSuite software libraries are included in :code:`automate_preproc.py`.

Once the data has been cleaned, additional processing is performed on the diffusion (dMRI) data. As discussed in the Introduction section, dMRI data measures the diffusion of water throughout the brain. The flow of water is constricted along the tube-like pathways (tracts) that connect regions of the brain, and the direction of diffusion can be traced from voxel to voxel to approximate the paths of tracts between brain regions. There are many algorithms and softwares that perform tractography, and the choice of algorithm can greatly affect the analysis results. We use the Diffusion Toolkit (DTK) [13]_ to perform multiple tractography algorithms on each diffusion image. In :code:`dtk.py` we employ four different diffusion tensor imaging (DTI)-based deterministic tractography algorithms: Fiber Assignment by Continuous Tracking (FACT; :cite:`MCCZ1999`), the second-order Runge–Kutta method (RK2; :cite:`BPPDA2000`), the tensorline method (TL; :cite:`LWTJAM2003`), and the interpolated streamline method (SL, :cite:`CLCASSMBR1999`). :cite:`ZZWJJPNLYT2015` provide more information on each method. :code:`dti_recon` first transforms the output file from Brainsuite into a usable format for DTK, and then :code:`dti_tracker` is called for each of the tractography algorithms. Finally, :code:`spline_filter` is used to smooth the generated tracts, denoising the outputs. Now that the images are processed, they can be efficiently loaded using python libraries :code:`nibabel` and :code:`dipy`, and subsequently operated on using standard data analysis packages such as :code:`numpy` and :code:`scipy`.

Defining Graph Nodes and Features
----------------------------------------------------------

Neuroimage data is readily applied to graph processing techniques and is often used as a benchmark application for new developments in graph processing :cite:`SNFOV2013`. Intuitively, the objective is to characterize the structural and functional relationships between brain regions, since correlations between PD and abnormal brain structure and function have been shown. As such, the first step is to define a graph structure for our data. This step alone has intuitive benefits. Even after preprocessing, individual voxels of MRI data contain significant noise that can affect analysis :cite:`GRSKMFZJHM2016`. Brain region sizes vary greatly across individuals and change over one individual's lifetime (e.g., due to natural aging :cite:`Peters2006`). Representing regions as vertices on a graph meaningfully groups individual voxels and mitigates these potential red herrings from analysis.

We use an undirected weighted graph :math:`\mathcal{G} = {\mathcal{V},\mathcal{E}, \textbf{W}}` with a set of vertices :math:`\mathcal{V}` with :math:`|\mathcal{V}| =` the number of brain regions :math:`N`, a set of edges :math:`\mathcal{E}`, and a weighted adjacency matrix :math:`\textbf{W}`, to represent our aMRI data. :math:`\mathcal{G}` is shared across the entire data set to represent general population-wide brain structure. Each vertex :math:`v_{i} \in \mathcal{V}` represents a brain region. Together, :math:` \mathcal{V}, \mathcal{E}, and \textbf{W}` form a *k-Nearest Neighbor adjacency matrix*, in which each vertex is connected to its *k* nearest neighbors (including itself) by an *edge*, and edges are weighted according to the average Euclidean distance between two vertices. The weight values are normalized by dividing each distance by the maximum distance from a given vertex to all of its neighbors, :math:`d_{ij} \in [0,1]`. (Refer to Appendix B for details.) :code:`gen_nodes.py` first defines the vertices of the graph using the anatomical MRI data, which has been cleaned and *segmented* into brain regions by Freesurfer. The center voxel for each segmentation volume in each image is calculated. Next, :code:`adj_mtx.py` calculates the mean center coordinate across all aMRI images for every brain region. The average center coordinate for each region :math:`i` is a vertex :math:`v_{i} \in \mathcal{V}` of the graph :math:`\mathcal{G}`. See Figure :ref:`adjmtx` for a depiction of the process.

.. figure:: adj_mtx_fig.png

    A depiction of the steps involved in forming the adjacency matrix. First, anatomical images are segmented into regions of interest (ROIs), which represent the vertices of the graph. The center voxel for each ROI is then calculated. An edge is placed between each node *i* and its *k*-nearest neighbors, calculated using the center coordinates.  Lastly, each edge is weighted by the normalized distance between each node *i* and its connected neighbor *j*. :label:`adjmtx`

Using these vertices, we wish to incorporate information from other modalities to characterize the relationships between the vertices. We define a *signal* on the vertices as a function :math:`f : \mathcal{V} \rightarrow \mathbb{R}`, returning a vector :math:`\textbf{f} \in \mathbb{R}^{N}`. These vectors can be analyzed as *signals* on each vertex, where the change in signal across vertices is used to define patterns throughout the overall graph structure. In our case, the vector signal defined on a vertex :math:`v_{i}` represents that vertex's weighted connectivity to all other vertices :cite:`SNFOV2013`, where the weights correspond to the strength of connectivity between :math:`v_{i}` and some other vertex :math:`v_{j}`, as calculated by a given tractography algorithm. As such, each signal is a vertex of size :math:`N` and there are :math:`N` signals defined on each graph, forming an :math:`N`x:math:`N` *weighted connectivity matrix*. Each dMRI image has one :math:`N`x:math:`N` set of signals for each tractography algorithm. In this way, the dimensionality of the data is drastically reduced, and information from multiple modalities and processing algorithms may be analyzed in a common data space.

:code:`gen_features.py` approximates the strength of connectivity between each pair of vertices. For this, the number of tracts (output by each tractography algorithm) connecting each pair of brain regions must be counted. Recall that each image carries with it an affine matrix that translates the voxel data to a coordinate space. Each preprocessing software uses a different coordinate space, so a new affine matrix must be calculated to align the segmented anatomical images and the diffusion tracts (i.e., *coregistration*). Freesurfer's :code:`mri_convert`, FSL's :code:`flirt`, and DTK's :code:`track_transform` are used to put the two modalities in the same coordinate space so that voxel-to-voxel comparisons can be made. Next, :code:`nibabel`'s i/o functionality is used to generate a mask file for each brain region, :code:`nibabel.streamlines` is used to read in the tractography data and :code:`dipy.tracking.utils.target` is used to identify which tracts travel through each volume mask. The tracts are encoded using a unique hashing function to save space and allow later identification.

To generate the signals for each vertex, :code:`utils.py` uses the encoded tract IDs assigned to each volume to count the number of tracts connecting each volume pair. The number of connections between pairs of brain regions approximate the connection strength, and these values are normalized similar to the normalization scheme mentioned above for the k-nearest neighbor weights. Figure :ref:`featsfig` offers a visualization.

.. figure:: feats_fig.png

  The process of generating the features from a single tractography algorithm is shown. Tractography streamlines are aligned to a corresponding anatomical image. The number of streamlines connecting each pair of brain regions is calculated to represent the strength of connection. Using each brain region as a vertex on the graph, the connection strengths between a given vertex to all other vertices are compiled to form the signal vector for that vertex. :label:`featsfig`

Graph Convolutional Networks
----------------------------------------------------------

Common to many areas of data analysis, *spectral graph processing* techniques (i.e., processing a signal defined on a graph structure) have capitalized on the highly flexible and complex modeling capacity of so-called deep learning neural network architectures. The layered construction of nonlinear calculations loosens rigid parameterizations of other classical methods. This is desirable, as changes in parameterizations have been shown to affect results in both neuroimage analysis (e.g., independent component analysis (ICA) :cite:`CJMRCMBD2017`) and in graph processing (e.g., the explicit parameterization used in Chebyshev approximation; see Appendix B for details :cite:`KW2017`).

In this paper, we utilize the Graph Convolutional Network (GCN) to compute signal processing on graphs. GCNs were originally used to classify the vertices of a single graph using a single set of signals defined on its vertices. Instead, our task is to learn signal patterns that generalize over many subjects' data. To this end, we designed a novel GCN architecture, which combines information from anatomical and diffusion MRI data, processes data from multiple diffusion MRI tractography algorithms, and consolidates this information into a single vector so as to compare many subjects' data side-by-side. A single complete forward pass of our model consists of multiple parallel Graph Convolutional Networks (one for each tractography algorithm), max pooling, and graph classification via Graph Attention Network layers. We will briefly explain each part in this subsection; see Appendix B for a deeper discussion.

The convolution operation measures the amount of change enacted on a function :math:`f_{1}` by combining it with another function :math:`f_{2}`. We can define :math:`f_{2}` such that its convolution with instances of :math:`f_{1}` from one class (e.g., PD) produce large changes while its convolution with instances of :math:`f_{1}` from another class (e.g., HC) produce small changes; this provides a way to discriminate instances of :math:`f_{1}` into classes without explicitly knowing the class values. Recall that we have defined a function :math:`f` over the vertices of our graph using dMRI data (i.e., the *signals*). We seek to learn functions, termed *filters*, that, when convolved with the input graph signals, transform the inputs into distinguishable groups according to class value (e.g., PD vs. healthy control). This is similar to the local filters used in convolutional neural networks, except that the filters of GCNs use the connections of the graph (i.e., the edges) to establish locality.

The convolution operator is made possible over a graph structure by utilizing a few insights from spectral graph theory. Namely, the normalized graph Laplacian is a representation of the graph written as

.. math::

    \textup{\L{}} = I - D^{\frac{-1}{2}} \textbf{W} D^{\frac{-1}{2}},

where :math:`I` is the identity matrix with 1's along the diagonal and 0's everywhere else, :math:`W` is the weighted adjacency matrix defined earlier w.r.t. :math:`\mathcal{G}`, and :math:`D` is a weighted degree matrix such that :math:`D_{ii} = \sum_{j} \textbf{W}_{ij}`. :math:`\textup{\L{}}` can be factorized via a process called *eigendecomposition*, or the graph Fourier transform, as :math:`\textup{\L{}} = U \Lambda U^{T}`, where :math:`U = (u_{1},...,u_{N})` is a complete set of orthonormal eigenvectors, and :math:`\Lambda` are the associated real, non-negative eigenvalues. This representation of :math:`\textup{\L{}}` is in the Fourier domain, in which the convolution operation becomes multiplication.

Recall that we wish to convolve functions :math:`f` (i.e., our input signals) and :math:`g` (i.e., our filters to be learned), which are both functions over the vertices of :math:`\mathcal{G}`. The graph Fourier transform can be applied to :math:`f` and :math:`g` by multiplication with :math:`U` :cite:`HBL2015`,

.. math::

    x*g_{\theta} = Ug_{\theta}U^{T}x

where :math:`x` is an input instance of :math:`f` (i.e., the signal at a single vertex), :math:`\theta` are the coefficients we wish to learn, and :math:`g_{\theta}` is a function of the eigenvalues of :math:`\textup{\L{}}`, :math:`g_{\theta}(\Lambda)` :cite:`KW2017`.

Our specific implementation is based off the :code:`GCN` class from :cite:`KW2017`'s PyTorch implementation [14]_, which has several computational improvements over the original graph convolution formula. In short, we define the graph convolutional operation as

.. math::

    Z = \tilde{D}^{\frac{-1}{2}}\tilde{W}\tilde{D}^{\frac{-1}{2}} X \Theta.

A so-called *renormalization trick* has been applied to :math:`\textup{\L{}}` wherein :math:`I_{N}` has been added. :math:`I_{N}+D^{\frac{-1}{2}}WD^{\frac{-1}{2}}` becomes :math:`\tilde{D}^{\frac{-1}{2}}\tilde{W}\tilde{D}^{\frac{-1}{2}}`, where :math:`\tilde{W} = W+I_{N}` and :math:`\tilde{D}_{ii} = \sum_{j} \tilde{W}_{ij}`. I.e., self-loops have been added to the adjacency matrix. :math:`\Theta \in \mathbb{R}^{CxF}` is a matrix of trainable coefficients, where :math:`C` is the length of the input signals at each node, and :math:`F` is the number of C-dimensional filters to be learned. :math:`X` is the matrix of input signals for all vertices (i.e., the signals from a single tractography output of a single dMRI image). :math:`Z \in \mathbb{R}^{NxF}` is the matrix of convolved signals, where :math:`F` is the number of filters and :math:`N` the number of vertices in :math:`\mathcal{G}`. We will call the output signals *features* going forward.

Generalizing :math:`\Theta` to the weight matrix :math:`\textbf{W}(l)` at a layer :math:`l` and :math:`X=H(l)` as the inputs to layer :math:`l`, where :math:`H(0)` is the original data, we can calculate a hidden layer of our GCN as

.. math::

    H(l+1) = \sigma(\tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}}H(l)\textbf{W}(l)),

where :math:`\sigma` represents a nonlinear activation function (e.g., :math:`ReLU`). The :code:`GCN` class from :cite:`KW2017`'s PyTorch implementation [14]_ defines a two layer graph convolutional network as

.. math::

    Z = f(X,A) = softmax(\hat{A} ReLU(\hat{A}X\textbf{W}(0))\textbf{W}(1)),

where :math:`\hat{A} = \tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}}`.

Multi-View Pooling
-------------------------
For each dMRI acquisition, *d* different tractography algorithms are used to compute multiple “views” of the diffusion data. To account for the variability in the outputs produced by each algorithm, we wish to compile the information from each before classifying the whole graph. As such, *d* GCNs are trained side-by-side, such that the GCNs share their weights [:cite:`KZS2015,KPFRLGR2017`]. This results in *d* output graphs, i.e. *d* output vectors for each vertex. The vectors corresponding to the same vertex are pooled using max pooling, which has been shown to outperform mean pooling :cite:`ZHCLZW2018`.

Graph Attention Networks
-------------------------
Recent development of attention-based mechanisms allows for a weighting of each vertex based on its individual contribution during learning, thus facilitating whole-graph classifications.
In order to convert the task from classifying each node to classifying the whole graph, the features on each vertex must be pooled to generate a single feature vector for each input. The *self-attention* mechanism, widely used to compute a concise representation of a signal sequence, has been used to effectively compute the importance of graph vertices in a neighborhood :cite:`VCCRLB2018`. This allows for a weighted sum of the vertices' features during pooling.

:cite:`VCCRLB2018` use a single-layer feedforward neural network as an attention mechanism :math:`a` to compute *attention coefficients e* across pairs of vertices in a graph. For a given vertex :math:`v_{i}`, the attention mechanism attends over its first-order neighbors :math:`v_{j}`; :math:`e_{ij} = a(\textbf{W_{a}}h_{i}, \textbf{W_{a}}h_{j})`, where :math:`h_{i}` and :math:`h_{j}` are the features on vertices :math:`v_{i}` and :math:`v_{j}`, and :math:`\textbf{W_{a}}` is a shared weight matrix applied to each vertex's features. :math:`e_{ij}` is normalized via the softmax function to compute :math:`a_{ij}`: :math:`a_{ij} = softmax(e_{ij}) = exp(e_{ij}) / \sum_{k \in \mathcal{N}_{i}} exp(e_{ik})`, where :math:`\mathcal{N}_{i}` is the neighborhood of vertex :math:`v_{i}`. The new features at :math:`v_{i}` are obtained via linear combination of the original features and the normalized attention coefficients, wrapped in a nonlinearity :math:`\sigma`: :math:`h_{i}' = \sigma(\sum_{j \in \mathcal{N}_{i}} a_{ij} \textbf{W_{a}}h_{j})`. “Multi-head” attention can be used, yielding :math:`K` independent attention mechanisms that are concatenated (or averaged for the final layer). This helps to stabilize the self-attention learning process.

.. math::

    h_{i} = ||_{k=1}^{K} \sigma(\sum_{j \in \mathcal{N}_{i}} a_{ij}^{k} \textbf{W_{a}}^{k} h_{j}),

or

.. math::

    h_{final} = \sigma(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_{i}} a_{jk}^{k}\textbf{W_{a}}^{k} h_{j}).

We employ a PyTorch implementation [15]_ of :cite:`VCCRLB2018`'s :code:`GAT` class to implement a graph attention network, learning attention coefficients as

.. math::

    a_{ij} = \frac{exp(LeakyReLU(a^{T}[\textbf{W_{a}}h_{i}||\textbf{W_{a}}h_{j}]))}{\sum_{k \in \mathcal{N}_{i}} exp(LeakyReLU(a^{T}[\textbf{W_{a}}h_{i}||\textbf{W_{a}}h_{k}]))},

where :math:`||` is concatenation.

Multi-Subject Training
-------------------------

.. figure:: GCNetwork_fig.png

    A depiction of the novel GCN architecture is shown. First, a GCN is trained for each “view” of the data, corresponding to a specific tractography algorithm. The GCNs share weights, and the resulting features are pooled for each vertex. This composite graph is then used to train a multi-head graph attention network, which assigns a weight (i.e., “attention”) to the feature computed at each vertex. The weight assigned to each vertex is used to compute a weighted sum of the features, yielding a single feature vector for graph classification. :label:`GCNfig`

The model is trained using :code:`train.py`. First, several helper functions in :code:`utils.py` are called to load the graph, input signals, and their labels, and prepare them for training. The model is built and run using the :code:`GCNetwork` class in :code:`GCN.py`. During training, the model reads in the signals for one dMRI acquisition at a time, where the signals from each tractography algorithm are processed in parallel, pooled into one graph, and then pooled into a single feature vector via the graph attention network. Using this final feature vector, a class prediction is made. Once a class prediction is made for every input dMRI instance, the error is computed and the weights of the model are updated through backpropagation. This is repeated over many epochs to iteratively fit the weights to the classification task. Figure :ref:`GCNfig` shows an outline of the network architecture.

Methods
============

Our data is downloaded from the Parkinson's Progression Markers Initiative (PPMI) [16]_ database. We download 1,182 images, consisting of [] aMRI images and 619 diffusion images. The images are from 127 individuals (each subject had multiple visits to the clinic (VCs) and data from multiple image modalities). Among the images, [] are from the Parkinson's Disease (PD) group and [] are healthy controls (HC). We preprocessed our data using the pipeline described above. We typically used the default parameters for the commands from the various softwares. We ran this preprocessing on five Google cloud virtual machines in parallel over the course of several days. In total, 479 CPU cores were utilized.

Following preprocessing, we constructed the shared adjacency matrix and trained the model on the dMRI signals, which totaled to 2,476 (619 dMRI acquisitions x 4 tractography algorithms) :math:`N`x:math:`N` connectivity matrices. To account for the class imbalance between PD and HC images, we use a bagging method wherein all the images from the HC group are combined with equally-sized randomly sampled subsets of the PD group. All of the images are used at least once during training, and the overall performance measures are averaged across folds. Using an 80/20 train-test split, we trained each fold for [] epochs and took the mean and standard deviation of the performance measures on the test set from each fold. We calculated the overall accuracy, AUC score, and F1 score, and compare these to baseline methods, including [].

In a subsequent experiment, we separated the instances from the PD class according the clinic visit (CV) number. As mentioned, most patients visited the clinic a number of times, each visit separated by 6 months to a year. Using the CV number as a proxy for the progression of PD, we hypothesized that later visits were correlated with later-stage PD and that our model would be able to identify this disease progression. Using 5-fold cross validation, we first trained our model

As overfitting is a risk when training deep neural

Results
============

We are currently running our preprocessing pipeline on a much larger dataset with hundreds of images, and are working to incorporate fMRI data into our results as well. Additionally, we have stored the visit number for each acquisition from the full data set, and we will train our model to predict the visit number for PD patients as a proxy of the disease progression. We will report results as soon as possible over the coming days.

Discussions and Conclusions
===================================

We have presented here a complete pipeline for preprocessing multi-modal neuroimage data and training a novel graph-based deep learning model to perform inference on the data.

Acknowledgements
=========================

Data used in the preparation of this article were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org/data). For up-to-date information on the study, visit www.ppmi-info.org.
PPMI - a public-private partnership - is funded by the Michael J. Fox Foundation for Parkinson's Research and funding partners, including Abbvie, Allergan, Avid, Biogen, BioLegend, Bristol-Mayers Squibb, Colgene, Denali, GE Healthcare, Genentech, GlaxoSmithKline, Lilly, Lundbeck, Merck, Meso Scale Discovery, Pfizer, Piramal, Prevail, Roche, Sanofi Genzyme, Servier, Takeda, TEVA, UCB, Verily, Voyager, and Golub Capital.

.. raw:: latex

   \bibliographystyle{plain}

.. raw:: latex

    \newpage

Appendix A: MRI Modalities
=============================

The modality which serves as the basis for the nodes of the graphs is anatomical T1-weighted MRI (aMRI) data. This modality provides high resolution images which are quite useful for distinguishing different tissue types and region boundaries. The speed and relative simplicity of aMRI imaging results in fewer and less severe artifacts. For a given subject, images from the other modalities are often aligned to aMRI images, and this modality is often used to obtain brain masks (via skull stripping) and perform volumetric segmentation. Typical preprocessing includes motion-correction, intensity normalization, magnetic susceptibility correction, skull stripping, registration to a common brain atlas, and segmentation [:cite:`Wang2015,HBL2010`].

Diffusion-weighted MR imaging (dMRI) introduces additional noise sources. dMRI measures the diffusion of water molecules in the brain by applying pulsed magnetic field gradients in numerous directions, resulting in multiple 3D volumes for a single image. Typically, a higher resolution image (resembling anatomical images) is taken as the first volume, and is termed the *b0* volume. During processing, all other volumes are aligned to this volume. dMRI data is usually obtained using an MRI variant known as spin-echo echo planar imaging (EPI), which results in artifacts such as eddy currents and magnetic susceptibility artifacts. Typical preprocessing includes correcting these artifacts and co-registering the diffusion data to aMRI images of the same acquisition, for comparison to the aMRI data during analysis [:cite:`Wang2015,HBL2010`].

Once dMRI data is cleaned, the information can be processed to trace the directionality of water diffusion across voxels, forming connected paths between them. This process, called *tractography* estimates white matter (WM) tracts, which are bundles of nerve fibers, or axons, that connect regions of the brain. The specific tractography algorithm can significantly affect the analysis results, so we incorporate the output from four different tractography algorithms in our model.

Appendix B: Graph Convolutional Networks
===========================================

Given an undirected weighted graph :math:`\mathcal{G} = {\mathcal{V},\mathcal{E}, \textbf{W}}` with a set of vertices :math:`\mathcal{V}` with :math:`|\mathcal{V}| = N`, a set of edges :math:`\mathcal{E}`, and a weighted adjacency matrix **W**, we define a signal on the vertices as a function :math:`\mathcal{f} : \mathcal{V} \rightarrow \mathbb{R}`, returning a vector :math:`\textbf{f} \in \mathbb{R}^{N}` for each vertex. The vector *signal* defined on each vertex represents that vertex's weighted connectivity to all other vertices :cite:`SNFOV2013`.

We seek to learn filters :math:`g` over the graph, similar to the local filters used in convolutional neural networks. The discrete Fourier transform (FT) matrix of the normalized graph Laplacian :math:`\textup{\L{}}` provides a means for doing this. :math:`\textup{\L{}}` is a real symmetric matrix represented as

.. math::

    \textup{\L{}} = I - D^{\frac{-1}{2}} \textbf{W} D^{\frac{-1}{2}}

and with eigendecomposition :math:`\textup{\L{}} = U \Lambda U^{T}`, where :math:`D` is a diagonal matrix with entries :math:`D_{ii} = \sum_{j} \textbf{W}_{ij} = \textbf{W} \cdot \textbf{1} U`, :math:`U = (u_{1},...,u_{N})` is a complete set of orthonormal eigenvectors, and :math:`\Lambda` are the associated real, non-negative eigenvalues.

The graph FT :math:`\hat{\textbf{f}}` of any function :math:`f \in \mathbb{R}^{N}` on the vertices of :math:`\mathcal{G}` gives the expansion of :math:`f` in terms of the eigenvectors of :math:`\textup{\L{}}` :cite:`SNFOV2013`. This allows us to define functions :math:`f` and :math:`g`, which are both functions on the vertices of :math:`\mathcal{G}`, in terms of the eigendecomposition of the graph Laplacian of :math:`\mathcal{G}`.

The Convolution Theorem :cite:`M2009` defines a linear operator that diagonalizes in the Fourier domain as a convolution operator in the vector domain. Commuting :math:`\textup{\L{}}` with the translation operator produces such an operator :cite:`HBL2015` and can be used to convolve functions :math:`f` and :math:`g`.

We can now define a graph convolution of input signals :math:`x` with filters :math:`g` on :math:`\mathcal{G}` by

.. math::

    x*g_{\theta} = Ug U^{T}x,

where :math:`U` is the matrix of eigenvectors of :math:`\textup{\L{}}` given by the graph FT, and :math:`\theta` are the parameters we wish to learn. We consider :math:`g_{\theta}` as a function of the eigenvalues :math:`\Lambda`, :math:`g_{\theta}(\Lambda) = diag(\theta)`; thus the parameters :math:`\theta` are the Fourier coefficients from the graph FT on :math:`\textup{\L{}}` :cite:`KW2017`.

Finding these parameters are computationally expensive as multiplication with :math:`U` is :math:`O(N^{2})`, and :math:`\textup{\L{}}` itself may be quite expensive to calculate. So, an approximation is made in terms of Chebyshev polynomials :math:`T_{k}(x)` up to the :math:`K^{th}` order :cite:`HVG2011`. Chebyshev polynomials are recursively defined :math:`T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x)`, with :math:`T_{0}(x) = 1` and :math:`T_{1}(x) = x`. Now, :math:`g_{\theta}'(\Lambda) \approx \sum_{k=0}^{K} \theta_{k}'T_{k}(\tilde{\Lambda})`, where rescaled :math:`\tilde{\Lambda} = \frac{2}{l_{max}} \Lambda - I_{N}` and :math:`l_{max}` is the largest eigenvalue of :math:`\Lambda`. Defining :math:`\tilde{\textup{\L{}}} = \frac{2}{l_{max}} \textup{\L{}}-I_{N}`, we have

.. math::

    g_{\theta}' * x \approx \sum_{k=0}^{K} \theta_{k}'T_{k}(\tilde{\textup{\L{}}})x

:cite:`KW2017`.

The expression is :math:`K`-localized, relying only on nodes that are :math:`K`-steps away from a given node (its :math:`K^{th}`-order neighborhood). Evaluating such a function is :math:`O(\mathcal{E})`. By limiting :math:`K=1` we have a linear function with respect to :math:`\textup{\L{}}` as the preactivation :math:`\hat{H}` of our convolutional layer. Wrapping :math:`\hat{H}` in a nonlinear activation function and stacking multiple layers gives us our graph convolutional network architecture. This so-called deep learning architecture removes the rigid parameterization enforced by Chebyshev polynomials :cite:`KW2017`.

:cite:`KW2017` further approximate :math:`l_{max} \approx 2` and simplify the equation for :math:`\hat{H}` to :math:`g_{\theta}' * x \approx \theta_{0}'(x) + \theta_{1}'(\textup{\L{}} - I_{N})x = \theta_{0}'(x) - \theta_{1}' D^{\frac{-1}{2}}AD^{\frac{-1}{2}}x`, reducing the task to learning two free parameters which can be shared over the whole graph. If :math:`\theta_{0}'` is set equal to :math:`-\theta_{1}'`, then the equation can be expressed with a single parameter :math:`\theta = \theta_{0}'`:

.. math::

    g_{theta} * x \approx \theta(I_{N} + D^{\frac{-1}{2}}AD^{\frac{-1}{2}})x.

:math:`k` successive applications of this operator effectively convolve the :math:`k^{th}`-order neighborhood of a given node, but may also lead to numerical instabilities and the exploding/vanishing gradient problem, since :math:`I_{N}+ D^{\frac{-1}{2}}AD^{\frac{-1}{2}}` now has eigenvalues in [0,2]. :cite:`KW2017` solve this issue via a *renormalization trick* such that :math:`I_{N}+ D^{\frac{-1}{2}}AD^{\frac{-1}{2}}` becomes :math:`\tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}}`, where :math:`\tilde{A} = A+I_{N}` and :math:`\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}`. I.e., self-loops have been added to the adjacency matrix. The weights given to these connections should bear similar importance to the other connections, e.g., using the mean edge weight.

Finally, the equation is generalized to a signal :math:`X \in \mathbb{R}^{NxC}` with :math:`C`-dimensional feature vectors at every node (each *element* will learn a single parameter) and :math:`F` filters:

.. math::

    Z = \tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}} X \Theta,

where :math:`\Theta \in \mathbb{R}^{CxF}` are the parameters and :math:`Z \in \mathbb{R}^{NxF}` is the convolved signal matrix. This equation is of complexity :math:`O(|\mathcal{E}|FC)`. Generalizing :math:`X=H(l)` as the inputs to a layer, where :math:`H(0)` is the original data and :math:`\Theta` to the weight matrix :math:`\textbf{W}(l)` at a layer :math:`l`, we can calculate a hidden layer as

.. math::

    H(l+1) = \sigma(\tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}}H(l)\textbf{W}(l)).

The time complexity of computing a single attention mechanism is :math:`O(|\mathcal{V}|FF' + |\mathcal{E}|F')`, where :math:`F` is the number of input features and :math:`F'` is the number of output features.

.. [1] In this paper we use “anatomical MRI” to refer to standard *T1-weighted* (T1w) MR imaging. “T1 weighted” refers to the specific sequence and timing of magnetic pulses and radio frequencies used during imaging. T1w MRI is a common MR imaging procedure; the important thing to note is that T1 weighting yields high-resolution images which show contrast between different tissue types, allowing for segmentation of different anatomical regions.
.. [2] https://github.com/xtianmcd/ppmi_dl
.. [3] https://nifti.nimh.nih.gov
.. [4] https://www.dicomlibrary.com
.. [5] https://github.com/rordenlab/dcm2niix
.. [6] https://bids.neuroimaging.io
.. [7] We install the softwares to the home (`~`) directory due to permission issues when connect to Google cloud virtual machines via the `ssh` command. Freesurfer's setup does not automatically adapt to installation in the home directory, so several of its environment variables need to be hard coded. See the `setup` bash script provided for details.
.. [8] https://surfer.nmr.mgh.harvard.edu
.. [9] In the release notes, it is recommended for multi-subject pipelines to use a single core per image and process subjects in parallel, and in the forums it is discussed that multiprocessing may only reduce the processing time to around 10 hours. It is also mentioned that the time required to transfer data on and off GPU cores may diminish the speedup provided by GPU processing. GPU support has not been provided by Freesurfer for quite some time, and we were unable to compile Freesurfer to use newer versions of CUDA. We tested multiple CPU multiprocessing approaches and found that running images in parallel with a single core per process was the fastest method.
.. [10] https://fsl.fmrib.ox.ac.uk/fsl/fslwiki
.. [11] Each subject has anatomical and diffusion MRI data for varying numbers of visits to the clinic. We use “clinic visit” or CV to refer to the MRI acquisitions (anatomical and diffusion) obtained during a single visit to the clinic.
.. [12] http://brainsuite.org
.. [13] http://trackvis.org/dtk/
.. [14] https://github.com/tkipf/pygcn
.. [15] https://github.com/Diego999/pyGAT
.. [16] https://www.ppmi-info.org
