:author: Christian McDaniel
:email: clm121@uga.edu
:institution: University of Georgia

:author: Shannon Quinn, PhD
:email: spq@uga.edu
:institution: University of Georgia
:bibliography: bib

---------------------------------------------------------
Developing a Graph Convolution-Based Analysis Pipeline for Multi-Modal Neuroimage Data: An Application to Parkinson's Disease
---------------------------------------------------------

.. class:: abstract

Parkinson’s disease (PD) is a highly prevalent neurodegenerative disease originating in subcortical areas of the brain and resulting in progressively worsening motor, cognitive, and psychological symptoms. Whereas clinical data tends to be somewhat arbitrary and genetic data inconclusive for many patients, neuroimaging data is an attractive research tool given the neurophysiological origins of the disease.

Despite the insights potentially available in magnetic resonance imaging (MRI) data, and the noninvasive and low-risk nature thereof, developing sound analytical techniques for this data has proven difficult. Principally, multiple imaging modalities are needed to provide clinicians with the most accurate view possible; the process of incorporating multiple image modalities into a single holistic model is both poorly defined and extremely challenging. Previous work has demonstrated that graph-based signal processing techniques can address this issue, while also drastically reducing the size and complexity of the data. Unfortunately, details for preprocessing and reformatting this data remain informal and incomplete, impeding the pace of advancement in the field and the reproduction of results.

In this paper, we propose a novel graph-based convolutional neural network architecture and present an end-to-end pipeline for preprocessing, formatting, and analyzing this data. On data downloaded from the Parkinson's Progression Markers Initiative (PPMI) database, we *... fill in RESULTS here.*


.. class:: keywords

Introduction
------------
Parkinson's disease (PD) affects approximately 1% of the United States population over the age of 60, a demographic expected to increase in the immediate future. The diagnosis of Parkinson’s disease has traditionally relied on clinical assessments with some degree of subjectivity, making diagnosing a specific stage of PD or differentiating between PD and similar conditions a somewhat arbitrary task [4]. As such, many efforts have emerged to identify quantitatively rigorous methods through which to distinguish PD.

While genetic and molecular biomarkers have exhibited some efficacy in developing a PD blueprint [4,10], many research efforts have turned to neuroimaging due to its noninvasive nature and alignment with existing knowledge of the disease. Namely, PD is a disease of the nigrostriatal dopaminergic pathway of the brain, and has been shown to result in various structural and functional abnormalities that can be captured by existing imaging modalities [14,10,GLHSA2014] .

While powerful machine learning techniques have been employed for analyzing neuroimaging data, algorithmic differences can result in vastly different results. [] found that when performing white matter (WM) tractography on diffusion-weighted MRI (dMRI) data, the choice of algorithm played as much a role as anatomical differences themselves. Furthermore, insights from a single image modality do not provide a full picture of the disease, while analyzing the disparate data types together is both poorly defined and extremely challenging.

Previous work has shown that graph-based signal processing techniques can address these issues. By representing the anatomical regions of the brain (from T1-weighted images) as nodes on a graph, the functional and structural relationships between the nodes can be defined in the graph space. The weighted connections between nodes are averaged across all subjects' anatomical T1-weighted MRI (T1w) data, mitigating the effect of inter-subject anatomical variability. For a given acquisition, diffusion and functional data are transformed from 4-dimensional sequences of volumes to simple one-dimensional vectors on each node (i.e., node features or signals on the nodes), drastically reducing the size and complexity of the data.

Intuitively, the graph-based representation has additional benefits over a standard image grid. The graphs group individual voxels into localized anatomical regions and characterize the structural and functional connections between them. This reduces noise from individual voxels and addresses the nonlinear relationship between image features and disease. For example, the fact that one subject's substantia nigra (SN), the dopamine-producing nucleus involved in PD, is smaller than the mean SN does not directly indicate the disease state of that subject. Additionally, the probability of connection between two regions is not a function of the distance between them, and connections often do not follow the shortest path between two regions. The edges of the graph offer an explicit metric for establishing meaningful neighborhoods of nodes and the features offer meaningful descriptors through which to group them.

Common to many areas of data analysis, spectral graph processing techniques have capitalized on the highly flexible and nonlinear calculations characteristic of so-called deep learning neural network architectures. The layered construction of nonlinear calculations loosens rigid parameterizations of other methods, such as the number of components used in independent component analysis (ICA) or explicit parameterizations required for a specific expansion function (e.g., Chebyshev polynomials, discussed in the Background section).

The graph convolutional network (GCN) defines a convolution on a graph in the spectral domain, using features defined on the nodes of the graph to learn filters - similar to standard convolutional neural networks - for node classification. Subsequent development of attention-based mechanisms allowed for a weighting of each node based on its significance, facilitating whole-graph classifications. In this paper we propose a novel architecture which first generates features for the nodes of the graph, pooling information from multiple dMRI tractography algorithms (and functional data??), and then performs whole-graph classification using graph attention network (GAT) layers.

We also seek to discuss the difficult challenge of processing neuroimaging data and preparing it for these models. Often, preprocessing is not performed prior to releasing data for research purposes, and it is left to the researcher to carry out. Many techniques exist, and many of the decisions along the way require a technical understanding of the imaging process and preprocessing steps. This is a major barrier to reproducibility and prevents data analysts from applying their skills in this domain. More work is needed to automate the procedure and provide better documentation for steps that need case-specific input. To that end, we have discuss our findings and methods below, and our code is available on github [add link in footnote].

Background
-------------
*MRI Acquisition and Data Preprocessing*
MRI data requires extensive artifact correction and removal before it can be used. MRI signals are acquired through the application of precisely coordinated magnetic fields and radiofrequency (RF) pulses. Each image is reconstructed from a series of recordings averaged over many individual signals. This inherently results in noisy measurements, magnetic-based artifacts, and artifacts from human error such as motion artifacts.

Before preprocessing, images should be converted to the Neuroimaging Informatics Technology Initiative (NIfTI) file format. Whereas many MRI data are initially in the Digital Information and Communications in Medicine (DICOM) format for standardized transfer of medical data and metadata, the NIfTI format is structured for ease of use when conducting computational analysis and processing on these files. The size, orientation, and location in space of the voxel data is dependent on settings used during image acquisition and requires an affine matrix to relate two images in a standard coordinate space. The NIfTI file format automatically associates each image with an affine matrix as well as a header file, which contains other helpful metadata.

Next, it is common practice to convert your data file structure to the Brain Imaging Data Structure (BIDS) format. Converting data to the BIDS format is required by certain softwares, and ensures a standardized and intuitive file structure.

The modality which serves as the basis for the nodes of the graphs is anatomical T1-weighted MRI (T1w) data. This modality provides high resolution images which are quite useful for distinguishing different tissue types and region boundaries. The speed and relative simplicity of T1w imaging results in fewer and less-severe artifacts. For a given subject, images from the other modalities are often aligned to T1w images, and this modality is often used to obtain brain masks (skull stripping) and perform volumetric segmentation. Typical preprocessing includes motion-correction, intensity normalization, magnetic susceptibility correction, skull stripping, registration to a common brain atlas, and segmentation.

Diffusion-weighted MR imaging (dMRI) introduces additional noise sources. dMRI measures the diffusion of water molecules in the brain by applying pulsed magnetic field gradients in numerous directions, resulting in multiple 3D volumes for a single image. Typically, an anatomical (e.g., T1w) image is taken as the first volume, and is termed the *b0* volume. During processing, all other volumes are aligned to this volume. dMRI data is usually obtained using an MRI variant known as spin-echo echo planar imaging (EPI), which results in artifacts such as eddy currents and magnetic susceptibility artifacts. Typical preprocessing includes correcting these artifacts and co-registering the diffusion data to T1w images of the same acquisition, for comparison to the T1w data during analysis.

Once dMRI data is cleaned, the information can be processed to trace the directionality of water diffusion across voxels, forming connected paths between them. This process, called *tractography* estimates white matter (WM) tracts, which are bundles of nerve fibers, or axons, that connect regions of the brain.

*If functional data is used, insert info re: fMRI acquisition/ preprocessing*

*Defining Graph Nodes and Features*
The underlying graph for the GCN is a weighted adjacency matrix sharing information from each acquisition’s T1w data. The matrix is generated by first computing the mean center voxel coordinate for each segmentation volume, averaged over all T1w acquisitions. A k-nearest neighbor (k-NN) adjacency matrix is then formed using these coordinates, and the edges between each node to its k nearest neighbors are weighted by the normalized distance to that neighbor. The values are normalized by dividing each distance by the maximum distance for a given node to all of its neighbors. Finally, self-loops are added for each node, given an edge weight equal to the mean of the edge weights for that node. (Refer to the Graph Convolutional Network subsection below for details.)

Each acquisition shares the same adjacency matrix and is distinguished by the features defined on its nodes. Each feature is a sparse vector representing the relative number of tracts connecting a given node to all other nodes, as calculated by a given tractography algorithm. As such, each acquisition has a set of features for each of the tractography algorithms used, and all the features for a given tractography algorithm can be viewed as a connectivity matrix, whose weights represent the estimated strength of connectivity between neighbors.

*Graph Convolutional Networks*
Neuroimaging data is readily applied to graph processing techniques and is often used as a benchmark application for new developments in the field (Shuman et al 2013). Intuitively, the objective is to establish localized anatomical regions and characterize the structural and functional connections between them. As such, given an undirected weighted graph :math:`\mathcal{G} = {\mathcal{V},\mathcal{E}, **W**}` with a set of vertices :math:`\mathcal{V}` with :math:`|\mathcal{V}| = N`, a set of edges :math:`\mathcal{E}`, and a weighted adjacency matrix **W**, we define a signal on the vertices as a function :math:`\mathcal{f} : \mathcal{V} \rightarrow \mathbb{R}`, returning a vector :math:`\textbf{f} \in \mathbb{R}^{N}`. I.e., the signal on the vertices is comprised of a given vertex’s weighted connectivity to all other vertices (Shuman et al).

We seek to learn filters over the graph, similar to the local filters used in convolutional neural networks. The discrete Fourier transform (FT) matrix of the normalized graph Laplacian :math:`\L{}` provides a means for doing this. :math:`\L{}` is a real symmetric matrix represented as
.. math::
\L{} = I - D^{\frac{-1}{2} \textbf{W} D^{-1}{2}
and with eigendecomposition :math:`\L{} = U \Lambda U^{T}`, where :math:`D` is a diagonal matrix with entries :math:`D_{ii} = \sum_{j} \textbf{W}_{ij} = \textbf{W} \cdot \textbf{1} U`, :math:`U = (u_{1},...,u_{N})` is a complete set of orthonormal eigenvectors and :math:`\Lamda` are the associated real, non-negative eigenvalues.

The graph Fourier transform :math:`\hat{\textbf{f}}` of any function :math:`\mathcal{f} \in \mathbb{R}^{N}` on the vertices of :math:`\mathcal{G}` gives the expansion of :math:`\mathcal{f}` in terms of the eigenvectors of :math:`\L{}` (Shuman et al). Given the Convolution Theorem [Mallat 1999] definition of a convolution as a linear operator that diagonalizes in the Fourier domain, commuting :math:`\L{}` with the translation operator produces such an equation (Henaff 2015) and can be used as a convolution operation on graph data.

We can now define a graph convolution of input signals :math:`x` with filters :math:`g_{\theta}` on :math:`\mathcal{G}` by
.. math::
x*g_{\theta} = Ug U^{T}x,
where :math:`U` is the matrix of eigenvectors of :math:`\L{}` given by the graph FT. We wish to learn the parameters :math:`theta` in :math:`g_{\theta)`. We consider :math:`g_{\theta}` as a function of the eigenvalues :math:`\Lambda`, :math:`g_{\theta}(\Lambda) = diag(\theta)`; thus the parameters :math:`\theta` are the Fourier coefficients from the graph FT on :math:`\L{}`. (Kipf and Welling 2017)

Finding these parameters are computationally expensive as multiplication with :math:`U` is :math:`O(N^{2})`, and :math:`\L{}` itself may be quite expensive to calculate. So, an approximation is made in terms of Chebyshev polynomials :math:`T_{k}(x)` up to the :math:`K^{th}` order (Hammond et al 2011). Now, :math:`g_{\theta}’(\Lambda) \approx \sum_{k=0}^{K} \theta_{k}’T_{k}(\tilde{\Lambda})`, where rescaled :math:`\tilde{\Lambda} = \frac{2}{l_{max}} \Lambda - I_{N}` and :math:`l_{max}` is the largest eigenvalue of :math:`\Lambda`. Chebyshev polynomials are recursively defined :math:`T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x), with T_{0}(x) = 1 and T_{1}(x) = x. Defining :math:`\tilde{\L{}} = \frac{2}{l_{max}} \L{}-I_{N}`, we now have
.. math::
g_{\theta}’ * x \approx \sum_{k=0}^{K} \theta_{k}’T_{k}(\tilde{\L{}})x
(Kipf and Welling 2017).

The expression is :math:`K`-localized, relying only on nodes that are :math:`K`-steps away from a given node (its :math:`K^{th}`-order neighborhood). Evaluating such a function is :math:`O(\mathcal{E})`. By limiting :math:`K=1` we have a linear function with respect to :math:`\L{}` as the preactivation :math:`\hat{H}` of our convolutional layer. Wrapping :math:`\hat{H}` in a nonlinear activation function and stacking multiple layers gives us our graph convolutional network architecture. This so-called deep learning architecture removes the rigid parameterization enforced by Chebyshev polynomials (Kipf and Welling 2017).

(Kipf and Welling 2017) further approximate :math:`l_{max} \approx 2` and simplify the equation for :math:`\hat{H}` to :math:`g_{\theta}’ * x \approx \theta_{0}’(x) + theta_{1}’(\L{} - I_{N})x = theta_{0}’(x) - theta_{1}’ D^{\frac{-1}{2}}AD^{\frac{-1}{2}}x`, reducing the task to learning two free parameters which can be shared over the whole graph. If :math:`\theta_{0}’` is set equal to :math:`-\theta_{1}’`, then the equation can be expressed with a single parameter :math:`\theta = \theta_{0}’`:
.. math::
g_{theta} * x \approx \theta(I_{N} + D^{\frac{-1}{2}}AD^{\frac{-1}{2}})x.

:math:`k` successive applications of this operator effectively convolve the :math:`k^{th}`-order neighborhood of a given node, but may also lead to numerical instabilities and the exploding/vanishing gradient problem, since :math:`I_{N}+ D^{\frac{-1}{2}}AD^{\frac{-1}{2}}` now has eigenvalues in [0,2]. Kipf and Welling 2017 solve this issue via a *renormalization trick* such that :math:`I_{N}+ D^{\frac{-1}{2}}AD^{\frac{-1}{2}}` becomes :math:`\tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}`, where :math:`\tilde{A} = A+I_{N}` and :math:`\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}`. I.e., self-loops have been added to the adjacency matrix. The weights given to these connections should bear similar importance to the other connections, e.g., using the mean edge weight.

Finally, the equation is generalized to a signal :math:`X \in \mathbb{R}^{NxC}` with :math:`C`-dimensional feature vectors at every node (each *element* will learn a single parameter) and :math:`F` filters:
.. math::
Z = \tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}} X \Theta`,
where :math:`\Theta \in \mathbb{R}^{CxF}` are the parameters and :math:`Z \in \mathbb{R}^{NxF}` is the convolved signal matrix. This equation is of complexity :math:`O(|\mathcal{E}|FC)`. Generalizing :math:`X=H(l)` as the inputs to a layer, where :math:`H(0)=X` to the weight matrix :math:`\textbf{W}(l)` at a layer :math:`l`, we can calculate a hidden layer as
.. :math::
H(l+1) = \sigma(\tilde{D}^{\frac{-1}{2}}\tilde{A}\tilde{D}^{\frac{-1}{2}}H(l)\textbf{W}(l)).

*Multi-View Pooling*
For each dMRI acquisition, *d* different tractography algorithms are used to compute multiple “views” of the diffusion data. To account for the variability in the outputs produced by each algorithm, we wish to compile the information from each before classifying the whole graph. As such, a GCN is trained for each algorithm. Each GCN shares weights [Koch et al 2015, Ktena et al 2017] and outputs the same number of features. The features from each GCN are pooled using max pooling, which has been shown to outperform mean pooling [Zhang et al 2018]. The final pooled vector is then passed to a graph attention network (GAT) to obtain an informed combination of the nodes for whole-graph classification.

*Graph Attention Networks*
In order convert the task from classifying each node to classifying the whole graph, the features on each node must be pooled to generate a single feature vector for a given graph. The *self-attention* mechanism, widely used to compute a single representation of a signal sequence, has been used to effectively compute the importance of graph nodes in a neighborhood (Velikcovic et al 2018). This allows for a weighted sum of the nodes’ features during pooling.

Velikcovic et al 2018 use a single-layer feedforward neural network as an attention mechanism :math:`a` to compute *attention coefficients e* across pairs of nodes in a graph. For a given node *i*, the attention mechanism attends over the first-order neighbors *j* of node *i* using the nodes’ features :math:`h_{i}` and :math:`h_{j}`: :math:`e_{ij} = a(\textbf{W}h_{i}, \textbf{W}h_{j}), where \textbf{W} is a shared weight matrix applied to each node’s features. :math:`e_{ij}` is normalized via the softmax function to compute :math:`a_{ij}`: :math:`a_{ij} = softmax(e_{ij}) = exp(e_{ij}) / \sum_{k \in \mathcal{N}_{i} exp(e_{ik}), where :math:`\mathcal{N}_{i}` is the neighborhood of node*i*. The new features at node *i* are obtained via linear combination of the original features and the normalized attention coefficients, wrapped in a nonlinearity :math:`sigma`: :math:`h_{i}’ = \sigma(\sum_{j \in \mathcal{N}_{i}) a_{ij} \textbf{W}h_{j}`. Multi-head attention can be used, yielding :math:`K` independent attention mechanisms that are concatenated (or averaged for the final layer). This helps to stabilize the self-attention learning process.
.. math::
h_{i} = ||_{k=1}^{K} \sigma(\sum_{j \in \mathcal{N}_{i} a_{ij}^{k} \textbf{W}^{k} h_{j},
or
h_{final} = \sigma(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_{i} a_{jk}^{k} \textbf{W}^{k} h_{j})`. The time complexity of computing a single attention mechanism is :math:	O(|\mathcal{V}|FF’ + |\mathcal{E}|F’), where :math:`F` is the number of input features and :math:`F’` is the number of output features.

*Multi-Subject Training*
GCNs were originally used to classify the nodes of a single graph using a single set of features defined on its nodes. Instead, our task is to learn features that generalize over many subjects’ data. To incorporate information from each acquisition, a single complete forward pass - consisting of multi-view GCN, max pooling, GAT - is conducted for every acquisition. A class prediction (e.g., Parkinson’s disease or Healthy control) is made for each forward pass output and the loss is calculated after all acquisitions have been used as input. Thus, a single epoch sees all acquisitions in the training set before weight updates are made.

Related Works
-------------

Powerful machine learning techniques have been employed for neuroimage data analysis and have been shown to perform quite well [14,10,8,2,3]. As concerns have arisen over limitations of these algorithms [2,3,8,14], there have been many applications of deep machine learning to neuroimage data analysis. For example, [7] proposes a 3D convolutional neural network (CNN) for skull stripping 3D brain images, [6] proposes a novel recurrent neural network plus independent component analysis (RNN-ICA) model for fMRI analysis, and [5] demonstrates the efficacy of the restricted Boltzmann machine (RBM) for network identification. LZCY2017 offer a comprehensive review of deep learning-based methods for medical image computing in general. To narrow our review, we will focus on the body of works that have used graph-based analysis for neuroimage data.

Many results have already been shared regarding the mathematical background of graph convolutional networks (GCNs) and graph attention networks (GATs). Principally, this paper is based on the advancements made by Kipf and Welling (2017) and Velickovic et al (2018) on GCNs and GATs respectively. Shuman et al, in addition to providing in-depth intuition behind spectral graph processing, demonstrate graph spectral filtering on diffusion signals defined a cerebral cortex graph. [Koch et al (2015), Kumar et al (2016), Zhang et al (2018)] develop siamese and multi-view neural networks which share weights across parallel neural networks for classifying objects based on multiple “views” or angles. These architectures group examples into pairs and train networks to classify the pairs as being from the same group or different groups. Ktena et al (2017, 2018) apply these techniques to learn similarity metrics between subjects with Autism Spectrum Disorder (ASD) and healthy controls (HC), using fMRI data from the Autism Brain Imaging Data Exchange (ABIDE) database. Zhang, et al (2018) apply a similar architecture to learn similarity metrics between subjects with Parkinson’s disease (PD) and HC, using dMRI data from the PPMI data set.

Methods
-------------
Our data is downloaded from the Parkinson’s Progression Markers Initiative (PPMI) database. We download ___ images for ___ individual acquisitions consisting of  ___ T1w images, ___ diffusion images (and ___ functional images). Among the acquisitions, ___ are from the Parkinson’s Disease (PD) group and ___ are healthy controls (HC). We preprocess the data and construct our novel GCN architecture as follows.

*Preprocessing*
The software `dcm2niix` is helpful for converting the data from its original DICOM format to the usable NIfTI format. We implement this conversion in `neuro-format.py`. We then reformat our data file structure to the BIDS format. There exist some readily available programs for doing this, but the file structure used by PPMI is quite nuanced, so we wrote our own function to do so in `make_bids.py`.

The standard software for T1w MRI data preprocessing is Freesurfer. Freesurfer is an actively developed software with responsive technical support and rich forums. The software is dense and the documentation is lacking in some aspects, so training may still be helpful, although not available in our case. The `recon-all` command performs all the steps needed for standard T1w preprocessing, including motion correction, registration to a common coordinate space using the Talairach atlas by default, intensity correction and thresholding, skull-stripping, region segmentation, surface tessellation and reconstruction, statistical compilation, etc.

The entire process takes around 15 or more hours per image. Although support for GPU-enabled processing was stopped years ago, the `-openmp <num_cores>` command allows parallel processing across the designated number of cores and can reduce the processing time to around 4-5 hours. However, in the release notes, it is recommended for multi-subject pipelines to use a single core per image and process subjects in parallel. We could not find any explicit comparison of the time requirements for each method. For segmentation, the Deskian/Killiany atlas is used, resulting in around 115 volume segmentations per image, to be used as the nodes for the graph.

The Functional Magnetic Resonance Imaging of the Brain (FMRIB) Software Library (FSL) is often used to preprocess diffusion data. The b0 volume is isolated (`fslroi`) and merged with other runs from the same subjects (`fslmerge`), the brain is isolated from the skull (skull stripped, `bet` with the help of `fslmaths -Tmean`), susceptibility correction is performed for specific cases using `topup` and eddy correction is performed using `eddy_openmp`.

The `topup` tool requires two or more acquisitions for a given subject, where the header parameters `TotalReadoutTime` and/or `PhaseEncodingDirection` differ from one another. Since the multiple acquisitions for a given subject typically span different visits to the clinic, the same parameters are often used and `topup` cannot be utilized.

We found another software, BrainSuite, which can perform susceptibility correction using a single acquisition. Although we still include FSL in our pipeline since it is a baseline in many other papers, we employ the BrainSuite software's Brain Diffusion Pipeline to perform susceptibility correction and to register the corrected dMRI data to the anatomical T1w data for a given subject.

First, a BrainSuite compatible brain mask is obtained using `bse`. Next, `bfc` is used for bias field (magnetic susceptibility) correction, and finally `bdp` performs co-registration of the diffusion data to the T1w image for the same subject. The calls to the Freesurfer, FSL, and BrainSuite software libraries are included in `automate_preproc.py`.

There are many algorithms and softwares that perform tractography, but we found that many researchers use the Diffusion Toolkit (DTK) in their experiments. In `dtk.py` we employ four different diffusion tensor imaging (DTI)-based deterministic tractography algorithms: Fiber Assignment by Continuous Tracking (FACT; Mori et al., 1999), the second-order Runge–Kutta method (RK2; Basser et al., 2000), the tensorline method (TL; Lazar et al., 2003), and the interpolated streamline method (SL). Zhan et al 2015 provides a more information on each method. `dti_recon` first transforms the output file from Brainsuite into a usable format for DTK, and then `dti_tracker` is called for each of the tractography algorithms. Finally, `spline_filter` is used to smooth the generated tracts, denoising the outputs.

*Graph Formation*
Now that the images are processed, they can be efficiently loaded using python libraries `nibabel` and `dipy`, and subsequently operated on using standard data analysis packages such as `numpy` and `scipy`.

`gen_nodes.py` uses the segmented T1w images to calculate the center voxel for each segmentation volume. Next, `adj_mtx.py` calculates the mean voxel coordinate for every volume across all acquisitions and forms the weighted adjacency matrix. See Figure :ref:`adj_mtx` 1 for a depiction of the process.

.. figure:: adj_mtx_fig.png

A depiction of the steps involved in forming the adjacency matrix. First, anatomical images from each acquisition are segmented into regions of interest (ROIs), which represent the vertices of the graph. The center voxel for each ROI is then calculated. An edge is placed between each node *i* and its *k*-nearest neighbors, calculated using the center coordinates.  Lastly, each edge is weighted by the normalized distance (:math:`d_{ij} \in [0,1]`) between each node *i* and its connected neighbor *j*.

`gen_features.py` uses Freesurfer’s 'mri_convert', FSL's `flirt`, and DTK's `track_transform` to co-register the final tractography outputs to the cleaned T1w images for each acquisition. Next, `nibabel` is used to generate a mask file for each segmentation volume, `nibabel.streamlines` is used to read in the tractography data and `dipy.tracking.utils.target` is used to identify which tracts travel through each volume mask. The tracts are encoding using a unique hashing function for later identification. To generate the features for each node, `utils.py` uses the encoded tract ID's assigned to each volume to count the number of tracts connecting each volume pair, and the connections are normalized by the maximum number of connections for a given node.

*Graph Convolutional Network*
The `GCN` class from Kipf and Welling 2017 implements a two layer graph convolutional network as
.. math::
  Z = f(X,A) = softmax(\hat{A} ReLU(\hat{A}X\textbf{W}(0))\textbf{W}(1)),
where :math:`\hat{A} = \tilde{D}^{\frac{-1}{2}\tilde{A}\tilde{D}^{\frac{-1}{2}`. We tweak this to use the tanh activation function instead of ReLU. (** compare to ReLU, may want to keep ReLU** ). Next, we implement Velikcovic et al’s `GAT` class to implement a graph attention network, learning attention coefficients as
.. math::
  a_{ij} = exp(LeakyReLU(a^{T}[\textbf{W}h_{i}||\textbf{W}h_{j}])) / \sum(k \in \mathcal{N}_{i}) exp(LeakyReLU(a^{T}[\textbf{W}h_{i}||\textbf{W}h_{k}])),
where :math:`||` is concatenation.

`GCN.py` contains these and helper classes as well as our GCNetwork class, which implements the multi-view GCN on the features derived from multiple tractography algorithms (and function data?), pools the multi-view features and calls the GAT class on the pooled data. The weighted attention assigned to each node’s feature is used to compute a weighted average across all the nodes’ output feature (of the same size as the number of classes). Figure :ref:`GCNetwork_fig` C shows an outline of the network architecture.  Finally `train.py` trains the network. For a given epoch, the network computes a forward pass on all acquisitions, calculates and backpropagates the loss using all the predictions, and updates the weights accordingly.

.. figure:: GCNetwork_fig.png

	A depiction of the novel GCN architecture is shown. First, a GCN is trained for each “view” of the data, corresponding to a specific tractography algorithm. The GCN shares weights, and the resulting features are pooled for each node. This composite graph is then used to train a multi-head graph attention network, which outputs features that have the same size as the number of classes. The attention weight assigned to each node is used to compute a weighted sum of each feature, yielding the predicted class :math:`\hat{y}` of the input acquisition.

Results
------------
Accuracy, F1score
ROC curve
Baseline comparisons

Discussions and Conclusions
-------------

Acknowledgements
--------------
Data used in the preparation of this article were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org/data). For up-to-date information on the study, visit www.ppmi-info.org.
PPMI - a public-private partnership - is funded by the Michael J. Fox Foundation for Parkinson's Research and funding partners, including [list the full names of all of the PPMI funding partners found at www.ppmi-info.org/fundingpartners].
