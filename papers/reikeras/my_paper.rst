.. role:: raw-latex(raw)
    :format: latex html

:author: Helge Reikeras
:email: helge@ml.sun.ac.za
:Institution: Stellenbosch University

:author: Ben Herbst
:email: herbst@sun.ac.za
:institution: Stellenbosch University

:author: Johan Du Preez
:email: dupreez@sun.ac.za
:institution: Stellenbosch University

:author: Herman Engelbrecht
:email: hebrecht@sun.ac.za
:institution: Stellenbosch University


-------------------------------------------
Audio-Visual Speech Recognition using SciPy
-------------------------------------------

.. class:: abstract

In audio-visual automatic speech recognition (AVASR) both acoustic and visual modalities of speech are used to identify what a person is saying. In this paper we propose a basic AVASR system implemented using SciPy, an open source Python library for scientific computing.  AVASR research draws from the fields of signal processing, computer vision and machine learning, all of which are active fields of development in the SciPy community. As such, AVASR researchers using SciPy are able to benefit from a wide range of tools available in SciPy.

The performance of the system is tested using the Clemson University audio-visual experiments (CUAVE) database. We find that visual speech information is in itself not sufficient for automatic speech recognition. However, by integrating visual and acoustic speech information we are able to obtain better performance than what is possible with audio-only ASR. 

Introduction
============

Motivated by the multi-modal manner humans perceive their environment, research in Audio-Visual Automatic Speech Recognition (AVASR) focuses on the integration of acoustic and visual speech information with the purpose of improving accuracy and robustness of speech recognition systems. AVASR is in general expected to perform better than audio-only automatic speech recognition (ASR), especially so in noisy environments as the visual channel is not affected by acoustic noise. 

Functional requirements for an AVASR system include acoustic and visual feature extraction, probabilistic model learning and classification. 

In this paper we propose a basic AVASR system implemented using  SciPy [1]_. In the proposed system mel-frequency cepstrum coefficients (MFCCs) and active appearance model (AAM) parameters are used as acoustic and visual features, respectively. Gaussian mixture models are used to learn the distributions of the feature vectors given a particular class such as a word or a phoneme. We present two alternatives for learning the GMMs, expectation maximization (EM) and variational Bayesian (VB) inference.

The performance of the system is tested using the CUAVE database. The performance is evaluated by calculating the misclassification rate of the system on a separate test data data.  We find that visual speech information is in itself not sufficient for automatic speech recognition. However, by integrating visual and acoustic speech we are able to obtain better performance than what is possible with audio-only ASR. 

.. [1] SciPy is available at http://www.scipy.org/

Feature extraction
==================

Acoustic speech
---------------

MFCCs are the standard acoustic features used in most modern speech recognition systems. In [Dav80]_ MFCCs are shown experimentally to give better recognition accuracy than alternative parametric representations.

MFCCs are calculated as the cosine transform of the logarithm of the short-term energy spectrum of the signal expressed on the mel-frequency scale. The result is a set of coefficients that approximates the way the human auditory system perceives sound. 

MFCCs may be used directly as acoustic features in an AVASR system. In this case the dimensionality of the feature vectors equals the number of MFCCs computed. Alternatively, velocity and acceleration information may be included by appending first and second order temporal differences to the feature vectors.

The total number of feature vectors obtained from an audio sample depends on the duration and sample rate of the original sample and the size of the window that is used in calculating the cepstrum (a windowed Fourier transform).

MFCCs are available in the ``scikits.talkbox.features.mfcc``. The default number of MFCCs computed is thirteen.

Example usage::

    from scikits.audiolab import wavread
    from scikits.talkbox.features import mfcc

    # data: raw audio data
    # fs: sample rate
    data, fs = wavread('sample.wav')[:2]

    # ceps: cepstral cofficients
    ceps = mfcc(input, fs=fs)[0]

Figure :raw-latex:`\ref{fig:mfcc}` shows the original audio sample and mel-frequency cepstrum for the word \`zero'.

.. raw:: latex

    \begin{figure}

        \begin{center}
        \includegraphics[width=\textwidth]{mfcc.pdf}
        \end{center}
        \caption{Acoustic feature extraction from an audio sample of the word `zero'. Mel-frequency cepstrum (top) and original audio sample (bottom).}
        \label{fig:mfcc}
    \end{figure}

Visual speech
-------------
While acoustic speech features can be extracted through a sequence of transformations applied to the input audio signal, extracting visual speech features is in general more complicated. The visual information relevant to speech is mostly contained in the motion of visible articulators such as lips, tongue and jaw. In order to extract this information from a sequence of video frames it is advantageous to track the complete motion of the face and facial features.

AAM [Coo98]_ fitting is an efficient and robust method for tracking the motion of deformable objects in a video sequence. AAMs model variations in shape and texture of the object of interest. To build an AAM it is necessary to provide sample images with the shape of the object annotated. Hence, in contrast to MFCCs, AAMs require prior training before being used for tracking and feature extraction.

The shape of an appearance model is given by a set of :raw-latex:`$(x,y)$` coordinates represented in the form of a column vector

.. raw:: latex

    \begin{equation}
        \mathbf{s} = (x_{1},y_{1},x_{2},y_{2},\ldots,x_{n},y_{n})^{\mathrm{T}}.
    \end{equation}

The coordinates are relative to the coordinate frame of the image.

Shape variations are restricted to a base shape :raw-latex:`$\mathbf{s}_{0}$` plus a linear combination of a set of :raw-latex:`$N$` shape vectors

.. raw:: latex

    \begin{equation}
        \mathbf{s}=\mathbf{s}_{0} + \sum_{i=1}^{N}p_{i}\mathbf{s}_{i}
    \end{equation}

where :raw-latex:`$p_{i}$` are called the shape parameters of the AAM.

The base shape and shape vectors are normally generated by applying principal component analysis (PCA) to a set of manually annotated training images. The base shape :raw-latex:`$\mathbf{s}_{0}$` is the mean of the object annotations in the training set, and the shape vectors are :raw-latex:`N` singular vectors corresponding to the :raw-latex:`N` largest singular values of the data matrix (constructed from the training shapes). Figure :raw-latex:`\ref{fig:shape}` shows an example of a base mesh and the first three shape vectors corresponding to the three largest singular values of the data matrix.

.. raw:: latex
    
    \begin{figure}
        \begin{center}
        \includegraphics[width=\columnwidth]{shapes.pdf}
        \end{center}
        \caption{Triangulated base shape $\mathbf{s}_{0}$ (top left), and first three shape vectors $\mathbf{p}_{1}$ (top right), $\mathbf{p}_{2}$ (bottom left) and $\mathbf{p}_{3}$ (bottom right) represented by arrows superimposed onto the triangulated base shape.}
        \label{fig:shape}
    \end{figure}
..

The appearance of an AAM is defined with respect to the base shape :raw-latex:`$\mathbf{s}_{0}$`. As with shape, appearance variation is restricted to a base appearance plus a linear combination of :raw-latex:`$M$` appearance vectors

.. raw:: latex

   \begin{equation}
        A(\mathbf{x})=A_{0}+\sum_{i=1}^{M}\lambda_{i}A_{i}(\mathbf{x})\qquad\forall \mathbf{x}\in \mathbf{s}_{0}.
    \end{equation}

To generate an appearance model, the training images are first shape-normalized by warping each image onto the base mesh using a piecewise affine transformation. Recall that two sets of three corresponding points are sufficient for determining an affine transformation. The shape mesh vertices are first triangulated. The collection of corresponding triangles in two shapes meshes then defines a piecewise affine transformation between the two shapes. The pixel values within each triangle in the training shape :raw-latex:`$\mathbf{s}$` are warped onto the corresponding triangle in the base shape :raw-latex:`$\mathbf{s}_{0}$` using the affine transformation defined by the two triangles.

The appearance model is generated from the shape-normalized images using PCA. Figure :raw-latex:`\ref{fig:appearance}` shows the base appearance and the first three appearance images.

.. raw:: latex

    \begin{figure}
        \begin{center}
        \includegraphics[width=\textwidth]{appearance.pdf}
        \end{center}
        \caption{Mean appearance $A_{0}$ (top left) and first three appearance images $A_{1}$ (top right), $A_{2}$ (bottom left) and $A_{3}$ (bottom right).}
        \label{fig:appearance}
    \end{figure}
..

Tracking of an appearance in a sequence of images is performed by minimizing the difference between the base model appearance, and the input image warped onto the coordinate frame of the AAM. For a given image :raw-latex:`$I$` we minimize

.. raw:: latex

    \begin{equation}
    \label{eqn:obj_func}
    \underset{\bm{\lambda},\mathbf{p}}{\operatorname{argmin}} \sum_{\mathbf{x}\in\mathbf{s}_{0}}\left[A_{0}(\mathbf{x})+\sum_{i=1}^{M}\lambda_{i}A_{i}(\mathbf{X})-I(\mathbf{W}(\mathbf{x};\mathbf{p}))\right]^{2}
    \end{equation}

where :raw-latex:`$\mathbf{p} = \{p_{1},\ldots,p_{N}\}$` and :raw-latex:`$\bm{\lambda} = \{\lambda_{1},\ldots,\lambda_{N}\}$`. For the rest of the discussion of AAMs  we assume that the variable :raw-latex:`$\mathbf{x}$` takes on the image coordinates contained within the base mesh :raw-latex:`$\mathbf{s}_{0}$` (as in Equation :raw-latex:`\ref{eqn:obj_func}`).

In Equation :raw-latex:`\ref{eqn:obj_func}` we are looking for the optimal alignment of the input image, warped backwards onto the frame of the base appearance :raw-latex:`$A_{0}(\mathbf{x})$`. 

.. 
    ----------------------------
    LUCAS-KANADE IMAGE ALIGNMENT
    ----------------------------

For simplicity we will limit the discussion to shape variation and ignore any variation in texture. The derivation for the case including texture variation is available in [Mat03]_. Consequently Equation :raw-latex:`\ref{eqn:obj_func}` now reduces to

.. raw:: latex

    \begin{equation}
    \label{eqn:lkia_p}
        \underset{\mathbf{p}}{\operatorname{argmin}}   \sum_{\mathbf{x}}[A_{0}(\mathbf{x}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}))]^{2}.
    \end{equation}

Solving Equation :raw-latex:`$\ref{eqn:lkia_p}$` for :raw-latex:`$\mathbf{p}$` is a non-linear optimization problem. This is the case even if :raw-latex:`$\mathbf{W}(\mathbf{x};\mathbf{p})$` is linear in :raw-latex:`$\mathbf{p}$` since the pixel values :raw-latex:`$I(\mathbf{x})$` are in general nonlinear in :raw-latex:`$\mathbf{x}$`. 

The quantity that is minimized in Equation :raw-latex:`\ref{eqn:lkia_p}` is the same as in the classic Lucas-Kanade image alignment algorithm [Luc81]_. In the Lukas-Kanade algorithm the problem is first reformulated as

.. raw:: latex

    \begin{equation}
        \label{eqn:lkia_dp}
        \underset{\Delta\mathbf{p}}{\operatorname{argmin}} \sum_{\mathbf{x}}[A_{0}(\mathbf{X}) - I(\mathbf{W}(\mathbf{x};\mathbf{p}+\Delta\mathbf{p}))]^{2}.
    \end{equation}

This equation differs from Equation :raw-latex:`$\ref{eqn:lkia_p}$` in that we are now optimizing with respect to :raw-latex:`$\Delta\mathbf{p}$` while assuming :raw-latex:`$\mathbf{p}$` is known. Given an initial estimate of :raw-latex:`$\mathbf{p}$` we update with the value of :raw-latex:`$\Delta\mathbf{p}$` that minimizes Equation :raw-latex:`\ref{eqn:lkia_dp}` to give

.. raw:: latex

    \begin{equation}
        \mathbf{p}^{\mathrm{new}} = \mathbf{p} + \Delta\mathbf{p}.
    \end{equation}

This will necessarily decrease the value of Equation :raw-latex:`\ref{eqn:lkia_p}` for the new value of :raw-latex:`$\mathbf{p}$`. Replaing :raw-latex:`$\mathbf{p}$` with the upadted value for :raw-latex:`$\mathbf{p}^{\mathrm{new}}$`, this procedure is iterated until convergence at which point :raw-latex:`$\mathbf{p}$` yields the (local) optimal shape parameters for the input image :raw-latex:`$I$`.

To solve Equation :raw-latex:`\ref{eqn:lkia_dp}` Taylor expansion is used [Bak01]_ which gives

.. raw:: latex

    \begin{equation}
    \label{eqn:taylor}
    \underset{\Delta\mathbf{p}}{\operatorname{argmin}} \sum_{\mathbf{x}}\left[A_{0}(\mathbf{W}(\mathbf{x};\mathbf{p}))-I(\mathbf{W}(\mathbf{x};\mathbf{p}))-\nabla I\frac{\partial \mathbf{W}}{\partial\mathbf{p}}\Delta\mathbf{p}\right]^{2}
    \end{equation}

where :raw-latex:`$\nabla I$` is the gradient of the input image and :raw-latex:`$\partial\mathbf{W}/\partial\mathbf{p}$` is the Jacobian of the warp evaluated at :raw-latex:`$\mathbf{p}$`.

The optimal solution to Equation :raw-latex:`\ref{eqn:taylor}` is found by setting the partial derivative with respect to :raw-latex:`$\Delta\mathbf{p}$` equal to zero which gives

.. raw:: latex

    \begin{equation}
        2\sum_{\mathbf{x}}\left[\nabla\mathbf{I}\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\right]^{\mathrm{T}}\left[A_{0}(\mathbf{x}) - I(\mathbf{W}(\mathbf{x}))-\nabla I\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\Delta\mathbf{p}\right] = 0.
    \end{equation}

Solving for :raw-latex:`$\Delta\mathbf{p}$` we get

.. raw:: latex

    \begin{equation}
        \Delta\mathbf{p} = \mathbf{H}^{-1}\sum_{\mathbf{x}}\left[\nabla I\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\right]^{\mathrm{T}}\left[A_{0}(\mathbf{x})-I(\mathbf{W}(\mathbf{x};\mathbf{p}))\right]
    \end{equation}
    
where :raw-latex:`$\mathbf{H}$` is the Gauss-Newton approximation to the Hessian matrix given by

.. raw:: latex

    \begin{equation}
        \mathbf{H} = \sum_{\mathbf{x}}\left[\nabla I\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\right]^{\mathrm{T}}\left[\nabla I\frac{\partial\mathbf{W}}{\partial\mathbf{p}}\right].
    \end{equation}

For a motivation for the backwards warp and further details on how to compute the piecewise linear affine warp and the Jacobian see [Mat03]_. 

A proper initialization of the shape parameters :raw-latex:`$\mathbf{p}$` is essential for the first frame. For subsequent frames :raw-latex:`$\mathbf{p}$` may be initialized as the optimal parameters from the previous frame.

The Lucas-Kanade algorithm is a Gauss-Newton gradient descent algorithm. Gauss-Newton gradient descent is available in ``scipy.optimize.fmin_ncg``. 

Example usage::

    from scipy import empty
    from scipy.optimize import fmin_ncg
    from scikits.image.io import imread

    # NOTE: The AAM module is currently under development
    import aam

    # Initialze AAM from visual speech training data
    vs_aam = aam.AAM('./training_data/')

    I = imread('face.jpg')

    def error_image(p):
        """ Compute error image given p """

        # Piecewise linear warp the image onto
        # the base AAM mesh
        IW = vs_aam.pw_affine(I,p)

        # Return error image
        return aam.A0-IW

    def gradient_descent_images(p):
        """ Compute gradient descent images given p """
        ...
        return gradIW_dWdP
    
    def hessian(p):
        """ Compute hessian matrix """"
        ...
        return H

    # Update p 
    p = fmin_ncg(f=error_image,
                     x0=p0,
                     fprime=gradient_descent_images,
                     fhess=hessian)

.. raw:: latex

    \begin{figure}

        \begin{center}
        \includegraphics[width=\textwidth]{aam_fit.pdf}
        \end{center}
        \caption{AAM fitted to an image}
        \label{fig:aam_fit}
    \end{figure}
..

Figure :raw-latex:`\ref{fig:aam_fit}` shows an AAM fitted to an input image. When tracking motion in a video sequence an AAM is fitted to each frame using the previous optimal fit as a starting point. 

In [Bak01]_ the AAM fitting method described above is referred to as \`forwards-additive`. 

As can be seen in Figure :raw-latex:`\ref{fig:shape}` the first two shape vectors mainly correspond to the movement in the up-down and left-right directions, respectively. As these components do not contain any speech related information we can ignore the corresponding shape parameters :raw-latex:`$p_{1}$` and :raw-latex:`$p_{2}$` when extracting visual speech features. The remaining shape parameters, :raw-latex:`$p_{3},\ldots,p_{N}$`, are used as visual features in the AVASR system. 

Models for audio-visual speech recognition
==========================================

Once acoustic and visual speech features have been extracted from respective modalities, we learn probabilistic models for each of the classes we need to discriminate between (e.g. words or phonemes). The models are learned from manually labeled training data. We require these models to `generalize` well; i.e. the models must be able to correctly classify novel samples that was not present in the training data.

Gaussian Mixture Models
-----------------------

Gaussian Mixture Models (GMMs) provide a powerful method for modeling data distributions under the assumption that the data is independent and identically distributed (i.i.d.). GMMs are defined as a weighted sum of Gaussian probability distributions

.. raw:: latex

    \begin{equation}\label{eqn:gauss}
        p(\mathbf{x}) = \sum_{k=1}^{K}\pi_{k}\mathcal{N}(\mathbf{x}|\bm{\mu}_{k},\bm{\Sigma}_{k})
    \end{equation}

where :raw-latex:`$\pi_{k}$` is the weight, :raw-latex:`$\bm{\mu}_{k}$` the mean, and :raw-latex:`$\bm{\Sigma}_{k}$` the covariance matrix of the :raw-latex:`$k\mathrm{th}$` mixture component.

Maximum likelihood
------------------

The log likelihood function of the GMM parameters :raw-latex:`$\bm{\pi}$`, :raw-latex:`$\bm{\mu}$` and :raw-latex:`$\bm{\Sigma}$` given a set of D-dimensional observations :raw-latex:`$\mathbf{X}=\{\mathbf{x}_{1},\ldots,\mathbf{x}_{N}\}$`  is given by

.. raw:: latex

    \begin{equation}
        \ln p(\mathbf{X}|\bm{\pi},\bm{\mu},\bm{\Sigma}) = \sum_{n=1}^{N}\ln\left\{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(\bm{x}_{n}|\bm{\mu}_{k},\bm{\Sigma}_{k})\right\}.
    \end{equation}

Note that the log likelihood is a function of the GMM parameters :raw-latex:`$\bm{\pi},\bm{\mu}$` and :raw-latex:`$\bm{\Sigma}$`. In order to fit a GMM to the observed data we maximize this likelihood with respect to the model parameters.

Expectation maximization
------------------------

The Expectation Maximization (EM) algorithm [Bis07]_ is an efficient iterative technique for optimizing the log likelihood function. As its name suggests, EM is a two stage algorithm. The first (`E` or `expectation`) step calculates the expectations for each data point to belong to each of the mixture components. It is also often expressed as the `responsibility` that the :raw-latex:`$k$th` mixture component takes for `explaining` the :raw-latex:`$n$th` data point, and is given by

.. raw:: latex

    \[r_{nk} = \frac{\pi_{k}\mathcal{N}(\mathbf{x}_{n}|\bm{\mu}_{k},\bm{\Sigma_{k}})}{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(\mathbf{x}_{n}|\bm{\mu}_{k},\bm{\Sigma}_{k})}.\]

Note that this is a \`soft' assignment where each data point is assigned to a given mixture component with a certain probability. Once the responsibilities are available the model parameters are updated (`M` or `maximization` step). The quantities

.. raw:: latex

    \begin{eqnarray}
        N_{k} &=& \sum_{n=1}^{N}r_{nk} \label{eqn:m_step_N}\\
        \mathbf{\bar{x}}_{k} &=& \sum_{n=1}^{N}r_{nk}\mathbf{x}_{n}\label{eqn:m_step_xbar}\\
        \mathrm{S}_{k} &=& \sum_{n=1}^{N}r_{nk}(\mathbf{x}_{n}-\mathbf{\bar{x}}_{k})(\mathbf{x}_{n}-\mathbf{\bar{x}}_{k})^{\mathrm{T}}\label{eqn:m_step_S}
    \end{eqnarray}


are first calculated. Finally the model parameters are updated as

.. raw:: latex

    \begin{eqnarray}
        \pi_{k}^{\mathrm{new}} &=& \frac{N_{k}}{N}\label{eqn:pi_k_new}\\
        \bm{\mu}_{k}^{\mathrm{new}} &=& \frac{\mathbf{\bar{x}}_{k}}{N_{k}}\label{eqn:mu_k_new}\\
        \mathbf{\Sigma}_{k}^{\mathrm{new}} &=& \frac{S_{k}}{N_{k}}\label{eqn:Sigma_k_new}.
    \end{eqnarray}

See [Bis07]_ for the derivation of these equations.

The EM algorithm in general only converges to a local optimum of the log likelihood function. Thus, the choice of initial parameters is crucial.

GMM-EM is available in ``scikits.learn.em``.

Example usage::

    from numpy import loadtxt
    from scikits.learn.em import GM, GMM, EM

    # Data dimensionality
    D = 8

    # Number of Gaussian Mixture Components
    K = 16        

    # Initialize Gaussian Mixture Model
    gmm = GMM(GM(D,K))

    # X is the feature data matrix

    # Learn GMM
    EM().train(X,gmm)

Figure :raw-latex:`\ref{fig:em}` shows a visual speech GMM learned using EM. For illustrative purposes only the first two speech-related shape parameters :raw-latex:`$p_{3}$` and :raw-latex:`$p_{4}$` are used. The shape parameters are obtained by fitting an AAM to each frame of a video of a speaker saying the word \`zero'. The crosses represent the training data, the circles are the means of the Gaussians and the ellipses are the standard deviation contours (scaled by the inverse of the weight of the corresponding mixture component for visualization purposes). The video frame rate is 30 frames per second (FPS) and the number of mixture components used is :raw-latex:`$16$`.

Note that in practice more than two shape parameters are used, which usually also requires an increase in the number of mixture components necessary to sufficiently capture the distribution of the data.

.. raw:: latex

    \begin{figure}
        \begin{center}
        \includegraphics[width=\columnwidth]{em.pdf}
        \end{center}
        \caption{Visual speech GMM of the word 'zero' learned using EM algorithm on two-dimensional feature vectors.}
        \label{fig:em}
    \end{figure}

Variational Bayes
-----------------

An important question that we have not yet answered is how to choose the number of mixture components. Too many components lead to redundancy in the number of computations, while too few may not be sufficient to represent the structure of the data. Additionally, too many components easily lead to overfitting. Overfitting occurs when the complexity of the model is not in proportion to the amount of available training data. In this case the data is not sufficient for accurately estimating the GMM parameters. 

The maximum likelihood criteria is unsuitable to estimate the number of mixture components since it increases monotonically with the number of mixture components. Variational Bayesian (VB) inference is an alternative learning method that is less sensitive than ML-EM to over-fitting and singular solutions while at the same time leads to automatic model complexity selection [Bis07]_.

As it simplifies calculation we work with the precision matrix :raw-latex:`$\mathbf{\Lambda} = \mathbf{\Sigma}^{-1}$` instead of the covariance matrix.

VB differs from EM in that the parameters are modeled as random variables. Suitable conjugate distributions are the Dirichlet distribution

.. raw:: latex

    \begin{equation}
        p(\bm{\pi}) = C(\bm{\alpha}_{0})\prod_{k=1}^{K}\pi_{k}^{\alpha_{0}-1}
    \end{equation}

for the mixture component weights, and the Gaussian-Wishart distribution

.. raw:: latex

    \begin{equation}
        p(\bm{\mu},\bm{\Lambda}) = \prod_{k=1}^{K}\mathcal{N}(\bm{\mu}_{k}|\bm{m}_{0},\beta_{0}\Lambda_{k})\mathcal{W}(\Lambda_{k}|\mathbf{W}_{0},\bm{\nu}_{0})
    \end{equation}

for the means and precisions of the mixture components.

In the VB framework, learning the GMM is performed by finding the posterior distribution over the model parameters given the observed data. This posterior distribution can be found using VB inference as described in [Bis07]_.  


VB is an iterative algorithm with steps analogous to the EM algorithm. Responsibilities are calculated as

.. raw:: latex

    \begin{equation}
        r_{nk} = \frac{\rho_{nk}}{\sum_{j=1}^{K}\rho_{nj}}.
    \end{equation}

The quantities :raw-latex:`$\rho_{nk}$` are given in the log domain by

.. raw:: latex

    \begin{eqnarray}
        \ln{\rho_{nk}} &=& \mathbb{E}[\ln{\pi_{k}}] + \frac{1}{2}\mathbb{E}[\ln{|\bm{\Lambda}|}] - \frac{D}{2}\ln{2\pi}\nonumber\\
        && - \frac{1}{2}\mathbb{E}_{\bm{\mu}_{k},\bm{\Lambda}_{k}}[(\mathbf{x}_{n}-\bm{\mu}_{k})^{\mathrm{T}}\bm{\Lambda}_{k}(\mathbf{x}_{n}-\bm{\mu}_{k})]
    \end{eqnarray}

where

.. raw:: latex

    \begin{eqnarray}
            \mathbb{E}_{\bm{\mu},\bm{\Lambda}}[(\mathbf{x}_{n}-\bm{\mu}_{k})^{\mathrm{T}}\bm{\Lambda}_{k}(\mathbf{x}_{n}-\bm{\mu}_{k})]
    &=& D\beta_{k}^{-1}\nonumber\\
    +\nu_{k}(\mathbf{x}_{n}-\mathbf{m}_{k})^{\mathrm{T}}\mathbf{W}_{k}(\mathbf{x}_{n}-\mathbf{m}_{k})&&
    \end{eqnarray}

and

.. raw:: latex

    \begin{eqnarray}
        \ln{\widetilde{\pi}_{k}} &=& \mathbb{E}[\ln{\pi_{k}}] = \psi(\alpha_{k})-\psi(\widehat{\alpha}_{k})\label{eqn:log_pi_tilde}\\
        \ln{\widetilde{\Lambda}_{k}} &=& \mathbb{E}[\ln|\bm{\Lambda}_{k}|] = \sum_{i=1}^{D}\psi\left(\frac{\nu_{k}+1-i}{2}\right)\nonumber\\&&+D\ln{2}+\ln{|\mathbf{W}_{k}|}\label{eqn:log_lambda_tilde}.
    \end{eqnarray}

Here :raw-latex:`$\widehat{\alpha}=\sum_{k}\alpha_{k}$` and :raw-latex:`$\psi$` is the derivative of the logarithm of the gamma function, also called the digamma function. The digamma function is available in SciPy as ``scipy.special.psi``.

The analogous M-step is performed using a set of equations similar to those found in EM. First the quantities

.. raw:: latex

    \begin{eqnarray}
        N_{k} &=& \sum_{n}r_{nk}\label{eqn:N_k}\\
        \mathbf{\bar{x}}_{k} &=& \frac{1}{N_{k}}\sum_{n}r_{nk}\mathbf{x}_{n}\label{eqn:xbar_k}\\
        \mathbf{S}_{k} &=& \frac{1}{N_{k}}\sum_{n}r_{nk}(\mathbf{x}_{n}-\mathbf{\bar{x}}_{k})(\mathbf{x}_{n}-\mathbf{\bar{x}}_{k})^{\mathrm{T}}\label{eqn:S_k}.
    \end{eqnarray}

are calculated. The posterior model parameters are then updated as

.. raw:: latex

    \begin{eqnarray}
        \alpha_{k}^{\mathrm{new}} &=& \alpha_{0}+N_{k}\label{eqn:alpha_k}\\
        \beta_{k}^{\mathrm{new}} &=& \beta_{0} + N_{k}\label{eqn:beta_k}\\
        \mathbf{m}_{k}^{\mathrm{new}} &=& \frac{1}{\beta_{k}}(\beta_{0}\mathbf{m}_{0}+N_{k}\mathbf{\bar{x}}_{k})\\
        \mathbf{W}_{k}^{\mathrm{new}} &=& \mathbf{W}_{0} + N_{k}\mathbf{S}_{k} + \nonumber\\&&\frac{\beta_{0}N_{k}}{\beta_{0}+N_{k}}(\mathbf{\bar{x}}-\mathbf{m}_{0})(\mathbf{\bar{x}}-\mathbf{m}_{0})^{\mathrm{T}} \\
        \nu_{k}^{\mathrm{new}} &=& \nu_{0} + N_{k} \label{eqn:nu_k}.
    \end{eqnarray}

Figure :raw-latex:`\ref{fig:vb}` shows a GMM learned using VB on the same data as in Figure :raw-latex:`\ref{fig:em}`. The initial number of components is again :raw-latex:`$16$`. Compared to Figure :raw-latex:`\ref{fig:em}` we observe that VB results in a much sparser model while still capturing the structure of the data. In fact, the redundant components have all converged to their prior distributions and have been assigned the weight of  :raw-latex:`0` indicating that these components do not contribute towards \`explaining' the data and can be pruned from the model. We also observe that outliers in the data (which is likely to be noise) is to a large extent ignored.

.. raw:: latex

    \begin{figure}
        \begin{center}
        \includegraphics[width=\textwidth]{vb.pdf}
        \end{center}
        \caption{Visual speech GMM of the word `zero' learned using VB algorithm on two-dimensional feature vectors.}

        \label{fig:vb}
    \end{figure}
..

We have recently developed a Python VB class for ``scikits.learn``. The class conforms to a similar interface as the EM class and will soon be available in the development version of ``scikits.learn``.

Experimental results
====================

A basic AVASR system was implemented using SciPy as outlined in the previous sections.

In order to test the system we use the CUAVE database [Pat02]_. The CUAVE database consists of 36 speakers, 19 male and 17 female, uttering isolated and continuous digits. Video of the speakers is recorded in frontal, profile and while moving. We only use the portion of the database where the speakers are stationary and facing the camera while uttering isolated digits. We use data from 24 speakers for training and the remaining 12 for testing. Hence, data from the speakers in the test data are not used for training. This allows us to evaluate how well the models generalize to speakers other than than those used for training. A sample frame from each speaker in the dataset is shown in Figure :raw-latex:`\ref{fig:data}`.

.. raw:: latex

    \begin{figure}
        \begin{center}
        \includegraphics[width=\textwidth]{thumb.png}
        \end{center}
        \caption{Frames from the CUAVE audio-visual data corpus}
        \label{fig:data}
    \end{figure}
..

In the experiment we build an individual AAM for each speaker by manually annotating every 50th frame. The visual features are then extracted by fitting the AAM to each frame in the video of the speaker.

Training the speech recognition system consists of learning acoustic and visual GMMs for each digit using samples from the training data. Learning is performed using VB inference. Testing is performed by classifying the test data. To evaluate the performance of the system we use the misclassification rate, i.e. the number of wrongly classified samples divided by the total number of samples.

We train acoustic and visual GMMs separately for each digit. The probability distributions (see Equation :raw-latex:`\ref{eqn:gauss}`) are denoted by :raw-latex:`$p(\mathbf{x}_{A})$` and :raw-latex:`$p(\mathbf{x}_{V})$` for the acoustic and visual components, respectively. The probability of a sample :raw-latex:`$(\mathbf{x}_{A},\mathbf{x}_{V})$` belonging to digit class :raw-latex:`$c$` is then  given by :raw-latex:`$p_{A}(\mathbf{x}_{A}|c)$` and :raw-latex:`$p_{V}(\mathbf{x}_{V}|c)$`.

As we wish to test the effect of noise in the audio channel, acoustic noise ranging from -5dB to 25dB signal-to-noise ratio (SNR) in steps of 5 dB is added to the test data. We use additive white Gaussian noise with zero mean and variance

.. raw:: latex 

    \begin{equation}
        \label{eqn:noise}
        \sigma_{\eta}^{2} = 10^{\frac{-\mathrm{SNR}}{10}}.
    \end{equation}

The acoustic and visual GMMs are combined into a single classifier by exponentially weighting each GMM in proportion to an estimate of the information content in each stream. As the result no longer represent probabilities we use the term `score`. For a given digit we get the combined audio-visual model

.. raw:: latex

    \begin{equation} 
        \label{eqn:decision}
        \mathrm{Score}(\mathbf{x}_{AV}|c) = p(\mathbf{x}_{A}|c)^{\lambda_{A}}p(\mathbf{x}_{V}|c)^{\lambda_{V}}
    \end{equation}

where 

.. raw:: latex

    \begin{eqnarray}
        0\leq\lambda_{A}\leq 1\\
        0\leq\lambda_{V}\leq 1
    \end{eqnarray}

and

.. raw:: latex

    \begin{equation}
        \lambda_{A}+\lambda_{V}=1\label{eqn:param_constraint}.
    \end{equation}

Note that Equation :raw-latex:`\ref{eqn:decision}` is equivalent to a linear combination of log likelihoods.

The stream exponents cannot be determined through a maximum likelihood estimation, as this will always result in a solution with the modality having the largest probability being assigned a weight of 1 and the other 0. Instead, we discriminatively estimate the stream exponents. As the number of classes in our experiment is relatively small we perform this optimization using a brute-force grid search, directly minimizing the misclassification rate. Due to the constraint (Equation :raw-latex:`\ref{eqn:param_constraint}`) it is only necessary to vary :raw-latex:`$\lambda_{A}$` from 0 to 1. The corresponding :raw-latex:`$\lambda_{V}$` will then be given by :raw-latex:`$1-\lambda_{A}$`.We vary :raw-latex:`$\lambda_{A}$` from 0 to 1 in steps of 0.1. The set of parameters :raw-latex:`$\lambda_{A}$` and :raw-latex:`$\lambda_{V}$` that results in the lowest misclassification rate are chosen as optimum parameters.

..
    Table :raw-latex:`\ref{tab:opt_stream_w}` shows the optimal stream weights found for the varying SNR. Note that for the most extreme noise case (-5 dB) the best result is obtained by weighting the visual component with weight 0.7. As the level of noise decreases there appears to be a convergence towards :raw-latex:`$\lambda_{A}=0.9$` and :raw-latex:`$\lambda_{V}=0.1$` indicating that for clean audio the audio channel contains significantly more speech information than the visual channel, but that the combination of audio and visual speech still performs better than audio only. In the case of clean we were unable to achieve better results with AVASR than audio-only ASR.
    .. raw:: latex
    \begin{table}
      \begin{center}
      \begin{tabular}{l | c | c | c | c | c | c | c | c}
        SNR & -5 & 0 & 5 & 10 & 15 & 20 & 25 & Clean \\
        \hline
        $\lambda_{A}$ & 0.3 & 0.7 & 0.9 & 0.9 & 0.9 & 0.9 & 0.9 & 1.0\\
        $\lambda_{V}$ & 0.7 & 0.3 & 0.1 & 0.1 & 0.1 & 0.1 & 0.1 & 0.0
      \end{tabular}
      \end{center}
      \caption{Optimal stream weights}
      \label{tab:opt_stream_w}
    \end{table}
..

In the experiment we perform classification for each of the SNR levels using Equation :raw-latex:`\ref{eqn:decision}` and calculate the average misclassification rate. We compare audio-only, visual-only, and audio-visual classifiers. For the audio-only classifier the stream weights are :raw-latex:`$\lambda_{A}= \textrm{ and } \lambda_{V}=0$` and for visual-only :raw-latex:`$\lambda_{A}=0 \textrm{ and } \lambda_{V}=1$`. For the audio-visual classifier the discriminatively trained stream weights are used. Figure :raw-latex:`\ref{fig:results}` shows average misclassification rate for the different models and noise levels.

From the results we observe that the visual channel does contain information relevant to speech, but that visual speech is not in itself sufficient for speech recognition. However, by combining acoustic and visual speech we are able to increase recognition performance above that of audio-only speech recognition, especially the presence of acoustic noise.

.. raw:: latex

    \begin{figure}
        \begin{center}
        \includegraphics[width=\textwidth]{results.pdf}
        \end{center}
        \caption{Misclassification rate}
        \label{fig:results}
    \end{figure}

Conclusion
==========

In this paper we propose a basic AVASR system that uses MFCCs as acoustic features, AAM parameters as visual features, and GMMs for modeling the distribution of audio-visual speech feature data. We present the EM and VB algorithms as two alternatives for learning the audio-visual speech GMMs and demonstrate how VB is less affected than EM by overfitting while leading to automatic model complexity selection.

The AVASR system is implemented in Python using SciPy and tested using the CUAVE database. Based on the results we conclude that the visual channel does contain relevant speech information, but is not in itself sufficient for speech recognition. However, by combining features of visual speech with audio features, we find that AVASR gives better performance than audio-only speech recognition, especially in noisy environments.

..
    Future Work
    ===========
    
    When optimizing Equation :raw-latex:`\ref{eqn:lkia_p}` we only consider variation in shape. However, additionally modeling appearance variation and including the appearance parameters in the speech feature vectors is likely to increase the robustness and performance of both the AAM tracker and speech recognizer.
    
    The GMM makes the fundamental assumption that the data points are independent. However, for speech this is not the case as speech sounds are highly context dependent. Assuming that an observation is conditionally dependent of the previous observation only we obtain the first-order Hidden Markov Model (HMM) which has been a highly successful model for speech recognition.
    
    Dynamic Bayesian networks is generalization of HMMs that allow us to additionally model asynchrony between the acoustic and visual stream. 
    
    EM is the standard method for training both HMMs and DBNs. However, it also possible to perform variational Bayesian analysis in DBNs. The resulting models should have similar properties as to GMMs. The performance of VB of DBNs used in speech recognition remains an unexplored research direction.
    
    Stream weights may be weighted adaptively.

Acknowledgments
===============
The authors wish to thank MIH Holdings for funding the research presented in this paper and for granting permission to contribute the research source code to SciPy.



References
==========
.. [Dav80] S. Davis, I. Matthews. *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*,
           IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(8),357-366, 1980

.. [Luc81] B.D. Lucas, T. Kanade. *An iterative image registration technique with an application to stereo vision*,
           Proceedings of Imaging understanding workshop, 121-130, 1981

.. [Coo98] T.F. Cootes, G.J. Edwards, C. J .Taylor, *Active appearance models*,
           Proceedings of the European Conference on Computer Vision, 1998

.. [Bak01] S. Baker and I. Matthews, *Lucas Kanade 20 Years On: A Unifying Framework*,
           International Journal of Computer Vision, 2000

.. [Pat02] E.K. Patterson, S. Gurbuz, Z. Tufekci, J.N. Gowdy,
           *CUAVE: A new audio-visual databse for multimodeal human-compuer inferface research*, 2002

.. [Mat03] I. Matthews, S. Baker, *Active Appearance Models Revisited*,
           International Journal of Computer Vision, 2003

.. [Bis07] C.M.Bishop. *Pattern recognition and machine learning*,
           Springer, 2007
