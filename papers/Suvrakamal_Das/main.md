## Abstract

The quest for more efficient and faster deep learning models has led to the development of various alternatives to Transformers, one of which is the Mamba model. This paper provides a comprehensive comparison between Mamba models and Transformers, focusing on their architectural differences, performance metrics, and underlying mechanisms. We analyze and synthesize findings from extensive research conducted by various authors on these models.

The synergy between Mamba models and the SciPy ecosystem enhances their integration into science.
By providing an in-depth comparison using Python and its scientific ecosystem, this paper aims to clarify the strengths and weaknesses of Mamba models relative to Transformers. We conclude with insights on the potential implications for future research and applications in various scientific and industrial domains.

### Introduction

### Background: State Space Models

The central goal of machine learning is to develop models capable of efficiently processing sequential data across a range of modalities and tasks. This is particularly challenging when dealing with **long sequences**, especially those exhibiting **long-range dependencies (LRDs)** – where information from distant past time steps significantly influences the current state or future predictions. Examples of such sequences abound in real-world applications, including speech, video, medical, time series, and natural language. However, traditional models struggle to effectively handle such long sequences.

**Recurrent Neural Networks (RNNs)**, often considered the natural choice for sequential data, are inherently stateful and require only constant computation per time step. However, they are slow to train and suffer from the well-known "**vanishing gradient problem**", which limits their ability to capture LRDs. **Convolutional Neural Networks (CNNs)**, while efficient for parallelizable training, are not inherently sequential and struggle with long context lengths, resulting in more expensive inference. **Transformers**, despite their recent success in various tasks, typically require specialized architectures and attention mechanisms to handle LRDs, which significantly increase computational complexity and memory usage.

A promising alternative for tackling LRDs in long sequences is **State Space Models (SSMs)**, a foundational mathematical framework deeply rooted in diverse scientific disciplines like control theory and computational neuroscience. SSMs provide a continuous-time representation of a system's state and evolution, offering a powerful paradigm for capturing LRDs. They represent a system's behavior in terms of its internal **state** and how this state evolves over time. SSMs are widely used in various fields, including control theory, signal processing, and computational neuroscience.

### Continuous-time Representation

The continuous-time SSM describes a system's evolution using differential equations. It maps a continuous-time input signal $u(t)$ to an output signal $y(t)$ through a latent state $x(t)$. The state is an internal representation that captures the system's history and influences its future behavior.

The core equations of the continuous-time SSM are:

* **State Evolution:**
  $${x'(t) = Ax(t) + Bu(t)}$$

* **Output Generation:**
  $${y(t) = Cx(t) + Du(t)}$$

where:

* $x(t)$ is the state vector at time $t$, belonging to a $N$-dimensional space.
* $u(t)$ is the input signal at time $t$.
* $y(t)$ is the output signal at time $t$.
* $A$ is the state matrix, controlling the evolution of the state vector $x(t)$.
* $B$ is the control matrix, mapping the input signal $u(t)$ to the state space.
* $C$ is the output matrix, projecting the state vector $x(t)$ onto the output space.
* $D$ is the command matrix, directly mapping the input signal $u(t)$ to the output. (For simplicity, we often assume $D$ = 0, as $Du(t)$ can be viewed as a skip connection.)

This system of equations defines a continuous-time mapping from input $u(t)$ to output $y(t)$ through a latent state $x(t)$. The state matrix $A$ plays a crucial role in determining the dynamics of the system and its ability to capture long-range dependencies.

### HiPPO Framework for Long-Range Dependencies

Despite their theoretical elegance, naive applications of SSMs often struggle with long sequences. This is due to the inherent limitations of simple linear differential equations in capturing long-range dependencies (LRDs). To overcome this, the **High-Order Polynomial Projection Operator (HiPPO)** framework provides a principled approach for designing SSMs specifically suited for LRDs.

HiPPO focuses on finding specific state matrices $A$ that allow the state vector $x(t)$ to effectively memorize the history of the input signal $u(t)$. It achieves this by leveraging the properties of orthogonal polynomials. The HiPPO framework derives several structured state matrices, including:

* **HiPPO-LegT (Translated Legendre):** Based on Legendre polynomials, this matrix enables the state to capture the history of the input within sliding windows of a fixed size.
* **HiPPO-LagT (Translated Laguerre):** Based on Laguerre polynomials, this matrix allows the state to capture a weighted history of the input, where older information decays exponentially.
* **HiPPO-LegS (Scaled Legendre):** Based on Legendre polynomials, this matrix captures the history of the input with respect to a linearly decaying weight.

### Discrete-time SSM: Recurrent Representation

To apply SSMs on discrete-time data sequences ($u_0$, $u_1$, ...), it's necessary to discretize the continuous-time model. This involves converting the differential equations into difference equations, where the state and input are defined at discrete time steps.

One common discretization method is the **bilinear transform**, also known as the **Tustin method**. This transform approximates the derivative $x'(t)$ by a weighted average of the state values at two consecutive time steps, introducing a **step size** $\delta$ that represents the time interval between samples.

Applying the bilinear transform to the continuous-time SSM yields the following discrete-time representation:

$$
x_k = \overline { A } x_{k - 1} + \overline B u_k ~~~~~~~~~
 \overline A = ( I - \Delta / 2 \cdot A )^ { - 1 } ( I + \Delta / 2 \cdot A )
$$

$$
y_k = \overline C x_k ~~~~~~~~~~
\overline B = ( I - \Delta / 2 \cdot A) ^ { - 1 } \Delta B ~~~~~~~~~
\overline { C } = C
$$

where:

* $x_k$ is the state vector at time step $k$.
* $u_k$ is the input signal at time step $k$.
* $y_k$ is the output signal at time step $k$.
* $A$ is the discretized state matrix, derived from the original state matrix $A$ and the step size $\delta$.
* $B$ is the discretized control matrix, derived from the original control matrix $B$ and the step size $\delta$.
* $C$ is the same output from the continuous-time SSM.

This discrete-time SSM can now be implemented as a **recurrent model**, where the state vector $x_k$ serves as the hidden state, and the discretized state matrix $A$ defines the state transition.

### Training SSMs: Convolutional Representation

While the recurrent representation provides a computationally efficient way to perform inference with an SSM, it is not optimal for training due to its sequential nature. To overcome this, SSM leverages the connections between linear time-invariant (LTI) SSMs and convolution.
The convolutional representation allows for efficient parallel training using Fast Fourier Transform (FFT) algorithms. However, the main challenge lies in computing the SSM convolution kernel $K$. Computing it naively with $L$ successive matrix multiplications by $A$ results in $O(N^2*L)$ operations and $O(NL)$ memory – a significant computational bottleneck for long sequences.
The discrete-time SSM can be expressed explicitly as a convolution:

 $$y = \overline K * u$$
$$
\begin{array} { c }
{ \overline K \in \mathbb{R}^L := \kappa _ L ( \overline A, \overline B, \overline C) : = ( \overline C \overline A ^ i \overline B ) _ { i \in [ L ] }} \ { = ( \overline C \overline B, \overline C \overline A \overline B,..., \overline C \overline { { A } } ^ { L - 1 } \overline B ) } \ \end{array}
$$

### S4: A Structured State Space Model

The theoretical advantages of State Space Models (SSMs) for handling long sequences, particularly their ability to capture long-range dependencies, make them a promising alternative to traditional sequence models. However, the computational limitations of existing SSM implementations, such as the LSSL, hinder their widespread adoption.
The Structured State Space (S4) model aims to overcome these limitations by introducing novel parameterization and efficient algorithms that preserve the theoretical strengths of SSMs.

### Diagonalization Problem

The core computational bottleneck in SSMs stems from repeated matrix multiplication by the state matrix $A$ when calculating the convolution kernel $K$. If $A$ were a diagonal matrix, this computation would become significantly more tractable. Diagonal matrices allow for efficient power calculations as well as multiplication by a vector, resulting in a time complexity of $O(N)$ for $N$ dimensions.

Diagonalization involves finding a change of basis that transforms $A$ into a diagonal form. However, this approach faces significant challenges when $A$ is **non-normal**. Non-normal matrices have complex eigenstructures, which can lead to several problems:

* **Numerically unstable diagonalization:** Diagonalizing non-normal matrices can be numerically unstable, especially for large matrices. This is because the eigenvectors may be highly sensitive to small perturbations in the matrix, leading to large errors in the computed eigenvalues and eigenvectors.
* **Exponentially large entries:** The diagonalization of some non-normal matrices, including the HiPPO matrices, can involve matrices with entries that grow exponentially with the dimension $N$. This can lead to overflow issues during computation and render the diagonalization infeasible in practice.

Therefore, naive diagonalization of non-normal matrices in SSMs is not a viable solution for efficient computation.

### The S4 Parameterization: Normal Plus Low-Rank (NPLR)

S4 overcomes the challenges of directly diagonalizing non-normal matrices by introducing a novel parameterization. It decomposes the state matrix *A* into a sum of a **normal matrix** and a **low-rank term**. This decomposition allows for efficient computation while preserving the structure necessary to handle long-range dependencies. The S4 parameterization is expressed as follows:

* SSM convolution kernel

$$ ~~~~~~~~
\overline K = \kappa _L(\overline A, \overline B, \overline C) \text{~~~for~~~} A = V \Lambda V^* − P Q^T$$

where:

* *V* is a unitary matrix that diagonalizes the normal matrix.
* *Λ* is a diagonal matrix containing the eigenvalues of the normal matrix.
* *P* and *Q* are low-rank matrices that capture the non-normal component.
* These matrices HiPPO - $LegS, LegT, LagT$ all satisfy $r$ = 1 or $r$ = 2.

This decomposition allows for efficient computation because:

* **Normal matrices are efficiently diagonalizable:** Normal matrices can be diagonalized stably and efficiently using unitary transformations.
* **Low-rank corrections are tractable:** The low-rank term can be corrected using the Woodbury identity, a powerful tool for inverting matrices perturbed by low-rank terms.

### S4 Algorithms and Complexity

S4 leverages its NPLR parameterization to develop efficient algorithms for computing both the recurrent representation ($A$) and the convolutional kernel ($K$).

#### S4 Recurrence

The S4 recurrent representation is computed by discretizing the state matrix $A$ using the bilinear transform. The crucial observation is that the inverse of a DPLR matrix is also DPLR (due to the Woodbury identity). Therefore, the discretized state matrix $A$ is the product of two DPLR matrices, allowing for efficient matrix-vector multiplication in O(N) time.

#### S4 Convolution

S4's convolutional representation is computed through a series of steps:

1. **SSM Generating Function:** Instead of directly computing the convolution kernel $K$, S4 calculates its spectrum by evaluating its truncated generating function. The generating function allows for efficiently expressing powers of $A$ as a single matrix inverse.
2. **Woodbury Correction:** The Woodbury identity is used to correct the low-rank term in the generating function, reducing the problem to evaluating the generating function for a diagonal matrix.
3. **Cauchy Kernel:** The generating function for a diagonal matrix is equivalent to computing a Cauchy kernel, which is a well-studied problem with efficient, numerically stable algorithms.

This process reduces the complexity of computing the convolution kernel $K$ to $O(N + L)$ operations and $O(N + L)$ memory, significantly improving upon the LSSL's complexity.

### S4 Architecture Details

The S4 layer, as defined by its NPLR parameterization, implements a mapping from a 1-D input sequence to a 1-D output sequence. To handle multiple features, the S4 architecture utilizes $H$ independent copies of the S4 layer, each processing one feature dimension. These outputs are then mixed using a position-wise linear layer, similar to a depthwise-separable convolution. This architecture allows for efficient computation while preserving the ability to capture relationships between different features.

Non-linear activation functions are typically added between S4 layers to enhance the model's expressivity, further paralleling the structure of CNNs. Thus, the overall deep S4 model resembles a depthwise-separable CNN, but with global convolution kernels that effectively capture long-range dependencies.

In summary, S4 offers a structured and efficient approach to SSMs, overcoming the limitations of previous implementations while preserving their theoretical strengths. Its NPLR parameterization allows for stable and efficient computation, while its efficient algorithms significantly reduce computational complexity. S4's ability to handle multiple features and its resemblance to CNNs further contribute to its versatility and potential as a powerful general sequence modeling solution.

### How SSMs Works

SSMs are usually part of larger neural network architecture, because on their own they are not much, from a high level perspective, they work like linear RNNs, where to output representation of the previous token and the embedding of the current input token are transformed and then combined. So Like in RNNs, SSMs process one input token after the other.

SSMs have 4 sets of matrices and parameters to process the input namely

$$
\Delta, A, B, C
$$

where:

* $\Delta $ modifies the weight in the $A$ and $B$ matrices
* $A$ (modified $\overline { A }$) determines how much of the Hidden state $h$ should propagate forward from token to token.
* $B$ (modified $\overline { B }$) Determines how of the input enters the hidden state.
* $C$ (modified $\overline { C }$) Determines how the hidden state transforms into output.

![Mamba Architecture](./ssmDiagram.drawio.svg)

1. **Discretization**

  The discretization technique facilitates the transformation of continuous differential equations into discrete time-step representations, leveraging the $\Delta$ matrix to decompose the infinitely continuous process into a discrete time-stepped process, thereby reducing computational complexity. In this approach, the $A$ and $B$ steps undergo discretization through the following equations:

  $$
  \overline{A} = \exp(\Delta A)
  $$

  $$
  \overline{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B
  $$

  This discretization scheme effectively reduces the continuous differential equation to a series of discrete time steps, enabling numerical approximations to be computed iteratively. By segmenting the continuous process into finite increments, the computational burden is alleviated, rendering the problem more tractable for numerical analysis and simulation.

2. **Linear RNNs**

The state-space models (SSMs) compute the output using a linear recurrent neural network (RNN) architecture, which operates on a hidden state $\Delta$. In this formulation, the hidden state propagates through a linear equation of the following form:

$$
h_t = \overline{A} h_{t-1} + \overline{B} X_t
$$
where
* $h_t$ is hidden state matrix at time step t
* $X_t$ is input vector at time t

The initial hidden state $h_0$ is computed as:
$$
h_0 = \overline{A} h_{-1} + \overline{B} x_0 = \overline{B} x_0
$$

Subsequently, the hidden state at the next time step, $h_1$, is obtained through the recursion:
$$
h_1 = \overline{A} h_0 + \overline{B} x_1 = \overline{A} \overline{B}
$$

The output $Y_t$ is then calculated from the hidden state $h_t$ and the input $X_t$ using the following linear transformation:

$$
Y_t = C h_t
$$

* C is the output control matrix
* $Y_t$ is output vector at time t
* internal hidden state at time t

$$
y_0 = C h_0 = C \overline{B} x_0 \\
y_1 = C h_1 = C \overline{A} \overline{B} x_0 + C \overline{B} x_1 \\
y_2 = C \overline{A}^2 \overline{B} x_0 + C \overline{A} \overline{B} x_1 + C \overline{B} x_2 \\
\vdots\\
y_t = C \overline{A}^t \overline{B} x_0 + C \overline{A}^{t-1} \overline{B} x_1 + \ldots + C \overline{A} \overline{B} x_{t-1} + C \overline{B} x_t
$$

$$
Y = K \cdot X
$$

where :
* $X$ is the input matrix *i.e.* $[x_0, x_1, \ldots, x_L]$
* $
K = \left( C \overline{B}, \, C \overline{A} \overline{B}, \, \ldots, \, C \overline{A}^{L-1} \overline{B} \right)
$

This linear RNN architecture effectively captures the temporal dependencies inherent in sequential data, enabling the model to learn and propagate relevant information through the recurrent connections. The linear formulation leverages the computational efficiency of matrix operations, facilitating scalable implementations.Continuos SSMs are differential equations that tell you how a hiddden state changes over time.

#### Parallel Associative Scan

The linear recurrent behavior inherent in the previous formulation is not efficiently implementable on GPU architectures, which favor parallel computing paradigms. This limitation renders convolutions inefficient in such environments. To address this challenge, the parallel associative scan technique is employed, which introduces a prefix sum-like operation to scan for all prefix sums. Although inherently sequential, this approach leverages an efficient parallel algorithm model to parallelize the SSM convolution, resulting in a significant performance boost. The parallel associative scan method exhibits linear time and space complexity, making it a computationally efficient solution.

##### SSM

$$
y_t = C \overline{A}^t \overline{B} x_0 + C \overline{A}^{t-1} \overline{B} x_1 + \ldots + C \overline{A} \overline{B} x_{t-1} + C \overline{B} x_t
$$

##### Selective SSM

$$
y_t = C_0 \overline{A}^t \overline{B}_0 x_0 + C_1 \overline{A}^{t-1} \overline{B}_1 x_1 + \ldots \\
\text{input-dependent }B\text{ and }C\text{ matrix}
$$

By leveraging the parallel associative scan technique, the selective SSM formulation can be efficiently implemented on parallel architectures, such as GPUs. This approach enables the exploitation of the inherent parallelism in the computation, leading to significant performance gains, particularly for large-scale applications and time-series data processing tasks.

## Mamba Model Architecture

One Mamba Layer is composed of one a selective state space module and some other layers as follows. Linear Layer Doubles the dimentionality of the input token embedding. A higher dimentionality gives the network more space to push around more information. Also some inseperable classes in the lower dimension might become seperable in the higher dimension. The authors use the 64 input dimensional embedding so this layers increases dimenionality from 64 to a 128.

Then a canonical 1D layers takes in the output of the previous layers. It's role is to push around dimension in the linearly upscaled 128 dimentional vector. It uses the SILU Activation function. Then comes the state space module to process the output of the convolution like a linear RNN.

![Mamba Architecture](./mamba.drawio.svg)

Then Mamba does the gated multiplication, we take the input, and pass it through another linear layer and
then pass it through another activation function. And this result of this is multiplied to the output of the selective SSM. The Author's intuition behind this operation is that the multiplication is the measure of similarity between the output of SSM which contains information from the previous tokens and the embedding of the current token. Then a linear layers reduces dimentionality back from 128 to 64.

To get mamba , we just need to stack multiple layers on top of each other.
And unlike other SSMs achitectures they do not need some other layers in between because they use the same layers all the time. In the same way in which transformers are composed of just Transformer layers on top of one another.

### Key Differences Between Mamba and Transformer Architectures

In this section, we present a detailed comparison of the Mamba and Transformer architectures. We focus on their core components, computational characteristics, and performance implications. Visualizations and equations are provided to illustrate these differences clearly.

Self attention, feed forward NN, normalization, residual layers and so on.

### Architecture Overview

#### Transformer Architecture
Transformers rely heavily on attention mechanisms to model dependencies between input and output sequences. The core components include:
* **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input sequence.
* **Position-wise Feed-Forward Networks**: Applied to each position separately.
* **Positional Encoding**: Adds information about the position of each token in the sequence, as Transformers lack inherent sequential information due to the parallel nature of their processing.

![Transformer Archtecture](https://imgur.com/ijGq9Z9.jpeg)

#### Mamba Architecture
Mamba models are based on Selective State Space Models (SSMs), combining aspects of RNNs, CNNs, and classical state space models. Key features include:
* **Selective State Space Models**: Allow input-dependent parameterization to selectively propagate or forget information.
* **Recurrent Mode**: Efficient recurrent computations with linear scaling.
* **Hardware-aware Algorithm**: Optimized for modern hardware to avoid inefficiencies.

### Key Differences

#### 1. Attention Mechanisms vs. Selective State Space Models

**Transformers** use multi-head self-attention to capture dependencies within the sequence:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
Where \( Q \), \( K \), and \( V \) are the query, key, and value matrices, respectively, and \( d_k \) is the dimension of the key vectors.

**Mamba Models** replace attention with selective state space parameters that change based on the input:
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) $$
Here, \( A \), \( B \), and \( C \) are state space parameters that vary with the input, allowing for efficient handling of long sequences without the quadratic complexity of attention mechanisms.

#### 2. Computational Complexity

**Transformers** have a quadratic complexity with respect to the sequence length \( n \):
$$ O(n^2 \cdot d) $$
This is due to the dot-product operations in the attention mechanism.

**Mamba Models** achieve linear complexity:
$$ O(n \cdot d) $$
This is facilitated by the recurrent nature of SSMs and the hardware-aware algorithms that optimize memory usage and computation.

#### 3. Sequence Handling and Memory Efficiency

**Transformers** require a cache of previous elements to handle long-range dependencies, leading to high memory usage.

**Mamba Models** utilize selective state spaces to maintain relevant information over long sequences without the need for extensive memory caches, providing a more memory-efficient solution.

Mamba integrates selective state spaces directly into the neural network architecture. The selective mechanism allows the model to focus on relevant parts of the input dynamically.

### Equations

**Transformer Attention Mechanism:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Mamba State Space Model:**
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) $$
Here, \( h(t) \) is the hidden state, \( x(t) \) is the input, and \( A \), \( B \), and \( C \) are the state space parameters.

### Mamba's Synergy with Scipy
Scipy provides a robust ecosystem for scientific computing in Python, offering a wide range of tools and libraries for numerical analysis, signal processing, optimization, and more. This ecosystem serves as a fertile ground for the development and integration of Mamba, facilitating its training, evaluation, and deployment in scientific applications. Leveraging Scipy's powerful data manipulation and visualization capabilities, Mamba models can be seamlessly integrated into scientific workflows, enabling in-depth analysis, rigorous statistical testing, and clear visualization of results.

The combination of Mamba's language understanding capabilities and Scipy's scientific computing tools opens up new avenues for exploring large-scale scientific datasets commonly encountered in scientific research domains such as astronomy, medicine, and beyond, extracting insights, and advancing scientific discoveries.

### Potential Applications and Future Directions:
* **Efficient Processing of Large Scientific Datasets:** Mamba's ability to handle long-range dependencies makes it well-suited for analyzing and summarizing vast amounts of scientific data, such as astronomical observations, medical records, or experimental results, thereby reducing the complexity and enabling more efficient analysis.
* **Enhancing Model Efficiency and Scalability:** Integrating Mamba with Scipy's optimization and parallelization techniques can potentially improve the efficiency and scalability of language models, enabling them to handle increasingly larger datasets and more complex scientific problems.
* **Advancing Scientific Computing through Interdisciplinary Collaboration:** The synergy between Mamba and Scipy fosters interdisciplinary collaboration between natural language processing researchers, scientific computing experts, and domain-specific scientists, paving the way for novel applications and pushing the boundaries of scientific computing.

### Conclusion

#### References

1. arXiv:2111.00396 [cs.LG]
2. ddsf
