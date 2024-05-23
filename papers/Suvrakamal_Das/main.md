# Style Guide for SciPy Conference Proceedings

Please refer to this guide along with the current [README](https://github.com/scipy-conference/scipy_proceedings/blob/2024/README.md) of the repository for the proceedings.

There is a page limit of 8 pages on the paper, excluding references.

For general Style Guide details please check [IEEE style guide](https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/style_references_manual.pdf). For inclusive language, please refer to [American Psychological Association’s style guide](https://www.apa.org/about/apa/equity-diversity-inclusion/language-guidelines). This style guide is based on both these references. Use [Strunk and White 4th edition](https://archive.org/details/TheElementsOfStyle4thEdition) as a grammar reference. We use [Merriam-Webster](https://www.merriam-webster.com/) as the English dictionary.

### Mamba Models a possible replacement for Transformers?

In the paper title, capitalize the first letter of the first and last word and all the nouns, pronouns, adjectives, verbs, adverbs, and subordinating conjunctions (If, Because, That, Which). Capitalize abbreviations that are otherwise lowercase (e.g., use DC, not dc or Dc) except for unit abbreviations and acronyms. Articles (a, an, the), coordinating conjunctions (and, but, for, or, nor), and most short prepositions are lowercase unless they are the first or last word. Prepositions of more than three letters (Before, Through, With, Without, Versus, Among, Under, Between) should be capitalized.

### Body of a Paper

This is the main section of your paper and should be divided into the following subsections with clear headings:

1. Introduction
  
      1. Sequence Modeling in Scientific Computing: Briefly explain the importance of sequence modeling in scientific domains like bioinformatics, time series analysis, and physical simulations. Provide examples of tasks like protein structure prediction, gene sequence analysis, and time series forecasting.

      2. Transformers: The Current Standard: Introduce Transformers as the leading architecture for sequence modeling. Briefly mention their core concepts like attention mechanisms. Acknowledge limitations of Transformers such as high computational complexity for scientific computing tasks.

      3. Mamba Models: A Promising Alternative: Briefly introduce Mamba models as a novel architecture with potential advantages for scientific computing. Highlight key features like selective matrix parameters and potential for memory efficiency.

2. Mamba Model Architecture
    2.1 Core Concepts of Mamba Models: Explain how Mamba models learn selective matrix parameters, reducing memory usage compared to Transformers. Discuss the role of the SRAM cache (if applicable) in improving efficiency for long sequences.
        2.2 Comparison to Transformers' Architecture: Clearly explain how Mamba models differ from Transformers, particularly the absence of attention mechanisms. Discuss potential trade-offs between the two approaches.

3. Applications in Scientific Computing
        3.1 Bioinformatics Applications: Explain how Mamba models can be used for tasks like protein structure prediction or gene sequence analysis. Emphasize how their memory efficiency is crucial for handling large biological datasets.
        3.2 Time Series Analysis Applications: Discuss how Mamba models can be used to analyze long time series data in finance, weather forecasting, or sensor readings. Highlight the advantage of their ability to handle long sequences effectively.
        3.3 Potential Applications in Physical Simulations (Optional): Briefly explore the potential of using Mamba models for simulating complex physical systems.

4. Comparison and Results
        4.1 Performance Comparison with Transformers: Compare the performance of Mamba models with Transformers on specific scientific computing tasks mentioned in section 3 (e.g., protein structure prediction, time series forecasting). Utilize relevant metrics like accuracy, memory footprint, and training time. Present results in tables or visualizations for clarity. Discuss potential limitations of Mamba models compared to Transformers in specific scenarios.

5. Conclusion
        Summarize the key findings on the potential of Mamba models for scientific computing.
        Discuss the future directions of Mamba model research and potential areas for improvement.
        Emphasize how Mamba models align with the focus of SciPy'24 by offering memory efficiency and handling long sequences effectively in scientific computing tasks.

6. References

Every reference should be a separate entry. Using one number for more than one reference is not allowed. Please refer to your specific publication guidelines for formatting references.

## Introduction

## Background: State Space Models

A central goal of machine learning is to develop models capable of efficiently processing sequential data across a range of modalities and tasks. This is particularly challenging when dealing with **long sequences**, especially those exhibiting **long-range dependencies (LRDs)** – where information from distant past time steps significantly influences the current state or future predictions. Examples of such sequences abound in real-world applications, including speech, video, medical, time series, and natural language. However, traditional models struggle to effectively handle such long sequences.

**Recurrent Neural Networks (RNNs)**, often considered the natural choice for sequential data, are inherently stateful and require only constant computation per time step. However, they are slow to train and suffer from the well-known "**vanishing gradient problem**", which limits their ability to capture LRDs. **Convolutional Neural Networks (CNNs)**, while efficient for parallelizable training, are not inherently sequential and struggle with long context lengths, resulting in more expensive inference. **Transformers**, despite their recent success in various tasks, typically require specialized architectures and attention mechanisms to handle LRDs, which significantly increase computational complexity and memory usage.

A promising alternative for tackling LRDs in long sequences is **State Space Models (SSMs)**, a foundational mathematical framework deeply rooted in diverse scientific disciplines like control theory and computational neuroscience. SSMs provide a continuous-time representation of a system's state and evolution, offering a powerful paradigm for capturing LRDs. They represent a system's behavior in terms of its internal **state** and how this state evolves over time. SSMs are widely used in various fields, including control theory, signal processing, and computational neuroscience.

### Continuous-time Representation

The continuous-time SSM describes a system's evolution using differential equations. It maps a continuous-time input signal $(u(t)$ to an output signal $y(t)$ through a latent state $x(t)$. The state is an internal representation that captures the system's history and influences its future behavior.

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

One common discretization method is the **bilinear transform**, also known as the **Tustin method**. This transform approximates the derivative $x'(t)$ by a weighted average of the state values at two consecutive time steps, introducing a **step size** $∆$ that represents the time interval between samples.

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
* $A$ is the discretized state matrix, derived from the original state matrix $A$ and the step size $∆$.
* $B$ is the discretized control matrix, derived from the original control matrix $B$ and the step size $∆$.
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

## S4: A Structured State Space Model

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

### Other Text

#### Footnotes

Footnotes should be numbered in consecutive order throughout the text. The footnote numbers are superscripts in text and in the actual footnotes. In text, place the superscript footnote numbers after the punctuation such as fullstops, commas, and parentheses, but before colons, dashes, quotation marks, and semicolons in a compound sentence. The footnotes should be placed at the bottom of the text column in which they are cited.

#### List in Text

The ordering of labeling for all lists is 1), 2), 3) followed by a), b), c), and then i), ii), iii).

For example, first list goes as this: 1) first item; 2) second item; and 3) third item.

#### Editorial style

##### Acronyms

Adding abbreviations to the metadata means we have accessible abbreviations across all instances of abbreviations in the manuscript. ([reference](https://mystmd.org/guide/glossaries-and-terms#abbreviations))

Define acronyms the first time they appear in the Abstract as well as the first time they appear in the body of the paper, written out as part of the sentence, followed by the acronym in parentheses. If the acronym is not repeated in the Abstract, do not include the acronym in parentheses. Coined plurals or plurals of acronyms do not take the apostrophe (e.g., FETs).

Possessive forms of the acronym do take the apostrophe (e.g., CPU’s speed). Indefinite articles are assigned to abbreviations to fit the sound of the first letter (e.g., an FCC regulation; a BRI).

##### Plurals

Plurals of units of measure usually do not take the "s". For example, the plural form of 3 mil is 3 mil, but 3 bits/s instead of 3 bit/s. Plural forms of calendar years do not take the apostrophe (e.g., 1990s). To avoid confusion, plural forms of variables in equations do take the apostrophe (e.g., x’s).

### Inclusive language

This section of the style guide is copied from the APA - [American Psychological Association’s style guide](https://www.apa.org/about/apa/equity-diversity-inclusion/language-guidelines). Refer to that for more details.

Avoid using identity-first language while talking about disabilities, either a person is born with or they are imposed later. This is not applicable for chosen identities, e.g, educators, programmers, etc.

| Terms to avoid      | Suggested alternative        |
| ------------------- | ---------------------------- |
| elderly             | senior citizen               |
| subject             | particiant                   |
| wheel-chair bound <br> confined to a wheelchair    | person who uses a wheelchair <br> wheelchair user|
| confined to a wheelchair | wheelchair user         |
| mentally ill <br>crazy <br>insane <br>mental defect <br>suffers from or is afflected with [condition]| person living with a mental illness <br>person with a preexisting mental health disorder <br>person with a behavioral health disorder <br>person with a diagnosis of a mental illness/mental health disorder/behavioral health disorder |
| asylum              | psychiatric hospital/facility |
| drug user / abuser <br>addict  | person who uses drugs <br>person who injects drugs <br> person with substance use disorder|
| alcoholic <br> alcohol abuser  | person with alcohol use disorder <br> person in recovery from substance use/alcohol disorder |
| person who relapsed | person who returned to use   |
| smoker             | person who smokes             |
| homeless people <br> the homeless <br> transient population | people without housing <br>people experiencing homelessness <br>people experiencing unstable housing/housing insecurity/people who are not securely housed <br>people experiencing unsheltered homelessness <br>clients/guests who are accessing homeless services <br>people experiencing houselessness <br> people experiencing housing or food insecurity |
| prostitute         | person who engages in sex work <br> sex worker (abbreviated as SWer) |
| prisoner <br>convict | person who is/has been incarcerated |
| slave                | person who is/was enslaved |

### General Language Suggestions

(Thanks to Chris Calloway and Renci for the following instructions.)

Shortest sentences are the best. Some ways to shorten the sentences and make those professional are as follows:

1. Avoid "which" and "that". For example, don’t use – "the model that we trained". Instead use – "we trained a model"
2. Avoid using pronouns like "I", "we", etc. For example, avoid "we trained the model"; instead use "the trained model …"
3. Avoid passive voice.
4. Avoid using questions, use statements instead. For example, avoid "which metrics would be useful for success"; instead use "the success metrics here are …"
5. Avoid words, verbs with emotions. For example, avoid "reflecting on; theme"; use "in short, or summarizing; topic"
