:author: James Bergstra
:email: james.bergstra@umontreal.ca
:institution: University of Montreal

:author: Olivier Breuleux
:email: breuleuo@iro.umontreal.ca
:institution: University of Montreal

:author: Frédéric Bastien
:email: bastienf@iro.umontreal.ca
:institution: University of Montreal

:author: Pascal Lamblin
:email: lamblinp@iro.umontreal.ca
:institution: University of Montreal

:author: Razvan Pascanu
:email: pascanur@iro.umontreal.ca
:institution: University of Montreal

:author: Guillaume Desjardins
:email: desjagui@iro.umontreal.ca
:institution: University of Montreal

:author: Joseph Turian
:email: turian@iro.umontreal.ca
:institution: University of Montreal

:author: David Warde-Farley
:email: wardefar@iro.umontreal.ca
:institution: University of Montreal

:author: Yoshua Bengio
:email: yoshua.bengio@umontreal.ca
:institution: University of Montreal

--------------------------------------------------------------------
Theano: A CPU and GPU Math Compiler in Python
--------------------------------------------------------------------

.. class:: abstract

    *Theano is a compiler for mathematical expressions in Python that
    combines the convenience of NumPy's syntax with the speed
    of optimized native machine language.
    The user composes mathematical expressions in a high-level
    description that mimics NumPy's syntax and semantics, while being statically
    typed and functional (as opposed to imperative).
    These expressions allow Theano to provide symbolic differentiation.
    Before performing computation, Theano optimizes the choice of expressions,
    translates them into C++ (or CUDA for GPU),
    compiles them into dynamically loaded Python modules, all automatically.
    Common machine learning algorithms implemented with Theano
    are from* :math:`$1.6\times$` *to* :math:`$7.5\times$` *faster
    than competitive alternatives (including those implemented with
    C/C++, NumPy/SciPy and Matlab) when compiled for the CPU
    and between* :math:`$6.5\times$` *and* :math:`$44\times$` *faster
    when compiled for the GPU.
    This paper illustrates how to use
    Theano, outlines the scope of the compiler, provides benchmarks
    on both CPU and GPU processors, and explains its overall design.*



Introduction
------------

Python is a powerful and flexible language for describing large-scale mathematical
calculations, but the Python interpreter is in many cases a poor engine for executing
them. One reason is that Python uses full-fledged Python objects on the heap to
represent simple numeric scalars.
To reduce the overhead in numeric calculations, it is important to use array
types such as NumPy's ``ndarray`` so that single Python objects on the heap can
stand for multidimensional arrays of numeric scalars, each stored efficiently in
the host processor's native format.

[NumPy]_ provides an N-dimensional array data type, and many functions
for indexing, reshaping, and performing elementary computations (``exp``, ``log``,
``sin``, etc.) on entire arrays at once. These functions are implemented in C for
use within Python programs. However, the composition of many such NumPy functions
can be unnecessarily slow when each call is dominated by the cost of transferring
memory rather than the cost of performing calculations [Alted]_.
[numexpr]_ goes one step further by providing a loop fusion optimization
that can glue several elementwise computations together.
Unfortunately, the syntax required by numexpr is a bit unusual (the expression
must be encoded as a string within the code), and at the time of this writing,
numexpr is limited to optimizing elementwise computations.  [Cython]_ and
[scipy.weave]_ address Python's performance issue by offering a simple way to
hand-write crucial segments of code in C (or a dialect of Python which can be
easily compiled to C, in Cython's case). While this approach can yield
significant speed gains, it is labor-intensive: if the bottleneck of a program
is a large mathematical expression comprising hundreds of elementary
operations, manual program optimization can be time-consuming and error-prone,
making an automated approach to performance optimization highly desirable.

Theano, on the other hand, works on a symbolic representation of mathematical
expressions, provided by the user in a NumPy-like syntax.  Access to the full
computational graph of an expression opens the door to advanced features such
as symbolic differentiation of complex expressions, but more importantly allows
Theano to perform local graph transformations that can correct many unnecessary,
slow or numerically unstable expression patterns.  Once optimized, the same
graph can be used to generate CPU as well as GPU implementations (the latter
using CUDA) without requiring changes to user code.

Theano is similar to [SymPy]_, in that both libraries manipulate symbolic
mathematical graphs, but the two projects have a distinctly different focus.
While SymPy implements a richer set of mathematical operations of the kind
expected in a modern computer algebra system, Theano focuses on fast, efficient
evaluation of primarily array-valued expressions.

Theano is free open source software, licensed under the New (3-clause) BSD license.
It depends upon NumPy, and can optionally use SciPy, as well as custom C and CUDA code
generators which are able to specialize for particular types, sizes, and shapes of
inputs. It can be easily extended with custom graph components, known as "ops", which can
leverage ``scipy.weave``, PyCUDA, Cython, and other
numerical libraries and compilation technologies at the user's discretion. Theano has been actively and
continuously developed and used since January 2008.
It has been used in the preparation of numerous scientific papers and as a teaching platform for machine
learning in graduate courses at the Université de Montréal.
Documentation and installation instructions can be found on Theano's website [theano]_.
All Theano users should subscribe to the
`announce <http://groups.google.com/group/theano-announce>`_ [#]_ mailing list
(low traffic). There are medium traffic mailing lists for
`developer discussion <http://groups.google.com/group/theano-dev>`_ [#]_
and `user support <http://groups.google.com/group/theano-users>`_ [#]_.

This paper is divided as follows:
`Case Study: Logistic Regression`_ shows how Theano can be used to solve
a simple problem in statistical prediction.
`Benchmarking Results`_ presents some results of performance
benchmarking on problems related to machine learning and expression evaluation.
`What's in Theano`_ gives an overview of the design of Theano.
`Limitations and Future Work`_ outlines current limitations
and planned future work.

.. [#] http://groups.google.com/group/theano-announce
.. [#] http://groups.google.com/group/theano-dev
.. [#] http://groups.google.com/group/theano-users

.. _example1:

.. _caseStudy:

Case Study: Logistic Regression
------------------------------------------

To get a sense of how Theano feels from a user's perspective,
we will look at how to solve a binary logistic regression problem.
Binary logistic regression is a classification model
parameterized by a weight matrix :math:`$W$` and
bias vector :math:`$b$`.
The model estimates the probability
:math:`$P(Y=1|x)$` (which we will denote with shorthand :math:`$p$`) that the input
`x` belongs to class `$y=1$` as:

.. raw:: latex

    \begin{equation}
    P(Y=1|x^{(i)}) = p^{(i)} = \frac {e^{W x^{(i)} + b}} {1 +  e^{Wx^{(i)} + b}}
    \end{equation}

The problem is to optimize the log probability of :math:`$N$` training examples,
:math:`$\mathcal{D} = \{(x^{(i)},y^{(i)}) , 0 < i \leq N\})$`,
with respect to :math:`W` and :math:`b`. To maximize the log likelihood we
will instead minimize the (average) negative log likelihood [#]_:

.. raw:: latex

    \begin{equation}
    \ell(W,b) = -\frac{1}{N}\sum_i y^{(i)} \log p^{(i)} + (1-y^{(i)}) \log (1 - p^{(i)})
    \end{equation}

.. [#] Taking the mean in this fashion decouples the choice of the regularization coefficient and the stochastic gradient step size from the number of training examples.

To make it a bit more interesting, we can also include an
:math:`$\ell_2$` penalty on :math:`$W$`, giving a cost function :math:`$E(W,b)$` defined as:

.. raw:: latex

    \begin{equation}
    E(W,b) = \ell(W, b) + 0.01 \sum_i \sum_j w_{ij}^2
    \end{equation}

In this example, tuning parameters :math:`W` and :math:`b` will be done through
stochastic gradient descent (SGD) of :math:`$E(W,b)$`, even though more sophisticated
algorithms could also be used. Implementing this minimization procedure in
Theano involves the following four conceptual steps:
(1) declaring symbolic variables,
(2) using these variables to build a symbolic expression graph,
(3) compiling a Theano function, and
(4) calling said function to perform numerical computations.
We will now step through each of these sections in more detail.


.. raw:: latex

    \begin{figure}[H]
        \includegraphics[scale=.75]{logreg1.pdf}
    \end{figure}

In the above code, we declare two symbolic variables ``x`` and ``y`` which will
serve as input to the rest of the computation graph. Theano variables are
strictly typed and include the data type, the number of dimensions, and the
dimensions along which it may broadcast in element-wise expressions. Here we
define ``x`` to be a matrix of the default data type (``float64``), where each
row of ``x`` corresponds to an example :math:`$x^{(i)}$`. Similarly, ``y`` is
declared as a vector of type ``int32`` whose entries correspond to the labels
:math:`$y^{(i)}$`. Treating multiple data points at the same time allows us to
implement SGD with mini-batches, a variant of SGD which is both computationally
and statistically efficient.

The ``shared()`` function creates *shared variables* for :math:`$W$` and :math:`$b$` and assigns them initial values.
Shared variables are similar to standard Theano variables, but differ in that
they have a persistent state. As we will see shortly, any Theano function can
operate directly on these shared variables, without having to declare them
explicitly as an input.
A shared variable's value is maintained
throughout the execution of the program and
can be accessed with ``.get_value()`` and ``.set_value()``, as shown in line 11.

.. raw:: latex

    \begin{figure}[H]
        \includegraphics[scale=.75]{logreg2.pdf}
    \end{figure}

The above code-block specifies the computational graph required to perform
gradient descent of our cost function. Since Theano's interface shares much in
common with that of ``numpy``, lines 13-17 should be self-explanatory for anyone
familiar with ``numpy``. On line 13, we start by defining :math:`$P(Y=1|x^{(i)}) = 1$`
as the symbolic variable ``p_1``. Notice that the dot product and element-wise exponential
functions are simply called via the ``T.dot`` and ``T.exp`` functions,
analoguous to ``numpy.dot`` and ``numpy.exp``. ``xent`` defines the
cross-entropy loss function, which is then combined with the :math:`$\ell_2$`
penalty on line 15, to form the cost function of Eq (2) and denoted by ``cost``.

Line 16 is crucial to our implementation of SGD, as it performs symbolic
differentiation of the scalar-valued ``cost`` variable with respect to variables
``w`` and ``b``.  ``T.grad`` operates by iterating backwards over the expression
graph, applying the chain rule of differentiation and building symbolic
expressions for the gradients on ``w`` and ``b``. As such, ``gw`` and ``gb`` are
also symbolic Theano variables, representing :math:`$\partial E / \partial W$` 
and :math:`$\partial E / \partial b$` respectively.
Finally, line 18 defines the actual prediction (``prediction``) of the logistic
regression by thresholding :math:`$P(Y=1|x^{(i)})$`.


.. raw:: latex

    \begin{figure}[H]
        \includegraphics[scale=.75]{logreg3.pdf}
    \end{figure}

The above code defines two Theano functions which are required to learn and
test our logistic regression module. Theano functions are in their simplest
form, callable objects which compute the value of certain nodes in the
computation graph, given values for the symbolic inputs indicated. For example, the
``predict`` function computes the actual output of the logistic regression
module (``prediction``). Since this value is a function of both ``x`` and ``y``,
these are given as input to the function. Parameters ``w`` and ``b`` are passed
implicitly, as is always the case with shared variables.

``train`` highlights two other important features of Theano functions. Firstly,
functions can compute multiple outputs. In this case, ``train`` computes both
the prediction (``prediction``) of the classifier as well as the cross-entropy
error function (``xent``). Computing both outputs together is computationally
efficient since it allows for sharing of all intermediate computations.
Secondly, the ``updates`` keyword argument enables functions to have
side-effects. It is a dictionary whose (key,value) pairs encode an update
to perform on a shared variable. This update is executed each time the
associated function is called. In this example, calling the ``train`` function
will also update the parameters ``w`` and ``b``, with the value obtained after a
single step of gradient descent. The update on ``w`` thus corresponds to the
expression 

:math:`$W \leftarrow W - \mu \frac{1}{N'} \sum_i \left. \frac{\partial E(W,b,x,y)}{\partial W} \right |_{x=x^{(i)},y=y^{(i)}}$`,

where :math:`$\mu=0.1$` is the learning rate and :math:`$N'$` the size of the
minibatch (number of rows of ``x``).


.. raw:: latex

    \begin{figure}[H]
        \includegraphics[scale=.75]{logreg4.pdf}
    \end{figure}

In this code-block, we finally show how Theano functions are used to perform the
task at hand. We start by generating four random training examples: ``D[0]``
is the input ``x`` and ``D[1]`` the labels we must learn to predict. We then
loop (lines 30-31) ten times, calling the ``train`` function repeatedly with
inputs ``x=D[0]`` and ``y=D[1]``. Notice that calling a Theano function is no
more complicated than calling a standard Python function: the graph
transformations, optimizations, compilation and calling of efficient C-functions
(whether targeted for the CPU or GPU) are all done under the hood, in a way
which is transparent to the user. Finally, we print the state of the model
parameters and show that the model accurately predicts the training labels.



.. _benchmark:

Benchmarking Results
--------------------

Theano started as a library for easing rapid development of complex machine 
learning algorithms. This section presents performance in two tasks from that
domain: training a multi-layer perceptron (MLP) and training a convolutional
network. More extensive benchmarks are forthcoming, and will be posted on our
website [theano]_.

We chose these architectures because of their popularity in the machine learning
community and their different computational demands. Large matrix-matrix
multiplications dominate in the MLP example while two-dimensional image
convolutions with small kernels dominate the convolutional network.
More information about these models and their learning algorithms is available 
from the Deep Learning Tutorials [DLT]_. 
The implementations used in these benchmarks are available online [dlb]_.

CPU timing was carried out on an
a Intel(R) Core(TM)2 Duo CPU E8500 @ 3.16GHz with 2 GB of RAM. 
All implementations were linked against the BLAS implemented in the Intel Math
Kernel Library, version 10.2.4.032 and allowed to use only one thread.
GPU timing was done on a GForce GTX 285.
CPU computations were done at double-precision.
GPU computations were done at single-precision.

Our first benchmark is training
a single layer MLP by mini-batch gradient descent.
Each implementation repeatedly carried out the following steps:
(1) multiply 60 784-element input vectors by a :math:`$784 \times 500$` weight matrix,
(2) compress the result by tanh,
(3) multiply the result by a :math:`$500 \times 10$` matrix,
(4) classify the result using a multi-class generalization of logistic regression,
(5) compute the gradient by performing similar calculations but in reverse, and finally
(6) add the gradients to the parameters.
This program tests the use of BLAS routines and elementwise computations.

.. _Figure 3:
.. _Benchmark1:
.. figure:: mlp.pdf
    :scale: 100

    **Figure 3:** Fitting a multi-layer perceptron to simulated data with 
    various implementations of stochastic gradient descent.  These models have
    784 inputs, 500 hidden units, a 10-way classification, and are trained 60
    examples at a time.

`Figure 3`_ looks at the number of examples processed per second 
by different implementations. We compared Theano (revision #ec057beb6c) against
NumPy 1.4.1, Matlab 7.9.0.529, and Torch 5 (a machine learning 
library written in C/C++) [torch5]_ on the CPU and  GPUMat 0.25 for Matlab
([gpumat]_) on the GPU.

When running on the CPU, Theano is 1.8x faster than NumPy,
1.6x faster than Matlab, and 7.5x faster than Torch 5. Torch was written
for flexibility, not speed (Ronan Collobert, p.c.).
Theano's speed increases 5.8x on the GPU from the CPU, a total increase of 11x over
NumPy (CPU) and 44x over Torch 5 (CPU).
GPUmat brings about a speed increase of only 1.4x when switching to the GPU
for the Matlab implementation, far
less than the 5.8x increase Theano achieves through CUDA specializations.

.. [#] Torch was designed and implemented with flexibility in mind, not speed (Ronan Collobert, p.c.).

.. _Benchmark2:
.. _Figure 4:
.. figure:: conv.pdf
    :scale: 100

    **Figure 4:** Fitting a convolutional network using different
    software. The benchmark stresses convolutions of medium-sized (256 by 256) images with
    small (7 by 7) filters.


Because of the difficulty in implementing efficient convolutional networks, we only
benchmark against known libraries that offer a pre-existing implementation.
We compare against EBLearn [EBL]_ and Torch, two libraries written in C++. 
EBLearn was implemented by Yann LeCun's lab at NYU, which has done extensive
research in convolutional networks.
To put these results into perspective, we implemented approximately half (no
gradient calculation) of the algorithm using SciPy's ``signal.convolve2d`` function. 
This benchmark uses convolutions of medium sized images
(:math:`$256 \times 256$`) with
small filters (:math:`$7 \times 7$`).
`Figure 4`_ shows the performance of Theano (both CPU and GPU)
against competing implementations.
On the CPU, Theano is 2.2x faster than EBLearn, its best competitor. This is because
Theano compiles more specialized convolution routines.
Theano's speed increases 4.9x on the GPU from the CPU, a total of 10.7x over
EBLearn (CPU).
On the CPU, Theano is 5.8x faster than SciPy even though SciPy is doing only
half the computations. This is because SciPy's convolution routine has not been
optimized for this application.

We also compared Theano with numexpr and NumPy for evaluating element-wise
expressions on the CPU (`Figure 5`_).
For small amounts of data, the extra function-call overhead of numexpr and
Theano makes them slower.  For larger amounts of data, and for more complicated
expressions, Theano is fastest because it uses an implementation specialized for
each expression.

.. _Figure 5:
.. _Benchmark3:
.. figure:: multiple_graph.pdf
    :scale: 100

    **Figure 5:** Speed comparison between NumPy,
    numexpr, and Theano for different sizes of input on four elementwise
    formulae.  In each subplot, the solid blue line represents Theano, the
    dashed red line represent numexpr, and performance is plotted with respect
    to NumPy.

.. _What's in Theano:
.. _intheano:

What's in Theano?
-----------------

Theano supports arrays of different dimensions 
(from scalar to n-dimensional tensors) and types (int, 
single-precision floats, double-precision floats etc.) as 
well as random streams of numbers (much as Numpy does). 
There is also limited support for sparse matrices and 
generic objects. `Table 1`_ presents 
a comprehensive list of operations that you would find 
in Theano. It also supports debugging and profiling functionalities.

.. _Table 1:
.. _Table1:

.. raw:: latex

    \begin{center}
    \begin{table}
    \centering \small
    \begin{tabular}{|p{1.6cm}|p{5.7cm}|}
    \hline
    Operators              &    {\tt +}, {\tt -}, {\tt /}, {\tt *}, {\tt **}, {\tt //},
                                {\tt eq}, {\tt neq}, {\tt <}, {\tt <=}, {\tt >}, {\tt >=},
                                {\tt \&}, \verb'|', \verb'^' 
                                \tabularnewline
                           &
                                \tabularnewline
    Allocation             &    {\tt alloc}, {\tt eye}, {\tt [ones,zeros]\_like},
                                {\tt identity\{\_like\} }
                                \tabularnewline
                           & 
                                \tabularnewline
    Indexing*              &    basic slicing (see {\tt set\_subtensor} and 
                                {\tt inc\_subtensor} for slicing lvalues);
                                limited support for advanced indexing
                                \tabularnewline
                           & 
                                \tabularnewline
    Mathematical \newline Functions        &    {\tt exp}, {\tt log}, {\tt tan[h]}, {\tt cos[h]}, {\tt sin[h]}, 
                                {\tt real}, {\tt imag}, {\tt sqrt}, {\tt floor}, {\tt ceil}, 
                                {\tt round}, {\tt abs}
                                \tabularnewline
                           &  
                                \tabularnewline
    Tensor \newline Operations      &    {\tt all}, {\tt any}, {\tt mean}, {\tt sum}, {\tt min}, {\tt max}, 
                                {\tt var}, {\tt prod}, {\tt argmin}, {\tt argmax},
                                {\tt reshape}, {\tt flatten},
                                {\tt dimshuffle}
                                \tabularnewline
                           &
                                \tabularnewline
    Conditional            &    {\tt cond}, {\tt switch}
                                \tabularnewline
                           & 
                                \tabularnewline
    Looping                &    {\tt Scan}
                                \tabularnewline
                           &
                                \tabularnewline
    Linear Algebra         &     {\tt dot}, {\tt outer}, {\tt tensordot}
                                \tabularnewline
                           & 
                                 \tabularnewline
    Calculus*              &     {\tt grad}
                                \tabularnewline
                           &
                                \tabularnewline
    Signal \newline Processing      &    {\tt conv2d}, {\tt FFT}, {\tt max\_pool\_2d}
                                \tabularnewline
                           &
                                \tabularnewline
    Random                 &    {\tt RandomStreams}, {\tt MRG\_RandomStreams}
                                \tabularnewline
                           &
                                \tabularnewline
    Printing               &    {\tt Print}
                                \tabularnewline
                           & 
                                \tabularnewline
    Sparse                 &    compressed row/col storage,
                                limited operator support,
                                {\tt dot}, {\tt transpose},
                                conversion to/from dense
                                \tabularnewline
    \hline
    \end{tabular}
    \caption{
    Overview of Theano's core Types and Ops set.
    This list is not exhaustive, and is superseded by the
    online documentation. More details are given in text for items marked with
    an asterisk. {\tt dimshuffle} is like {\tt numpy.swapaxes}.
    }
    \end{table}
    \end{center}

    \vspace{-1cm}

Ops & Functionality
~~~~~~~~~~~~~~~~~~~

*Ops* are objects that define computations.
Most of the ops (e.g. ``add``, ``exp``) behave like NumPy counterparts.
`Table 1`_ lists the core functionality offered by Theano's
Ops. More extensive reference documentation is available online
[theano]_.

Allocating random number variables
and seeding generators is typically done via a ``RandomStreams`` instance, which
replicates the ``numpy.random.RandomState`` interface
and wraps ``numpy.random.RandomState`` functionality.
Theano also provides an experimental new ``MRG_RandomStreams`` generator which
provides a few distributions using an ``MRG`` algorithm with both a CPU and GPU
implementation [Ecu]_.


There is a narrower range of Ops that work on SparseType Variables: packing and
unpacking of compressed sparse row/column
sparse matrices into dense variables is supported,
as is conversion between sparse and dense matrices.  Transpose, negation,
addition, and subtraction are supported.  Scalar and elementwise multiplication
with a dense matrix is supported, and matrix multiplication between sparse and
dense is supported.

Roughly 90\% of Ops for tensors have implementations for the GPU, notable
exceptions being advanced indexing, summation over certain combinations of
axes, and reductions max, min and prod.
Our goal is extend coverage to all ops.

Theano does *not* currently have ops for sparse or dense matrix inversion, nor linear
algebra decompositions.  Ops for complex number dtypes are also not as widely
implemented or well-tested as those for integer and float dtypes. Object dtypes
are not implemented in Theano.


Transformations
~~~~~~~~~~~~~~~~

Theano uses graph transformations to implement a range of tasks from
merging redundant calculations to transferring computations to the
GPU. The optimization of expression graphs is a pipeline comprising
several stages.

The first stage merges duplicate expressions, so as to only compute
them once. Two expressions are considered duplicates if they carry out
the same operation and that all inputs are the same - since Theano is
purely functional, these expressions must return the same value and
thus the operation is safe to carry. The symbolic gradient mechanism
often introduces redundancy, so this phase is quite important.

The second stage transforms the expression into an equivalent
canonical form. For example, sub-expressions involving only
multiplication and division are put into a standard fraction form
(e.g. ``a / (((a * b) / c) / d) -> (a * c * d) / (a * b) -> (c * d) /
(b)``). Some useless calculations are eliminated in this phase, for
instance crossing out uses of the ``a`` term in the previous example,
reducing ``exp(log(x))`` to ``x``, and doing constant
folding. Furthermore, since the canonicalization collapses many
different expressions into a single normal form, it becomes easier to
define reliable transformations on them.

The third stage transforms expressions to improve numerical
stability. For instance, consider the function ``log(1 + exp(x))``,
which is zero in the limit of negative ``x`` and ``x`` in the limit of
large ``x``. Due to limitations in the representation of double
precision numbers, the expression yields infinity for ``x >
709``. Theano is able to identify this pattern and replace it with an
implementation that simply returns ``x`` if ``x`` is sufficiently
large (using doubles, this is accurate beyond the least significant
digit).

The fourth stage specializes generic expressions and subgraphs.
Expressions like ``pow(x,2)`` become ``sqr(x)``. Theano also performs
more elaborate specializations: for example, expressions involving
scalar-multiplied matrix additions and multiplications may become BLAS
General matrix multiply (GEMM) nodes and ``reshape``, ``transpose``,
and ``subtensor`` Ops (which create copies by default) are replaced by
constant-time versions that work by aliasing memory.

After this stage of specialization, Sub-expressions involving
element-wise operations are fused together in order to avoid the
creation of unnecessary temporaries. For instance, denoting the ``a +
b`` operation on tensors as ``map(+, a, b)``, then an expression such
as ``map(+, map(*, a, b), c)`` would become ``map(lambda ai,bi,ci:
ai*bi+ci, a, b, c)``. If the user desires to use the GPU, Ops with
corresponding GPU implementations are substituted in, and transfer Ops
are introduced where needed.

Lastly, Theano replaces Ops with equivalents that reuse the memory of
their inputs (which means, as a side effect, that no subsequent Ops
may use the original values). Many Ops (e.g. GEMM and all elementwise
Ops) have such equivalents.  Reusing memory this way can improve speed
by reducing cache misses and allowing more computations to fit on GPUs
where memory is at a premium.

.. verify with Fred

Code Generators
~~~~~~~~~~~~~~~~


Many (roughly 80%) of Theano's Ops generate and compile C or CUDA code during
``theano.function``.
The majority of Ops (such as all elementwise Ops and ``Sum``) that generate C code specialize the code based on the dtype and
number of dimensions of their arguments.
Some Ops, such as the small-filter convolution (``conv2d``), further specialize code based on
the size the arguments will have.

Modern x86 architectures are relatively forgiving if code is not perfectly
specialized to the input dimensions, and only the ``conv2d`` Op goes to any great
length to generate many special case implementations for the CPU.
By comparison, GPU architectures are much less forgiving of code that is not carefully specialized
for the size and physical layout of function arguments.
Theano's code generators for ``GpuSum``, ``GpuElementwise``, and ``GpuConv2d``
generate a wider variety of implementations than
their respective CPU-targeting Ops.
The difference in speed on a GPU between 
a naïve and an optimal implementation of even a simple algorithm like row/column
summation in a matrix can be an order of magnitude or more.
Theano's ability to generate custom-made CUDA kernels for many important
mathematical operations accounts for the good GPU performance in our benchmarks.

Moving Computation to the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each expression in Theano is associated with an implementation that runs on
either the host (a host expression) or a GPU device (a GPU expression).
One important application of graph transformations is to replace host
expressions with GPU expressions.
The majority of host expression types have GPU equivalents and the proportion is
always growing.

The heuristic that guides GPU allocation is simple:
if any input or output of an expression resides on the GPU and the expression
has a GPU equivalent, then we replace it.
How does this chain reaction get started?
Shared variables storing float32 tensors default to GPU storage,
and the expressions derived from them consequently default to using GPU
implementations.
It is possible to explicitly force any float32 variable to reside on the GPU,
so you can start the chain reaction of optimizations and use the GPU even
in graphs with no shared variables.
It is possible (though awkward, and discouraged)
to specify exactly which computations to perform on the GPU
by disabling the default GPU optimizations.

Tensors stored on the GPU use a special internal data type with an interface
similar to the ``ndarray``.
This datatype fully supports strided tensors, and
arbitrary numbers of dimensions.
The support for strides means that several operations such as the transpose and
simple slice indexing can be performed in constant time.


Limitations and Future Work
---------------------------

While most of the development effort went into making Theano produce fast code,
not as much went into optimizing the compilation process itself, thus 
the compilation time tends to grow super-linearly with the size of 
the expression graph. Theano can deal with graphs up to a few thousand
nodes, with compilation times typically in the range of seconds. Beyond 
that, it can be impractically slow, unless you disable some of the more 
expensive optimizations, or compile pieces of the graph separately. 

A Theano function call also requires more overhead (on the order of microseconds)
than a native Python function call. For this reason, Theano is suited to
applications where functions correspond to expressions that are not too
small (see `Figure 5`_).

The set of types and operations that Theano provides continues to grow, but it does not
cover all the functionality of NumPy and covers only a few features of SciPy.
Wrapping functions from these and other libraries is often straightforward,
but implementing their gradients or related graph transformations
can be more difficult.

We expect to improve support for advanced indexing and linear algebra in the
coming months. Documentation online describes how to add new operations, 
new type or new graph transformations. There are also experimental version
of the scan operation, used for looping, for the GPU and an experimental lazy-evaluation 
enabled Theano.

Also the library has been tuned towards expressions related to machine 
learning with neural networks, and it was not as well tested outside 
thist domain. Theano is not a powerful computer algebra system, and 
it is an important area of future work to improve its ability to recognize
numerical instability in complicated elementwise expression graphs.

Debugging Theano functions can require non-standard techniques and
Theano-specific tools.  The reason is two-fold: 1) definition
of Theano expressions is separate from their execution, and 2) optimizations
can introduce many changes to the computation graph.

We plan to extend GPU support to the full range of C data types, but only float32
tensors are supported as of writing.
There no support for sparse vectors or matrices on the GPU,
although algorithms from the CUSPARSE package should make it easy to add at least basic
support for sparse GPU objects.


Conclusion
------------

Theano is a mathematical expression compiler for Python 
that translates high level NumPy-like code
into machine language for efficient CPU and GPU computation.
Theano achieves good performance by minimizing the use
of temporary variables, minimizing pressure on fast memory caches,
making full use of ``gemm`` and ``gemv`` BLAS subroutines, and generating fast C code
that is specialized to sizes and constants in the expression graph.
Theano implementations of machine learning algorithms related to neural networks
on one core of an E8500 CPU are up to 1.8 times faster than implementations in NumPy, 1.6 times faster than
MATLAB, and 7.6 times faster than a related C++ library.  Using a Nvidia GTX285 GPU, Theano
is 5.8 times faster again.
One of
Theano's greatest strengths is its ability to generate custom-made CUDA
kernels, 
which can not only significantly outperform CPU implementations but alternative
GPU implementations as well.


Acknowledgements
----------------

Theano has benefited from the contributions of many members
of Yoshua Bengio's machine learning group in the computer science department
(Informatique) at the University of Montreal,
especially: 
Arnaud Bergeron, Thierry Bertin-Mahieux, Olivier Delalleau, 
Douglas Eck, Dumitru Erhan, Philippe Hamel, Simon Lemieux,
Pierre-Antoine Manzagol, and François Savard.
David Warde-Farley contributed to the preparation of this paper.
The authors acknowledge the support of the following agencies for
research funding and computing support: NSERC, RQCHP, CIFAR, SHARCNET and CLUMEQ.

References
----------

.. [theano] Theano, http://www.deeplearning.net/software/theano

.. [NumPy] T. E. Oliphant, Python for Scientific Computing, 
           Computing in Science & Engineering 9, 10 (2007).

.. [numexpr] D. Cooke et al., numexpr, http://code.google.com/p/numexpr/

.. [Cython] S. Behnel, R. Bradshaw, and D. S. Seljebotn, 
            Cython C-Extensions for Python,
            http://www.cython.org/

.. [scipy.weave] SciPy Weave module, 
                 http://www.scipy.org/Weave

.. [Alted]  F. Alted, Why Modern CPUs Are Starving And What Can
    Be Done About It, Computing in Science and Engineering, 12(2):68-71, 2010.

.. [SymPy] SymPy, http://code.google.com/p/sympy/

.. [BLAS] J. J. Dongarra, J. Du Croz, I. S. Duff, and S. Hammarling, 
          Algorithm 679: A set of Level 3 Basic Linear Algebra Subprograms, ACM Trans. Math. Soft., 16:18-28, 1990. 
          http://www.netlib.org/blas

.. [LAPACK] E. Anderson et al., 
            LAPACK Users' Guide Third Edition,
            http://www.netlib.org/lapack/lug/index.html

.. [DLT] Deep Learning Tutorials, 
         http://deeplearning.net/tutorial/

.. [dlb] Benchmarking code, 
         http://github.com/pascanur/DeepLearningBenchmarks

.. [torch5] Torch 5, http://torch5.sourceforge.net

.. [EBL] EBLearn: Energy Based Learning, http://eblearn.sourceforge.net/

.. [gpumat] GPUmat: GPU toolbox for MATLAB, http://gp-you.org

.. [Ecu] P. L'Ecuyer, F. Blouin, and R. Couture,
         A Search for Good Multiple Recursive Generators,
         ACM Transactions on Modeling and Computer Simulation, 3:87-98, 1993. 
