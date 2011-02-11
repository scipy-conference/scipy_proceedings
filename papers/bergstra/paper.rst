:author: James Bergstra
:email: james.bergstra@umontreal.ca
:institution: University of Montreal

:author: Olivier Breuleux
:email: breuleuo@iro.umontreal.ca
:institution: University of Montreal

:author: Frederic Bastien
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
:email: dwfarley@iro.umontreal.ca
:institution: University of Montreal

:author: Yoshua Bengio
:email: yoshua.bengio@umontreal.ca
:institution: University of Montreal

--------------------------------------------------------------------
Theano: A CPU and GPU Math Compiler in Python
--------------------------------------------------------------------

.. class:: abstract


    *Theano is a compiler for mathematical expressions in Python. It combines the convenience of NumPy with the speed of optimized native machine language.
    The user writes mathematical expressions in a high-level
    description that mimics NumPy's syntax and semantics, while being statically typed and purely functional.
    Theano optimizes the expression, translates the optimized expression into C/C++, and runs an optimizing compiler on the C/C++ code, all automatically.
    Theano can also generate CUDA code to produce GPU implementations for those high-level description.
    Common machine learning algorithms
    are from* :math:`$1.6\times$` *to* :math:`$7.5\times$` *faster than competitive alternatives (including those in C/C++, NumPy,
    SciPy, and Matlab) when compiled for the CPU
    and between* :math:`$6.5\times$` *and* :math:`$44\times$` *faster when compiled for the GPU.
    Theano's speed comes from optimizing at a high level of granularity, namely over the symbolic graph representing a complicated mathematical expression.
    Theano's speed on GPUs also comes from its  ability to generate custom-made CUDA kernels for many important
    mathematical operations.
    Theano uses a library of graph transformation
    heuristics to optimize expression graphs for fast and
    numerically stable computation.
    Theano has been designed to hide the implementation details of a variety of backend technologies.
    With a single line of code, expressions in Theano can be automatically differentiated. These derivative expressions are also compiled by Theano.
    This paper illustrates how to use
    Theano, outlines the scope of the compiler,
    provides benchmarks on both CPU and GPU processors, and explains its overall design.
    Theano development and use began in January 2008.*



Introduction
------------

Python is a great language for describing large-scale mathematical calculations in a high level way,
but the Python interpreter is not a good engine for executing them. Python
numbers are full-fledged objects on the heap, but large-scale mathematical
calculations should be carried out using a processor's native types with minimal overhead.
There are several options available to a Python programmer to get the speed 
of native machine-language code for numerical calculations, including [NumPy]_, [numexpr]_, [Cython]_, and [scipy.weave]_.
[NumPy]_ provides an N-dimensional array data type, and many functions
for indexing, reshaping, and performing elementary computations (``exp``, ``log``, ``sin``, etc.)
on entire arrays at once. These functions are implemented in C for use within Python programs.
However, the composition of such NumPy functions
can be unnecessarily slow when each call is dominated by the cost of transfering memory rather than the cost of performing calculations [Alted]_.
As a partial solution, [numexpr]_ provides faster implementations than NumPy when
several elementwise computations are composed, by implementing a loop fusion optimization.
However,
numexpr requires unusual syntax (the expression must be encoded as a string within the code),
and it is limited to elementwise computations.
In many use-cases, hand-written native machine language
implementations are much faster than these common alternatives.
[Cython]_ and [scipy.weave]_ make it easier to write just the crucial bottleneck code in C,
and keep the rest of the program in Python. However, if the bottleneck
is a large mathematical expression comprising hundreds of operations,
manual optimization of the math can be time-consuming and suboptimal
compared to automatic optimization.

Theano combines the convenience of NumPy with the speed of hand-optimized
C/C++ code by generating and compiling native implementations
from a complete, high-level description of the computation graph, which it
optimizes before translating it into C/C++ code.
In our benchmarks, this can lead
to significant performance gains over competing math libraries.
Theano is capable of generating CPU as well as GPU implementations
(the latter using CUDA), without requiring changes to user code.
It supports full program transformations and macros,
including automatic differentiation.
It also performs more local transformations that correct many unnecessary, slow, or numerically unstable
expression patterns.
This pattern is appropriate for applications where the same computation will be repeated many times on different inputs,
so that the time invested in the initial optimization and compilation (typically on the order of seconds) is
negligible in comparison to time saved on the repeated calculations.
Theano is similar to [SymPy]_, in that both manipulate symbolic
mathematical graphs. SymPy implements a more extensive set of mathematical
operations, but it does not offer as efficient numerical evaluation.

Theano is free open source software, licensed under the New (3-clause) BSD license.
It depends upon NumPy, and can optionally use SciPy, as well as custom C and CUDA code generators which are able to specialize for particular types, sizes, and shapes of inputs. 
Theano can be extended to use ``scipy.weave``, PyCUDA, Cython, and other
numerical libraries and compilation technologies.
Theano has been actively and continuously developed and used since January 2008.
It has been used in
several scientific papers and it is used to teach machine learning in
graduate courses at the Université de Montréal.
Documentation and installation instructions can be found on Theano's website [theano]_.
All Theano users should subscribe to the
`announce <http://groups.google.com/group/theano-announce>`_ [#]_ mailing list (low traffic).
There are medium traffic mailing lists for `developer discussion <http://groups.google.com/group/theano-dev>`_ [#]_ and `user support <http://groups.google.com/group/theano-users>`_ [#]_.

This paper is divided as follows:
Section `Case Study: Logistic Regression`_ shows how Theano can be used to solve
a simple problem in statistical prediction.
GPU use, and some of the expression transformations Theano performs.
Section `Benchmarking Results`_ presents some results of performance
benchmarking on problems related to machine learning and expression evaluation.
Section `What's in Theano`_ gives an overview of the design of Theano.
Section `Limitations and Future Work`_ outlines current limitations
outlines planned future work.

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
parametrized by a weight matrix :math:`W` and
bias vector :math:`b`.
The model estimates the probability
:math:`$P(Y=1|x)$` (which we will denote with shorthand :math:`$p$`) that the input `x` belongs to class `y=1` as:

.. raw:: latex

    \begin{align}
    P(Y=1|x^{(i)}) &= p^{(i)} = \frac {e^{W x^{(i)} + b}} {1 +  e^{Wx^{(i)} + b}}
    \end{align}

The problem is to optimize the log probability of
:math:`N` training examples, :math:`$\mathcal{D} = \{(x^{(i)},y^{(i)}) , 0 < i \leq N\})$`,
with respect to :math:`W` and :math:`b`.
To make it a bit more interesting, we can also include an
:math:`$\ell_2$` penalty on :math:`$W$`, giving a cost function defined as:

.. raw:: latex

    \begin{align*}
    cost = 0.01 \cdot W^2 - \frac{1}{N} \sum_i ( \ & y^{(i)} \cdot p^{(i)} + \\
        & (1-y^{(i)}) \cdot (1 - p^{(i)}) )
    \end{align*}

Tuning parameters :math:`W` and :math:`b` to minimize this cost can be
performed by more sophisticated algorithms, but for our example we will
use stochastic gradient descent.

.. _Listing 2:
.. _ListingLogReg:

.. figure:: logreg.pdf
    :scale: 100

    **Listing 2:** A Theano program for fitting and 
    applying a logistic regression model.

The code in `Listing 2`_ implements this minimization.
The code is organized into four conceptual steps with respect to Theano:
  1. declare symbolic variables
  2. use these variables to build a symbolic expression graph,
  3. compile a function, and
  4. call the compiled function to perform computations.

Lines 7-10 declare the symbolic inputs for our logistic regression problem.
Notice that ``x`` is defined as a matrix, and ``y`` as a vector.
The Type of a Theano variables includes its number of dimensions,
its data type,
and the dimensions along which it may broadcast in element-wise expressions.
Here, ``x`` and ``y`` have the default data type which is ``float64``.

We did we not make ``x`` a vector and ``y`` a scalar, because it would limit the
speed of the program.
Matrix-matrix multiplication is more efficient on modern x86
architecture than matrix-vector multiplication
and Theano function calls involve overhead.
Treating several examples in parallel mitigates that overhead.

The ``shared()`` function (Lines 9+10 of `Listing 2`_) creates *shared variables* for :math:`$W$` and :math:`$b$` and assigns them initial values.
Shared variables are
similar to standard Theano variables, but are stateful. In
a sense, they behave like global variables which any Theano function
may use without having to declare them in its inputs list.
A shared variable's value is maintained
throughout the execution of the program and
can be accessed with ``.get_value()`` and ``.set_value()``, as shown in Line 12.
Theano manages the storage of
these values. In particular, it stores single-precision dense *shared* tensors on the GPU by
default when a GPU is available.  In such cases it uses a different
Theano-specific data type for internal storage in place of the NumPy ``ndarray``.

Line 15 defines :math:`$P(Y=1|x^{(i)}) = 1$` as ``p_1``.
Line 16 defines the cross-entropy term in :math:`cost` as ``xent``.
Line 17 defines the predictor by thresholding over :math:`$P(Y=1|x^{(i)}) = 1$` as ``prediction``.
Line 18 defines :math:`cost` as ``cost``, by adding the cross-entropy term to the :math:`$\ell_2$` penalty.

Line 19 (``gw,gb = T.grad(cost, [w,b])``) performs automatic
differentiation of scalar-valued ``cost`` with respect to variables ``w`` and ``b``.
It works like a macro, iterating backward over the expression
graph, applying the chain rule of differentiation and building expressions for the
gradients on ``w`` and ``b``.

Lines 22-25 (``train = function...``) introduce the ``updates`` argument to ``function``.
An update is an expression that will be computed by the function, like a return
value, but the computed result is stored in a shared variable instead of returned to the caller.
On a GPU, this means that a shared variable and its updated value can all reside
on the device. Having both on the device can be
important for performance, because it is slow to copy between the host and the GPU.
Here we adjust ``w`` and ``b`` by their gradients, the direction that causes the cost to drop most sharply. This update step implements stochastic gradient descent.

Line 26 compiles a second function (``predict = function...``) from the same expression graph.
This is a standard pattern when using Theano - we define one big
expression graph that corresponds to some application domain, and then compile
several functions from it to compute various sub-regions of the graph. Note that
all these functions may read and write the states of the various shared variables,
hence their name.

Lines 28-30 randomly generate four training examples, each with 100 feature values. 
(In practice, training examples would be inputs to the program.)
Line 31-33 runs the ``train`` gradient update step, ten times.
Lines 34-41 print some debug output.

Theano applies some graph transformations to optimize the ``train`` and ``predict``
functions for speed and numerical stability, when compiling them in Lines 22-25 and 26, respectively.
For example, in the ``predict``
function, ``1/(1+exp(-u))`` is recognized as the logistic sigmoid
function and replaced with an implementation that is faster for large positive
and negative values of ``u``.
All the element-wise operations are fused together after
the vector-matrix multiplication and compiled as a specialized C function with a
single loop over the data.  
In the ``train`` function, Theano additionally recognizes ``log(sigmoid(u))``
and ``log(1-sigmoid(u))`` as instances of the softplus function:
``log1p(exp(u))``, for which Theano has an implementation that avoids a
dangerous potential overflow.
When updating ``w`` with its new value, Theano also
recognizes that a single call to the BLAS ``dgemv`` routine can implement the
:math:`$\ell_2$`-regularization of ``w``, scale its gradient, 
and decrement ``w`` by its scaled gradient.

.. _benchmark:

Benchmarking Results
--------------------

Theano was developed to allow the rapid development of algorithms
in machine learning.
This section presents performance in two tasks from that domain:
the training of a multi-layer perceptron (MLP) and a convolutional network. 
More extensive benchmarks are forthcoming, and will be posted on our website.

We chose these
architectures because of their popularity in machine learning and their different 
computational demands. Large matrix-matrix multiplications dominate in the MLP example, 
and two-dimensional image convolutions with small kernels dominate 
computations in the convolutional network.
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
Each implementation multiplied 60 784-element
input vectors by a :math:`$784 \times 500$` weight matrix, compressed by a tanh
function, then multiplied by a :math:`$500 \times 10$` matrix, and finally classified using a
multi-class generalization of logistic regression.  The gradient was calculated
by performing similar calculations, but in reverse.

.. _Figure 3:
.. _Benchmark1:
.. figure:: mlp.pdf
    :scale: 100

    **Figure 3:** Fitting a multi-layer perceptron to simulated data with 
    various implementations of stochastic gradient descent.  These models have
    784 inputs, 500 hidden units, a 10-way classification, and are trained 60
    examples at a time.

`Figure 3`_ compares the number of examples processed per second 
by different implementations.
We compared Theano (revision #ec057beb6c), NumPy 1.4.1, Matlab 7.9.0.529, and
Torch 5 (a machine learning 
library written in C/C++) [torch5]_.  On the GPU we compared Theano with GPUMat 0.25 for Matlab
([gpumat]_).
As shown in `Figure 3`_, on the CPU Theano is 1.8x faster than NumPy,
1.6x faster than Matlab, and 7.5x faster than Torch 5. Torch was written
for flexibility, not speed (Ronan Collobert, p.c.).
Theano's speed increases 5.8x on the GPU from the CPU, a total increase of 11x over NumPy on the CPU and 44x over Torch 5 on the CPU.
GPUmat increases the Matlab speed on the GPU only 1.4x from the CPU, far
less than the 5.8x increase Theano achieves through CUDA specializations.

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
EBLearn was implemented by Yann LeCun's lab at NYU, which has done extensive research in convolutional networks, so EBLearn is a solid baseline.
To put these results
into perspective, we implemented approximately half (no gradient calculation)
of the algorithm using SciPy's ``signal.convolve2d`` function.  This benchmark
uses convolutions of medium sized images
(:math:`$256 \times 256$`) with
small filters (:math:`$7 \times 7$`).
`Figure 4`_ shows the performance of Theano (both CPU and GPU)
against competing implementations.
On the CPU, Theano is 2.2x faster than EBLearn, its best competitor. This is because Theano compiles more specialized convolution routines.
Theano's speed increases 4.9x on the GPU from the CPU, a total of 10.7x over EBLearn on the CPU.
On the CPU, Theano is 5.8x faster than SciPy even though SciPy is doing only half the algorithm because 
SciPy's convolution routine has not been optimized for this application.

We also compared Theano with numexpr and NumPy for evaluating elementwise
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

This section gives an overview the design of Theano.

A Theano expression graph is a bi-partite directed acyclic graph.
It is bi-partite because there are two kinds of nodes: *variable* nodes are the
inputs to and outputs from *apply* nodes.
A *variable* node represents input or an intermediate mathematical result.
It has a *Type* (``.type``) that signals the sort of value the variable might take at
runtime.
An *apply* node represents the application of the *Op* (``.op``) to some input *variables* (``.inputs``) producing some output *variables* (``.outputs``).
Figures 1 and 2 have been simplified for clarity.
Technically there is an
intermediate result for the output of the ``Elemwise{pow,no_inplace}``,
and the variable nodes (box) and apply nodes (ellipse) are distinct from the
Type and Op instances respectively (not shown) that give them meaning.


Variables
~~~~~~~~~~~~~~~~~~~

Theano supports three kinds of variable nodes: *Variables*, *Constants*, and *Shared variables*. 
*Variable* nodes (with a capital V) are the most common kind - a Variable is either found as a
leaf of the graph (if it was created explicitly with a call like ``theano.tensor.vector()``),
or as the output of an *apply* node (if it was defined by the application
of an Op).
In the latter case, the Variable will have a ``.owner`` attribute pointing to the *apply* node.
``a`` and ``b`` in `Listing 1`_ are Variables (without ``.owner``).
``p_1`` in `Listing 2`_ is also a Variable (with ``.owner``).
``theano.function`` takes two arguments: the input list, which is a list of Variables; and the output value or list, which is a Variable or list of Variables.
*Constant* nodes each have a ``.value`` attribute, which is the immutable (read-only) value of this variable.
``10`` in `Listing 1`_ was converted to a Constant node.
*Shared Variable* nodes have ``.get_value()`` and ``.set_value(new_val)`` methods that
behave by default as if they are transfering from and to (respectively) Theano-managed
memory. Sometimes this is done for consistency, and other times (like when a
type conversion takes place, or the transfer requires moving data to or from a
GPU) it is a necessary copy.
This value can also be modified by calling a Theano function that was defined with ``updates``, like ``train`` in `Listing 2`_.

Types
~~~~~~~~~~~~~~~~~~~

The important variable Types in Theano are:

 * ``TensorType`` - 
   denotes a ``numpy.ndarray`` with specific number of dimensions,
   a record of which of these dimensions are broadcastable, and *dtype*. The dtype is the data types,
   e.g. ``int32``, ``float64``, etc.

 * ``SparseType`` -
   denotes one of the ``csr`` or ``csc`` formats in ``scipy.sparse``.

 * ``RandomStateType`` -
   denotes a NumPy ``RandomState`` object. They are rarely used directly
   by Theano user code. They are storage containers for the random
   number generator.

 * ``Generic`` -
   denotes any Python value.
   They are rarely used directly by Theano user code.
   Generic Variables exist mainly for Ops to be able
   to allocate workspace outputs.


Theano types are often stricter
than their NumPy/SciPy equivalents. For example,
there are different versions of ``SparseType`` in Theano, which are specific
to different encodings like ``csr`` or ``csc``. The Theano ``TensorType`` that 
corresponds to a ``numpy.ndarray`` also specifies
the number of dimensions (scalar=0, vector=1, etc.), which of them are
broadcastable, and what *dtype* should be used. This information is used 
when performing graph transformations.

For *Shared Variables* and *Constants*, the type is inferred 
automatically based on the value given during initialization.


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
    Allocation             &    {\tt alloc}, {\tt eye}, {\tt [ones,zeros]\_like},
                                {\tt identity\{\_like\} }
                                \tabularnewline
    Indexing*              &    basic slicing (see {\tt set\_subtensor} and 
                                {\tt inc\_subtensor} for slicing lvalues);
                                limited support for advanced indexing
                                \tabularnewline
    Math. Functions        &    {\tt exp}, {\tt log}, {\tt tan[h]}, {\tt cos[h]}, {\tt sin[h]}, 
                                {\tt real}, {\tt imag}, {\tt sqrt}, {\tt floor}, {\tt ceil}, 
                                {\tt round}, {\tt abs}
                                \tabularnewline
    Tensor Operations      &    {\tt all}, {\tt any}, {\tt mean}, {\tt sum}, {\tt min}, {\tt max}, 
                                {\tt var}, {\tt prod}, {\tt argmin} , {\tt argmax}
                                {\tt reshape}, {\tt flatten},
                                {\tt dimshuffle}
                                \tabularnewline
    Conditional            &    {\tt cond}, {\tt switch}
                                \tabularnewline
    Looping                &    {\tt Scan}
                                \tabularnewline
    Linear Algebra         &     {\tt dot}, {\tt outer}, {\tt tensordot}
                                \tabularnewline
    Calculus*              &     {\tt grad}
                                \tabularnewline
    Signal Processing      &    {\tt conv2d}, {\tt FFT}, {\tt max\_pool\_2d}
                                \tabularnewline
    Random                 &    {\tt RandomStreams}, {\tt MRG\_RandomStreams}
                                \tabularnewline
    Printing               &    {\tt Print} Op
                                \tabularnewline
    Sparse                 &    limited operator support, {\tt dot}
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
exceptions being advanced indexing, scan, summation over certain combinations of
axes, and reductions max, min and prod.
Our goal is extend coverage to all ops.

Theano does *not* currently have ops for sparse or dense matrix inversion, nor linear
algebra decompositions.  Ops for complex number dtypes are also not as widely
implemented or well-tested as those for integer and float dtypes. Object dtypes
are not implemented in Theano.


Transformations
~~~~~~~~~~~~~~~~

Theano uses graph transformations to implement a range of
tasks from merging redundant calculations to transferring computations to the
GPU.
The optimization of expression graphs is carried out several stages.

The first stage removes duplicate expressions, and when several constants are
actually equal, they are replaced with a single node.
Theano treats two apply nodes with the same inputs and the same Op as being
duplicates and only keeps one.
The automatic gradient mechanism often introduces this sort of redundancy,
so this phase is quite important.  The ``'FAST_COMPILE'`` mode includes only this
stage.

The second stage transforms the graph into an equivalent, canonical form,
so that subsequent patterns do not have to recognize as
wide a variety of equivalent expressions.
For example, expression subgraphs involving just multiplication and division are
put into a standard fraction form (e.g. ``a / (b * c / d) -> (a * d) / (b * c)``),
and terms in both numerator and denominator are cancelled.

The third stage replaces expressions to improve numerical stability. The
logistic sigmoid substitution described at the end of Section `Case Study: Logistic Regression`_ is an example.
After numerically unstable subgraphs have been replaced with more stable ones,
Theano pre-calculates expressions involving only constants.

The fourth stage specializes generic expressions and subgraphs.
Expressions like ``pow(x,2)`` become ``sqr(x)``.
Theano also performs more elaborate specializations:
expressions involving scalar-multiplied matrix additions and multiplications may
become
BLAS General matrix multiply (GEMM) nodes, sums of incremented tensors become incremented
sums, and ``reshape``, ``dimshuffle``, and ``subtensor`` Ops
are replaced by constant-time versions that work by aliasing memory.

After this stage of specialization, Elementwise subgraphs are fused into
Compound ones that permit loop fusion (such as the ``Elemwise{Composite{...}}``
Op in `Figure 2`_).  If Theano is using a GPU, Ops with corresponding GPU
implementations are substituted in.

Lastly, Theano replaces Ops with equivalents that reuse the memory of
their inputs and also invalidate those inputs by side-effect of running.
Many Ops (e.g. GEMM and all elementwise Ops) have such equivalents.
Reusing memory this way can improve speed by reducing cache misses
and allowing more computations to fit on GPUs where memory is at a premium.

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


Limitations and Future Work
---------------------------

Theano does not make significant efforts to optimize the compilation process itself.
Theano can take up to a few seconds to construct a Theano function
(especially when it must compile freshly-generated C code), even when a naïve
implementation of the function's expression would require only a fraction of a
second. So Theano takes time when creating Theano functions, which is not the case
for libraries such as NumPy
and SciPy whose functions have already been compiled.
Theano is therefore suited to applications where a function will be called enough times
that the time spent on the initial compilation is negligible.
Theano has been tested primarily with graphs from 10-1000 nodes, which is
sufficient for many algorithms.
The time spent on applying graph transformations tends to grow super-linearly with the size
of the expression graph. Beyond a few thousand nodes, Theano's optimization
algorithm can be impractically slow, unless you disable some of the more
expensive optimizations, or compile pieces of the graph separately.

A Theano function call also requires more overhead (on the order of microseconds) than a native Python function
call. For this reason, Theano is suited to applications where functions correspond to
expressions that are not too small (see `Figure 5`_).

The set of Types and Ops that Theano provides continues to grow, but it does not
cover all the functionality of NumPy and covers only a few features of SciPy.
Wrapping functions from these and other libraries is often straightforward,
but implementing related graph transformations and implementing Ops for
gradients can be more difficult.
We expect to improve support for advanced indexing and linear algebra in the
coming months.
Documentation online describes how to add new Ops, Types, and transformations.

Theano's graph transformations give good results for expressions related to
machine learning with neural networks, but they are not as well tested outside
that domain.  Theano is not a powerful computer algebra system, and 
it is an important area of future work to improve its ability to recognize
numerical instability in complicated elementwise expression graphs.

Debugging Theano functions can require non-standard techniques and
Theano-specific tools.  The reason is two-fold: 1) definition
of Theano expressions is separate from their execution, and 2) optimizations
can introduce many changes to the computation graph.


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

.. [NumPy] D. Ascher et al., Numerical Python, tech. report UCRL-MA-128569, 
           Lawrence Livermore National Laboratory, 2001, 
           http://numpy.scipy.org

.. [numexpr] D. Cooke et al., 
             numexpr, 
             http://code.google.com/p/numexpr/

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


