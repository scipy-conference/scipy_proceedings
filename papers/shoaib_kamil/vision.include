Scientists and other knowledge workers whose main professional activity
is not programming nonetheless need to write programs to do their work.
They need performance, but usually have neither the interest nor the
esoteric skills needed to write high-performance parllel or sequential
code, a task that typically requires not only knowledge of the problem
or algorithm but also intimate knowledge of threading models and
hardware capabilities.  In short, these application writers would
prefer to program in a familiar sequential model in a
*productivity-level language (PLL)* such as Python if they could attain
performance comparable to an expert-coded *efficiency-level language (ELL)*
implementation, possibly taking advantage of parallelism.

The usual solution to bridging this gap is to provide compiled native
libraries for certain functions, as the SciPy package does.  However, in
some cases libraries may be inadequate or insufficient.  Various
families of computational patterns share the property that while the
*strategy* for mapping the computation onto a particular hardware family
is common to all problem instances, the specifics of the problem are
not.  For example, consider a stencil computation, in which each point
in an n-dimensional grid is updated with a new value that is some
function of its neighbors' values.  The general strategy for optimizing
sequential or parallel code given  a particular target platform
(multicore, GPU, etc.) is independent of the specific function, but
because that function is unique to each application, capturing the
stencil abstraction in a traditional compiled library is awkward,
especially in the ELLs typically used for performant code
(C, C++, etc.) that don't support higher-order functions gracefully.

Even if the function doesn't change much across applications, work on
autotuning (AF: NEED CITATIONS HERE) has shown that for algorithms with
tunable implementation parameters, the performance gain from fine-tuning
these parameters compared to setting them naively can be up to 10x
(AF: CHECK THIS).  Indeed, the complex internal structure of autotuning
libraries such as the Optimized Sparse Kernel Interface [OSKI]_ is
driven by the fact that often runtime information is necessary to choose
the best execution strategy or tuning-parameter values.

We propose a new approach to bridging the productivity/efficiency gap
that overcomes these problems.  The insight of our approach is that if
we provide high-level abstractions for the type of computational pattern
the application writer is trying to use (stencil, matrix multiplication,
etc.), we can select an implementation strategy and apply optimizations
that take advantage of this information *before* the efficiency-level
code is generated and compiled.  We therefore leverage the dynamic features of
modern languages like Python to 
defer until runtime what most libraries must do at compile time.

Specifically, we provide a framework for creating classes each of which
represents a particular computational pattern, and appropriate methods
for expressing problems (instances) of that pattern in Python.  When
these methods are called, they use a combination of pre-supplied
templated source code snippets and
manipulation of the Python AST  to
generate low-level source code in an ELL such as C, C++ or Cuda.  In
cases where the problem calls for passing in a function, such as the
stencil example above, the application writer codes their function in
Python (subject to some restrictions) and we use its AST to lower the
function to the target ELL and inline it into the generated source
code.  This source code is then compiled by an appropriate conventional
compilers, the resulting object file is linked to the Python
interpreter, and called like a native library.  

Because each class specializes one type of computation only, we
can apply optimizations at the level of problm instance rather than
limit ourselves to optimizations that would work generally.  For
example, returning to the  stencil example above,
a fundamental stencil "primitive" is applying the function to each
neighbor of a stencil point.  
Because we know the semantics of the stencil operation,
optimizations such as  loop unrolling or loop transposition can take
advantage of this knowledge, which would be impossible if we were trying
to perform loop unrolling or transposition without knowing the context.
(AF: need a crisper example of this, ie what optimizations can we do to
optimize neighbor iteration that would not necessarily apply to loops in
general)
If a class is provided that implements our approach for a particular
family of problems and generates output code for a particular target
(say, C++ with OpenMP extensions, or nVIDIA's Cuda), we say that we have
a *specializer* for that type of problem on that specific target, and we
refer to the public methods of that class as specializable methods.

Mechanically, we intercept the first call to a
specializable method, grab the AST of the Python code at call site, and
immediately transform it to a domain-specific AST, or DAST.  That is, we
immediately move the computation into a domain where problem-specific
optimiations and knowledge can be applied, by applying transformations
to the DAST.  Returning once again to the stencil, the DAST might have
nodes such as "iterate over neighbors" or "iterate over all stencil
points".  These abstract node types will be converted to generated ELL
code according to the code generation strategy chosen, but at this level
of representation, one can talk about optimizations that make sense *for
stencils specifically* as opposed to those that make sense *for
iteration  generally*. 

We start with an existing ELL solution of a particular problem type on
particular hardware.  Such solutions are devised by human experts, such 
as researchers working on autotuning or numerical programmers who have
optimized some algorithm.  The specializer writer's task is to factor
this working solution into a set of source code templates that can be
instantiated and optionally a set of rules for transforming the DAST of
this type of problem in order to realize the optimizations present in
the ELL code.

At runtime, then, the input to the specialization process is one or more
templates of ELL 
source code, optionally a set of methods for transforming or optimizing the
AST corresponding to the problem instance, and some Python code to drive
the process of assembling the snippets and/or transforming the DAST.  

Conversion of the DAST into ELL code is handled largely by CodePy, after
which the generated source code is compiled by an appropriate downstream
compiler (gcc, cudac, proprietary compilers, etc) into an object file that
can be called from Python.  Code caching strategies avoid
the cost of code generation and compilation on subsequent calls.

[need diagram showing human expert, strategy consisting of templates and
AST transformation rules (for each of N platforms), app writer, Asp,
generated code]

Features:

- Leverage downstream compilers; indeed, can emit code that's easier for
  them to optimize

- Not a magic compiler/parallelizer: relies on human expert providing a
  specialization strategy packages as templates + AST transformation
  rules

- separates concerns by hiding complexity: in ELL code, even "simple"
  computation like stencil expands 10x when the extra code necessary to
  get performance (loop unrolling, software pipelining, parallel
  annotations, etc) is added to the main application logic

- separates concern by allowing separate innovation.  New specializers
  can be provided into a CPAN-like ecosystem ("the Asp store").  New
  apps can be written in a PLL that hides the hardware.  New optimizing
  compilers that target specific hardware features can be inserted in
  the final phase.  Etc.

- performance portability: same Python source can be used to
  JIT-specialize to radically different platforms, so source code has
  nothing in it that ties its performance to a specific platform

Open research issues:

- when to respecialize

- composition
