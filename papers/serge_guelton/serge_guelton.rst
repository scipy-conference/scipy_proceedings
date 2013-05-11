:author: Serge Guelton
:email: serge.guelton@telecom-bretagne.eu
:institution: ENS, Paris, France

:author: Pierrick Brunet
:email: pierrick.brunet@telecom-bretagne.eu
:institution: Télécom Bretagne, Plouzané, France

:author: Alan Raynaud
:email: alan.raynaud@telecom-bretagne.eu
:institution: Télécom Bretagne, Plouzané, France

:author: Mehdi Amini
:email: mehdi.amini@silkan.com
:institution: Silkan, Meudon Val-Fleury, France


-------------------------------------------------------------------
Pythran: Enabling Static Optimization of Scientific Python Programs
-------------------------------------------------------------------

.. class:: abstract


    Pythran is a young open source static compiler that turns Python modules
    into native ones. Based on the fact that scientific modules do not rely
    much on the dynamic features of the language, it trades them against
    powerful, eventually inter procedural, optimizations, such as automatic
    detection of pure functions, temporary allocation removal, constant
    folding, numpy ufunc fusion and parallelization, explicit parallelism
    through OpenMP annotations, false variable polymorphism pruning and AVX/SSE
    vector instruction generation.

    In addition to these compilation steps, Pythran provides a C++ runtime that
    leverages on the C++ STL for generic containers, and the Numeric Template
    Toolbox (nt2) for numpy support. It takes advantage of modern C++11
    features such as variadic templates, type inference, move semantics and
    perfect forwarding, as well as classical ones such as expression templates.

    The input code remains compatible with the Python interpreter, and output
    code is generally as efficient as the annotated Cython equivalent, if not
    more, without the backward compatibility loss. Numpy expressions runs as
    fast as if compiled with numexpr, without change on the original code.

.. class:: keywords

   static compilation, numpy, c++

Introduction
------------

The Python language is growing in popularity as a language for scientific
computing, mainly thanks to a concise syntax, a high level standard library and
several scientific packages.

However, the overhead of running a scientific or mathematic application written
in Python compared to the same algorithm written in a statically compiled
language such as C is high, due to numerous dynamic lookup, interpretation cost
and high level programming. Additionally, the Python compiler performs no
optimisation on the bytecode, while scientific applications are first-class
candidates for many of them.

Several tools  have been proposed by an active community to fill this
performance gap, either through static compilation or Just In Time(JIT)
compilation.

An approach pioneered by Psyco[ref]_ is to suppress the interpretation
overhead by translating Python Programs to C programs calling the Python C
API[ref]_. More recently, Nuitka[ref]_ has taken the same approach using
C++ has a backend. Going a step further Cython[ref]_ uses an hybrid
C/Python language that can efficiently be translated to C code, relying on the
Python C API for some parts and on plain C for others.  Shedskin[ref]_
translates implicitly strongly typed Python program into C++, without any call
to the Python C API.

The alternate approach consist in writing a Just In Time(JIT) compiler embeded
into the interpreter, to dynamically turn the computation intensive parts into
native code. The `numexpr` module [ref]_ does so for Numpy expressions
by JIT-compiling them from a string representation to native code.  PyPy[ref]_
applies this approach to the whole language.

To the notable exception of PyPy, these compilers do not apply any of the
static optimizations techniques that have been known for decades and
successfully applied to statically compiled language such as C or C++.
Translators to statically compiled languages do take advantage of them
indirectly, but the quality of generated code may prevent advanced
optimisations, such as vectorization, while they are available at higher level,
i.e. at the Python level.

Taking into account the specificities of the Python language can unlock many
new transformations. For instance, PyPy automates the conversion of the `range`
builtin into `xrange` through the use of a dedicated structure called
`range-list`. This article presents Pythran, an optimizing compiler for a large
subset of the Python language that turns implicitly statically typed modules
into parametric C++ code. Unlike existing alternatives, it does perform various
compiler optimizations such as detection of pure functions, temporary
allocation removal or constant folding. These transformations are backed up by
code analysis such as alias aliasing, inter-procedural memory effect
computations or use-def chains.

The article is structured as follows: Section 1 introduces the Pythran
compiler compilation flow and internal representation.  Section 2  presents
several code analysis while Section 3 focuses on code optimizations. Section
4 illustrates the performance of generated code on a few synthetic benchmarks
and concludes.

Pythran Compiler Infrastructure
-------------------------------

The Pythran compiler is built as a traditional static compiler: a front-end
turns Python code into an Internal Representation(IR), a middle-end performs
various code optimizations on this IR and a back-end turns the IR into native
code. The front-end performs two steps:

1. turn Python code into Python Abstract Syntax Tree(AST) thanks to the `ast`
   module from the standard library;

2. turn the Python AST into a type-agnostic Pythran IR, which remains a subset
   of the Python AST.

Pythran IR is similar to Python AST, as defined in the `ast` module, except
that several nodes are forbidden (most notably Pythran does not support
used-defined classes or the `exec` instruction), and some nodes are converted
to others to form a simpler AST easier to deal with for further analyse and
optimizations. The transformations applied by Pythran on Python AST are the
following:

- list/set/dict comprehension are expanded into loops wrapped into a function call;

- tuple unpacking is expanded into several variable assignments;

- lambda functions are turned into named nested functions;

- the closure of nested functions is statically computed to turn the nested
  function into a global function taking the closure as parameter;

- implicit `return None` are made explicit;

- all imports are fully expanded to make function access path explicits

- method calls are turned into function calls;

- implicit `__builtin__` function calls are made explicit;

- `try ... finally` are turned into nested `try ... except` blocks;

- identifier whose name may clash with C++ keywords are renamed. 



The back-end works in three steps:

1. turn the Pythran IR into parametric C++ code;

2. instanciate the C++ code for the desired types;

3. compile the generated C++ code into native code.

First step requires to map polymorphic variables and polymorphic functions from
the Python world to C++. Pythran only supports polymorphic variables for
functions, i.e. a variable can hold several function pointers during its life
time, but it cannot hold an integer then a string. As shown later, it is
possible to detect several false variable polymorphism cases using use-def
chains. Function polymorphism is achieved through template parameters: a
template function can be applied to several types as long as an implicit
structural typing is respected, which is very similar to Python's duck typing,
except that it is checked at compile time.

Second step only consists in the instantiation of the top-level function of the
module, using user-provided signature. Template instantiation then triggers the
instantiation of the correctly typed of all function written in the program.
Note that the user only needs to provide the type of the outermost functions.
The type of all internal functions is then inferred from the call site.

Last step involves a template library, called `pythonic` that contains a
polymorphic implementation of many functions from the Python standard library
in the form of C++ template functions. Several optimizations, most notably
expression template, are delegated to this library. Pythran relies on a
C++11-aware compiler for the native code generation and on `boost::python` for
the Python-to-C++ glue. Generated code is compatible with g++ 4.7.2 and clang++
3.2.

It is important to note that all Pythran analysis are type-agnostic, i.e. they
do not assume any type for the variables manipulated by the program. Type
specialization is only done in the back-end, right before native code
generation. Said otherwise, the Pythran compiler manipulates polymorphic
functions and polymorphic variables.

Figure :ref:`compilation-flow` summarizes the compilation flow and the involved
tools.

.. figure:: compilation-flow.pdf

   Pythran compilation flow.

Code Analysis
-------------

Code Optimizations
------------------

Benchmarks
----------

Conclusion
----------

References
----------
.. [ref] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


