:author: Serge Guelton
:email: serge.guelton@telecom-bretagne.eu
:institution: ENS, Paris, France

:author: Pierrick Brunet
:email: pierrick.brunet@telecom-bretagne.eu
:institution: Télécom Bretagne, Plouzané, France

:author: Alan Raynaud
:email: alan.raynaud@telecom-bretagne.eu
:institution: Télécom Bretagne, Plouzané, France

:author: Adrien Merlini
:email: adrien.merlini.eu
:institution: Télécom Bretagne, Plouzané, France

:author: Mehdi Amini
:email: mehdi.amini@silkan.com
:institution: Silkan, Meudon Val-Fleury, France


-------------------------------------------------------------------
Pythran: Enabling Static Optimization of Scientific Python Programs
-------------------------------------------------------------------

.. class:: abstract


    Pythran is a young open source static compiler that turns modules written
    in a Python subset into native ones. Based on the fact that scientific
    modules do not rely much on the dynamic features of the language, it trades
    them against powerful, eventually inter procedural, optimizations, such as
    automatic detection of pure functions, temporary allocation removal,
    constant folding, numpy ufunc fusion and parallelization, explicit
    parallelism through OpenMP annotations, false variable polymorphism pruning
    and AVX/SSE vector instruction generation.

    In addition to these compilation steps, Pythran provides a C++ runtime that
    leverages on the C++ STL for generic containers, and the Numeric Template
    Toolbox (nt2) for Numpy support. It takes advantage of modern C++11
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

An approach used by Cython[cython]_ is to suppress the interpretation overhead
by translating Python Programs to C programs calling the Python C
API[pythoncapi]_. More recently, Nuitka[nuitka]_ has taken the same approach
using C++ has a back-end. Going a step further Cython also uses an hybrid
C/Python language that can efficiently be translated to C code, relying on the
Python C API for some parts and on plain C for others.  Shedskin[shedskin]_
translates implicitly strongly typed Python program into C++, without any call
to the Python C API.

The alternate approach consist in writing a Just In Time(JIT) compiler embeded
into the interpreter, to dynamically turn the computation intensive parts into
native code. The `numexpr` module [numexpr]_ does so for Numpy expressions by
JIT-compiling them from a string representation to native code. Numba[numba]_
extends it to Numpy-centric applications and PyPy[pypy]_ applies the approach
to the whole language.

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
`range-list`. This article presents Pythran, an optimizing compiler for a
subset of the Python language that turns implicitly statically typed modules
into parametric C++ code. It supports many high-level constructs of the 2.7
version of the Python language such as list comprehension, set comprehension,
dict comprehension, generator expression, lambda functions, nested functions or
polymorphic functions. It does *not* support global variables, user classes or
any dynamic feature such as introspection, polymorphic variables.

Unlike existing alternatives, Pythran does not solely performs static typing of
Python programs. It also performs various compiler optimizations such as
detection of pure functions, temporary allocation removal or constant folding.
These transformations are backed up by code analysis such as aliasing,
inter-procedural memory effect computations or use-def chains.

The article is structured as follows: Section 1 introduces the Pythran compiler
compilation flow and internal representation.  Section 2  presents several code
analysis while Section 3 focuses on code optimizations. Section 4 presents
back-end optimizations for the `numpy` expressions. Section 5 illustrates the
performance of generated code on a few synthetic benchmarks and concludes.


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
except that it is checked at compile time, as illustrated by the following
implementation of a generic dot product in Python and C++:

.. code-block:: python

    def dot(l0, l1):
        return sum(x*y for x,y in zip(l0,l1))

.. code-block:: c++

    template<class T0, class T1>
        auto dot(T0&& t0, T1&& t1)
        -> decltype(/* skipped */)
        {
            return pythonic::sum(
                pythonic::map(
                    operator_::multiply(),
                        pythonic::zip(
                            std::forward<T0>(t0),
                            std::forward<T1>(t1))
                )
            );
        }

Provided `sum`, `map` and `zip` are implemented in a third party library. The
only assumption these two version make are that `l0` and `l1` are iterable,
their content can be multiplied and the result of the multiplication is
accumulatable.

Second step only consists in the instantiation of the top-level function of the
module, using user-provided signature. Template instantiation then triggers the
instantiation of the correctly typed of all function written in the program.
Note that the user only needs to provide the type of the outermost functions.
The type of all internal functions is then inferred from the call site.

Last step involves a template library, called `pythonic` that contains a
polymorphic implementation of many functions from the Python standard library
in the form of C++ template functions. Several optimizations, most notably
expression template, are delegated to this library. Pythran relies on the
C++11[cxx11]_ language, as it makes heavy use of recent features such as move
semantics, type inference through `decltype(...)` and variadic templates. As a
consequence it requires a compatible C++ compiler for the native code
generation and on Boost.Python[boost_python]_ for the Python-to-C++ glue.
Generated code is compatible with g++ 4.7.2 and clang++ 3.2.

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

A code analyse is a function that takes a part of the IR (or the whole module's
IR) as input and returns aggregated high-level information. For instance, a
simple Pythran analyse calld `Identifiers` gathers the set of all identifiers
used throughout the program. It is used to create new identifiers that do not
conflict with existing ones.

One of the most important analyse in Pythran is *alias analysis*, sometimes
referred as point-to analysis. For each identifiers, it computes an
approximation of the set of locations this identifier may point to. For
instance, let us consider the polymorphic function `foo` defined as follows:

.. code-block:: python

    def foo(a,b):
        c = a or b
        return c*2

The identifier `c` involved in the multiplication may refer to

- a fresh location if `a` and `b` are scalars

- the same location as `a` if `a` evaluates to `True`

- the same location as `b` otherwise.

As we do not specialise the analyse for different type and the truth value of
`a` is unknown at compilation time, the alias analysis yields the approximated
result that `c` may points to a fresh location, `a` or `b`.

Without this kind of information, even a simple instruction like `sum(a)` would
yield very few informations as there is no guarantee that the `sum` identifiers
points to the `sum` built-in.

When turning Python AST to Pythran IR, nested functions are turned into global
functions taking their closure as parameter. This closure is computed using the
information provided by the `Globals` analyse that statically computes the
state of the dictionary of globals, and `ImportedIds` that computes the set of
identifiers used by an instruction but not declared in this instruction. For
instance in the following snippet:

.. code-block:: python

    def outer(outer_argument):
        def inner(inner_argument):
            return cos(outer_argument) + inner_argument
        return inner

The `Globals` analyse called on the `inner` function definition marks `cos` as
a global variable, and `ImportedIds` marks `outer_argument` and `cos` as
imported identifiers.

A rather high-level analyse is the `PureFunctions` analyse, that computes the
set of functions declared in the module that are pure, i.e. whose return value
only depends from the value of their argument. This analyse depends on two
other analyse, namely `GlobalEffects` that computes for each function whether
this function modifies the global state (including I/O, random generators etc)
and `ArgumentEffects` that computes for each argument of each function whether
this argument may be updated in the function body. These three analyse works
inter-procedurally, as illustrated by the following example:

.. code-block:: python

    def fibo(n):
        return n if n < 2 else fibo(n-1) + fibo(n-2)

    def bar(l):
        return map(fibo, l)

    def foo(l):
        return map(fibo, random.sample(l, 3))

The `fibo` function is pure as it has no global effects or argument effects and
only calls itself. As a consequence the `bar` function is also pure has the
`map` intrinsic is pure when its first argument is pure. However the `foo`
function is not pure as it calls the `sample` function from the `random`
module, which has a global effect (on the underlying random number generator).

Several analysis depends on the `PureFunctions` analyse. `ParallelMaps` uses
aliasing information to check if an identifier points to the `map` intrinsic,
and checks if the first argument is a pure function using `PureFunctions`. In
that case the `map` is added to the set of parallel maps, because it can be
executed in any order. This is the case for the first `map` in the following snippet,
but not for the second.

.. code-block:: python

    def pure(a):
        return a**2

    def guilty(a):
        b = pure(a)
        print b
        return b

    l = list(...)
    map(pure, l)
    map(guilty, l)

`ConstantExpressions` uses function purity to decide
whether a given expression is constant, i.e. its value only depends from
literals. For instance the expression `fibo(12)` is a constant expression
because `fibo` is pure and its argument is a literal.

`UsedDefChains` is a typical analyse from the static compilation world. For
each variable defined in a function, it computes the chain of *use* and *def*.
The result can be used to perform various code transformation, for instance to
remove dead code, as a *def* not followed by a *use* is useless. It is used in
Pythran to avoid false polymorphism. An intuitive way to represent used-def
chains is illustrated on next code snippet:

.. code-block:: python

    a = 1
    if cond:
        a = a + 2
    else:
        a = 3
    print a
    a = 4

In this example, there are two possible chains starting from the first
assignment. Using `U` to denote *use* and `D` to denote *def*, one gets::

    D U D U D

and::

    D D U D

The fact that all chains finish by a *def* indicates that the last assignment
can be removed (but not necessarily its right hand part that could have a
side-effect).

All the above analyse are used by the Pythran developer to build code
transformation to optimize the execution time of the generated code.

Code Optimizations
------------------

One of the benefit of translating Python code to C++ code is that it removes
most of the dynamic lookups. It also unveils all the optimizations available at
C++ level. For instance, a function call is quite costly in Python, which
advocates in favor of using inlining. This transformation comes at no cost when
using C++ as the back-end language, as the C++ compiler does it.

However, there are some informations available at the Python level that cannot
be recovered at the C++ level. For instance, Pythran uses functor with an
internal state and a goto dispatch table to represent generators. Although
effective, this approach is not very efficient, especially for trivial cases.
Such trivial cases appear when a generator expression is converted, in the
front-end, to a looping generator. To avoid this extra cost, Pythran turns
generator expressions into call to `imap` and `ifilter` from the `itertools`
module whenever possible, removing the unnecessary goto dispatching table. This
kind of transformation cannot be made by the C++ compiler. For instance, the
one-liner `len(set(vec[i]+i for i in cols))` extracted from the `nqueens`
benchmarks from the Unladen Swallow project is rewritten as
`len(set(itertools.imap(lambda i: vec[i]+i,cols)))`. This new form is less
efficient in pure Python (it implies one extra function call per iteration),
but can be compiled into C++ more efficiently than a general generator.

A similar optimization consists in turning `map`, `zip` or `filter` into their
equivalent version from the `itertool` module. The benefit is double: first it
removes a temporary allocation, second it gives an opportunity to the compiler
to replaces list accesses by scalar accesses. This transformation is not always
valid, nor profitable. It is not valid if the content of the output list is
written later on, and not profitable if the content of the output list is read
several times, as each read implies the (re) computation, as illustrated in the
following code:

.. code-block:: python

    def valid_conversion(n):
        # this map can be converted to imap
        l = map(math.cos, range(n))
        return sum(l) # sum iterates once on its input

    def invalid_conversion(n):
        # this map cannot be converted to imap
        l = map(math.cos, range(n))
        return sum(l) + max(l) # sum iterates once

The information concerning constant expressions is used to perform a classical
transformation called constant unfolding, which consists in the compile-time
evaluation of constant expressions. The validity is guaranteed by the
`ConstantExpressions` analyse, and the evaluation relies on Python ability to
compile an AST into byte code and run it, benefiting from the fact that Pythran
IR is a subset of Python AST. A typical illustration is the initialization of a
cache at compile-time:

.. code-block:: python

    def esieve(n):
        candidates = range(2, n+1)
        return sorted(
            set(candidates)
            -
            set(p*i
                for p in candidates
                for i in range(p, n+1))
            )

    cache = esieve(100) 

Pythran automatically detects that `eseive` is a pure function and evaluates
the `cache` variable value at compile time.


Sometimes, coders use the same variable in a function to represent value with
different types, which leads to false polymorphism, as in:

.. code-block:: python

    a = cos(1)
    a = str(a)

These instructions cannot be translated to C++ directly because `a` would have
both `double` and `str` type. However, using `UsedDefChains` it is possible to
assert the validity of the renaming of the instructions into:

.. code-block:: python

    a = cos(1)
    a_ = str(a)

that does not have the same typing issue.

In addition to this python-level optimizations, the Pythran backend library,
`pythonic`, uses several well known optimisations, especially for `numpy`
expressions.

Library Level Optimizations
---------------------------

Using the proper library, the C++ language provides an abstraction level close
to what Python proposes. Pythran provides a wrapper library, `pythonic`, that
leverage on the Standard Template Library(STL), the GNU Multiple Precision
Arithmetic Library(GMP) and the Numerical Template Toolbox(NT2)[nt2]_ to emulate
Python standard library. The STL is used to provide a typed version of the
standard containers (`list`, `set`, `dict` and `str`), as well as
reference-based memory management through `shared_ptr`. Generic algorithms such
as `accumulate` are used when possible. GMP is the natural pick to represent
Python's `long` in C++. NT2 provides a generic vector library called
`boost.simd` that makes it possible to access the vector instruction unit of
modern processors in a generic way. It is used to efficiently compile `numpy`
expressions.

`numpy` expressions are the perfect candidate for library level optimization.
Pythran implements three optimizations on such expressions:

1. Expression templates[expression_templates]_ are used to avoid multiple iterations and the
   creation of intermediate arrays. Because they aggregates all `ufunc` into a single
   expression at compile time, they also increase the computation intensity of the
   loop body, which increases the impact of the two following optimizations.

2. Loop vectorization. All modern processors have a vector instruction unit
   capable of applying the same operation on a vector of data instead of a
   single data. For instance Intel's i7 can run 8 single-precision additions in
   a single instruction. One can directly use the vector instruction set
   assembly to use these vector units, or use C/C++ intrinsics. Pythran relies
   on `boost.simd` from NT2 that offers a generic vector implementation of all
   standard math functions to generate a vectorized version of `numpy`
   expressions. Again, the aggregation of operators performed by the expression
   templates proves to be beneficial, as it reduces the number of (costly) load
   from the main memory to the vector unit.

3. Loop parallelization through OpenMP[openmp]_. Numpy expression computation do
   not carry any loop-dependency. They are perfect candidates for loop
   parallelization, especially after the aggregation from expression templates,
   as OpenMP generally performs better on loops with a higher computation
   intensity that masks the scheduling overhead.

To illustrate the benefits of these three optimizations, let us consider the
simple numpy expression:

.. code-block:: python

    d = numpy.sqrt(b*b+c*c)

When benchmarked with the `timeit` module on an hyper threaded quadcore i7, the
standard versions yields:

.. code-block:: python

    >>> %timeit np.sqrt(b*b+c*c)
    1000 loops, best of 3: 1.23 ms per loop


then with Pythran and expression templates:

.. code-block:: python

    >>> %timeit my.pythranized(b,c)
    1000 loops, best of 3: 621 us per loop

Expression templates replace 4 temporary array creations and 4 loops by a
single allocation and a single loop.

Going a step further and vectorizing the generated loop yields an extra performance boost:

.. code-block:: python

    >>> %timeit my.pythranized(b,c)
    1000 loops, best of 3: 418 us per loop

Although the AVX instruction sets makes it possible to store 4 double precision
float, one does not get a 4x speed up because of the unaligned memory transfer
to and from vector registers.

Finally, with expression templates, vectorization and OpenMP:

.. code-block:: python

    >>> %timeit my.pythranized(b,c)
    1000 loops, best of 3: 105 us per loop

The 4 hyper threaded cores give an extra performance boost. Unfortunately, the
load is not sufficient to get more than an average 4x speed up compared to the
vectorized version. In the end, Pythran generates a native module that performs
roughly 11 times faster than the original version.

As a reference, the `numexpr` module that performs JIT optimization of the
expression yields the following timings:

.. code-block:: python

    >>> %timeit numexpr.evaluate("sqrt(b*b+c*c)")
    1000 loops, best of 3: 395 us per loop
 
Next section performs an in-depth comparison of Pythran with three Python
optimizers: PyPy, Shedskin and numexpr.

Benchmarks
----------

All benchmarks presented in this section are run on an hyper-threaded i7
quadcore, using the code available in the Pythran sources available at
https://github.com/serge-sans-paille/pythran in the `pythran/test/cases`
directory. The Pythran version used is `deqzffzr`, Shedskin 0.9.2, PyPy 2.0
compiled with the `-jit` flag, CPython 2.7.3 and numexpr 2.0.1.

Pystone is a Python translation of whetstone, a famous floating point number
benchmarks that dates back to Algol60 and the 70's. Although non representative
of real applications, it illustrates the general performance of floating point
number manipulations. Figure :ref:`pystone-table` illustrates the benchmark
result for CPython, PyPy, Shedskin and Pythran.

.. table:: Benchmarking result on the Pystone program. :label:`pystone-table`

    +-------------+---------------+------------+
    |  bla        |   bla         |     bla    |
    +-------------+---------------+------------+

It shows that...

Nqueen is a benchmark extracted from the now dead project Unladen Swallow. It
is particularly interesting as it makes an intensive use of non-trivial
generator and integer sets. Figure :ref:`nqueen-table` illustrates the benchmark
result for CPython, PyPy, Shedskin and Pythran. 

.. table:: Benchmarking result on the NQueen program. :label:`nqueen-table`

    +-------------+---------------+------------+
    |  bla        |   bla         |     bla    |
    +-------------+---------------+------------+

It shows that...

Hyantes is a geomatic application that exhibits typical usage of numpy
array using loops instead of generalized expressions. It is helpful to measure
the performance of direct array indexing. Figure :ref:`hyantes-table`
illustrates the benchmark result for CPython, PyPy and Pythran[*]_.

.. [*] Shedsking does not support numpy

.. table:: Benchmarking result on the hyantes program. :label:`hyantes-table`

    +-------------+---------------+------------+
    |  bla        |   bla         |     bla    |
    +-------------+---------------+------------+

It shows that...

Finally, `arc_distance` is a typical usage of numpy expression that is
typically more efficient with CPython than its loop alternative as all the
looping is done directly in C. Figure :ref:`hyantes-table`
illustrates the benchmark result for CPython, PyPy, Numexpr and Pythran.

.. table:: Benchmarking result on the hyantes program. :label:`hyantes-table`

    +-------------+---------------+------------+
    |  bla        |   bla         |     bla    |
    +-------------+---------------+------------+

It shows that...


Conclusion
----------

This paper presents the Pytran compiler, a translator and optimizer from Python
to C++. Unlike existing static compilers for Python, this compiler leverages on
several function-level or module-level analysis to provide several generic or
Python-centric code optimizations. Additionally, it uses a C++ library that
makes heavy use of template programming to provide an efficient API similar to
a subset of Python standard library. This library takes advantage of modern
hardware capabilities --- vector instruction unit and multi-cores --- in its
implementation of part of the Numpy package.

The paper gives an overview of the compilation flow, the analysis involved and
the optimization used. It also compares the performance of compiled python
module against CPython and other optimizers: Shedskin, PyPy and numexpr.

To conclude, limiting Python to a statically typed subset does not hinders the
expressively when it comes to scientific or mathematic computations, but makes
it possible to use a wide variety of classical optimizations to have Python
match the performance of statically compiled language. Moreover, one can use
high level informations to generate efficient code that would proved to be
difficult to write to the average programmer.

References
----------

.. [boost_python] D. Abrahams and R. W. Grosse-Kunstleve.
                    *Building Hybrid Systems with Boost.Python*,
                    C/C++ Users Journal, 21(7), July 2003.

.. [cython]  S. Behnel, R. Bradshaw, C. Citro, L. Dalcin, D. S. Seljebotn and K. Smith.
                *Cython: The Best of Both Worlds*,
                Computing in Science Engineering, 13(2):31-39, March 2011.

.. [cxx11] ISO, Geneva, Switzerland.
            *Programming Languages -- C++*,
            ISO/IEC 14882:2011.

.. [expression_templates] T. Veldhuizen.
            *Expression Templates*,
            C++ Report, 7:26-31, 1995.

.. [nt2] M. Gaunard, J. Falcou and J-T. Lapresté.
            *The Numerical Template Toolbox*,
            https://github.com/MetaScale/nt2.

.. [nuitka] K. Hayen.
            *Nuitka - The Python Compiler*,
            Talk at EuroPython2012.

.. [numba] T. Oliphant et al.
            *Numba*,
            http://numba.pydata.org/.

.. [numexpr] D. Cooke, T. Hochberg et al.
            *Numexpr - Fast numerical array expression evaluator for Python and NumPy*,
            http://code.google.com/p/numexpr/.

.. [openmp] *OpenMP Application Program Interface*,
            http://www.openmp.org/mp-documents/OpenMP3.1.pdf,
            July 2011.

.. [pypy] C. F. Bolz, A. Cuni, M. Fijalkowski and A. Rigo.
            *Tracing the meta-level: PyPy's tracing JIT compiler*,
            Proceedings of the 4th workshop on the
            Implementation, Compilation, Optimization of
            Object-Oriented Languages and Programming Systems,
            18-25, 2009.

.. [pythoncapi] G. v. Rossum and F. L. Jr. Drake.
                *Python/C API Reference Manual*,
                September 20012.

.. [shedskin] M. Dufour.
                *Shed skin: An optimizing python-to-c++ compiler*,
                Delft University of Technology, 2006.


