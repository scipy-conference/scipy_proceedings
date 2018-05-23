:author: Luiz Irber
:email: lcirberjr@ucdavis.edu
:institution: University of California, Davis


--------------------------------------------
Oxidizing Python: writing extensions in Rust
--------------------------------------------

.. class:: abstract

   Python has a mature ecosystem for extensions using C/C++,
   with Cython being part of the standard toolset for scientific programming.
   Even so,  C/C++ still have many drawbacks,
   ranging from smaller annoyances (like library packaging, versioning and build systems)
   to serious one like buffer overflows and undefined behavior leading to security issues.

   Rust is a system programming language trying to avoid many of the C/C++ pitfalls,
   on top of providing a good development workflow and memory safety guarantees.

   This talk is about how we can write extensions in Rust and use them in Python,
   using as example a Python codebase with a core written in C++ (for performance reasons)
   and how I replaced the C++ parts with Rust.
   This also let the core be used in other languages and even unusual platforms,
   like a browser (using WebAssembly).

   I'll discuss what works, what doesn't (yet) and why this might be a good idea.
   I will also cover how writing extensions changed over the years,
   and what lessons we can incorporate in this new approach.

.. class:: keywords

   rust, extension, FFI, minhash, bioinformatics

Introduction
------------

The process of converting a codebase to Rust is referred as "Oxidation" [#]_ in
the Rust community, following the codename Mozilla chose for the process of
integrating Rust components into the Firefox codebase.
Many of these components were tested and derived in Servo, an experimental
browser engine written in Rust, and are being integrated into Gecko,
the current browser engine (mostly written in C++).


.. [#] The creator of the language is known to keep making up different
       explanations for the name of the language [RustName]_,
       but in this case "oxidation" refers to the chemical process that creates
       rust, and rust is the closest thing to metal (metal being the hardware).
       There are many terrible puns in the Rust community.

Why Rust?
---------

There are other programming languages more focused on scientific software
that could be used instead, like Julia.

Many programming languages start from a specific niche (like R and statistics,
or Maple and mathematics) and grow into larger languages over time.
While Rust goal is not to be a scientific language,
its focus on being a general purpose language allows a phenomenon similar
to what happened with Python, where people from many areas pushed the
language in different directions (system scripting, web development,
numerical programming...) allowing developers to combine all these things
in their systems.

Brings many best practices to the default experience:
integrated package management (with Cargo)
supporting documentation, testing and benchmarking.

It's understandable that older languages like C/C++ need
more effort to support some of these features (like modules and an unified
build system), since they are designed by standards and need to keep backward
compatibility with codebases that already exist.
Nonetheless, the lack of features increase the effort needed to have good
software engineering practices, since you need to choose a solution that might
not be compatible with other similar but slightly different options,
leading to fragmentation and increasing the impedance to use these features.

Is compatible with C/C++ through an explicit layer (the FFI),
avoiding the semantic issues that C++ backwards compatibility with C introduces.

Converting from a C++ extension to Rust
---------------------------------------

The current implementation of the core data structures in sourmash is in a
C++ extension wrapped with Cython.

The goals of converting the code are:

support additional languages and platforms (browser, JS, webassembly)

reducing the number of wheel packages necessary (one for each OS/platform)

use the Rust memory management concepts (lifetimes, borrowing) to lead
to increased parallelism in the code

Many of these goals are attainable with our current C++ codebase, and
"rewrite in a new language" is rarely the best way to solve a problem.
But the reduced burden in maintenance due to better tooling,
on top of features that would require careful planning to execute
(increasing the parallelism without data races) while maintaining compatibility
with the current codebase are promising enough to justify this experiment.


Cython provides a nice gradual path to migrate code from Python to C++,
since it is a superset of the Python syntax. It also provides low overhead
for many C++ features, especially the STL containers, with makes it easier
to map C++ features to the Python equivalent.
For research software this also lead to faster exploration of solutions before
having to commit to lower level code, but without a good process it might also
lead to code never crossing into the C++ layer and being stuck in the Cython
layer. This doesn't make any difference for a Python user, but it becomes
harder from users from other languages to benefit from this code (since your
language would need some kind of support to calling Python code, which is not
as readily available as calling C code).

Depending on the requirements, a downside is that Cython is tied to the CPython API,
so generating the extension requires a development environment set up with
the appropriate headers and compiler. This also makes the extension specific
to a Python version: while this is not a problem for source distributions,
generating wheels lead to one wheel for each OS and Python version supported.



Unified libraries for Bioinformatics
------------------------------------

Bioinformatics is an umbrella term for many different methods, depending on
what analysis you want to do with your data (or model).
In this sense, it's distinct from other scientific areas where it is possible
to rely on a common set of libraries (numpy and linear algebra), since a
library supporting many disjoint methods tend to grow too big and hard to
maintain.

The environment also tends to be very diverse, with many different languages
needing to interact (?!?)


Exploratory vs production-ready software
----------------------------------------

Bioinformatics analysis is usually written as a pipeline: a workflow
describing how to connect the input and output of many different tools to
generate results. The basic unit is a tool, usually with a command-line interface,
and so pipelines tend to rely on standard operating system abstractions like
files and pipes to make the tools communicate with each other. But since tools
might have input requirements distinct from what the previous tool provides,
many times it is necessary to do format conversion or adapting to make the
pipeline work.

Using tools as blackboxes, controllable through specific parameters at the
command-line level, make exploratory analysis and algorithm reuse harder:
if something needs to be investigated the user needs to resort to perturbations
of the parameters or the input data, without access to the more feature-rich and
meaningful abstraction happening inside the tool.


References
----------
.. [RustName] “Internet Archaeology: The Definitive, End-All Source for Why Rust Is Named ‘Rust’”.
              Accessed May 10, 2018.
              https://www.reddit.com/r/rust/comments/27jvdt/internet_archaeology_the_definitive_endall_source/.
