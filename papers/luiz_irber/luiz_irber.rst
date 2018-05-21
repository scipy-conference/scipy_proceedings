:author: Luiz Irber
:email: lcirberjr@ucdavis.edu
:institution: University of California, Davis

:bibliography: rust


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
