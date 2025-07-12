---
title: Advancing High Energy Physics Data Analysis with Julia -- A Case for JuliaHEP
abstract: |
  For the past 25 years, the High-Energy Physics (HEP) community has steadily adopted Python as its primary language for data analysis, supported by compiled-backend libraries like NumPy [@numpy] and [Awkward Array](https://github.com/scikit-hep/awkward), along with other Python tools like [Uproot](https://github.com/scikit-hep/uproot5) for I/O. However, the Julia programming language offers a compelling alternative, addressing the two-language problem with C++-comparable performance and Python-like ease of use. [JuliaHEP](https://github.com/JuliaHEP/) is an informal organization that aims to unify effort in developing Julia projects related to HEP, outlining its advantages, ongoing developments, and integration with existing Python-based tools.
---

## Introduction

The High-Energy Physics (HEP) community has long relied on Python and C++ for data analysis. While Python provides ease of use and a rich scientific ecosystem, it struggles with performance for large-scale analyses. C++, on the other hand, offers speed but comes with increased complexity and slower development cycles. Julia, a modern language designed for scientific computing, promises the best of both worlds: high-level expressiveness with near-native execution speed.

:::{figure} heplanguages.png
:label: fig:heplanguages
HEP Software: it all revolves around a language, or several.
:::

JuliaHEP, with its emerging set of tools and libraries [@juliahep], aims to facilitate HEP data analysis in Julia. However, given the timescale of the HEP experiments and the legacy software they rely on, this introduces a three-language problem — though it is intended as a temporary measure.

Currently, three main approaches are being pursued in the development of these tools. The first involves wrapping mature C++ libraries — for example, Geant4.jl provides a Julia interface to the widely used Geant4 particle transportation toolkit. The second approach entails a complete reimplementation of existing tools in native Julia, as demonstrated by BAT.jl. The third, and the one we follow, focuses on integrating Julia with the Python-based Awkward Array library via its dedicated Julia backend, AwkwardArray.jl.

Data sharing — especially across language boundaries — is one of the key challenges. Our approach addresses this by exposing shared data buffers between Python and Julia, avoiding costly copies and offering a gentle, incremental path for physicists to explore Julia’s capabilities within existing workflows.

## Awkward Array in Julia

Awkward Array is a key library for handling complex and jagged HEP data in the Python ecosystem. It is also integrated into Julia via its AwkwardArray.jl backend.

This backend differs from other Awkward Array backends in that it exposes the underlying data buffers to Julia. By using Vector of Vectors, a Julia-native type that provides similar capabilities, we have compared performance and usability against the traditional Python stack. There is no significant overhead in combining the two.

Additionally, we have explored Julia’s code interoperability with Python and its native compilation strategies to optimize HEP computations.

## Performance Benefits

A key focus of this paper is the performance benefits of Julia for HEP workloads. We benchmark the same operation comparing JuliaHEP tools against Python and C++ implementations. The @fig:combinations highlights the case where Julia offers significant speedup with less boilerplate code.

:::{figure} combinations.png
:label: fig:combinations
This is the combinations performance plot, comparing the ak.combinations function with a Julia kernel on an awkward array. The Julia kernel uses JuliaMath/Combinatirics.jl[@Combinatorics.jl].
:::

 This is without exploiting Julia’s potential for parallel computing and built-in GPU support, which provides an efficient pathway for scaling HEP computations.

## Scaling it up

Beyond performance, the introduction of Julia via AwkwardArray.jl addresses the practical aspects of integrating Julia into existing HEP workflows. Physicists can gradually adopt Julia without abandoning Python-based tools, thanks to interlanguage operability. The PythonCall[@PythonCall.jl] package provides a useful interface between the languages, enabling bi-directional integration between the ecosystems.

The AwkwardArray.jl Julia package can be considered a backend for Awkward Arrays in Python, but it differs from all other backends. In a sense, it’s a transitional project that allows Python users to speed up computations on Awkward Arrays using Julia kernels — or even Julia packages optimized for specific needs, such as combinatorics. The underlying data representing an Awkward Array is shared between the languages, making it highly efficient by avoiding unnecessary data copying.

Case studies from CMS and other experimental collaborations illustrate Julia’s real-world applicability in tasks like event processing, data analysis, and simulation.

Inevitably, this approach introduces a third language for physicists. Is it wise? Will Julia replace the other two languages in HEP? Perhaps — but it’s too early to say. Legacy software still needs to be supported, mature Python tools remain widely used, and new languages like Rust are emerging.

There are two directions JuliaHEP packages are currently taking: one involves a complete rewrite of code in Julia, while the other wraps existing C++ libraries via a Julia interface. The most difficult part of bridging the language barrier is the data. However, looking ahead, this 'awkward' approach offers a gentle introduction of a new concept to current users. The question remains whether this will ultimately prove useful — but it creates a valuable space for experimentation and gradual adoption.

We exploited the Awkward design — specifically the contiguous buffers and their form descriptions — in other optimizations, such as the virtual array implementation. It’s a powerful concept.

The main data formats for HEP are currently ROOT TTrees, with RNTuple expected to become more prominent in the future. Both formats can be read by uproot and unroot — Python and Julia counterparts.

Managing a two-language configuration and shared environment is feasible if approached carefully. Conda for Python, alongside juliaup and Julia’s integrated package manager, can work together. The main challenge lies in manually controlling Julia package version updates, as they may introduce newer versions incompatible with the currently installed Python packages. At the time of writing, a possible coordination mechanism via pixi is under testing.

This setup allows physicists to gradually adopt Julia within existing Python-based workflows — and vice versa — while also tapping into Julia’s strengths, including just-in-time (JIT) compilation, multiple dispatch, and GPU acceleration.

## Acknowledgements

This work was supported by the National Science Foundation under Cooperative Agreement PHY-2323298.