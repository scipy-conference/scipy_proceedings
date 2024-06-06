---
# Ensure that this title is the same as the one in `myst.yml`
title: "Scikit-build-core: A modern build-backend for CPython C/C++/Fortran/Cython extensions"
abstract: |
  Discover how scikit-build-core revolutionizes Python extension building with its seamless integration of CMake and Python packaging standards. Learn about its enhanced features for cross-compilation, multi-platform support, and simplified configuration, which enable writing binary extensions with pybind11, Nanobind, Fortran, Cython, C++, and more. Dive into the transition from the classic scikit-build to the robust scikit-build-core and explore its potential to streamline package distribution across various environments.
---

## Introduction

Python packaging has evolved significantly in the last few years. Standards
have been written to allow the development of new build backends not
constrained by setuptool's complex legacy. Build time dependencies, once
nearly impossible to rely on, are now the standard, and required for any build
system, including the original setuptools. And this new system is controlled by
one central location, the `pyproject.toml` file.

We present scikit-build-core, a new build system based on these standards that
brings together Python packaging and the CMake build system, the popular
general purpose build system for languages like C, C++, Fortran, Cython, and
CUDA. Together, this provides a simple entry point to building extensions that
still scales to major projects. This allows everyone to take advantage of
existing libraries and harness the performance available in these languages.

We will look at creating a package with scikit-build-core. Then we will cover
some of it's most interesting and innovative features. Then we will peer into
the design and internal workings. Finally, we will look at some of the packages
that adopted scikit-build-core. A lot of the ecosystem was improved, even
beyond scikit-build-core, as part of this project, so we will also highlight
some of that work.

## History of packaging

Python has a long history, and packaging isn't something that was considered
important for quite a while. Python gained a standard library module to help
with packaging called `distutils` in Python 1.6 and 2.0, in the year 2000. Distribution
was difficult, leading to packages containing large numbers of unrelated
modules (SciPy) to reduce the number of packages one had to figure out how to
install, and "distributions" of Python, such as the Enthought distribution.

Several developments greatly improved Python's packaging story, such the
addition of a new binary format, the "wheel" made distributing binaries easier.
This, along with new hardware and a changing installation landscape highlighted
an issue with putting a packaging tool in the standard library: you can't
freely update the standard library. Third party "extensions" to distutils
appeared; the one that became the main tool for many years is setuptools. It
wasn't long before setuptools became required; building directly with distutils
was too buggy, and setuptools was injecting patches into distutils when it
loaded.

Package installers, originally `easy_install` and then the more full featured
`pip` came along. `pip` was tightly coupled to setuptools, and even helped it
out by making sure it's injections to distutils were done even if packages
didn't import setuptools first or at all. Pip was directly and deeply tied to
setuptools; if you wanted to build a package, setuptools was the only answer.
Even if you make the sdist and wheel(s) yourself, which was actually pretty
easy, as those were standardized, you couldn't make sure that `pip` wouldn't
try to use setuptools to build a wheel from an SDist.

Faults in the organically grown distutils/setuptools quickly became apparent.
You couldn't tell the installer to update setuptools from `setup.py`, because
it was just a Python file that was being run. It was hard to declare built-time
dependencies (though they really did try by running the Python file twice, the
first time with stubs). You couldn't parse metadata without running the
setup.py. It was hard to extend the setuptools commands, and the entire API was
public, which meant it was really hard to fix something without breaking
everyone else.

Third party tools for building packages started showing up, like Flit and
Poetry. These worked by making a compatibility `setup.py` and injecting it into
the SDist, just in case pip needed to build the wheel. This is when
standardization efforts, in the form of PEPs, began to change the packaging
landscape forever.

PEP 517 defined an API for build-frontends (like pip) to talk to build-backends
(like setuptools). PEP 518 defined an isolated build environment, which would
allow builders to download build dependencies, like a specific version of
setuptools or a plugin. Later, PEP 621 would add a standard way to define basic
package metadata, and PEP 660 would add an API for editable installs.

Python build backends started to appear. Most of the initial build backends
were were designed for Python only packages. `flit-core`, `poetry-core`,
`pdm-backend`, and `hatchling` are some popular build backends that support
some or all of these PEPs. Setuptools also ended up gaining support for all
these PEPs, as well.

Compiled backends were a bit slower, but today we have several great choices.
`scikit-build-core` for CMake, `meson-python` for Meson, and `maturin` for
Cargo (Rust) are the most popular. `enscons` for SCons should get a special
mention as the first binary build backend, even though it is mostly a
historical curiosity at this point.

## Scikit-build (classic)

The original scikit-build was released as PyCMake at SciPy 2014, and renamed
two years later, at SciPy 2016. Being developed well before the packaging PEPs,
it was designed as a wrapper around distutils and/or setuptools. This design
had flaws, but was the only way this could be done at the time.

Because it was deeply tied to setuptools internals, updates to setuptools or
wheel often would break scikit-build. There were a lot of limitations of
setuptools that scikit-build couldn't alleviate properly.

However, it did allow users to use a real build system (CMake) with their
Python projects. A notable example are two packages produced by the
scikit-build team: ninja and cmake redistributions on PyPI. Users could pip
install `cmake` and `ninja` anywhere that wheels worked.

In 2021, a proposal was written and accepted by the NSF to fund development on
a new package, scikit-build-core, built on top of the packaging standards and
free from setuptools and other historical cruft. Work started in 2022, and the
first major package to switch was awkward array, at the end of that year.

## Using scikit-build-core

### The simplest example

Scikit-build-core was designed to be easy to use as possible, with reasonable
defaults and simple configuration. A minimal, working example is possible with
just three files. You need a file to build, for example, this is a simple
pybind11 module in a `main.cpp`:

```cpp
#include <pybind11/pybind11.h>

PYBIND11_MODULE(example, m) {
    m.def("square", [](double x) { return x*x; });
}
```

You then need a `pyproject.toml`, and it can be as little as six lines long, just like for pure Python:

```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "example"
version = "0.0.1"
```

And finally, you need a `CMakeLists.txt` for CMake, which is also as little as six lines:

```cmake
cmake_minimum_required(VERSION 3.15...3.26)
project(example LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(example example.cpp)
install(TARGETS example LIBRARY DESTINATION .)
```

And that all that is needed to get started.

### Using binding tools

### Common needs as simple configuration

## Innovative features of scikit-build-core

### Dynamic requirement on CMake/Ninja

### Integration with external packages

### Dual editable modes with automatic recompile

## Scikit-build-core's design

### The configuration system

### Dynamic metadata

### The File API

## Adoption

### Rapids.ai

### Ninja / CMake / clang-format

### ZeroMQ

### Smaller projects

## Related work

## Summary
