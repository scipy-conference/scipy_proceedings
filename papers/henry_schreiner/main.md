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

### Common needs as simple configuration

Customizing scikit-build-core is done through the `[tool.scikit-build]` table
in the `pyproject.toml`. An example configuration is shown below:

```{code-block}
:linenos:

[tool.scikit-build]
minimum-version = "0.10"

build.verbose = true
logging.level = "INFO"

wheel.expand-macos-universal-tags = true
```

On line 2, you can see the `minimum-version` setting. This is a special setting
that allows scikit-build-core to make changes to default values in the future
while ensuring your configuration continues to build with the defaults present
the version you specify. This is very similar to CMake's
`cmake_minimum_requires` feature.

On line 4-5, you can see an example of increasing the verbosity, both to print
out build commands as well as see scikit-build-core's logs.

On line 7, you can see that scikit-build-core can expand the macOS universal
tags to include both ARM and Intel variants for maximum compatibility with
older Pip versions; this is impossible to do in setuptools unless you use API
declared private and unstable in wheel.

## Innovative features of scikit-build-core

Scikit-build-core has some interesting and powerful innovations. This section
is by no means exhaustive; it doesn't cover scikit-build-core's over forty
different configuration options, for example. It is instead just meant to
highlight some of the most interesting features of scikit-build-core.

### Dynamic requirement on CMake/Ninja

Scikit-build-core needs CMake to build, and often ninja as well. Both are
available as Python wheels, but simply adding them to your
`build-system.requires` list is not a good idea, because some platforms aren't
supported by wheels on PyPI, such as WebAssembly, the various BSD's,
ClearLinux, or Android. But these systems can get CMake other ways, such as
from a package manager, so users shouldn't have a build fail because they can't
get a cmake wheel if they have a sufficient version of CMake already. As it
turns out, PEP 517 is flexible enough to handle this very elegantly.

PEP 517 has an API for a build tool to declare it's dependencies. This was
initially used for setuptools to only depend on wheel when making wheels, but
it works perfectly for optional binary dependencies as well. Scikit-build-core
will check to see if `cmake` is present on the system. If it is, and it reports
a version sufficient for the package, it will not be added to the requested
dependencies. This is also done for ninja. This same system was added to
`meson-python`, as well, since Meson also requires ninja.

### Integration with external packages

Scikit-build-core has several mechanisms to allow packages on PyPI to provide
CMake configuration and tools. There are three mechanisms provided:

First, the site-packages directory is added to the CMake search path by
scikit-build-core. Due to the way CMake config file discovery works, this
allows a package to provide a `<PkgName>Config.cmake` file in any GNU-like
standard location in the package (like
`pybind11/share/cmake/pybind11/pybind11Config.cmake`, for example).

The second way is with an entry point `cmake.prefix`, which adds prefix
dirs, useful for adding module files that don't have to match the installed
package name or are in arbitrary locations.

And the final way is with the `cmake.module` entry point, which can be used
to add any CMake helper files.

### Dual editable modes with automatic recompile

Editable installs are supported in two modes. The default mode installs a
custom finder that combines the Python files in your development directory (as
specified by packages) with CMake installed files in site-packages. It is
important to rely on importlib.resources instead of file manipulation in this
mode. This mode supports installs into multiple environments.

This mode also supports automatic rebuilds. If you enable it, then importing
your package will rebuild on any changes to the source files, something that
could not be done with setuptools.

There is also an opt-in "inplace" mode that uses CMake's inplace build to build
your files in the source directory. This is very similar to setuptools
`build_ext --inplace`, and works with tools that can't run Python finders, like
type checkers. This mode works with a single environment, since the build dir
_is_ the source dir. There is no dynamic finder in this mode, so automatic
rebuilds are not supported.

### Dynamic metadata

A system for allowing custom metadata plugins was contributed and has been one
of the most successful parts of scikit-build-core. Any metadata field other
than the `name` can be dynamically specified in Python packaging; the
dynamic-metadata mechanism provides a way for plugins to be written by anyone,
including "in-tree" plugins written inside a specific package. This gives authors
quite a bit of freedom to do things like customize the package description or
read the version from a file.

This system is being worked on as a standard for other build backends to use
and a stand-alone namespace. Changes are expected, but the current implementation
in scikit-build-core uses the following API:

```python
def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any],
) -> str:
   ...
```

Plugins provide this function. Three built-in plugins are provided in
scikit-build-core: one for regex, one that wraps `setuptools_scm`, and one that
wraps `hatch-fancy-pypi-readme`. Using one of these looks like this:

```toml
name = "mypackage"
dynamic = ["version"]

[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "src/mypackage/__init__.py"
```

The `provider` key tells scikit-build-core where to look for the plugin. There
is also a provider-path`for local plugins. All other keys are implemented by
the plugin; the regex plugin implements`input`and`regex`, for exmaple.

An extra feature that is not being proposed for broader adoption is a
genearation mechanim. Scikit-build-core can generate a file for you with
metadata provided via templating. It looks like this:

```toml
[[tool.scikit-build.generate]]
path = "package/_version.py"
template = '''
version = "${version}"
'''
```

This is extreamly useful due to the `location` key, which defaults to
`install`, which will put the file only in the built wheel, `build`, which puts
the file in the build directory, and `source`, which writes it out to the
source directory and SDist, much like setuptools_scm normally does. In the
default mode, though, no generated files are placed in your source tree.

### Overrides

Static configuration has many benefits, but it has a significant drawback; you often
want to configure different situations differently. For example, you might want a higher
minimum CMake version on Windows, or a pure Python version for unreleaed Python versions.
These sorts of things can be expressed with overrides, which was designed after
overrides in cibuildwheel, which in turn were based on mypy's overrides.
Scikit-build-core has the most powerful version of the system, however, with an
`.if` table that can do version comparisons and regexs, and supports any/all
merging of conditions, and inhertance from a prevous override (also added to
cibuildwheel). It includes conditions for the state of teh build (wheel, sdist,
editable, or metadata builds) and environment variables. An example is shown below:

```toml
[[tool.scikit-build.overrides]]
if.platform-system = "darwin"
cmake.version = ">=3.18"
```

## Scikit-build-core's design

This section is devoted to the internals of scikit-build-core.

### The configuration system

### The File API

### Plugins for other systems

## Adoption

### Rapids.ai

### Ninja / CMake / clang-format

### ZeroMQ

### Smaller projects

## Related work

## Summary
