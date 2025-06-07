## CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs) [@CUDA_paper; @CUDA_slides].
The CUDA ecosystem provides Software Development Kits (SDKs) with APIs to CUDA that allow for software developers to write hardware accelerated programs with CUDA in various languages for NVIDIA GPUs.
CUDA has official language support for C++, Fortran, and Python, with community support for Julia and other languages.
While there are other types of hardware acceleration development platforms, as of 2025 CUDA is the most abundant platform for scientific computing that uses GPUs and effectively the default choice for major machine learning libraries and applications.

CUDA is closed source and proprietary to NVIDIA, which means that NVIDIA has historically limited the download access of the CUDA toolkits and drivers to registered NVIDIA developers (while keeping the software free (monetarily) to use).
CUDA then required a multi-step installation process [@CUDA_install_guide] with manual steps and decisions based on the target platform and particular CUDA version.
This meant that when CUDA enabled environments were setup on a particular machine they were powerful and optimized, but brittle to change and could easily be broken if system wide updates (like for security fixes) occurred.
CUDA software environments were bespoke and not many scientists understood how to construct and curate them.

## CUDA packages on conda-forge

### Initial implementation

After discussion in late 2018 [@conda-forge_github_io_issue_687] to better support the scientific developer community, the CUDA packaging community agreed to use the Anaconda `defaults` channel's [@anaconda-defaults-channel] `cudatoolkit` package.
Initially the `cudatoolkit` package was designed around Numba's CUDA needs [@conda-recipe-cudatoolkit], though it evolved to a bundle of redistributable CUDA libraries.
In 2019, NVIDIA began packaging the `cudatoolkit` package in the [`nvidia` conda channel](https://anaconda.org/nvidia).
With help from the broader community, the `cudatoolkit` package was added to `conda-forge` in 2020 [@staged-recipes-pr-12882].
For the first time, this provided users the _ability to specify different versions of CUDA libraries_ and download them in newly created conda environments.

Supporting initial conda-forge CUDA builds required additional components:
* [A conda-forge Docker image](https://github.com/conda-forge/docker-images/pull/93) using [the NVIDIA CUDA Docker images](https://hub.docker.com/r/nvidia/cuda/), which provided the NVIDIA build tools for compiling packages.
* [A shim package](https://github.com/conda-forge/staged-recipes/pull/8229) to leverage the NVIDIA build tools within a conda package build.
* [A CUDA build matrix in conda-forge's global pinnings](https://github.com/conda-forge/conda-forge-pinning-feedstock/pull/285), which tied these two pieces together.

These ideas were tied together in the first package build on September 20, 2019 [@ucx-split-feedstock-pr-14], and the initial implementation of this work was completed later in 2019.
In 2020, support was expanded to [Windows CUDA builds](https://github.com/conda-forge/conda-forge-pinning-feedstock/pull/914).
Lots of iteration on this work happened after, all using the same basic foundation.

### Revised implementation

After some time using these packages and build process, a few observations became clear.
First, some packages used only a subset of the libraries, like the driver, the CUDA runtime library, or particular library components like cuBLAS.
However, the `cudatoolkit` package shipped considerably more than that, so having finer specifications of dependencies would provide a better package maintainer and end-user experience.
Second, some packages needed components that were not part of the `cudatoolkit` bundle like other libraries or parts of the build toolchain.
Having some way to depend on these components would improve usability.
Third, the infrastructure management overhead of custom Docker images and their integration into the conda-forge build matrix was cumbersome for conda-forge maintainers.
Being able to install and use the build tools directly would simplify maintenance and benefit end-users wishing to use these build tools.

To address these issues, NVIDIA began working on a revised set of packages.
These more closely matched packages in other distribution channels (like Linux distribution package managers) and were adapted to the conda user experience.
For example, Linux distributions often install packages at the system level, which differs from the first-class userspace environment experience that conda package environments provides.
As a result, some distinctions that a Linux distribution provides are unneeded in conda.
There are additional differences around behavior in pinning versions of dependencies or how compilers are packaged and expected to work in their installed environments.
Initial production of the packages were made on the `nvidia` channel, however, all of this work was being done internally in NVIDIA and published to a separate channel.
This made the packages less visible and required additional knowledge to use.

In [2023](https://youtu.be/WgKwlGgVzYE?si=hfyAo6qLma8hnJ-N), NVIDIA began adding the releases of CUDA conda packages from the `nvidia` channel to conda-forge, making it easier to discover and allowing for community support.
Given the new package structure, NVIDIA added the packages for CUDA `12.0` to indicate the breaking change.
Also with significant advancements in system driver specification support, CUDA `12` became the first version of CUDA to be released as conda packages through conda-forge and included all CUDA libraries from the [CUDA compiler `nvcc`](https://github.com/conda-forge/cuda-nvcc-feedstock) to the [CUDA development libraries](https://github.com/conda-forge/cuda-libraries-dev-feedstock).
[CUDA metapackages](https://github.com/conda-forge/cuda-feedstock/) were also released, which allow users to easily describe the version of CUDA they require (e.g. `cuda-version=12.5`) and the CUDA conda packages they want (e.g. `cuda`).
This significantly improved the ability for researchers to easily create CUDA accelerated computing environments.
