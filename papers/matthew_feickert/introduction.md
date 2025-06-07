## Introduction

A critical component of research software sustainability is the reproducibility of the software and computing environments software operates and "lives" in.
Providing software as a "package" &mdash; a standardized distribution of all source or binary components of the software required for use along with identifying metadata &mdash; goes a long way to improved reproducibility of software libraries.
However, more researchers are consumers of libraries than developers of them, but still need reproducible computing environments for research software applications that may be run across multiple computing platforms &mdash; e.g. scientific analyses, visualization tools, data transformation pipelines, and artificial intelligence (AI) and machine learning (ML) applications on hardware accelerator platforms (e.g. GPUs).
While workflow engines and Linux containers offer a gold standard for scientific computing reproducibility, they require additional layers of training and software engineering knowledge.
Modern open source multi-platform environment management tools, e.g. Pixi [@pixi], provide automatic multi-platform digest-level lock file support for all dependencies &mdash; down to the compiler level &mdash; of software on public package indexes (e.g. PyPI [@PyPI_website] and conda-forge [@conda-forge_community]) while still providing a high level interface well suited for researchers.
Combined with the arrival of the full CUDA [@CUDA_paper] stack on conda-forge, it is now possible to declaratively specify a full CUDA accelerated software environment.
We are now at a point where well supported, robust technological solutions exist, even for applications with highly complex software environments.
What is currently lacking is the education and training by the broader scientific software community to adopt these technologies and build community standards of practice around them, as well as an understanding of what are the most actionably useful features of adopting computational reproducibility tools.

## Reproducibility

"Reproducible" research is a term that can mean multiple things across various fields.
Some fields may view work as "reproducible" if the full process is documented, and other may view "reproducible" as meaning that all computations will give the same numerical outputs barring entropy variations.
As there are multiple levels of reproducibility, we will restrict "reproducibility" to software environment reproducibility.
We define this as be limited to the ability to define and programmatically create a software environment composed of packages that specifies all software, and its dependencies, with exact URLs and binary digests ("hashes").
Reproducible environments need to be machine agnostic in that for a specified computing platform in the environment they must be installable without modification across multiple instances.

### Hardware accelerated environments

Software the involves hardware acceleration on computing resources like GPUs requires additional information to be provided for full computational reproducibility.
In addition to the computer platform, information about the hardware acceleration device, its supported drivers, and compatible hardware accelerated versions of the software in the environment (GPU enabled builds) are required.
While this information is straightforward to collect, traditionally this has been difficult to make use of in practice given software access restrictions and the lack of declarative human interfaces for defining relationships between system-level drivers and user software.
Multiple recent technological advancements (made possible by social agreements and collaborations) in the scientific open source world now provide solutions to these problems.
