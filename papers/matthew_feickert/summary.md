## Summary

As hardware accelerated code becomes more common across scientific computing, especially CUDA accelerated software for machine learning, the need for simple but powerful solutions for software environment management has grown too.
The simple and flexible structure of conda packages allows for complex projects to be packaged as directory trees of built binaries on a platform specific level.
This has allowed for the complexity of the CUDA software stack to be efficiently built as conda packages using the conda-forge cyberinfrastructure and then distributed on the conda-forge conda channel for public use.
Distribution of CUDA conda packages on conda-forge additionally allows for other conda-forge projects to use CUDA conda packages in their builds, resulting in a wide selection of CUDA enabled projects, including many machine learning packages.
Through use of Pixi's declarative specification of dependencies in the project manifest and non-optional digest level lock file generation, software environments can now be declaratively and rapidly constructed, resolved, and locked using semantic operations well designed for scientific researchers.
With these powerful technologies and abstractions, researchers can now construct machine learning and data science environments for multiple platforms at once and use trusted patterns to develop locally and deploy to remote computational resources.

In addition to the long term reproducibility provided by the combination of these technologies, the maintenance burden and complexity reduction should not be overlooked.
With the CUDA v12 distributions on conda-forge, researches no longer need to have experience in CUDA internals and distribution installation to accelerate their software projects.
They need only know the supported versions of CUDA by the NVIDIA drivers on their target machines.
Researchers also no longer need to use multiple tools to build bespoke workflows for constructing and maintaining lock files for multiple environments and platforms, while keeping environment definition files and lock files synced.
Pixi provides a single tool and unified interface to achieve the same results faster while using high level abstractions &mdash; removing most of the work of software environment reproducibility from the user workflow.[^timing_footnote]
Having the full specification of the software environment including the CUDA dependencies also removes runtime failures due to missing, unspecified, or incompatible system-level requirements on remote compute resources.
Most importantly, reducing cognitive overhead and the latency to reach a usable software environment reduces the time to insight for researchers, transferring the problems of scientific computing back into their domains of expertise.

[^timing_footnote]: It is worth reflecting on that the auspicious interplay of these technologies is a recent advancement that could not have happened significantly earlier.
Conda-forge was created as a project in 2015 (at the SciPy 2015 conference), and the first distributions of CUDA v12 were in 2022, with Pixi being created in 2023.
The advancements that have occurred in less than two years after their coexistence is a reflection of the power of strong design standards and collaboration across the conda-forge community.
