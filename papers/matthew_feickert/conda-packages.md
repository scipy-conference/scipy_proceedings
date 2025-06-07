## Conda packages

Conda packages (`.conda` files) are language agnostic file archives that contain built code distributions and metadata.
This is quite powerful, as it allows for arbitrary code to be built for any target platform and then packaged with its metadata.
When a conda package is downloaded and then unpacked with a conda package management tool (e.g. Pixi, conda, mamba) it is then "installed" by copying the package's file directory tree to the base of the environment's directory tree.
Package contents are also simple; they can only contain files and symbolic links.

### conda-forge

Conda packages can be distributed on package indexes that support the concept of "channels" which redirect URLs to directory trees of conda packages.
Channel names serve as the base path for hosting packages.
The most broadly used community channel for conda-packages is the `conda-forge` channel, which hosts the conda packages generated from builds on the global conda-forge community cyberinfrastructure.
The conda-forge community operates in a GitHub organization that hosts "feedstock" Git repositions that contain conda package build recipes as well as automation infrastructure and continuous integration (CI) and continuous delivery (CD) workflows.
This allows for conda-forge community members to submit and maintain recipes &mdash; instructions for conda package build systems &mdash; to build and distribute conda packages for multiple variants of computing platforms &mdash; combinations of operating systems and hardware architectures &mdash; for Linux, macOS, and Windows.
