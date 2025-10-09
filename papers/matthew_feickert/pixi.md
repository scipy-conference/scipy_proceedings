## Pixi

### Conceptual overview

[Pixi](https://www.pixi.sh/) is a cross-platform package and environment manager that can handle complex development workflows [@pixi; @pixi-docs].
Importantly, Pixi automatically and non-optionally will produce or update a lock file &mdash; a structured file that contains a full list of all environments defined with a complete list of all packages, as well as definition of each packages with digest information on the binary &mdash; for the software environments defined by the user whenever any actions mutate the environment.
Pixi is written in Rust, and leverages the language's speed and technologies to solve environments fast.

Pixi addresses the concept of computational reproducibility by focusing on a set of main features

1. **Virtual environment management**: Pixi can create environments that contain conda packages and Python packages and use or switch between environments easily.
1. **Package management**: Pixi enables the user to install, update, and remove packages from these environments through the `pixi` command line.
1. **Task management**: Pixi has a task runner system built-in, which allows for tasks with custom logic and dependencies on other tasks to be created.

These features become powerful when combined with robust behaviors

1. **Automatic lock files**: Any changes to a Pixi workspace that can mutate the environments defined in it will automatically and non-optionally result in the Pixi lock file for the workspace being updated.
This ensures that any state of a Pixi project is trivially computationally reproducible.
1. **Solving environments for other platforms**: Pixi allows the user to solve environment for platforms other than the current user machine's.
This allows for users to solve and share environment to any collaborator with confidence that all environments will work with no additional setup.
1. **Pairity of conda and Python packages**: Pixi allows for conda packages and Python packages to be used together seamlessly, and is unique in its ability to handle overlap in dependencies between them.
Pixi will first solve all conda package requirements for the target environment, lock the environment, and then solve all the dependencies of the Python packages for the environment, determine if there are any overlaps with the existing conda environment, and the only install the missing Python dependencies.
This ensures allows for fully reproducible solves and for the two package ecosystems to compliment each other rather than potentially cause conflicts.
1. **Efficient caching**: Pixi uses an efficient global cache shared between all Pixi projects and globally installed tools on a machine.
The first time Pixi installs a package it will download the files to the global cache and link the files into the environment.
When Pixi has to reinstall the same package in a different environment, the package will be linked from the same cache, making sure internet bandwidth for downloads and disk space is used as efficiently as possible.

Pixi users declaratively specify their project dependencies which are recorded in a Pixi manifest `pixi.toml` file (which for Python projects can optionally be embedded in a `pyproject.toml` `[pixi]` table) and automatically resolved in the `pixi.lock` lock file.
This declarative nature allows for users to efficiently specify their project requirements while being guaranteed a static and reproducible environment from the lock file.

### CUDA hardware accelerated environment creation

Combining the features of modern CUDA `12` conda packages with Pixi's environment management, it is now possible to efficiently manage multiple software environments that can include both hardware accelerated and CPU environments.
An example Pixi workspace is presented in @pixi-ml-example-workspace

```{literalinclude} code/ml-example/pixi.toml
:linenos:
:end-line: 49
:label: pixi-ml-example-workspace
:caption: Example of a multi-platform and multi-environment Pixi manifest with all required information and constraints to resolve and install CUDA accelerated conda packages.
```

where the definition of multiple platforms allows for solving the declared environments for all platforms while on other platforms

```{literalinclude} code/ml-example/pixi.toml
:linenos:
:start-line: 0
:end-line: 4
:emphasize-lines: 4
```

the `cpu` feature defines `dependencies` and `tasks` that are accessible from the `cpu` environment

```{code} toml
:filename: pixi.toml

...

[feature.cpu.dependencies]
pytorch-cpu = ">=2.7.1,<3"
torchvision = ">=0.22.0,<0.23"

[feature.cpu.tasks.train-cpu]
description = "Train a PyTorch CNN on MNIST on CPU"
cmd = "python ./src/torch_MNIST.py --epochs 2 --save-model --data-dir data"

...

[environments]
cpu = ["cpu"]
```

The `gpu` feature does the same for the `gpu` environment, but it also importantly defines a [`system-requirements` table](https://pixi.sh/v0.50.2/workspace/system_requirements/) that define the system specifications needed to install and run a Pixi workspace's environments.


```{code} toml
:filename: pixi.toml

...

[feature.gpu.system-requirements]
cuda = "12"

[feature.gpu.target.linux-64.dependencies]
pytorch-gpu = ">=2.7.1,<3"
torchvision = ">=0.22.0,<0.23"

[feature.gpu.target.win-64.dependencies]
pytorch-gpu = ">=2.7.1,<3"
torchvision = ">=0.22.0,<0.23"

[feature.gpu.tasks.train-gpu]
description = "Train a PyTorch CNN on MNIST on GPU"
cmd = "python ./src/torch_MNIST.py --epochs 14 --save-model --data-dir data"

...

[environments]
...
gpu = ["gpu"]
```

`system-requirements` build upon the concept of conda "[virtual packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html)", allowing for the dependency resolver to enforce constraints declared by defining compatibility of the system with virtual packages, like `__cuda`.
In the particular case of CUDA, the `system-requirements` table specifies the CUDA version the workspace expects the host system to support, as detected through the host system's NVIDIA driver API.
While the `system-requirements` field values do not correspond to lower or upper bounds, specifying that the workspace is expected to work on systems that support CUDA 12

```{code} toml
:filename: pixi.toml

...

[feature.gpu.system-requirements]
cuda = "12"

...

```

ensures that packages depending on `__cuda >= 12` are resolved correctly.
This effectively means that declaring the system requirement will cause the Pixi dependency resolver to find CUDA enabled packages that are compatible with CUDA 12, disallowing for incompatible package builds to be resolved.
Once these package dependencies have been resolved and locked, this ensures that any system capable of meeting the system requirement will get working CUDA accelerated conda packages installed.

Not all machines will have an NVIDIA GPU on them to allow for the system requirements to be resolved correctly.
To allow for non-CUDA-supported-machines to still resolve Pixi workspace requirements, shell environment overrides exist through the `CONDA_OVERRIDE_CUDA` environmental variable.
Setting `CONDA_OVERRIDE_CUDA=12` on a machine that doesn't meet the CUDA version requirements, will override the supported virtual packages and set a value of `__cuda=12` for the system.
This can be clearly understood from setting the override and then querying the workspace summary with `pixi info`, as seen in @conda-override-cuda-example.
This is a powerful functionality as it allows for environment specification, resolution, and locking for target platforms that users might not have access to, but can be assured are valid.

```{code} console
:label: conda-override-cuda-example
:caption: Demonstration of using the `CONDA_OVERRIDE_CUDA` environmental variable on a system with no CUDA support (an Apple silicon machine) to allow dependency resolution as if it supported CUDA 12.

% pixi info
System
------------
       Pixi version: 0.54.1
           Platform: osx-arm64
   Virtual packages: __unix=0=0
                   : __osx=15.3.2=0
                   : __archspec=1=m2
...

% CONDA_OVERRIDE_CUDA=12 pixi info
System
------------
       Pixi version: 0.54.1
           Platform: osx-arm64
   Virtual packages: __unix=0=0
                   : __osx=15.3.2=0
                   : __cuda=12=0
                   : __archspec=1=m2
...
```

Pixi also allows for feature composition to efficiently create new environments.
@pixi-ml-example-workspace's `gpu` and `inference` features are combined and resolved collectively to provide a new CUDA accelerated `inference` environment that does not affect the `gpu` environment.
The same applies for the `lab` feature and environment, which additionally provides JupyterLab for interactive programming with notebooks.

```{code} toml
:filename: pixi.toml

...

[feature.inference.dependencies]
matplotlib = ">=3.10.3,<4"

[feature.lab.dependencies]
notebook = ">=7.4.5,<8"
jupyterlab = ">=4.4.7,<5"

...

[environments]
...
gpu = ["gpu"]
inference = ["gpu", "inference"]
lab = ["gpu", "inference", "lab"]
```

Composing multiple environments from Pixi features allows for separating conceptual steps of scientific analysis into bespoke software environments that contain only the necessary dependencies.
This allows for each step's environment to be better defined, potentially with radically different or conflicting dependencies from other steps, and for clean separation between interactive and non-interactive ("batch") computing models.

### Locked environments

Once the workspace has been defined, any Pixi operation on the workspace will result in all environments in the workspace having their dependencies resolved and then fully specified ("locked") at the digest ("hash") level in a single `pixi.lock` Pixi lock file, as seen in @example-pixi-lockfile.
The lock file is a YAML file that contains two definition groups: `environments` and `packages`.
The `environments` group lists every environment in the workspace for every platform with a complete listing of all packages in the environment.
The `packages` group lists a full definition of every package that appears in the `environments` lists, including the package's URL and digests (e.g. sha256, md5).
These groups provide a full description of every package described in the Pixi workspace and its dependencies and constraints on other packages.
Versioning the lock file along with the manifest file in a version control system allows for workspaces to be fully reproducible to the byte level indefinitely into the future, conditioned on the continued existence of the package indexes the workspace pulls from (e.g. conda-forge, PyPI, the nvidia conda channel).
In the event that long term preservation and reproducibility are of importance, there are community projects [@pixi-pack] that allow for downloading all dependencies of a Pixi environment and generating a tar archive containing all of the packages, which can later be unpacked and installed.

```{literalinclude} example.lock
:filename: pixi.lock
:label: example-pixi-lockfile
:caption: Example structure of a `pixi.lock` Pixi lock file showing the definition of the environments as well as a full description of each package used in each environment.
```
