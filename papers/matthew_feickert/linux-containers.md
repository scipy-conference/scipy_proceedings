## Deploying environments to remote compute

Often researchers are running scientific and machine learning workflows on remote computational resources that use batch computing systems (e.g. HTCondor, SLURM).
For systems with shared filesystems (e.g. SLURM) it is possible to use Pixi workspaces in workflows in a similar manner to local machine (e.g. laptop or workstation).
Other systems (e.g. HTCondor) do not have a shared filesystem (e.g. HTCondor), requiring that each worker node receive its own copy of the software environment.
While locked Pixi environments significantly help with this, it is often advantageous to distribute the environment in the form of a Linux container image to the compute resources.
These systems are able to mount Linux container images to worker nodes in ways that reduce the disk and memory cost to the user's session, compared to installing Pixi and then downloading all dependencies of the software environment from the package indexes used.
This also reduces the bandwidth use as the Linux container image can be cached at the compute resource host and efficiently replicated to the worker nodes, paying the bandwidth cost of download once.
While Linux container technology historically has presented additional engineering and design overhead to researchers, Linux container construction of Pixi environments is simple and can be reduced to templated format.
An example in the form of a templated Dockerfile is seen in @example-pixi-dockerfile.[^docker_footnote]
The template requires user input to define the target CUDA version (`CUDA_VERSION`) and the name of the Pixi environment to install (`ENVIRONMENT`).
As the Pixi environment is already fully defined and locked it can be directly installed as normal in the `build` stage of the container image build, along with an entrypoint shell script that will activate the environment, and then copied from the `build` stage into the `final` stage to reduce the total image size by removing the cache and reducing the total number of layers in the final image.

```{literalinclude} code/ml-example/Dockerfile
:label: example-pixi-dockerfile
:caption: The template structure of a Dockerfile for a locked Pixi environment with CUDA dependencies. The only values that need user input are the CUDA version and the name of the target environment.
```

The Dockerfile can then be built into a Linux container image binary file which can be distributed to a container image registry.
Batch computing system workflow definition files can use these container images to provide the software environment for the computing jobs, which pull the images from the container image registry when requested by the job.

[^docker_footnote]: As many compute facilities do not allow for use of Docker directly given security concerns, Apptainer container image formats are more common.
Apptainer definition files are similarly easy to write as compared to Dockerfiles and Docker container images can be converted into a format that Apptainer can use.
As Docker is a more common format in the broader computing world, including commercial settings, it has been used for this example.
These workflows are not limited to a single container image format.
