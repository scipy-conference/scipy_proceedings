:author: Adam Richie-Halford
:email: richiehalford@gmail.com
:institution: University of Washington, Seattle, WA

:author: Ariel Rokem
:email: arokem@gmail.com
:institution: University of Washington, Seattle, WA

:bibliography: mybib

------------------------------------------------------------------
Cloudknot: A Python Library to Run your Existing Code on AWS Batch 
------------------------------------------------------------------

.. class:: abstract

   We introduce cloudknot, a software library that simplifies
   cloud-based distributed computing by programmatically executing
   user-defined functions (UDFs) in AWS Batch. It takes as input
   a Python function, packages it as a container, creates all the
   necessary AWS constituent resources to submit jobs, monitors their
   execution and gathers the results, all from within the Python
   environment. Cloudknot overcomes limitations of previous similar
   libraries, such as pywren, that runs UDFs on AWS Lambda, because most
   data science workloads exceed the AWS Lambda limits on execution
   time, RAM, and local storage.

.. class:: keywords

   cloud computing, Amazon AWS


Introduction
------------

In the quest to minimize time-to-first-result, data scientists are
increasingly turning to cloud-based distributed computing with
commercial vendors like Amazon Web Services (AWS). However, because of
the complexity and steep learning curve associated with a transition to
cloud computing, it remains inaccessible. A number of Python libraries
have sought to close this gap by allowing users to interact seamlessly
with AWS resources from within their Python environment. For example
pywren :cite:`jonas2017` enables users to run their existing Python
code on AWS Lambda, providing convenient distributed execution for
jobs that fall within the limits of this service (maximum 300 seconds
of execution time, 1.5 GB of RAM, 512 MB of local storage, and no
root access). However, these limitations are impractical for many
data-oriented workloads, that require more RAM and local storage, longer
compute times, and complex dependencies. Here, we introduce a new Python
library: cloudknot :cite:`cloudknot-docs` :cite:`cloudknot-repo`, that
launches Python functions as jobs on the AWS Batch service, thereby
lifting these limitations.


Methods
-------

The primary object in cloudknot is the :code:`Knot`, which employs the
single program, multiple data (SPMD) paradigm to achieve parallelism.
In this section, we describe cloudknot's approach to establishing the
single program (SP) and managing the multiple data (MD). :code:`Knot`'s
user-facing API and interactions with cloud-based resources are depicted
in Figure :ref:`fig.workflow`.

.. figure:: figures/cloudknot_workflow.pdf

   Cloudknot's SPMD workflow. The left two columns depict steps
   cloudknot takes to create the SP. The right columns depicts
   cloudknot's management of the MD. Blue rounded squares represent
   components of cloudknot's user-facing API. Yellow circles represent
   AWS resources. Grey document shapes represent containers,
   templates, or data used to communicate with cloud resources.
   :label:`fig.workflow`


Single Program (SP)
~~~~~~~~~~~~~~~~~~~

:code:`Knot` creates the single program on initialization, taking a
user-defined function (UDF) as input and wrapping it in a command line
interface (CLI) that downloads data from an Amazon S3 bucket specified
by an input URL. The UDF is also wrapped in a Python decorator that
sends its output back to an S3 bucket. So in total, the command line
program downloads input data from S3, executes the UDF, and sends
output back to S3. :code:`Knot` then packages the CLI, along with its
dependencies, into a Docker container. The container is uploaded into
the Amazon Elastic Container Registry (ECR). Cloudknot's use of Docker
allows it to handle non-trivial software and data dependencies (see the
microscopy examples later in this paper).

Separately, :code:`Knot` uses an AWS CloudFormation template to create
the AWS resources required by AWS Batch:

- AWS Identity and Access Management (IAM) Roles

  - A batch service IAM role to allow AWS Batch to make calls to other
    AWS services on the user's behalf

  - An ECS instance role to be attached to each container instance when
    it is launched

  - A EC2 Spot Fleet role to allow Spot Fleet to bid on, launch, and
    terminate instances if the user chooses to use Spot Fleet instances
    instead of dedicated EC2 instances.

- An AWS Virtual Private Cloud (VPC) with subnets and a security group

- An AWS Batch job definition specifying the job to be run. :code:`Knot`
  passes the location of the Docker container on AWS ECR to this job
  definition so that all jobs execute the SP.

- An AWS Batch job queue that schedules jobs onto a compute environment.

- An AWS Batch compute environment, which is a set of compute resources
  that will be used to run jobs. The user may ask :code:`Knot` to
  restrict the compute environment to only certain instance types (e.g.
  ``c4.2xlarge``) or may choose a specific Amazon Machine Image (AMI)
  to be loaded on each compute resource. Or thay may simply request a
  minimum, desires, and maximum number of virtual CPUs and let AWS Batch
  select and manage the EC2 instances.

:code:`Knot` uses sensible defaults for the job definition and compute
environment parameters so that the casual user may never need to concern
themselves with selecting an instance type or specifying an AMI. More
advanced users can their jobs' memory requirements, instance types, or
AMIs. This might be necessary if the jobs require special hardware (e.g.
GPGPU computing) or if the user wants more fine-grained control over
which resources are launched.

Finally, :code:`Knot` exposes AWS resource tags to the user so that
they can assign metadata to each created resource. This facilitates
management of cloudknot generated resources and allows the user to
quickly recognize cloudknot resources in the AWS console.


Multiple Data (MD)
~~~~~~~~~~~~~~~~~~

To operate on the MD, the :code:`Knot.map()` method serializes each
element of the input and sends it to S3, organizing the data in a schema
that is internally consistent with the expectations of the CLI. It then
launches an AWS Batch array job (or optionally, separate individual
Batch jobs) to execute the program over these data. When run, each batch
job selects its own input, executes the UDF, and returns its serialized
output to S3.

Finally, :code:`Knot.map()` downloads the output from S3 and returns
it to the user. Since AWS Batch, and therefore cloudknot, allows
arbitrarily long execution times, :code:`Knot.map()` returns a list
of futures for the results, mimicking Python's concurrent futures'
:code:`Executor` objects.

Under the hood, :code:`Knot.map()` creates a
:code:`concurrent.futures.ThreadPoolExecutor` instance where each
thread intermittently queries S3 for its returned output. The results
are encapsulated in :code:`concurrent.futures.Future` objects, allowing
asynchronous execution. The user can use :code:`Future` methods such
as :code:`done()` and :code:`result()` to test for success or view the
results. This also allows them to attach callbacks to the results using
the :code:`add_done_callback()` method. For example a user may want to
perform a local reduction on results generated on AWS Batch.


API
---

|warning| The above interactions with AWS resources are hidden from the
user. The advanced or curious user can customize the Docker container or
cloudformation template. But for basic use cases, here is an example of
using the API

.. code-block:: python

   # Insert really awesome code example here
   import cloudknot as ck

|warning|


Results
-------

Because cloudknot's approach favors "embarrassingly parallel"
applications, one should expect near-linear scaling with an additional
fixed overhead for creating AWS resources and transmitting results
through S3. This suits use-cases for which execution time is much
greater than the time required to create the necessary resources on AWS
(infrastructure setup time can be minimized, reusing AWS resources that
have already been created). We show near-linear scaling for a scientific
use-case: analysis of human brain MRI data. This use-case demonstrates
that cloudknot does not introduce undue overhead burden, exploiting the
scaling efficiency of underlying AWS Batch infrastructure.


Conclusion
----------

cloudknot simplifies cloud-based distributed computing by
programmatically executing UDFs in AWS Batch. This lowers the barrier to
cloud computing and allows users to launch massive compute workloads at
scale from within their Python environment.


Examples
--------

In this section, we will present a few use-cases of cloudknot. We will start with examples that have minimal software and data dependencies, and increase the complexity by adding first data dependencies and subsequently complex software and resource dependencies.


Simulations
~~~~~~~~~~~

Simulation use-cases are straightforward. In contrast to pywren, simulations executed with cloudknot do not have to comply with any particular memory or time limitations.
While pywren's limitations stem from the use of the AWS Lambda service.


Data Dependencies: Analysis of magnetic resonance imaging data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because cloudknot is run on the standard AWS infrastructure, it allows specification of complex and large data dependencies. Dependency of individual tasks on data can be addressed by preloading the data into object storage on S3, and the downloading of individual bits of data needed to complete each task into the individual worker machines.

As an example, we implemented a pipeline for analysis of human MRI data. Human MRI data is a good use-case for a system such as cloudknot, because much of the analysis in computational pipelines that analyze this type of data proceeds in an embarassingly parallel manner: even for large data-sets with multiple subjects, a large part of the analysis is conducted first at the level of each individual brain, and aggregation of information across brains is typically done after many preprocessing and analysis stages are done at the level of each individual.

For example, diffusion MRI (dMRI) is a method that measures the properties of the connections between different regions of the brain. Over the last few decades, this method has been used to establish the role of these connections in many different cognitive and behavioral properties of the human brain, and to delineate the role that the biology of these connections plays in neurological and psychiatric disorders [XXX]. Because of the interest in these connections, several large consortium efforts for data collection have aggregated large datasets of human dMRI data from multiple different subjects.

In analysis of dMRI data, the first few steps are done at the individual level: selection of regions of interest within each image, denoising and initial modeling of the data. These are the steps that were implemented in the pipeline that we used in a previous study :cite:`mehta2017comparative`, and we reused this pipeline in the current study. This allows us to compare the performance of cloudknot directly against the performance of several alternative systems for distributed computing that were studied in our previous work: Spark :cite:`Zaharia2010-rp`, Myria :cite:`Halperin2014-vu` and Dask :cite:`Rocklin2015-ra`

In cloudknot, we used the reference implementation from this previous study written in Python and using methods implemented in Python and Cython in Dipy :cite:`Garyfallidis2014`. In contrast to all of these other systems, essentially no changes had to be made to the reference implementation when using cloudknot, except to download data from S3 into the individual instances. Parallelization was implemented only at the level of individual subjects, and a naive serial approach was taken at the level of each individual.

As expected, with a small number of subjects this reference implementation is significantly slower with cloudknot compared with the parallelized implementation in these other systems. But the relative advantage of these systems diminshes substantially as the number of subjects grows larger (Figure XXX), and the benefits of parallelization across subjects starts to be more substantial.

Two important caveats to this analysis: the first is that the analysis with the other systems was all conducted on a 16-node cluster (each node was an AWS r3.2xlarge instance with 8 vCPUs). The benchmark code does run faster with more nodes added to the cluster. Notably, even for the largest amount of data (25 subjects) that was executed in cloudknot, AWS chooses to deploy only two instances of the r4.16xlarge type -- each with 64 vCPUs and 488 GB of RAM. In terms of RAM, this is the equivalent of a 16 node cluster of r3.2xlarge, but the number of CPUs deployed to the task is much half. The other is that that the timing data for the other systems is from early 2017, and some of these systems have evolved and improved since.


Data and software dependencies: analysis of microscopy data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MRI example demonstrates the use of a large and rather complex dataset. In addition, cloudknot can manage complex software dependencies. Researchers in cell biology, molecular engineering and nano-engineering are also increasingly relying on methods that generate large amounts of data and on analysis that requires large amounts of compute power. For example, in experiments that evaluate the mobility of synthetically designed nano-particles in biological tissue :cite:`Nance2017-xp`, :cite:`Nance2012-nu`, researchers may record movies of microscopic images of the tissue at high spatial and temporal resolution and with wide field of view, resulting in large amounts image data, often stored in multiple large image files. To analyze these experiments, researchers rely on software implemented in ImageJ for particle segmentation and tracking, such as TrackMate :cite:`Tinevez2017-ti`. However, when applied to large amounts of data, using TrackMate serially in each experiment can be prohibitively time consuming. One solution is to divide the movies spatially into smaller field of view movies, and analyze them in parallel :cite:`Curtis2018`.

Another field that has seen a dramatic increase in data volumes is the field of cell biology and molecular engineering. These fields often rely on the ImageJ software. This software, written in Java, can be scripted using Jython. However, this requires installation of the ImageJ Jython run-time.
Because cloudknot relies on docker, this installation can be managed using the command line interface (i.e. :code:`wget`). Once a docker image is created that contains the software dependencies for a particular analysis, Python code can be written on top of it to execute system calls that will run the analysis. This is the approach taken here. We do not provide a quantitative benchmark for this example.

Because of the data size in this case, a custom AMI had to be created from the AWS Batch AMI, that includes a larger volume (Batch AMI volumes are limited to XXX GB of disk-space).

In summary: rather complex sets of dependencies both in terms of the software required, as well as the data and resources that are required can be managed with the combination of docker, AWS and cloudknot, but putting together such combinations may require more work and more expertise in managing each of these parts.


Acknowledgements
----------------
This work was funded through a grant from the Gordon & Betty Moore Foundation and the Alfred P. Sloan Foundation to the University of Washington eScience Institute. Thanks to Chad Curtis and Elizabth Nance for the collaboration on the implementation of a cloudknot pipeline for analysis of microscopy data.


References
----------

.. |warning| image:: figures/warning.jpg
             :scale: 3%
