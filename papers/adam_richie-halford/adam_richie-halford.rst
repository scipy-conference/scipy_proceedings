:author: Adam Richie-Halford
:email: richiehalford@gmail.com
:institution: University of Washington, Seattle, WA

:author: Ariel Rokem
:email: arokem@gmail.com
:institution: University of Washington, Seattle, WA

:bibliography: mybib

------------------------------------------------
A Numerical Perspective to Terraforming a Desert
------------------------------------------------

.. class:: abstract

   A short version of the long version that is way too long to be written as a
   short version anyway.  Still, when considering the facts from first
   principles, we find that the outcomes of this introspective approach is
   compatible with the guidelines previously established.

   In such an experiment it is then clear that the potential for further
   development not only depends on previous relationships found but also on
   connections made during exploitation of this novel new experimental
   protocol.

.. class:: keywords

   terraforming, desert, numerical perspective

Introduction
------------

In the quest to minimize time-to-first-result, data scientists are
increasingly turning to cloud-based distributed computing with
commercial vendors like Amazon Web Services (AWS). However, because of
the complexity and steep learning curve associated with a transition to
cloud computing, it remains inaccessible. A number of Python libraries
have sought to close this gap by allowing users to interact seamlessly
with AWS resources from within their Python environment. For example
Pywren enables users to run their existing Python code on AWS Lambda,
providing convenient distributed execution for jobs that fall within
the limits of this service (maximum 300 seconds of execution time, 1.5
GB of RAM, 512 MB of local storage, and no root access). However, these
limitations are impractical for many data-oriented workloads, that
require more RAM and local storage, longer compute times, and complex
dependencies. Here, we introduce a new Python library: Cloudknot [1][2],
that launches Python functions as jobs on the AWS Batch service, thereby
lifting these limitations.

Methods
-------

Cloudknot employs the single program, multiple data (SPMD) paradigm
to achieve parallelism. A Python user-defined function (UDF) is
automatically wrapped in a command line interface (CLI) and packaged
into a Docker container, together with its dependencies. The container
is uploaded into the AWS Elastic Container Registry (ECR), and the
location of this container on AWS ECR is supplied as an AWS Batch job
definition.

A list of inputs (i.e. the MD in SPMD) is provided to AWS Batch, and
the job definition is executed in parallel on each element of the list.
Cloudknot uses Amazon Simple Storage Service (S3) as an intermediary,
storing the inputs in a format that conforms to the expectations of the
CLI. Finally, Cloudknot directs AWS Batch to spin up one EC2 or Spot
instance for each element of the inputs, pointing each instance to its
own element.

All of the interaction with AWS ECR, S3, and Batch is automated:
the user-facing API is one object (a `Knot`) that orchestrates this
interaction. Inputs are provided using the `Knot.map()` method, which
returns a list of ‘futures’ for the results, mimicking Python’s
concurrent futures’ `Executor` objects.

Results
-------

Because Cloudknot’s approach favors “embarrassingly parallel”
applications, one should expect near-linear scaling with an additional
fixed overhead for creating AWS resources and transmitting results
through S3. This suits use-cases for which execution time is much
greater than the time required to create the necessary resources on AWS
(infrastructure setup time can be minimized, reusing AWS resources that
have already been created). We show near-linear scaling for a scientific
use-case: analysis of human brain MRI data. This use-case demonstrates
that Cloudknot does not introduce undue overhead burden, exploiting the
scaling efficiency of underlying AWS Batch infrastructure.

Conclusion
----------

Cloudknot simplifies cloud-based distributed computing by
programmatically executing UDFs in AWS Batch. This lowers the barrier to
cloud computing and allows users to launch massive compute workloads at
scale from within their Python environment.

Examples
--------

In this section, we will present a few use-cases of `Cloudknot`.
We will start with examples that have minimal software and data dependencies, and increase the complexity by adding first data dependencies and subsequently complex software dependencies.


Simulations
~~~~~~~~~~~
Simulation use-cases are straightforward. In contrast to `pywren`, simulations executed with `Cloudknot` do not have to comply with any particular memory or time limitations.
While `pywren`'s limitations stem from the use of the AWS Lambda service.


Data Dependencies: Analysis of magnetic resonance imaging data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependency of individual tasks on data can be addressed by preloading the data into object storage on S3, and the downloading of individual bits of data needed to complete each task  into the individual worker machines. For example, we implemented an analysis pipeline for human MRI data.

We replicated the pipeline that we used in a previous study [mehta2017comparative]_. This allows us to compare the performance of `Cloudknot` directly against the performance of several alternative systems for distributed computing: Spark [Zaharia2010-rp]_, Myria [Halperin2014-vu]_ and Dask [Rocklin2015-ra].



Two important caveats to this analysis An important caveat is that the timing data for the other systems is from early 2017, and some of these systems have evolved since then.


Data and software dependencies: analysis of microscopy data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another field that has seen a dramatic increase in data volumes is the field of cell biology and molecular engineering. In this example,




Acknowledgements
----------------

References
----------
