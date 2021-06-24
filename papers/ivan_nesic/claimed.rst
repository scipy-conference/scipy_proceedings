:author: Romeo Kienzler
:email: romeo.kienzler@ch.ibm.com
:institution: IBM, Center for Open Source Data and AI Technologies (CODAIT)

:author: Ivan Nesic
:email: ivan.nesic@usb.ch
:institution: University Hospital of Basel
:orcid: 0000-0002-4373-8860

---------------------------------------------------------------
CLAIMED, a visual and scalable component library for Trusted AI
---------------------------------------------------------------

.. class:: abstract

   We propose an Open Source low-code / no-code tool suite for data science supporting rapid
   prototyping with visual editing and jupyter notebooks, seamless scaling during development 
   and deployment (including GPU), pre-build components for various business domains, 
   support for the complete python and R tooling including Apache Spark, TensorFlow, 
   PyTorch, pandas and scikit-learn, straightforward extensibility 
   (anything which runs in a Docker container), reproducibility of work, data lineage and collaboration support.
   More specifically we provide CLAIMED, the component library for AI, Machine Learning, ETL
   and Data Science driven by the JupyterLab extension "Elyra Pipeline Editor", a framework
   agnostic low-code pipeline editor currently supporting local execution, Airflow and Kubeflow.
   To exemplify its use, we constructed a workflow composed exclusively of components in CLAIMED, 
   to train, evaluate and deploy a deep learning model, that determines if CT scans are positive 
   for COVID-19 [covidata]_.
   


.. class:: keywords

    Kubernetes, Kubeflow, JupyterLab, Elyra, KFServing, TrustedAI,
    AI Explainability, AI Fairness, AI Adversarial Robustness

Introduction
============

Monolithic scripts are often used in prototyping. For production deployments, tools like 
Slurm [slurm]_, Snakemake [snakemake]_, QSub [qsub]_, HTCondor [htcondor]_, Apache Nifi [nifi]_,
NodeRED [nodered]_, KNIME [knime]_, Galaxy [galaxy]_, 
Reana [reana]_, WEKA [weka]_, Rabix [rabix]_, Nextflow [nextflow]_, OpenWDL [openwdl]_, CWL [cwl]_
or Cromwell [cromwell]_ are used. 
We found that these tools, even when used in conjunction, support only a subset of our requirements:

- low-code / no-code environment for rapid prototyping with visual editing and jupyter notebooks
- seamless scaling during development and deployment
- GPU support
- pre-build components for various business domains
- support for the complete python and R tooling including Apache Spark,
  TensorFlow, PyTorch, pandas and scikit-learn
- seamless extensibility
- reproducibility of work
- data lineage
- collaboration support

Therefore we have built an extensible component library to be used in low-code / no-code
environments called CLAIMED, the visual
**C**\ omponent **L**\ ibrary for **A**\rtificial **I**\nteligence (AI), **M**\achine Learning (ML),
**E**\xtract, Transform, Load (ETL) and **D**\ ata Science.
In the following section we elaborate on the implementation
details followed by a description of an exemplary pipeline to showcase
the capabilities of CLAIMED. Then, we elaborate on different ideas
how CLAIMED can be improved in the "Future Work" section, followed by the conclusion.

Implementation
==============

Before we can elaborate on how the requirements have been addressed with CLAIMED and how the
presented workflow has been implemented, we need to introduce some
terms and technology in the technology breakdown section.

Technology breakdown
--------------------

Containerization and Kubernetes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Virtualization opened up a lot of potential for managing the
infrastructure, mainly the ability to run different operating systems on
the same hardware at the same time. Next step of isolation can be
performed for each of the microservices running on the server, but
instead of managing access rights and resources on the host operating
system, we can containerize these in separate packages with their own
environments. Practical effect of this is that we are running each of
the microservices as if they have their own dedicated virtual machine,
but without the overhead of such endeavour. This is accomplished by
running containers on top of the host operating system. An example of
the containerization platform is Docker.

Containerization made it possible to run a large number of containers,
which introduced the need of their orchestration. This means something,
and hopefully not someone, needs to constantly take care that the system
is in the desired state. It needs to scale up and down, manage
communication between containers, schedule them, manage authentications,
balance the load etc. Although there are other options like Docker
Swarm, Kubernetes is the market leader in this domain. It was donated to
Cloud Native Computing Foundation (CNCF) [cncf]_ by Google,
which means a lot of Google’s know-how and years of
experience went into it. The system can run on public, on-premises or on
hybrid clouds. On-premises installation is very important for institutions
dealing with sensitive data. For IBM, Kubernetes is also strategic. 
This is mainly because Kubernetes enables the hybrid cloud scenario 
backed by an
open source common runtime capable of transparently moving workload 
across different on-premises, remote and cloud data centers seamlessly.
Besides acting as a Kubernetes runtime provider in the cloud and - 
through the acquisition of RedHat - becoming the major vendor for 
on-premises Kubernetes, IBM is now able to deliver software solutions
- so called "Cloud Paks" - 
on top of Kubernetes, making them run everywhere. Therefore, 
IBM joined CNCF [ibmcncf]_, moved all Watson Services to Kubernetes 
and acquired RedHat. This makes IBM the 3rd largest committer to
Kubernetes.

Deep Learning with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow is the second incarnation of the Google Brain project’s scalable
distributed training and inference system named DistBelief
[tf]_. It supports myriad of hardware platforms, from
mobile phones to GPU/TPU clusters, for both training and inference. It
can even run in browser on the client’s side, without the data ever
leaving the machine. Apart from being a valuable tool in research domain, it is
also being used in demanding production environments. On a development
side, representing machine learning algorithms in tree-like structures
makes it a very good expression interface. Lastly, on the performance vs
usability side, both graph and eager modes are supported. Eager mode allows for
easier debugging since the code is executed in Python control flow, as opposed to
the TensorFlow specific graph control flow [tfeager]_.
The advantages of graph mode is usage in distributed training, performance
optimization and production deployment.
In-depth analysis of these two modes can be found here [tfbook]_.

Kubeflow
~~~~~~~~

Kubeflow [kubeflow]_ is a machine learning pipeline management and execution system
running as first class citizen on top of Kubernetes. Besides making use
of Kubernetes scalability it allows for reproducible work as machine
learning pipelines and the results and intermediate artifacts of their
executions are stored in a meta data repository.

Elyra
~~~~~~~~~~~~~~~~

Elyra [elyra]_ started as a set of extensions for the JupyterLab ecosystem.
Here we concentrate on the pipeline editor of Elyra which allows for
expression of machine learning workflows using a drag’n’drop editor and
send them for execution on top of different engines like Kubeflow or
Airflow. This means non-programmers can understand and create machine 
learning workflows on their own. Elyra also supports visualizing
such pipelines in the browser (e.g. from a github repository).

JupyterLab
~~~~~~~~~~

JupyterLab [jupyter]_ is one of the most popular development environments for data
science. Therefore we started to support JupyterLab first. But the
pipeline editor of Elyra will be supported in other environments as
well, VSCode [vscode]_ being next on the list.

AI Explainability
~~~~~~~~~~~~~~~~~

Despite the good performance, deep learning models are viewed as
being black box approaches.
Technically, deep learning models are a series of non-linear feature
space transformations, but an intuitive understanding of each of the
individual processing steps is not trivial.
There are techniques with which we can look over a deep learning model’s shoulder.
The one we are using here is called LIME [lime]_. LIME takes the
existing classification model and permutes images taken from the
validation set (therefore the real class label is known to LIME) as long as a
misclassification is happening. That way LIME can be used to create heat
maps as image overlays to indicate regions of images which are most
relevant for the classifier. In other words, we identify
regions of the image the classifier is looking at.

As Fig. :ref:`limefig` illustrates, the most relevant areas in an image
for classifying for COVID-19 are areas containing bones over lung tissue
which indicates a problem with that particular classifier.

.. figure:: lime2.png

   Example on how LIME helps to identify classification relevant
   areas of an image. :label:`limefig`

AI Fairness and Bias
~~~~~~~~~~~~~~~~~~~~

So what is bias? "Bias is a disproportionate weight in
favor of or against an idea or thing, usually in a way that is
closed-minded, prejudicial, or unfair" [bias]_. But what we want from 
our model is to be fair and unbiased towards protected attributes like 
race, age, socioeconomic status, religion and so on. So wouldn't
it be easier if we just "hid" those columns from the model during the training?
Unfortunately the problem is convoluted. Protected attributes are often
encoded inside the other attributes (latent features).
For example, race, religion and
socioeconomic status are latently encoded in attributes like zip codes,
contact methods or types of products purchased. Therefore, fairness assessment and
bias detection is quite challenging. Luckily, a huge number of single
number metrics exist to assess bias in data and models. Here, we are
using the AIF360 [aif360]_ library. IBM donated it to
the Linux Foundation AI, which puts it under open governance.

AI Adversarial Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~

Another pillar of Trusted AI is adversarial robustness. For example, 
as researchers found out, adversarial noise can be introduced in data (data poisoning)
or models (model poisoning) to influence models decisions in favor of
the adversarial. Libraries like the Adversarial Robustness Toolbox
ART [art]_ support all state-of-the-art attacks and
defenses.

Requirements and System Architecture 
------------------------------------
In the following we cover the system architecture as well as the
requirements for the different parts of the
system architecture: Execution Engine and Visual Workflow Editor.

Execution Engine
~~~~~~~~~~~~~~~~
An execution engine takes a pipeline description and executes it on top
of physical machines reading source data and creating output data.
The following requirements have been defined in order to assess the
adequacy of the execution engine.

- Kubernetes Support

  We defined Kubernetes as the lowest layer of abstraction because that
  way the executor layer is agnostic of the underlying Infrastructure as a service (IaaS)
  architecture. In addition, Kubernetes provides better resource utilization if multiple
  pipelines are run in parallel on the system. We can consume Kubernetes as a service (aaS) offered by a variety
  of Cloud providers like IBM, Amazon, Google, Microsoft, OVH or Linode.
  A lot of workload for this particular project has been envisioned to be outsourced
  to SciCore [scicore]_ - a scientific computing data center part of the Swiss Personalized
  Health Network (SPHN) [sphn]_ and the Swiss Institute of Bioinformatics [sib]_ which runs
  on OpenStack and provides Kubernetes as part of it (Magnum), while on premises of the
  on premises of the University Hospital Basel we have RedHat OpenShift.

- GPU support

  GPU support is essential since a large fraction of the workload is
  training of deep learning neural networks on TensorFlow and PyTorch.
  Training those models on CPU doesn't make sense economically and
  ecologically.

- Component Library

  Predefined, ready to use components, are convenient to use,
  save time and, if well tested, reduce the probability of an error.
  Kubeflow for example has components for parallel training of TensorFlow models (TFJob),
  parallel execution of Apache Spark jobs as a pipeline step,
  parallel hyperparameter tuning (Katib) and model serving (KFServing/
  KNative)

- Reproducibility

  From a legal point of view, in certain domains, it is
  necessary to reconstruct a certain decision, model or output
  dataset for verification and audit. Therefore the ability to reproduce
  and re-run a pipeline is a critical requirement. Of course,
  there are other examples where this is imperative, like in science.

- Data Lineage

  Although a subset of reproducibility, Data Lineage is a crucial
  feature when it comes to visualizing the changes the datasets went
  through   during the pipeline execution.

.. table:: Execution engines against requirements. (Abbreviations: KF=Kubeflow, AF=Airflow, SM=Snakemake) :label:`engxreq`

    ================== == == ===== == ==== ======== =====
    Requirement        KF AF Slurm SM Qsub HTCondor Reana
    ================== == == ===== == ==== ======== =====
    Kubernetes Support X  X        X       X        X
    GPU support        X  X  X     X  X    X        X
    Component Library  X
    Reproducibility    X  X        X       X        X
    Data Lineage       X                            X
    ================== == == ===== == ==== ======== =====

Integrated tools
~~~~~~~~~~~~~~~~
Integrated tools are tools which include a visual data flow editor,
a component library and an execution engine.
Prominent candidates in the open source space are Apache Nifi,
NodeRED, KNIME and Galaxy.

The following additional requirements have been defined for a suitable
tool:

- Low-Code/No-Code/Visual Editing

  Citizen data scientists (in our demo example, medical doctors) need to
  work with the tool, so visual editing is necessary. But apart from
  being a visual editing tool only, support for creating custom
  pipeline components on the fly using R and python is necessary
  as well.

- Jupyter Notebooks

  Researchers in general are used to implement tasks in jupyter notebooks.
  This makes support for JupyterLab, as well as having an
  easy way of making jupyter notebooks part of the data processing pipeline,
  a key requirement.

.. table:: Integrated tools against requirements. :label:`toolxreq`

    ================== ==== ======= ===== ====== =====
    Requirement        Nifi NodeRED KNIME Galaxy Elyra
    ================== ==== ======= ===== ====== =====
    Kubernetes Support                    X      X
    GPU support                           X      X
    Component Library  X    X       X     X      X
    Reproducibility    X            X     X      X
    Data Lineage       X                  X      X
    Visual Editing     X    X       X     X      X
    Jupyter Notebooks                            X
    ================== ==== ======= ===== ====== =====


Final technology choice
~~~~~~~~~~~~~~~~~~~~~~~
As it can be seen from the tables :ref:`engxreq` and :ref:`toolxreq`,
only Kubeflow on the execution engine side, and Elyra as the
integrated tool are capable of covering all of the requirements.
Therefore we select this pair as our primary technology choice.

Elyra's pipeline editor supports drag’n’drop functionality, for adding arbitrary
scripts (shell, R, python) and Jupyter notebooks from the file explorer
to the canvas. To each one of them can be assigned a container image
which they would use to run on.
Elyra supports submission of the pipelines to Airflow and Kubeflow at the
moment.

Together with Kubeflow and JupyterLab (where Elyra runs as an extension)
all our requirements are fulfilled.
Kubernetes support, GPU support, an existing and growing component
library, Reproducibility and Data Lineage is provided by Kubeflow
and visual editing with low code support through Jupyter notebooks
and collaboration support with Git is supported by Elyra and 
JupyterLab.

.. figure:: architecture.png

   Runtime architecture of CLAIMED. :label:`architecture`

As it can be seen on Figure :ref:`architecture`, Elyra -  specifically
the pipeline editor of the Elyra Extension to JupyterLab - allows
for visually building data pipelines with a set of assets like
notebooks and scripts dragged on a canvas and transparently published
to Kubeflow as a Kubeflow pipeline.

The only thing missing is a set of re-usable notebooks for different
kinds of tasks. This is where CLAIMED kicks in. We've created CLAIMED
as open source library [complib]_. In the next sections we
will introduce the demo use case, along with how components found in CLAIMED
have been used to implement this pipeline.

System Implementation and Demo Use Case
=======================================

A TrustedAI image classification pipeline
-----------------------------------------

As already mentioned previously, pipelines are a great way to introduce
reproducibility, scaling, auditability and collaboration in machine
learning. Pipelines are often a central part of a ML-Ops strategy. This
especially holds for TrustedAI pipelines since reproducibility and
auditability are even more important there. Figure :ref:`pipeline`
illustrates the exemplary TrustedAI pipeline we have built using the
component library and Figure :ref:`kfp` is a screenshot taken from
Kubeflow displaying the pipeline after finishing it’s run.

.. figure:: elyra_pipeline_zoomed.png

   The exemplary TrustedAI pipeline for the health care use case. :label:`pipeline`

Pipeline Components
-------------------



This section exemplifies each existing category with at
least one component which has been used for this particular pipeline. 
There are also other components that are not part of the pipeline,
so they haven't been exemplified here.
Please note that the core feature of our software is threefold:

- the CLAIMED component library
- Elyra with it's capability to use CLAIMED to create a pipeline 
  and push it to Kubeflow
- the pipeline itself



Input Components
~~~~~~~~~~~~~~~~

There are input components for different types
of data source, like files and databases.

In this particular case, we’re pulling data directly from the GitHub
repository via a public and permanent link [covidata]_. We only pull the
metadata.csv and images directory.


.. figure:: kfp.png

   The pipeline once executed in Kubeflow. :label:`kfp`

Transform Components
~~~~~~~~~~~~~~~~~~~~

Sometimes, transformations on the metadata, or any other structured
dataset, are necessary. Therefore, we provide a generic transformation
component - in this case we just used it to change to format of the
categories as the original file contained forward slashes which made it
hard to use on the underlying operating system. This is performed by
simply specifying the column name and function to be applied on that column.

Filter Components
~~~~~~~~~~~~~~~~~

Similar to changing content of rows in a dataset, removing rows is also
a common task in data engineering. The filter stage allows doing exactly that.
It is enough to provide a predicate - in this case the
predicate ``~metadata.filename.str.contains('.gz')`` removes invalid
images.

Image Transformer Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: images_folder_tree.png

   Example of directory structure supported by TensorFlow Dataset API. :label:`imgdir`

One supported standard for the conversion of image datasets into the TensorFlow's
dataset supported format, is to organize images into directories representing
their classes [tfimgprep]_. TensorFlow Dataset is an API that
allows for a convenient way to create datasets from various input data,
apply transformations and preprocessing steps and make iteration over
the data easier and memory efficient [tfdataset]_.

In our example, the data aren’t in the required format. It is organized as
a directory full of images and alongside it is a CSV file which defines the
attributes. Available attributes are exam finding, sex and age,
from which we only require the finding for our example.
The images are then arranged by following the previously
described directory structure, as illustrated by Fig. :ref:`imgdir`.
After performing this step, the data can be consumed by the Tensorflow Dataset API.

Training Components
~~~~~~~~~~~~~~~~~~~

Understanding, defining and training deep learning models is not simple.
Training a deep learning image classification model requires a
properly designed neural network architecture. Luckily, the community
trends towards predefined model architectures, which are parameterized
through hyper-parameters. At this stage, we are using the MobileNetV2, a
small deep learning neural network architecture with the set of the most
common parameters. It ships with the TensorFlow distribution - ready to
use, without any further definition of neurons or layers. As shown in
figure :ref:`trainingstage`, only a couple of parameters
need to be specified.

Although possible, hyper-parameter search is not considered in this
processing stage as we want to make use of Kubeflow’s hyper-parameter
search capabilities leveraged through Katib [katib]_ in
the future.

.. figure:: trainstage.png

   Source code of the wrapped training component. :label:`trainingstage`


Evaluation Components
~~~~~~~~~~~~~~~~~~~~~

Besides define, compile and fit, a model needs to be evaluated before it
goes into production. Evaluating classification performance against the
target labels has been state-of-the-art since the beginning of machine
learning, therefore we have added components like confusion matrix. But
taking TrustedAI measures into account is a newly emerging practice.
Therefore, components for AI Fairness, AI Explainability and AI
Adversarial Robustness have been added to the component library.

Blessing Components
~~~~~~~~~~~~~~~~~~~

In Trusted AI (but not limited to) it is important to obtain a blessing of assets like
generated data, model or report to be published and used by other
subsystems or humans. Therefore, a blessing component uses the results
of the evaluation components to decide if the assets are ready for
publishing.

Publishing Components
~~~~~~~~~~~~~~~~~~~~~

Depending on the asset type, publishing means either persisting a data
set to a data store, deploying a machine learning model for consumption
of other subsystems or publishing a report to be consumed by humans.
Here, we exemplify this category by a KFServing [kfserving]_ component which
publishes the trained TensorFlow deep learning model to Kubernetes.
KFServing, on top of KNative, is particular interesting as it draws from
Kubernetes capabilities like canary deployment and scalability (including
scale to zero) in addition to built-in Trusted AI functionality.

Future Work
===========
We have financial support to add functionality to CLAIMED in multiple
dimensions. Below we give a summary of the next steps.

Extend component library
------------------------

To this date, at least one representative component for each category has
been released. Components are added to the library on a regular basis. 
The components due to be published are: Parallel Tensorflow Training with
TFJob, Parallel Hyperparameter Tuning with Katib and Parallel Data
Processing with Apache Spark.

Component exporter for Kubeflow
-------------------------------

Containerizing notebooks and scripts is a frequent task in the data science community.
In our environment, this involves attaching the arbitrary assets,
like jupyter notebooks and scripts, to a container image and then
transpiling a Kubeflow component out of it.
We are currently in the process of implementing a tool that would facilitate this workflow.
The name of the tool is C3 [c3]_, and it stands for CLAIMED component compiler.

Import/Export of components to/from Galaxy
------------------------------------------
As seen in Table 2,
Galaxy covers a majority of our requirements already.
Unfortunately, Galaxy components - called "tools" - are very skewed 
towards genomics.
Adding new components and extending functionality onto other domains would make
the tool interesting for a wider audience.
Reverse is also true, the existing component library Galaxy is extensive,
well established and tested. It makes sense to automatically transpile
those tools as components into CLAIMED. We are currently looking into
adding import/export support between CLAIMED and Galaxy into C3.


UX improvements of the Elyra pipeline editor
--------------------------------------------

The components are isolated, so only explicitly shared information can be put into context for
all of them. In order for the components' executor, e.g. Kubflow, to do this,
it must be provided a configuration. We envision for Elyra to automatically deduce
interesting parameters from the code and from the environment, upon which it would create
dynamic forms. For example, fields like checkboxes and dropdowns where one can select
input and output files mentioned in the code. Currently, only environment variables
are provided in a rudimentary UI with one text field per variable.
One proposal is to introduce an optional configuration block to the scripts and notebooks.
It would then be interpreted by Elyra and the appropriate UI would be rendered.

One successful example of such implementation is Galaxy's UI [galaxy_ui]_.
A complex UI behavior is expressed by XML configuration. So we are also exploring an
option of either using Galaxy's XML Schema or defining a new one and support the
transformation from one into the other.


Add CWL support to the Elyra pipeline editor
--------------------------------------------

CWL is a powerful workflow expression language supported already by
various tools we've evaluated. Currently, Elyra uses its own, 
proprietary pipeline representation format. Addin support of CWL to Elyra would
improve interoperability between different software components. 
For example, Reana, execution engine used in the particle physics community, and Galaxy
partially already support CWL. This means it would be possible export
pipelines from Elyra to Reana, without the need of transpiling the pipeline.
Alternatively, Elyra could integrate export and import of CWL into its pipeline editor.


Import 3rd party component libraries
------------------------------------
Since the only thing needed for something to become a CLAIMED component
is to be wrapped in a container image and assigned some meta data,
it is possible for 3rd party component libraries like those from KNIME or
Nifi and to be imported into CLAIMED.
This also holds true for Kubeflow components.
It is also possible to wrap different components from KNIME, Nifi or
similar tools in this manner and use it within Elyra, as well as in
the other execution engines CLAIMED supports.


Create more (exemplary) pipelines
---------------------------------
At the moment, CLAIMED ships with three exemplary pipelines. The health
care inspired TrustedAI pipeline which was covered in this paper,
a pipeline to visualize and predict soil temperature from a historic
data set and an IoT sensor data analysis pipeline. The next pipeline
in line is a genomics pipeline for the Swiss Institute of
Bioinformatics affiliates University Hospital Berne/Berne University
and potentially for particle physics at CERN.


Conclusion
==========

We’ve build and proposed a trustable, low-code, scalable and open source
visual AI pipeline system on top of various common components
used in the machine learning community. Using Kubeflow Pipelines
provides reproducibility and auditability. Using Kubernetes provides
scalability and standardization. Using Elyra for visual development
provides ease of use, such that all internal and external stakeholders
are empowered to audit the system in all dimensions.

References
----------
.. [bias] Steinbock, Bonnie (1978). *Speciesism and the Idea of Equality*, Philosophy, 53 (204): 247–256, doi:10.1017/S0031819100016582

.. [aif360] AI Fairness 360 Toolkit, https://github.com/Trusted-AI/AIF360. Last accessed 18 Feb 2021

.. [aix360] AI Explainability 360 Toolkit, https://github.com/Trusted-AI/AIX360 Last accessed 18 Feb 2021

.. [elyra] Elyra AI, https://github.com/elyra-ai. Last accessed 18 Feb 2021

.. [kubernetes] Kubernetes, https://kubernetes.io/. Last accessed 18 Feb 2021

.. [jupyter] JupyterLab, https://jupyter.org/. Last accessed 18 Feb 2021

.. [kfserving] KFServing, https://www.kubeflow.org/docs/components/serving/kfserving Last accessed 18 Feb 2021

.. [lime] Marco Tulio Ribeiro et al. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, pp. 1135–1144 (2016), doi:10.1145/2939672.2939778

.. [kubeflow] https://www.kubeflow.org/ Last accessed 18 Feb 2021

.. [katib] Katib, https://github.com/kubeflow/katib. Last accessed 18 Feb 2021

.. [tf] Martín Abadi et al. *TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems*, arXiv:1603.04467v2, March 2016

.. [art] Adversarial Robustness Toolbox, https://github.com/Trusted-AI/adversarial-robustness-toolbox. Last accessed 18 Feb 2021

.. [ibmcncf] IBM joining CNCF, https://developer.ibm.com/technologies/containers/blogs/ibms-dedication-to-open-source-and-its-involvement-with-the-cncf Last accessed 18 Feb 2021

.. [cncf] Cloud Native Computing Foundation, https://www.cncf.io. Last accessed 18 Feb 2021

.. [complib] https://github.com/elyra-ai/component-library

.. [ect] https://github.com/cloud-annotations/elyra-classification-training/tree/developer_article

.. [slurm] https://slurm.schedmd.com/documentation.html

.. [snakemake] https://snakemake.github.io/

.. [qsub] https://en.wikipedia.org/wiki/Qsub

.. [htcondor] https://research.cs.wisc.edu/htcondor/

.. [galaxy] https://galaxyproject.org/

.. [reana] https://reanahub.io/

.. [nifi] https://nifi.apache.org/

.. [nodered] https://nodered.org/

.. [knime] https://www.knime.com/

.. [weka] https://www.cs.waikato.ac.nz/ml/weka/

.. [rabix] https://rabix.io/

.. [nextflow] https://www.nextflow.io/

.. [openwdl] https://openwdl.org/

.. [cwl] https://www.commonwl.org/

.. [cromwell] https://cromwell.readthedocs.io/en/stable/

.. [covidata] Joseph Paul Cohen et al. *COVID-19 Image Data Collection: Prospective Predictions Are the Future*, arXiv:2006.11988, 2020

.. [tfeager] https://www.tensorflow.org/guide/eager/

.. [tfdataset] https://www.tensorflow.org/api_docs/python/tf/data/Dataset

.. [tfimgprep] https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

.. [galaxy_ui] https://github.com/bgruening/galaxytools/blob/c1027a3f78bca2fd8a53f076ef718ea5adbf4a8a/tools/sklearn/pca.xml#L75

.. [c3] https://github.com/romeokienzler/c3

.. [tfbook] https://www.oreilly.com/library/view/whats-new-in/9781492073727/

.. [vscode] https://code.visualstudio.com/

.. [scicore] https://scicore.unibas.ch/

.. [sphn] https://sphn.ch/

.. [sib] https://www.sib.swiss/
