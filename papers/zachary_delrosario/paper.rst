:author: Zachary del Rosario
:email: zdelrosario@olin.edu
:institution: Assistant Professor of Engineering and Applied Statistics, Olin College of Engineering

:bibliography: references

=====================================================================================
Enabling Active Learning Pedagogy and Insight Mining with a Grammar of Model Analysis
=====================================================================================

.. class:: abstract

Modern engineering models are complex, with dozens of inputs, uncertainties arising from simplifying assumptions, and dense output data. While major strides have been made in the computational scalability of complex models, relatively less attention has been paid to user-friendly, reusable tools to explore and make sense of these models. Grama is a python package aimed at supporting these activities. Grama is a grammar of model analysis: an ontology that specifies data (in tidy form), models (with quantified uncertainties), and the verbs that connect these objects. This definition enables a reusable set of evaluation "verbs" that provide a consistent analysis toolkit across different grama models. This paper presents three case studies that illustrate pedagogy and engineering work with grama: 1. Providing teachable moments through planned errors, 2. Providing reusable tools to help users self-initiate productive modeling behaviors, and 3. Enabling *exploratory model analysis* (EMA)---exploratory data analysis augmented with data generation.

.. class:: keywords

   engineering, engineering education, exploratory model analysis, software design, uncertainty quantification

Introduction
============

Modern engineering relies on scientific computing. Computational advances enable faster analysis and design cycles by reducing the need for physical experiments; for instance, finite-element analysis enables computational study of aerodynamic flutter, and reynolds-averaged Navier-Stokes simulation supports the simulation of jet engines---both of these are enabling technologies that support the design of modern aircraft :cite:`keane2005computational`. Modern areas of computational research include heterogeneous computing environments :cite:`mittal2015survey`, task-based parallelism :cite:`bauer2012legion`, and big data :cite:`sagiroglu2013big`. Another line of work considers the development of *integrated tools* to unite diverse disciplinary perspectives in a single, unified environment, e.g. the integration of multiple physical phenomena in a single code :cite:`esmaily2020benchmark` or the integration of a computational solver and data analysis tools :cite:`maeda2022integrated`. Such integrated computational frameworks are highlighted as *essential* for applications such as computational analysis and design of aircraft :cite:`slotnick2014cfd`. While engineering computation has advanced along the aforementioned axes, the conceptual understanding of practicing engineers has lagged in key areas.

Every aircraft you have ever flown on has been designed using probabilistically-flawed, potentially dangerous criteria :cite:`zdr2021allowables`. The fundamental issue underlying these criteria is a flawed heuristic for uncertainty propagation; initial human subjects work suggests that engineers' tendency to mis-diagnose sources of variability as inconsequential noise may contribute to the persistent application of flawed design criteria :cite:`aggarwal2021qualitative`. These flawed treatments of uncertainty are not limited to engineering design; recent work by Kahneman et al. :cite:`kahneman2021noise` highlights widespread failures to recognize or address variability in human judgment, leading to bias in hiring, economic loss, and an unacceptably capricious application of justice.

Grama was originally developed to support model analysis under uncertainty; in particular, to enable active learning pedagogy to promote deeper student learning :cite:`freeman2014active`. This toolkit aims to *integrate* the disciplinary perspectives of computational engineering and statistical analysis within a unified environment to support a *coding to learn* pedagogy :cite:`barba2016computational`. The design of grama is heavily inspired by the Tidyverse :cite:`wickham2019welcome`, an integrated set of R packages organized around the 'tidy data' concept :cite:`wickham2014tidy`. Grama uses the tidy data concept and introduces an analogous concepts for *models*.

Grama: A Grammar of Model Analysis
==================================

Grama :cite:`zdr2020grama` is an integrated set of tools for working with *data* and *models*. Pandas :cite:`mckinney2011pandas` is used as the underlying data class, while grama implements the :code:`Model` class. A grama model includes a number of functions---mathematical expressions or simulations---and domain/distribution information for the deterministic/random inputs. The following code illustrates a simple grama model with both deterministic and random inputs [#]_.

.. [#] Throughout, :code:`import grama as gr` is assumed.

.. code-block:: python

		# Each cp_* function adds information to the model
		md_example = (
		    gr.Model("An example model")
		    # Overloaded `>>` provides pipe syntax
		    >> gr.cp_vec_function(
		        fun=lambda df: gr.df_make(f=df.x+df.y+df.z),
			var=["x", "y", "z"],
			out=["f"],
		    )
		    >> gr.cp_bounds(x=(-1, +1))
		    >> gr.cp_marginals(
		        y=gr.marg_mom("norm", mean=0, sd=1),
		        z=gr.marg_mom("uniform", mean=0, sd=1),
		    )
		    >> gr.cp_copula_gaussian(
		        df_corr=gr.df_make(var1="y", var2="z", corr=0.5)
		    )
		)

While an engineer's interpretation of the term "model" focuses on the input-to-output mapping (the simulation), and a statistician's interpretation of the term "model" focuses on a distribution, the grama model integrates both perspectives in a single model.

Grama models are intended to be *evaluated* to generate data. The data can then be analyzed using visual and statistical means. Models can be *composed* to add more information, or *fit* to a dataset. Figure :ref:`verbs` illustrates this interplay between data and models in terms of the four categories of function "verbs" provided in grama.

.. figure:: verb-classes-bw.png
   :scale: 40%
   :figclass: bht

   Verb categories in grama. These grama functions start with an identifying prefix, e.g. :code:`ev_*` for evaluation verbs. :label:`verbs`

Defaults for Concise Code
-------------------------

Grama verbs are designed with sensible default arguments to enable concise code. For instance, the following code visualizes input sweeps across its three inputs, similar to a *ceteris paribus* profile :cite:`kuzba2019pyceterisparibus,biecek2020paribus`.

.. code-block:: python

		(
		    ## Concise default analysis
		    md_example
		    >> gr.ev_sinews(df_det="swp")
		    >> gr.pt_auto()
		)

This code uses the default number of sweeps and sweep density, and constructs a visualization of the results. The resulting plot is shown in Figure :ref:`example-sweep`.

.. figure:: example-sweep.png
   :scale: 50%
   :figclass: bht

   Input sweep generated from the code above. Each panel visualizes the effect of changing a single input, with all other inputs held constant. :label:`example-sweep`

Grama imports the plotnine package for data visualization :cite:`kibirige2021plotnine`, both to provide an expressive grammar of graphics, but also to implement a variety of "autoplot" routines. These are called via a dispatcher ``gr.pt_auto()`` which uses metadata from evaluation verbs to construct a default visual. Combined with sensible defaults for keyword arguments, these tools provide a concise syntax even for sophisticated analyses. The same code can be slightly modified to change a default argument value, or to use plotnine to create a more tailored visual.

.. code-block:: python

		(
		    md_example
		    ## Override default parameters
		    >> gr.ev_sinews(df_det="swp", n_sweeps=10)
		    >> gr.pt_auto()
		)

		(
		    md_example
		    >> gr.ev_sinews(df_det="swp")
		    ## Construct a targeted plot
		    >> gr.tf_filter(DF.sweep_var == "x")
		    >> gr.ggplot(gr.aes("x", "f", group="sweep_ind"))
		    + gr.geom_line()
		)

This system of defaults is important for pedagogical design: Introductory grama code can be made extremely simple when first introducing a concept. However, the defaults can be overridden to carry out sophisticated and targeted analyses. We will see in the Case Studies below how this concise syntax encourages sound analysis among students.

Case Studies
============

Planned Errors as Teachable Moments
-----------------------------------

.. code-block:: python

		md_flawed = (
		    gr.Model("An example model")
		    >> gr.cp_vec_function(
		        fun=lambda df: gr.df_make(f=df.x+df.y+df.z),
			var=["x", "y", "z"],
			out=["f"],
		    )
		    >> gr.cp_bounds(x=(-1, +1))
		    >> gr.cp_marginals(
		        y=gr.marg_mom("norm", mean=0, sd=1),
		        z=gr.marg_mom("uniform", mean=0, sd=1),
		    )
		    ## NOTE: No copula specified
		)

		(
		    md_flawed
		    ## This code will throw the following Error
		    >> gr.ev_sample(n=1000, df_det="nom")
		)


.. error::

   ``ValueError``: Present model copula must be defined for sampling. Use ``CopulaIndependence`` only when inputs can be guaranteed independent. See the Documentation chapter on Random Variable Modeling for more information. https://py-grama.readthedocs.io/en/latest/source/rv_modeling.html

Encouraging Sound Analysis
--------------------------

Exploratory Model Analysis
--------------------------
