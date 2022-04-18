:author: Zachary del Rosario
:email: zdelrosario@olin.edu
:institution: Assistant Professor of Engineering and Applied Statistics, Olin College of Engineering

:bibliography: references

=====================================================================================
Enabling Active Learning Pedagogy and Insight Mining with a Grammar of Model Analysis
=====================================================================================

.. class:: abstract

Modern engineering models are complex, with dozens of inputs, uncertainties arising from simplifying assumptions, and dense output data. While major strides have been made in the computational scalability of complex models, relatively less attention has been paid to user-friendly, reusable tools to explore and make sense of these models. Such tools have the potential to help engineers mine their models for valuable insights, and to enable more effective engineering pedagogy through hands-on active learning approaches. Grama is a python package aimed at supporting these activities.

Grama is a grammar of model analysis: an ontology that specifies data (in tidy form), models (with quantified uncertainties), and the verbs that connect these objects. Grama implements a :code:`Model` class that combines input metadata with an input-to-output mapping. This definition enables a reusable set of evaluation "verbs" that provide a consistent analysis toolkit across different grama models.

This paper presents three case studies that illustrate pedagogy and engineering work with grama: 1. Invariants in the "grama model" design provide teachable moments that encourage more sound analysis. 2. Reusable tools encourage users to self-initiate healthy modeling behaviors. 3. Reusable tools enable *exploratory model analysis* (EMA), an analog of exploratory data analysis augmented with a data generation loop.

.. class:: keywords

   engineering, engineering education, exploratory model analysis, software design, uncertainty quantification

Introduction
============

(TODO Background on computational engineering models)

 Every aircraft you have ever flown on has been designed using probabilistically-flawed, potentially dangerous criteria :cite:`zdr2021allowables`. The fundamental issue underlying these criteria is a flawed heuristic for uncertainty propagation; initial human subjects work suggests that engineers' tendency to mis-diagnose sources of variability as inconsequential noise may contribute to the persistent application of flawed design criteria :cite:`aggarwal2021qualitative`. These flawed treatments of uncertainty are not limited to engineering design; recent work by Kahneman et al. :cite:`kahneman2021noise` highlights widespread failures to recognize or address variability in human judgment, leading to bias in hiring, economic loss, and an unacceptably capricious application of justice.

Grama was originally developed to support model analysis under uncertainty; in particular, to enable active learning pedagogy to promote deeper student learning :cite:`freeman2014active`. The design of grama is heavily inspired by the Tidyverse :cite:`wickham2019welcome`, an interoperable set of R packages organized around the 'tidy data' concept :cite:`wickham2014tidy`. Grama uses the tidy data concept and introduces an analogous concepts for *models*.

Grama: A Grammar of Model Analysis
==================================

Grama :cite:`zdr2020grama` is a set of tools for working with *data* and *models*. Pandas :cite:`mckinney2011pandas` is used as the underlying data class, while grama implements the :code:`Model` class. A grama model includes a number of functions---mathematical expressions or simulations---and domain/distribution information for the deterministic/random inputs. The following code illustrates a simple grama model with both deterministic and random inputs [#]_.

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

Grama models are intended to be *evaluated* to generate data. The data can then be analyzed using visual and statistical means. Models can be *composed* to add more information, or *fit* to a dataset. Figure :ref:`verbs` illustrates this interplay between data and models in terms of the four categories of function "verbs" provided in grama.

.. figure:: verb-classes-bw.png
   :scale: 40%
   :figclass: bht

   Verb categories in grama. These grama functions start with an identifying prefix, e.g. :code:`ev_*` for evaluation verbs. :label:`verbs`
