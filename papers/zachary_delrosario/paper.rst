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

Grama :cite:`zdr2020grama` is an integrated set of tools for working with *data* and *models*. Pandas :cite:`mckinney2011pandas` is used as the underlying data class, while grama implements a :code:`Model` class. A grama model includes a number of functions---mathematical expressions or simulations---and domain/distribution information for the deterministic/random inputs. The following code illustrates a simple grama model with both deterministic and random inputs [#]_.

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

An advantage of a unified modeling environment like grama is the opportunity to introduce *planned errors as techable moments*.

It is common in probabilistic modeling to make problematic assumptions. For instance, Cullen and Frey :cite:`cullen1999probabilistic` note that modelers frequently and erroneously treat the normal distribution as a default choice for all unknown quantities. Another common issue is to assume, by default, the independence of all random inputs to a model. This is often done *tacitly*---with the independence assumption unstated. These assumptions are problematic, as they can adversely impact the validity of a probabilistic analysis :cite:`zdr2021allowables`.

To highlight the dependency issue for novice modelers, grama uses error messages to provide just-in-time feedback to a user who does not articulate their modeling choices. For example, the following code builds a model with no dependency structure specified. The result is an error message that summarizes the conceptual issue and points the user to a primer on random variable modeling.

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
		    ## NOTE: No dependency specified
		)
		(
		    md_flawed
		    ## This code will throw an Error
		    >> gr.ev_sample(n=1000, df_det="nom")
		)


.. error::

   ``ValueError``: Present model copula must be defined for sampling. Use ``CopulaIndependence`` only when inputs can be guaranteed independent. See the Documentation chapter on Random Variable Modeling for more information. https://py-grama.readthedocs.io/en/latest/source/rv_modeling.html

Grama is designed both as a teaching tool and a scientific modeling toolkit. For the student, grama offers teachable moments to help the novice grow as a modeler. For the scientist, enforces practices that promote scientific reproducibility.

Encouraging Sound Analysis
--------------------------

As mentioned above, concise grama syntax is desirable to *encourage sound analysis practices*. Grama is designed to support higher-level learning outcomes :cite:`bloom1956taxonomy`; for instance, rather than focusing on *applying* programming constructs to generate model results, grama is intended to help users *study* model results ("evaluate", according to Bloom's Taxonomy). Sound computational analysis demands study of simulation results, e.g. to check for numerical instabilities. This case study makes this learning outcome distinction concrete by considering *parameter sweeps*.

Generating a parameter sweep similar to Figure :ref:`example-sweep` with standard Python libraries requires a considerable amount of boilerplate code, manual coordination of model data, and explicit loop construction: The following code generates parameter sweep data using standard libraries. Note that this code sweeps through values of ``x`` holding values of ``y`` fixed; additional code would be necessary to construct a sweep through ``y`` [#]_.

.. [#] Code assumes ``import numpy as np; import pandas as pd``.

.. code-block:: python

    ## Manual approach
    # Gather model data
    x_lo = -1; x_up = +1;
    y_lo = -1; y_up = +1;
    f_model = lambda x, y: x**2 * y
    # Analysis parameters
    nx = 10               # Grid resolution for x
    y_const = [-1, 0, +1] # Constant values for y
    # Generate data
    data = np.zeros((nx * len(y_const), 3))
    for i, x in enumerate(np.linspace(x_lo, x_up, num=nx)):
        for j, y in enumerate(y_const):
            data[i + j*nx, 0] = f_model(x, y)
            data[i + j*nx, 1] = x
            data[i + j*nx, 2] = y
    # Package data for visual
    df_manual = pd.DataFrame(
        data=data,
        columns=["f", "x", "y"],
    )

The ability to write low-level programming constructs---such as the loops above---is an obviously worthy learning outcome in a course on scientific computing. However, not all courses should focus on low-level programming constructs. Grama is not designed to support low-level learning outcomes; instead, the package is designed to support a "coding to learn" philosophy :cite:`barba2016computational` focused on higher-order learning outcomes to support sound modeling practices.

Parameter sweep functionality can be achieved in grama without explicit loop management and with sensible defaults for the analysis parameters. This provides a "quick and dirty" tool to inspect a model's behavior. A grama approach to parameter sweeps is shown below.

.. code-block:: python

    ## Grama approach
    # Gather model data
    md_gr = (
        gr.Model()
        >> gr.cp_vec_function(
            fun=lambda df: gr.df_make(f=df.x**2 * df.y),
            var=["x", "y"],
            out=["f"],
        )
        >> gr.cp_bounds(
            x=(-1, +1),
            y=(-1, +1),
        )
    )
    # Generate data
    df_gr = gr.eval_sinews(
        md_gr,
        df_det="swp",
        n_sweeps=3,
    )

Once a model is implemented in grama, performing a parameter sweep is trivial, requiring just two lines of code and zero initial choices for analysis parameters. The practical outcome of this software design is that users will tend to *self-initiate* parameter sweeps: While students will rarely choose to write the extensive boilerplate code necessary for a parameter sweep (unless required to do so), students writing code in grama will tend to self-initiate sound analysis practices.

For example, the following code is unmodified from a student report [#]_. The original author implemented an ordinary differential equation model to simulate the track time ``"finish_time"`` of an electric formula car, and sought to study the impact of variables such as the gear ratio ``"GR"`` on ``"finish_time"``. While the assignment did not require a parameter sweep, the student chose to carry out their own study. The code below is a self-initiated parameter sweep of the track time model.

.. [#] Included with permission of the author, on condition of anonymity.

.. code-block:: python

		## Unedited student code
		md_car = (
		    gr.Model("Accel Model")
		    >> gr.cp_function(
		        fun = calculate_finish_time,
		        var = ["GR", "dt_mass", "I_net" ],
		        out = ["finish_time"],
		    )

		    >> gr.cp_bounds(
		        GR=(+1,+4),
		        dt_mass=(+5,+15),
		        I_net=(+.2,+.3),
		    )
		)

		gr.plot_auto(
		    gr.eval_sinews(
		        md_car,
		        df_det="swp",
		        #skip=True,
		        n_density=20,
		        n_sweeps=5,
		        seed=101,
		    )
		)


.. figure:: student-sweep-focus.png
   :scale: 40%
   :figclass: bht

   Input sweep generated from the student code above. The image has been cropped for space, and the results are generated with an older version of grama. The jagged response at higher values of the input are evidence of solver instabilities. :label:`example-sweep`

The parameter sweep shown in Figure :ref:`example-sweep` gives an overall impression of the effect of input ``"GR"`` on the output ``"finish_time"``---this particular input tends to dominate the results. However, variable results at higher values of ``"GR"`` provide evidence of numerical instability in the ODE solver underlying the model. Without this sort of model evaluation, the student author would not have discovered the limitations of the model.

Exploratory Model Analysis
--------------------------


.. figure:: hull-schematic-stable.png
   :scale: 40%
   :figclass: bht

   Schematic boat hull rotated to :math:`22.5^{\circ}`. The forces due to gravity and buoyancy act at the center of mass (COM) and center of buoyancy (COB), respectively. Note that this hull is stable, as the couple will rotate the boat to upright. :label:`boat-stable`

.. figure:: hull-schematic-unstable.png
   :scale: 40%
   :figclass: bht

   Schematic boat hull rotated to :math:`22.5^{\circ}`. Gravity and buoyancy are annotated as in Figure :ref:`boat-stable`. Note that this hull is unstable, as the couple will rotate the boat away from upright. :label:`boat-unstable`
