:author: W\. Scott Shambaugh
:email: wsshambaugh@gmail.com
:bibliography: refs


-----------------------------------------------------------------------------------
Monaco: A Monte Carlo Framework for Performing Uncertainty and Sensitivity Analyses
-----------------------------------------------------------------------------------

.. class:: abstract

   This paper introduces *monaco*, a Python library for conducting Monte Carlo simulations of computational models, and performing uncertainty analysis (UA) and sensitivity analysis (SA) on the results. UA and SA are critical to effective and responsible use of models in science, engineering, and public policy, however their use is uncommon and hampered by lack of tools and need for advanced statistical knowledge. By providing a simple, general, and rigorous-by-default framework that wraps around existing models, *monaco* makes UA and SA easy and accessible to practitioners with a basic knowledge of statistics.


.. class:: keywords

   Monte Carlo, Modeling, Uncertainty Quantification, Uncertainty Analysis, Sensitivity Analysis, Ensemble Prediction, VARS, D-VARS


Introduction
============

Computational models form the backbone of decision-making processes in science, engineering, and public policy. However, our increased reliance on these models stands in contrast to the difficulty in understanding them as we add increasing complexity to try and capture ever more of the fine details of real-world interactions. Practitioners will often take the results of their large, complex model as a point estimate, with no knowledge of how uncertain those results are. Multiple-scenario modeling (e.g. looking at a worst-case, most-likely, and best-case scenario) is an improvement, but what is needed is a complete global exploration of the input space. That gives insight into the overall distribution of results (UA) as well as the relative influence of the different input factors on the ouput variance (SA). This complete understanding is critical for effective and responsible use of models in any decision-making process.

Despite the importance of UA and SA in modeling, a recent literature review of the most highly cited papers across a wide range of disciplines showed that they are uncommon in most fields. And even when performed, best practices are usually lacking - amongst papers which specifically claimed to perform sensitivity analysis, only 21% performed global (as opposed to local or zero) UA, and 41% performed global SA. :cite:`saltelli2019so` 

Typically UA and SA are done using Monte Carlo simulations, for reasons explored in the following section. There are Monte Carlo frameworks available, however existing options are domain-specific, focused on narrow sub-problems (i.e. integration), tailored towards training neural nets, or require a deep statistical background to use. See the introduction of :cite:`OLIVIER2020101204` for an overview of the currently available Python tools for performing UA. For the domain expert who is not a statistician and wants to perform UA and SA on their existing models, there are no good options. *monaco* was written to address this gap.

.. figure:: monaco_logo.png
   :align: center
   :figclass: h
   :scale: 20 %

   The monaco project logo. :label:`monacologo`

Why Monte Carlos?
=================

Mathematical Grounding
----------------------

Randomized Monte Carlo sampling offers a cure to the curse of dimensionality - consider an investigation of the output from :math:`k` input factors :math:`y = f(x_1, x_2, ..., x_k)` where each factor is uniformly sampled between 0 and 1, :math:`x_i \in U[0, 1]`. The input space is then a :math:`k`-dimensional hypercube with volume 1. If each input is varied one at a time (OAT), then points which are sampled lie on the principle axes passing through the point :math:`x_{i\in[1, ..., k]} = 0.5`. The volume of the sample space as :math:`n \to \infty` is the convex hull of these axes, which for the :math:`l_1` norm forms a :math:`k`-dimensional hyperoctahedron with volume :math:`\frac{1}{k!}` (or most optimistically, a :math:`l_2` norm hypersphere with volume :math:`\frac{\pi^{k/2}}{2^k \Gamma(k/2 + 1)}`), both of which decrease super-exponentially as :math:`k` increases. Unless the model is known to be linear, this leaves the input space wholly unexplored. In contrast, the volume of the convex hull of random samples as is obtained with a Monte Carlo approach will converge to the entire input volume as :math:`n \to \infty` (with much better coverage within that volume as well) :cite:`dyer1992volumes`. See Fig. :ref:`figvolume`.

.. figure:: hypersphere_volume.png
   :align: center
   :figclass: h

   Volume fraction :math:`V` of a :math:`k`-dimensional hypercube enclosed by the convex hull of :math:`n \to \infty` of random samples versus OAT samples along the principle axes of the input space. :label:`figvolume`

To sample :math:`k` input factors where each factor :math:`x_{i \in [1, ..., k]}` is drawn from a unique probability distribution, it is a common practice to uniformly draw a random percentile between 0 and 1, :math:`p_i \in U[0, 1]`, and transform this to the target distribution through that distribution's inverse cumulative density function (CDF), :math:`x_i = F_i^{-1}(p_i)`. Thus the preceding result on the unit hypercube is easily generalized to an arbitrarily distributed input space.


When to Use and When to Avoid Monte Carlo Analysis
--------------------------------------------------

A complete exploration of a model's input space is necessary to fully characterize its behavior, and responsible use of computational models requires understanding the total range of predictions they imply. Policy papers have identified UA and SA as a critical good modeling practice :cite:`azzini2020uncertainty` :cite:`us2009guidance`. The short answer to the question of when to conduct UA and SA is, "always".

There are some important considerations to keep in mind, however. With computational power making running large numbers of cases ever easier, a Monte Carlo analysis can result in highly statistically signficant conclusions. However these results are *conditional on the correctness of the underlying model and input distributions*. If the underlying model has not been throroughly validated, then any precise quantification of uncertainty and sensitivities will be washed out by the mismatch between the model and reality.

This validation can be difficult - outputs are a function of the combined inputs and model, and cannot be used to validate either the inputs or the model on their own. Generally validation requires significant domain expertise to ensure a mechanistic model has a solid theoretical foundation, and to diagnose errors in its implementation. 

This is not to say that UA and SA should not be conducted early in the model development process - obtaining the range of plausible output uncertainties is a critical step in input and model validation. Test data cannot be well compared against a single point estimate of a model's output, and it is necessary to have the full distribution of output values to compare test data against. Once a Monte Carlo analysis has generated these distributions, hypothesis testing or probablistic prediction measures like loss scores can be used to anchor the outputs against real-life test data.

Some benefits are more qualitative. Monte Carlo analysis is an excellent way to uncover edge cases in a model through unexpected combinations of inputs, especially in highly nonlinear models. This is the core concept behind "fuzzing" techniques in software testing. And for any practitioner, outlier cases often contain the most useful information. :cite:`saltelli2019so` identifies this as one reason why researchers might (reflexively or unscrupulously) avoid UA and SA - it forces addressing gaps in models and makes it more difficult to explain away inconvenient results.

While Monte Carlo analysis is not strictly necessary for linear models, it is often an easier and conceptually simpler way to compute the propogation of uncertainties and sensitivities through a model than using linear methods. And by making nonlinear models easier to examine, there should be less of a need to make linearity assumptions about a system in the first place.

One inherent pitfall of Monte Carlo approaches is that rare events may be undersampled. For example, NASA uses Monte Carlo simulations extensively during launch vehicle design to predict the rocket trajectory and performance. :cite:`hanson2010applying` However, they must prove robustness to anomalous or stressing scenarios which may occur only one or two times in a run of thousands of cases, which is not enough to draw conclusions from. In instances such as this, rare event scenarios should be investigated directly.

Note that *monaco*'s computational and storage overhead in creating easily-iterrogatable objects for each variable, value, and case makes it an inefficient choice for computationally simple applications with high :math:`n`, such as Monte Carlo integration. It is best suited for models with moderate to high computational cost. 


Why Use Basic Monte Carlo over Bayesian Methods?
------------------------------------------------

*monaco* purposefully eschews the greater computational efficiency and built-in input parameter estimation of Bayesian methods such as Markov Chain Monte Carlo in favor of basic sampling. For its target audience of scientists, engineers, and policy analysts with mechanistic models, this is useful for many reasons:

* `monaco` favors conceptual simplicity for greater accessibility. The goal is to be usable by someone at the level of knowing what a Uniform and Normal distribution are, and not require an in-depth data science background in inference. For one example of a barrier to entry that would not be covered in an undergraduate education, Bayesian methods generate inherently correlated sample points, which invalidates the independence assumption of many basic statistical methods one might want to apply to outputs. 
* Many modeling domains are not data-rich, especially in contrast to the dimensionality :math:`k` of the model. This precludes using standard Bayesian approaches to obtain results with any useful level of confidence. Consider NASA's use of Monte Carlo in predicting a rocket's flight :cite:`hanson2010applying` - they might have high levels of confidence that the hundreds of parameters describing their rocket are properly bound, and that their model correctly implements the physics of flight, but before their first launch they won't have any flight data to anchor against at all. UA and SA are still critical tools in these situations.
* Bayesian approaches require knowing a-priori which output statistics need to be calculated, and will undersample regions of low importance. In practice, the author has found that much of the value of UA is exploratory - using it to probe regions of the model that were previously unexamined. This can't be done if the practitioner has pidgeonholed their purview to particular posteriors.
* Bayesian methods have poor repeatability of specific cases. If different posteriors need to be calculated, this will require a re-run that changes the sample points and erases previous cases of interest.


Workflow
--------

UA and SA of any model follows a common workflow. Probability distributions for the model inputs are defined, and randomly sampled values for a large number of cases are fed to the model. The outputs from each case are collected and the full set of inputs and outputs can be analyzed. Typically UA is performed by generating histograms, scatter plots, and summary statistics for the output variables, and SA is performed by looking at the effect of input on output variables through scatter plots and calculating sensitivity indices. These results can then be compared to real-world test data to validate the model or inform revisions to the model and input variables. See Fig. :ref:`figanalysisprocess`.

Note that *monaco* does not currently have tools for model or parameter validation, and closing that part of the workflow loop is left up to the user.

.. figure:: analysis_process.png
   :align: center
   :figclass: h

   Monte Carlo workflow for understanding the full behavior of a computational model, inspired by :cite:`saltelli2019so`. :label:`figanalysisprocess`


*monaco* Structure
==================

Overall Structure
-----------------

Broadly, each input factor and model output is a *variable* that can be thought of as lists (rows) containing the full range of randomized *values*. *Cases* are slices (columns) that take the *i*'th input and output value for each variable, and represent a single run of the model. Each case is run on its own in parallel, and the output values are collected into output variables. Fig. :ref:`figarchitecture` shows a visual representation of this.

.. figure:: val_var_case_architecture.png
   :align: center
   :figclass: h

   Structure of a monaco simulation, showing the relationship between the major objects and functions. This maps onto the central block in Fig. :ref:`figanalysisprocess`. :label:`figarchitecture`


Simulation Setup
----------------
The base of a *monaco* simulation is the `Sim` object. This object is formed by passing it a name, the number of random cases `ncases`, and a dict `fcns` of the handles for three user-defined functions defined in the next section. A random seed that then seeds the entire simulation can also be passed in here, and is highly recommended for repeatability of results.

Input variables then need to be defined. *monaco* takes in the handle to any of `scipy.stat`'s continuous or discrete probability distributions, as well as the required arguments for that probability distribution :cite:`virtanen2020scipy`. If nonnumeric inputs are desired, the method can also take in a `nummap` dictionary which maps the randomly drawn integers to values of other types.

At this point the sim can be run. The randomized drawing of input values, creation of cases, running of those cases, and extraction of output values are automatically executed. 


User-Defined Functions
----------------------

The user needs to define three functions to wrap *monaco*'s Monte Carlo framework around their existing computational model. First is a `run` function which either calls or directly implements their model. Second is a `preprocess` function which takes in a `Case` object, extracts the randomized inputs, and structures them with any other invariant data to pass to the `run` function. Third is a `postprocess` function which takes in a `Case` object as well as the results from the model, and extracts the desired output values. The Python call chain is as:

.. code-block:: python
    
    postprocess(case, *run(*preprocess(case)))

Or equivalently to expand the Python star notation into pseudocode:

.. code-block:: python
    
    siminput = (siminput1, siminput2, ...) 
                 = preprocess(case)
    simoutput = (simoutput1, simoutput2, ...)
                  = run(*siminput) 
                  = run(siminput1, siminput2, ...)
    _ = postprocess(case, *simoutput)
      = postprocess(case, simoutput1, simoutput2, ...)

These three functions must be passed to the simulation in a dict with keys `'preprocess'`, `'run'`, and `'postprocess'`. See the example code at the end of the paper for a simple worked example.


Examining Results
-----------------

After running, users should generally do all of the following UA and SA tasks to get a full picture of the behavior of their computational model.

* Plot the results (UA & SA). :code:`sim.plot()` is a useful method to automatically generate histograms and scatter plots for all scalar variables.

* Calculate statistics for input or output variables (UA).

* Calculate sensitivity indices to rank importance of the input variables on variance of the output variables (SA).

* Investigate specific cases with outlier or puzzling results.

* Save the results to file or pass them to other programs. 


Data Flow
---------

A summary of the process and data flow:

1) Instantiate a `Sim` object.
2) Add input variables to the sim with specified probability distributions.
3) Run the simulation. This executes the following:    

 a) Random percentiles are drawn `ndraws` times for each of the input variables.
 b) These percentiles are transformed into random values via the inverse CDF of the target probability distribution.
 c) If nonnumeric inputs are desired, the random numbers are converted to objects via a `nummap` dict.
 d) `Case` objects are created and populated with the input values for each case.
 e) Each case is run by structuring the inputs values with the `preprocess` function, passing them to the `run` function, and collecting the output values with the `postprocess` function.
 f) The output values are collected into output variables and saved back to the sim. If the values are nonnumeric, a `valmap` dict assigning numbers to each unique value is automatically generated.

4) Calculate statistics & sensitivities for input & output variables.
5) Plot variables, their statistics, and sensitivities.


Technical Features
==================

Sampling Methods
----------------

Random sampling of the percentiles for each variable can be done using scipy's pseudo-random number generator (PRNG), or with any of the low-discrepancy methods in `scip.stats.qmc` Quasi-Monte Carlo module. In general, the `'sobol_random'` method that generates Sobol sequences with Owen scrambling :cite:`sobol1967distribution` :cite:`owen2020dropping` is recommended in nearly all cases as a well-performing quasi-random sequence with the best known convergence, balanced integration properties as long as the number of cases is a power of 2, and a fairly flat frequency spectra :cite:`perrier2018sequences`. This is set as default. In cases where computing sample points takes a  prohibitively long amount of time, users may fall back to `'random'` sampling directly from the PRNG at the cost of less even distribution of points in the input space. See Fig. :ref:`figsampling` for a visual comparison.


.. figure:: sampling.png
   :align: center
   :figclass: h

   256 uniform and normal samples along with the 2D frequency spectra for scrambled Sobol sampling (top, default) and PRNG random sampling (bottom). :label:`figsampling`


Order Statistics, or, How Many Cases to Run?
--------------------------------------------

How many Monte Carlo cases should one run? One answer would be to choose :math:`n \geq 2^k` with a sampling method that implements a (t,m,s) digital net (such as a Sobol or Halton sequence), which guarentees that there will be at least one sample point in every hyperoctant of the input space :cite:`joe2008constructing`. This should be considered a lower bound for SA, with the number of cases run being some integer multiple of :math:`2^k`.

Along a similar vein, :cite:`dyer1992volumes` suggests that with random sampling :math:`n \geq 2.136^k` is sufficient to ensure that the volume fraction :math:`V` approaches 1. The author hypothesizes that for a digital net, the :math:`n \geq \lambda^k` condition will be satisfied with :math:`\lambda \leq 2`, and so :math:`n \geq 2^k` will suffice for this condition to hold. However, these methods of choosing the number of cases may undersample for low :math:`k` and be infeasible for high :math:`k`.

A rigorous way of choosing the number of cases is to first choose a statistical interval (confidence interval for a percentile, or a tolerance interval to contain a percent of the population), and then use order statistics to calculate the minimum :math:`n` required to obtain that result at a desired confidence level. *monaco* implements routines for calculating these statistical intervals via an order statistics distribution-free approach that makes no assumptions about the normality or other shape characteristics of the output distribution. See Chaper 5 of :cite:`hahn1991statistical`.

A more qualitative UA method would simply to choose a reasonably high :math:`n` (say, :math:`n=2^{10}`), manually examine the results to ensure high-interest areas are not being undersampled, and rely on bootstrapping of the desired variable statistics to obtain the required confidence levels. 


Variable Statistics
-------------------

For any input or output variable, a statistic can be calculated for the ensemble of values. *monaco* builds in some common statistics (mean, percentile, etc), or alternatively the user can pass in a custom one. To obtain a confidence interval for this statistic, the results are resampled with replacement using the `scipy.stats.bootstrap` module. The number of bootstrap samples is determined using an order statistic approach as outlined in the previous section, and multiplying that number by a scaling factor (default 10x) for smoothness of results.


Sensitivity Indices
-------------------

Sensitivity indices give a measure of the relationship between the variance of a scalar output variable to the variance of each of the input variables. In other words, they measure which of the inputs has the largest effect on an output. It is crucial that sensitivity indices are global rather than local measures - global sensitivity has the stronger theoretical grounding and there is no reason to rely on local measures in scenarios such as automated computer experiments where data can be easily and aribitrarily sampled :cite:`saltelli2008global` :cite:`puy2022comprehensive`.

With computer-designed experiments, it is possible to contruct a specially constructed sample set to directly calculate global sensitivity indices such as the Total-Order Sobol index :cite:`sobol2001global`, or the IVARS100 index :cite:`razavi2016new`. However, this special construction requires either sacrificing the desirable UA properties of low-discrepancy sampling, or conducting an additional Monte Carlo analysis of the model with a different sample set. For this reason, *monaco* uses the D-VARS approach to calculating global sensitivity indices, which allows for using a set of given data :cite:`sheikholeslami2020fresh`. This is the first publically available implementation of the D-VARS algorithm.


Plotting
--------
*monaco* includes a plotting module that takes in input and output variables and quickly creates histograms, empirical CDFs, scatter plots, or 2D or 3D "spaghetti plots" depending on what is most appropriate for each variable. Variable statistics and their confidence intervals are automatically shown on plots when applicable.


Parallel Processing
-------------------

*monaco* uses *dask distributed* :cite:`rocklin2015dask` as a parallel processing backend, and supports preprocessing, running, and postprocessing cases in a parallel arrangement. Users familiar with *dask* can extend the parallelization of their simulation from their single machine to a distributed cluster.

For simple simulations such as the example code at the end of the paper, the overhead of setting up a *dask* server may outweigh the speedup from parallel computation, and in those cases *monaco* also supports running single-threaded in a single for-loop.


Example
=======
Presented here is a simple example showing a Monte Carlo simulation of rolling two 6-sided dice and looking at their sum.

The user starts with their `run` function which here directly implements their computational model. They must then create `preprocess` and `postprocess` functions to feed in the randomized input values and collect the outputs from that model.

.. code-block:: python
    
    # The 'run' function, which implements the
    # existing computational model (or wraps it)
    def example_run(die1, die2):
        sum = die1 + die2
        return (sum, )
    
    # The 'preprocess' function grabs the random
    # input values for each case and structures it 
    # with any other data in the format the 'run' 
    # function expects
    def example_preprocess(case):
        die1 = case.invals['die1'].val
        die2 = case.invals['die2'].val
        return (die1, die2)
    
    # The 'postprocess' function takes the output
    # from the 'run' function and saves off the
    # outputs for each case
    def example_postprocess(case, sum):
        case.addOutVal(name='Sum', val=sum)
        case.addOutVal(name='Roll Number',
                       val=case.ncase)
        return None

The *monaco* simulation is initialized, given input variables with specified probability distributions (here a random integer between 1 and 6), and run.

.. code-block:: python
    
    import monaco as mc
    from scipy.stats import randint
    
    # dict structure for the three input functions
    fcns = {'preprocess' : example_preprocess,
            'run'        : example_run,
            'postprocess': example_postprocess}
    
    # Initialize the simulation
    ndraws = 1024  # Arbitrary for this example
    seed = 123456  # Recommended for repeatability
    
    sim = mc.Sim(name='Dice Roll', ndraws=ndraws,
                 fcns=fcns, seed=seed)
    
    # Generate the input variables
    sim.addInVar(name='die1', dist=randint,
                 distkwargs={'low': 1, 'high': 6+1})
    sim.addInVar(name='die2', dist=randint,
                 distkwargs={'low': 1, 'high': 6+1})
    
    # Run the Simulation
    sim.runSim()

The results of the simulation can then be postprocessed and examined. Fig. :ref:`figexample` shows the plots this code generates.

.. code-block:: python
    
    # Calculate the mean and 5-95th percentile
    # statistics for the dice sum
    sim.outvars['Sum'].addVarStat('mean')
    sim.outvars['Sum'].addVarStat('percentile',
                                  {'p':[0.05, 0.95]})
    
    # Plots a histogram of the dice sum
    mc.plot(sim.outvars['Sum'])
    
    # Creates a scatter plot of the sum vs the roll
    # number, showing randomness
    mc.plot(sim.outvars['Sum'],
            sim.outvars['Roll Number'])
    
    # Calculate the sensitivity of the dice sum to 
    # each of the input variables
    sim.calcSensitivities('Sum')
    sim.outvars['Sum'].plotSensitivities()


.. figure:: example.png
   :align: center
   :figclass: h

   Output from the example code which calculates the sum of two random dice rolls. The top plot shows a histogram of the 2-dice sum with the mean and 5 - 95th percentiles marked, the middle plot shows the randomness over the set of rolls, and the bottom plot shows that each of the dice contributes 50% to the variance of the sum. :label:`figexample`


Conclusion
==========

This paper has introduced the ideas underlying Monte Carlo analysis and discussed when it is appropriate to use for conducting UA and SA. It has shown how *monaco* implements a rigorous, parallel Monte Carlo framework, and how to use it through a simple example. This library is geared towards scientists, engineers, and policy analysts that have a computational model in their domain of expertise, enough statistical knowledge to define a probability distribution, and a desire to ensure their model will make accurate predictions of reality. The author hopes this tool will help contribute to easier and more widespread use of UA and SA in improved descision-making.


Further Information
===================

*monaco* is available on PyPI, has API documentation at https://monaco.rtfd.io/, and is hosted on github at https://github.com/scottshambaugh/monaco/. Please see the "examples" directory in the github source for many more Monte Carlo implementation examples across a range of domains such as physics simulation, election prediction, financial modeling, pandemic spread, and integration.
