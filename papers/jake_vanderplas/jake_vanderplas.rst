:author: Jake VanderPlas
:email: jakevdp@cs.washington.edu
:institution: University of Washington

---------------------------------------------------
Frequentism and Bayesianism: A Python-driven Primer
---------------------------------------------------

.. class:: abstract

   This paper gives a brief introduction to the essential differences between the frequentist and Bayesian approaches to statistical inference, with examples implemented in Python

.. class:: keywords

   statistics, bayesian inference

Introduction
------------

One of the first things a scientist in a data-intensive field hears about statistics is that there is are two different approaches: frequentism and Bayesianism. Despite their importance, many scientific researchers never have opportunity to learn the distinctions between them and the different practical approaches that result.

This paper seeks to synthesize the philosophical and pragmatic aspects of this debate, so that scientists who use these approaches might be better prepared to understand the tools available to them. Along the way we'll explore the fundamental philosophical divergence of frequentism and Bayesianism, explore the practical aspects of how this divergence affects data analysis, and discuss the ways that these practices may affect scientific results.

The Definition of Probability
-----------------------------
Fundamentally, the disagreement between frequentists and Bayesians concerns the definition of probability.

For frequentists, probability only has meaning in terms of **a limiting case of repeated measurements**. That is, if I measure the photon flux :math:`F` from a given star (we'll assume for now that the star's flux does not vary with time), then measure it again, then again, and so on, each time I will get a slightly different answer due to the statistical error of my measuring device. In the limit of a large number of measurements, the *frequency* of any given value indicates the probability of measuring that value.  For frequentists **probabilities are fundamentally related to frequencies of events**. This means, for example, that in a strict frequentist view, it is meaningless to talk about the probability of the *true* flux of the star: the true flux is (by definition) a single fixed value, and to talk about a frequency distribution for a fixed value is nonsense.

For Bayesians, the concept of probability is extended to cover **degrees of certainty about statements**.  Say a Bayesian claims to measure the flux :math:`F` of a star with some probability :math:`P(F)`: that probability can certainly be estimated from frequencies in the limit of a large number of repeated experiments, but this is not fundamental. The probability is a statement of my knowledge of what the measurement reasult will be. For Bayesians, **probabilities are fundamentally related to our own knowledge about an event**. This means, for example, that in a Bayesian view, we can meaningfully talk about the probability that the *true* flux of a star lies in a given range.  That probability codifies our knowledge of the value based on prior information and/or available data.

The surprising thing is that this arguably subtle difference in philosophy can lead, in practice, to vastly different approaches to the statistical analysis of data.  Below I will give a few practical examples of the differences in approach, along with associated Python code to demonstrate the practical aspects of the resulting methods.

Example #1: Simple Photon Counts
--------------------------------
Here we'll take a look at an extremely simple problem, and compare the frequentist and Bayesian approaches to solving it. There's necessarily a bit of mathematical formalism involved, but I won't go into too much depth or discuss too many of the subtleties.

Imagine that we point our telescope to the sky, and observe the light coming from a single star. For the time being, we'll assume that the star's true flux is constant with time, i.e. that is it has a fixed value :math:`F_{\rm true}` (we'll also ignore effects like sky noise and other sources of systematic error). We'll assume that we perform a series of :math:`N` measurements with our telescope, where the :math:`i^{\rm th}` measurement reports the observed photon flux :math:`F_i` and error :math:`e_i`. [#note_about_errors_]

The question is, given this set of measurements :math:`D = \{F_i,e_i\}`, what is our best estimate of the true flux :math:`F_{\rm true}`?

.. [#note_about_errors] We'll make the reasonable assumption that measurement errors are Gaussian. In a Frequentist perspective, :math:`e_i` is the standard deviation of the results of a single measurement event in the limit of repetitions of *that event*. In the Bayesian perspective, :math:`e_i` is the standard deviation of the (Gaussian) probability distribution describing our knowledge of that particular measurement given its observed value.

Here we'll use Python to generate some toy data to demonstrate the two approaches to the problem. Because the measurements are number counts, a Poisson distribution is a good approximation to the measurement process:

.. code-block:: python

    # Draw 50 samples with mean 1000
    F = scipy.stats.poisson(1000).rvs(50)
    e = numpy.sqrt(F)  # Poisson Errors

The data is visualized in Figure :ref:`fig1`.

.. figure:: figure1.png

   Data for Example 1: simple photon counts. :label:`fig1`

These measurements each have a different error :math:`e_i` which is estimated from [Poisson statistics](http://en.wikipedia.org/wiki/Poisson_distribution) using the standard square-root rule. In this toy example we already know the
true flux :math:`F_{\rm true}`, but the question is this: **given our measurements and errors, what is our best point estimate of the true flux?**

Let's take a look at the frequentist and Bayesian approaches to solving this.


Frequentist Approach to Simple Photon Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll start with the classical frequentist **maximum likelihood** approach. Given a single observation :math:`D_i = (F_i, e_i)`, we can compute the probability distribution of the measurement given the true flux :math:`F_{\rm true}` given our assumption of Gaussian errors:

.. math::

    P(D_i~|~F_{\rm true}) = \frac{1}{\sqrt{2\pi e_i^2}} \exp{\left[\frac{-(F_i - F_{\rm true})^2}{2 e_i^2}\right]}

This should be read "the probability of :math:`D_i` given :math:`F_{\rm true}` equals ...". You should recognize this as a normal distribution with mean :math:`F_{\rm true}` and standard deviation :math:`e_i`. We construct the **likelihood function** by computing the product of the probabilities for each data point:

.. math::

    \mathcal{L}(D~|~F_{\rm true}) = \prod_{i=1}^N P(D_i~|~F_{\rm true})

Here :math:`D = \{D_i\}` represents the entire set of measurements. Because the value of the likelihood can become very small, it is often more convenient to instead compute the log-likelihood.  Combining the previous two equations and computing the log, we have

.. math::

    \log\mathcal{L} = -\frac{1}{2} \sum_{i=1}^N \left[ \log(2\pi  e_i^2) + \frac{(F_i - F_{\rm true})^2}{e_i^2} \right]

What we'd like to do is determine :math:`F_{\rm true}` such that the likelihood is maximized. For this simple problem, the maximization can be computed analytically (i.e. by setting :math:`d\log\mathcal{L}/dF_{\rm true} = 0`). This results in the following observed estimate of :math:`F_{\rm true}`:

.. math::

    F_{\rm est} = \frac{\sum w_i F_i}{\sum w_i};~~w_i = 1/e_i^2

Notice that in the special case of all errors :math:`e_i` being equal, this reduces to

.. math::

    F_{\rm est} = \frac{1}{N}\sum_{i=1}^N F_i

That is, in agreement with intuition, :math:`F_{\rm est}` is simply the mean of the observed data when errors are equal.

We can go further and ask what the error of our estimate is. In the frequentist approach, this can be accomplished by fitting a Gaussian approximation to the likelihood curve at maximum; in this simple case this can also be solved analytically. It can be shown that the standard deviation of this Gaussian approximation is:

.. math::

    \sigma_{\rm est} = \left(\sum_{i=1}^N w_i \right)^{-1/2}

These results are fairly simple calculations; for the above dataset the result is :math:`F_{\rm est} = 998 \pm 4` photons.


Bayesian Approach to Simple Photon Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bayesian approach, as you might expect, begins and ends with probabilities.  It recognizes that what we fundamentally want to compute is our knowledge of the parameters in question, i.e. in this case,

.. math::

    P(F_{\rm true}~|~D)

Note that this formulation of the problem is *fundamentally contrary* to the frequentist philosophy, which says that *probabilities have no meaning for model parameters* like :math:`F_{\rm true}`. Nevertheless, within the Bayesian philosophy this is perfectly acceptable. 

To compute this result, Bayesians next apply [Bayes' Theorem](http://en.wikipedia.org/wiki/Bayes\'_theorem), a fundamental law of probability:


.. math::

    P(F_{\rm true}~|~D) = \frac{P(D~|~F_{\rm true})~P(F_{\rm true})}{P(D)}

Though Bayes' theorem is where Bayesians get their name, it is not this law itself that is controversial, but the Bayesian *interpretation of probability* implied by the term :math:`P(F_{\rm true}~|~D)`.

Let's take a look at each of the terms in this expression:

- :math:`P(F_{\rm true}~|~D)`: The **posterior**, or the probability of the model parameters given the data: this is the result we want to compute.
- :math:`P(D~|~F_{\rm true})`: The **likelihood**, which is proportional to the :math:`\mathcal{L}(D~|~F_{\rm true})` in the frequentist approach, above.
- :math:`P(F_{\rm true})`: The **model prior**, which encodes what we knew about the model prior to the application of the data :math:`D`.
- :math:`P(D)`: The **data probability**, which in practice amounts to simply a normalization term.

If we set the prior :math:`P(F_{\rm true}) \propto 1` (a *flat prior*), we find

.. math::

    P(F_{\rm true}|D) \propto \mathcal{L}(D|F_{\rm true}).

That is, with a flat prior in :math:`F_{\rm true}`, the Bayesian posterior is maximized at precisely the same value as the frequentist result! So despite the philosophical differences, we see that (for this simple problem at least) the Bayesian and frequentist point estimates are equivalent.

You'll notice that we glossed over one important piece here: the prior.
You'll noticed that I glossed over something here: the prior, :math:`P(F_{\rm true})`. The prior allows inclusion of other information into the computation, which becomes very useful in cases where multiple measurement strategies are being combined to constrain a single model (as is the case in, e.g. cosmological parameter estimation). The necessity to specify a prior, however, is one of the more controversial pieces of Bayesian analysis.

A frequentist will point out that the prior is problematic when no true prior information is available. Though it might seem straightforward to use a **noninformative prior** like the flat prior mentioned above, there are some surprisingly subtleties involved. It turns out that in many situations, a truly noninformative prior does not exist! Frequentists point out that the subjective choice of a prior which necessarily biases your result has no place in statistical data analysis.

A Bayesian would counter that frequentism doesn't solve this problem, but simply skirts the question. Frequentism can often be viewed as simply a special case of the Bayesian approach for some (implicit) choice of the prior: a Bayesian would say that it's better to make this implicit choice explicit, even if the choice might include some subjectivity.

Discussion
~~~~~~~~~~
You might come away with the impression that the Bayesian method is unnecessarily complicated, and in this case it certainly is. Using an Affine Invariant Markov Chain Monte Carlo Ensemble sampler to characterize a one-dimensional normal distribution is a bit like using the Death Star to destroy a beach ball, but I did this here because it demonstrates an approach that can scale to complicated posteriors in many, many dimensions, and can provide nice results in more complicated situations where an analytic likelihood approach is not possible.

As a side note, you might also have noticed one little sleight of hand: at the end, we use a frequentist approach to characterize our posterior samples!  When we computed the sample mean and standard deviation above, we were employing a distinctly frequentist technique to characterize the posterior distribution. The pure Bayesian result for a problem like this would be to report the posterior distribution itself (i.e. its representative sample), and leave it at that. That is, in pure Bayesianism the answer to a question is not a single number with error bars; the answer is the posterior distribution over the model parameters!

Where The Results Diverge
-------------------------
- nuisance parameters
- confidence vs. credibility



Example #2: Bayes' Billiards Game
---------------------------------
Following http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/


Example #3: Jaynes' Truncated Exponential
-----------------------------------------
Following http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/



Bayesianism in Practice: MCMC in Python
---------------------------------------
Following http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/


Example #4: Linear Fit with Unknown Errors
------------------------------------------
Following http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/



References
----------

- Jaynes
- Eddy
- Wasserman
