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

A Simple Example: Photon Counts
-------------------------------
Here we'll take a look at an extremely simple problem, and compare the frequentist and Bayesian approaches to solving it. There's necessarily a bit of mathematical formalism involved, but I won't go into too much depth or discuss too many of the subtleties.

Imagine that we point our telescope to the sky, and observe the light coming from a single star. For the time being, we'll assume that the star's true flux is constant with time, i.e. that is it has a fixed value :math:`F_{\rm true}` (we'll also ignore effects like sky noise and other sources of systematic error). We'll assume that we perform a series of :math:`N` measurements with our telescope, where the :math:`i^{\rm th}` measurement reports the observed photon flux :math:`F_i` and error :math:`e_i`. [#note_about_errors_]

The question is, given this set of measurements :math:`D = \{F_i,e_i\}`, what is our best estimate of the true flux :math:`F_{\rm true}`?

.. [#note_about_errors] We'll make the reasonable assumption that measurement errors are Gaussian. In a Frequentist perspective, :math:`e_i` is the standard deviation of the results of a single measurement event in the limit of repetitions of *that event*. In the Bayesian perspective, :math:`e_i` is the standard deviation of the (Gaussian) probability distribution describing our knowledge of that particular measurement given its observed value.

Here we'll use Python to generate some toy data to demonstrate the two approaches to the problem. Because the measurements are number counts, a Poisson distribution is a good approximation to the measurement process:

.. code-block:: python

    >>> # Draw 50 samples with mean 1000
    >>> F = scipy.stats.poisson(1000).rvs(50)
    >>> e = numpy.sqrt(F)  # Poisson Errors

The data is visualized in Figure :ref:`fig1`.

.. figure:: figure1.png

   Data for Example 1: simple photon counts. :label:`fig1`

These measurements each have a different error :math:`e_i` which is estimated from Poisson statistics using the standard square-root rule. In this toy example we already know the true flux :math:`F_{\rm true}`, but the question is this: **given our measurements and errors, what is our best point estimate of the true flux?**

Let's take a look at the frequentist and Bayesian approaches to solving this.


Frequentist Approach to Photon Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll start with the classical frequentist **maximum likelihood** approach. Given a single observation :math:`D_i = (F_i, e_i)`, we can compute the probability distribution of the measurement given the true flux :math:`F_{\rm true}` given our assumption of Gaussian errors:

.. math::

    P(D_i|F_{\rm true}) = \frac{1}{\sqrt{2\pi e_i^2}} \exp{\left[\frac{-(F_i - F_{\rm true})^2}{2 e_i^2}\right]}

This should be read "the probability of :math:`D_i` given :math:`F_{\rm true}` equals ...". You should recognize this as a normal distribution with mean :math:`F_{\rm true}` and standard deviation :math:`e_i`. We construct the **likelihood function** by computing the product of the probabilities for each data point:

.. math::

    \mathcal{L}(D|F_{\rm true}) = \prod_{i=1}^N P(D_i|F_{\rm true})

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


Bayesian Approach to Photon Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bayesian approach, as you might expect, begins and ends with probabilities.  It recognizes that what we fundamentally want to compute is our knowledge of the parameters in question, i.e. in this case,

.. math::

    P(F_{\rm true}|D)

Note that this formulation of the problem is *fundamentally contrary* to the frequentist philosophy, which says that *probabilities have no meaning for model parameters* like :math:`F_{\rm true}`. Nevertheless, within the Bayesian philosophy this is perfectly acceptable. 

To compute this result, Bayesians next apply Bayes' Theorem, a fundamental law of probability:

.. math::

    P(F_{\rm true}|D) = \frac{P(D|F_{\rm true})~P(F_{\rm true})}{P(D)}

Though Bayes' theorem is where Bayesians get their name, it is not this law itself that is controversial, but the Bayesian *interpretation of probability* implied by the term :math:`P(F_{\rm true}|D)`.

Let's take a look at each of the terms in this expression:

- :math:`P(F_{\rm true}|D)`: The **posterior**, or the probability of the model parameters given the data: this is the result we want to compute.
- :math:`P(D|F_{\rm true})`: The **likelihood**, which is proportional to the :math:`\mathcal{L}(D|F_{\rm true})` in the frequentist approach, above.
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
In the simple example above, the frequentist and Bayesian approaches give basically the same result. While it is easy to show that the two approaches are often equivalent for simple problems, it is also true that they can diverge greatly for more complicated problems. In practice, this divergence most often makes itself most clear in two different ways:

1. The handling of nuisance parameters
2. The subtle (and often overlooked) difference between frequentist confidence intervals and Bayesian credible intervals

We'll discuss these two situations in more detail in the following sections.

Nusiance Parameters: Bayes' Billiards Game
------------------------------------------
We'll start by discussing the first type of situation: nuisance parameters. A nuisance parameter is any quantity whose value is not relevant to the goal of an analysis, but is nevertheless required to determine some quantity of interest. For example, we might have a situation similar to example #1 above, but in which the errors :math:`e_i` are unknown. One potential approach s to treat these errors as nuisance parameters: that is, parameters which vary within the analysis, but have their effects somehow removed in the final results.

I'll start with an example of nuisance parameters that, in one form or another, dates all the way back to the posthumous 1763 paper written by Thomas Bayes himself [Bayes1763]_. The particular version of this problem used here is borrowed from [Eddy2004]_. The setting is a gambling game in which Alice and Bob bet on the outcome of a process they can't directly observe:

Alice and Bob enter a room. Behind a curtain there is a billiard table, which they cannot see, but their friend Carol can. Carol rolls a ball down the table, and marks where it lands. Once this mark is in place, Carol begins rolling new balls down the table. If the ball lands to the left of the mark, Alice gets a point; if it lands to the right of the mark, Bob gets a point.  We can assume for the sake of example that Carol's rolls are unbiased: that is, the balls have an equal chance of ending up anywhere on the table.  The first person to reach **six points** wins the game.

Here the location of the mark (determined by the first roll) can be considered a nuisance parameter: it is unknown, and not of immediate interest, but it clearly must be accounted for when predicting the outcome of subsequent rolls. If the first roll settles far to the right, then subsequent rolls will favor Alice. If it settles far to the left, Bob will be favored instead.

Given this setup, here is the question we ask of ourselves: *In a particular game, after eight rolls, Alice has five points and Bob has three points. What is the probability that Bob will go on to win the game?*

Intuitively, you probably realize that because Alice received five of the eight points, the marker placement likely favors her. And given this, it's more likely that the next roll will go her way as well. And she has three opportunities to get a favorable roll before Bob can win; she seems to have clinched it.  But, **quantitatively**, what is the probability that Bob will squeak-out a win?


A Naïve Frequentist Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Someone following a classical frequentist approach might reason as follows:

To determine the result, we need an intermediate estimate of where the marker sits. We'll quantify this marker placement as a probability :math:`p` that any given roll lands in Alice's favor.  Because five balls out of eight fell on Alice's side of the marker, we can quickly show that the maximum likelihood estimate of :math:`p` is given by:

.. math::

    \hat{p} = 5/8,

a result follows in a straightforward manner from the binomial likelihood. Assuming this maximum likelihood probability, we can compute the probability that Bob will win, which is given by:

.. math::

    P(B) = (1 - \hat{p})^3

That is, he needs to win three rolls in a row. Thus, we find that the probability of Bob winning is 0.053. In other words, we expect that the odds against Bob winning are 18 to 1.


A Bayesian Approach
~~~~~~~~~~~~~~~~~~~
A Bayesian approach to this problem involves treating the unknown :math:`p` as a nuisance parameter, and integrating over it so that, in some sense, our result is agnostic to the unknown value of :math:`p`.

We can also approach this problem from a Bayesian standpoint. This is slightly more involved, and requires us to first define some notation.

We'll consider the following random variables:

- :math:`B` = Bob Wins
- :math:`D` = observed data, i.e. :math:`D = (n_A, n_B) = (5, 3)`
- :math:`p` = unknown probability that a ball lands on Alice's side during the current game

We want to compute :math:`P(B|D)`; that is, the probability that Bob wins given our observation that Alice currently has five points to Bob's three.

The general Bayesian method of treating nuisance parameters is *marginalization*, or integrating the joint probability over the entire range of the nuisance parameter. In this case, that means that we will first calculate the joint distribution

.. math::

    P(B,p|D)

and then marginalize over :math:`p` using the following identity:

.. math::

    P(B|D) \equiv \int_{-\infty}^\infty P(B,p|D) {\mathrm d}p

This identity follows from the definition of conditional probability, and the law of total probability: that is, it is a fundamental consequence of probability axioms and will always be true. Even a frequentist would recognize this; they would simply disagree with our interpretation of :math:`P(p)` as being a measure of uncertainty of our own knowledge.

To compute this result, we will manipulate the above expression for :math:`P(B|D)` until we can express it in terms of other quantities that we can compute.

We'll start by applying the following definition of conditional probability to expand the term :math:`P(B,p|D)`:

.. math::

    P(B|D) = \int P(B|p, D) P(p|D) dp

Next we use Bayes' rule to rewrite :math:`P(p|D)`:

.. math::

    P(B|D) = \int P(B|p, D) \frac{P(D|p)P(p)}{P(D)} dp

Finally, using the same probability identity we started with, we can expand :math:`P(D)` in the denominator to find:

.. math::

    P(B|D) = \frac{\int P(B|p,D) P(D|p) P(p) dp}{\int P(D|p)P(p) dp}

Now the desired probability is expressed in terms of three quantities that we can compute. Let's look at each of these in turn:

- :math:`P(B|p,D)`: This term is exactly the frequentist likelihood we used above. In words: given a marker placement :math:`p` and the fact that Alice has won 5 times and Bob 3 times, what is the probability that Bob will go on to six wins?  Bob needs three wins in a row, i.e. :math:`P(B|p,D) = (1 - p) ^ 3`.
- :math:`P(D|p)`: this is another easy-to-compute term. In words: given a probability :math:`p`, what is the likelihood of exactly 5 positive outcomes out of eight trials? The answer comes from the well-known Binomial distribution: in this case :math:`P(D|p) \propto p^5 (1-p)^3`
- :math:`P(p)`: this is our prior on the probability :math:`p`. By the problem definition, we can assume that :math:`p` is evenly drawn between 0 and 1.  That is, :math:`P(p) \propto 1`, and the integrals range from 0 to 1.

Putting this all together and simplifying gives

.. math::

    P(B|D) = \frac{\int_0^1 (1 - p)^6 p^5 dp}{\int_0^1 (1 - p)^3 p^5 dp}

where both integrals are evaluated from 0 to 1. These integrals are instances of the beta function, so we can quickly evaluate the result using scipy:

.. code-block:: python

    >>> from scipy.special import beta
    >>> bayes_prob = beta(6+1, 5+1) / beta(3+1, 5+1)

This gives :math:`P(B|D) = 0.091`, which is equivalent to odds of 10 to 1 against Bob winning.


Discussion
~~~~~~~~~~
The Bayesian approach gives odds of 10 to 1 against Bob, while the naive frequentist approach gives odds of 18 to 1 against Bob. So which one is correct?

For a simple problem like this, we can answer this question empirically by using a monte carlo simulation in which we simulate a large number of games and count the fraction of suitable games which Bob goes on to win. This can be coded in a couple dozen lines of Python (see [VanderPlas2014]_, part I). The result of the simulation confirms our Bayesian odds: 10 to 1 against Bob winning.

This should not be construed to imply that frequentism is wrong, however. Its incorrect result is more a matter of the approach being "naïve" than it being "frequentist". There certainly exist frequentist methods for handling this sort of nuisance parameter – for example, it is theoretically possible to apply a transformation and conditioning of the data to isolate the dependence on :math:`p` – but I've not been able to find any approach to this particular problem that does not somehow take advantage of Bayesian-like marginalization over :math:`p`.

Another potential point of contention is that the question itself is posed in a way that is perhaps unfair to the classical, frequentist approach. A frequentist might instead hope to give the answer in terms of null tests or confidence intervals: that is, they might devise a procedure to construct limits which would provably bound the correct answer in :math:`100\times(1 - \alpha)` percent of similar trials, for some value of :math:`\alpha` – say, 0.0. This might be classically accurate, but it doesn't quite answer the question at hand. We'll further discuss the meaning of such confidence intervals below.

There is one clear common point of these two potential frequentist responses: both require some degree of effort and/or special expertise; perhaps a suitable frequentist approach would be immediately obvious to someone with a PhD in statistics, but is most definitely *not* obvious to a statistical lay-person simply trying to answer the question at hand. In this sense, I think Bayesianism provides a better approach for this sort of problem: by simple algebraic manipulation of a few well-known axioms of probability within a Bayesian framework, we can straightforwardly arrive at the correct answer without need for other special expertise.


Confidence vs. Credibility: Jaynes' Truncated Exponential
---------------------------------------------------------
A second consequence of the philosophical difference between frequentism and Bayesianism is the difference between frequentist confidence intervals (CI) and Bayesian credible regions (CR), which are, respectively, the standard frequentist and Bayesian methods for constructing uncertainty bounds on unknown parameters. Despite their apparent similarity, the two approaches are fundamentally different. Both are statements of probability, but the probability refers to different aspects of the bound. For example, when constructing a standard 95% (:math:`2\sigma`) bound about a parameter :math:`\theta`:

- A Bayesian would say: "Given our observed data, there is a 95% probability that the true value of :math:`\theta` lies within the credible region".
- A frequentist would say: "There is a 95% probability that when I compute a confidence interval from data of this sort, the true value of :math:`\theta` will lie in this confidence interval.

Notice the subtle difference: the Bayesian makes a statement of probability about the *parameter value* given a *fixed credible region*. The frequentist makes a statement of probability about the *confidence interval itself* given a *fixed parameter value*. This distinction follows straightforwardly from the definition of probability discussed above: the Bayesian probability is a statement of degree of knowledge about a parameter; the frequentist probability is a statement of long-term limiting frequency of a particular recipe for constructing the interval.

As an example of how this affects the interpretation of results, consider that for a 95% CI, it is incorrect to say that there is a 95% chance of the parameter lying within the interval; this is a Bayesian rather than a frequentist interpretation. Under most circumstances (such as the first example above), the Bayesian CR and frequentist CI will more-or-less overlap, so this frequentist oversight can perhaps be forgiven. But, as we'll see below, this overlap does not always hold, especially in the case of non-Gaussian distributions constrained by few data points. As a result, this common misinterpretation of the frequentist CI can lead to dangerously erroneous interpretations.

To demonstrate a situation in which the frequentist confidence interval and the Bayesian credibility region do not overlap, I'm going to turn to an example given by E.T. Jaynes, a 20th century physicist who wrote extensively on statistical inference in Physics. In his words:

    A device will operate without failure for a time :math:`\theta` because of a protective chemical inhibitor injected into it; but at time :math:`\theta` the supply of the chemical is exhausted, and failures then commence, following the exponential failure law. It is not feasible to observe the depletion of this inhibitor directly; one can observe only the resulting failures. From data on actual failure times, estimate the time :math:`\theta` of guaranteed safe operation... [Jaynes1976]_

Essentially, we have data :math:`D` drawn from the following model:

.. math::

    p(x|\theta) = \left\{
    \begin{array}{lll}
    \exp(\theta - x) &,& x > \theta\\
    0                &,& x < \theta
    \end{array}
    \right\}

where :math:`p(x|\theta)` gives the probability of failure at time :math:`x`, given an inhibitor which lasts for a time :math:`\theta`. We observe some failure times :math:`D = \{10, 12, 15\}` and ask for 95% uncertainty bounds on the value of :math:`\theta`.

First, let's think about what common-sense would tell us. Given the model, an event can only happen after a time :math:`\theta`. Turning this around tells us that the upper-bound for :math:`\theta` must be :math:`\min_i\{x_i\}`. So, for our particular example, we would immediately write :math:`\theta \le 10`. Let's explore how a frequentist and a Bayesian approach compare to this observation.

Truncated Exponential: A Frequentist Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the frequentist paradigm, we'd like to compute a confidence interval on the value of :math:`\theta`. We can start by observing that the population mean is given by

.. math::

    E(x) = \int_0^\infty xp(x)dx = \theta + 1

So, using the sample mean as the point estimate of :math:`E(x)`, we have an unbiased estimator for :math:`\theta` given by

.. math::

    \hat{\theta} = \frac{1}{N} \sum_{i=1}^N x_i - 1

In the large-:math:`N` limit, the central limit theorem tells us that the sampling distribution is normal with standard deviation given by the standard error of the mean: :math:`\sigma_{\hat{\theta}}^2 = 1/N`, and we can write the 95% (i.e. :math:`2\sigma`) confidence interval as

.. math::

    CI_{\rm large~N} = \left(\hat{\theta} - 2 N^{-1/2},~\hat{\theta} + 2 N^{-1/2}\right)

For our particular observed data, this gives a confidence interval around our unbiased estimator of :math:`CI(\theta) = (10.2, 12.5)`, entirely above our common-sense bound of :math:`\theta < 10`! We might hope that this discrepancy is due to our use of the large-:math:`N` approximation with a paltry :math:`N=3` samples. A more careful treatment of the problem (See [Jaynes1976]_ or [VanderPlas2014]_, III) gives the exact confidence interval :math:`CI(\theta) = (10.2, 12.2)`: the 95% confidence interval entirely excludes the sensible bound :math:`\theta < 10`!


Truncated Exponential: A Bayesian Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's see if the Bayesian approach can do better. For the Bayesian solution, we start by writing Bayes' rule:

.. math::

    p(\theta|D) = \frac{p(D|\theta)p(\theta)}{P(D)}

Using a constant prior :math:`p(\theta)`, and with the likelihood

.. math::

    p(D|\theta) = \prod_{i=1}^N p(x|\theta)

we find

.. math::

    p(\theta|D) \propto \left\{
    \begin{array}{lll}
    N\exp\left[N(\theta - \min(D))\right] &,& \theta < \min(D)\\
    0                &,& \theta > \min(D)
    \end{array}
    \right\}

where :math:`\min(D)` is the smallest value in the data :math:`D`, which enters because of the truncation of :math:`p(x|\theta)`. Because :math:`p(\theta|D)` increases exponentially up to the cutoff, the shortest 95% credibility interval :math:`(\theta_1, \theta_2)` will be given by

.. math::

    \theta_2 = \min(D)

and :math:`\theta_1` given by the solution to the equation

.. math::

    \int_{\theta_1}^{\theta_2} N\exp[N(\theta - \theta_2)]d\theta = f

which can be simplified to

.. math::

    \theta_1 = \theta_2 + \frac{\log(1 - f)}{N}

For our particular data, this results in a Bayesian credible region

.. math::

    CR(\theta) = (9.0, 10.0)

which agrees with our common-sense bound.

Discussion
~~~~~~~~~~

*(TODO: tone this down a bit; mention biased vs unbiased estimators)*

Why do the frequentist CI and Bayesian CR give such divergent results? The reason goes back to the definitions of the CI and CR, and to the fact that *the two approaches are answering different questions*. The Bayesian CR answers a question about the value of :math:`\theta` itself, while the frequentist CI answers a question about the validity of the procedure used to construct the CI.

Recall the statements about confidence intervals and credible regions that I made above. From the Bayesians:

    "Given our observed data, there is a 95% probability that the true value of :math:`\theta` falls within the credible region" - Bayesians

And from the frequentists:

    "There is a 95% probability that when I compute a confidence interval from data of this sort, the true value of :math:`\theta` will fall within it." - Frequentists

Now think about what this means. Suppose you've measured three failure times of your device, and you want to estimate :math:`\theta`. I would assert that "data of this sort" is not your primary concern: you should be concerned with what you can learn from **those particular three observations**, not the entire hypothetical space of observations like them. As we saw above, if you follow the frequentists in considering "data of this sort", you are in danger at arriving at an answer that tells you nothing meaningful about the particular data you have measured.

Suppose you attempt to change the question and ask what the frequentist confidence interval can tell you *given the particular data that you've observed*. Here's what it has to say:

    "*Given this observed data*, the true value of :math:`\theta` is either in our confidence interval or it isn't" - Frequentists

That's all the credibility region means – and all it can mean! – for **this particular data** that you have observed. Really. I'm not making this up. You might notice that this is simply a tautology, and can be put more succinctly:

    "*Given this observed data*, I can put no constraint on the value of :math:`\theta`" - Frequentists

If you're interested in what your particular, observed data are telling you, frequentism is useless.


Bayesianism in Practice: MCMC in Python
---------------------------------------
One of the weaknesses of Bayesianism is that it tends to be extremely computationally intensive: while simple problems like the billiard game above lend themselves to relatively easy analytical integration, real-life Bayesian computations require numerical integration of high-dimensional parameter spaces. A turning-point in Bayesian computation was the development of sampling methods such as Markov Chain Monte Carlo (MCMC), a class of algorithms which can efficiently draw samples from even high-dimensional posterior distributions.

There are several excellent MCMC packages available in Python. I'll discuss three of them here: emcee [#emcee]_ [ForemanMackey2013]_ , PyMC [#pymc]_ [Patil2010]_ , and PyStan [#pystan]_. Here we'll propose a straightforward problem with some nontrivial elements, and compare how it is implemented in these three packages.

.. [#emcee] emcee: the MCMC Hammer http://dan.iel.fm/emcee

.. [#pymc] PyMC: Bayesian Inference in Python http://pymc-devs.github.io/pymc/

.. [#pystan] The Python Interface to Stan https://pystan.readthedocs.org/en/latest/

A Bayesian Linear Model
~~~~~~~~~~~~~~~~~~~~~~~
For our test problem, we'll consider a three-parameter linear model which fits a straight-line to data. The parameters will be the the y-intercept :math:`\alpha`, the slope :math:`\beta`, and the normal scatter :math:`\sigma` about the line; the scatter in this case will be treated as a nuisance parameter.

For data :math:`D = \{x_i, y_i\}`, the model is

.. math::

    \hat{y}(x_i|\alpha,\beta) = \alpha + \beta x_i,

and the likelihood is

.. math::

    P(D|\alpha,\beta,\sigma) = (2\pi\sigma^2)^{-N/2} \prod_{i=1}^N \exp\left[\frac{-[y_i - \hat{y}(x_i|\alpha, \beta)]^2}{2\sigma^2}\right].

The posterior is proportional to the product of the likelihood and the prior; in this case we must be aware that a flat prior is not uninformative. Through symmetry arguments, it can be shown that an uninformative prior for this problem is given by

.. math::

    P(\alpha,\beta,\sigma) \propto \frac{1}{\sigma}(1 + \beta^2)^{-3/2}.

With the likelihood and prior determined, we can mode on to sampling the posterior using the three packages.


Solution with emcee
~~~~~~~~~~~~~~~~~~~
emcee version 2.0

*TODO: discuss*

For ``emcee``, all that is required is to define a Python function representing the logarithm of the posterior. For clarity, we'll factor this definition into two functions, the log-prior and the log-likelihood:

.. code-block:: python

    import numpy as np
    import emcee

    def log_prior(theta):
        alpha, beta, sigma = theta
        if sigma < 0:
            return -np.inf  # log(0)
        else:
            return (-1.5 * np.log(1 + beta**2)
                    - np.log(sigma))

    def log_likelihood(theta, x, y):
       alpha, beta, sigma = theta
       y_model = alpha + beta * x
       return -0.5 * np.sum(np.log(2 * np.pi * sigma**2) +
                            (y - y_model)**2 / sigma**2)

   def log_posterior(theta, x, y):
       return log_prior(theta) + log_likelihood(theta,x,y)

Next we set up the computation. ``emcee`` combines multiple "walkers", each of which is its own markov chain. We'll also specify a burn-in period, to allow the chains to stabilize before

.. code-block:: python

   ndim = 3  # number of parameters in the model
   nwalkers = 50  # number of MCMC walkers
   nburn = 1000  # "burn-in" to stabilize chains
   nsteps = 2000  # number of MCMC steps to take

   # set theta near the maximum likelihood, with 
   np.random.seed(0)
   starting_guesses = np.random.random((nwalkers, ndim))


Now we call the sampler and extract the trace:

.. code-block:: python

   sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                   log_posterior,
                                   args=[xdata, ydata])
   sampler.run_mcmc(starting_guesses, nsteps)

   # sampler.chain is of shape (nwalkers, nsteps, ndim)
   # we'll throw-out the burn-in points and reshape:
   trace = sampler.chain[:, nburn:, :]
   trace = trace.reshape(-1, ndim).T


Solution with PyMC
~~~~~~~~~~~~~~~~~~
PyMC version 2.3

*TODO: discuss*

First we define all the variables

.. code-block:: python

    import pymc

    alpha = pymc.Uniform('alpha', -100, 100)

    @pymc.stochastic(observed=False)
    def beta(value=0):
		return -1.5 * np.log(1 + value ** 2)

		@pymc.stochastic(observed=False)
		def sigma(value=1):
   return -np.log(abs(value))

   # Define the form of the model and likelihood
   @pymc.deterministic
   def y_model(x=xdata, alpha=alpha, beta=beta):
   return alpha + beta * x

   y = pymc.Normal('y', mu=y_model, tau=1. / sigma ** 2,
                   observed=True, value=ydata)

   # package the full model in a dictionary
   model1 = dict(alpha=alpha, beta=beta, sigma=sigma,
                 y_model=y_model, y=y)

Next we run the chain and extract the trace:

.. code-block:: python

    S = pymc.MCMC(model1)
    S.sample(iter=100000, burn=50000)
    pymc_trace = [S.trace('alpha')[:],
                  S.trace('beta')[:],
                  S.trace('sigma')[:]]


Solution with PyStan
~~~~~~~~~~~~~~~~~~~~
PyStan version 2.2

*TODO: discuss*

.. code-block:: python

    import pystan

    fit_code = """
    data {
        int<lower=0> N; // number of points
        real x[N]; // x values
        real y[N]; // y values
    }

    parameters {
        real alpha_perp;
        real<lower=-pi()/2, upper=pi()/2> theta;
        real log_sigma;
    }

    transformed parameters {
        real alpha;
        real beta;
        real sigma;
        real ymodel[N];
   
        alpha <- alpha_perp / cos(theta);
        beta <- sin(theta);
        sigma <- exp(log_sigma);
        for (j in 1:N)
          ymodel[j] <- alpha + beta * x[j];
        }

    model {
        y ~ normal(ymodel, sigma);
    }
    """

    # perform the fit
    fit_data = {'N': len(xdata), 'x': xdata, 'y': ydata}
    fit = pystan.stan(model_code=fit_code, data=fit_data,
                      iter=25000, chains=4)

    # extract the traces
    traces = fit.extract()
    pystan_trace = [traces['alpha'],
                    traces['beta'],
                    traces['sigma']]

That's all there is to it!

Comparison
~~~~~~~~~~
*TODO: add figure comparing the three*

Conclusion
----------
*TODO: add summary*


References
----------
.. [Bayes1763] T. Bayes.
               *An essay towards solving a problem in the doctrine of chances*.
               Philosophical Transactions of the Royal Society of London
               53(0):370-418, 1763

.. [Eddy2004] S.R. Eddy. *What is Bayesian statistics?*.
              Nature Biotechnology 22:1177-1178, 2004

.. [ForemanMackey2013] D. Foreman-Mackey, D.W. Hogg, D. Lang, J.Goodman.
            *emcee: the MCMC Hammer*. PASP 125(925):306-312, 2014

.. [Jaynes1976] E.T. Jaynes. *Confidence Intervals vs Bayesian Intervals (1976)*
                Papers on Probability, Statistics and Statistical Physics
                Synthese Library 158:149-209, 1989

.. [Patil2010] A. Patil, D. Huard, C.J. Fonnesbeck.
               *PyMC: Bayesian Stochastic Modelling in Python* 
               Journal of Statistical Software, 35(4):1-81, 2010.

.. [VanderPlas2014] J.T VanderPlas. *Frequentism and Bayesianism*.
                    Four-part series (`I <http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/>`_, 
                    `II <http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/>`_,
                    `III <http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/>`_,
                    `IV <http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/>`_) on *Pythonic Perambulations*
                    http://jakevdp.github.io/, 2014.
                    
                    
