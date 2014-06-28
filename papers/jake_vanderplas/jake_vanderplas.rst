:author: Jake VanderPlas
:email: jakevdp@cs.washington.edu
:institution: eScience Institute, University of Washington

---------------------------------------------------
Frequentism and Bayesianism: A Python-driven Primer
---------------------------------------------------

.. class:: abstract

   This paper gives a brief, semi-technical introduction to the essential differences between the frequentist and Bayesian approaches to statistical inference, with examples implemented in Python. The differences between the approaches stem from differing conceptions of probability. This difference of philosophy which leads to different approaches to certain problems, and different ways of asking and answering questions about unknown parameters. After discussing these and giving some examples, we briefly compare three leading Python packages for performing Bayesian inference in Python via Markov Chain Monte Carlo. [#blog]_

.. class:: keywords

   statistics, frequentism, Bayesian inference

.. [#blog] This paper draws from content originally published on the author's blog, *Pythonic Perambulations* [VanderPlas2014]_.

Introduction
------------

One of the first things a scientist in a data-intensive field hears about statistics is that there are two different approaches: frequentism and Bayesianism. Despite their importance, many researchers never have opportunity to learn the distinctions between them and the different practical approaches that result.

This paper seeks to synthesize the philosophical and pragmatic aspects of this debate, so that scientists who use these approaches might be better prepared to understand the tools available to them. Along the way we'll explore the fundamental philosophical divergence of frequentism and Bayesianism, explore the practical aspects of how this divergence affects data analysis, and discuss the ways that these practices may affect scientific results.


The Disagreement: The Definition of Probability
-----------------------------------------------
Fundamentally, the disagreement between frequentists and Bayesians concerns the definition of probability.

For frequentists, probability only has meaning in terms of **a limiting case of repeated measurements**. That is, if you measure the flux :math:`F` from a given star (we'll assume for now that the star's flux does not vary with time), then measure it again, then again, and so on, each time you will get a slightly different answer due to the statistical error of my measuring device. In the limit of many measurements, the *frequency* of any given value indicates the probability of measuring that value.  For frequentists, **probabilities are fundamentally related to frequencies of events**. This means, for example, that in a strict frequentist view, it is meaningless to talk about the probability of the *true* flux of the star: the true flux is (by definition) a single fixed value, and to talk about a frequency distribution for a fixed value (delta functions aside) is nonsense.

For Bayesians, the concept of probability is extended to cover **degrees of certainty about statements**. A Bayesian might claim to know the flux :math:`F` of a star with some probability :math:`P(F)`: that probability can certainly be estimated from frequencies in the limit of a large number of repeated experiments, but this is not fundamental. The probability is a statement of the researcher's knowledge of what the true flux is. For Bayesians, **probabilities are fundamentally related to their own knowledge about an event**. This means, for example, that in a Bayesian view, we can meaningfully talk about the probability that the *true* flux of a star lies in a given range.  That probability codifies our knowledge of the value based on prior information and available data.

The surprising thing is that this arguably subtle difference in philosophy can lead, in practice, to vastly different approaches to the statistical analysis of data.  Below I will give a few practical examples of the differences in approach, along with associated Python code to demonstrate the practical aspects of the resulting methods. Due to space limitations, we'll not go into much depth regarding the subtleties of the mathematical formalism involved. For this purpose, there are other excellent resources available; e.g. [Wasserman2004]_.


A Simple Example: Flux Measurements
-----------------------------------
Here we'll compare the frequentist and Bayesian approaches to the solution of an extremely simple problem. Imagine that we point a telescope to the sky, and observe the light coming from a single star. For simplicity, we'll assume that the star's true flux is constant with time, i.e. that is it has a fixed value :math:`F`; we'll also ignore effects like sky background systematic errors. We'll assume that a series of :math:`N` measurements are performed, where the :math:`i^{\rm th}` measurement reports the observed flux :math:`F_i` and error :math:`e_i`. The question is, given this set of measurements :math:`D = \{F_i,e_i\}`, what is our best estimate of the true flux :math:`F`? [#note_about_errors]_ 

.. [#note_about_errors] We'll make the reasonable assumption of normally-distributed measurement errors. In a Frequentist perspective, :math:`e_i` is the standard deviation of the results of the single measurement event in the limit of (imaginary) repetitions of *that event*. In the Bayesian perspective, :math:`e_i` describes the probability distribution which quantifies our knowledge of :math:`x` given :math:`x_i`.

First we'll use Python to generate some toy data to demonstrate the two approaches to the problem. We'll draw 50 samples :math:`F_i` with a mean of 1000 (in arbitrary units) and a (known) error :math:`e_i`:

.. code-block:: python

    >>> np.random.seed(2)  # for reproducibility
    >>> e = np.random.normal(30, 3, 50)
    >>> F = np.random.normal(1000, e)

In this toy example we already know the true flux :math:`F`, but the question is this: given our measurements and errors, what is our best point estimate of the true flux? Let's look at a frequentist and a Bayesian approach to solving this.


Frequentist Approach to Flux Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll start with the classical frequentist maximum likelihood approach. Given a single observation :math:`D_i = (F_i, e_i)`, we can compute the probability distribution of the measurement given the true flux :math:`F` given our assumption of Gaussian errors:

.. math::

    P(D_i|F) = \left(2\pi e_i^2\right)^{-1/2} \exp{\left(\frac{-(F_i - F)^2}{2 e_i^2}\right)}.

This should be read "the probability of :math:`D_i` given :math:`F` equals ...". You should recognize this as a normal distribution with mean :math:`F` and standard deviation :math:`e_i`. We construct the *likelihood* by computing the product of the probabilities for each data point:

.. math::

    \mathcal{L}(D|F) = \prod_{i=1}^N P(D_i|F)

Here :math:`D = \{D_i\}` represents the entire set of measurements. For both analytic and numerical reasons, it is often more convenient to instead consider the log-likelihood. Combining the previous two equations gives

.. math::

    \log\mathcal{L} = -\frac{1}{2} \sum_{i=1}^N \left[ \log(2\pi  e_i^2) + \frac{(F_i - F)^2}{e_i^2} \right].

We would like to determine :math:`F` such that this likelihood is maximized. For this simple problem, the maximization can be computed analytically (e.g. by setting :math:`d\log\mathcal{L}/dF|_{\hat{F}} = 0`), which results in the following point estimate of :math:`F`:

.. math::

    \hat{F} = \frac{\sum w_i F_i}{\sum w_i};~~w_i = 1/e_i^2

The result is a simple weighted mean of the observed values. Notice that in the case of equal errors :math:`e_i`, the weights cancel and :math:`\hat{F}` is simply the mean of the observed data.

We can go further and ask what the uncertainty of our estimate is. In the frequentist approach, this can be accomplished using a Gaussian approximation to the peak likelihood; in this simple case this fit can also be solved analytically to give:

.. math::

    \sigma_{\hat{F}} = \left(\sum_{i=1}^N w_i \right)^{-1/2}

This result can be evaluated this in Python as follows:

.. code-block:: python

    >>> w = 1. / e ** 2
    >>> F_hat = np.sum(w * F) / np.sum(w)
    >>> sigma_F = w.sum() ** -0.5

For our particular data, the result is :math:`\hat{F} = 999 \pm 4`.


Bayesian Approach to Flux Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bayesian approach, as you might expect, begins and ends with probabilities. The fundamental result of interest is our knowledge of the parameters in question: in this case, :math:`P(F|D)`.

Note that while this formulation makes sense given the Bayesian view of probability, the setup is *fundamentally contrary* to the frequentist philosophy, which says that *probabilities have no meaning for fixed model parameters* like :math:`F`.

To compute this result, Bayesians next apply Bayes' theorem, a fundamental law of probability:

.. math::

    P(F|D) = \frac{P(D|F)~P(F)}{P(D)}

Though Bayes' theorem is where Bayesians get their name, it is not this theorem itself that is controversial, but the Bayesian *interpretation of probability* implied by the term :math:`P(F|D)`.

Let's take a look at each of the terms in this expression:

- :math:`P(F|D)`: The **posterior**, or the probability of the model parameters given the data.
- :math:`P(D|F)`: The **likelihood**, which is proportional to the :math:`\mathcal{L}(D|F)` used in the frequentist approach.
- :math:`P(F)`: The **model prior**, which encodes what we knew about the model prior to the application of the data :math:`D`.
- :math:`P(D)`: The **model evidence**, which in practice amounts to simply a normalization term.

If we set the prior :math:`P(F) \propto 1` (a *flat prior*) [#note_flat]_, we find

.. math::

    P(F|D) \propto \mathcal{L}(D|F).

That is, with a flat prior in :math:`F`, the Bayesian posterior is maximized at precisely the same value as the frequentist result! So despite the philosophical differences, we see that (for this simple problem at least) the Bayesian and frequentist point estimates are equivalent.

.. [#note_flat] A flat prior is an example of an improper prior: that is, it cannot be normalized. In practice, we could remedy this by imposing some bounds on possible values: say, :math:`0 < F < F_{tot}`, the total flux of all sources in the universe.

You might notice that we glossed over one important piece here: the prior, :math:`P(F)`. The prior allows inclusion of other information into the computation, which becomes very useful in cases where multiple measurement strategies are being combined to constrain a single model (as is the case in, e.g. cosmological parameter estimation). The necessity to specify a prior, however, is one of the more controversial pieces of Bayesian analysis.

A frequentist will point out that the prior is problematic when no true prior information is available. Though it might seem straightforward to use a **non-informative prior** like the flat prior mentioned above, there are some surprising subtleties involved. [#stark]_ It turns out that in many situations, a truly uninformative prior cannot exist! Frequentists point out that the subjective choice of a prior which necessarily biases the result has no place in scientific data analysis.

A Bayesian would counter that frequentism doesn't solve this problem, but simply skirts the question. Frequentism can often be viewed as simply a special case of the Bayesian approach for some (implicit) choice of the prior: a Bayesian would say that it's better to make this implicit choice explicit, even if the choice might include some subjectivity. Furthermore, as we'll see below, the question frequentism answers is not always the question the researcher wants to ask.

In simple problems like this where the results of the frequentist and Bayesian approaches agree, arguments over the use of a prior and the philosophy of probability may seem frivolous. But as we'll show below, there are situations where the different approaches can lead to very different results and interpretations.

.. [#stark] For an enlightening discussion, see Philip B. Stark, *Constraints versus Priors*, http://www.stat.berkeley.edu/~stark/Preprints/constraintsPriors13.pdf

Where The Results Diverge
-------------------------
In the simple example above, the frequentist and Bayesian approaches give basically the same result. While it is easy to show that the two approaches are often equivalent for simple problems, it is also true that they can diverge greatly for more complicated problems. In practice, this divergence most often makes itself most clear in two different ways:

1. The handling of nuisance parameters: i.e. parameters which affect the final result, but are not otherwise of interest.
2. The different handling of uncertainty: for example, the subtle (and often overlooked) difference between frequentist confidence intervals and Bayesian credible regions.

We'll discuss examples of these below.

Nuisance Parameters: Bayes' Billiards Game
------------------------------------------
We'll start by discussing the first point: nuisance parameters. A nuisance parameter is any quantity whose value is not relevant to the goal of an analysis, but is nevertheless required to determine the result which is of interest. For example, we might have a situation similar to the flux measurement above, but in which the errors :math:`e_i` are unknown. One potential approach is to treat these errors as nuisance parameters.

Let's consider an example of nuisance parameters borrowed from [Eddy2004]_ that, in one form or another, dates all the way back to the posthumous 1763 paper written by Thomas Bayes himself [Bayes1763]_. The setting is a gambling game in which Alice and Bob bet on the outcome of a process they can't directly observe.

Alice and Bob enter a room. Behind a curtain there is a billiard table, which they cannot see. Their friend Carol rolls a ball down the table, and marks where it lands. Once this mark is in place, Carol begins rolling new balls down the table. If the ball lands to the left of the mark, Alice gets a point; if it lands to the right of the mark, Bob gets a point.  We can assume for the sake of example that Carol's rolls are unbiased: that is, the balls have an equal chance of ending up anywhere on the table.  The first person to reach six points wins the game.

Here the location of the mark (determined by the first roll) can be considered a nuisance parameter: it is unknown -- perhaps even unknowable -- and not of immediate interest, but it clearly must be accounted for when predicting the outcome of subsequent rolls. If the first roll settles far to the right, then subsequent rolls will favor Alice. If it settles far to the left, Bob will be favored instead.

Given this setup, here is the question to answer: *In a particular game, after eight rolls, Alice has five points and Bob has three points. What is the probability that Bob will go on to win the game?*

Intuitively, we realize that because Alice received five of the eight points, the marker placement likely favors her. Given that she has three opportunities to get a favorable roll before Bob can win, she seems to have clinched it.  But quantitatively speaking, what is the probability that Bob will persist to win?


A Naïve Frequentist Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Someone following a classical frequentist approach might reason as follows:

To determine the result, we need to estimate where the marker sits. We'll quantify this marker placement as a probability :math:`p` that any given roll lands in Alice's favor.  Because five balls out of eight fell on Alice's side of the marker, we compute the maximum likelihood estimate of :math:`p`, given by:

.. math::

    \hat{p} = 5/8,

a result follows in a straightforward manner from the binomial likelihood. Assuming this maximum likelihood probability, we can compute the probability that Bob will win, which is given by:

.. math::

    P(B) = (1 - \hat{p})^3

That is, he needs to win three rolls in a row. Thus, we find that the probability of Bob winning is 0.053. In other words, we expect that the odds against Bob winning are 18 to 1.


A Bayesian Approach
~~~~~~~~~~~~~~~~~~~
A Bayesian approach to this problem involves *marginalizing* (i.e. integrating) over the unknown :math:`p` so that, assuming the prior is accurate,  our result is agnostic to its actual value. In this vein, we'll consider the following quantities:

- :math:`B` = Bob Wins
- :math:`D` = observed data, i.e. :math:`D = (n_A, n_B) = (5, 3)`
- :math:`p` = unknown probability that a ball lands on Alice's side during the current game

We want to compute :math:`P(B|D)`; that is, the probability that Bob wins given our observation that Alice currently has five points to Bob's three. A Bayesian would recognize that this expression can be computed by integrating over the joint distribution :math:`P(B,p|D)`:

.. math::

    P(B|D) \equiv \int_{-\infty}^\infty P(B,p|D) {\mathrm d}p

This identity follows from the definition of conditional probability, and the law of total probability: that is, it is a fundamental consequence of probability axioms and will always be true. Even a frequentist would recognize this; they would simply disagree with our interpretation of :math:`P(p)` as being a measure of uncertainty of knowledge.

To compute this result, we will manipulate the above expression for :math:`P(B|D)` until we can express it in terms of other quantities that we can compute.

We'll start by applying the definition of conditional probability to expand the term :math:`P(B,p|D)`:

.. math::

    P(B|D) = \int P(B|p, D) P(p|D) dp

Next we use Bayes' rule to rewrite :math:`P(p|D)`:

.. math::

    P(B|D) = \int P(B|p, D) \frac{P(D|p)P(p)}{P(D)} dp

Finally, using the same probability identity we started with, we can expand :math:`P(D)` in the denominator to find:

.. math::

    P(B|D) = \frac{\int P(B|p,D) P(D|p) P(p) dp}{\int P(D|p)P(p) dp}

Now the desired probability is expressed in terms of three quantities that we can compute:

- :math:`P(B|p,D)`: This term is proportional to the frequentist likelihood we used above. In words: given a marker placement :math:`p` and Alice's 5 wins to Bob's 3, what is the probability that Bob will go on to six wins?  Bob needs three wins in a row, i.e. :math:`P(B|p,D) = (1 - p) ^ 3`.
- :math:`P(D|p)`: this is another easy-to-compute term. In words: given a probability :math:`p`, what is the likelihood of exactly 5 positive outcomes out of eight trials? The answer comes from the Binomial distribution: :math:`P(D|p) \propto p^5 (1-p)^3`
- :math:`P(p)`: this is our prior on the probability :math:`p`. By the problem definition, we can assume that :math:`p` is evenly drawn between 0 and 1.  That is, :math:`P(p) \propto 1` for :math:`0 \le p \le 1`.

Putting this all together and simplifying gives

.. math::

    P(B|D) = \frac{\int_0^1 (1 - p)^6 p^5 dp}{\int_0^1 (1 - p)^3 p^5 dp}.

These integrals are instances of the beta function, so we can quickly evaluate the result using scipy:

.. code-block:: python

    >>> from scipy.special import beta
    >>> P_B_D = beta(6+1, 5+1) / beta(3+1, 5+1)

This gives :math:`P(B|D) = 0.091`, which is equivalent to odds of 10 to 1 against Bob winning.


Discussion
~~~~~~~~~~
The Bayesian approach gives odds of 10 to 1 against Bob, while the naïve frequentist approach gives odds of 18 to 1 against Bob. So which one is correct?

For a simple problem like this, we can answer this question empirically by simulating a large number of games and count the fraction of suitable games which Bob goes on to win. This can be coded in a couple dozen lines of Python (see part II of [VanderPlas2014]_). The result of such a simulation confirms the Bayesian result: 10 to 1 against Bob winning.

So what is the takeaway: is frequentism wrong? On the contrary: in this case, the incorrect result is more a matter of the approach being "naïve" than it being "frequentist". The approach above does not consider how :math:`p` may vary. Frequentism can certainly address this by, e.g. applying a transformation and conditioning of the data to isolate dependence on :math:`p`, or by performing a Bayesian-like integral over the sampling distribution of the frequentist estimator :math:`\hat{p}`.

Another potential point of contention is that the question itself is posed in a way that is perhaps unfair to the classical, frequentist approach. A frequentist might instead hope to give the answer in terms of null tests or confidence intervals: that is, they might devise a procedure to construct limits which would provably bound the correct answer in :math:`100\times(1 - \alpha)` percent of similar trials, for some value of :math:`\alpha` – say, 0.05. We'll further discuss the meaning of such confidence intervals below.

There is one clear common point of these two frequentist responses: both require some degree of effort and/or special expertise in classical methods; perhaps a suitable frequentist approach would be immediately obvious to an expert statistician, but is not particularly obvious to a statistical lay-person simply trying to answer the question at hand. In this sense, it could be argued that for a problem like this (i.e. with a well-motivated prior), Bayesianism provides a better framework for handling nuisance parameters: by simple algebraic manipulation of a few well-known axioms of probability interpreted in a Bayesian sense, we straightforwardly arrive at the correct answer without need for other special statistical expertise.


Confidence vs. Credibility: Jaynes' Truncated Exponential
---------------------------------------------------------
A second major consequence of the philosophical difference between frequentism and Bayesianism is in the handling of uncertainty, exemplified by the standard tools of each method: frequentist confidence intervals (CIs) and Bayesian credible regions (CRs). Despite their apparent similarity, the two approaches are fundamentally different. Both are statements of probability, but the probability refers to different aspects of the bound. For example, when constructing a standard 95% bound about a parameter :math:`\theta`:

- A Bayesian would say: "Given our observed data, there is a 95% probability that the true value of :math:`\theta` lies within the credible region".
- A frequentist would say: "If this experiment is repeated many times, in 95% of these cases the computed confidence interval will contain the true :math:`\theta`." [#wasserman_note]_

.. [#wasserman_note] [Wasserman2004]_ notes on p. 92 that we need not consider repetitions of the same experiment; it's sufficient to consider repetitions of any correctly-performed frequentist procedure.

Notice the subtle difference: the Bayesian makes a statement of probability about the *parameter value* given a *fixed credible region*. The frequentist makes a statement of probability about the *confidence interval itself* given a *fixed parameter value*. This distinction follows straightforwardly from the definition of probability discussed above: the Bayesian probability is a statement of degree of knowledge about a parameter; the frequentist probability is a statement of long-term limiting frequency of quantities (such as the CI) derived from the data.

This difference must necessarily affect our interpretation of results. For example, it is common in scientific literature to see it claimed that it is 95% certain that an unknown parameter lies within a given 95% CI, but this is not the case! This is erroneously applying the Bayesian interpretation to a frequentist construction. This frequentist oversight can perhaps be forgiven, as under most circumstances (such as the simple flux measurement example above), the Bayesian CR and frequentist CI will more-or-less overlap. But, as we'll see below, this overlap cannot always be assumed, especially in the case of non-Gaussian distributions constrained by few data points. As a result, this common misinterpretation of the frequentist CI can lead to dangerously erroneous interpretations.

To demonstrate a situation in which the frequentist confidence interval and the Bayesian credibility region do not overlap, let us turn to an example given by E.T. Jaynes, a 20th century physicist who wrote extensively on statistical inference in Physics. In his words, consider a device that

    "...will operate without failure for a time :math:`\theta` because of a protective chemical inhibitor injected into it; but at time :math:`\theta` the supply of the chemical is exhausted, and failures then commence, following the exponential failure law. It is not feasible to observe the depletion of this inhibitor directly; one can observe only the resulting failures. From data on actual failure times, estimate the time :math:`\theta` of guaranteed safe operation..." [Jaynes1976]_

Essentially, we have data :math:`D` drawn from the model:

.. math::

    P(x|\theta) = \left\{
    \begin{array}{lll}
    \exp(\theta - x) &,& x > \theta\\
    0                &,& x < \theta
    \end{array}
    \right\}

where :math:`p(x|\theta)` gives the probability of failure at time :math:`x`, given an inhibitor which lasts for a time :math:`\theta`. We observe some failure times :math:`D = \{10, 12, 15\}` and ask for 95% uncertainty bounds on the value of :math:`\theta`.

First, let's think about what common-sense would tell us. Given the model, an event can only happen after a time :math:`\theta`. Turning this around tells us that the upper-bound for :math:`\theta` must be :math:`\min(D)`. So, for our particular example, we would immediately write :math:`\theta \le 10`. With this in mind, let's explore how a frequentist and a Bayesian approach compare to this observation.

Truncated Exponential: A Frequentist Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the frequentist paradigm, we'd like to compute a confidence interval on the value of :math:`\theta`. We can start by observing that the population mean is given by

.. math::

    E(x) = \int_0^\infty xp(x)dx = \theta + 1.

So, using the sample mean as the point estimate of :math:`E(x)`, we have an unbiased estimator for :math:`\theta` given by

.. math::

    \hat{\theta} = \frac{1}{N} \sum_{i=1}^N x_i - 1.

In the large-:math:`N` limit, the central limit theorem tells us that the sampling distribution is normal with standard deviation given by the standard error of the mean: :math:`\sigma_{\hat{\theta}}^2 = 1/N`, and we can write the 95% (i.e. :math:`2\sigma`) confidence interval as

.. math::

    CI_{\rm large~N} = \left(\hat{\theta} - 2 N^{-1/2},~\hat{\theta} + 2 N^{-1/2}\right)

For our particular observed data, this gives a confidence interval around our unbiased estimator of :math:`CI(\theta) = (10.2, 12.5)`, entirely above our common-sense bound of :math:`\theta < 10`! We might hope that this discrepancy is due to our use of the large-:math:`N` approximation with a paltry :math:`N=3` samples. A more careful treatment of the problem (See [Jaynes1976]_ or part III of [VanderPlas2014]_) gives the exact confidence interval :math:`(10.2, 12.2)`: the 95% confidence interval entirely excludes the sensible bound :math:`\theta < 10`!

Though this may seem counter-intuitive, this result is in fact correct: the approach has successfully answered the frequentist question. 95% of CIs so constructed on data from this model will in fact contain the true :math:`\theta`; this particular draw of :math:`D` just happens to be in the unlucky 5%.


Truncated Exponential: A Bayesian Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bayesian approach to the problem starts with Bayes' rule:

.. math::

    P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}.

We use the likelihood given by 

.. math::

    P(D|\theta) \propto \prod_{i=1}^N P(x_i|\theta)

and, in the absence of other information, use an uninformative flat prior [#note_uninformative]_ to find

.. math::

    P(\theta|D) \propto \left\{
    \begin{array}{lll}
    N\exp\left[N(\theta - \min(D))\right] &,& \theta < \min(D)\\
    0                &,& \theta > \min(D)
    \end{array}
    \right\}

where :math:`\min(D)` is the smallest value in the data :math:`D`, which enters because of the truncation of :math:`P(x_i|\theta)`. Because :math:`P(\theta|D)` increases exponentially up to the cutoff, the shortest 95% credibility interval :math:`(\theta_1, \theta_2)` will be given by :math:`\theta_2 = \min(D)`, and :math:`\theta_1` given by the solution to the equation

.. math::

     \int_{\theta_1}^{\theta_2} P(\theta|D){\rm d}\theta = f

which has the solution

.. math::

    \theta_1 = \theta_2 + \frac{1}{N}\ln\left[1 - f(1 - e^{-N\theta_2})\right].

For our particular data, the Bayesian credible region is

.. math::

    CR(\theta) = (9.0, 10.0)

which agrees with our common-sense bound.

.. [#note_uninformative] The flat prior in this case can be motivated by maximum entropy; see, e.g. [Jeffreys1946]_. Still, the use of uninformative priors like this often raises eyebrows among frequentists: there are good arguments that even "uninformative" priors add information; see e.g. [Evans2002]_.

Discussion
~~~~~~~~~~
Why do the frequentist CI and Bayesian CR give such different results? The reason goes back to the definitions of the CI and CR, and to the fact that *the two approaches are answering different questions*. The Bayesian CR answers a question about the value of :math:`\theta` itself (the probability that the parameter is in the fixed CR), while the frequentist CI answers a question about the procedure used to construct the CI (the probability that the constructed CI will contain the fixed parameter).

Using Monte Carlo simulations, it is possible to confirm that both the above results correctly answer their respective questions (see [VanderPlas2014]_, III). In particular, 95% of frequentist CIs constructed using data drawn from this model in fact contain the true :math:`\theta`. Our particular data are simply among the unhappy 5% which the confidence interval misses. But this makes clear the danger of misapplying the Bayesian interpretation to a CI: this particular CI is not 95% likely to contain the true value; it is in fact 0% likely!

Does this mean that frequentism is incorrect? No: it simply shows that we must carefully keep in mind what question frequentism is answering. Frequentism does not seek probabilities of *parameter values given data*, as the Bayesian approach does; it seeks probabilities of *computed limits given a recipe for constructing them*. Despite this, it is common to see a 95% confidence interval interpreted in the Bayesian sense: as a fixed interval that the parameter is expected to be found in 95% of the time. As we see clearly here, this interpretation is flawed, and should be carefully avoided. For sensible parameter constraints from a single dataset, Bayesianism may be preferred, especially if the difficulties of uninformative priors can be avoided through the use of true prior information. [#note_unbiased]_

.. [#note_unbiased] Note that this example is a bit unfair as it relies on the *unbiased* frequentist estimator. Other estimators are available: if the (biased) maximum likelihood estimator were used instead, the confidence interval would be very similar to the Bayesian credible region derived above. It is well-known that the unbiased estimator is not always the optimal, especially with small :math:`N` and censored models; see, e.g. [Hardy2003]_. Nevertheless, as an illustration of the correct interpretation of the CI, this remains a useful example.


Bayesianism in Practice: Markov Chain Monte Carlo
-------------------------------------------------
Though Bayesianism has some nice features in theory, in practice it can be extremely computationally intensive: while simple problems like those examined above lend themselves to relatively easy analytic integration, real-life Bayesian computations often require numerical integration of high-dimensional parameter spaces.

A turning-point in practical Bayesian computation was the development and application of sampling methods such as Markov Chain Monte Carlo (MCMC). MCMC is a class of algorithms which can efficiently characterize even high-dimensional posterior distributions through drawing of randomized samples such that the points are distributed according to the posterior. A detailed discussion of MCMC is well beyond the scope of this paper; an excellent introduction can be found in [Gelman2004]_. Below, we'll propose a straightforward model and compare three MCMC implementations available in Python.


Application: A Simple Linear Model
----------------------------------
As an example of a more realistic data-driven analysis, let's consider a simple three-parameter linear model which fits a straight-line to data with unknown errors. The parameters will be the the y-intercept :math:`\alpha`, the slope :math:`\beta`, and the (unknown) normal scatter :math:`\sigma` about the line.

For data :math:`D = \{x_i, y_i\}`, the model is

.. math::

    \hat{y}(x_i|\alpha,\beta) = \alpha + \beta x_i,

and the likelihood is

.. math::

    \mathcal{L}(D|\alpha,\beta,\sigma) = (2\pi\sigma^2)^{-N/2} \prod_{i=1}^N \exp\left[\frac{-[y_i - \hat{y}(x_i|\alpha, \beta)]^2}{2\sigma^2}\right].

We'll evaluate this model on the following data set:

.. code-block:: python

    import numpy as np
    np.random.seed(42)
    theta_true = (25, 0.5)
    xdata = 100 * np.random.random(20)
    ydata = theta_true[0] + theta_true[1] * xdata
    ydata = np.random.normal(ydata, 10) # add error

Below we'll consider a frequentist solution to this problem, as well as a Bayesian solution computed with several MCMC implementations in Python: emcee [#emcee]_, PyMC [#pymc]_, and PyStan [#pystan]_. A full discussion of the strengths and weaknesses of the various MCMC algorithms used by the packages is out of scope for this paper, as is a full discussion of performance benchmarks for the three packages (for all three, this example runs in under 20 seconds on a single machine). Rather, the purpose of this section is to show side-by-side examples of the Python APIs of the three packages. First, though, we'll consider a frequentist solution.

.. [#emcee] emcee: The MCMC Hammer http://dan.iel.fm/emcee/

.. [#pymc] PyMC: Bayesian Inference in Python http://pymc-devs.github.io/pymc/

.. [#pystan] PyStan: The Python Interface to Stan https://pystan.readthedocs.org/


Frequentist Solution
~~~~~~~~~~~~~~~~~~~~
A frequentist solution can be found by computing the maximum likelihood estimate. For standard linear problems such as this, the result can be computed using efficient linear algebra. If we define the *parameter vector*, :math:`\theta = [\alpha~\beta]^T`; the *response vector*, :math:`Y = [y_1~y_2~y_3~\cdots~y_N]^T`; and the *design matrix*,

.. math::

    X = \left[
           \begin{array}{lllll}
               1 & 1 & 1 &\cdots & 1\\
               x_1 & x_2 & x_3 & \cdots & x_N
           \end{array}\right]^T,

it can be shown that the maximum likelihood solution is

.. math::

    \hat{\theta} = (X^TX)^{-1}(X^T Y).

The confidence interval around this value is an ellipse in parameter space defined by the following matrix:

.. math::

    \Sigma_{\hat{\theta}}
                   \equiv \left[
                      \begin{array}{ll}
                         \sigma_\alpha^2 & \sigma_{\alpha\beta} \\
                          \sigma_{\alpha\beta} & \sigma_\beta^2
                      \end{array}
                    \right]
                   = \sigma^2 (M^TM)^{-1}.

Here :math:`\sigma` is our unknown error term; it can be estimated based on the variance of the residuals about the fit. The off-diagonal elements of :math:`\Sigma_{\hat{\theta}}` are the correlated uncertainty between the estimates. In code, this is what it looks like:

.. code-block:: python

    X = np.vstack([np.ones_like(xdata), xdata]).T
    theta_hat = np.linalg.solve(np.dot(X.T, X),
                                np.dot(X.T, ydata))
    y_hat = np.dot(X, theta_hat)
    sigma_hat = np.std(ydata - y_hat)
    Sigma = sigma_hat ** 2 * np.linalg.inv(np.dot(X.T, X))

The result is shown by the black ellipse in Figure :ref:`fig1`.


Bayesian Solution: Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bayesian result is encapsulated in the posterior, which is proportional to the product of the likelihood and the prior; in this case we must be aware that a flat prior is not uninformative. Through symmetry arguments, first developed by [Jeffreys1946]_, it can be shown that an uninformative prior for this problem is given by

.. math::

    P(\alpha,\beta,\sigma) \propto \frac{1}{\sigma}(1 + \beta^2)^{-3/2}.

(See [VanderPlas2014]_, part IV for a straightforward derivation of this). With this prior and the above likelihood, we are prepared to numerically evaluate the posterior via MCMC.


Solution with emcee
~~~~~~~~~~~~~~~~~~~
The emcee package [ForemanMackey2013]_ is a lightweight pure-Python package which implements Affine Invariant Ensemble MCMC [Goodman2010]_, a sophisticated version of MCMC sampling. To use ``emcee``, all that is required is to define a Python function representing the logarithm of the posterior. For clarity, we'll factor this definition into two functions, the log-prior and the log-likelihood:

.. code-block:: python

    import emcee  # version 2.0

    def log_prior(theta):
        alpha, beta, sigma = theta
        if sigma < 0:
            return -np.inf  # log(0)
        else:
            return (-1.5 * np.log(1 + beta**2)
                    - np.log(sigma))

    def log_like(theta, x, y):
       alpha, beta, sigma = theta
       y_model = alpha + beta * x
       return -0.5 * np.sum(np.log(2*np.pi*sigma**2) +
                            (y-y_model)**2 / sigma**2)

    def log_posterior(theta, x, y):
        return log_prior(theta) + log_like(theta,x,y)

Next we set up the computation. ``emcee`` combines multiple "walkers", each of which is its own Markov chain. We'll also specify a burn-in period, to allow the chains to stabilize prior to drawing our final traces:

.. code-block:: python

   ndim = 3  # number of parameters in the model
   nwalkers = 50  # number of MCMC walkers
   nburn = 1000  # "burn-in" to stabilize chains
   nsteps = 2000  # number of MCMC steps to take
   starting_guesses = np.random.rand(nwalkers, ndim)


Now we call the sampler and extract the trace:

.. code-block:: python

    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    log_posterior,
                                    args=[xdata,ydata])
    sampler.run_mcmc(starting_guesses, nsteps)

    # chain is of shape (nwalkers, nsteps, ndim):
    # discard burn-in points and reshape:
    trace = sampler.chain[:, nburn:, :]
    trace = trace.reshape(-1, ndim).T

The result is shown by the blue curve in Figure :ref:`fig1`.


Solution with PyMC
~~~~~~~~~~~~~~~~~~
The PyMC package [Patil2010]_ is an MCMC implementation written in Python and Fortran. It makes use of the classic Metropolis-Hastings MCMC sampler [Gelman2004]_, and includes many built-in features, such as support for efficient sampling of common prior distributions. Because of this, it requires more specialized boilerplate than does emcee, but the result is a very powerful tool for flexible Bayesian inference.

The example below uses PyMC version 2.3; as of this writing, there exists an early release of version 3.0, which is a complete rewrite of the package with a more streamlined API and more efficient computational backend. To use PyMC, we first we define all the variables using its classes and decorators:

.. code-block:: python

    import pymc  # version 2.3

    alpha = pymc.Uniform('alpha', -100, 100)

    @pymc.stochastic(observed=False)
    def beta(value=0):
        return -1.5 * np.log(1 + value**2)

    @pymc.stochastic(observed=False)
    def sigma(value=1):
        return -np.log(abs(value))

    # Define the form of the model and likelihood
    @pymc.deterministic
    def y_model(x=xdata, alpha=alpha, beta=beta):
        return alpha + beta * x

    y = pymc.Normal('y', mu=y_model, tau=1./sigma**2,
                    observed=True, value=ydata)

    # package the full model in a dictionary
    model1 = dict(alpha=alpha, beta=beta, sigma=sigma,
                  y_model=y_model, y=y)

Next we run the chain and extract the trace:

.. code-block:: python

    S = pymc.MCMC(model1)
    S.sample(iter=100000, burn=50000)
    trace = [S.trace('alpha')[:], S.trace('beta')[:],
             S.trace('sigma')[:]]

The result is shown by the red curve in Figure :ref:`fig1`.


Solution with PyStan
~~~~~~~~~~~~~~~~~~~~
PyStan is the official Python interface to Stan, a probabilistic programming language implemented in C++ and making use of a Hamiltonian MCMC using a No U-Turn Sampler [Hoffman2014]_. The Stan language is specifically designed for the expression of probabilistic models; PyStan lets Stan models specified in the form of Python strings be parsed, compiled, and executed by the Stan library. Because of this, PyStan is the least "Pythonic" of the three frameworks:

.. code-block:: python

    import pystan  # version 2.2

    model_code = """
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

    # perform the fit & extract traces
    data = {'N': len(xdata), 'x': xdata, 'y': ydata}
    fit = pystan.stan(model_code=model_code, data=data,
                      iter=25000, chains=4)
    tr = fit.extract()
    trace = [tr['alpha'], tr['beta'], tr['sigma']]

The result is shown by the green curve in Figure :ref:`fig1`.


Comparison
~~~~~~~~~~
.. figure:: figure1.png

   Comparison of model fits using frequentist maximum likelihood, and Bayesian MCMC using three Python packages: emcee, PyMC, and PyStan. :label:`fig1`

The :math:`1\sigma` and :math:`2\sigma` posterior credible regions computed with these three packages are shown beside the corresponding frequentist confidence intervals in Figure :ref:`fig1`. The frequentist result gives slightly tighter bounds; this is primarily due to the confidence interval being computed assuming a single maximum likelihood estimate of the unknown scatter, :math:`\sigma`. This interpretation can be confirmed by plotting the Bayesian posterior conditioned on the maximum likelihood estimate :math:`\hat{\sigma}`; this gives a credible region much closer to the frequentist confidence interval.

The similarity of the three MCMC results belie the differences in algorithms used to compute them: by default, PyMC uses a Metropolis-Hastings sampler, PyStan uses a No U-Turn Sampler (NUTS), while emcee uses an affine-invariant ensemble sampler.  These approaches are known to have differing performance characteristics depending on the features of the posterior being explored. As expected for the near-Gaussian posterior used here, the three approaches give very similar results.

A main apparent difference between the packages is the Python interface. Emcee is perhaps the simplest, while PyMC requires more package-specific boilerplate code. PyStan is the most complicated, as the model specification requires directly writing a string of Stan code.


Conclusion
----------
This paper has offered a brief practical glimpse at the differences between frequentist and Bayesian statistics, which have their root in differing conceptions of probability. Though the two approaches often give indistinguishable results in simple problems, we considered several specific situations in which the results differ: namely the treatment of nuisance parameters, and the interpretation of uncertainties in estimates. Finally, we took a detailed look at the application of these approaches to a simple linear model, and demonstrated how the frequentist and Bayesian results can be computed using tools available in the Python programming language.



References
----------
.. [Bayes1763] T. Bayes.
               *An essay towards solving a problem in the doctrine of chances*.
               Philosophical Transactions of the Royal Society of London
               53(0):370-418, 1763

.. [Eddy2004] S.R. Eddy. *What is Bayesian statistics?*.
              Nature Biotechnology 22:1177-1178, 2004

.. [Evans2002] S.N. Evans & P.B. Stark. *Inverse Problems as Statistics*.
               Mathematics Statistics Library, 609, 2002.

.. [ForemanMackey2013] D. Foreman-Mackey, D.W. Hogg, D. Lang, J.Goodman.
                       *emcee: the MCMC Hammer*. PASP 125(925):306-312, 2014

.. [Gelman2004] A. Gelman, J.B. Carlin, H.S. Stern, and D.B. Rubin.
                *Bayesian Data Analysis, Second Edition.*
                Chapman and Hall/CRC, Boca Raton, FL, 2004.

.. [Goodman2010] J. Goodman & J. Weare.
                 *Ensemble Samplers with Affine Invariance*.
                 Comm. in Applied Mathematics and
                 Computational Science 5(1):65-80, 2010.

.. [Hardy2003]  M. Hardy. *An illuminating counterexample*.
                Am. Math. Monthly 110:234–238, 2003.

.. [Hoffman2014] M.C. Hoffman & A. Gelman.
                 *The No-U-Turn Sampler: Adaptively Setting Path Lengths
                 in Hamiltonian Monte Carlo*. JMLR, submitted, 2014.

.. [Jaynes1976] E.T. Jaynes. *Confidence Intervals vs Bayesian Intervals (1976)*
                Papers on Probability, Statistics and Statistical Physics
                Synthese Library 158:149, 1989

.. [Jeffreys1946] H. Jeffreys *An Invariant Form for the Prior Probability in Estimation Problems*.
                  Proc. of the Royal Society of London. Series A
                  186(1007): 453, 1946

.. [Patil2010] A. Patil, D. Huard, C.J. Fonnesbeck.
               *PyMC: Bayesian Stochastic Modelling in Python* 
               Journal of Statistical Software, 35(4):1-81, 2010.

.. [VanderPlas2014] J. VanderPlas. *Frequentism and Bayesianism*.
                    Four-part series (`I <http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/>`_, 
                    `II <http://jakevdp.github.io/blog/2014/06/06/frequentism-and-bayesianism-2-when-results-differ/>`_,
                    `III <http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/>`_,
                    `IV <http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/>`_) on *Pythonic Perambulations*
                    http://jakevdp.github.io/, 2014.

.. [Wasserman2004] L. Wasserman.
                 *All of statistics: a concise course in statistical inference*.
                 Springer, 2004.
