:author: Allen B. Downey
:email: allen.downey@olin.edu
:institution: Olin College of Engineering

------------------------------------------------
Will Millennials Ever Get Married?
------------------------------------------------

We investigate marriage patterns among women in the United States with
data from the National Survey of Family Growth (NSFG). Using survival
analysis methods implemented in Python, we describe age at first
marriage for successive cohorts based on decade of birth. Of women born
in the 1980s, 60% are married by age 32; of women born in 1990s, 13% are
married by age 22. For both groups these results reflect substantial and
statistically significant decreases compared to previous cohorts.

Survival analysis, marriage patterns.

Introduction
============

A recent study from the Pew Research Center reports that the fraction of
adults in the U.S. who have never married is increasing. In 2012, 23% of
men 25 and older had never married, and 17% of women. In 1960, only 10%
of men were unmarried and 8% of women. Their report focuses on the
causes of these trends, but does not address this question: is the
fraction of people who never marry increasing, are people marrying
later, or both? That is the subject of this paper.

To answer these questions, we apply tools of survival analysis to data
from the National Survey of Family Growth (NSFG). Since 1973 the
U.S. Centers for Disease Control and Prevention (CDC) have conducted
this survey, intended to gather “information on family life, marriage
and divorce, pregnancy, infertility, use of contraception, and men’s and
women’shealth.” See http://cdc.gov/nchs/nsfg.htm.

NSFG data is organized in cycles; during each cycle several thousand
respondents were interviewed, including women ages 14–44. Men were
included starting with Cycle 6 in 2002, but for this study we use only
data from female respondents.

Table [tab:nsfg] shows the interview dates for each cycle, the number of
respondents, and the birth years of the respondents. We did not use data
from Cycles 1 and 2 because they included only married women. The total
sample size for this study is 52 789.

[h]

+---------+-------------+---------------+-----------+
| Cycle   | Interview   | Number of     | Birth     |
+---------+-------------+---------------+-----------+
|         | Dates       | Respondents   | Years     |
+---------+-------------+---------------+-----------+
| 3       | 1982–83     | 7 969         | 1937–68   |
+---------+-------------+---------------+-----------+
| 4       | 1988–88     | 8 450         | 1943–73   |
+---------+-------------+---------------+-----------+
| 5       | 1995        | 10 847        | 1950–80   |
+---------+-------------+---------------+-----------+
| 6       | 2002–03     | 7 643         | 1957–88   |
+---------+-------------+---------------+-----------+
| 7       | 2006–10     | 12 279        | 1961–95   |
+---------+-------------+---------------+-----------+
| 8       | 2011–13     | 5 601         | 1966–98   |
+---------+-------------+---------------+-----------+

[tab:nsfg]

For each respondent we have date of birth (year and month), date of
interview, and date of first marriage, if applicable. So we can compute
with resolution of one month the respondent’s age at interview, age, and
age at first marriage, agemarry.

To study changes in marriage patterns over time, we group the
respondents into cohorts by decade of birth. For each cohort,
Table [tab:cohorts] reports the number of respondents, range of ages
when they were interviewed, number who had been married at least once at
time of interview, and the number of married respondents whose date of
marriage was not ascertained.

Cohort 30 refers to women who born in the 1930s. The focus of this paper
is to describe and predict marriage patterns of the last two cohorts,
women born in the 1980s and 90s.

[ht]

+----------+---------------+-------------+-----------+---------------+
| Cohort   | Number of     | Age at      | Number    | Missing Age   |
+----------+---------------+-------------+-----------+---------------+
|          | Respondents   | Interview   | Married   | at Marriage   |
+----------+---------------+-------------+-----------+---------------+
| 30       | 325           | 42–44       | 310       | 0             |
+----------+---------------+-------------+-----------+---------------+
| 40       | 3 608         | 32–44       | 3275      | 0             |
+----------+---------------+-------------+-----------+---------------+
| 50       | 10 631        | 22–44       | 8658      | 10            |
+----------+---------------+-------------+-----------+---------------+
| 60       | 14 484        | 15–44       | 8421      | 27            |
+----------+---------------+-------------+-----------+---------------+
| 70       | 12 083        | 14–43       | 5908      | 25            |
+----------+---------------+-------------+-----------+---------------+
| 80       | 8 536         | 14–33       | 2203      | 8             |
+----------+---------------+-------------+-----------+---------------+
| 90       | 3 122         | 15–23       | 93        | 0             |
+----------+---------------+-------------+-----------+---------------+

[tab:cohorts]

Methodology
===========

Survival analysis
-----------------

Survival analysis is a powerful set of tools with applications in many
domains, but it is often considered a specialized topic. One goal of
this paper is to introduce survival analysis, using Python, for people
who are not already familiar with it.

Survival analysis is used to study and predict the time until an event:
in medicine, the event might be the death of a patient, hence
“survival”; but more generally we might be interested in the time until
failure of a mechanical part, the lifetimes of civilizations, species,
or stars; or in this study the time from birth until first marriage.

The result of survival analysis is most often a **survival function**,
which shows the fraction of the population that survives after
:math:`t`, for any time, :math:`t`. If :math:`T` is a random variable
that represents the time until an event, the survival function,
:math:`S(t)`, is the probability that :math:`T` exceeds :math:`t`:

.. math:: S(t) \equiv \mathrm{Pr}(T > t)

If the distribution of :math:`T` is known, or can be estimated from a
representative sample, computing :math:`S(t)` is simple: it is the
complement of the cumulative distribution function (CDF):

.. math:: S(t) = 1 - \mathrm{CDF_T}(t)

In Python we can compute the survival function like this:

::

    from collections import Counter
    import numpy as np

    def MakeSurvivalFunction(values):
        counter = Counter(values)
        ts, fs = zip(*sorted(counter.items()))
        ts = np.asarray(ts)
        ps = np.cumsum(fs, dtype=np.float)
        ps /= ps[-1]
        ss = 1 - ps
        return SurvivalFunction(ts, ss)

values is a sequence of observed lifetimes. Counter makes a map from
each unique value to the number of times it appears, which we split into
a sorted sequence of times, ts, and their frequencies, fs.

We convert ts to a NumPy array (see http://www.numpy.org/). Then ps is
the cumulative sum of the frequencies, normalized to go from 0 to 1, so
it represents the CDF of the observed values. ss, which is the
complement of ps, is the survival function.

SurvivalFunction is defined in marriage.py, a Python module we wrote for
this project. The code and data for this project are available in a
public Git repository at https://github.com/AllenDowney/MarriageNSFG.

Given a survival curve, we can compute the **hazard function**, which is
the instantaneous death rate at time :math:`t`; that is, the fraction of
people who survive until time :math:`t` and then die at time :math:`t`.
When :math:`t` is continuous, the hazard function, :math:`\lambda(t)`,
is

.. math:: \lambda(t) = -S'(t) / S(t)

Where :math:`S'(t)` is the derivative of :math:`S(t)`. Since the
survival function decreases monotonically, its derivative is
nonpositive, so the hazard function is nonnegative.

With a survival function represented by discrete ts and ss, we can
compute the hazard function like this:

::

    import pandas as pd

    # class SurvivalFunction
    def MakeHazardFunction(self):
        lams = pd.Series(index=self.ts)
        prev = 1.0
        for t, s in zip(self.ts, self.ss):
            lams[t] = (prev - s) / prev
            prev = s
        return HazardFunction(lams)

MakeHazardFunction is a method of SurvivalFunction, which provides
attributes ts and ss. The result, lams, is a Pandas Series object that
maps from the same set of ts to the estimated hazard function,
:math:`\lambda(t)` (see http://pandas.pydata.org).

|Survival and hazard functions for 1930s cohort.| [fig:marriage1]

Figure [fig:marriage1] shows the survival and hazard functions for women
born in the 1930s. These women were interviewed when they were 42–44
years old. At that point more than 95% of them had been married; for the
others we set age at marriage to infinity (np.inf). In this cohort, the
hazard function is highest at ages 18–22, and lower as age increases.

This example demonstrates the simple case, where the respondents are the
same age and most events are complete. But for most applications of
survival analysis, the sample also includes incomplete events. For
example, the 1960s cohort includes women from ages 14–44; for the ones
that are not married, we don’t know when they will marry, if ever. These
missing data are said to be “censored”.

It might be tempting to ignore unmarried women and compute the survival
function for women whose ages at marriage are known. But that would
discard useful information and seriously bias the results.

For women who are not married yet, their age at interview is a lower
bound on their age at marriage. We can use both groups to estimate the
hazard function, then compute the survival function. One common way to
do that is Kaplan-Meier estimation (see
https://en.wikipedia.org/wiki/Kaplan-Meier_estimator).

The fundamental idea is that at each time, :math:`t`, we know the number
of events that occurred and the number of respondents who were “at
risk”; that is, known to to be unmarried. The ratio of these factors
estimates the hazard function.

Initially, the entire sample is considered at risk. At each time step,
we subtract people who got married at age :math:`t` as well as people
who were interviewed at age :math:`t` (and therefore no longer in the
observation pool at the next time step). The following function
implements this algorithm:

::

    def EstimateHazard(complete, ongoing):
        hist_complete = Counter(complete)
        hist_ongoing = Counter(ongoing)

        ts = list(hist_complete|hist_ongoing)
        ts.sort()

        at_risk = len(complete)+len(ongoing)

        lams = pd.Series(index=ts)
        for t in ts:
            ended = hist_complete[t]
            censored = hist_ongoing[t]

            lams[t] = ended / at_risk
            at_risk -= ended + censored

        return HazardFunction(lams)

complete is a sequence of lifetimes for complete events, in this case
age at marriage. ongoing is a sequence of lower bounds for incomplete
observations, in this case age at interview.

``hist_complete`` counts how many respondents were married at each age;
``hist_ongoing`` counts how many unmarried respondents were interviewed
at each age.

ts is a sorted list of observation times, which is the union of unique
values from complete and ongoing.

``at_risk`` is the number of respondents at risk; initially it is the
total number of respondents.

lams is a Pandas Series that maps from each observation time to the
estimated hazard rate.

For each value of t we compute ended, which is the number of people
married at t, and censored, which is the number of people interviewed at
t. The hazard function at t is just the ratio of ended and ``at_risk``.

At the end of each time step, we subtract ended and censored from
``at_risk``.

The result is a HazardFunction object that contains the Series lams and
provides methods to access it.

With this estimated HazardFunction, we can compute the SurvivalFunction.
The hazard function, :math:`\lambda(t)`, is the probability of ending at
time :math:`t` conditioned on surviving until :math:`t`. Therefore, the
probability of surviving until :math:`t` is the cumulative product of
the complementary hazard function:

.. math:: S(t) = \prod_{t_i < t} \left[1 - \lambda(t_i)\right]

Here’s the Python implementation:

::

    # class HazardFunction
    def MakeSurvival(self):
        series = (1 - self.series).cumprod()
        ts = series.index.values
        ss = series.values
        return SurvivalFunction(ts, ss)

We wrote our own implementation of these methods in order to demonstrate
the methodology, and also to make them work efficiently with the
resampling methods described in the next section. But Kaplan-Meier
estimation and other survival analysis algorithms are also implemented
in a Python package called Lifelines (see
http://lifelines.readthedocs.org).

Resampling
----------

The NSFG is intended to be representative of the adult U.S. population,
but it uses stratified sampling to systematically oversample certain
subpopulations, including teenagers and racial minorities. Our analysis
takes this design into account to generate results that are
representative of the population.

As an example of stratified sampling, suppose there are 10 000 people in
the population you are studying, and you sample 100. Each person in the
sample represents 100 people in the population, so each respondent has
the same “sampling weight”.

Now suppose there are two subgroups, a minority of 1 000 people and a
majority of 9 000. A sample of 100 people will have 10 members of the
minority group, on average, which might not be enough for reliable
statistical inference.

In a stratified sample, you might survey 40 people from the minority
group and only 60 from the majority group. This design improves the
power of the sample, but it changes the weight associated with each
respondent. Each of the 40 minorities represents :math:`1000 / 40 = 25`
people in the population, while each of the 60 others represents
:math:`9000 / 60 = 150` people. In general, respondents from oversampled
groups have lower weights.

The NSFG includes a computed weight for each respondent that indicates
how many people in the U.S. population she represents. Some statistical
methods, like regression, can be extended to take these weights into
account, but in general it is not easy.

However, bootstrapping provides a simple and effective approach. The
idea behind bootstrapping is to use the actual sample as a model of the
population, then simulate the results of additional experiments by
drawing new samples (with replacement) from the actual sample.

With stratified sampling, we can modify the bootstrap process to take
sampling weights into account. The following function performs weighted
resampling on the NSFG data:

::

    def ResampleRowsWeighted(df):
        weights = df.finalwgt
        cdf = thinkstats2.Cdf(dict(weights))
        indices = cdf.Sample(len(weights))
        sample = df.loc[indices]
        return sample

df is a Pandas DataFrame with one row per respondent and a column that
contains sampling weights, called finalwgt.

weights is a Series that maps from respondent index to sampling weight.
cdf represents a cumulative distribution function that maps from each
index to its cumulative probability. The Cdf class is provided by
thinkstats2, a Python module that accompanies the second edition of
*Think Stats*.

Sample generates a random sample of indices based on the sampling
weights. The return value, sample, is a Pandas DataFrame that contains
the selected rows. Since the sample is generated with replacement, some
respondents might appear more than once; others might not appear at all.

After resampling, we jitter the data by adding Gaussian noise (mean 0,
standard deviation 1 year) to the respondents’ age at interview and age
at marriage. Jittering contributes some smoothing, which makes the
figures easier to interpret, and some robustness, making the results
less prone to the effect of a small number of idiosyncratic data points.

Jittering makes sense in the context of bootstrapping: each respondent
in the sample represents several thousand people in the population. It
is reasonable to assume that there is variation within each represented
subgroup.

Finally, we discretize age at interview and age at marriage, rounding
down to integer values.

Results
=======

|Survival functions.| [fig:marriage2]

Figure [fig:marriage2] shows the estimated survival curve for each
cohort (we omit the 1930s cohort because it only includes people born
after 1936, so it is not representative of the decade). The colored
lines show the median of 101 resampling runs; the gray regions show 90%
confidence intervals.

Two trends are apparent in this figure: women are getting married later,
and the fraction of women who remain unmarried is increasing.

Table [tab:cohorts2] shows the percentage of married women in each
cohort at ages 22, 32, and 42 (which are the last observed ages for
cohorts 90, 80, and 70).

[ht]

| c\|r\|r\|r Cohort &
| & 22 & 32 & 42
| 40 & 69 & 90 & 92
| 50 & 57 & 85 & 90
| 60 & 41 & 79 & 87
| 70 & 32 & 75 & 82
| 80 & 23 & 60 & –
| 90 & 13 & – & –

[tab:cohorts2]

Two features of this data are striking:

-  At age 22, only 13% of the 90s cohort have been married, contrasted
   with 69% of the 40s cohort. Between these cohorts, the fraction of
   women married by age 22 has dropped more than 11 percentage points
   per decade.

-  At age 32, only 60% of the 80s cohort is married, and their survival
   curve seems to have gone flat. In this cohort, 259 were at risk at
   age 30, and only 9 were married that year; 155 were at risk at age
   31, and none were married; 63 were are risk at age 32, and again none
   were married. These low hazard rates are strange, but they are based
   on sample sizes large enough that it is hard to dismiss them.

Projection
----------

Predicting these kinds of social trends is nearly futile. As we saw in
the previous section, the 80s cohort seems to be on strike, with
unprecedented low marriage rates in their early thirties. Simple
extrapolation of their survival curve predicts that 40% of them will
remain unmarried, more than double the fraction of previous generations.

But at the same time the fraction of women getting married at ages 35–45
has been increasing for several generations, so we might expect that
trend to continue. In that case the gap between the 80s and 70s cohorts
might close.

These prediction methods yield contradictory results. A simple middle
ground is to assume that the hazard function from the previous
generation will apply to the next. For example, for the 70s cohort, the
hazard rate at age 33 was 4.6%. In the 80s cohort, there will be 14
women at risk at age 33, so we predict that :math:`0.64` of them will
get married.

To make these projections (and avoid the word “prediction”), we extend
each HazardFunction using data from the previous cohort:

::

    # class HazardFunction
    def Extend(self, other):
        last_t = self.series.index[-1]
        other_ts = other.series.index
        hs = other.series[other_ts > last_t]
        seq = self.series, hs
        self.series = pd.concat(seq)

Then we convert the extended hazard functions to survival functions,
using HazardFunction.MakeSurvival.

|Survival functions with projections.| [fig:marriage3]

Figure [fig:marriage3] shows the results. Again, the gray regions show a
90% confidence interval. For the 80s cohort, the median projection is
that 72% will marry by age 44, down from 82% in the previous cohort.

For the 90s cohort, the median projection is that only 68% will marry by
age 44. But the projection assumes that this cohort will also go on a
“marriage strike” in their early thirties. This event is probably
idiosyncratic and unlikely to be repeated. So, again, we should not take
these predictions too seriously.

Future work
===========

This work is preliminary, and there are many avenues for future
investigation:

-  The NSFG includes data from male respondents, starting with Cycle 6
   in 2002. We plan to repeat our analysis for male respondents.

-  There are many subgroups in the U.S. that would be interesting to
   explore, including different regions, education and income levels,
   racial and religious groups.

-  We have data from the Canadian General Social Survey, which will
   allow us to compare marriage patterns between countries (see
   http://tinyurl.com/canadagss).

-  We are interested in finding similar data from other countries.

Acknowledgment
==============

Many thanks to Lindsey Vanderlyn for help with data acquisition,
preparation, and analysis.

1

Allen Downey, *Think Stats: Exploratory Data Analysis*", 2nd edition,
O’Reilly Media, October 2014. http://thinkstats2.com

Wendy Wang and Kim Parker, “Record Share of Americans Have Never
Married”, Washington D.C.: Pew Research Center’s Social and Demographic
Trends project, September 2014. http://tinyurl.com/wang14pew

[|image|]Allen B. Downey is a Professor of Computer Science at Olin
College of Engineering in Needham MA, where he teaches programming, data
science, and scientific computing. He is the author of several free
textbooks, including *Think Bayes*, *Think Stats*, and *Think Python*,
all published by O’Reilly Media and available from Green Tea Press:
http://greenteapress.com. He offers workshops and consults on data
science and Bayesian statistics.

.. |Survival and hazard functions for 1930s cohort.| image:: marriage1
.. |Survival functions.| image:: marriage2
.. |Survival functions with projections.| image:: marriage3
.. |image| image:: allen_downey



References
----------

.. [Wang14] Wendy Wang and Kim Parker, 2014. *Record Share of Americans 
           Have Never Married*, Washington D.C.: Pew Research Center's
           Social and Demographic Trends project.  http://www.pewsocialtrends.org/files/2014/09/2014-09-24_Never-Married-Americans.pdf
