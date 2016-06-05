:author: Matar Haller
:email: matar@berkeley.edu
:institution: Helen Wills Neuroscience Institute, University of California, Berkeley
:corresponding:

:author: Jaya Narasimhan
:email: jnaras@berkeley.edu
:institution: Department of Electrical Engineering and Computer Science, University of California, Berkeley

:author: Juan Shishido
:email: juanshishido@berkeley.edu
:institution: School of Information, University of California, Berkeley

----------------------------------------------------------
Tell Me Something I Don't Know: Analyzing OkCupid Profiles
----------------------------------------------------------

.. class:: abstract

The way that people self-present online while dating has broad implications for
the relationships they pursue. Here we present an analysis of 59,000+ OkCupid
user profiles that examines word usage patterns across sexual orientations,
drug usage status, and ethnicities. We find that individuals in particular
demographic groups self-present in consistent ways. Our results also suggest
that users may unintentionally reveal demographic attributes in their online
profiles.

In this paper we analyze online self-presentation by combining natural language
processing (NLP) with machine learning. We use this approach to motivate a
discussion of how to successfully mine text data. We describe our initial
approach, explain why it was problematic, and discuss modifications that made
it successful. In doing so, we review standard NLP techniques, challenge
accepted "norms," and cover several ways to represent text data. We explain
clustering and topic modeling, including k-means and nonnegative matrix
factorization, and discuss why we chose the latter. We then present unexpected
similarities between self-identified drug users and those who declined to
reveal their drug status. Finally, we discuss utilizing permutation testing for
identifying deceit in online self-presentation and discuss how we're working to
extend this analysis.

This work was completed using SciPy, NumPy, pandas, Scikit-Learn, and NLTK.

.. class:: keywords

   natural language processing, machine learning, okcupid, online dating

Introduction
------------

Online dating has become a common way of finding mates, with 22% of 25-34 year
olds having used online dating [Pew15]_. Previous studies suggest that the
free text portion of online dating profiles is an important factor (after
photographs) for assessing attractiveness [Fio08]_. Research into the
principle of homophily suggests that people tend to associate and bond with
similar others and that this principle strongly structures social networks and
ties, most prominently by race and ethnicity [McP01]_. Perhaps not
surprisingly, research suggests that homophily extends to online dating, with
people seeing mates similar to themselves [Fio05]_. However, it remains unclear
if people within a particular demographic group self-present in a similar way
when searching for a mate online.

The original intent of this project was to analyze demographic trends in
self-presentation in online profiles. Specifically, we were interested in
whether demographic groups are distinct in the ways in which they present
themselves online, and specifically whether people from different demographic
groups would self-present in distinct ways. We extended the previous natural
language processing analyses of online dating [Nag09]_ by combining natural
language processing with machine learning on a larger scale. We leveraged
multiple approaches including clustering and topic modeling, as well as feature
selection and modeling strategies. By exploring the relationship between free
text self-descriptions and demographics, we find that we can predict a user's
demographic makeup based on their user essays, and we discover some unexpected
insights into deception.

Data
----

Description
~~~~~~~~~~~

The data was obtained from: https://github.com/everett-wetchler/okcupid
[Wet15]_. Profile information from 59,000+ OkCupid users that were online on
06/30/2012. The users were within 25 miles of the San Francisco Bay Area. Each
profile contains the user’s height, body type, religion, ethnicity, religion,
as well as 10 essays on different topics. We used the 10 essays for text
analysis and the demographic data for trends.

Processing
~~~~~~~~~~

First, we identified essays that we wanted to run analysis on. The essay text
was cleaned by removing line break characters, urls, and stripping white
spaces. If a user wrote less than 5 words for the given essay, s/he was removed
from the analysis. Finally, stopwords and punctuation were removed.

The essay was tokenized using the happyfuntokenizer
(http://sentiment.christopherpotts.net/tokenizing.html), which is well suited
for online communication as it maintains emoticons as discrete tokens. Finally,
a tfidf matrix was created with unigrams, bigrams and trigrams, dropping words
with <0.01 document frequency.

To reduce cardinality, we combined demographic categories. For example,there
were 7,648 unique values for languages spoken, so we reduced the language
category to a number representing the number of languages listed by the user.

Methodology
-----------

Algorithms
~~~~~~~~~~

K-means
*******

K-means clustering.

The goal of any clustering approach is to find subgroups of observations in a
data set. Typically, we wish to find groups, or clusters, whose members are
similar to each other.

K-means splits a data set into k non-overlapping clusters. Each cluster is
described by its "centroid," which corresponds to the mean value of the
observations in the cluster.

The first step with K-means is to choose the initial centroids. One way to do
this is to randomly select k data set observations. However, centroids do not
need to correspond to actual data points. The next step is to assign every
observation to its nearest centroid. The groups of observations associated with
each centroid constitute a cluster. Cluster centroids are then updated by
finding the new mean value of each cluster. Finally, observations are
reassigned to the nearest cluster centroid. This continues until cluster
assignments no longer change or until the change is below some specified
threshold.

Choosing k and defining a similarity metric&mdash;recall that the goal with
clustering is to find *similar* groups&mdash;are important considerations.

NMF
***

Non-negative matrix factorization.

For document clustering, the document corpus can be projected onto a
"k-dimensional semantic space," with each axis corresponding to a particular
topic and each document being represented as a linear combination of those
topics [Xu_03]_. With methods such as latent semantic indexing, the derived
latent semantic space is orthogonal. Because of this, these type of methods
can't easily handle cases where corpus topics overlap, as can often be the
case. NMF, on the other hand, because the derived latent semantic space is not
required to be orthogonal, can find directions for related or overlapping
topics.

We applied NMF to each essay of interest using Scikit-Learn version 0.16, which
uses the projected gradient solver [Lin07]_. NMF utilizes document frequency
counts, so we calculated the tfidf matrix for unigrams, bigrams, and trigrams.
We limited the tokens to those that appeared in at least 1% of the documents.
We calculated NMF with k = 25, which factorized the tfidf matrix into two
matrices, W and H. The dimensions are n_samples x 25 and 25 x n_features for W
and H, respectively. Group descriptions were given by top-ranked terms (the
most distinctive) in the columns of H. Document membership weights were given
by the rows of W. We calculated the maximum value in each row of W to determine
essay group membership. We chose to have 25 groupings somewhat arbitrarily,
though we did try using cosine similarity measures to determine when the
groupings were the most dissimilar.

Initial Approach
~~~~~~~~~~~~~~~~

We initially proposed using LWIC to featurize the data as was done in numerous
previous studies of online dating [Nag09]_ [Tom12]_ [Bon05]_. However, a more
recent study analyzing a much larger dataset [Sch13]_ reported that using an
"open" vocabulary resulted in more generalizable results than the "closed" set
available through LIWC. Instead of using LIWC, we created and selected our
features using unigram, bigram, and trigram tokens from the combined text
across the 10 essays for each user. The bigrams and trigram tokens were
selected using their PMI, calculated as the frequency of the phrase divided by
the product of the individual words in that phrase. Bigrams were restricted to
those with a PMI > 4 and trigrams were restricted to those with a PMI > 6, as
described in Schwartz et al. (2013). We restricted all tokens to those that
appeared in at least 1% of the documents.

After tokenizing the data, we applied PCA on the resulting features. The data
was noisier than we anticipated, and required 50 principal components to
account for 48% of the variance (Fig 1). PCA performed much worse when we
scaled the data values. Whitening the data did not have an effect. We had also
intended to apply varimax rotation in order to have a clearer interpretation of
the principal components, but after clustering on the reduced dimensions we
decided that it would not have that great of an effect.

We subsequently applied k-means clustering to the users in the reduced feature
space. To choose the number of clusters, we used the silhouette score, which
calculates a ratio using the mean intra-cluster distance and the mean
nearest-cluster distance for each sample. Because the dataset was so large, we
calculated the score on subsets of the data, creating a bootstrap estimate of
the score. We applied k-means clustering with 3, 5, 7, and 9 clusters, and
chose the number of clusters with the highest silhouette score.

After selecting the number of clusters based on the silhouette score, we
analyzed the distribution of users across clusters and found that some clusters
had very few users and that the overall distribution of users across clusters
was very unbalanced, with most clusters consisting of a single user. When
analyzing the essays in clusters of single users, we discovered that those
clusters corresponded to users who had written very little. As a result, we
decided to limit our analysis to users who had used at least 100 tokens across
their 10 essays.

Yet even with the additional preprocessing, dimensionality reduction, and
clustering, the groupings were sparse and uneven. We decided that PCA and
k-means clustering were probably not well-suited for our dataset. We attempted
using k-modes (from: https://github.com/nicodv/kmodes) instead of k-means, and
although the clusters were more evenly distributed (Fig 2), the costs were
extremely high.

Final Approach
~~~~~~~~~~~~~~

The initial essay preprocessing contributed a significant amount of noise to
our analysis. Specifically, while preprocessing, we combined all essays into a
single text block for each user. In grouping all essays together introduced
unnecessary noise into our analysis. Instead, we decided to focus our analysis
on two separate essays: "My Self Summary" and "Favorite Books, Movies, TV."

We began by exploring the lexical features of the text. Our goal was to
determine whether there existed inherent differences in writing styles by
demographic split. We considered essay length, the use of profanity and slang
terms, and part-of-speech usage.

When determining essay length, we used our tokenizer, as opposed to, say,
simply splitting the text on white-space. This was mainly to be consistent with
our downstream analyses, such as predictive modeling.

A list of profane words was obtained from the "Comprehensive Perl Archive
Network" website. Slang terms include words such as "dough," which refers to
money, and acronyms like "LMAO," which stands for "laughing my ass off." These
terms come from the Wiktionary Category:Slang page. Note that there is overlap
between the profane and slang lists.

Finally, we were interested in whether there were differences in the types of
words used by different groups of individuals. For example, do certain
users tend to use verbs ("action" words) more often than other groups of users?
To answer questions like these, we first had to associate parts of speech (also
known as "lexical categories") with each term (or "token") in our corpus. To do
this, we used spaCy's part-of-speech tagger. We use spaCy's "coarse-grained"
tags, of which there are 19, in order to maintain low-cardinality. These tags
expand upon Petrov, Das, and McDonald's "universal part-of-speech tagset."

In addition to lexical characteristics, we were interested in understanding
the semantics of the text. To do this, we used non-negative matrix
factorization (NMF) in order to find latent (or hidden) structure in the text.
This structure is in the form of "topics" or "clusters" which can be described
by particular tokens. With this, we then examined the distribution of users
across clusters by demographic split. The idea was to determine whether
particular groups of users were more likely to write about particular topics or
themes in their essays.

After running NMF to cluster users, we ran a keyword analysis on the essays for
each resulting group. The keyword algorithm takes the 1000 most frequent
unigrams and extracts hypernyms from them using WordNet. After the hypernyms
are calculated, it uses examples of these hypernyms as seeds to find contextual
4-grams. It then filters the 4-grams to keep only those that occur more than 20
times. While the keywords adequately summarized and described the terms in a
given cluster, they were not sufficiently distinct between clusters because the
most frequent words were used across many groups. Instead we focused our next
analyses and visualizations on words which defined differences between groups
(that would characterize one cluster relative to the others).

Results
-------

Our first attempt at visualizing tokens that make the groupings distinctive was
to plot a histogram of the difference in relative frequency between tokens for
a given group versus other groups (Fig 3). This can be done between clusters
(comparing a single cluster to all of the other clusters) or within clusters
(comparing a single demographic group to all other groups). This visualization
doesn’t penalize infrequent tokens like word clouds do, but it was more
difficult to view the individual words.

In order to determine if there were demographic differences in the distribution
of users across groups, we calculated the percentage of each demographic
represented in each cluster (or NMF group), normalized by the total number of
people in that particular demographic group. We visually examined the clusters
and noted the ones dominated by a particular demographic group. For example,
in the "Favorite books, movie, TV" essay, we noted that group 7 was dominated
by people identifying as black relative to other ethnic groups (Fig 5, left).

An added benefit of plotting the distribution of cluster membership is that we
can determine how much certain themes are being written about for a particular
essay. For example, in Fig 5 above, group 6 has a low number of users across
all demographic splits, indicating that relatively few individuals wrote about
the theme associated that cluster.

We were also able to split and combine demographic groups, for example gender
and sexual orientation (Fig 6). For the “Self Summary” essay we noted that
clusters 1 and 6 were dominated by gay men, while group 2 was dominated by
women, both gay and straight. Upon examining the word clouds for these
demographic splits we noted that cluster 1 discussed San Francisco and the act
of moving or locations and changing. The word cloud for group 6 was represented
by n-grams discussing searching for a relationship and types of relationships.
This is in contrast to cluster 2, which was dominated by women and whose word
cloud seemed to encompass stereotypical cliches.

We also created superordinate groupings. For example, for the "Favorite book,
movies, TV" essay, clusters 2, 15, and 24 were all characterized by n-grams
referring to movies. We decided to combine those groups into a superordinate
"movie" grouping and look at its distribution across males and females (Fig 7).
We found the male cluster dominated by Star Wars, Fight Club, and Lord of the
Rings, while the female cluster was dominated by love, Harry Potter, Hunger
Games, Little Miss Sunshine, and Pride and Prejudice.

We were able to explore several other demographic splits. The most interesting
was related to drug usage. Fig 8 shows the distribution of the drug usage
categories (yes, no, and unknown) for each cluster. Interestingly, individuals
in the "unknown" category—those for whom we had no information—closely followed
the self-identified drug users. That is, when the proportion of drug users in a
particular cluster was high (or low), so we was the proportion of unknown drug
users. This was most distinctive when the separation between drug users and
non-drug users was large (e.g., groups 3, 7, 21, and others). Our hypothesis is
that it is possible that individuals who do not respond to the drug usage
question might not want to admit that they use drugs.

Future Work
-----------

Future

Conclusion
----------

Conclude

Acknowledgements
----------------
Acknowledge

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Pew15] 5 Facts About Online Dating.

.. [Fio08] Assessing Attractiveness in Online Dating Profiles.

.. [McP01] Birds of a feather: Homophily in social networks.

.. [Fio05] Homophily in Online Dating: When Do You Like Someone Like Yourself?.

.. [Nag09] Nagarajan and Hearst, An Examination of Language Use in Online Dating Profiles, 2009

.. [Wet15] Everett Wetchler, okcupid, (2015), GitHub repository,
           `<https://github.com/everett-wetchler/okcupid.git>`_

.. [Xu_03] Document clustering based on non-negative matrix factorization.

.. [Lin07] Projected gradient methods for non-negative matrix factorization.

.. [Tom12] What lies beneath: The linguistic traces of deception in online
           dating profiles.

.. [Bon05] Language of lies in prison: Linguistic classification of prisoners'
           truthful and deceptive natural language.

.. [Sch13] Personality, gender, and age in the language of social media: The
           open-vocabulary approach.
