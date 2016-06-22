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

In this section, we describe our findings. We start with a discussion of our
lexical-based analyses before discussing our semantic-based results.
Lexical-based characteristics include essay length, use of profanity and slang
terms, as well as part-of-speech usage. Our analyses focus on two demographic
dimensions—sex and drug usage—and on two essays—"my self summary" and "favorite
books, movies, shows, music, food."

We first compare lexical-based characteristics on the self-summary text by sex.
Our sample includes 21,321 females and 31,637 females. Note that the difference
between this and the 59,946 users in the data set is due to our dropping users
with less than five tokens in a particular essay. We find that, on average,
females write just under 150 words (tokens, actually) compared to males' 139,
though the variance is higher for the males than the females. This difference
is statistically significant.

For profanity and slang, instead of comparing frequencies across demographic
splits, we compare the proportion of users who use these terms.

In the self-summary essay, profanity is rarely used. Overall, only 6% of users
include such terms in their descriptions. 5.8% of females use profanity in
their self-summaries compared to 6.1% of males. This difference is not
statistically significant.

Not surprisingly, slang is much more prevalent (on a per-user basis) than
profanity. 56% of users use some form of slang in their self-summary essays.
Females use slang at a lower rate than males—54% vs. 57%—a difference that is
statistically significant.

In order to compare part-of-speech usage, we first associate part-of-speech
tags with every token in the self-summary corpus. This results in counts by
user and tag. Because of the difference in essay length we saw above, we
normalize these values based on the essay length. For example, if, out of 100
tokens, a particular user uses 25 verbs, a value of 0.25 would be associated
with the verb tag for that user. Of the 15 possible tags, we focused on three:
adjectives, nouns, and verbs. This is summarized in the following table.

   +----------------+--------+--------+-------------+
   | Part-of-Speech | Female | Male   | Significant |
   +================+========+========+=============+
   | Adjectives     | 10.61% | 10.16% | *           |
   +----------------+--------+--------+-------------+
   | Nouns          | 18.65% | 18.86% | *           |
   +----------------+--------+--------+-------------+
   | Verbs          | 18.28% | 18.27% |             |
   +----------------+--------+--------+-------------+

We find that, in the self-summary essay, females use more adjectives than
males do. For nouns, it's the other way around. Interestingly, neither sex uses
verbs more often than the other.

In addition to part-of-speech usage, we can explore particular terms associated
with parts-of-speech that are distinctive to a particular group. We do this by
comparing relative token frequencies. The 15 most-distinctive adjective,
noun, and verb tokens, by sex, are summarized below.

   +----------------+----------------------------+----------------------------+
   | Part-of-Speech | Female                     | Male                       |
   +================+============================+============================+
   | Adjectives     | my happy independent       | that nice more few other   |
   |                | favorite sweet silly       | most its cool interesting  |
   |                | important passionate warm  | your easy good which must  |
   |                | amazing beautiful          | last                       |
   |                | adventurous creative loyal |                            |
   |                | social                     |                            |
   +----------------+----------------------------+----------------------------+
   | Nouns          | who girl family friends    | guy sports music something |
   |                | love someone life person   | francisco women what       |
   |                | yoga heart men wine things | guitar video computer      |
   |                | adventures dancing         | stuff games years company  |
   |                |                            | name                       |
   +----------------+----------------------------+----------------------------+
   | Verbs          | love am laugh have being   | was 's been m 've 'll play |
   |                | are loving travel be       | moved working get playing  |
   |                | laughing appreciate        | 'm like know laid          |
   |                | traveling dancing          |                            |
   |                | exploring loves            |                            |
   +----------------+----------------------------+----------------------------+

We use NMF to help us understand the subject matter that users find interesting
and important about themselves and, thus, choose to write about. This provides
insight into the way they choose to self-present. In addition to particular
themes, NMF also allows us to consider stylistic expression. Choosing the
number of NMF components—these can be thought of as topics—is an arbitrary and
iterative process. For the self-summary essay, we chose to start with 25.

Several expected themes emerged. Some users, for example, chose to highlight
personality traits. Some did so by mentioning specific characteristics such as
humor while others were less specific, mentioning phrases such as, "easy going."
Other users focused on describing the types of activities they enjoyed. Hiking,
traveling, and cooking were popular choices. Others chose to mention what they
were looking for, whether that be a long-term relationship, a friendship, or
sex. Topics and a selection of their highest weighted tokens are summarized in
the table below. (Note that a complete list of the 50 highest weighted tokens
for each topic is available in the appendix.)

   +----------------+---------------------------------------------------------+
   | Topic          | Tokens                                                  |
   +================+=========================================================+
   | meet & greet   | meet new people, looking meet new, love meeting new,    |
   |                | new friends, enjoy meeting, interesting people,         |
   |                | want meet, 'm new, people love, experiences             |
   +----------------+---------------------------------------------------------+
   | the city       | san francisco, moved san francisco, city,               |
   |                | living san francisco, just moved san, native,           |
   |                | san diego, grew, originally, recently                   |
   +----------------+---------------------------------------------------------+
   | enthusiastic   | love travel, love laugh, love outdoors, love love,      |
   |                | laugh, dance, love cook, especially, life love,         |
   |                | love life                                               |
   +----------------+---------------------------------------------------------+
   | straight talk  | know, just, want, ask, message, just ask, really,       |
   |                | talk, write, questions                                  |
   +----------------+---------------------------------------------------------+
   | about me       | 'm pretty, 'm really, 'm looking, 'm just, say 'm,      |
   |                | think 'm, 'm good, 'm trying, nerd, 'm working          |
   +----------------+---------------------------------------------------------+
   | novelty        | new things, trying new, trying new things, new places,  |
   |                | learning new things, exploring, restaurants,            |
   |                | things love, love trying, different                     |
   +----------------+---------------------------------------------------------+
   | seeking        | 'm looking, guy, relationship, looking meet, share,     |
   |                | woman, nice, just looking, man, partner                 |
   +----------------+---------------------------------------------------------+
   | carefree       | easy going, 'm easy going, easy going guy,              |
   |                | pretty easy going, laid, love going, enjoy going,       |
   |                | simple, friendly, likes                                 |
   +----------------+---------------------------------------------------------+
   | casual         | guy, lol, chill, nice, old, pretty, alot, laid, kinda,  |
   |                | wanna                                                   |
   +----------------+---------------------------------------------------------+
   | enjoy          | like, 'd like, things like, really like, n't like,      |
   |                | feel like, stuff, like people, like going, watch        |
   +----------------+---------------------------------------------------------+
   | transplant     | moved, sf, years ago, school, east coast, city,         |
   |                | just moved, college, went, california                   |
   +----------------+---------------------------------------------------------+
   | nots           | n't, ca n't, does n't, really, wo n't, n't like,        |
   |                | n't know, n't really, did n't, probably                 |
   +----------------+---------------------------------------------------------+
   | moments        | spend time, good time, lot, free time, spending time,   |
   |                | lot time, spend lot, time friends, time 'm, working     |
   +----------------+---------------------------------------------------------+
   | personality    | humor, good sense humor, good time, good conversation,  |
   |                | sarcastic, love good, dry, good company, appreciate,    |
   |                | listener                                                |
   +----------------+---------------------------------------------------------+
   | amusing        | fun loving, 'm fun, having fun, outgoing, guy, girl,    |
   |                | adventurous, like fun, looking fun, spontaneous         |
   +----------------+---------------------------------------------------------+
   | review         | let 's, think, way, self, right, thing, say, little,    |
   |                | profile, summary                                        |
   +----------------+---------------------------------------------------------+
   | region         | bay area, moved bay area, bay area native, grew,        |
   |                | living, 'm bay area, east bay, raised bay area, east,   |
   |                | originally
   +----------------+---------------------------------------------------------+
   | career-focused | work hard, play hard, hard working, progress, harder,   |
   |                | job, try, love work, company, busy                      |
   +----------------+---------------------------------------------------------+
   | locals         | born, raised, born raised, california, raised bay area, |
   |                | college, school, sf, berkeley, oakland                  |
   +----------------+---------------------------------------------------------+
   | unconstrained  | open minded, creative, honest, relationship,            |
   |                | adventurous, curious, passionate, intelligent, heart,   |
   |                | independent                                             |
   +----------------+---------------------------------------------------------+
   | active         | enjoy, friends, family, hiking, watching, outdoors,     |
   |                | travelling, hanging, cooking, sports                    |
   +----------------+---------------------------------------------------------+
   | creative       | music, art, live, movies, live music, play, food,       |
   |                | games, dancing, books                                   |
   +----------------+---------------------------------------------------------+
   | carpe diem     | live, world, fullest, enjoy life, experiences,          |
   |                | passionate, love life, moment, living life, life short  |
   +----------------+---------------------------------------------------------+
   | cheerful       | person, people, make, laugh, think, funny, kind, happy, |
   |                | honest, smile                                           |
   +----------------+---------------------------------------------------------+
   | jet setter     | 've, lived, years, world, traveled, year, spent,        |
   |                | countries, different, europe                            |
   +----------------+---------------------------------------------------------+

In order to determine whether there are differences in the topics or themes
that OkCupid users choose to write about in their self-summaries, we plot the
distribution over topics by demographic split. This allows us to identify how
often certain themes are being written about and whether those themes are
distinct to particular demographic groups.

The following figure shows the distribution over topics by sex. We see that
the highest proportion of users, of either sex, are in the "about me" group.
This is not surprising given that we're analyzing the self-summary essays. For
most topics, the proportion of females and males is faily even. One notable
exception is with the "enthusiastic" group, which females belong to at almost
twice the rate of males. Users in this group use modifiers such as, "love,"
"really," and "absolutely" regardless of the activities they are describing.

We can further examine online self-presentation by considering the other
available essays in the OkCupid data set. It has been noted that, "people do
actually define themselves through music and relate to other people through
it" [Col15]_. It is possible that this extends to other media, such as books or
movies, too. We consider the "favorite books, movies, shows, music, food" essay
next. Note that because the favorites text is less expository and more
list-like, we do not consider a lexical-based analysis. Instead, we use NMF to
identify themes (or genres). Like with the self-summaries, we choose 25 topics.
The following table lists the topics and a selection of their highest weighted
tokens.

   +----------------+---------------------------------------------------------+
   | Topic          | Tokens                                                  |
   +================+=========================================================+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   | TV-hits        | mad men, arrested development, breaking bad, 30 rock,   |
   |                | tv, parks, sunny, wire, dexter, office                  |
   +----------------+---------------------------------------------------------+
   | enthusiastic   | love food, love music, love movies, love love, cook,    |
   |                | love good, eat, food, love read, books love             |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   | genres-movies  | sci fi, action, comedy, horror, fantasy, movies, drama, |
   |                | romantic, classic, adventure                            |
   +----------------+---------------------------------------------------------+
   | genres-music   | hip hop, rock, r&b, jazz, reggae, rap, pop, country,    |
   |                | classic, old                                            |
   +----------------+---------------------------------------------------------+
   | TV-comedies-0  | big bang theory, met mother, big lebowski, friends,     |
   |                | house, office, community, walking dead, new girl, bones |
   +----------------+---------------------------------------------------------+
   | genres-food    | italian, thai, mexican, food, indian, chinese,          |
   |                | japanese, sushi, french, vietnamese                     |
   +----------------+---------------------------------------------------------+
   | teen           | harry potter, hunger games, twilight, dragon tattoo,    |
   |                | pride prejudice, harry met sally, disney, vampire,      |
   |                | trilogy, lady gaga                                      |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   | movies-drama   | eternal sunshine, spotless mind, litte miss sunshine,   |
   |                | amelie, garden state, lost, life, beautiful,            |
   |                | lost translation, beauty                                |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   | music-rock     | david, black, john, tom, radiohead, bob, brothers,      |
   |                | beatles, black keys, bowie                              |
   +----------------+---------------------------------------------------------+
   | movies-sci-fi  | star, lord, wars, rings, star trek, trilogy, series,    |
   |                | matrix, princess, bride                                 |
   +----------------+---------------------------------------------------------+
   | TV-comedies-1  | modern family, family guy, office, south park,          |
   |                | met mother, glee, simpsons, american dad, 30 rock,      |
   |                | colbert                                                 |
   +----------------+---------------------------------------------------------+
   |                | fight club, shawshank redemption, pulp fiction,         |
   |                | fear loathing, peppers, red hot, vegas, american,       |
   |                | catcher rye, big lebowski                               |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+
   |                |                                                         |
   +----------------+---------------------------------------------------------+

The favorites topics are more difficult to categorize than the self-summaries.
In some cases, genres (or media) overlap. For example, in the TV-comedies-0
group, "The Walking Dead," a drama, is listed. In other cases, we see groups
that are potentially similar. However, it is possible that these groups (e.g.,
the multiple TV comedies groups) are, indeed, different, even if only slightly.
This might suggest that the number of NMF components is too high, but we prefer
the granularity it provides. In fact, we'll show that we are able to create
subordinate groupings from the above topics from which we can extract
distinctive tokens and compare demographic splits. We'll begin by examining the
distribution over topics by sex.

The most popular topics, for both females and males, are "TV-hits" and
"music-rock," with about 16% of each sex writing about shows or artists in
those groups. We see more separation between the sexes in the favorites essay
than we did with the self-summaries. The enthusiastic group is, again,
distinctly female. A distinctly male category includes films such as "Fight
Club" and "The Shawshank Redemption" and musical artists such as the Red Hot
Chili Peppers.

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

.. [Col15] Collingwood, J. (2015). Preferred Music Style Is Tied to Personality.
           Psych Central. Retrieved on June 22, 2016, from
           http://psychcentral.com/lib/preferred-music-style-is-tied-to-personality/
