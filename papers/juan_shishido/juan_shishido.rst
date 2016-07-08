:author: Juan Shishido
:email: juanshishido@berkeley.edu
:institution: School of Information, University of California, Berkeley
:corresponding:

:author: Jaya Narasimhan
:email: jnaras@berkeley.edu
:institution: Department of Electrical Engineering and Computer Science, University of California, Berkeley
:equal-contributor:

:author: Matar Haller
:email: matar@berkeley.edu
:institution: Helen Wills Neuroscience Institute, University of California, Berkeley
:equal-contributor:

----------------------------------------------------------
Tell Me Something I Don't Know: Analyzing OkCupid Profiles
----------------------------------------------------------

.. class:: abstract

In this paper, we present an analysis of 59,000 OkCupid user profiles that
examines online self-presentation by combining natural language processing
(NLP) with machine learning. We analyze word usage patterns by self-reported
sex and drug usage status. In doing so, we review standard NLP techniques,
cover several ways to represent text data, and explain topic modeling. We find
that individuals in particular demographic groups self-present in consistent
ways. Our results also suggest that users may unintentionally reveal
demographic attributes in their online profiles.

.. class:: keywords

   natural language processing, machine learning, supervised learning,
   unsupervised learning, topic modeling, okcupid, online dating

Introduction
------------

Online dating has become a common and acceptable way of finding mates. 
Almost half of Americans know someone who uses
or who has met a partner through online dating and 59% believe online dating is
a good way to meet people [Pew16]_. Online dating sites or mobile dating apps
are used by 27 percent of 18-24 year olds, 22 percent of 25-34 year olds, 21
percent of 35-44 year olds, 13 percent of 45-54 year olds, and 12 percent of
55-64 year olds [Pew16]_. Given the popularity of online dating, the way that people 
self-present online has broad implications for the relationships they pursue. 

Previous studies suggest that the free-text portion of online dating profiles
is an important factor (after photographs) for assessing attractiveness
[Fio08]_. The principle of homophily posits that people tend to
associate and bond with similar others and that this principle strongly
structures social networks and ties, most prominently by race and ethnicity
[McP01]_. Perhaps not surprisingly, research suggests that homophily extends to
online dating, with people seeking mates similar to themselves [Fio05]_.
However, it remains unclear whether people within particular demographic groups,
such as sex or ethnicity, self-present in similar ways when searching for a
mate online.

In this paper, we analyze demographic trends in online self-presentation.
Specifically, we focus on whether people signal demographic characteristics
through the way they present themselves online. We extend previous natural
language processing analyses of online dating [Nag09]_ by combining NLP
with supervised and unsupervised machine learning on a larger scale. We
leverage multiple approaches including clustering and topic modeling as well as
feature selection and modeling strategies. By exploring the relationships
between free-text self-descriptions and demographics, we discover that we can
predict users' demographic makeup and also find some unexpected insights into
unintentional demographic signals in free-text writing.

Code and data for this work are available in our ``okcupid`` GitHub repository [1]_.

Data
----

Description
~~~~~~~~~~~

The data was obtained from Albert Y. Kim's JSE_OkCupid repository [2]_. Profile
information was available for 59,946 OkCupid users that were members as of
06/26/2012, lived within 25 miles of San Francisco, had been active in the
previous year, and had at least one photo in their profile [Wet15]_.
The data set contained free-text responses to 10 essay prompts as well as the
following user characteristics: age, body type, diet, drinking status, drug
usage status, education level, ethnicity, height, income, job type, location,
number of children, sexual orientation, attitude toward pets, religion, sex,
astrological sign, smoking status, number of language spoken, and relationship
status.

This data set was selected because its diverse set of essay prompts and because the
availability of detailed user characteristics provided ideal means for
examining the link between online self-presentation and demographics.
Previous work using this data was primarily exploratory, with some basic text analysis 
and with logistic regression to predict sex using only height [Kim15]_. 
The present study extends previous work by analyzing additional features and by
introducing novel analyses.

This data set is public, as authorized by OkCupid president and co-founder
Christian Rudder [Kim15]_.

Preprocessing
~~~~~~~~~~~~~

Line break characters and URLs were removed from the essay text. Multiple
periods, dashes, and white spaces were replaced by single instances. Essays
were tokenized (segmented) into individual words using spaCy's default
tokenizer, which is well suited for online communication as it maintains
emoticons as discrete tokens, and removes punctuation. Users who wrote less
than 5 words for a given essay were removed from the analysis.

We combined drug usage status levels. Specifically, users who responded
"sometimes" or "often" were grouped into a "yes" category. Individuals who
answered "never" were assigned to the "no" group and we created an "unknown"
category for users who did not answer.


Methods
-------

Term Frequency-Inverse Document Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Machine learning tasks require numerical inputs. There are several ways to
represent text as numerical feature vectors. Features typically correspond to
distinct tokens or to sequences of adjacent tokens. A token is a series of
characters, such as a word, that is treated as a distinct unit [Bir10]_.

One way to represent a corpus, or collection of text documents, is as a matrix
of token counts. This weights terms by their absolute frequencies. Often,
highly-weighted terms, such as "a" or "the," are not informative, so token
counts are weighted using term frequency-inverse document frequency (tf-idf).

Tf-idf is the product of the term frequency and the inverse document frequency.
The term frequency refers to the *relative* frequency of term :math:`t` in
document :math:`d`. The inverse document frequency is the log of the total
number of documents :math:`N` to the number of documents that contain term
:math:`t`.


Log-Odds-Ratio
~~~~~~~~~~~~~~

One metric for comparing word usage across groups is to calculate the
log-odds-ratio. The odds for word :math:`w` in the usage of group :math:`g`
are defined as :math:`O_{iw} = \frac{f_{iw}}{(1 - f_{iw})}` where :math:`f_{iw}`
is the frequency count of word :math:`w` normalized by total count of words
used by group :math:`i`. If a word is used only by one group, its
log-odds-ratio is infinite. Therefore, a constant is added to each frequency
when calculating the odds. The log of the ratio of the adjusted odds between
groups can then be used to compare word usage across groups. 


Non-negative Matrix Factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For document clustering, the document corpus is projected onto a
k-dimensional semantic space, with each axis corresponding to a particular
topic and each document being represented as a linear combination of those
topics [Xu_03]_. Methods such as latent semantic indexing require the derived
latent semantic space to be orthogonal, so this class of methods
does not work well when corpus topics overlap, as is often the case.
Conversely, non-negative matrix factorization (NMF) does not require the latent
semantic space to be orthogonal, and therefore is able to find directions for
related or overlapping topics.

NMF was applied to each essay of interest using scikit-learn (version 0.17.1),
which uses the coordinate descent solver. NMF utilizes document frequency
counts, so the tf-idf matrix for unigrams, bigrams, and trigrams was calculated,
while limiting tokens to those appearing in at least 0.5% of the documents
(minimum frequency). NMF was calculated with :math:`k` dimensions, which
factorized the tf-idf matrix into two matrices, :math:`W` and :math:`H`. The
dimensions were ``n_samples x k`` and ``k x n_features`` for :math:`W` and
:math:`H`, respectively. Group descriptions were given by top-ranked terms (the
most distinctive) in the columns of :math:`H`. Document membership weights were
given by the rows of :math:`W`. The maximum value in each row of :math:`W`
determined essay group membership.

Permutation Testing
~~~~~~~~~~~~~~~~~~~

Permutation tests provide an exact sampling distribution of a test statistic
under the null hypothesis [Ger12]_ by computing the test statistic
for every manner by which labels can be associated with the observed data. In
practice, permutations are rarely ever completely enumerated. Instead, the
sampling distribution is approximated by randomly shuffling the labels :math:`P`
times.

The likelihood of the observed test statistic is determined as the proportion
of times that the absolute value of the permuted test statistics are greater
than or equal to the absolute value of the observed test statistic. This is the
:math:`p`-value for a two-tailed hypothesis. Permutation-based methods can be
used to compare two samples or to assess the performance of classifiers [Oja10]_.

There are several advantages to using randomization to make inferences as
opposed to parametric methods. Permutation tests do not assume normality, do
not require large samples and "can be applied to all sorts of outcomes,
including counts, durations, or ranks" [Ger12]_.

Approach
--------

Our analyses focus on two demographic dimensions — sex and drug usage — and on
two essays — "My self summary" and "Favorite books, movies, shows, music, food."
These essays were selected because they were answered by most users. For
example, "The most private thing I am willing to admit" prompt was ignored by
32 percent of users. Other essays in this data set may provide additional
insight into self-presentation and will be analyzed in the future.

We began by exploring the lexical features of the text as a way to determine
whether there were differences in writing styles by demographic split. We
considered essay length, the use of profanity and slang terms, and
part-of-speech usage. 

Essay length was determined based on the tokenized essays. A list of profane
words was obtained from the "Comprehensive Perl Archive Network" website. Slang
terms include words such as "dough," which refers to money, and acronyms like
"LOL." These terms come from the Wiktionary Category:Slang page [3]_. Note that
there is overlap between the profane and slang lists.

Each token in the corpus was associated with a lexical category using spaCy's
part-of-speech tagger. spaCy supports 19 coarse-grained tags [4]_ that expand
upon Petrov, Das, and McDonald's universal part-of-speech tagset [Pet11]_.

Differences in lexical features by demographic were analyzed using permutation
testing. We first compared average essay length by sex. Next, we examined
whether the proportion of females using profanity was different than the
proportion of males using such terms. The same was done for slang words.
Finally, we compared the average proportion of adjectives, nouns, and verbs and
identified the most distinctive terms in each lexical category by sex using the
smoothed log-odds-ratio, which accounts for variance.

Text semantics were also analyzed. The corpus was transformed into a tf-idf
matrix using spaCy's default tokenizer with punctuation removed. We chose to
include unigrams, bigrams, and trigrams [5]_. Stop words [6]_ and terms that
appeared in less than 0.5% of documents were removed. Stemming, the process of
of removing word affixes, was not performed.

Non-negative matrix factorization was used to identify latent structure in the
text. This structure was in the form of "topics" or "clusters" which were
described by particular tokens. In order to determine whether particular
demographic groups were more likely to write about particular topics, the
distribution of users across topics was calculated relative to each demographic
group. In cases where we are able to create superordinate groupings from NMF
topics — for example, by combining semantically similar clusters — we use the
log-odds-ratio to find distinctive tokens for the superordinate grouping.

Finally, we fit a logistic regression model to predict drug usage status for
users in the "Unknown" drug usage category.

Results
-------

In this section, we describe our findings. We start with a discussion of our
lexical-based analyses before discussing semantic-based results. Lexical-based
characteristics include essay length, use of profanity and slang terms, as well
as part-of-speech usage.

We first compared lexical-based characteristics on the self-summary text by sex.
Our sample included 21,321 females and 31,637 males [7]_. On average, females wrote significantly 
longer essays than males (150 terms compared to 139, :math:`p` < 0.001).

We next compared the proportion of users who utilized profanity and slang. Profanity was
rarely used in the self-summary essay. Overall, only 6 percent of users
included profanity in their self-descriptions. The difference was not
significantly significant by sex (5.8% of females versus 6.1% of males,
:math:`p` = 0.14).

Not surprisingly, slang was much more prevalent (on a per-user basis) than
profanity. 56 percent of users used some form of slang in their self-summary
essays. Females used slang at a significantly lower rate than males (54% versus
57%, :math:`p` < 0.001).

To compare part-of-speech usage, we first associated part-of-speech tags with
every token in the self-summary corpus. This results in counts by user and
part-of-speech. Each user's counts were then normalized by the user's essay
length to account for essay length differences between users. Of the 19
possible part-of-speech tags, we focused on adjectives, nouns, and verbs.

.. table:: Proportion of part-of-speech terms used, by sex. Asterisks (``**``)
           denote statistically significant differences at the 0.01 level.

   +-------------------+--------+--------+
   | Part-of-Speech    | Female | Male   |
   +===================+========+========+
   | Adjectives ``**`` | 10.61% | 10.16% |
   +-------------------+--------+--------+
   | Nouns ``**``      | 18.65% | 18.86% |
   +-------------------+--------+--------+
   | Verbs             | 18.28% | 18.27% |
   +-------------------+--------+--------+

Females used significantly more adjectives than males, while males used
significantly more nouns than females (:math:`p` < 0.001 for both). There was no
difference in verb usage between the sexes (:math:`p` = 0.91).

In addition to part-of-speech usage, we explored specific terms associated
with parts-of-speech that were distinctive to a particular sex. We did this
using the log-odds-ratio. The 10 most-distinctive adjective, noun, and verb
tokens for each sex are summarized below.

   +----------------+----------------------------+----------------------------+
   | Part-of-Speech | Female                     | Male                       |
   +================+============================+============================+
   | Adjectives     | independent sweet my sassy | nice cool its that few     |
   |                | silly happy warm favorite  | interesting martial most   |
   |                | girly fabulous             | masculine more             |
   +----------------+----------------------------+----------------------------+
   | Nouns          | girl family who yoga men   | guy computer engineer      |
   |                | gal heels love dancing     | guitar sports software     |
   |                | friends                    | women video technology     |
   |                |                            | geek                       |
   +----------------+----------------------------+----------------------------+
   | Verbs          | love am laugh laughing     | m was play playing laid    |
   |                | dancing adore loving       | 'll working hit moved been |
   |                | dance appreciate being     |                            |
   +----------------+----------------------------+----------------------------+

NMF was then used to provide insights into the underlying topics that users chose to
describe themselves. Selecting the number of NMF components (topics to which
users are clustered) is an arbitrary and iterative process. For the
self-summary essay, we chose 25 components, which resulted in a diverse, but
manageable set of topics.

Several expected themes emerged. Many users chose to highlight personality
traits, for example "humor" or "easy-going," while others focused on describing
the types of activities they enjoyed. Hiking, traveling, and cooking were
popular choices. Others chose to mention what they were looking, whether that
be a long-term relationship, a friendship, or sex. Topics and the highest
weighted tokens for each are summarized in the table below. Note that topic names were hand-labeled.

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
   |                | originally                                              |
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
   |                | traveling, hanging, cooking, sports                     |
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

In order to determine whether there were differences in the topics that OkCupid
users chose to write about in their self-summaries, we plotted the distribution
over topics by demographic split. This allowed us to identify if specific
topics were distinct to particular demographic groups.

The following figure shows the distribution over topics by sex for the
self-summary essay. The highest proportion of users, of either sex, were in the
"about me" topic. This is not surprising given the essay prompt. For most
topics, females and males were mostly evenly distributed. One notable
exception was with the "enthusiastic" topic, to which females belonged at
almost twice the rate of males. Users in this group used modifiers such as,
"love," "really," and "absolutely" regardless of the activities they were
describing.

.. figure:: self-summary-sex.png

   Self-Summaries

We further examined online self-presentation by considering the other available
essays in the OkCupid data set. Previous psychology research suggests that
people's preferred music styles are tied to their personalities [Col15]_, and it
is possible that this extends to other media, such as books or movies. We next
analyzed the "Favorite books, movies, shows, music, food" essay.

As with the self-summaries, we removed users who wrote less than 5 tokens for
this essay (11,836 such cases). Note that because the favorites text is less
expository and more list-like, we did not perform a lexical-based analysis.
Instead, we used NMF to identify topics (or genres). Like with the
self-summaries, we chose 25 topics. The following table lists the topics and a
selection of their highest weighted tokens.

   +----------------+---------------------------------------------------------+
   | Topic          | Tokens                                                  |
   +================+=========================================================+
   | like           | like, music like, movies like, really like, stuff,      |
   |                | food like, things, like music, books like, like movies  |
   +----------------+---------------------------------------------------------+
   | TV-hits        | mad men, arrested development, breaking bad, 30 rock,   |
   |                | tv, parks, sunny, wire, dexter, office                  |
   +----------------+---------------------------------------------------------+
   | enthusiastic   | love food, love music, love movies, love love, cook,    |
   |                | love good, eat, food, love read, books love             |
   +----------------+---------------------------------------------------------+
   | favorite-0     | favorite, favorite food, favorite movies,               |
   |                | favorite books, favorite music, favorite movie,         |
   |                | favorite book, favorite shows, favorite tv,             |
   |                | time favorite                                           |
   +----------------+---------------------------------------------------------+
   | genres-movies  | sci fi, action, comedy, horror, fantasy, movies, drama, |
   |                | romantic, classic, adventure                            |
   +----------------+---------------------------------------------------------+
   | genres-music   | hip hop, rock, r&b, jazz, reggae, rap, pop, country,    |
   |                | classic, old                                            |
   +----------------+---------------------------------------------------------+
   | misc-0         | fan, reading, food 'm, right, 'm big, really,           |
   |                | currently, music 'm, just, open                         |
   +----------------+---------------------------------------------------------+
   | TV-comedies-0  | big bang theory, met mother, big lebowski, friends,     |
   |                | house, office, community, walking dead, new girl, bones |
   +----------------+---------------------------------------------------------+
   | genres-food    | italian, thai, mexican, food, indian, chinese,          |
   |                | japanese, sushi, french, vietnamese                     |
   +----------------+---------------------------------------------------------+
   | nots           | ca n't, watch, n't really, does, n't like, does n't,    |
   |                | think, eat, n't watch tv, n't read                      |
   +----------------+---------------------------------------------------------+
   | teen           | harry potter, hunger games, twilight, dragon tattoo,    |
   |                | pride prejudice, harry met sally, disney, vampire,      |
   |                | trilogy, lady gaga                                      |
   +----------------+---------------------------------------------------------+
   | everything     | books, movies, food, music, shows, country, dance,      |
   |                | action, lots, horror                                    |
   +----------------+---------------------------------------------------------+
   | movies-drama-0 | eternal sunshine, spotless mind, litte miss sunshine,   |
   |                | amelie, garden state, lost, life, beautiful,            |
   |                | lost translation, beauty                                |
   +----------------+---------------------------------------------------------+
   | time periods   | 80, let, good, 90, life, just, 70, world, time, man     |
   +----------------+---------------------------------------------------------+
   | avid           | read lot, time, watch, listen, recently, lately,        |
   |                | love read, watch lot, favorites, just read              |
   +----------------+---------------------------------------------------------+
   | misc-1         | list, just, long, ask, way, goes, things, try,          |
   |                | favorites, far                                          |
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
   | movies-drama-1 | fight club, shawshank redemption, pulp fiction,         |
   |                | fear loathing, peppers, red hot, vegas, american,       |
   |                | catcher rye, big lebowski                               |
   +----------------+---------------------------------------------------------+
   | kinds          | kinds music, love kinds, kinds food, kinds movies,      |
   |                | listen, different, country, foods, comedy, action       |
   +----------------+---------------------------------------------------------+
   | favorite-1     | favorite book, favorite movie, food, music, good, fav,  |
   |                | book read, reading, great, best                         |
   +----------------+---------------------------------------------------------+
   | novelty        | enjoy, new, types, trying, reading, things, foods,      |
   |                | types music, films, different                           |
   +----------------+---------------------------------------------------------+
   | TV-drama       | game thrones, ender 's game, walking dead, true blood,  |
   |                | series, currently, hunger games, dexter, song ice,      |
   |                | boardwalk empire                                        |
   +----------------+---------------------------------------------------------+
   | genres-books   | fiction, non fiction, science fiction, fiction books,   |
   |                | read non fiction, historical fiction, films, books,     |
   |                | documentaries, biographies                              |
   +----------------+---------------------------------------------------------+

The topics for this essay were less distinctive than the topics for the
self-summaries. In some cases, genres (or media) overlapped. For example, the
"TV-comedies-0" group included "The Walking Dead," which is a drama. There was
also overlap between groups. This might suggest that the number of NMF
components was too high, but the granularity these topics provided was used for
further analyses. We created superordinate groupings from the topics from which
we extracted distinctive tokens for particular demographic groups. We first
examined the distribution over topics by sex.

.. figure:: favorites-sex.png

   Favorites

The most popular topics, for both females and males, were "TV-hits" and
"music-rock," with about 16 percent of each sex writing about shows or artists
in those groups. We found more separation between the sexes in the favorites
essay than we did with the self-summaries. As with the self-summary essay, the enthusiastic group was
distinctly female. A distinctly male category included films such as "Fight
Club" and "The Shawshank Redemption" and musicians such as the Red Hot
Chili Peppers.

We created superordinate groupings by combining clusters. There were four
groups related to movies. In order to extract demographic-distinctive tokens,
we used the smoothed log-odds-ratio which accounts for variance as described by
Monroe, Colaresi, and Quinn [Mon09]_. The top movies for females were Harry
Potter, Pride & Prejudice, and Hunger Games while males favored Star Wars, The
Matrix, and Fight Club. The "movies-sci-fi" and "movies-drama-1" groups,
whose highest weighted tokens referred to the male-favored movies, had a higher
proportion of males than females. Similarly, the "teen" group, which
which corresponded to female-favored movies, had a higher proportion of females.
This reflects the terms found by the log-odds-ratio.

We next examined the distribution over topics by drug usage. In this
demographic category, users self-identified as drug users or non-drug users. To
this, we added a third level for users who declined the state their drug usage
status. There were 6,859 drug users, 29,402 non-drug users, and 11,849 users who did not state their drug usage status
("unknown").

There was more intra-cluster variation in the distribution of users across topics than for the demographic split by sex.
Interestingly, the distribution across topics of users for whom we had no drug
usage information — those in the "unknown" category — tended to track the
distribution of self-identified drug users. In other words, the
proportion of drugs users and unknown users in most topics was similar.
This was especially true in cases where difference in proportions of drug users
and non-drug users was large. This unexpected finding may suggest that
individuals who did not respond to the drug usage question abstained in order
to avoid admitting they did use drugs.

.. figure:: favorites-drugs.png

   Favorites

Although we were unable to test this hypothesis directly due to lack of
ground-truth drug-usage status for these users, the manner by which free-text
writing styles may unintentionally disclose demographic attributes is an
intriguing avenue for research. We used a predictive modeling approach to
attempt to gain insights into this question. Specifically, we trained a logistic
regression model on a binary outcome, using only drug users and non-drug users.
We used tf-idf weights on unigrams, bigrams, and trigrams as in the previous
analyses. We also balanced the classes by randomly sampling 6,859 accounts
from the non-drug user population. The data was split into training (80%) and
test (20%) sets in order to assess model accuracy. We then predicted class
labels on the group of unknown drug usage status.

Our initial model, which used only the "Favorites" essay text, accurately predicted
68.0 percent of drug users. When applied to the unknown users upon which the model was not trained,
the model predicted that 55 percent of the unknown users
were drug users and that 45 percent were not. When we examined the proportion
of predicted user by NMF cluster, however, we found intriguing patterns. In the
"music-rock" group — the group with the largest disparity between users and
non-users — 84 percent of unknowns were classified as drug users. In contrast,
only 25 percent of the unknowns in the "TV-comedies-0" group were classified as
such. While this cluster included "The Big Lebowski," which is identified as a
"stoner film" [She13]_, it also features "The Big Bang Theory," "How I Met Your
Mother," "NCIS," "New Girl," and "Seinfeld," which we would argue are decidedly
not drug-related.

These results prompted us examine if we could predict drug usage status
based on text alone. For this, we combined the text of all 10 essays and
dropped the 2,496 users who used less than five tokens in the full-text. As
before, we randomly sampled from the non-users in order to balance the classes
and split the data into training and test sets.

The full-text model accuracy increased to 72.7 percent. We used the feature
weights to find the 25 most-predictive drug-usage terms. These are listed below.

::

  sex, shit, music, weed, party, beer, dubstep, fuck,
  drinking, smoking, partying, chill, hair, park,
  fucking, dj, burning, electronic, drunk, ass,
  reggae, robbins, dude, smoke, cat

Drug users in this data set reference drinking, smoking, partying, and music
more than non-users and also use particular profane terms.

Conclusion
----------

The current study extended previous NLP analyses of online dating profiles. The
scope of this work was larger than previous studies, both because of the size
of the datasets (previous studies used much smaller datasets) and because of
the novel combination of NLP with both supervised and unsupervised machine
learning techniques (NMF and logistic regression). To our knowledge, there is currently no study that
combines these techniques to identify unintentional cues in online
self-presentation as well as predict demographics from free-text self
descriptions. The idea that people may unintentionally be providing information
about themselves in the way that they answer questions online is an intriguing
avenue for future research and can also be extended to deception online.

Acknowledgements
----------------

This work began as a final project for the Applied Natural Language Processing
course at the School of Information at the University of California, Berkeley.
We would like to thank Marti Hearst for her guidance in the "right" way to do
NLP and in pushing us to explore new and exciting data sets. We would also like
to thank David Bamman for fruitful discussions on NLP and ideas for permutation
testing. We are also thankful for our reviewer's time and effort. Their
comments and suggestions were invaluable, helping us organize our
thoughts and analyses.

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

Footnotes
---------
.. [1] https://github.com/juanshishido/okcupid. Jupyter notebook with analysis
       results: https://github.com/juanshishido/okcupid/blob/master/OkNLP-paper.ipynb

.. [2] https://github.com/rudeboybert/JSE_OkCupid. Our original data source was
       Everett Wetchler's okcupid repository (https://github.com/everett-wetchler/okcupid).
       However, after commit ``0d62e62``, in which the data was "fully
       anonimized" to exclude essays, we switched to Kim's repository. As far
       as we can tell, this data set is the same as the Wetchler original.

.. [3] https://simple.wiktionary.org/wiki/Category:Slang.

.. [4] https://spacy.io/docs#token-postags.

.. [5] Unigrams are single tokens. Bigrams refer to two adjacent and trigrams
       to three adjacent tokens.

.. [6] Stop words are words that appear with very high frequency, such as "the"
       or "to."

.. [7] The difference between the number of users in the data set and the
       number of users in the analysis is due to the fact that we drop users
       that write less than five tokens for a particular essay.

References
----------
.. [Bir10] Bird, S., Klein, E., & Loper, E. (2009). Natural language processing
           with Python. "O'Reilly Media, Inc.".

.. [Col15] Collingwood, J. (2015). Preferred Music Style Is Tied to Personality.
           Psych Central. Retrieved on June 22, 2016, from
           http://psychcentral.com/lib/preferred-music-style-is-tied-to-personality/

.. [Fio05] Fiore, A. T., & Donath, J. S. (2005, April). Homophily in online
           dating: when do you like someone like yourself?. In CHI'05 Extended
           Abstracts on Human Factors in Computing Systems (pp. 1371-1374). ACM.

.. [Fio08] Fiore, A. T., Taylor, L. S., Mendelsohn, G. A., & Hearst, M. (2008,
           April). Assessing attractiveness in online dating profiles. In
           Proceedings of the SIGCHI Conference on Human Factors in Computing
           Systems (pp. 797-806). ACM.

.. [Ger12] Gerber, A. S., & Green, D. P. (2012). Field experiments: Design,
           analysis, and interpretation. WW Norton.

.. [Kim15] Kim, A. Y., & Escobedo-Land, A. (2015). OkCupid Data for Introductory
           Statistics and Data Science Courses. Journal of Statistics Education,
           23(2), n2.

.. [McP01] McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a
           feather: Homophily in social networks. Annual review of sociology,
           415-444.

.. [Mon09] Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin'words:
           Lexical feature selection and evaluation for identifying the content
           of political conflict. Political Analysis, 16(4), 372-403.

.. [Nag09] Nagarajan, M., & Hearst, M. A. (2009, March). An Examination of
           Language Use in Online Dating Profiles. In ICWSM.

.. [Oja10] Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying
           classifier performance. Journal of Machine Learning Research,
           11(Jun), 1833-1863.

.. [Pet11] Petrov, S., Das, D., & McDonald, R. (2011). A universal part-of-speech
           tagset. arXiv preprint arXiv:1104.2086.

.. [Pew16] Smith, Aaron, & Anderson, Monica (2016). 5 Facts About Online Dating.
           Retrieved from http://www.pewresearch.org/fact-tank/2016/02/29/5-facts-about-online-dating/.

.. [She13] Sheffield, Rob (2013). 10 Best Stoner Movies of All Time. Rolling
           Stones. Retrieved on June 23, 2016, from
           http://www.rollingstone.com/movies/lists/the-greatest-stoner-movies-of-all-time-20130606

.. [Wet15] Everett Wetchler, okcupid, (2015), GitHub repository,
           `<https://github.com/everett-wetchler/okcupid.git>`_

.. [Xu_03] Xu, W., Liu, X., & Gong, Y. (2003, July). Document clustering based
           on non-negative matrix factorization. In Proceedings of the 26th
           annual international ACM SIGIR conference on Research and
           development in informaion retrieval (pp. 267-273). ACM.
