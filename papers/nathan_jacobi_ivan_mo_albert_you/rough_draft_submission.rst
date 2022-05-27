1. **Abstract**

Human languages’ semantics and structure constantly change over time
through mediums such as culturally significant events. By viewing the
semantic changes of words during notable events, contexts of existing
and novel words can be predicted for similar, current events. By
studying the initial outbreak of a disease and the associated semantic
shifts of select words, we hope to be able to spot social media trends
to prevent future outbreaks faster than traditional methods. To explore
this idea, we generate a temporal word embedding model that allows us to
study word semantics evolving over time.

2. **Introduction & background**

Human languages experience continual changes to their semantic
structures. Natural language processing techniques then allow us to
examine these semantic alterations through methods such as word
embeddings. Word embeddings provide low dimension numerical
representations of words, mapping lexical meanings into a vector space.
Words that lie close together in this vector space represent close
semantic similarities [vec13]. This numerical vector space allows for
quantitative analysis of semantics and contextual meanings, allowing for
stronger machine learning models that utilize human language.

We hypothesize that disease outbreaks can be predicted faster than
traditional methods by studying word embeddings and their semantic
shifts during past outbreaks. By surveying the context of select medical
terms and other words associated with a disease during the initial
outbreak, we create a generalized model that can be used to catch future
similar outbreaks quickly. By leveraging social media activity, we
predict similar semantic trends can be found in real time. Additionally,
this allows novel terms to be evaluated in context without requiring a
priori knowledge of them, allowing potential outbreaks to be detected
early in their lifespans, thus minimizing the resultant damage to public
health.

Given a corpus spanning a fixed time period, multiple word embeddings
can be created at set temporal intervals, which can then be studied to
track contextual drift over time. However, a common issue in these
so-called “temporal word embeddings” is that they are often unaligned -
or the embeddings do not lie within the same embedding space. Past
proposed solutions to aligning temporal word embeddings require multiple
separate alignment problems to be solved, or for “anchor words” – words
that have no contextual shifts between times – to be used for mapping
one time period to the next [dia16]. Yao et al. propose a solution to
this alignment issue, shown to produce accurate and aligned temporal
word embeddings, through solving one joint alignment problem across all
time slices [dwe18].

3. **Methodology**

**3.1 Data Collection & Pre-Processing**

Our data set is a corpus \*D\* of over 7 million Tweets collected from
Scott County, Indiana from the dates January 1st, 2014 until January
17th, 2017. During this time period, an HIV outbreak was taking place in
Scott County, with an eventual 215 confirmed cases being linked to the
outbreak [oxy16]. Gonsalves et al. predicts an additional 126
undiagnosed HIV cases were linked to this same outbreak [sco18]. The
state's response led to questioning if the outbreak could have been
stemmed or further prevented with an earlier response [pol17]. Our
corpus was selected with a focus on Tweets related to the outbreak. By
closely studying the semantic shifts during this outbreak, we hope to
accurately predict similar future outbreaks before they reach large case
numbers, allowing for a critical, earlier response.

To study semantic shifts through time, the corpus was split into 18
temporal buckets, each spanning a 2 month period. The corpus within each
bucket is represented by \*D_t*, with t representing the temporal slice.
Within each 2 month period, Tweets were split into 12 pre-processed
output csv files. Pre-processing steps first removed retweets, links,
images, emojis, and punctuation. Common stop words were removed from the
Tweets using the NLTK Python Package, and each Tweet was then tokenized.
A vocabulary dictionary was then generated for each of the 18 temporal
buckets, containing each unique word and a count of its occurrences
within its respective bucket. The vocabulary dictionaries for each
bucket were then combined into a global vocabulary dictionary,
containing the total counts for each unique word across all 18 buckets.
Our experiments utilized two vocabulary dictionaries: the first being
the 10,000 most frequently occurring words from the global vocabulary;
the second being a list of medical terms taken from a `published
list <https://github.com/glutanimate/wordlist-medicalterms-en>`__ of
terms taken from combining two medical spell check libraries. The
combined vocabulary \*V\* consisted of the top 10,000 words across \*D\*
as well as an additional 8,156 medical terms that occurred within all
data handled in scripts utilized the pandas Python package.
Additionally, we created a vocabulary of HIV/AIDS specific medical terms
to be used in analysis.

**3.2 Temporally Aligned Vector Generation**

Generating word embeddings can be done through 2 primary methods:
continuous bag-of-words (CBOW) and skip-gram [vec13]. Our methods use a
CBOW approach at generating embeddings, which generates a word’s vector
embedding based on the context the word appears in, i.e the words in a
window range surrounding the target word. Following pre-processing of
our corpus, steps for generating word embeddings were applied to each
temporal bucket. Each time bucket First, co-occurrence matrices were
created, with a window size *w* = 5. These matrices contained the total
occurrences of each word against every other within a range of 5 words
within the corpus \*D_t*. Each co-occurrence matrix was of dimensions
\*V\* x \*V*. Following the generation of each of these co-occurrence
matrices, a \*V\* x \*V\* dimensioned Positive Pointwise Mutual
Information matrix was calculated. The value in each cell was calculated
as follows: PPMI(t, L)w,c = max{PMI(*D_t \*, L)w,c , 0}, where w and c
are two words in \*V\* Embeddings generated by word2vec utilize PMI
matrices, where given embedding vectors \*u_w\* and \*u_c\* EQUATION 2
from paper (dynamic word embeddings). Each embedding \*u\* has a reduced
dimensionality *d*, typically around 25 - 200. Each PPMI from our data
set is created independently from each other temporal bucket.

After these PPMI matrices are made, temporal word embeddings can be
created using the method proposed by Yao et al. [dwe18]. The proposed
solution focuses on EQUATION 4 from paper (dynamic word embeddings),
where U is a set of embeddings from time period t. Decomposing each
PPMI(t) will yield embedding U(t), however each U(t) is not guaranteed
to be in the same embedding space. [dwe18] derives U(t)A = B where
EQUATION 8 from paper (dynamic word embeddings, replace W(t) with
U(t),Y(t) represents PPMI(t)). To decompose PPMI(t) in our model,
SciPy’s linear algebra package was utilized to solve for
eigendecomposition of each PPMI(t), and the top 100 terms were kept to
generate an embedding of d = 100. The alignment was then applied,
yielding 18 temporally aligned word embedding sets of our vocabulary,
with dimensions V x d, or 18,156 x 100.

**3.3 Predictions for Detecting Modern Shifts**

Following the generation of temporally aligned word embedding, they can
be used for semantic shift analysis. Machine learning algorithms can
easily recognize patterns between past and present semantic shifts, but
for our purposes, the problem becomes determining which patterns are
indicative of the disease outbreak that was ongoing during the initial
Scott County HIV outbreak. Our initial solution involves determining
patterns within medical related terms, as those words are semantically
linked to a medical emergency such as this outbreak. Using the word
embedding vectors generated for each temporal bucket, a new data set was
created to use for determining semantic shift patterns. All 18 temporal
observations of each word were included in this data set, however rather
than using the embedding for each word for each temporal bucket, the
change in the embeddings between each consecutive bucket was used
instead, subtracting the first temporal bucket's embedding from the
second. Additionally, the two dimensional representation of initial and
next positions of each embedding were listed as features. These two
dimensional representations of the word embeddings were generated using
UMAP for dimensionality reduction, with a set random state to ensure a
shared space. This yielded each word having 17 observations and 104
features: {d_vec0 … d_vec99, x0, y0, x1, y1}.

The data was then split into 80% training and 20% testing. Using these
training data, K-means clustering was performed to try to classify each
observation. Several iterations with various parameters were attempted,
but all led had converging inertia values of over 20,000. Therefore
features were reassessed, and embedding vectors were created again with
dimension *d* = 10, yielding 14 features per observation. Inertia at
convergence on 8 cluster K-Means was reduced to around 3,000, yielding
significantly better results. Following the clustering, the results were
analyzed to determine which clusters contained the higher than average
incidence rates of medical terms and HIV/AIDS related terms. These
clusters are then considered target clusters, and large incidences of
words being clustered within these can be flagged as indicative as a
possible outbreak.

4. **Results**

**4.1 Quantitative Analysis of Embeddings**

To ensure accuracy in word embeddings generated in this model, we
utilized word2vec (w2v), a proven neural network method of embeddings
[vec13]. For each temporal bucket, a static w2v embedding of d = 100 was
generated to compare to the temporal embedding generated from the same
bucket. As the vectors do not lie within the same embedding space, the
vectors cannot be directly compared. Instead, we compare shared nearby
words between the vectors. As the temporal embeddings generated by the
alignment model are influenced by other temporal buckets, we hypothesize
notably different vectors. Methods for testing quality in [dwe18] rely
on a semi-supervised approach: the corpus used is an annotated set of
New York Times articles, and the section (*Sports, Business, Politics*,
etc.) are given alongside the text, and can be used to assess strength
of an embedding. Additionally, the corpus used spans over 20 years,
allowing for metrics such as checking the closest word to leaders or
titles, such as “president” or “NYC mayor” throughout time. These
methods show that the dynamic word embedding alignment model yields
accurate results. Given that our corpus spans a significantly shorter
time period, and does not have annotations, we use a very rudimentary
method of analysis, comparing the closest *n* = 15 words between the w2v
embeddings and the temporal embeddings. Add more here

**4.2 Prediction of Modern Shifts**

|image1|\ |image2|

The results of clustering led to medical related terms and HIV related
terms having higher incidences than other terms in 2 clusters each:
clusters 3 and 7 for HIV terms, and clusters 4 and 7 for medical related
terms. Incidence rates for all terms and medical terms in each cluster
can be seen in table (TABLE NUMBER) and figure, and HIV related terms in
table (TABLE NUMBER) and figure. FOR TEST DATA: CLASSIFY, FIND
PERCENTAGE OF MEDICAL TERMS THAT END UP IN SAME CLUSTERS

======= ========= ============= ==========
Cluster All Words Medical Terms Difference
======= ========= ============= ==========
0       0.055184  0.077877      0.022693
1       0.132719  0.070984      -0.06173
2       0.093325  0.09203       -0.0013
3       0.188303  0.132459      -0.05584
4       0.187044  0.277972      0.090929
5       0.071675  0.099538      0.027864
6       0.142118  0.062721      -0.0794
7       0.129633  0.186419      0.056786
======= ========= ============= ==========

======= ========= ========= ==========
Cluster All Words HIV Terms Difference
======= ========= ========= ==========
0       0.055184  0.031584  -0.0236
1       0.132719  0.137035  0.004317
2       0.093325  0.020886  -0.07244
3       0.188303  0.25675   0.068447
4       0.187044  0.151808  -0.03524
5       0.071675  0.059603  -0.01207
6       0.142118  0.120734  -0.02138
7       0.129633  0.2216    0.091967
======= ========= ========= ==========

5. **Future Work**

Case studies of previous datasets related to other diseases and
collection of more modern Tweets could not only provide critical insight
into relevant medical activity, but also further strengthen our model
and its credibility. One potent example is the 220 United States
counties determined by the CDC to be considered vulnerable to HIV and/or
viral hepatitis outbreaks due to injection drug use, similar to the
outbreak that occurred in Scott County [vul16]. Using the model
generated by our experiments can allow us to set up an early detection
system for an HIV outbreak in these counties, by analyzing social media
data in these select areas. The end goal is to create a pipeline that
can perform semantic shift analysis at set intervals of time, and detect
words that fit our classification of “outbreak indicative” terms. If
enough of these terms become detected, public health officials can be
notified the severity of a possible outbreak has the potential to be
mitigated if properly handled.

Expansion into other social media platforms would increase the variety
of data our model has access to, and therefore what our model is able to
respond to. With the foundational model established, we would be able to
focus on converting the data and addressing the differences between
social networks (e.g. audience and online etiquette). Reddit and
Instagram are two points of interest due to their increasing prevalence,
as well as vastness of available data.

An idea introduced by previous members of the project is to develop a
client and web application for general use of our model. The ideal
audience would be medical officials and organizations, but even public
or research use for trend prediction could be potent. The application
would give users the ability to pick from a given glossary of medical
terms, defining their own set of significant words to run our model on.
Our model would then expose any potential trends or insight for the
given terms in contemporary data, allowing for quicker responses to
activity. Customization of the data pool could also be a feature, where
Tweets and other social media posts are narrowed down to specific
geographic regions or smaller time windows, yielding more specified
results.

6. **Bibliography**

**[oxy16] Peters, P. J., Pontones, P., Hoover, K. W., Patel, M. R.,
Galang, R. R., Shields, J., Blosser, S. J., Spiller, M. W., Combs, B.,
Switzer, W. M., Conrad, C., Gentry, J., Khudyakov, Y., Waterhouse, D.,
Owen, S. M., Chapman, E., Roseberry, J. C., McCants, V., Weidle, P. J.,
… Duwve, J. M. (2016). HIV infection linked to injection use of
oxymorphone in Indiana, 2014–2015. New England Journal of Medicine,
375\ (3), 229–239.** **https://doi.org/10.1056/nejmoa1515195**

**[sco18] Gonsalves, G. S., & Crawford, F. W. (2018). Dynamics of the
HIV outbreak and response in Scott County, in, USA, 2011–15: A modeling
study. The Lancet HIV, 5\ (10).**
**https://doi.org/10.1016/s2352-3018(18)30176-0**

**[pol17] Golding, N. J. (2017). The Needle and the damage done:
Indiana's response to the 2015 HIV epidemic and the need to change state
and federal policies regarding needle exchanges and intravenous drug
users. Indiana Health Law Review, 14\ (2), 173.**
**https://doi.org/10.18060/3911.0038**

**[vul16] Van Handel, M. M., Rose, C. E., Hallisey, E. J., Kolling, J.
L., Zibbell, J. E., Lewis, B., Bohm, M. K., Jones, C. M., Flanagan, B.
E., Siddiqi, A.-E.-A., Iqbal, K., Dent, A. L., Mermin, J. H., McCray,
E., Ward, J. W., & Brooks, J. T. (2016). County-level vulnerability
assessment for rapid dissemination of HIV or HCV infections among
persons who inject drugs, United States. JAIDS Journal of Acquired
Immune Deficiency Syndromes,** **73\ (3), 323–331.**
**https://doi.org/10.1097/qai.0000000000001098**

**[vec13] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).
Efficient Estimation of Word Representations in Vector Space.**
**arXiv.** **https://doi.org/10.48550/arXiv.1301.3781**

**[dwe18] Yao, Z., Sun, Y., Ding, W., Rao, N., & Xiong, H. (2018).
Dynamic word embeddings for evolving semantic discovery. Proceedings of
the Eleventh ACM International Conference on Web Search and Data Mining.
https://doi.org/10.1145/3159652.3159703**

**[dia16] Hamilton, W., Leskovec, J., and Jurafsky, D. (2016).
Diachronic word embeddings reveal statistical laws of semantic change.
arXiv. https://doi.org/10.48550/arxiv.1605.09096**

.. |image1| image:: vertopal_57c8abe9d30a45a29277994ae3ab25cd/media/image1.png
   :width: 2.79688in
   :height: 2.08177in
.. |image2| image:: vertopal_57c8abe9d30a45a29277994ae3ab25cd/media/image2.png
   :width: 2.82813in
   :height: 2.10219in
