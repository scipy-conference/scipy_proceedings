:author: Anthony Breitzman
:email: breitzman@rowan.edu
:institution: Rowan University Department of Computer Science
:bibliography: mybib


--------------------------------------------------------
An Accessible Python based Author Identification Process
--------------------------------------------------------

.. class:: abstract

   The Federalist Papers are 85 documents written anonymously by a combination of Alexander Hamilton, John Jay, and James Madison in the late 
   1780's supporting adoption of the American Constitution.  All but 12 documents have confirmed authors based on lists provided before the 
   author’s deaths.  Mosteller and Wallace in 1963 provided evidence of authorship for the 12 disputed documents, however the analysis is 
   not readily accessible to non-statisticians.  In this paper we replicate the analysis but in a much more accessible way using modern 
   text mining methods and Python, culminating in a decision tree that predicts authorship using only three words.  One surprising result is the usefulness of filler-words in identifying writing styles.  The method 
   described here can be applied to other authorship questions such as linking the Unabomber manifesto with Ted Kaczynski, 
   identifying Shakespeare's collaborators, etc.

.. class:: keywords

   Federalist, Author Identification, Attribution, Forensic Linguistics, Text-Mining

Introduction
------------

Author attribution is a long-standing problem involving identifying true authors in anonymous texts. Recently 
the problem has garnered headlines with several high profile cases that were made 
possible with computers and text mining methods.  In 2017 *The Discovery Channel* created 
a TV series called Manhunt:Unabomber that showed how Forensic Linguistics was used to 
determine that Ted Kaczynski was the author of the Unabomber manifesto [Luu17]_. In 2016 a 
headline from *The Guardian* shook the literary world: "Christopher Marlowe credited as 
one of Shakespeare's co-writers" [Alb16]_. It was long suspected that Shakespeare collaborated with 
others, but since Marlowe was always considered his biggest rival, it was quite a surprise that the two collaborated. 

While forensic linguistics may be a recent name for such attribution, the idea of using statistical modeling to identify authors goes back to at 
least 1963 when Mosteller and Wallace published their ground-breaking study of the Federalist Papers [Mos63]_. Since that study was published 
in a Statistics journal, it requires a thorough understanding of statistics to understand it.  Since our audience consists 
mostly of Software Engineers instead of Statisticians, we present a more accessible analysis of the Federalist Papers, which can be applied to other author attribution problems.


History of the Federalist Papers
--------------------------------

This brief history is shortened from [Mos63]_ which itself is a much shortened history from [Ada1]_ and [Ada2]_.  The Federalist Papers were a series of essays written by Alexander Hamilton, John Jay, 
and James Madison published under the pseudonym "Publius" 
in New York newspapers in 1787 and 1788 in support of ratification of the constitution. It is surmised that the authors were not anxious to claim the essays for decades because the 
opinions in the essays sometimes opposed positions each later supported [Mos63]_.  Hamilton was famously killed in a duel in 1804 but he left a list of the essays he wrote with his lawyer before his death.  Madison later
generated his own list a decade later and attributed any discrepancies between the lists as "owing doubtless to the hurry in which (Hamilton's) memorandum was made out" [Ada1]_.  
Of the 85 essays, the 5 essays written by Jay are not in dispute.  Another 51 by Hamilton, 14 by Madison, and 3 joint essays coauthored by Hamilton and Madison are also not in dispute.  
However, 12 essays (Federalist Nos. 49-58, 62 amd 63) were claimed by both Hamilton and Madison in their respective lists [Mos63]_.




Similarities of Hamilton and Madison as Writers
-----------------------------------------------
Before Mosteller used advanced statistical modeling for author attribution, the standard approach was to look 
at things like sentence length to identify authors.  In [Mos87]_ Mosteller explains why this won't work with Hamilton and Madison because they are too similar.  
"The writings of Hamilton and Madison are difficult to tell apart because both authors were masters of the popular Spectator style of writing-complicated and oratorical. Never use a short word if a long one will do. Double
negatives are good, triple even better. To illustrate, in 1941 Frederick Williams and I counted sentence lengths for the undisputed papers and got
means of 34.55 and 34.59 words, respectively, for Hamilton and Madison,
and average standard deviations for sentence lengths of 19.2 and 20.3." [Mos87]_

With Python libraries such as NLTK [nltk02]_ it's quite easy to confirm the author similarities with just a few lines of code:

.. code-block:: python

   from nltk.tokenize import sent_tokenize
   from nltk.tokenize import word_tokenize
   
   #return a list of sentence lengths for
   #a document
   def sentence_stats(document): 
    sent_stats = []
    sentences = sent_tokenize(document)
    for sentence in sentences:
        words = word_tokenize(sentence)
        sent_stats.append(len(words))
    return(sent_stats)
 
Note that all of the code examples used in this paper are pasted in from a Jupyter notebook and are not commercial 
quality code.  Readability is prioritized over efficiency in the code examples provided. 

Using the function above we can produce comparison box-plots (See Figure :ref:`box`) of the non-disputed Federalist Papers of Hamilton, Madison, and Jay.  For comparison 
we also show sentence lengths for "Moby Dick" by Herman Melville and a "Tale of Two Cities" by Charles Dickens.

.. figure:: boxplots.png
   :align: left

   Boxplots showing Sentence Length Statistics for Five Authors :label:`box`

We see in Figure :ref:`box` that not only do Hamilton and Madison have the same median sentence length, but they have the same 25-percentile and 75-percentile sentence length and very similar minimum and maximum sentence lengths.  In comparison John Jay tends to use 
longer sentences while Melville and Dickens use shorter sentences.


.. raw:: latex

   \begin{table*}[htbp]
   \centering
    \begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|}
    \toprule
          & \multicolumn{5}{c|}{\% of Papers Containing Word} & \multicolumn{5}{c|}{Usage Per 1000 Words} \\
    \midrule
    word & \multicolumn{1}{c|}{Hamilton} & \multicolumn{1}{c|}{Madison} & \multicolumn{1}{c|}{Joint} & \multicolumn{1}{c|}{Disputed} & \multicolumn{1}{c|}{Jay} & \multicolumn{1}{c|}{Hamilton} & \multicolumn{1}{c|}{Madison} & \multicolumn{1}{c|}{Joint} & \multicolumn{1}{c|}{Disputed} & \multicolumn{1}{c|}{Jay} \\
    \midrule
    states & 96    & 100   & 100   & 91.6  & 100   & 3.636 & 5.662 & 1.562 & 4.231 & 2.038 \\
    \midrule
    constitution & 90.1  & 100   & 100   & 91.6  & 20    & 1.878 & 3.304 & 1.562 & 2.396 & 0.643 \\
    \midrule
    national & 84.3  & 78.5  & 33.3  & 33.3  & 80    & 1.976 & 1.224 & 0.312 & 0.449 & 3.326 \\
    \midrule
    executive & 43.1  & 64.2  & 66.6  & 50    & 40    & 1.044 & 2.172 & 0.312 & 0.674 & 0.321 \\
    \midrule
    senate & 35.2  & 42.8  & 33.3  & 41.6  & 20    & 0.907 & 0.416 & 0.312 & 1.572 & 0.965 \\
    \midrule
    president & 29.4  & 35.7  & 0     & 25    & 20    & 0.753 & 0.277 & 0     & 0.149 & 1.072 \\
    \midrule
    congress & 37.2  & 78.5  & 33.3  & 50    & 20    & 0.315 & 1.363 & 0.156 & 0.374 & 0.751 \\
    \midrule
    war   & 49    & 35.7  & 100   & 25    & 100   & 0.591 & 0.369 & 2.031 & 0.112 & 2.145 \\
    \bottomrule
    \end{tabular}%
	\caption{Percent of Documents Containing Content Word and Usage Rate by Author}
    \DUrole{label}{contentPerc}%
    \end{table*}%


Why a Traditional Text Mining Approach is likely to Fail
--------------------------------------------------------

A typical text mining approach might be to gather each author's documents, then convert to lower case and remove stopwords such as "the", "and", "is", etc. and 
treat each document as a vector so that we could 
do clustering or topic modeling on "content" words as described in [TD22]_ for example.  The 
problem with such a method is grouping by topics or clustering by themes will not help to identify author 
styles.  As 
we see in Table :ref:`contentPerc`, content words that give rise to topics such as "war" or "senate" will group a portion of each author's articles 
together into the "war" cluster along with portions of the disputed documents and similarly group together a portion of each author's articles into a "senate" cluster 
and so on, while not giving us any idea of true authorship of the disputed documents. 


A Quick Introduction to Naive Bayes Classification
--------------------------------------------------

In order to make this paper self-contained we will give a quick review of Naive Bayes classification, using a sentiment classifier as an example.  For 
a longer explanation see [Jur23]_ or many other 
books.  Essentially Baye's Theorem allows us to reverse a probability as follows:
:math:`P(C_i |a_1,a_2,...,a_k)=(P(a_1,a_2,...,a_k|C_i)P(C_i))/P(a_1,a_2,...,a_k).` In other words, the probability of 
Class :math:`i` given attributes :math:`a_1,a_2,...a_k`
is the same as the probability of those attributes given class :math:`i` multiplied by the probability of the class and divided by the probability of the attributes.

One might ask how is this an advantage?  If we think of a sentiment analyzer then we have two classes we're trying to decide: :math:`P(+)` or :math:`P(-)`. Thus 
we are just trying 
to decide if :math:`P(+|a_1,a_2,...,a_k)=(P(a_1,a_2,...,a_k|+)P(+))/P(a_1,a_2,...,a_k)` is larger or smaller 
than :math:`P(-|a_1,a_2,...,a_k)=(P(a_1,a_2,...,a_k|-)P(-))/P(a_1,a_2,...,a_k).`  However, since the denominators are the same and we can usually 
assume that :math:`P(+)` and :math:`P(-)` are roughly the same, we need only determine if :math:`P(a_1,a_2,...,a_k|+)` is larger or smaller 
than :math:`P(a_1,a_2,...,a_k|-).`

The last trick is if the attributes are independent, then :math:`P(a_1,a_2,...,a_k|C_i) = P(a_1|C_i)*P(a_2|C_i)...*P(a_k|C_i).`  If the attributes are
independent then the previous equality is always true, if they are not independent then the equality may not be true but still helpful.  The "Naive" part 
of Naive Bayes comes from assuming independence of attributes even if we can't confirm it.  Statisticians bristle at this assumption, and sometimes derisively 
refer to this as "Wrong Bayes" or "Idiot Bayes."  In reality, Naive Bayes is often an effective classifier, but because the independence of attributes is not guaranteed
we can generally believe the direction of the prediction (e.g. positive or negative sentiment), but not the specific probability that it 
computes.  That is, if a Naive Bayes classifier tells us a tweet is positive with 0.98 probability, we can trust it is likely positive, but we can't
trust the 0.98 probability of it being positive.

As an example, let's suppose we want to predict whether the tweet "yay it's taco tuesday." is positive or negative. Now let's assume we have the dictionary in Table :ref:`dict`. 
If we completely ignore the words that are not in the dictionary then it's pretty clear that :math:`P(+|yay) > P(-|yay)` and thus the sentiment of the example tweet is positive.


The next question is how might we build such a dictionary?  We can actually build a much better dictionary fairly easily.  Download a million tweets that contain a smile emoji and label those positive 
tweets and download another million tweets that have a frown emoji and 
label those negative tweets.  If "yay" appears 90,000 times in the million positive tweets and 2,000 times in the negative tweets then for the "yay" entry we store 0.09 for :math:`P(+)` and 0.002 for :math:`P(-)`.
This is much different then the 0.90 and 0.10 entries in Table :ref:`dict` but the net effect will be the same. It's also likely that "taco" will end up in this dictionary with a slightly positive sentiment as well.

The only risk is that there is a chance that "tuesday" makes it into the negative dictionary but not the positive dictionary, so the Naive Bayes calculation will have a zero product for the positive sentiment and a very small but non-zero product on the negative side.  The net
result is that Naive Bayes will accidentally predict a negative sentiment for the tweet "yay it's taco tuesday."

The usual method of dealing with this case is something called Laplace Smoothing.  As an example, if "tuesday" appears in 3 out of a million negative tweets and 0 out of a million positive tweets, we change the 
probabilities from 0/1,000,000 and 3/1,000,000 by adding 1 to each numerator and denominator (i.e. 1/1,000,001 and 4/1,000,001). We can get the same effect by creating an artificial tweet that 
contains one of each word in the union
of positive and negative tweets.  We add the artificial tweet to both the negative tweets and positive tweets before building our dictionary, and then we need not worry about Laplace 
smoothing or accidental zero products.

.. table:: A simplified sentiment dictionary. :label:`dict`

   +---------+------+------+
   | word    | P(+) | P(-) |
   +=========+======+======+
   | yay     |  .90 | .10  |
   +---------+------+------+
   | awesome |  .95 | .05  |
   +---------+------+------+
   | boo     |  .02 | .98  |
   +---------+------+------+
   | awful   |  .07 | .93  |
   +---------+------+------+

Building Naive Bayes Dictionaries for Author Identification
-----------------------------------------------------------

In the previous section we discussed a Naive Bayes classifier for predicting the sentiment of a document.  To predict whether an essay is 
likely to have been written by Hamilton or Madison is really not much different. Instead of building a positive and negative dictionary, 
we instead build a Hamilton dictionary based on his Federalist Papers and do the same for Madison's papers.  We can build an artificial 
essay that contains the union of all words in both author's papers as before to avoid the need for Laplace smoothing.  Note we need not 
involve Jay's papers since there is no dispute about which papers he authored.

Because we have limited space in this paper, we will not present all of the code needed, but 
will provide a sketch with key code snippets so that the reader could replicate the experiment easily.

Project Gutenberg [Fed1]_ has the Federalist Papers as a plain-text e-book with each essay as an individual chapter.  We 
place each chapter in a list of the form :code:`[number,[text]]` and thus each variable :code:`hamilton, madison, jay, joint`, and :code:`disputed` contains 
each author's Federalist Papers as a list of lists.

As an example the code snippet below prints out the first 35 characters of John Jay's Federalist Papers.  We show Jay since he 
has only 5 papers, but the other variables :code:`hamilton, madison`, and :code:`disputed` are in the same format.

.. code-block:: python

   for a in jay:
    print('Fed.No.'+str(a[0])+': '+a[1][0][:35]+'...')

**Output:**
::

   Fed.No.2: WHEN the people of America reflect ...
   Fed.No.3: IT IS not a new observation that th...
   Fed.No.4: MY LAST paper assigned several reas...
   Fed.No.5: QUEEN ANNE, in her letter of the 1s...
   Fed.No.64: IT IS a just and not a new observat...  

.. raw:: latex

   \begin{table*}[htbp]
   \centering
    \begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|}
    \toprule
          & \multicolumn{5}{c|}{\% of Papers Containing Word} & \multicolumn{5}{c|}{Usage Per 1000 Words} \\
    \midrule
    word & \multicolumn{1}{c|}{Hamilton} & \multicolumn{1}{c|}{Madison} & \multicolumn{1}{c|}{Joint} & \multicolumn{1}{c|}{Disputed} & \multicolumn{1}{c|}{Jay} & \multicolumn{1}{c|}{Hamilton} & \multicolumn{1}{c|}{Madison} & \multicolumn{1}{c|}{Joint} & \multicolumn{1}{c|}{Disputed} & \multicolumn{1}{c|}{Jay} \\
    \midrule
    upon  & 100   & 21.4  & 66.6  & 16.6  & 20    & 3.012 & 0.161 & 0.312 & 0.112 & 0.107 \\
    \midrule
    on    & 98    & 100   & 100   & 100   & 100   & 3.037 & 6.817 & 6.094 & 7.077 & 4.721 \\
    \midrule
    very  & 72.5  & 85.7  & 100   & 91.6  & 60    & 0.583 & 1.04  & 0.937 & 2.209 & 1.394 \\
    \midrule
    community & 62.7  & 14.2  & 33.3  & 25    & 20    & 0.558 & 0.046 & 0.156 & 0.187 & 0.107 \\
    \midrule
    while & 39.2  & 0     & 0     & 0     & 40    & 0.291 & 0     & 0     & 0     & 0.214 \\
    \midrule
    enough & 35.2  & 0     & 33.3  & 0     & 0     & 0.267 & 0     & 0.156 & 0     & 0 \\
    \midrule
    nomination & 13.7  & 0     & 0     & 0     & 0     & 0.178 & 0     & 0     & 0     & 0 \\
    \midrule
    consequently & 5.8   & 57.1  & 0     & 41.6  & 40    & 0.032 & 0.277 & 0     & 0.337 & 0.429 \\
    \midrule
    lesser & 3.9   & 35.7  & 0     & 16.6  & 20    & 0.016 & 0.161 & 0     & 0.149 & 0.107 \\
    \midrule
    whilst & 1.9   & 57.1  & 66.6  & 50    & 0     & 0.008 & 0.277 & 0.312 & 0.337 & 0 \\
    \midrule
    although & 1.9   & 42.8  & 0     & 33.3  & 80    & 0.008 & 0.161 & 0     & 0.149 & 0.536 \\
    \midrule
    composing & 1.9   & 42.8  & 33.3  & 16.6  & 0     & 0.008 & 0.254 & 0.156 & 0.074 & 0 \\
    \midrule
    recommended & 1.9   & 35.7  & 0     & 8.3   & 20    & 0.008 & 0.138 & 0     & 0.037 & 0.429 \\
    \midrule
    sphere & 1.9   & 35.7  & 0     & 16.6  & 0     & 0.008 & 0.184 & 0     & 0.112 & 0 \\
    \midrule
    pronounced & 1.9   & 28.5  & 0     & 16.6  & 0     & 0.008 & 0.115 & 0     & 0.074 & 0 \\
    \midrule
    respectively & 1.9   & 28.5  & 0     & 16.6  & 0     & 0.008 & 0.138 & 0     & 0.074 & 0 \\
    \midrule
    enlarge & 0     & 28.5  & 0     & 16.6  & 0     & 0     & 0.115 & 0     & 0.074 & 0 \\
    \midrule
    involves & 0     & 28.5  & 0     & 16.6  & 0     & 0     & 0.092 & 0     & 0.074 & 0 \\
    \midrule
    stamped & 0     & 28.5  & 33.3  & 0     & 0     & 0     & 0.092 & 0.156 & 0     & 0 \\
    \midrule
    crushed & 0     & 21.4  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    democratic & 0     & 21.4  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    dishonorable & 0     & 21.4  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    precision & 0     & 21.4  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    reform & 0     & 21.4  & 33.3  & 16.6  & 0     & 0     & 0.161 & 0.156 & 0.074 & 0 \\
    \midrule
    transferred & 0     & 21.4  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    universally & 0     & 21.4  & 0     & 8.3   & 20    & 0     & 0.069 & 0     & 0.037 & 0.107 \\
    \midrule
    bind  & 0     & 14.2  & 0     & 8.3   & 20    & 0     & 0.069 & 0     & 0.037 & 0.107 \\
    \midrule
    derives & 0     & 14.2  & 33.3  & 8.3   & 0     & 0     & 0.069 & 0.156 & 0.037 & 0 \\
    \midrule
    drawing & 0     & 14.2  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    function & 0     & 14.2  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    inconveniency & 0     & 14.2  & 0     & 16.6  & 0     & 0     & 0.069 & 0     & 0.074 & 0 \\
    \midrule
    obviated & 0     & 14.2  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \midrule
    patriotic & 0     & 14.2  & 0     & 25    & 20    & 0     & 0.069 & 0     & 0.112 & 0.107 \\
    \midrule
    speedy & 0     & 14.2  & 0     & 8.3   & 0     & 0     & 0.069 & 0     & 0.037 & 0 \\
    \bottomrule
    \end{tabular}%
    \caption{Favorite Words of Hamilton and Madison}
    \DUrole{label}{favorite1}%
    \end{table*}%
	
It is much easier to work with a list of dictionaries 
containing words and frequencies for each essay than it is to use full-text.  For example, if an essay mentions 'constitution' ten times 
an entry :code:`constitution:10` is much more useful 
to us than the raw text of the essay.  As mentioned above, the NLTK library [nltk02]_ makes it fairly easy to convert our list of lists into a list of dictionaries via the code below.

.. code-block:: python

  from nltk.tokenize import word_tokenize

  def get_document_dict(str1):
  #returns a dictonary with frequencies of any 
  #word in str1.
  #e.g. str1 = 'quick brown fox is quick.'
  # returns {quick:2, brown:1, fox:1, is:1}
  x = {}
  words = word_tokenize(str1.lower().strip())
  for b in words:
        if b in x:
            x[b]+=1
        else:
            x[b]=1
  return(x)
  
With a few calls to the function above we now have the variables :code:`hamiltonDicts`, :code:`madisonDicts`, :code:`jayDicts`, :code:`jointDicts`, and :code:`disputedDicts`.  Each 
of these variables contains a list of dictionaries with one dictionary for each essay.

Because of the analysis above on "content" words, we are reluctant to remove so called stopwords like "the", "and", "is" etc. 
However, we will remove any word that appears in 80 or more of the essays or in fewer than 3 of the essays, 
since such words cannot discriminate between authors.  As a result, we are likely to remove words like 
"the", "and", "is" but keep other words like "while" and "whilst" which might otherwise be removed via a stopword 
list but might be useful for discriminating between a Hamilton and Madison paper.

.. code-block:: python

  completeDict={} #dictionary containing every 
                  #word along with doc frequency

  kills = [',','.',"''",'',';','-',')','(']
  authDicts = [hamiltonDicts,madisonDicts,
               jointDicts,disputedDicts]
  for authDict in authDicts:
   for a in authDict:
    for x in a:
        if (x not in kills):
         if x in completeDict:
            completeDict[x]+=1
         else:
            completeDict[x]=1
            
  trimDict = set()
  for a in completeDict:
    x = completeDict[a]
    if (x >= 3 and x < 80):
        trimDict.add(a)
  
After running the code snippet above, :code:`completeDict` has 8,492 unique words and :code:`trimDict` has 3,967 unique words.  The Naive Bayes dictionary for 
both Hamilton and Madison will have 3,967 unique words and 
each word entry will contain the total frequency of each author's usage of that word plus 1 because of the Laplace smoothing 
step where 1 is added for each word in :code:`trimDict`.  Thus 
for example at this stage Hamilton's Naive Baye's dictionary will contain the value 374 for the word "upon" while Madison's dictionary 
will contain the value 9 for the word "upon". To convert the word frequencies to Bayesian probabilities, denominators 
are computed by running through each dictionary and summing up all of the word frequency counts.  Since Hamilton has more documents, his denominator 
is 64,314 while Madison's is 27,766.  Thus if an essay has :code:`N` instances 
of "upon" in it, the Naive Bayes contribution for "upon" would be :code:`N*0.00581521908138197` for Hamilton and :code:`N*0.0003241374342721314` for 
Madison. Both are small numbers but the Hamilton value for "upon" is an order of magnitude larger than for Madison.

Running a Naive Bayes model at this point yields 100% accuracy for the known Hamilton and Madison authored papers, but in an effort 
to make the model more accessible it is worth simplifying the dictionaries.  Just like the example for "upon" above, the same values can be computed for 
all words. It is then easy to highlight the favorite words of Hamilton and Madison.  In the code snippet below a list of words 
that Hamilton uses 5 times as often as Madison (or vice-versa) is generated.  The wordlist generated by the code below can be used to create a 
simpler Naive Bayes 
model that only uses a limited dictionary.  Again, it will achieve 100% accuracy for the known Hamilton and Madison authored papers.

.. code-block:: python

   smallVocab1 = []
   for i,a in enumerate(trimDict):
    h1 = hamiltonNBwordDicts[a]/hamiltonNBdenom
    m1 = madisonNBwordDicts[a]/madisonNBdenom
    if (m1/h1 > 5 or m1/h1 <.2):
      smallVocab1.append(a)
  
A subset of the most interesting words from the code above can be found in Table :ref:`favorite1`.  We've also added "on" and "very" to the list, just to 
illustrate that stopwords will often be used quite differently for different authors.

Table :ref:`favorite1` shows that "upon" is one of Hamilton's favorite 
words.  He uses it in all of his papers while Madison uses it in only 21% of his papers.  Also, 
Hamilton uses it at a rate of 3 times per 1,000 words, which is roughly 20 times as often as Madison.  
If we look at the disputed papers, only 2 out of 12 use the word "upon" and the usage rate is much 
closer to Madison than to Hamilton for the word "upon."  "On" is another interesting word from 
Table :ref:`favorite1`.  This word would often be discarded in a text analysis, but we see that while it 
is used frequently by both Madison and Hamilton, it is used at twice the rate by Madison as by Hamilton.  
Probably because Hamilton uses "upon" in place of "on" in many cases.  We note that in the disputed 
papers, "on" is used at the Madison rate instead of the Hamilton rate.  Another word that might 
be discarded in many text analyses is "very" which is used by both authors, but used much more often by Madison.  
Again, the use in the disputed texts, indicates Madison authorship for many. 

We see from Table :ref:`favorite1` that Hamilton has a few words that he uses often that Madison uses rarely. 
The previously mentioned "upon" is used 5 times as often by Hamilton 
than Madison. We see that "community" is used 4 times as often by Hamilton than Madison, 
and "while," "enough," and "nomination" are used by Hamilton but never Madison.  None of these words are used at the Hamilton rates in the disputed papers however.

Madison has many more words that he uses that Hamilton rarely or never uses.  Both "whilst" and "consequently" 
are used in 57% of Madison's papers but "whilst" is used in only 2% of Hamilton's papers and "consequently" 
is used in only 6% of Hamilton's papers.  Both of these words are used at the Madison rate in the disputed papers. There are also several words 
that Madison uses that Hamilton never uses including "crushed," "dishonerable," "precision," "reform," "bind," "inconveniency," "obviated," and several 
others.  All of these words appear in at least one of the disputed papers.

We aren't addressing the 3 joint papers in this analysis, but Table :ref:`favorite1` also gives us some insight into those.  From the table we see both 
author's favorite words, but it looks like Madison is the dominant partner or perhaps the final editor.  The evidence for this is that 
although "upon" appears in the joint papers, it appears at only about 10% the rate of a typical Hamilton paper.  
Also, "on" and "very" are used at the Madison rate in the joint papers.  "While" which is another 
Hamilton favorite, doesn't appear in the joint papers and "whilst" appears at just over the Madison rate.

The code and output for running Naive Bayes on several subsets of the small vocabulary (generated above) is shown below.

.. code-block:: python

   def report1(words):
    print(words)
    print(str(check_accuracy(words))+'% accuracy')
    madison = hamilton = 0
    for a in disputedDicts:
        if (NB_federalist_predict(a,words)=='madison'):
            madison+=1
        else:
            hamilton+=1
    print("disputed papers: madison:"+str(madison)+
          ', hamilton:'+str(hamilton)+'\n')

   report1(smallVocab1)
   report1(['although','composing','involves',
         'confederation','upon'])
   report1(['although','obviated','composing',
         'whilst','consequently','upon'])
   report1(['against','within','inhabitants',
         'whilst','powers','upon','while'])
   report1(['against','upon','whilst',
         'inhabitants','within'])
   report1(['against','within','inhabitants',
         'whilst','upon'])
   report1(['against','while','whilst','upon','on'])
   report1(['concurrent','upon','on',
         'very','natural'])
   report1(['while','upon','on','inconveniency'])

**Output:**
::

   ['composing', 'respectively', 'bind', 'dishonorable',
   'compilers', 'chain', 'enumeration', 'involves',
   'consequently', 'jury', 'indebted', 'matter',
   'assumed', 'nomination', 'kind', 'enlarge', 'speedy',
   'surely', 'slaves', 'residence', 'crushed',
   'democratic', 'transferred', 'eyes', 'novelty',
   'obviated', 'including', 'disregarded', 'readily',
   'administering', 'obscurity', 'complied', 'reform',
   'works', 'although', 'department', 'term',
   'community', 'confederation', 'relief', 'pronounced',
   'pointing', 'precision', 'courts', 'lesser',
   'commonly', 'enough', 'while', 'drawing', 'sphere',
   'democracy', 'coin', 'indispensably', 'patriotic',
   'universally', 'recommended', 'function',
   'unanimous', 'whilst', 'violating', 'annually',
   'stamped', 'intended', 'indirectly', 'alloy',
   'cantons', 'inconveniency', 'upon', 'intermixed',
   'derives']
   100.0% accuracy
   disputed papers: madison:12, hamilton:0

   ['although', 'composing', 'involves',
   'confederation', 'upon']
   100.0% accuracy
   disputed papers: madison:12, hamilton:0

   ['although', 'obviated', 'composing', 'whilst',
   'consequently', 'upon']
   96.92307692307692% accuracy
   disputed papers: madison:12, hamilton:0

   ['against', 'within', 'inhabitants', 'whilst',
   'powers', 'upon', 'while']
   100.0% accuracy
   disputed papers: madison:12, hamilton:0

   ['against', 'upon', 'whilst', 'inhabitants',
   'within']
   96.92307692307692% accuracy
   disputed papers: madison:12, hamilton:0

   ['against', 'while', 'whilst', 'upon', 'on']
   96.92307692307692% accuracy
   disputed papers: madison:12, hamilton:0

   ['concurrent', 'upon', 'on', 'very', 'natural']
   98.46153846153847% accuracy
   disputed papers: madison:12, hamilton:0

   ['while', 'upon', 'on', 'inconveniency']
   95.38461538461539% accuracy
   disputed papers: madison:12, hamilton:0

The snippet above shows a number of Naive Bayes outputs based on several subsets of the small vocabulary of 70 words generated previously.  
When run on the full 70 words the model accurately identifies all known Hamilton papers and all known Madison papers and also predicts that the disputed 
papers are all authored by Madison.  When the subset of words is reduced to "although," "composing", "involves," "confederation," and "upon" the same results are obtained.  Other subsets may give slightly lower accuracy, but all predict that Madison is the author of the disputed essays.

The :code:`report1` function shown above calls two other functions.  The :code:`check_accuracy` function is straightforward.  
It just checks whether the model predicts the right author for the known Hamilton and Madison essays.  This is just a couple of if statements 
and is not shown to conserve space.  The other function is the Naive Bayes model function. It is shown below.

.. code-block:: python

   import math
   """
   given a document return 'hamilton' if NaiveBayes 
   prob suggests Hamilton authored it. similarly 
   return 'madison' if he is the likely author.
   
   use trimDict unless another word list passed in.
   """
   def NB_federalist_predict(docDict,vocab1=trimDict):
    h_pr = m_pr = 0
    for word in docDict:
      if (word in vocab1):
        h_pr += float(docDict[word])*(math.log(
           hamiltonNBwordDicts[word]/hamiltonNBdenom))
        m_pr += float(docDict[word])*(math.log(
           madisonNBwordDicts[word]/madisonNBdenom))
        
    if (h_pr > m_pr):
         return('hamilton')
    else:
         return('madison')


Note the use of :code:`math.log` and that there is no attempt 
to compute an actual probability.  This is an old coding trick to prevent underflow back 
when floating point numbers were all low precision.  Rather than multiply several very small quantities 
as we saw for "upon" above, we take the sum of the logarithms of the small quantities.  We need not exponentiate 
the sums to get the true answers, because as was discussed above, the actual values from Naive 
Bayes computations are not accurate probability values.  Thus, just identifying which sum of logs is larger is enough to predict the author. 

A Very Simple Decision Tree Model of Author Identification
----------------------------------------------------------

From Table :ref:`favorite1` above, it's clear that "while", "whilst", and "upon" can somewhat 
distinguish between papers authored by Hamilton or Madison.  The use of "while" suggests Hamilton, 
while the use of "whilst" often suggests Madison, particularly if the rate is above 0.25 mentions per 1,000 
words.  If neither "while," or "whilst" is mentioned we can look for "upon."  Both authors use "upon", 
but if the rate of "upon" is at 0.9 mentions per 1,000 words or above, then it is almost certainly authored by Hamilton.

The description above can be made into a very simple decision tree.  A decision tree can be made into a series of 
if-then statements, yielding the simple model below.

.. code-block:: python

   #return usage rate per 1000 words of a target word
   #e.g. if target=='upon' appears 3 times in a 1500 
   #word essay, we return a rate of 2 per 1000 words.
   def rate_per_1000(docDict,target):
    if (target in docDict):
        wordCount=0
        for a in docDict:
            wordCount+=docDict[a]
        return(1000*docDict[target]/wordCount)
    else:
        return(0)
    
   #given a document dictionary, predict if it was 
   #authored by Hamilton or Madison
   def federalist_decison_tree(docDict):
    if ('while' in docDict):
        return('hamilton')
    else:
        if (rate_per_1000(docDict,'whilst') >= .25):
                return('madison')
        if (rate_per_1000(docDict,'upon') >= .9):
                return('hamilton')
        else:
                return('madison')
				
The simple model above is 100% accurate on the known documents, and predicts Madison as the author of the 12 disputed documents.			

Additional Models with Sci-Kit Learning
---------------------------------------
In the sections above we hand-built a Naive Bayes model and Decision tree model that both predicted that Madison authored the disputed 
Federalist Papers.  Since this paper is being presented to an audience with Python experience, it is worth showing how easy it is 
to create and test multiple machine learning models using the Sci-Kit Learn [SKlearn]_ library.  For these models we'll use a very small vocabulary of the words "against," "within," "inhabitants," "whilst," and "upon."

.. code-block:: python

   """
   Build and test multiple models via SKlearn.
   X is a dataframe consisting of known
   Hamilton and Madison papers.

   y is a data frameconsisting of author labels.

   X_test is a dataframe consisting of disputed
   papers
   """
 
   import sklearn
   from sklearn.feature_extraction.text 
                import TfidfVectorizer
   from sklearn.feature_extraction.text 
                import CountVectorizer
   from sklearn.ensemble 
                import RandomForestClassifier
   #other model imports such as 
   #KNeighborsClassifier similar but
   #not shown for space consideration
   from sklearn.model_selection 
                import cross_val_score



   smallVocab5 = ['against','within','inhabitants',
                  'whilst','upon']


   tfidf = sklearn.feature_extraction.text.
                TfidfVectorizer(analyzer="word", 
                                binary=False,
                                min_df=2,
                                vocabulary=smallVocab5)
								
   X_transformed = tfidf.fit_transform(X)
   lb = sklearn.preprocessing.LabelEncoder()
   y_transformed = lb.fit_transform(y)

   X_test_transformed = tfidf.transform(X_test)

   models = [
      KNeighborsClassifier(3),
      DecisionTreeClassifier(max_depth=5),
      RandomForestClassifier(n_estimators=25, 
	                         max_depth=3),
      LinearSVC(),
      SVC(gamma=2, C=1),
      ComplementNB(),
      AdaBoostClassifier()
      ]

   CV = 5
   cv_df = pd.DataFrame(index=range(CV * len(models)))
   for model in models:
     model_name = model.__class__.__name__
     accuracies = cross_val_score(model, X_transformed, 
                                  y_transformed, 
                                  scoring='accuracy', 
                                  cv=CV)
     avgAccur = 0
     for fold_idx, accuracy in enumerate(accuracies):
       print(model_name,"fold:",fold_idx,
	         "accuracy:",str(accuracy)[:5])
     print(model_name,"avg accuracy:",
	       str(accuracies.mean())[:5])

   model.fit(X_transformed, y_transformed)
   y_final_predicted = model.predict(X_test_transformed)
   y_final_predicted_labeled = 
                 lb.inverse_transform(y_final_predicted)
  
   mPercent(y_final_predicted_labeled)

**Output:**
::
 
   KNeighborsClassifier fold: 0 accuracy: 1.0
   KNeighborsClassifier fold: 1 accuracy: 1.0
   KNeighborsClassifier fold: 2 accuracy: 1.0
   KNeighborsClassifier fold: 3 accuracy: 1.0
   KNeighborsClassifier fold: 4 accuracy: 1.0
   KNeighborsClassifier avg accuracy: 1.0
   % Disputed attributed to Madison: 100.0

   DecisionTreeClassifier fold: 0 accuracy: 1.0
   DecisionTreeClassifier fold: 1 accuracy: 0.846
   DecisionTreeClassifier fold: 2 accuracy: 1.0
   DecisionTreeClassifier fold: 3 accuracy: 1.0
   DecisionTreeClassifier fold: 4 accuracy: 1.0
   DecisionTreeClassifier avg accuracy: 0.969
   % Disputed attributed to Madison: 100.0

   RandomForestClassifier fold: 0 accuracy: 1.0
   RandomForestClassifier fold: 1 accuracy: 0.846
   RandomForestClassifier fold: 2 accuracy: 1.0
   RandomForestClassifier fold: 3 accuracy: 1.0
   RandomForestClassifier fold: 4 accuracy: 1.0
   RandomForestClassifier avg accuracy: 0.969
   % Disputed attributed to Madison: 100.0

   LinearSVC fold: 0 accuracy: 1.0 
   LinearSVC fold: 1 accuracy: 1.0
   LinearSVC fold: 2 accuracy: 1.0
   LinearSVC fold: 3 accuracy: 1.0
   LinearSVC fold: 4 accuracy: 1.0
   LinearSVC avg accuracy: 1.0
   % Disputed attributed to Madison: 100.0

   SVC fold: 0 accuracy: 1.0
   SVC fold: 1 accuracy: 1.0
   SVC fold: 2 accuracy: 1.0
   SVC fold: 3 accuracy: 1.0
   SVC fold: 4 accuracy: 1.0
   SVC avg accuracy: 1.0
   % Disputed attributed to Madison: 100.0

   ComplementNB fold: 0 accuracy: 0.923
   ComplementNB fold: 1 accuracy: 1.0
   ComplementNB fold: 2 accuracy: 1.0
   ComplementNB fold: 3 accuracy: 1.0
   ComplementNB fold: 4 accuracy: 1.0
   ComplementNB avg accuracy: 0.985
   % Disputed attributed to Madison: 100.0

   AdaBoostClassifier fold: 0 accuracy: 1.0
   AdaBoostClassifier fold: 1 accuracy: 0.846
   AdaBoostClassifier fold: 2 accuracy: 1.0
   AdaBoostClassifier fold: 3 accuracy: 1.0
   AdaBoostClassifier fold: 4 accuracy: 1.0
   AdaBoostClassifier avg accuracy: 0.969
   % Disputed attributed to Madison: 100.0
   
 
The code snippet above puts multiple Sci-Kit Learn models [SKlearn]_ into a list and loops through each.  Inside the 
loop a 5-fold cross validation is run on the training data consisting of all known Hamilton and Madison essays.  The models are 
then run on the disputed papers and a function called :code:`mPercent` is called that calculates how many of the disputed 
papers were written by Madison.  The latter function is not shown to save space, but is very straightforward to program 
since the 
predictor function gives either an "m" for Madison or an "h" for Hamilton.

We note that the 5-fold cross validation is 100% accurate for each fold for the K-Nearest Neighbors model, and 
the Support-Vector classifiers.  For the other models 4 out of 5 folds were 100% accurate and overall the models
were 97% accurate or better.  All of the models predicted that the disputed papers were written by Madison.

Note Sci-Kit Learn offers multiple Naive Bayes classifiers.  The Complement Naive Bayes model was chosen above 
because it was empirically shown by [CNB]_ to outperform other Naive Bayes models on text classification tasks.


Conclusions
-----------
In this brief paper we presented a number of ways to solve the problem of disputed author identification.  First we showed 
that a Naive Bayes dictionary could be built in a way that is similar to models used in sentiment analysis.  We then showed 
how such a dictionary could be used to identify favorite words for each target author.  We built a Naive Bayes model that 
suggested that James Madison is the likely author of the disputed Federalist Papers.  We then built a very simple decision 
tree using only the words "while," "whilst," and "upon" which also points to Madison as the author.  Finally, we showed 
how the Sci-Kit Learn [SKlearn]_ library could be used to build and test numerous models very quickly.  Each of the Sci-Kit 
Learn models also point to Madison as the author.  Note that whilte this is a case-study of the Federalist Papers, the methods 
shown here can easily be applied to any author 
identification process.  

.. [Luu17] C. Luu, "Fighting Words With the Unabomber", JSTOR.org, August 1, 2017, https://daily.jstor.org/fighting-words-unabomber/.

.. [Alb16] D. Alberge, "Christopher Marlowe credited as one of Shakespeare's co-writers," The Guardian, 23 Oct. 2016, https://www.theguardian.com/culture/2016/oct/23/christopher-marlowe-credited-as-one-of-shakespeares-co-writers.

.. [Mos63] F. Mosteller and D. L. Wallace, "Inference in an Authorship Problem", Journal of the American Statistical Association, 1963, pp. 275-309,https://www.jstor.org/stable/2283270?origin=JSTOR-pdf"

.. [Mos87] F. Mosteller, "A Statistical Study of the Writing Styles of the Authors of The Federalist Papers," Proceedings of the American Philosophical Society , Jun., 1987, Vol. 131, No. 2, pp. 132-140, https://www.jstor.org/stable/986786

.. [nltk02] E. Loper and S. Bird, "NLTK: The Natural Language Toolkit," arXiv, cs/0205028, 2002. 

.. [Ada1] D. Adair, "The Authorship of the Disputed Federalist Papers," The William and Mary Quarterly 1, no. 2 (April 1944): 97–122. 

.. [Ada2] D. Adair, "The Authorship of the Disputed Federalist Papers: Part II," The William and Mary Quarterly 1, no. 3 (July 1944): 235–264.

.. [Jur23] D. Jurafsky and J. Martin, "Speech and Language Processing (Draft of 3rd edition)," Draft of January 7, 2023.

.. [TD22] J. Grimmer and M.E. Roberts and B.M. Stewart,"Text as Data: A New Framework for Machine Learning and the Social Sciences," Princeton University Press, 2022.

.. [Fed1] Alexander Hamilton and John Jay and James Madison,"The Project Gutenberg eBook of The Federalist Papers", Available at \url{https://www.gutenberg.org/cache/epub/1404/pg1404.txt, Last Accessed May 1, 2023

.. [SKlearn] F. Pedregosa and others.,"Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research},2011, 2825--2830.

.. [CNB] J.D. Rennie and L. Shih and J. Teevan and D.R. Karger, "Tackling the poor assumptions of naive bayes text classifiers" ICML (Vol. 3, pp. 616-623), 2023.