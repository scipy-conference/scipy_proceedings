:author: Jyotika Singh
:email: singhjyotika811@gmail.com
:institution: ICX Media, Inc.

---------------------------------------------------
Analyzing YouTube using Python and Machine Learning
---------------------------------------------------

.. class:: abstract

   Social media is very popularly used every day with daily content viewing
   and/or posting that in turn influences people around this world in a variety
   of ways. Social media platforms, such as YouTube, have a lot of activity that
   goes on every day in terms of video posting, watching and commenting. While
   we can open the YouTube app on our phones and look at videos and what people
   are commenting, it only gives us a limited view as to kind of things others
   around us care about and what is trending amongst other consumers of our
   favorite topics or videos. Crawling some of this raw data and performing
   analysis on it using NLP can be tricky given the different styles of language
   usage by people in today’s world. This talk will highlight the YouTube’s open
   Data API and how to use it in python to get the raw data, data cleaning using
   Natural Language Processing tricks and Machine Learning in python for social
   media interactions, and extraction of trends and key influential factors from
   this data in an automated fashion. All these steps towards trend analysis will
   be talked about and demonstrated with examples that use different open-source
   python tools.

.. class:: keywords

   nlp, natural language processing, social media data, youtube, named entitity
   recognition, ner


Introduction
------------

Social media has large amounts of activity every second across the globe. Analyzing
text similar to text coming from a social media data source can be tricky due to
the absence of writing style rules and norms. Since this kind of data entails
user written text from a diverse set of locations, writing styles, languages and
topics, it is difficult to normalize data cleaning, extraction and NLP methods.

Social media data can be extracted using some official and open APIs. Examples
of such APIs include YouTube Data API and Twitter API. One important
thing to note would be to ensure one’s use case fits within compliance of API
guidelines. In this effort, the YouTube Data API will be discussed along
with common gotchas and useful tools that can be leveraged to access data.

One can perform NLP if the text data type is available for analysis. The nature
of noise seen in text from social media sources will be discussed and presented.
Cleaning of the noisy text using python techniques and open-source packages will
be further analyzed. Social media data additionally entails statistics of content
popularity, likes, dislikes and more. Analysis on statistical and text extracted
from YouTube API will be discussed and evaluated.

Finally, trend analysis will be performed using open-source python tools,
social media data, statistics, NLP techniques for data cleaning and named entity
recognition (NER) for a story-telling analytics piece.


Natural Language Processing
---------------------------

Natural language processing (NLP) is the computer manipulation of natural language.
Natural language refers to language coming from a human, either written or spoken.
[enwiki]_ defined NLP as follows: NLP is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze
large amounts of natural language data. The result is a computer capable of
"understanding" the contents of documents, including the contextual nuances of
the language within them. At one extreme, it could be as simple as counting word
frequencies to compare different writing styles.
[bird]_ mentions, "At the other extreme, NLP involves “understanding” complete
human utterances, at least to the extent of being able to give useful responses
to them. NLP is challenging because Natural language is messy. There are few
rules and human language is ever evolving.".

Some of the common NLP tasks on text data include the following.

1. Named Entity Recognition

  Named-entity recognition (NER) (also known as (named) entity identification,
  entity chunking, and entity extraction) is a subtask of information extraction
  that seeks to locate and classify named entities mentioned in unstructured text
  into predefined categories such as person names, organizations, locations,
  medical codes, time expressions, quantities, monetary values, percentages, etc.
  Some popular python libraries that can be leveraged include SpaCy [spacy]_ and NLTK [bird]_.

2. Keyphrase extraction

  Keyphrase extraction is the task of automatically selecting a small set of
  phrases that best describe a given free text document. [CoNLL]_
  Some popular tools that can be used for Keyphrase extraction are mentioned in
  this article [#]_.

  .. [#] https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f

3. Unigrams/Bigrams/Trigrams analysis

  Breaking down text into single words, a pair of consecutive written words
  or three consecutively written words and analyzing occurrence patterns.

4. Custom classifier building (public dataset -> features -> ML models)

  If out-of-box solutions do not exist for one's NLP task, building custom
  models to help solve for the problem is an option with the help of available
  data, NLP libraries (such as NLTK [#]_, SpaCy [#]_ and gensim [#]_), and
  Machine Learning libraries (scikit-learn [#]_).

  .. [#] https://www.nltk.org/
  .. [#] https://spacy.io/
  .. [#] https://radimrehurek.com/gensim/
  .. [#] https://scikit-learn.org/

5. Others

  Tokenization, Part-of-speech tagging, Lemmatization & Stemming, Word Sense
  Disambiguation, Topic modeling, Sentiment Analysis and Text summarization are
  some other popularly used NLP tasks. This list is not all inclusive.


Potential use cases include the following.

1. Analytics, intelligence and trends

  Analyzing patterns in text based on word occurrences, language usage, combining
  text occurrences with other available data, topics and sentiment information,
  NLP method outputs, or combinations thereof.

2. Story Telling

  Analyzing text using the various NLP techniques along with other statistical
  and available data aids in converting raw data to an informative story piece
  that helps understand and make sense of patterns observed. Depending on the
  data available, a time-window analysis can help study patterns as they change.

3. Mass analysis

  A human can only see N number of text samples a day to learn, whereas a machine
  can analyze a lot greater than N. Leveraging machines for NLP tasks along with
  several processing solutions available with Python, such as multiprocessing [#]_,
  can help analyze large amounts of data in a reasonable time-frame.

.. [#] https://docs.python.org/3/library/multiprocessing.html

Social Media APIs
-----------------

There are several social media platforms that let you programmatically collect
publicly available data and/or your own published data via APIs. Whatever you
intend to do with this data, it is important to ensure that you use the data in
compliance with the API’s guidelines and terms and services.

Some types of available requests on YouTube include search, video, channel,
comments, etc.
YouTube Data API documentation [#]_ is a great resource to learn more and get started.
At a high level, the getting started [#]_ steps include registering a project,
enabling the project and using the API key generated.

.. [#] https://developers.google.com/youtube/v3/docs
.. [#] https://developers.google.com/youtube/v3/getting-started

Gotchas
~~~~~~~

There are a few items to keep in mind when using the YouTube Data API. Some of
the gotchas while using the api include the following.

1. Rate limits

  The API key registered to you comes with a daily quota. The quota-spend depends
  on the kind of requests you make. API does not warn you in API request response
  if you are about to finish your daily quota but does throw that error once you
  have exceeded the daily quota. It is important to know how your application will
  behave if you hit the quota to avoid unexpected behavior and premature script
  termination.

2. Error handling

  If trying to query for a video, comment or channel that is set to private by the
  owner, the API throws an error. Your code could end prematurely if you are
  querying in a loop and one or a few of the requests have that issue. Error
  handling could help automate one’s process better on such expected errors.

Interacting with the YouTube Data API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to interact with the YouTube Data API. Some of them are
as follows.

1. Use the API web explorer's "Try this API" section [#]_

2. Build your own code using API documentation examples [#]_

3. Open-source tools

  1. Wrappers of YouTube Data API [#]_ : Libraries that act as wrappers and
  provide a way to use YouTube Data API V3.

  2. pyYouTubeAnalysis [#]_ : This library allows one to run searches, collect
  comments and define search params (search keywords, timeframe and type).
  Furthermore, the project includes error handling so code execution does not
  stop due to unforeseen errors while interacting with YouTube API. Additional
  features include NLP methods such as SpaCy based Named Entity Recognition (NER)
  that runs location extraction on text.

.. [#] https://developers.google.com/youtube/v3/docs/search/list
.. [#] https://developers.google.com/youtube/v3/quickstart/python
.. [#] https://github.com/rohitkhatri/youtube-python, https://github.com/sns-sdks/python-youtube
.. [#] https://github.com/jsingh811/pyYouTubeAnalysis


Social media / YouTube data noise
---------------------------------
Text fields are available within several places on YouTube, including video title,
description, tags, comments, channel title and channel description. Video title,
description, tags and channel title and description are filled by the
content/channel owner. Comments on the other hand are made by individuals
reacting to a video.

.. figure:: comments.png
   :scale: 60%
   :figclass: w

   Random sample of YouTube comments representing writing style diversity.
   :label:`commentsfig`

The challenges in such a data source arise due to writing style diversity,
language diversity and topic diversity. Figure :ref:`commentsfig` shows a few
examples of language diversity. On social media, people use abbreviations, and
sometimes these abbreviations may not be the most popular ones. Other than the
non-traditional abbreviation usage, different languages, different text lengths
and emojis used by commenters are observed.

Data Cleaning Techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Based on some noise seen on YouTube and other social media platforms, the
following data cleaning techniques have been found to be helpful cleaning methods.

1. Removing URLs

  Social media text data comes with a lot of URLs. Depending on the task at hand,
  removing the urls have been observed to come in handy for cleaning the text.

  .. code-block:: python

     import re

     URL_PATTERN = re.compile(
         r"https?://\S+|www\.\S+",
         re.X
     )

     def remove_urls(txt):
         """
         Remove urls from input text
         """
         clean_txt = URL_PATTERN.sub(" ", txt)
         return clean_txt


2. Removing emojis

  Emojis are widely used across social media and provide benefit in certain NLP
  tasks, such as sentiment analysis. On the contrary, for many other NLP tasks,
  removing emojis from text can be a useful cleaning method.

  .. code-block:: python

     import re

     EMOJI_PATTERN = re.compile(
         "[\U00010000-\U0010ffff]",
         flags=re.UNICODE
     )


     def remove_emojis(txt):
         """
         Remove emojis from input text
         """
         clean_txt = EMOJI_PATTERN.sub(" ", txt)
         return clean_txt


3. Spelling / typo corrections

  Some NLP models tend to do very well for a particular style of language and
  word usage. On social media, the language seen can be accompanied with
  various incorrectly spelled words, also known as typos.
  PySpellChecker [#]_, Autocorrect [#]_ and Textblob [#]_ are examples of open-source
  tools that can be used for spelling corrections.

.. [#] https://pypi.org/project/pyspellchecker/
.. [#] https://pypi.org/project/autocorrect/
.. [#] https://textblob.readthedocs.io/en/dev/


4. Language detection and translations

  Developing NLP methods on different languages is a challenging and popular
  problem. Often when one has developed NLP methods for english language text,
  detection of a foreign language and translation to english serves as a good
  solution and allows one to keep their NLP methods fixed. Such tasks introduce
  other challenges such as the quality of language detection and translation.
  Nonetheless, detection and translation is a popular technique while dealing
  with multiple different languages.
  Some examples of Python libraries that can be used for language detection
  include langdetect [#]_, Pycld2 [#]_, Textblob [#]_ and Googletrans [#]_.
  Translate [#]_ and Googletrans can be used for language translations.

.. [#] https://pypi.org/project/langdetect/
.. [#] https://pypi.org/project/pycld2/
.. [#] https://textblob.readthedocs.io/en/dev/
.. [#] https://pypi.org/project/googletrans/
.. [#] https://pypi.org/project/translate/


Trend analysis case study
---------------------------

.. figure:: flights.png

   Domestic and international flight search patterns in 2020.
   :label:`flightsfig`

In the year 2020, COVID hit us all hard. The world went through a lot of changes
in the matter of no time to reduce the spread of the virus. One such impact was
observed massively in the travel and hospitality industry. Figure :ref:`flightsfig`
shows the flight search trends between February and November 2020 for domestic and
international flight searches from the US. Right before lockdown and restrictions
were enforced starting in March across different places across the globe, a big
spike can be seen in flight searches, correlating with the activity of people
trying to fly back home if they were elsewhere before restrictions disabled
them to do so. A massive reduction in flight searches [#]_ can be seen after March and
beyond as travel was reduced due to COVID imposed restrictions.

.. [#] https://www.kayak.com/news/category/travel-trends/

.. figure:: hotels.png

   Hotel booking search patterns in 2020.
   :label:`hotelfig`

Aligning with reduced flight searches, reduced hotel search [#]_ were also reports
from March onwards as can be seen in figure :ref:`hotelfig`

.. [#] https://www.sojern.com/blog/covid-19-insights-on-travel-impact-hotel-agency/

Let’s try to correlate these findings and understand content consumption within
those time periods on YouTube.

First, a search was performed to gather videos about “travel vlogs”. Travel vlogs
are a popular content genre on YouTube where a lot of people are able to find
reviews, advice and sneak peaks of different destinations that wows them and
inspires travel plans. Such videos typically consist of people traveling to
different locations and recording themselves at different spots.

.. figure:: views_year.png

   Yearly video views. :label:`viewsyearfig`

.. figure:: likes_year.png

   Yearly video likes. :label:`likesyearfig`

.. figure:: comments_year.png

   Yearly video comments. :label:`commentsyearfig`

Statistically, it can be seen from figures :ref:`viewsyearfig`, :ref:`likesyearfig`
and :ref:`commentsyearfig` that travel vlog has been a growing
topic of interest and has been growing along with online content consumption over
the years up till 2019. A downward trend was seen in average views, comments and
likes on travel vlog videos in 2020.

.. figure:: views_month.png

   Monthly video views for 2019 and 2020. :label:`viewsmonthfig`

.. figure:: likes_month.png

   Monthly video likes for 2019 and 2020. :label:`likesmonthfig`

.. figure:: comments_month.png

   Monthly video comments for 2019 and 2020. :label:`commentsmonthfig`

.. figure:: stats_shift.png

   Difference in video engagements between 2019 and 2020. :label:`statsshiftfig`


Figures :ref:`viewsmonthfig`, :ref:`likesmonthfig` and :ref:`commentsmonthfig`
show a month over month comparison between 2019
and 2020 to analyze average audience engagement patterns. The viewership trends
reflect the reduction from March onwards when COVID hit most locations across the
globe. Figure :ref:`statsshiftfig` further shows engagement shift between 2019
and 2020. The trend slopes downwards, picks up a little July onwards, which
correlates with the time Europe lifted a lot of the travel restrictions. It can
be seen however, people were still creating travel vlogs and commenting on such
videos. Between June and September 2020, amidst a much-reduced travel, what were
these videos, what content was getting created, who was creating it and what were
the commenters talking about?

.. figure:: videos.png

   Word cloud of video topics.
   :label:`videofig`

Figure :ref:`videofig` shows a word cloud representation of what these videos
were about. Travel that would entail easier implementation of social distance
was seen popping up in 2020, such as hiking, beach trips and road traveling.
Location names such as Italy, France and Spain were also seen showing up in the videos.

YouTube influencer channels that drove high engagement during summer and fall of 2020
include the following.

1. 4K Walk [#]_ – YouTube channel creating videos about walking tours all over Europe and America.

2. BeachTuber [#]_ – YouTube channel creating vlogs from different beaches all over Europe.

3. Beach Walk [#]_ – YouTube channel posting about different beaches all over Europe and America. 

4. DesiGirl Traveller [#]_ – YouTube channel creating videos about India travel.

5. Euro Trotter [#]_ – YouTube channel creating videos about Europe travel.

.. [#] https://youtube.com/c/4KWALK
.. [#] https://youtube.com/c/BeachTuber
.. [#] https://youtube.com/c/BeachWalk
.. [#] https://youtube.com/c/DesiGirlTraveller
.. [#] https://youtube.com/c/EuroTrotter

.. figure:: locs.png

  Word cloud of location names used in comments.
  :label:`locsfig`

A few examples of comments that were being left by audiences of such videos are
as follows.

  "i’m going to sorrento in 10 days and i’m so excited. i’ve been watching tonnes
  of sorrento and italy vlogs and yours are so lush X) <3"

  "Did they require you to have a prior covid test?"

  "I loved the tour looked like you guys had fun. im going there next week, how
  long ago were you there and were there lots of restrictions and closing due to
  covid"

  "Great video man, this place looks amazing. I have never been to Iceland, would
  love to visit some day.  Honestly can't wait for the lockdown to be lifted so I
  can start travelling again. Thanks for sharing your experience. :)"

It was seen that people expressed interest in inquiring about the lifting of the
travel ban due to COVID, pre-travel COVID test requirements along with the
sentiment of waiting to travel again. People were seen mentioning a lot of
location names in their comments. With the help of SpaCy’s NER, location
extractions were performed and figure :ref:`locsfig` shows the location popularly
mentioned by commenters. One can see European locations, along with some Asian
and American locations which correlate with travel restriction reductions in
some of the places.

This analysis, including data collection and NER, was performed
using pyYouTubeAnalysis library [#]_.

.. [#] https://github.com/jsingh811/pyYouTubeAnalysis


References
----------
.. [enwiki] Wikipedia contributors. "Natural language processing." Wikipedia,
          The Free Encyclopedia. Wikipedia, The Free Encyclopedia,
          21 May. 2021. Web. 27 May. 2021.

.. [bird] Bird, Steven, et al. Natural Language Processing with Python.
          O’Reilly Media, 2009.

.. [CoNLL] Bennani-Smires, Kamil & Musat, Claudiu & Hossmann, Andreea & Baeriswyl,
          Michael & Jaggi, Martin. (2018). Simple Unsupervised Keyphrase Extraction
          using Sentence Embeddings. 10.18653/v1/K18-1022.

.. [spacy] Honnibal, Matthew and Montani, Ines and Van Landeghem, Sofie and Boyd, Adriane.
          (2020). spaCy: Industrial-strength Natural Language Processing in Python.
          Zenodo. 10.5281/zenodo.1212303. https://doi.org/10.5281/zenodo.1212303.
