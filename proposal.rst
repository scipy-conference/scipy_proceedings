Title: Text and data mining scientific articles with allofplos
--------------------------------------------------------------

Short Abstract
--------------

Mining scientific articles is hard when many of them are inaccessible
behind paywalls. The Public Library of Science (PLOS) is a non-profit
Open Access science publisher of the single largest journal (*PLOS
ONE*), whose articles are all freely available to read and re-use.
allofplos is a Python package for maintaining a constantly growing
collection of PLOS's 227,000+ articles. It also quickly and easily
parses these article files into Python data structures. This talk will
cover how allofplos keeps your articles up-to-date, and how to use it to
easilyy access common article metadata and fuel your meta-research, with
actual use cases from inside PLOS. ## Long Abstract

Background/motivation
~~~~~~~~~~~~~~~~~~~~~

Why mine scientific articles?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Science articles represent scientific knowledge
-  Text- and data-mining techniques used for meta-research and
   meta-science
-  Quickly identify sets of articles of interest
-  Identify research literature trends over time (study findings, jargon
   usage, citation networks)
-  Quality Assurance (QA)
-  But, many scientific articles are behind a paywall

PLOS and its article corpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  227,000+ scientific articles from a wide array of research fields,
   focusing on the medical and life sciences
-  Open Access: free to read, free to re-use
-  Creative Commons license (CC-BY, CC0)
-  Adheres to cross-publisher standard Journal Article Tag Suite (JATS)
   schema

allofplos
~~~~~~~~~

What is allofplos?
^^^^^^^^^^^^^^^^^^

-  Python package for

   -  downloading and maintaining up-to-date article corpora
   -  parsing PLOS XML articles

-  Separates downloading initial corpus from updating existing corpus
-  Turns PLOS articles into Python data structures

   -  Focuses on article metadata (e.g., title, authors, date of
      publication)
   -  Enables TDM without in-depth XML knowledge

How allofplos maintains corpora
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Multiple corpora for different purposes

   -  Ships with a starter directory of 122 articles
   -  10,000 article demo corpus is easily downloadable
   -  Full corpus managed separately

-  Full corpus initial download: Downloads and unzips allofplos .zip
   file
-  Corpus updating:

   -  Queries PLOS's search API for DOIs of new articles
   -  Smartly scrapes PLOS's journal pages only for new or updated
      articles

How allofplos uses corpora and parses articles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Initialize Corpus class with folder of articles to iterate through
   DOIs

   -  Configure corpus location/identity with environment variable

-  Parsing articles with ``Article``

   -  Initialize an Article class object w/DOI or XML filename
   -  Parser uses the fast, C-based lxml library under the hood
   -  Access JATS metadata fields in common Python data structures
      (mostly lists and dicts)
   -  Properties follow intuitive naming scheme (article.title,
      article.journal, article.authors, article.pubdate,
      article.license)
   -  Handles previous XML standards (all JATS compatible) on older
      articles
   -  Reconciles and combines data from multiple locations within the
      article into clean standard form
   -  For efficiency, Article memoizes the lxml ElementTree object for
      rapid querying of multiple properties #### How to construct
      queries

-  Use Corpus & Article classes together to rapidly iterate through
   files
-  Query the full-text of the article using article.tree & XPath
-  Query the corpus using *peewee* ORM
-  Included "starter" SQLite database
-  SQLite database constructor available
-  Use case examples from inside PLOS

Relevant external links
~~~~~~~~~~~~~~~~~~~~~~~

-  allofplos on `GitHub <https://github.com/PLOS/allofplos>`__
-  allofplos on `PyPI <https://pypi.python.org/pypi/allofplos>`__
-  allofplos on `DockerHub <https://hub.docker.com/r/plos/allofplos/>`__
-  allofplos `tutorial <https://github.com/eseiver/xml_tutorial>`__ and
   `slides <https://github.com/eseiver/xml_tutorial/blob/master/allofplos_presentation%20slides.pdf>`__
   from The Hacker Within at UC Berkeley, Nov 2017
-  Lead author's presentation at Peer Review Congress, Sep 2017:
   https://www.youtube.com/watch?v=3wxsLIpcg80
