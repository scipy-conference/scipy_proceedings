:author: Elizabeth Seiver
:email: elizabeth.seiver@gmail.com
:corresponding:

:author: M Pacer
:email: mpacer.phd@gmail.com
:institution: Netflix

:author: Sebastian Bassi
:email: sbassi@gmail.com
:institution: Globant

-------------------------------------------------------
Text and data mining scientific articles with allofplos
-------------------------------------------------------

.. class:: abstract

   Mining scientific articles is hard when many of them are inaccessible
   behind paywalls. The Public Library of Science (PLOS) is a non-profit
   Open Access science publisher of the single largest journal (*PLOS
   ONE*), whose articles are all freely available to read and re-use.
   allofplos is a Python package for maintaining a constantly growing
   collection of PLOS's 230,000+ articles. It also quickly and easily
   parses these article files into Python data structures. This talk will
   cover how allofplos keeps your articles up-to-date, and how to use it to
   easily access common article metadata and fuel your meta-research, with
   actual use cases from inside PLOS.

.. class:: keywords

   

Introduction
------------


What is allofplos?
------------------
allofplos is a Python package for downloading and maintaining up-to-date scientific article corpora, as well as parsing PLOS XML articles in the JATS format. It's available on PyPI as well as a GitHub repository. Many existing Python packages for parsing XML and/or JATS focus on defensive parsing, where the structure is assumed not to be reliable or the document is immediately converted to another intermediate format (often JSON) and XML is just a temporary stepping stone. allofplos uses lxml[CITE], which is compiled in C, for fast XML parsing and conversion to familiar Python data structures like lists, dictionaries, and datetime objects.

allofplos's parsing focuses on article metadata (e.g., article title, author names and institutions, date of publication), which are conveniently located in the 'front' section of the XML. We designed the parsing API around the Article() object which takes an article, turns it into an lxml tree, and quickly locates and parses fields as such:
>>> art = Article()
>>> art.title
'Title of article'
>>> art.journal
'PLOS ONE'
>>> art.pubdate
datetime.datetime(2016, 9, 8, 0, 0)
It's geared at researchers who are familiar with scientific articles and Python, but may not be familiar with the JATS XML.

How allofplos maintains corpora
-------------------------------
allofplos ships with a starter directory of 122 articles ('starterdir'), and includes commands for downloading a 10,000 article demo corpus as well. The default path to a corpus is stored as the variable 'corpusdir' in the Python program, and first checks for the environment variable 'PLOS_CORPUS' which overrides that default location. If you've used pip to install the program, specifying 'PLOS_CORPUS' will ensure that the article data won't get overwritten when you update the allofplos package, as the default location is within the package. (Forking/cloning the GitHub repository doesn't have the same problem, because the default corpus location is in the .gitignore file.)
>>> import os
>>> os.environ['PLOS_CORPUS'] = 'path/to/corpus_directory'
>>> from allofplos import update
>>> update.main()

Downloading new articles can also be accessed via the command line:
$ export PLOS_CORPUS="path/to/corpus_directory"
$ python -m allofplos.update

If no articles are found at the specified corpus location, it will initiate a download of the full corpus. This is a 4.6 GB zip file stored on Google Drive, updated daily via an internal PLOS server, that then is unzipped in that location to around 25 GB of 230,000+ XML articles. For incremental updates of the corpus, allofplos first scans the corpus directory for all DOIs of all articles (constructed from filenames) and diffs that with every article DOI from the PLOS search API. That list of missing articles are downloaded individually in a rate-limited fashion from links that are constructed using the DOIs. Those files are identical to the ones in the .zip file. The .zip file prevents users from needing to scrape the entire PLOS website for the XML files, and 'smartly' scrapes only the latest articles. It also checks for a subset of provisional articles called 'uncorrected proofs' if the final version is available and downloads a new version if so.


How allofplos uses corpora and parses articles
----------------------------------------------

