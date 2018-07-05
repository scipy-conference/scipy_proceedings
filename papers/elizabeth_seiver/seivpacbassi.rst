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
   collection of PLOS's 230,000+ articles. It also efficiently
   parses these article files into Python data structures. This article will
   cover how allofplos keeps your articles up-to-date, and how to use it to
   easily access common article metadata and fuel your meta-research, with
   actual use cases from inside PLOS.

.. class:: keywords

   Text and data mining, metascience, open access, science publishing, scientific articles, XML

Introduction
------------

Why mine scientific articles?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scientific articles are the standard mechanism of communication in science.
They embody a clear way by which human minds across centuries and continents
are able to communicate with one another, growing the total sum of knowledge.
Scientific articles are unique resources in that they are the material
artifacts by which this cultural exchange is made concrete and persistent.
They offer a unique source of insight into the history of carefully argued,
hard-won knowledge. Accordingly because they are made of annotated text, they
offer unique opportunities for well-defined text and data mining problems.
Importantly because PLOS represents the largest single journal in the history
of publishing it has collected an excellent corpus for this study. Equally
importantly because PLOS is Open Access the opportunity to use this data set is
available to everyone capable of downloading and analyzing it. The allofplos
library enables more people to do that more easily.

What is allofplos?
------------------

``allofplos`` is a Python package for downloading and maintaining up-to-date
scientific article corpora, as well as parsing PLOS XML articles in the JATS
(Journal Article Tag Suite)[jats]
format. It is available on PyPI[allofplospypi] as well as a GitHub repository[allofplosgh]. Many existing
Python packages for parsing XML and/or JATS focus on defensive parsing, where
the structure is assumed not to be reliable or the document is immediately
converted to another intermediate format (often JSON) and XML is just a
temporary stepping stone. allofplos uses lxml[lxml05], which is compiled in C, for
fast XML parsing and conversion to familiar Python data structures like lists,
dictionaries, and datetime objects. The intended audience is researchers who are
familiar with scientific articles and Python, but may not be familiar with JATS
XML. Other related tools include one from fellow Open Access publisher eLife[elife]. 

Functionality
-------------

``allofplos`` ships with a starter directory of 122 articles (``starterdir``), and
includes commands for downloading a 10,000 article demo corpus as well. The
default path to a corpus is stored as the variable ``corpusdir`` in the Python
program, and first checks for the environment variable ``$PLOS_CORPUS`` which
overrides that default location. If you have used pip to install the program,
specifying ``$PLOS_CORPUS`` will ensure that the article data will not be overwritten
when you update the ``allofplos`` package, as the default location is within the
package. (Forking/cloning the GitHub repository avoids this problem, because the
default corpus location is in the ``.gitignore`` file.)
  

.. code-block:: python

    import os
    os.environ['PLOS_CORPUS'] = 'path/to/corpus_directory'
    from allofplos import update
    update.main()

Downloading new articles can also be accessed via the command line:: 
  
    $ export PLOS_CORPUS="path/to/corpus_directory"
    $ python -m allofplos.update

If no articles are found at the specified corpus location, it will initiate a
download of the full corpus. This is a 4.6 GB zip file stored on Google Drive,
updated daily via an internal PLOS server, that then is unzipped in that
location to around 25 GB of 230,000+ XML articles. For incremental updates of
the corpus, allofplos first scans the corpus directory for all DOIs (Digital Object
Identifiers)[doi] of all articles (constructed from filenames) and compares that with
every article DOI from the PLOS search API. The missing articles are then downloaded individually in a rate-limited fashion from links that are constructed using the DOIs.
Those files are identical to the ones in the .zip file. The .zip file prevents users
from needing to scrape the entire PLOS website for the XML files, and "smartly"
scrapes only the latest articles. For a subset of provisional articles called
"uncorrected proofs", it checks whether the final version is available, and
downloads the updated version if so.


Article corpora and parsing
---------------------------

To initialize a corpus (defaults to ``corpusdir``, or the location set by the
``$PLOS_CORPUS`` environmental variable), use the ``Corpus`` class.


.. code-block:: python
  
   from allofplos import Corpus
   corpus = Corpus()
   
The number of articles in the corpus can be found with ``len(corpus)``. The list
of every DOI for every article in the corpus can be found at ``corpus.dois``, and
the path to every XML file in the corpus directory at ``corpus.filenames``. To
select a random Article object, use ``corpus.random_article``. To select a random
list of ten Article objects, use ``corpus.random_sample(10)``. You can also iterate
through articles as such:


.. code-block:: python

    for article in corpus[:10]:
        print(article.title)

Because DOIs contain semantic meaning and XML filenames are based on the DOI, if
you systematically loop through the corpus, it will not be a representative sample
but rather will implicitly progress first by journal name and then by publication
date. The iterator for ``Corpus()`` puts the articles in a random order to avoid
this problem.

The ``Article`` class
~~~~~~~~~~~~~~~~~~~~~

As mentioned above, you can use the Corpus class to initialize an Article()
object without calling it directly. An Article takes a DOI and the location of
the corpus directory to read the accompanying XML document into lxml.

.. code-block:: python

   art = Article('10.1371/journal.pone.0052669')

The lxml tree of the article is memoized in ``art.tree`` so it can be repeatedly
called without needing to re-read the XML file.

.. code-block:: python
    
    >>> type(art.tree)
    lxml.etree._ElementTree
    
Article parsing in ``allofplos`` focuses on metadata (e.g., article title, author
names and institutions, date of publication, Creative Commons copyright
license[cc], JATS version/DTD), which are conveniently located in the ``front``
section of the XML. We designed the parsing API to quickly locate and parse XML
elements as properties:

.. code-block:: python
    
    >>> art.doi
    '10.1371/journal.pone.0052669'
    >>> art.title
    'Statistical Basis for Predicting Technological Progress'
    >>> art.journal
    'PLOS ONE'
    >>> art.pubdate
    datetime.datetime(2013, 2, 28, 0, 0)
    >>> art.license
    {'license': 'CC-BY 4.0',
     'license_link': 'https://creativecommons.org/licenses/by/4.0/',
     'copyright_holder': 'Nagy et al',
     'copyright_year': 2013}
    >>> art.dtd
    'NLM 3.0'

For author information, ``Article`` reconciles and combines data from multiple
elements within the article into a clean standard form. Property names match XML
tags whenever possible.

Using XPath
~~~~~~~~~~~

You can also do XPath searches on `art.tree`, which works well for finding
article elements that are not Article class properties.

.. code-block:: python
  
    >>> acknowledge = art.tree.xpath('//ack/p')[0]
    >>> acknowledge.text
    'We thank all contributors to the Performance Curve Database (pcdb.santafe.edu).'

Use case: searching Methods sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
We can put these pieces together to make a list of articles that use PCR in their
Methods section (``pcr_list``). The body of an article is divided into sections
(with the element tag 'sec') and the element attributes of Methods sections are
either ``{'sec-type': 'materials|methods'}`` or ``{'sec-type': 'methods'}``. The
``lxml.etree`` module needs to be imported to turn XML elements into strings via
the ``tostring()`` method.

.. code-block:: python

    import lxml.etree as et
    pcr_list = []
    for article in corpus.random_sample(20):

        # Step 1: find Method sections
        methods_sections = article.root.xpath("//sec[@sec-type='materials|methods']")
        if not methods_sections:
            methods_sections = article.root.xpath("//sec[@sec-type='methods']")

        for sec in methods_sections:

            # Step 2: turn the method sections into strings
            method_string = et.tostring(sec, method='text', encoding='unicode')

            # Step 3: add DOI if 'PCR' in string
            if 'PCR' in method_string:
                pcr_list.append(article.doi)
                break
            else:
                pass

Query with peewee & SQLite
~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Query the corpus using *peewee* ORM
-  Included "starter" SQLite database
-  SQLite database constructor available

Future directions
-----------------
We also have plans for future updates to allofplos. First, we plan to make the article
parsing publisher-neutral, allowing for reading JATS content from other publishers
in addition to PLOS. Second, we want to improve incremental corpus updates so that all
changes can be downloaded and updated via a standardized mechanism such as a hash
table. This includes 'silent republications', where articles are updated online without
an official correction notice (the substance of the article is unchanged but
the XML has been updated). While the local allofplos server has methods for
catching these changes and updating the zip file appropriately, there is not currently
a way to make sure a user's local corpus copy reflects all of those changes. Third, we
want to expand the possibilities of multiple corpora and allow for article versioning,
such as for comparing older and newer versions of articles instead of just replacing
them entirely. And finally, we want to expand and integrate the functionality of the
sqlite database so that selecting a subset of articles based on metadata criteria such
as journal, publication date, or author is faster and easier than looping through each
XML file individually.

References
----------
.. [lxml05] Behnel, S., Faassen, M. et al. (2005),
            lxml: XML and HTML with Python, http://lxml.de.
.. [cc] Creative Commons Licenses. https://creativecommons.org/licenses/
.. [allofplosgh] allofplos GitHub repository. https://github.com/PLOS/allofplos
.. [allofplospypi] allofplos PyPI repository. https://pypi.org/project/allofplos/
.. [jats] JATS NIH/NISO standard.
          https://jats.nlm.nih.gov/publishing/tag-library/1.1d3/chapter/how-to-read.html
.. [elife] elife-tools GitHub repository. https://github.com/elifesciences/elife-tools
.. [doi] Digital Object Identifiers.
         https://www.doi.org/doi_handbook/1_Introduction.html
