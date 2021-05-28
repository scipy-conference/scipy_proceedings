:author: Frederick Reiss
:email: frreiss@us.ibm.com
:institution: IBM Research

:author: Bryan Cutler
:email: bjcutler@us.ibm.com
:institution: IBM

:author: Zachary Eichenberger
:email: zachary.eichen@gmail.com
:institution: University of Michigan
:institution: IBM Research

:bibliography: references

:video: https://youtu.be/dQw4w9WgXcQ

--------------------------------------------------
Natural Language Processing with Pandas DataFrames
--------------------------------------------------

.. TODO update abstract; this version copied from submission

.. class:: abstract

    Most areas of Python data science have standardized on using Pandas
    DataFrames for representing and manipulating structured data in memory.
    Natural Language Processing, not so much.
    
    We believe that Pandas has the potential to serve as a universal data
    structure for NLP data. DataFrames could make every phase of NLP easier,
    from creating new models, to evaluating their effectiveness, to building
    applications that integrate those models.  However, Pandas currently lacks
    important data types and operations for representing and manipulating
    crucial types of data in many of these NLP tasks.

    This paper describes *Text Extensions for Pandas*, a library of extensions
    to Pandas that make it possible to build end-to-end NLP applications while
    representing all of the applications' internal data with DataFrames.
    We leverage the extension points built into Pandas library to add new data
    types, and we provide important NLP-specfific operations over these data
    types and and integrations with popular NLP libraries and data formats.
    
.. class:: keywords

   terraforming, desert, numerical perspective

Background and Motivation
-------------------------

This paper describes our work on applying general purpose data analysis tools
from the Python data science stack to natural language processing (NLP)
applications.  This work is motivated by our experiences working on NLP
products from IBM's *Watson* portfolio, including IBM Watson Natural Language
Understanding and IBM Watson Discovery.  

*TODO: Citations for NLU and Discovery*

These products include many NLP components, such as state-of-the-art machine
learning models, rule engines for subject matter experts to write business
rules, and user interfaces for displaying model results.  However, the bulk of
the development work on these products involves not the core NLP components,
but data manipulation tasks like converting between the output formats of
different models; manipulating training data; analyzing the outputs of models
for correctness; and serializing data for transfer across programming language
and machine boundaries.

Although the raw input to our NLP algorithms is natural language text, most of
the code in our NLP systems operates over machine data. Examples of this
machine data include:

* Relational tables of training data in formats like CoNLL-U :cite:`DBLP:journals/corr/abs-2004-10643`
* Model outputs formatted as tables for comparison against training data
* Arrays of dense tensors that represent BERT embeddings :cite:`DBLP:journals/corr/abs-1810-04805`
* Trees that represent dependency parses 
* Relational tables that represent document structure

This focus on mundane engineering tasks at the expense of core AI algorithms is
not unique to IBM, or indeed to natural language processing.

*TODO: Cite the "Technical Debt in Machine Learning" paper and note Google does
a lot of NLP*

However, NLP is unique in the quantity of redundant data structures and
low-level algorithms that different systems reimplement over and over again.
One can see this trend clearly in open source NLP libraries, where free access
to internal code also exposes the internal data structures.  Each of the major
NLP libraries implements its own custom data structures for basic NLP concepts. 

Consider the concept of a *span*: a region of a document, usually expressed as
a range of characters or tokens.  NLP systems use spans to represent the
locations of information they extract from text. This information includes
tokens, named entities, arguments to semantic role labeling predicates, and
many others.

Here is how some popular Python NLP libraries represent spans:

* ``SpaCy`` has a Python class ``Span`` that represents a range of tokens. The
  locations of these tokens are stored inside the class ``Doc``. The  
  ``__getitem__`` method of ``Doc`` returns instances of the class ``Token``, which
  encodes the location of the token as a beginning character offset and a
  length in characters.
* ``Stanza`` has a Python class ``Span`` that represents a range of *characters*.
  Information about the tokens that are contained within the character range
  are stored in the ``tokens`` property of the ``Span`` as objects of type
  ``Token``.  These classes ``Span`` and ``Token`` are different from the
  ``SpaCy`` classes of the same names.
* ``nltk`` models text as a Python list. The elements of the list can be Python 
  strings or tuples, depending on the stage of processing. Spans over 
  tokens are represented by slices of the list. Character location information
  is not generally available.
* ``transformers`` does not generally model spans, instead leaving that aspect
  up to the user.  One exception is the library's
  ``TokenClassificationPipeline`` class, which has a method ``group_entities`` that
  returns a Python dictionary for each entity.  The fields ``start`` and
  ``end`` in this dictionary hold the span of the entity, measured in
  characters.
* *TensorFlow Text* represents lists of spans as either a pair of
  one-dimensional tensors (for tokenization) or as a single two-dimensional
  tensor (for span comparison operations).  The elements of the tensors can
  represent byte, character, or token offsets.  Users need to track which type
  of offset is stored in a given tensor.


*TODO: Citations for API docs of SpaCy, Stanza, nltk, transformers, TensorFlow
Text*

All of these representations are incompatible with each other. Users who want
to use two of these libraries together will need to write code to convert
between their outputs.  Users are also left to invent their own algorithms for
even the most basic operations over spans, including serializing them, finding
their covered text, determining whether two spans overlap, and finding matches
between two sets of spans.

The redundancy that these libraries display at the level of individual spans is
pervasive across all the more complex structures that they extract from text.
Users and library developers both spend considerable amounts of time reading
the documentation for these different data structures, writing code to convert
between them, and reimplementing basic operations over them.


An Alternative Approach
-----------------------

The Python data science community has developed effective tools for managing
and analyzing data in memory, chief among them being the DataFrame library
*Pandas*.

*TODO: Cite Pandas paper*

Could we use these general-purpose tools instead of continually reinventing
data structures and algorithms for basic NLP tasks?

We prototyped some use cases and quickly discovered that NLP-related data
involves domain-specific concepts; and some of these concepts are inconvenient
to express in Pandas.  For example, the *span* concept that we described in the
previous section is a crucial part of many applications.  The closest analog to
a span in Pandas' data model is the ``interval`` type, which represents an
inteval using a pair of numbers.  When we prototyped some common NLP
applications using ``interval`` to represent spans, we needed additional code
and data structures to track the relationships between intervals and target
strings; as well as between spans and different tokenizations.  We also needed
code to distinguish between intervals measured characters and in tokens. All of
this additional code negated much of the benefit of the general-purpose tool.

To reduce the amount of code that users would need to write, we started working
on extensions to Pandas to better cover represent NLP-specific data and to
support key operations over that data.  We call the library that we eventually
developed *Text Extensions for Pandas*.

Extending Pandas
++++++++++++++++

Text Extensions for Pandas includes three types of extensions:

* NLP-specific **data types (dtypes)** for Pandas DataFrames
* NLP-specific **operations** over these new data types
* **Integrations** between Pandas and common NLP libraries

Pandas includes APIs for library developers to add new data types to Pandas,
and we used these facilities to implement the NLP-specific data types in Text
Extensions for Pandas.

The core component of the Pandas extension type system is the *extension
array*. The Python class ``pandas.api.extensions.ExtensionArray`` defines key
operations for a columnar array object that backs a Pandas ``Series``.  Classes
that extend ``ExtensionArray`` and implement a relatively short list of
required operations can serve as the backing stores for Pandas ``Series``
objects while supporting   support nearly all the operations that Pandas
built-in types support, including filtering, slicing, aggregation, and binary
I/O.

*TODO: Citation or hyperref to "extending Pandas" documentation*

Indeed, many of the newer built-in types in Pandas, such as the ``interval`` 
and ``categorical``, are implemented as subclasses of ``ExtensionArray``.
Text Extensions for Pandas includes three different extension types based on
this API. The first two extension types are for spans with character- and
token-based offsets, respectively. The third extension type that we add
represents tensors.

Spans
-----

We implement character-based spans with a Python class called ``SpanArray``,
which derives from Pandas' ``ExtensionArray`` base class.  A ``SpanArray``
object represents a column of span data, and it stores this data internally
using three Numpy arrays, plus a shared reference to the underlying text.

*TODO: Insert a diagram of the layout of a TokenSpanArray with the backing
SpanArray, plus the DataFrame representation that the user sees*

The three arrays that represent a column of spand data consist of arrays of
begin and end offsets (in characters), plus a third array of indices into a
dictionary of unique document texts. The ``SpanArray`` object also stores a
shared reference to a dictionary data structure that tracks unique document
texts.

The dictionary data structure is necessary because a Pandas series can contain
spans from multiple different documents.  Users need to be able to perform
operations over the containing DataFrames without creating many copies of the
text of each document.  Dictionaries are append-only and are shared among
SpanArray objects to facilitate zero-copy operations like filtering and slicing.

In addition to spans with character offsets, we also support spans whose begin
and end offsets are measured in tokens.  Token-based spans are a useful
construct because most machine learning models and rule engines for NLP operate
over tokens, not characters.  Evaluation metrics for model result quality also
tend to operate over tokens.  Representing spans with token offsets can
facilitate operations like computing token distances between spans and can
prevent errors that could lead to spans not starting or ending on a token
boundary.

There can be multiple different tokenizations of the same document, even within
a single application. When storing token-based span offsets, it is important to
retain information about which tokenization of which document each token offset
corresponds to.  The ``TokenSpanArray`` class represents each distinct
tokenization of a document with an instance of ``SpanArray`` containing the
locations of the tokens.  The representation of the token-based spans
themselves consists of three Numpy arrays, holding begin and end offsets (in
tokens) and a pointer to the ``SpanArray`` containing the token offsts.

Although it stores the locations of spans as token offsets, the
``TokenSpanArray`` class can generate character-based begin and offsets on
demand from its internal tables of token locations.  This facility allows
``TokenSpanArray`` to be used in any code that works over instances of
``SpanArray``. For example, code that detects pairs of overlapping spans can
easily work over arbitrary combinations of token- and character-based spans,
which is useful when merging the outputs of models that represent span offsets
differently. 

The internal structure of our ``SpanArray`` and ``TokenSpanArray`` extension
arrays allows for efficient vectorized implementations of common Pandas
oeprations like slicing, filtering, and aggregation.  Slicing operations over a
``SpanArray`` produce a new ``SpanArray`` with views of the original
``SpanArray`` object's internal Numpy arrays, avoiding unneccessary copying of
span data.


Tensors
-------

*Tensors* |---| dense n-dimensional arrays |---| are another common concept in
modern natural language processing.  The deep learning models that drive much
of state-of-the-art NLP today take tensors as inputs and outputs and operate
internally over other tensors.  Embeddings, a key part of many NLP algorithms,
can be efficiently represented with tensors.  Tensors are also useful for more
traditional types of NLP data, such as n-grams and one-hot-encoded feature
vectors.

Our ``TensorArray`` extension array class represents a Pandas series where each
element is a tensor.  Internally, we represent the entire series' data as a
single dense NumPy array The TensorArray class translates Pandas array
operations to vectorized operations over the underlying Numpy array.  These
vectorized operations are much more efficient than iterating over a list of
tensors.

*TODO: Add some more details about the capabilities of TensorArray*

There are other libraries that provide Pandas-like dataframes specialized for
numeric tensor or array data these libraries are useful for cases where
dataframes consist almost entirely of tensor data.

*TODO: list and cite other libraries*

Our TensorArray extension type is a complementary alternative for applications
where the data is a mixture of tensors, spans, and built-in Pandas data types
with a wide variety of different schemas. For these applications, our tensor
type allows users to leveage Pandas' collection of built-in operations and
third-party visualizations, while still operating efficiently over
tensor-valued data series.

*TODO: Screenshot of DataFrame with tensors and spans and a segue between it
and the preceding paragraph*

Serialization
-------------

Many areas of modern NLP involve large collections of documents, and common NLP
operations expand the size of data by orders of magnitude

*TODO: Cite the huge corpus used to train BERT, and mention some specifics*

Pandas includes facilities for efficient serialization of Pandas data types
with Apache Arrow, and Text Extensions for Pandas uses these facilities to
allow NLP data to be stored in Arrow records for efficient storage and
transfer.

*TODO: Comparison of serialization size and time between us and a few libraries
for CoNLL-2003*

We also support reading files in the text based CoNLL formats and the related
CoNLL-U format.  Many benchmark data sets for natural language processing are
released in these formats.

*TODO: Example code that reads in CoNLL-2003 data*


Spanner Algebra
---------------

In addition to representing span data, NLP applications need to filter,
transform, and aggregate this data, often in ways that are unique to natural
language processing.

The *document spanners* formalism :cite:`10.1145/2699442` extends the
relational algebra with additional operations to cover a wide gamut of
critical NLP operations.

Since it is an extension of the relational algebra, much of document spanners
can already be expressed with Pandas core operations.  We have implemented
several of the remaining parts of document spanners as operations over series
of type Span.

Specifically, we have NLP-specific *join* operations (sometimes referred to as
"merge") for identifying matching pairs of spans from two input sets, where the
spans in a matching pair have an overlap, containment, or adjacency
relationship.  These join operations are crucial for combining the results of
multiple NLP models, and they also play a role in rule-based business logic.
For example, a domain expert might need to filter out matches of one model that
overlap with matches of a different model.

TODO: Example code for the above use case

We include two implementations of the *extract* operator, which produces a set
of spans over the current document that satisfy a constraint.  Our current
implementations of *extract* support extracting the set of spans that match a
regular expression or a gazetteer (dictionary).

We also include a version of the *consolidate* operator, which takes as input a
set of spans and removes overlap among the spans by applying a consolidation
policy.  This operator is useful for business logic that combines that results
of multiple models and/or extraction rules as well as for resolving ambiguity
when a single model produces overlapping spans in its output.


TODO: Use the proper mathematical notation for the operators in the preceding
paragraphs

Other Span Operations
+++++++++++++++++++++

We also support span operations that are not part of the document spanners
formalism but are important for key NLP tasks:

* aligning spans based on one tokenization of the document to a different
  tokenization

* lemmatizing spans -- converting the text of the span to a normalized form

* converting between spans and tokens with inside-outside-beginning (IOB) tags


Jupyter Notebook Integration
----------------------------

Jupyter notebooks have built-in facilities for displaying Pandas DataFrames.
Our extensions to Pandas also work with these facilities.

If the last line of a notebook cell returns a DataFrame containing span and
tensor data, then Jupyter will display an HTML representation of the DataFrame
with the 

TODO: Example of displaying a DataFrame with Span and Tensor data
(maybe share this figure with another section)

Other Python development tools, including Visual Studio Code, PyCharm, and
Google Colab, use extended versions of the Jupyter DataFrame display facilities
to show DataFrames in their own user interfaces. Our extension types also work
with these interfaces.

TODO: Example of displaying a DataFrame in PyCharm

There is also an ecosystem of interactive libraries for exploring and
visualizaing Pandas DataFrames.  These libraries also work with our extension
types.

TODO: Screenshot of displaying a DataFrame of span information with D-tale

Because our extension types for tensors use Numpy's `ndarray` type for
individual cell values, 

TODO: Example of using matplotlib to render time series from a Pandas Series of
tensors in a Jupyter notebook

The 

TODO: Word cloud from the `text` attribute

It is often useful to visualize spans in the context of the source text.

We use Jupyter's built-in application programming interface (API) for HTML
rendering to 

If the last expression in a notebook cell returns a `SpanArray` or
`TokenSpanArray` object, then Jupyter will automatically display the spans in
the context of the target text.  

TODO: Example of asking a series of spans to display itself 

Taken together with JupyterLab's ability to display multiple widgets and views
of the same notebook, these facilities allow users to visualize NLP data from
several perspectives at once.

for example, this UI for flagging and examining potentially incorrect labeling

TODO: Screenshot of relabeling 


NLP Library Integrations
------------------------

Text Extensions for Pandas provides facilities for transforming the outputs of
several common NLP libraries into Pandas DataFrames, using our extensions to
Pandas to represent NLP concepts.


SpaCy
+++++

Our SpaCy integration converts the output of a SpaCy language model into a
DataFrame of token information.

TODO: Citation of SpaCy

Converting from SpaCy's internal representation to DataFrames allows users to
use Pandas operations to analyze and transform the outputs of the language
model.

For example, users can use Pandas' grouping and aggregation to count the number
of nouns in each sentence:

TODO: Code example

Or they could use our span-specific join operations and the Pandas `merge`
function to match all pronouns in the document with the person entities that
are in the same sentence:

TODO: Code snippet

We also support using SpaCy's `DisplaCy` visualization library to display
dependency parse trees stored in DataFrames.  Users can filter the output of
the language model using Pandas operations, then display the resulting subgraph
of the parse tree in a Jupyter notebook:

TODO: Picture of visualizing a partial parse tree, with code

This display facility will work with any DataFrame that encodes a dependency
parse as Pandas Series of token spans, token IDs, and head IDs.


`transformers`
++++++++++++++

Our integration with the `transformers` library for transformer-based masked
language models

TODO: Citation of `transformers`

tokenize text using the subword tokenizations commonly used for models such as
BERT and convert 


TODO: Example for transformers


IBM Watson Natural Languague Understanding
++++++++++++++++++++++++++++++++++++++++++

Watson Natural Language Understanding is a RESTful API that provides access to
prebuilt NLP models for common tasks across a wide variety of natural
languagues.  Users can use these APIs to process several thousands documents per
month for free, with paid tiers of the service available for higher data rates.

Our Pandas integration with Watson Natural Language Understanding can translate
the outputs of all of Watson Natural Language Understanding's information
extraction models into Pandas DataFrames. The supported models are:

* `syntax`, which performs syntax analysis tasks like tokenization,
  lemmatization, and part of speech tagging.
* `entities`, which identifies mentions of named entities such as persons,
  organizations, and locations.
* `keywords`, which identifies instances of a user-configurable set of keywords
  as well as information about the sentiment that the document expresses
  towards each keyword.
* `semantic_roles`, which performs *semantic role labeling*, extracting
  subject-verb-object triples that describe events that occurred in the text.
* `relations`, which identifies relationships betwen pairs of named entities

TODO: Example for Watson NLU 

IBM Watson Discovery
++++++++++++++++++++

One of the key features of the IBM Watson Discovery product is *Table
Understanding*, a document enrichment model that identifies and parses
human-readable tables of data in PDF and HTML documents.

Text Extensions for Pandas can convert the output of Watson Discovery's Table
Understanding enrichment into Pandas DataFrames.

This facility allows users to reconstruct the contents and layout of the
original table as a DataFrame, which is useful for debugging and analysis of
these outputs.

TODO: Before and after picture of a table

Our conversion also produces a the "shredded" representation of the table as a
DataFrame with one line for each cell of the original table. This data format
facilitates data integration and cleaning of the extracted information.  Pandas'
facilities for data cleaning, filtering, and aggregation are extremely useful
for turning raw information about extracted tables into clean, deduplicated
data suitable to insert into a database.

TODO: Table extracted from 10 years of IBM annual reports



Usage in NLP Research
---------------------

we are using Text Extensions for Pandas in ongoing research on semisupervised
identification of errors in NLP corpora.

Pandas' data analysis facilities for provide a powerful substrate
for cross-referencing and analyzing the outputs of NLP models in order to
pinpoint 

one example of this type of application is work that we and several other
coauthors recently published on correcting errors in the highly-cited
CoNLL-2003 corpus for named entity recognition.

TODO: Is there space to include a detailed description of the application?

our paper in CoNLL 2020 describes the corrections in detail, as well as results
from revisiting key results

TODO: Citation of CoNLL-2003 dataset
TODO: Citation of CoNLL paper




Community
---------

The source code for Text Extensions for Pandas is available at
`<https://github.com/CODAIT/text-extensions-for-pandas>`_.  We welcome
community contributions to the code as well as feedback from users about bugs
and feature requests.

We also publish installable packages on the PyPI and Conda-Forge package
repositories. Since our library is implemented in pure Python, these packages
work on most operating systems.


