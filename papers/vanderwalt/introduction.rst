:author: Stéfan van der Walt
:email: stefan@sun.ac.za
:department: Helen Wills Neuroscience Institute
:institution: University of California, Berkeley

:author: Jarrod Millman
:email: millman@berkeley.edu
:department: Helen Wills Neuroscience Institute
:institution: University of California, Berkeley

.. |emdash| unicode:: U+02014
   :trim:

-------
Preface
-------

Introduction
------------
Welcome to the fourth publication of the SciPy conference proceedings.  This
year marks the 10th Python in Science Conference (SciPy2011), held in Austin,
Texas from July 11th to 16th where more than 190 participants, in a truly
global attendance from North and South America, Europe, Africa, and
Asia, represented both academia and industry.

Ten years and counting ...
--------------------------
The first "Python for Scientific Computing Workshop" was held at Caltech in
2002--a single track, 2-day workshop with about 70 attendees.  Since then, the
number of attendees nearly tripled; the 2-days of talks are now preceeded by 2
days of tutorials and followed by 2 days of developer sprints.  Multiple
conference tracks accommodate the increasing number of high-quality talk
submissions, and satellite conferences are held annually in Europe and India.
We give a brief overview of these developments:

In 2004 the workshop is renamed the SciPy conference and extended by 2 days of
developer sprints; the number of attendees increases to about 90.  The first
keynote is given by Jim Hugunin, creator of numeric (the precursor to NumPy),
Jython, IronPython, and co-designer of AspectJ.

In 2005, the community is split between two different core numerical libraries
-- numeric and numarray, making code exchange and shared development hard.
Travis Oliphant, who had been working on a successor to both libraries, opens
the conference with a presentation of the new core package now known as
numpy. In order to move the new library forward, the number and length of talks
are limited, and additional break sessions are held for informal discussions.

By 2006, NumPy approaches a 1.0 release, with support from a re-united
community. Guido van Rossum, the creator of Python, delivers the keynote, and
tutorials appear for the first time.  The number of attendees grows to 138.

The 2007 conference includes the first open call for student sponsorship,
continued and expanded every year since.  Ivan Krstić, then the director of
security architecture at One Laptop per Child, gives the keynote.

The 2008 keynote is delivered by Alex Martelli, "Über Tech Lead" for Google,
perhaps better known in the Python community as the MartelliBot.  A
peer-reviewed conference proceedings appears, and EuroSciPy is held for the
first time.

In 2009, keynotes are delivered by Peter Norvig, Director of Research at
Google, and Jon Guyer, Materials Scientist at NIST.  In addition to the second
European SciPy conference, the 1st SciPy India is held.

The 2010 conference moves to Austin, TX, has multiple tracks and 187
participants.  Keynotes are delivered by David Beazley (the author of the
Python Essential Reference and creator of SWIG) and Travis Oliphant (then
President of Enthought, Inc. one of the original co-authors of SciPy).

2011 Conference
---------------
This year's conference consisted of several events: tutorials, followed by the
main conference and sprints, with some birds-of-a-feather sessions sprinkled
throughout.  Aric Hagberg and the NetworkX team also held a satellite workshop
preceding the main conference.

Tutorials
~~~~~~~~~
Tutorials were divided into an introductory and an advanced track.
This year's introductory track started with Jonathan Rocher's overview
of numpy, ipython and matplotlib, followed by an introduction to SciPy
by Anthony Scopatz. Mateusz Paprocki and Aaron Muller, both from
SymPy, showed how to use their project to do symbolic computation, and
Corran Webster from Enthought illustrated the power of Matplotlib,
Traits and Chaco.  Traits, of course, being Enthought's popular GUI
building tool.

On the advanced end we had Chris Fonnesbeck and Abie Flaxman discuss
PyMonteCarlo for statistical modelling, followed by Gaël
Varoquaux's overview of scikits.learn, a collection of machine
learning algorithms.  On the parallel computing front Jeff Daily spoke
about the Global Arrays Toolkit.

The last tutorial was given by Min Ragan-Kelley, who taught us how
IPython can now be used to perform interactive parallel computing.

Keynote speakers
~~~~~~~~~~~~~~~~
The first keynote address was delivered by Eric Jones, a founder of Enthought
and one of the first SciPy authors.  In his talk, titled "What matters in
Scientific Software Projects: 10 Years of Success and Failures Distilled",
Eric touched on some of the lessons we've learnt during the past decade.  For
example, he mentioned the rare skills intersection that we currently have
between scientists or engineers and computer scientists, and how valuable
those skills are in today's world.  He also emphasised the importance of
engaged participation with your customer or users, in order to develop
practical and useful software specification; and then the trust required to
build such relationships.  Finally, he warned against over-emphasising
development process, which often has less of an impact on success than we'd
like to admit.  Process cannot substitute for intelligent reasoning brought to
a project by talented people.

The second Monday plenary, "How the Digital Age Affects Research" was
delivered by Kaitlin Thanay.  Kaitlin has an interesting background as manager
of the science division of Creative Commons, and is now with Digital Science,
a company that spun out of Nature.  She started her talk by introducing what
some think to be the first computation machine: the Antikythera mechanism.
Her talk then followed the interaction of science and technology as both
developed, and how this influences the research cycle.  This talk is
especially relevant to scientists at institutions that place a high premium on
publishing.

On Wednesday morning, Hilary Mason from bit.ly spoke about "Science in the
Service of Awesome".  bit.ly generates about 1% of all new URLs that appear on
the internet per day, and hosts more than 10**9 unique URLs.  Interestingly,
they do quite a bit of data science and generate statistics on things like
click-throughs etc.

The last plenary was given by Perry Greenfield, who is a familiar face at
SciPy conferences all around.  Perry currently leads the Science Software
Branch of the Space Telescope Science Institute, and is one of the pioneers of
Python in astronomy.  His talk, "How Python Slithered into Astronomy", a
follow-up of his talk delivered at SciPy India last year, looked at the
history of Python in this very active field.  The astronomy community has been
and continues to be central to the development of tools such as NumPy.

Talks
~~~~~
The conference had three sessions this year, and at any one time two
would run in parallel.

Anthony Scopatz chaired the Python and Core Technologies session,
which was an effort to bring science related talks submitted to
PyCon to this conference.  Peter Wang, author of Chaco and now with
Streamitive, hosted the session on data science |emdash| which has become
quite a trending topic in Python, if I may borrow from the Twitter
nomenclature.

Highlighting the growing role of Python in statistical computing was
Wes McKinney's talk titled "Time Series Analysis in
Python with `statsmodels`".  Wes, Skipper
Seabold, Josef Perktold, Jonathan Taylor and others have made huge improvements in
the statistics tools available for Python, as
witnessed by all the other stats talks.  In addition, there was also a
birds-of-a-feather session, following up from last year's on DatArray,
discussing the best Python data structures for representing
statistical data.

Scott Determan's Vision Spreadsheet is a lot of fun to watch in
action, so check it out at

    ::

      http://visionspreadsheet.com

After a couple of years' silence from the IPython team, we were blown
out of the water by Fernando Perez's report on their progress.  He
introduced all the new features in the recently released IPython 0.11,
such as the swanky new Qt and web-based front-ends, and parallel
computation backed by 0MQ.

Two other talks from the Python Core Technologies talks that I enjoyed
were "Twiggy: A Pythonic Logger" by Pete Fein, and "Lessons Learned
Building a Scalable Distributed Storage System in Python" by Chuck
Thier.  The first, because I have a long running feud with the Python
standard library's logging module, and the second because it is great
to know that you can implement crazy things like Distributed Storage
Systems in Python.

Gaël Varoquaux gave a great talk on Python for Brain Mining, but his
tool that interested me most was Joblib--a lightweight system for
building scientific pipelines.  In the quest for reproducibility, this
is a great find!

Then, on the business side of things, Josh Heman from Sports Authority
spoke about the challenges of getting Python into a Multi-billion
Dollar Enterprise.

As usual, we also had the very entertaining lightning talks session on
the last day of the conference.  Travis Oliphant's talk on "What I
would like to see in NumPy" should probably have made its way into the
main conference, so find his slides online if you'd like to have a
look at some of the ideas that may well form part of NumPy 2.0.

The slides and videos for all these talks and many others are
available from the conference website.

Sprints
~~~~~~~

With open source development, bringing developers to the same physical
location can be challenging.  The sprints provide one ideal such an
opportunity, during which developers can have some good old
face-to-face conversations, and put their brains together to solve
long-standing problems or implement exciting new features.  It is
interesting that, in today's connected world, these real-world
conversations still have such a large impact.


Special issue: IEEE Computing in Science and Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attendees this year were given copies of the March/April 2011 issue of
Computing in Science and Engineering.
This was a special issue on Python for Scientists and Engineers
Computing in Science & Engineering or CiSE
CiSE is a peer-reviewed technical magazine
jointly published by the American Institute of Physics and the IEEE Computer
Society

A follow-up to May/June 2007 special issue,
“Python: Batteries Included.”
The 2007 issue had a couple articles presenting the core
Python libraries for scientific computing:
numpy, scipy, ipython, and matplotlib.
As well as a series of shorter pieces presenting specific
scientific, engineering, and educational applications.

This year's special issue had fewer, but longer articles
focusing on some of the more advanced features of
the core stack of scientific tools for Python.

1st article
“Python: An Ecosystem for Scientific Computing,”

core stack of tools developed specifically for scientific computing
that makes Python such a highly productive environment for modern scientific
computing.

The next two articles focus on two complimentary approaches to improving the
efficiency of Python code while retaining Python’s ease of use.

“The NumPy Array: A Structure for Efficient Numerical Computation,”
describe how NumPy provides a high-level multidimensional
array structure, that also allows fine-grained
control over performance and memory-management.

In “Cython: The Best of Both Worlds,”
discuss this popular tool for creating
Python extension modules in C, C++, and Fortran.

“Mayavi: 3D Visualization of Scientific Data,”
a 3D scientific visualization package for Python.
simple scripts to visualize their data; to load and
explore their data with a full-blown interactive,
graphical application; and to assemble their own
custom applications from Mayavi widgets.


Proceedings
-----------

Each of the **XX** submitted abstracts was reviewed by both the program chairs
and two additional members of the program committee. The committee consisted of
**XX** members from **XX** countries, and represented both industry and academia.

Abstracts were evaluated according to the following criteria:
applicability, novelty, and general impression.

We accepted **XX** submissions for oral presentation at the conference. At the end
of the conference, we invited the presenters to submit their work for
publication in the proceedings. These submissions were reviewed by **XX**
proceedings reviewers from **XX** countries.  Each reviewer provided an overall
score for each reviewed paper, based on the quality of approach and writing.
Reviewers were also asked to provide more specific feedback according to the
questionnaire shown in the appendix.

The tools used to produce this document are made available under an open source
license, and may be obtained from the code repository at
https://github.com/scipy/scipy-proceedings.


Acknowledgements
----------------

A conference the size of SciPy is only possible through the hard work and
dedication of a large number of volunteers.  Once again Enthought
provided significant administrative support.

We thank sponsors 

These proceedings are the result of many hours of work by authors and reviewers
alike.  We thank them for their significant investment in these manuscripts.
The names of all contributers are listed in the "Organization" section, which
forms part of the cover material.

----------

Appendix: Reviewer Questionnaire
--------------------------------

- Is the code made publicly available and does the article sufficiently
  describe how to access it?  We aim not to publish papers that essentially
  advertise propetiary software.  Therefore, if the code is not publicly
  available, please provide a one- to two- sentence response to each of the
  following questions:

  - Does the article focus on a topic other than the features
    of the software itself?
  - Can the majority of statements made be externally validated
    (i.e., without the use of the software)?
  - Is the information presented of interest to readers other than
    those at whom the software is aimed?
  - Is there any other aspect of the article that would
    justify including it despite the fact that the code
    isn't available?
  - Does the article discuss the reasons the software is closed?

- Does the article present the problem in an appropriate context?
  Specifically, does it:

  - explain why the problem is important,
  - describe in which situations it arises,
  - outline relevant previous work,
  - provide background information for non-experts

- Is the content of the paper accessible to a computational scientist
  with no specific knowledge in the given field?

- Does the paper describe a well-formulated scientific or technical
  achievement?

- Are the technical and scientific decisions well-motivated and
  clearly explained?

- Are the code examples (if any) sound, clear, and well-written?

- Is the paper factually correct?

- Is the language and grammar of sufficient quality?

- Are the conclusions justified?

- Is prior work properly and fully cited?

- Should any part of the article be shortened or expanded?

- In your view, is the paper fit for publication in the conference proceedings?
  Please suggest specific improvements and indicate whether you think the
  article needs a significant rewrite (rather than a minor revision).
