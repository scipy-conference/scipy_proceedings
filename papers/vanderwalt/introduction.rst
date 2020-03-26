:author: Stéfan van der Walt
:email: stefan@sun.ac.za
:institution: Stellenbosch University

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

Welcome to the third publication of the SciPy conference proceedings.  This
year marked the 9th Python in Science conference (SciPy2010), which was held in
Austin, Texas from June 28th to July 3rd 2010.  The conference was attended by
187 participants from both academia and industry.  Attendees came from far and
wide, including North America, Columbia, Spain, South Africa, Turkey, Germany,
Norway, Italy, Singapore, and Great Britain.

For the first eight years, the conference was held at the California Institute
of Technology.  After the 2009 conference, we began looking for a new venue to
enable continued growth of the event.  With Enthought playing a prominent
organizational role, we chose the AT&T Conference Center in Austin, Texas.  We
thank our colleagues and co-organisers at Caltech for all their support and
effort over the years, and for playing such an integral part in establishing
this event.

What started as a small, informal meeting is now a world-wide series of
conferences, with the third EuroSciPy held at the Ecole Normale Supérieure in
Paris from July 8th to the 11th as well as the second SciPy India in Hyderabad,
Andra Pradesh from December 13th to 18th.

Conference overview
-------------------

The conference kicked off with a keynote by David Beazley on *Python
Concurrency*.  David is well known in the Python community as an author,
speaker, and trainer.  From his early career in scientific computing, David is
well known as the author of SWIG |emdash| a popular tool that generates C++/Python
wrappers.

The second keynote, *Moving Forward from the Last Decade of SciPy*, was
delivered by Travis Oliphant.  From Travis's unique position as the original
author of NumPy and a founding contributor to SciPy, he provided an historical
overview of the growth of the community and tool chain, and shared his
vision for potential future development.

The new, larger venue allowed us to host parallel tracks for the first time; in
addition to the main conference, we had specialized tracks in *bioinformatics*
as well as *parallel processing and cloud computing*.

The latter was a central topic of the conference, dealt with in several papers
in these proceedings.  At the conference, tutorials on the topic included *High
Performance and Parallel Computing* by Brian Granger and *GPUs and Python* by
Andreas Klockner.

The continuing strong presence of the astronomy community was evident at the
conference, and is also reflected in the proceedings, with papers on projects
such as the Chandra, James Webb, and Hubble telescopes.

The conference saw a rising interest in statistical computing and biological
modelling.  Of particular note are the papers *Statsmodels: Econometrics and
Statistical Modeling with Python* by Skipper Seabold and Josef Perktold, and
*Data Structures for Statistical Computing in Python* by Wes McKinney.

The conference was followed by the traditional two days of sprints.  The new
location provided a number of rooms where groups could interact, helping to
draw in some of the smaller projects.  It was a productive time for all
involved, and led to improvements in projects such as *NumPy*, *SciPy*,
*IPython*, *matplotlib*, *scikits.learn*, *scikits.image*, *theano*,
and several others.

Proceedings
-----------

Each of the 33 submitted abstracts was reviewed by both the program chairs and
two additional members of the program committee. The committee consisted of 11
members from 6 countries, and represented both industry and academia.

Abstracts were evaluated according to the following criteria:
applicability, novelty, and general impression.

We accepted 32 submissions for oral presentation at the conference. At the end
of the conference, we invited the presenters to submit their work for
publication in the proceedings. These submissions were reviewed by 14
proceedings reviewers from 8 countries.  Each reviewer provided an overall
score for each reviewed paper, based on the quality of approach and writing.
Reviewers were also asked to provide more specific feedback according to the
questionnaire shown in the appendix.

Due to time constraints of the editors, as well as a complete rewrite of the
publication framework, completion of this year's proceedings took longer than
anticipated.  The new framework, however, is much easier to use and extend and
will enable us to publish future editions much more rapidly. The tools used to
produce this document are made available under an open source license, and may
be obtained from the code repository at
https://github.com/scipy/scipy-proceedings.


Acknowledgements
----------------

A conference the size of SciPy is only possible through the hard work and
dedication of a large number of volunteers.  Once again Enthought
provided significant administrative support.  In particular, we would like to
thank Amenity Applewhite, Jodi Havranek, and Leah Jones, who not only carried a
significant administrative burden, but also did the enormous footwork required
in securing a new venue, negotating vendor prices, and numerous other tasks
both small and large due to the move from Caltech to Austin this year.

We thank Enthought, Dell, Microsoft, D.E. Shaw & Co., AQR Financial Management,
the Python Software Foundation, and one anonymous donor, for funding 14
students to travel and attend SciPy 2010.  We also acknowledge our media
sponsor, the IEEE/AIP Computing in Science and Engineering magazine, for
publicizing the conference and providing magazines to participants.

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
