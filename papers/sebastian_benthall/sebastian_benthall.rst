:author: Sebastian Benthall
:email: sb@ischool.berkeley.edu
:institution: UC Berkeley School of Information

--------------------------------------------------------
Generative Modeling of Online Collaboration with BigBang
--------------------------------------------------------

.. class:: abstract

   We introduce BigBang, a new Python toolkit for analyzing 
   online collaborative communities such as those that 
   build open source software.
   Mailing lists serve as critical communications infrastructure for
   many communities, including several of the open source software 
   development communities that build scientific Python packages.
   BigBang provides tools for analyzing mailing lists.
   In this paper, we present results from testing a generative
   model of online collaboration in mailing lists.
   We derive a social network fro archival mailing list history
   and test the Barabasi-Alpert model of network formation
   against this data.
   We find the model does not fit the data, but that mailist list
   social networks share statistical regularities not explained in
   existing complex systems literature.
   This suggests room for a new generative model of network formation
   in the open collaborative setting.

.. class:: keywords

   mailing lists, clustering


Introduction
------------

Open source software communites such as those that produce many scientific 
Python packages
are a critical part of contemporary scientific organization.
A distinguishing feature of these communities is their heavy use of
Internet-based infrastructure, such as mailing lists, version control systems, and
issue trackers, for managing communications and organizing work on distributed teams.
This data is often deliberately publicly accessible as open source best practices
include the "conspicuous use of archives" (cite Fogel).
The availability of these digital records are also an excellent resource for
the researcher interested in sociotechnical organization and collaboration
within science.

This paper introduces BigBang, a Python project whose purpose is the collection,
preprocessing, and analysis of data from open collaborative communities.
Section I will provide an introduction to the project's goals and technical 
organization.
Section II will outline some of the features available in the present version 
of the software.
Section III will illustrated with an example how BigBang has been used to explore
a novel clustering algorithm based on.

BigBang Overview
----------------

BigBang is a software project that aims to provide researchers a complete
toolkit for the scientific analysis of open online collaborative communities.
Launched last year and presented at SciPy 2014, our team has made progress in 
building BigBang as a platform for computational social science and reflexive data science. 
We will present results gained from using BigBang to study scientific Python communities in 
comparison to other open source projects and social data.
The study of online collaboration is a subdomain of the social sciences that
is of critical importance to the practice of scientific computing as so many
of the tools used in scientific computing are developed largely through
online collaboration.

Thorough study of these kinds of communities requires the collection and
rationalization of many heterogenous and high-dimensional data sources,
including but not limited to:

- mailing lists
- version control systems such as Git
- issue trackers such as GitHub and Bugzilla

These are complex data sources in that they have the following dimensions:

- time. data are timestamped
- text. email message bodies, issue contents, and commit messages in version
  control are all text suitable for natural language processing techniques
- social network. participants in the project are individuals linked by relational
  ties of communication. Hence these data afford study through social
  network analysis techniques.

In order to facilitate study of collaborative dynamics (which I should totally cite
prior work on eg. Thomas's syllabus...) a variety of techniques may be brought to bear
on this data.
Since these techniques are widely available in various scientific open source packages,
BigBang's role is to import and apply these packages to the available data and demonstrate
the success of various approaches.
These demonstrations are available in the project's ``examples`` directory, which contains
many Jupyter notebooks that illustrate BigBang's functionality.

In the context of Scientific Python, BigBang is deliberately recursive.
It is a scientific Python project that depends on many other scientific Python projects
that can study they dependencies and interactions between the scientific Python
technologies and communities.
We envision BigBang in providing a new means for these communities, among others,
to engage in scientific self-management, thus bringing into being a harmonious
sociotechnical superintelligence that takes over the world.

Features
--------

BigBang currently supports many kinds of analysis on data from open source
communities.

BigBang supports collection of email data.
It can do this either by scraping the archival pages of a Mailman instance,
or by importing an `.mbox` formatted file. On email data, BigBang includes functionality for the following operations:


*Cohort clustering*.
- by quintile
- by spectral analysis, see below


The conundrum in network science is how to define generative processes for
networks that create networks with the empirical properties of networks
found in nature.
As these interaction graphs are anomalous with respect to the existing
literature on complex systems (to the best of my knowledge), modeling these
networks presents a challenge that will be undertaken in future work.


Deriving interaction graphs
---------------------------

Email is archived in the same text format that email is sent in, as specified in
RFCXYZ (cite).
Every email is comprised of a message body and a number of metadata headers
that are essential for email processing and presentation.

For our study, we have been interested in extracting the implied social
network from an email archive of a public mailing list. To construct this
network, we have used the `From`, `Message-ID`, and `In-Reply-To` headers
of the email.

The `From` field of an email contains data about the sender of the message.
This often includes both a full name and an email address of the sender.
As this is set by the email client (note: or is it the mailing list server?)
a single person is often represented differently in the `From` field
across many emails. See *Entity Resolution* for our strategies for
resolving entities from divergent email headers.

The `Message-ID` header provides a globally unique identification string
for every email.
This is accomplished by (how?).
The `In-Reply-To` header is set when an email is sent as a reply to
another email.
The reply's `In-Reply-To` header will match the `Message-ID` of the
original email.

We construct the directed *interaction graph* :math:`G` for a set of emails as follows:

* For every email :math:`e`, add a node :math:`v_{f(e)}` to :math:`G` corresponding 
  to its `From` header :math:`f(e)`  and set its `sent` attribute :math:`v_{f(e)}` 
  (notation is sloppy here -- can we write a pseudocode algorithm?)
* If :math:`e_1` is a reply to another email :math:`e_2`, create an edge between
  their corresponding nodes or, if it already exists, increment its weight by 1

BigBang implements this interaction graph creation using Python's native
email processing libraries, `pandas`, and `networkx`.

The motivation for constructing interaction graphs in this way is to build a
concise representation of the social network implied by email data.
We posit that building a social network representation based on actual messages
sent between people provides a more granular and faithful description of
social relationships than one based on higher-level descriptions of social
relationships or ties from web services such as Facebook 'friends' and
Twitter 'followers'

It may be objected that since the data we are considering comes from public
mailing lists where every message has a potentially large audience, it may be
misleading to build a network representation on the assumption that a reply
is directed primarily at the person who was replied to and not more broadly
to other participants in a thread or, even more broadly, to the mailing list
as a whole.

While this is a valid objection, it points to the heart of what is distinctive
about this research.
While there have been many studies of social network formation in conventional
settings, the conditions of open collaboration are potentially quite different.
Theoretically, we expect them to be explicitly and collectively goal-directed,
are self-organized for efficient action as opposed to positional 
social advantage, and designed around an archiving system for the sake of
efficiency.
Understanding the statistical properties of this particular form of social
organization, as opposed to others, is the very purpose of this empirical work.

*Interaction graph study*.
By looking at the *Reply-To* header of the emails, we
are able to construct a graph of who replies to who in the email list. (How?)
Drawing on (x,y, and z) we are studying the empirical properties of these
networks in order to come up with a generative model of community graph.
Whereas (Barabasi reference) models graphs with power-law degree distribution,
we find (verify using Clauset et al. method) that degree distribution in
these graphs is log normally distributed. Moreover, whereas [Newman2002]_
hypothesizes that socially generated graphs will be characterized by high
degree assortativity, we find empirically that these interaction graphs
have degree assortativity comparable with biological and technical networks.

Entity Resolution
-----------------

Empirically, over the extent of a mailing list's archival
data it is common for the *From* fields of emails to vary even when the
email is coming from the same person. Not only do people sometimes change their
email address or use multiple addresses to interact with the same list, but
also different email clients may represent the same email address in the *From*
header in different ways. BigBang includes automated techniques for resolving
these entities, cleaning the data for downstream processing.

Data from the `From` header of messages stored by Mailman is most often represented
in a form that includes both a full name representation and an email representation.
Unfortunately these fields can vary widely for one person. The following list
shows some of the variability that might appear for a single prolific sender.

* tyrion.lannister at gmail.com (Tyrion Lannister)
* Tyrion.Lannister at gmail.com (Tyrion Lannister)
* Tyrion.Lannister at gmail.com (Tyrion.Lannister@gmail.com)
* Tyrion.Lannister at gmail.com (Tyrion.Lannister at gmail.com)
* Tyrion.Lannister@gmail.com (Tyrion Lannister)
* Tyrion.Lannister@gmail.... (Tyrion Lannister)
* Tyrion.Lannister@gmail.com
* Tyrion.Lannister at gmail.com (Tyrion)
* tyrion at lanister.net (Tyrion Lannister)
* halfman@council.kings-landing.gov (Tyrion Lannister)
* halfman@council.kings-landing.gov (Tyrion Lannister, Hand of the King)
* halfman@council.kings-landing.gov (halfman@council.kings-landing.gov)
* tyrion+hand at lanister.net (Tyrion Lannister)
* tyrion.lannister at gmail.com (=?UTF-8?B?RGF2aWQgQWJpw6Fu?=)
* "Tyrion Lannister" <Tyrion.Lannister@gmail.com>

Variation in entity representation is a source of noise in our research and an
ongoing area of development for BigBang.
For the study in this paper, we have implemented a heuristic system for
entity matching.
First we standardize the data by converting it to lower case and normalizing
" at " and "@". Then we parse out the email and full name subfields of the
data and look for exact matches.
Under this procedure, all of the above email addresses would be collapsed into a single
entity.

In our interaction graph study, this has the effect of combining several nodes into a single one
in a way that's similar to the (cite) blockmodel technique.
The edges to and from the derived node are weighted by the sum of the edges of the original
nodes.
The `sent` attribute of the new node is also set as the sum of the `sent` attribute of the
original nodes.

Preferential attachment model
-----------------------------

Discuss Barabasi-Alpert here.



Discussion network formation
----------------------------

Background
----------

- Barabasi points out power laws in degree distributions of networks and proposes preferential
attachment model. [BarabasiAlbert]_
- Degree assortativity of social and technical networks [Newman2002]_ [Alstott2014]_

and so

Methods
-------

- build interaction graph
  - In-Reply-To header
- compute goodness of fit of power law distribution to degree distribution [Clauset2007] (cite Alstott)
- compare degree assortativity 


Results
-------

Every mailing list of the 10 we analyzed exhibits degree disassortivity and a significantly
(:math:`p` > .05) better fit to log normal instead of power law distribution.

.. table:: Results of analysis. For each mailing list archive, number of participants :math:`n`,
           computed degree assortativity of the interaction graph, and loglikelihood ratio R and
           statistical significance :math:`p` of comparison of fit between power law and log normal
           distributions. In all cases the interaction graph is disassortative with significantly
           more log normal degree distribution. :label:`mtable`

   +---------------+----------------+-----------+-----------------------+---------+------------+
   | list name     | Source         | :math:`n` |  Degree Assorativity  | R value | :math:`p`  |
   +===============+================+===========+=======================+=========+============+
   | ipython-dev   | SciPy          | 689       | -0.246441169106       | -0.518  |  0.080     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | potlatch-dev  | OpenStreetMap  | 75        | -0.0568958403876      | -0.001  |  0.969     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | scipy-dev     | SciPy          | 1056      | -0.276991197113       | -0.331  |  0.578     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | ipython-user  | SciPy          | 1085      | -0.267104106913       | -0.334  |  0.227     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | scipy-user    | SciPy          | 2735      | -0.111360803079       | -0.024  |  0.307     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | design        | WikiMedia      | 111       | -0.17722303449        | -3.618  |  0.095     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | gendergap     | WikiMedia      | 301       | -0.172107714006       | -0.858  |  0.399     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | maps-l        | WikiMedia      | 118       | -0.186099913331       | -0.003  |  0.945     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | wikimedia-l   | WikiMedia      | 1729      | -0.155694746786       | -3.645  |  0.074     |
   +---------------+----------------+-----------+-----------------------+---------+------------+
   | hot           | OpenStreetMap  | 524       | -0.199048173004       | -0.851  |  0.403     |
   +---------------+----------------+-----------+-----------------------+---------+------------+





Discussion
----------

The regularity in these data sets suggests that there is a need for a new network generation
model that results in disassortative graphs with log normal degree distribution.
Building this graph generation model may help us understand better how collaborative communities
grow and form over time.

References
----------

.. [Alstott2014] Alstott J, Bullmore E, Plenz D (2014) powerlaw: A Python Package 
                 for Analysis of Heavy-Tailed Distributions. PLoS ONE 9(1): e85777. 
                 doi:10.1371/journal.pone.0085777

.. [BarabasiAlbert] Albert-László Barabási & Reka Albert. Emergence of Scaling 
                    in Random Networks, Science, Vol 286, Issue 5439, 15 October 
                    1999, pages 509-512.

.. [Clauset2007]  A. Clauset, C.R. Shalizi, and M.E.J. Newman. Power-law distributions 
                  in empirical data. arXiv:0706.1062, June 2007.

.. [Newman2002] Newman, 2002.

.. [SocWik] Howard T. Welser, Dan Cosley, Gueorgi Kossinets, Austin Lin, Fedor Dokshin, 
            Geri Gay, and Marc Smith. 2011. *Finding social roles in Wikipedia.* 
            In Proceedings of the 2011 iConference (iConference '11). ACM, New York, NY, USA, 122-129.  

.. [LaborWik] R. Stuart Geiger and Aaron Halfaker. 2013. 
              *Using edit sessions to measure participation in wikipedia.* 
              In Proceedings of the 2013 conference on Computer supported cooperative work (CSCW '13). 
              ACM, New York, NY, USA, 861-870.

.. [SocRole] Gleave, E.; Welser, H.T.; Lento, T.M.; Smith, M.A., 
           *"A Conceptual and Operational Definition of 'Social Role' in Online Community,"* 
           System Sciences, 2009. HICSS '09. 42nd Hawaii International Conference on , 
           vol., no., pp.1,11, 5-8 Jan. 2009

.. [Zanetti2012] Zanetti, M. and Schweitzer, F. 2012.
                 "A Network Perspective on Software Modularity"
                 ARCS Workshops 2012, pp. 175-186.
