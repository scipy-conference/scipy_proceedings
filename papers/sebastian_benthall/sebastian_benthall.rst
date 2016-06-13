:author: Sebastian Benthall
:email: sb@ischool.berkeley.edu
:institution: Ion Channel
:institution: UC Berkeley School of Information
:corresponding:

:author: Travis Pinney
:email: travis.pinney@ionchannel.io
:institution: Ion Channel

:author: Kit Plummer
:email: kit.plummer@ionchannel.io
:institution: Ion Channel

---------------------------------------------------------------
An Ecological Approach to Software Supply Chain Risk Management
---------------------------------------------------------------

.. class:: abstract

   Robustness and resilience of software infrastructure are essential for
   cybersecurity. Existing approaches to software assurance focus on
   evaluating individual software projects in isolation. We propose
   a novel method of risk management that considers the total ecosystem
   of software development, inclusive of interdependent software packages
   and the communities of people that develop them. By making minimal
   analytic assumptions about the propogation of vulnerability and exposure
   through a software supply chain, we can predict high risk "hot spots"
   in the ecosystem in need of additional investment. We present features of
   Ion Channel that assist with the collection and analysis of software
   dependency information. We present statistical properties of
   empirical data from the Python Package Index and GitHub as complex networks,
   and demonstrate the results of our ecosystem modeling strategy on this data
   when combined with simulated data.
   We consider the research and business ethics of this work. 

.. class:: keywords

   risk management, software dependencies, complex networks

Introduction
------------

Critical systems that depend on complex software are open
to many kinds of risk.
Typical approaches to software security have mitigated this
risk with static analysis.
We are developing novel ways to manage software risk through
supply chain intelligence, with a focus on open source software
ecosystems.

The Heartbleed bug in OpenSSL is anas an example of community failure
example of how vulnerabilities
in open source software can be a major security risk. [Wheeler2014]_
(**Reference recent nodejs leftpad bug as well.**)
Open source software projects make their source code and developer
activity data openly available for analysis.
Despite its openness, there are many ways this data can be used
to mitigate software risk that have not been explored.

Our approach builds on prior work in software assurance, community evaluation,
and supply chain risk assessment by taking the entire open source software ecosystem
into consideration in the analysis.
By software ecosystem, we mean the whole system of interdependent software packages
under development, the developers themselves, their communications, and the infrastructure 
they use to manage their development practices.

With a small number of analytic assumptions about the propagation of vulnerability
and exposure through the software dependency network, we have developed a model
of ecosystem risk that predicts "hot spots" in need of more investment.
In this paper, we demonstrate this model using real software dependency data extracted
from PyPI using Ion Channel, GitHub data, as well as simulated data.

We will also discuss the ethics of performing this research and using it in a commercial
product. We conclude that establishing trust the trust of the developer community is
an important part of our research and business model.


Prior work
----------

[Verdon2004]_ outline the diversity of methods used for risk analysis in software design.
Their emphasis is on architecture-level analysis and its iterative role in software development.
Security is achieved through managing information flows through architecturally distinct tiers of trust.
They argue for a team-based approach with diverse knowledge and experience because "risk analysis
is not a science".

In computer science, however, there is a long history of security achieved through static analysis.
[Wagner2000]_ points out that the dependency of modern Internet systems on legacy code and the
sheer complexity of source code involved makes manual source code level auditing infeasible.
Therefore, static analysis tools based on firm mathematical foundations are significant
for providing computer security at scale. 

[Wheeler2015]_ develops a risk index for determing which open source software projects need 
security investments. This work is part of the Linux Foundation (LF) Core Infrastructure 
Initiative (CII) and published by the Institute for Defense Analysis. 
This metric is based on their expertise in software development analytics and an 
extensive literature review of scholarly and commercial work on the subject. 
They then apply this metric to Debian packages and have successfully identified 
projects needing investment. This work is available on-line as the CII Census project.

[Schweik2012]_ is a comprehensive study of the success and failure of open source
projects based on large-scale analysis of SourceForge data, as well as survey and
interview data. They define successful project as one that performs a useful function
and has had at least three releases. They identify several key predictive factors to
project success, including data that indicates usefulness (such as number of downloads),
number of hours contributed to the project, 

Our approach synthesizes these precedents in computer security and software community analysis.
We see risk analysis as a science that at its best adopts techniques from static analysis.
However, we look not only at the source code of computing systems, but also more broadly
at developers and their communications. We are looking for mathematically firm principles
of software supply chain risk management.

As this supply chain resembles a complex ecosystem more than a simple 'chain' or stack,
our risk management strategy is adopted from work on disaster risk reduction and
climate change adaptation research. [Cordona2012]_ and others use a framework for
evaluating expected cost of low-probability events that distinguishes three factors
of risk. *Hazards* are potentially damaging factors from the environment; the
cybersecurity equivalent are *threats*. *Exposure* refers to the inventory of elements
in place where hazards occur; the cybersecurity equivalent is *assets*. *Vulnerabilities*
are defined as the propensity of exposed elements to suffer adverse effects when impacted
by a hazard. Expected risk is then straightforwardly computed as the product of the
probability of hazard and the vulnerable exposure of the system. By adapting this
framework to cybersecurity in the software ecosystem, we are able to consider a wide
range of threats--including novel threats such as attacks of software communities themselves.


Modeling Ecological Risk in Software
------------------------------------

Software dependency and project risk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While it has been suggested that using software dependency information can augment assessments 
of project risk, suggested uses of this information are unspecific. This is partly due to a 
larger problem, the ambiguity of how 'risk' is used in a software development context.

If we break down the sources of risk and how these effect the need for security investments analytically, 
we can distinguish between several different factors.

* Vulnerability. A software project's vulnerability is its intrinsic susceptability to attack. 
  CVE data is about software vulnerability. Being written in a language in which it is hard to 
  write secure code (such as C and C++) can be a predictor of vulnerability.
* Exposure. A software project's exposure is its extrinsic availability to attack. Being directly exposed to a network is a source of exposure.

Vulnerability and exposure are distinct elements of a software project's risk. 
Analyzing them separately and then combining them in a principled way will give us a better understanding of a project's risk.

Dependencies complicate the way we think about vulnerability and exposure. 
A software project does not just include the code in its own repository; 
it also includes the code of all of its dependencies. 
And a project does not need to be installed directly to be exposed--it can be installed 
as a dependency of another project. 
Based on these observations, we can articulate two heuristics for use of 
dependency topology in assessing project risk.

* If A depends on B, then a vulnerability in B implies a corresponding vulnerability in A.
* If A depends on B, then an exposure to A implies an exposure to B.

While there are exceptions to these rules, they are a principled analytic way of related vulnerability, exposure, 
and software dependency that can be implemented as a heuristic and tested as a hypothesis.

Robustness and fragility, resilience and brittleness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The risk analysis framework described above is very general.
Due to this generality, it suffers from the ambiguity of its terms.
In particular, "vulnerability" can, dependent on the application of this
framework, be literal software vulnerabilities such as would be reported
in a CVE.
But when we analyze the software ecosystem as a supply chain, we are
often concerned about higher level properties that serve as general proxies
for whole classes of error or failure.

We find the distinction between system *robustness* and system *resilience* helpful.
We define the *robustness* of a system as its invulnerability to threats and hazards,
as a function of its current state. A system that is not robust is *fragile*.
We define the *resilience* of a system as its capacity to recover quickly from injury
or failure. A system that is not resilient is *brittle*.
A mature, well-tested system will be robust.
A system with an active community ready to respond to the discovered of a new exploit
will be resilient.

A system can be robust, or resilient, or both, or neither.
Robustness and resilience can be in tension with each other.
The more dynamic a software project is, measured as a function of the activity
of the community and frequency of new commits, the more likely that it will
be resilient, responding to new threat information. But it is also likely to
be less robust, as new code might introduce new software flaws.


Computing risk
~~~~~~~~~~~~~~~

The risk analysis framework presented above is designed to be very
generalizable, factoring risk into abstract *exposure* and *vulnerability*
factors and then making minimal assumptions about how these factors propagate
through the dependency graph.

In practice, the application of this framework will depend on the selection
of package metadata used to measure exposure and vulnerability. Below is a
Python implementation of efficient risk computation using a directed graph
representation of package dependencies and NetworkX. [Hagberg2008]_
In this code, we use a precomputed 'fragility' metric as the vulnerability
variable, and the number of downloads of each package as the exposure variable.

.. code-block:: python

    import networkx as nx

    G = nx.read_gexf('pkg.gexf')

    # select proxy empirical variables for
    # vulnerability and exposure

    vulnerability_metric = 'fragility'
    exposure_metric = 'downloads'

    # efficiently compute ecosystem vulnerability
    # and assign as attribute

    ecosystem_vulnerability = {}

    for i in nx.topological_sort(G,reverse=True):
    
        ecosystem_vulnerability[i] = 
                G.node[i][vulnerability_metric] 
                + sum([ecosystem_vulnerability[j]
                       for j in G.neighbors(i)]) 

    nx.set_node_attributes(G,
                           'ecosystem_vulnerability',
                           ecosystem_vulnerability)

    # efficiently compute ecosystem exposure 
    # and assign as attribute
    
    ecosystem_exposure = {}

    for i in nx.topological_sort(G):
    
         ecosystem_exposure[i] = 
                G.node[i][exposure_metric]
                + sum([ecosystem_exposure[j]
                       for j in G.predecessors(i)]) 

    nx.set_node_attributes(G,
                           'ecosystem_exposure',
                           ecosystem_exposure)

    # efficiently compute ecosystem risk
    # and assign as attribute
    
    ecosystem_risk= {}

    for i in nx.topological_sort(G):
        ecosystem_risk[i] = 
                G.node[i]['ecosystem_vulnerability'] 
                * G.node[i]['ecosystem_exposure']




**Algorithms, with source code, for computing risk on a dependency network.**


Data collection and publication
-------------------------------

Data for this analysis comes from two source. For package and release metadata,
we used data requested from PyPI, the Python Package Index.
This data provides for data about the publication date and number of
downloads for each software release.

We also downloaded each Python release and inspected it for the presence of a ``setup.py``
file. We then extracted package dependency information from ``setup.py`` through
its ``install_requires`` field.

Python dependencies are determined through executing Python install scripts.
Therefore, our method of discovering package dependencies through static
analysis of the source code does not capture all cases.

For each package, we consider its dependencies to be the union of all requirements
for all releases. While this loses some of the available information, it is sufficient
for this first analysis of the PyPI ecosystem. We will use more of the available information
and take into account more of the complexity of Python package management in future work.

Empirical and Simulation Results
--------------------------------

.. figure:: dependencies-1.png
   :scale: 20%
   :figclass: bht

   Visualization of PyPi dependency network, created using Gephi [Bastian2009]_. This visualization does not include singleton nodes with zero degree, which are the vast majority of nodes. :label:`depfig`




Statistical properties of the software dependency network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyPI package dependency network resembles classical complex networks, with
some notable departures.

A early claim in complex network theory by [Newman2002]_, [Newman2003]_ is that
random complex networks will exhibit negative degree assortativity, and that social
networks will exhibit positive degree assortativity due to homophily or other
effects of group membership on network growth.
[Noldus2015]_ notes that in directed graphs, there are four variations on the
degree assortativity metric as for each pair of adjacent nodes one can consider
each node's in-degree and out-degree.
The degree assortativity metrics for the PyPI dependency graph are given
in :ref:`datable`.

The PyPI package dependency network notably  has *in-in* degree assortativity of
:math:`0.19`, and *out-in* degree assortativity of :math:`-0.16`.
The *in-out* and *out-out* degree assortativities are both close to zero.
We have constructed the graph with the semantics that an edge from A to B
implies that A depends on B.

.. table:: Degree assortativity metrics for the PyPI dependency graph. :label:`datable`

   +------------+----------------+
   | Metric     | Value          |
   +============+================+
   | *in-in*    |   :math:`0.19` |
   +------------+----------------+
   | *in-out*   |   :math:`0.05` |
   +------------+----------------+
   | *out-in*   |  :math:`-0.16` |
   +------------+----------------+
   | *out-out*  |  :math:`-0.04` |
   +------------+----------------+

What explains this strange network structure? One reason is that
there is much greater variation in out-degree than in in-degree.
:label:`odtable` shows the top ten most depended on packages.
:label:`idtable` shows the top ten packages with the most dependencies.
Four packages, ``requests``, ``six``, ``django``, and ``pyyaml`` have
out-degree over 1000. 

.. table:: Top ten most depended on packages. :label:`odtable`

   +-----------------------+----------------+
   | Package               | Out-Degree     |
   +=======================+================+
   | ``requests``          |   :math:`2125` |
   +-----------------------+----------------+
   | ``six``               |   :math:`1381` |
   +-----------------------+----------------+
   | ``django``            |   :math:`1174` |
   +-----------------------+----------------+
   | ``pyyaml``            |    :math:`775` |
   +-----------------------+----------------+
   | ``zope.interface``    |    :math:`663` |
   +-----------------------+----------------+
   | ``lxml``              |    :math:`619` |
   +-----------------------+----------------+
   | ``flask``             |    :math:`607` |
   +-----------------------+----------------+
   | ``python-dateutil``   |    :math:`599` |
   +-----------------------+----------------+
   | ``zope.component``    |    :math:`550` |
   +-----------------------+----------------+
   | ``jinja2``            |    :math:`507` |
   +-----------------------+----------------+

.. table:: Top ten packages by number of dependencies. :label:`idtable`

   +------------------------+---------------+
   | Package                | Out-Degree    |
   +========================+===============+
   | ``plone``              |    :math:`92` |
   +------------------------+---------------+
   | ``mypypi``             |    :math:`53` |
   +------------------------+---------------+
   | ``invenio``            |    :math:`52` |
   +------------------------+---------------+
   | ``ztfy.sendit``        |    :math:`48` |
   +------------------------+---------------+
   | ``ztfy.blog``          |    :math:`47` |
   +------------------------+---------------+
   | ``smartybot``          |    :math:`47` |
   +------------------------+---------------+
   | ``icemac.addressbook`` |    :math:`41` |
   +------------------------+---------------+
   | ``sentry``             |    :math:`40` |
   +------------------------+---------------+
   | ``products.silva``     |    :math:`38` |
   +------------------------+---------------+
   | ``ztfy.scheduler``     |    :math:`37` |
   +------------------------+---------------+


.. figure:: exposure-vulnerability-plot.png
   :figclass: bht

   Hex plot of log vulnerbality and log exposure of each package, with bin density scored on log scale. All logs are base 10. Exposure is more widely distributed than vulnerability, the vast majority of packages score low. There is a fringe of packages that are either highy vulnerable, highly exposed, or both. There is a log-linear tradeoff between high vulnerability and high exposure. This is most likely due to the fact that ecosystem vulnerability and ecosystem exposure both depend on an package's position in the dependency network. :label:`depfig`




Hot spot analysis
~~~~~~~~~~~~~~~~~



**Visualization of hot spots based on the data here.**

**Discussion of implications of network statistics on distribution of risk.**

Research and Business Ethics
----------------------------

**Discussion of research ethics.**

* As "Big Data" research goes, this is benign because open source developers generally consider their activity public.
* Study of dependency structure is study of technology and institutions, not human subjects research.
* As we begin to look at developer activity data, this raises some issues of surveillance
* Approaching problem as a search for positive quality/reliability rather than labeling
  particular developers as liabilities eases problems of reputation.





References
----------

.. [Bastian2009] Bastian, Mathieu, Sebastien Heymann, and Mathieu Jacomy. "Gephi: an open source software for exploring and manipulating networks." ICWSM 8 (2009): 361-362.

.. [Clauset2007]  A. Clauset, C.R. Shalizi, and M.E.J. Newman. Power-law distributions 
                  in empirical data. arXiv:0706.1062, June 2007.

.. [Mitzenmacher2003] Mitzenmacher, M. 2003.
                      "A Brief History of Generative Models for Power Law
                      and Lognormal Distributions."
                      Internet Mathematics Vol. 1, No. 2: 226-251

.. [Cordona2012] Cardona, Omar-Daria, et al. "Determinants of risk: exposure and vulnerability." (2012).

.. [Girardot2013] O. Girardot. STATE OF THE PYTHON/PYPI DEPENDENCY GRAPH. 2013

.. [Hagberg2008] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

.. [LaBelle2004] N. LaBelle, E. Wallingford. 2004. Inter-package dependency networks in open-source software.

.. [Newman2002] Newman, M. E. J. 2002. "Assortative mixing in networks."

.. [Newman2003] Newman, M. E. J. 2003. "Mixing patterns in networks."
                Phys. Rev. E 67, 026126

.. [Noldus2015] Noldus, R and Mieghem, P. 2015. "Assortativity in Complex Networks"
                Journal of Complex Networks. doi: 10.1093/comnet/cnv005


.. [Schweik2012] C. Schweik and R. English. *Internet Success: A Study of Open-Source Software Commons*,
      The MIT Press. 2012

.. [Verdon2004] D. Verdon and G. McGraw, "Risk analysis in software design," in IEEE Security & Privacy, vol. 2, no. 4, pp. 79-84, July-Aug. 2004.

.. [Wagner2000] David A. Wagner. 2000. Static Analysis and Computer Security: New Techniques for Software Assurance. Ph.D. Dissertation. University of California, Berkeley. AAI3002306.

.. [Wheeler2014] Wheeler, David A. How to Prevent the next Heartbleed. 2014-10-20.
      ``http://www.dwheeler.com/essays/heartbleed.html``

.. [Wheeler2015] D. Wheeler and S. Khakimov. *Open Source Security Census: Open Source Software Projects Needing Security Investments*, Institute for Defense Analysis. 2015


