:author: Mike Pacer
:email: mpacer@berkeley.edu
:institution: University of California at Berkeley
:bibliography: myBibliography

:video: youtube_link_to_come

..  latex::
    :usepackage: booktabs

    \newcommand{\bigCI}{\mathrel{\text{\scalebox{1.07}{$\perp\mkern-10mu\perp$}}}}

    \newcommand\independent{\protect\mathpalette{\protect\independenT}{\perp}}

    \def\independenT#1#2{\mathrel{\rlap{$#1#2$}\mkern2mu{#1#2}}}


.. raw:: latex

    \newcommand{\DUrolesc}{\textsc}
    \newcommand{\DUrolesf}{\textsf}

.. role:: sc

.. role:: sf

.. |cbnx| replace:: :sc:`cbnx`
.. _cbnx: https://github.com/michaelpacer/Causal-Bayesian-NetworkX


------------------------
Causal Bayesian NetworkX
------------------------

..  class:: abstract

    Probabilistic graphical models are useful tools for modeling systems governed by probabilistic structure. *Bayesian networks* are one class of probabilistic graphical model that have proven useful for characterizing both formal systems and for reasoning with those systems. Probabilistic dependencies in Bayesian networks are graphically expressed in terms of directed links from parents to their children. *Casual Bayesian networks* are a generalization of Bayesian networks that allow one to "intervene" and perform "graph surgery" — cutting nodes off from their parents. *Causal theories* are a formal framework for generating causal Bayesian networks.

    This report provides a brief introduction to the formal tools needed to comprehend Bayesian networks, including probability theory and graph theory. Then, it describes Bayesian networks and causal Bayesian networks. It introduces some of the most basic functionality of the extensive NetworkX python package for working with complex graphs and networks :cite:`networkx`. I introduce some utilities I have build on top of NetworkX including conditional graph enumeration and sampling from discrete valued Bayesian networks encoded in NetworkX graphs :cite:`pacer2015cbnx`. I call this Causal Bayesian NetworkX, or |cbnx|. I conclude by introducing a formal framework for generating causal Bayesian networks called theory based causal induction :cite:`griffithst09`, out of which these utilities emerged. I discuss the background motivations for frameworks of this sort, their use in computational cognitive science, and the use of computational cognitive science for the machine learning community at large.

..  class:: keywords

    probabilistic graphical models, causal theories, Bayesian networks

Introduction and Aims
---------------------

My first goal in this paper is to provide enough of an introduction to some formal/mathematical tools such that those familiar with :code:`python` and programming more generally will be able to appreciate both why and how one might implement causal Bayesian networks. Especially to exhibit *how*, I have developed parts of a toolkit that allows the creation of these models on top of the :code:`NetworkX` python package:cite:`networkx`. Given the coincidence of the names, it seemed most apt to refer to this toolkit as :code:`Causal Bayesian NetworkX` abbreviated as |cbnx| [#]_.

.. [#] in addition to the frozen version which you can find in the citations :cite:`pacer2015cbnx`, the most up to date version can be found at |cbnx|_.

In order to understand the tool-set requires the basics of probabilistic graphical models, which requires understanding some graph theory and some probability theory. The first few pages are devoted to providing necessary background and illustrative cases for conveying that understanding. 

Notably, contrary to how Bayesian networks are commonly introduced, I say relatively little about inference from observed data. This is intentional, as is this discussion of it. Many of the most trenchant problems with Bayesian networks are found in critiques of their use to infer these networks from observed data. But, many of the aspects of Bayesian networks (especially causal Bayesian networks) that are most useful for thinking about problems of structure and probabilistic relations do not rely on inference from observed data. In fact, I think the immediate focus on inference has greatly hampered widespread understanding of the power and representative capacity of this class of models. Equally – if not more – importantly, I aim to discuss generalizations of Bayesian networks such as those that appear in :cite:`griffithst09`, and inference in these cases requires a much longer treatment (if a comprehensive treatment can be provided at all). If you are dissatisfied with this approach and wish to read a more conventional introduction to (causal) Bayesian networks I suggest consulting :cite:`pearl2000`.

The Causal Bayesian NetworkX toolkit can be seen as consisting of two main parts: graph enumeration/filtering and the storage and use of probabilistic graphical models in a NetworkX compatible format:cite:`networkx`. Because this topic will more be the focus of my talk which can be viewed at the youtube link above and the source code of the most basic implementation is available at |cbnx|_:cite:`pacer2015cbnx` [#]_, in this paper I focus more on the other aspects of the problem. Nonetheless, appreciating these other aspects is made easier by also appreciating the problems of implementation/representation and the early solutions that I propose.

.. [#] This code is licensed with the standard BSD 3-clause license.

I focus first on establishing a means of building iterators over sets of directed graphs. I then apply operations to those sets. Beginning with the complete directed graph, we enumerate over the subgraphs of that complete graph and enforce graph theoretic conditions such as acyclicity over the entire graph, guarantees on paths between nodes that are known to be able to communicate with one another, or orphan-hood for individual nodes known to have no parents. We accomplish this by using closures that take graphs as their input along with any explicitly defined arguments needed to define the exact desired conditions. 

I then shift focus to a case where there is a specific known directed acyclic graph that is imbued with a simple probabilistic semantics over its nodes and edges, also known as a Bayesian network. I demonstrate how to sample independent trials from these variables in a way consistent with these semantics. I discuss briefly some of the challenges of encoding these semantics in dictionaries as afforded by NetworkX without resorting to :code:`eval` statements and discuss compatibility issues I have found with JSON storage formats. 

I conclude with a discussion of some of the problems that have been addressed in Cognitive Science through the use of graphical models like those described. In particular, I will discuss a framework called **theory based causal induction** :cite:`griffithst09`, or my preferred term: **causal theories**, which allows for defining problems of causal induction. It is out of this framework the perspective expressed in this paper, the associated talk, and the Causal Bayesian NetworkX toolkit developed. 

Graphical Models
----------------

Graphs are formal models defined by a set of nodes (:math:`X, |X| = N`) and edges between those nodes (:math:`e \in E \equiv e \in (X \times X)`, where *E* is the set of edges). 

Notes on notation
=================

Nodes
^^^^^

In the examples in |cbnx|_, nodes are given explicit labels individuating them such as :math:`\{A,B,C,\ldots\}` or {'rain','sprinkler','grass_wet'}. Oftentimes, for the purposes of mathematical notation, it will be helpful to index nodes by the integers over a common variable label, e.g., using  :math:`\{X_1,X_2,X_3,\ldots\}`. [#]_ 

.. [#] Despite pythonic counting beginning with 0, I chose not to begin this series with 0 because when dealing with variables that might be used in statistical regressions, the 0 subscript will have a specific meaning that separates it from the rest of the notation. For example when expressing multivariate regression as :math:`Y = \beta X + \epsilon, \epsilon \sim \mathcal{N}(0,\Sigma)`, :math:`\beta_0` refers to the parameter associated with a constant variable :math:`x_0 = 1` and :math:`X` is normally defined as :math:`x_1, x_2, x_3, \ldots`. This allows a simple additive constant to be estimated, which usually(but not always) is not of interest to statistical tests, acting as a scaling constant more than anything else. This also makes for simpler notation than saying :math:`Y = \beta_0 + \beta X + \epsilon`, since that is equivalent to the previous notation (:math:`Y = \beta X + \epsilon`) if :math:`x_0 = 1`. In other cases :cite:`griffithst05,pacerg12`, the 0 index will be used to indicate background sources for events in a system.

Edges
^^^^^

Defined in this way, edges are all *directed* in the sense that an edge from :math:`X_1 \textrm{ to } X_2` is not the same as the edge from :math:`X_2 \textrm{ to } X_1`, or :math:`(X_1,X_2) \neq (X_2,X_1)`. An edge :math:`(X_1,X_2)` will sometimes be written as :math:`X_1 \rightarrow X_2`, and the relation may be described using language like ":math:`X_1` is the parent of :math:`X_2`" or ":math:`X_2` is the child of :math:`X_1`".

Directed paths
^^^^^^^^^^^^^^

Paths are a useful way to understand sequences of edges and the structure of a graph. Informally, to say there is a path between :math:`X_i` and :math:`X_j` is to say that one can start at :math:`X_i` and by traveling from parent to child along the edges leading out from the node that you are currently at, you can eventually reach :math:`X_j`.

To define it recursively and more precisely, if the edge :math:`(X_i,X_j)` is in the edge set or if the edges :math:`(X_i,X_k)` and :math:`(X_k,X_j)` are in the edge set there is a path from :math:`X_i` to :math:`X_j`. Otherwise, a graph has a path from node :math:`X_i` to :math:`X_j` if there is a subset of its set of edges such that the set contains edges :math:`(X_i,X_k)` and :math:`(X_l,X_j)` and there is a path from :math:`X_k` to :math:`X_l`. 


Adjacency Matrix Perspective
============================

For a fixed set of nodes :math:`X` of size :math:`N`, each graph is uniquely defined by its edge set, which can be seen as a binary :math:`N \times N` matrix, where each index :math:`(i,j)` in the matrix is :math:`1` if the graph contains an edge from :math:`X_i \rightarrow X_j`, and :math:`0` if it does not contain such an edge. We will refer to this matrix as :math:`A(G)`.

This means that any values of :math:`1` found on the diagonal of the adjacency matrix (i.e., where :math:`X_i \rightarrow X_j, i=j`) indicate a self-loop on the respective node.

Undirected Graphs
=================

We can still have a coherent view of *undirected* graphs, despite the fact that our primitive notion of an edge is that of a *directed* edge. If a graph is undirected, then if it has an edge from :math:`X_i \rightarrow X_j` then it has an edge from :math:`X_j \rightarrow X_i`. Equivalently, this means that the adjacency matrix of the graph is symmetric, or :math:`A(G)=A(G)^\top`. However from the viewpoint of the undirected graph, that means that it has only a single edge.


Directed Graphs
===============

From the adjacency matrix perspective we've been considering, all graphs are technically directed, and undirected graphs are a special case where one (undirected) edge would be represented as two symmetric edges.

The number of directed graphs that can be obtained from a set of nodes of size :math:`n` can be defined explicitly using the fact that they can be encoded as a unique :math:`n \times n` matrix:

.. math::

    R_n = 2^{n^2}


Directed Acyclic Graphs
^^^^^^^^^^^^^^^^^^^^^^^

A cycle in a directed graph can be understood as the existence of a path from a node to itself. This can be as simple as a self-loop (i.e., if there is an edge :math:`(X_i,X_i)` for any node :math:`X_i`). 

Directed acyclic graphs(:sc:`dag`\s) are directed graphs that contain no cycles.

The number of :sc:`dag`\s that can be obtained from a set of nodes of size :math:`n` can be defined recursively as follows :cite:`mckay2004acyclic`:

.. math::

    R_n = \sum_{k=1}^{n} (-1)^{k+1} {\binom{n}{k}} 2^{k(n-k)} R_{n-k}

Note, because :sc:`dag`\s do not allow any cycles, this means that there can be no self-loops. As a result, every value on the diagonal of a  :sc:`dag`\'s adjacency matrix will be 0. 

.. Topological ordering in :sc:`dag`\s
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. It is possible to reorder 


Probability Distributions: Conditional, Joint and Marginal
----------------------------------------------------------

A random variable defined by a conditional probability distribution [#]_ has a distribution indexed by the realization of some other variable (which itself is often a random variable, especially in the context of Bayesian networks). 

.. [#] Rather than choose a particular interpretation of probability over event sets (e.g., Bayesian or  frequentist), I will attempt to remain neutral, as those concerns are not central to the issues of graphs and simple sampling.

The probability mass function (pmf) of a discrete random variable(:math:`X`) taking on value :math:`x` will be designated with :math:`P(X=x)`. Oftentimes, when one is discussing the full set of potential values (and not just a single value), one leaves out the :math:`=x` and just indicates :math:`P(X)`. [#]_ 
.. This interpretation works most easily when considering mutually exclusive values, and if one is instead considering the possibility of a more complex event such as a variable taking on one of a set of values, the notation will often need adjusting. 

.. [#] If one is dealing with continuous quantities rather than discrete quantities one will have to use a probability density function (pdf) which does not have as straightforward an interpretation as a probability mass function. This difficult stems from the fact that (under most cases) the probability of any particular event occurring is "measure zero", or "almost surely" impossible. Without getting into measure theory and the foundation of calculus and continuity we can simply note that it is not that any individual event has non-zero probability, but that sets of events have non-zero probability.As a result, continuous random variables are more easily understood in terms a cumulative density function (cdf), which states not how likely any individual event is, but how likely it is that the event in question is less than a value :math:`x`. The notation usually given for a cdf of this sort is :math:`F(X\leq x) = \int_{-\infty}^{x}f(u)du`, where :math:`f(u)` is the associated probability density function.

The conditional probability of :math:`X` with value :math:`x` given another variable :math:`Y` with value :math:`y` is :math:`P(X=x~|Y=y)`. Much like above, if we want to consider the probability of each possible event without specifying one, sometimes this will be written as :math:`P(X|Y=y)`. If we are considering conditioning on any of the possible values of the known variable, we might use the notation :math:`P(X|Y)`, but that is a slight abuse of the notation. 

You *can* view :math:`P(X|Y)` as a function over the space defined by :math:`X\times Y`. However, if you do so, do not interpret this as a probability function (of any kind). Rather, this defines a probability function for :math:`X` relative to each value of :math:`Y`. Without conditioning on :math:`Y` we have many potential probability functions of X. Equivalently, it denotes a *family* of probability functions on X indexed by the values :math:`Y=y`.

The *joint probability* of :math:`X` and :math:`Y` is the probability that both :math:`X` and  :math:`Y` occur in the event set in question. This is noted as :math:`P(X,Y)` or :math:`P(X \cap Y)` (using the set theoretic intersection operation). Similar to :math:`P(X|Y)`, you *can* view :math:`P(X,Y)` as a function over the space defined by :math:`X\times Y`. However, :math:`P(X,Y)` is a probability function in the sense that the sum of :math:`P(X=x,Y=y)` over all the possible events in the space defined by :math:`(x,y)\in X\times Y` equals 1.

The *marginal probability* of :math:`X` is the same :math:`P(X)` that we have seen before. However, the term refers to the notion of summing over values of :math:`Y` in the joint probability, and these summed probabilities were recorded in the *margins* of a probability table. Formally, this can be stated as :math:`P(X) = \sum_{y\in Y}P(X,Y)`.

Relating conditional and joint probabilities
============================================

Conditional probabilities are related to joint probabilities using the following form:

.. math::

    P(X|Y=y) = \frac{P(X,Y=y)}{P(Y=y)} = \frac{P(X,Y=y)}{\sum_{x \in X}P(X=x,Y=y)}

Equivalently:

.. math::

    P(X,Y=y) = P(X|Y=y)P(X)

Bayes' Theorem
==============

Bayes' Theorem can be seen as a result of how to relate conditional and joint probabilities. Or more importantly, how to compute the probability of a variable once you know something about some other variable.

Namely, if we want to know :math:`P(X|Y)` we can transform it into :math:`\frac{P(X,Y)}{\sum_{x \in X}P(X=x,Y)}`, but then can also transform joint probabilities (:math:`P(X,Y)`) into statements about conditional and marginal probabilities (:math:`P(X|Y)P(X)`).

This leaves us with

..  math::

    P(X|Y) = \frac{P(X|Y)P(X)}{\sum_{x \in X}P(X=x|Y)P(X=x)}

Probabilistic Independence
==========================

To say that two variables are independent of each other means that knowing/conditioning on the realization of one variable is irrelevant to the distribution of the other variable. This is equivalent to saying that the joint probability is equal to the multiplication of the probabilities of the two events. 

If two variables are conditionally independent, that means that conditional on some set of variables, condition

Example: Marginal Independence :math:`\neq` Conditional Independence
====================================================================

Consider the following example:

..  math::
    :type: eqnarray

    X &\sim& \textrm{Bernoulli}_{\{0,1\}}(.5),~ Y \sim \textrm{Bernoulli}_{\{0,1\}}(.5)\\
    Z &=& X \oplus Y , \oplus \equiv \textsc{xor}

Note that, :math:`X \independent Y` but :math:`X \not\independent Y|Z`.

Bayesian Networks
-----------------

Bayesian networks are a class of graphical models that have particular probabilistic semantics attached to their nodes and edges. This makes them probabilistic graphical models. 

In Bayesian networks when a variable is conditioned on the total set of its parents and children, it is conditionally independent of any other variables in the graph. This is known as the "Markov blanket" of that node. [#]_

.. [#] The word "Markov" refers to Andrei Markov and appears as a prefix to many other terms. It most often indicates that some kind of independence property holds. For example, a Markov chain is a sequence (chain) of variables in which each variable depends only dependent on the value of the immediate preceding (and by implication) postceding variables in the chain. 

Common assumptions in Bayesian networks
=======================================

While there are extensions to these models [#]_ , a number of assumptions commonly hold. 

.. [#] An important class of extensions to Bayesian networks that I will not have time to discuss at length includes those that consider temporal dependencies: Dynamic Bayesian Networks (:sc:`dbn`\s) :cite:`deank1989time,ghahramani1998learning`, continuous-time dependencies with Continuous Time Bayesian Networks (:sc:`ctbn`\s) :cite:`nodelman02`, Poisson Cascades :cite:`simma10`, Continuous Time Causal Theories (:sc:`ct`:math:`^2`) :cite:`pacerg12, pacerg15`, Reciprocal Hawkes Processes :cite:`blundell2012modelling` and the Network Hawkes Model :cite:`lindermana2014`.

Fixed node set
^^^^^^^^^^^^^^

The network is considered to be comprehensive in the sense that there is a fixed set of known nodes with finite cardinality :math:`N`. This rules out the possibility of hidden/latent variables as being part of the network. From this perspective inducing hidden nodes requires postulating a new graph that is potentially unrelated to the previous graph. 

Trial-based events, complete activation and :sc:`dag`\-hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within a trial, all events are presumed to occur simultaneously. This means two things. First, there is no notion of temporal asynchrony, where one node/variable takes on a value before its children take on a value (even if in reality – i.e., outside the model – that variable is known to occur before its child). Secondly, the probabilistic semantics will be defined over the entirety of the graph meaning that one cannot sample a proper subset of the nodes of a graph unless they have no effects or are marginalized out with their effects being incorporated into their children.

This property also explains why Bayesian networks need to be acyclic. Most of the time when we consider causal cycles in the world the cycle relies on a temporal delay between the causes and their effects to take place. If the cause and its effect is simultaneous, it becomes difficult (if not nonsensical) to determine which is the cause and which is the effect — they seem instead to be mutually definitional. But, as noted above, when sampling in Bayesian networks simultaneity is presumed for *all* of the nodes.

Independence in Bayes Nets
==========================

One of the standard ways of describing the relation between the semantics (probability values) and syntax (graphical structure) of Bayesian networks is in terms of the graph encoding particular conditional independence assumptions between the nodes of the graph. Indeed, in some cases Bayesian networks are *defined as* a convenient representation for the conditional and marginal independence relationships between different variables. 

It is the perspective of the graphs as *merely* representing the independence relationships and the focus on inference that leads to the focus on equivalence classes of Bayes nets. The set of graphs :math:`\{A \rightarrow B \rightarrow C,~ A \leftarrow B \rightarrow C, \textrm{ and } A \leftarrow B \leftarrow C\}` represent the same conditional independence relationships, and thus cannot be distinguished on the basis of observational evidence alone. This also leads to the emphasis on finding :sf:`V`\-structures or common-cause structures where (at least) two arrows are directed into the same child with no direct link between those parents(e.g., :math:`A \rightarrow B \leftarrow C`). :sf:`V`\-structures are observationally distinguishable because any reversing the direction of any of the arrows will alter the conditional independence relations that are guaranteed by the graphical structure. [#]_

.. [#] A more thorough analysis of this relation between graph structures and implied conditional independence relations invokes the discussion of *d-separation*. However, d-separation (despite claims that "[t]he intuition behind [it] is simple") is a more subtle concept than it at first appears as it involves both which nodes are observed and the underlying structure.

Though accurate, this eschews some important aspects of the semantics that distinguish arrows with different directions when you consider the kinds of values that the variables take on.

Directional semantics between different types of nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conditional distributions of child nodes are usually defined with parameter functions that take as arguments their parents' realizations for that trial. Bayes nets often are used to exclusively represent discrete (usually, binary) nodes the distribution is usually defined as an arbitrary probability distribution associated with the label of it's parent's realization. 

If we allow (for example) positive continuous valued nodes to exist in relation to discrete nodes the kind of distributions available to describe relations between these nodes changes depending upon the direction of the arrow. A continuous node taking on positive real values mapping to an arbitrarily labeled binary node taking on values :math:`\{a,b\}` will require a function that maps from :math:`\mathbb{R} \rightarrow [0,1]`, where it maps to the probability that the child node takes on (for instance) the value :math:`a` [#]_.However, if the relationship goes the other direction, one would need to have a function that maps from :math:`\{a,b\} \rightarrow \mathbb{R}`. For example, this might be a Gaussian distributions for *a* and *b* (:math:`(\mu_a,\sigma_a),(\mu_b,\sigma_b)`). Regardless of the particular distributions, the key is that the functional form of the distributions are radically different 

.. [#] If the function maps directly to one of the labeled binary values this can be represented as having probability 1 of mapping to either :math:`a` or :math:`b`.


Sampling and semantics in Bayes Nets
====================================

This procedure for sampling a trial from Bayesian networks relies heavily on using what I call the *active sample set*. This is the set of nodes for which we have well-defined distributions at the time of sampling.

There will always be at least one node in a Bayesian network that has no parents (for a given trial). We will call these nodes **orphans**. To sample a trial from the Bayesian network we begin with the orphans. 

Because orphans have no parents – in order for the Bayes net to be well-defined – each orphan will have a well-defined marginal probability distribution that we can directly sample from. Thus we start with the set of orphans as the *active sample set*. 

After sampling from all of the orphans, we will take the union of the sets of children of the orphans, and at least one of these nodes will have values sampled for all of its parents. We take the set of orphans whose entire parent-set has sampled values, and sample from the conditional distributions defined relative to their parents' sampled values and make this the *active sample set*.

After each set of samples from the *active sample set* we will either have new variables whose distributions are well-defined or will have sampled all of the variables in the graph for that trial. [#]_ If we have multiple trials, we repeat this procedure for each trial. 

.. [#] One potential worry is the case of disconnected graphs (i.e., graphs that can be divided into at least 2 disjoint sets of nodes where there will be no edges between nodes of different sets). However, because disconnected subgraphs of a :sc:`dag` will also be :sc:`dag`\s, we can count on at least one orphan existing for each of those graphs, and thus we will be able to sample from all disconnected subgraph by following the same algorithm above (they will just be sampled in parallel).

Example: Rain, Sprinkler & Ground
=================================


..  figure:: sprinkler.pdf
    :figclass: bht


    An Bayesian network describing the sprinkler example. :label:`sprinkler`

In the sprinkler Bayesian network in Figure :ref:`sprinkler` [#]_, there three discrete nodes that represent whether it *Rains* (yes or no), whether the *Sprinkler* is on (on or off) and whether the *Ground* is wet (wet or dry). The edges encode the fact that the rain listens to no one, that the rain can alter the probability of whether the sprinkler is on, and the rain and the sprinkler together determine whether the ground is wet.

.. [#] This technically is not a well specified example of a Bayesian network, because while I have specified the states and their relations, I left open the potential interpretation of the parameters and how they relate to one another. I did so because it shows the limits of what is encoded knowing only the Bayes net structure and its 


Causal Bayesian Networks
------------------------

Causal Bayesian networks are Bayesian networks that are given an interventional operation that allows for "graph surgery" by cutting nodes off from their parents. [#]_ The central idea is that interventions are cases where some external causal force is able to "reach in" and set the values of individual nodes, rendering intervened on independent of their parent nodes. 

.. [#] This is technically a more general definition than that given in :cite:`pearl2000` as in that case there is a specific semantic flavor given to interventions as they affect the probabilistic semantics of the variables within the network. Because here we are considering a version of intervention that affects the *structure* of a set of graphs rather than an intervention's results on a specific parameterized graph, this greater specificity is unnecessary.

NetworkX :cite:`networkx`
-------------------------

This is a package for representing, manipulating and analyzing graphs and complex networks. It stores different kinds of graphs as variations on a "dict of dicts of dicts" structure. For example, directed graphs are stored as two dict-of-dicts-of-dicts structures. It can also represent multi-graphs (graphs where multiple versions of "the same" edge from the adjacency matrix perspective can exist and will (usually) carry different semantics). We will not be using the multigraph feature of NetworkX, as multigraphs are not traditionally used in the context of Bayesian networks.

Basic NetworkX operations
=========================

NetworkX is usually imported using the :code:`nx` abbreviation, and you input nodes and edges as lists of tuples, which can be assigned dictionaries as their last argument, which stores the dictionary as the nodes' or edges' data.

..  code-block:: python
    
    import networkx as nx  

    G = nx.DiGraph() # init directed graph
    G.add_edges_from(edge_list) # input edges 
    G.add_nodes_from(node_list) # input nodes
    edge_list = G.edges(data=True) # output edges
    node_list = G.nodes(data=True) # output nodes

Causal Bayesian NetworkX: Graphs
--------------------------------

Here we will look at some of the basic operations described in the `ipython notebook` :cite:`scipy` found at |cbnx|_.

Other packages
==============

In addition to networkX, we need to import numpy :cite:`numpy`, scipy :cite:`scipy`, and functions from itertools.

..  code-block::python

    import numpy as np
    import scipy
    from itertools import chain, combinations, tee


Beginning with a max-graph
==========================

Starting with the max graph for a set of nodes (i.e., the graph with :math:`N^2` edges), we build an iterator that returns graphs by successively removing subsets of edges. Because we start with the max graph, this procedure will visit all possible subgraphs. One challenge that arises when visiting *all* possible subgraphs is the sheer magnitude of that search space (:math:`2^{N^2}`).

..  code-block:: python

    def completeDiGraph(nodes):
        """
        Building a max-graph from a set of n nodes.
        This graph has :math:`n^2` edges.
        Variables:
        nodes are a list of strings comprising node names
        """

        G = nx.DiGraph() # Creates new graph
        G.add_nodes_from(nodes) # adds nodes to graph
        edgelist = list(combinations(nodes,2)) 
        # list of directed edges
        edgelist.extend([(y,x) for x,y in edgelist)
        #add symmetric edges
        edgelist.extend([(x,x) for x in nodes]) 
        # add self-loops
        G.add_edges_from(edgelist) # add edges to graph
        return G

Preëmptive Filters
==================

In order to reduce the set of edges that we need to iterate over, rather than working over the max-graph for *any* of nodes, it helps to determine which individual edges are known to always be present and which ones are known to never be present. In this way we can reduce the size of the edgeset over which we will be iterating. 

Interestingly, this allows us to include more variables/nodes without the explosion of edges that would be the consequence of adding additional nodes were we not to include preëmptive filters.

One of the most powerful uses I have found for this is the ability to modify a graph set to include interventional nodes without seeing a corresponding explosion in the number of graphs. On the assumption that interventions apply only to a single node () example nodes representing interventions, as nodes without on the preëxisting variables that.

..  code-block:: python

    def filter_Graph(G,filter_set):
        """
        This allows us to apply a set of filters encoded 
        as closures that take a graph as input
        and return a graph as output.
        """
        graph = G.copy()
        for f in filter_set:
            graph = f(graph)
        return graph

Example filter: remove self-loops
=================================

By default the graph completed by :code:`completeDiGraph()` will have self-loops, often we will not want this (e.g., :sc:`dag`\s cannot contain self-loops).

.. code-block:: python

    def extract_remove_self_loops_filter():
        def remove_self_loops_filter(G):
            graph = G.copy()
            graph.remove_edges_from(graph.selfloop_edges())
            return graph
        return remove_self_loops_filter

.. Example filter use-case: add intervening nodes to a existing graph
.. ==================================================================

.. By default the graph completed by :code:`completeDiGraph()` will have self-loops, often we will not want this (e.g., :sc:`dag`\s cannot contain self-loops).

.. .. code-block:: python

..     def extract_remove_self_loops_filter():
..         def remove_self_loops_filter(G):
..             graph = G.copy()
..             graph.remove_edges_from(graph.selfloop_edges())
..             return graph
..         return remove_self_loops_filter



Conditions
==========

The enumeration portion of this approach is defined in this :code:`conditionalSubgraphs` function.[#]_ This allows you to pass in a graph from which you will want to sample subgraphs that meet the conditions that you also pass in. 

.. [#] Note that powerset will need to be built (see |cbnx|_ for details).

..  code-block:: python

    def conditionalSubgraphs(G,condition_list):
        """
        Returns a graph iterator of subgraphs of G 
        meeting conditions in condition_list.

        Variables: 
        G: a graph from which subgraphs will be taken.
        condition_list: a list of condition functions.
        
        Functions in condition_list have i/o defined as
        input: graph, generated as a subgraph of G
        output: Bool, whether graph passes condition
        """

        for edges in powerset(G.edges()):
            G_test = G.copy()
            G_test.remove_edges_from(edges)
            if all([c(G_test) for c in condition_list]):
                
                yield G_test


Example condition: detecting :sc:`dag`\s
========================================

If we wanted to have examples of all dags that are subgraphs of a passed in graph, we can use a convenient networkX utility.

..  code-block:: python

    def create_path_complete_condition(node_pairs):
        def path_complete_condition(G):
            return all([nx.has_path(G,x,y) for x,y in node_pairs])
        return path_complete_condition

Non-destructive conditional subgraph generators
===============================================

Because the :code:`conditionalSubgraph` generator produces an iterable, if we want to apply a conditional after that initial set is generated, we need to split it into two copies of the iterable. This involves the :code:`tee` function from the :code:`itertools` core package.

.. code-block:: python

    def new_conditional_graph_set(graph_set,cond_list):
        """
        Returns graph_set & a new iterator which has 
        conditions in cond_list applied to it.
        
        Warning: This function will devour the iterator 
        you include as the `graph_set` input, 
        you need to redeclare the variable as 
        one of the return values of the function.
        
        Thus a correct use would be:    
        a,b = new_conditional_graph_set(a,c)
        
        The following would not be a correct use:
        x,y = new_conditional_graph_set(a,c)
        
        Variables: 
        graph_set: graph-set iterator generator
        cond_list: list conditions
            input: a graph.
            output: boolean value
        """
        
        graph_set_newer, graph_set_test = tee(graph_set,2)
        def gen():
            for G in graph_set_test:
                G_test = G.copy()
                if all([c(G_test) for c in condition_list]):
                    yield G_test
        return graph_set_newer, gen()

Filters versus Conditions: which to use
=======================================

The most obvious structural differences between filters and conditions give insight to how they are to be used. 

Filters are intended to apply to the max graph to reduce the edge set. They take (at least) a graph as an argument and return a graph. This is meant to be a transformation of the graph, or a way to change the value of a graph in place. It does not have any notion of producing both copies of the graph (though that could be done as well).

Conditions are intended to be applied to a series of graphs generated by an iterator taking subgraphs of some other graph

.. Naming conventions for filters and conditions
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. The convention I have been following for distinguishing filter and condition functions is that the higher-order function in the case of filters beginning with the word :code:`extract_`, and then both the returned function and the higher-order function ending with the word :code:`filter`. Similarly, conditions have begun with :code:`create_` and finished with :code:`condition`.

.. ..  code-block:: python

..     def extract_name_filter(node_list):
..         def name_filter(G):
..             graph = G.copy()
..             # operations removing edges
..             return graph
..         return name_filter


.. ..  code-block:: python

..     def create_name_condition(node_list):
..         def name_condition(G):
..             # operations checking for whether conditions hold
..             return # truth value
..         return name_condition

.. Complex example: adding interventional nodes
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ..  code-block:: python

..     def add_interventions(G):
..         node_list = G.nodes()
..         edge_list = G.edges()

..         int_node_list = [str(x)+"_int" for x in node_list]

..     def completeDiGraph(nodes):
..         """
..         Building a max-graph from a set of nodes. This graph has
..         :math:`n^2` edges in terms of len(nodes).
..         Variables:
..         nodes are a list of strings that specify the node names
..         """

..         G = nx.DiGraph() # Creates new graph
..         G.add_nodes_from(nodes) # adds nodes to graph
..         edgelist = list(combinations(nodes,2)) 
..         # list of directed edges
..         edgelist.extend([(y,x) for x,y in list(combinations(nodes,2))]) 
..         #add symmetric edges
..         edgelist.extend([(x,x) for x in nodes]) # add self-loops
..         G.add_edges_from(edgelist) # add edges to graph
..         return G

..     def extract_remove_self_loops_filter():
..         def remove_self_loops_filter(G):
..             graph = G.copy()
..             graph.remove_edges_from(graph.selfloop_edges())
..             return graph
..         return remove_self_loops_filter


.. Gates: Context-sensitive causal Bayesian networks
.. -------------------------------------------------

|cbnx|: Representing probabilistic relations and sampling
---------------------------------------------------------

It is possible to identify first those nodes with no parents, and then sample them. Then (and this is the part that is iterated), sample those in the children set whose parent set has a full sample, removing children from the children set when they are sampled from and adding them to the parent set. This is continued until there are no children in the children set. 

This is the algorithm that sampling follows as can be observed in |cbnx|_. This approach only works for :sc:`dag`\s and is formally equivalent to identifying a *topological ordering* for the nodes and then sampling accordingly. A graph having a topological ordering is biconditionally equivalent to being a :sc:`dag`. This criterion can roughly be seen as assigning each node an integer such that every child will always have an integer greater than any of its parent nodes (and by recursion any of its ancestor nodes). This provides an order in which to visit the nodes for sampling purposes that will ensure that any nodes in a child's parent set will always be visited first. This also results in choosing those nodes with an empty set as a parent set (i.e., orphans) to have the lowest integers, and therefore to be sampled first.

Most of the difficult parts of encoding a sampling procedure though have nothing to do with the algorithm(it is standard for :sc:`dag`\s). Rather, they arise from attempting to store the relevant information within the NetworkX data nodes, so that a self-contained graphical object can be imported and exported. There is a general problem of a lack of standard storage format for Bayesian networks (and probabilistic graphical models in general). This is just one flavor of that problem. 

Example: |cbnx| implementation for sprinkler graph
==================================================

.. code:: python

    node_prop_list = [("rain",{
        "state_space":("yes","no"), 
        "sample_function": "choice",
        "parents":[], 
        "dist":[.2,.8]}),
        ("sprinkler",{
        "state_space":("on","off"),
        "sample_function": "choice",
        "parents":["rain"], 
        "dist":{(("rain","yes"),):[.01,.99],
                (("rain","no"),):[.4,.6]}}),
        ("grass_wet",{
        "state_space":("wet","dry"),
        "sample_function": "choice",
        "parents":["rain","sprinkler"],
        "dist":{
            (("rain","yes"),("sprinkler","on")):[.99,.01],
            (("rain","yes"),("sprinkler","off")):[.8,.2],
            (("rain","no"),("sprinkler","on")):[.9,.1],
            (("rain","no"),("sprinkler","off")):[0,1]}})]


    edge_list = [("sprinkler","grass_wet"),
        ("rain","sprinkler"),
        ("rain","grass_wet")]

    def sample_from_graph(G,f_dict=None,k = 1):
        if f_dict == None:
            f_dict = {"choice": np.random.choice}
        n_dict = G.nodes(data = True)
        n_ids = np.array(G.nodes())
        n_states = [(n[0],n[1]["state_space"]) for n in n_dict]
        orphans = [n for n in n_dict if n[1]["parents"]==[]]
        s_values = np.empty([len(n_states),k],dtype='U20')
        s_nodes = []
        for n in orphans:
            samp_f = str_to_f(n[1]["sample_function"],
                f_dict)
            s_states = n[1]["state_space"]
            s_dist = n[1]["dist"]
            s_idx = G.nodes().index(n[0])
            s_values[s_idx,:]  = samp_f(s_states,
                size=[1,k],p=s_dist)
            s_nodes.append(n[0])
        while set(s_nodes) < set(G.nodes()):
            nodes_to_sample = has_full_parents(G,s_nodes)
            for n in nodes_to_sample:
                par_indices = [(par,G.nodes().index(par)) 
                    for par in G.node[n]["parents"]]
                par_vals = [(par[0],s_values[par[1],:]) 
                    for par in par_indices]
                samp_index = G.nodes().index(n)
                s_values[samp_index,:] = cond_samp(G,n,
                    par_vals,f_dict,k)
                s_nodes.append(n)
        return s_values
    
    def has_full_parents(G,s_n):
        check_n = [x for x in G.nodes() if x not in s_n]
        nodes_to_be_sampled = []
        for n in G.nodes(data = True):
            if (n[0] in check_n) & (n[1]["parents"]<=s_n):
                nodes_to_be_sampled.append(n[0])
        if len(nodes_to_be_sampled)==0: 
            raise RuntimeError("A node must be sampled")
        return nodes_to_be_sampled

    def nodeset_query(G,n_set,n_atr=[]):
        if len(n_atr)==0:
            return [n for n in G.nodes(data = True) 
                if n[0] in n_set]
        else:
            return_val = []
            for n in G.nodes(data=True):
                if n[0] in node_set:
                    return_val.append((n[0],
                    {attr:n[1][attr] for attr in n_atr}))
            return return_val
        
        
    def cond_samp(G,n,par_vals,f_dict, k = 1):
        try: n in G
        except KeyError:
            print("{} is not in graph".format(n))
        output = np.empty(k,dtype="U20")
        for i in np.arange(k):
            val_list = []
            for p in par_vals:
                val_list.append(tuple([p[0],p[1][i]]))
            samp_dist = G.node[n]["dist"][tuple(val_list)]
            samp_f = str_to_f(
                G.node[n]["sample_function"],f_dict)
            samp_states = G.node[n]["state_space"]
            temp_output = samp_f(samp_states,
                size=1,p=samp_dist)
            output[i] = temp_output[0]
        return output

    def str_to_f(f_name, f_dict=None):
        if f_dict == None:
            f_dict = {"choice": np.random.choice}
        try: f_dict[f_name]
        except KeyError:
            print("{} is not defined.".format(f_name))
        return f_dict[f_name]



Lack of JSON compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^

The data-structures described above, notably the node_prop_listI discovered that my method of storing groups of parent-variable realizations as tuple-keys with distribution arguments broke the JSON compatibility that I was able to maintain in other circumstances. This was useful for being able to call the distributions of variables who have more than one parent nodes. I have not yet fixed this problem. 

Causal Theories and Computational Cognitive Science
---------------------------------------------------

**Theory based causal induction** is a formal framework that arose out of the tradition in computational cognitive science to approach problems of human cognition with rational, computational-level analyses :cite:`griffithst09`. In particular, causal theories form generative models for defining classes of parameterized probabilistic graphical models. They rely on defining a set of classes of entities (ontology), potential relationships between those classes of entities and particular entities (plausible relations), and particular parameterizations of how those relations manifest in observable data (or in how other relations eventually ground out into observable data). This allows Griffiths and Tenenbaum to subsume the prediction of a wide array of human causal inductive, learning and reasoning behavior using this framework for generating graphical models and doing inference over the structures they generate.

Rational analysis
=================

A technique used that allows us to model not cognition per se, but the situation into which cognitive capacities are to be placed. If we assume that we know the inputs, the outputs and the goal state of the arbitrary cognitive agent, we can iteratively predict the agent's behavior [#]_.

Precisely Specify what are the goals of the cognitive system.
Develop a formal model of the environment to which the system is adapted(almost certainly less structured than the standard experimental situation)
make the minimal assumption about computational limitations. This is where one speifies the constraints of evolutionary history. To the extent that these assumptions are minimal, the analysis is powerful.
Derive the optimal behavioral functiong iven 1–3
examine empirical literature to see if the predictions of the behaviorl function are confirmed.
If the predictions are off, iterate. In my own experience my problems have been with the mathematical analysis required in step 4, which can often be quite complex.

Predictions flow from 1–3. Framing of the information processing problems A mechanistic-free casting of a psychological theory. Most of the interesting assumptions come ins tep 2 because the structure of the environment is what is easiest to verify.

This is often coupled with computational-level analysis inspired by Marr's :cite:`marr82` levels of analysis.  

.. [#] This is not a well-sourced definition. I need to go back to :cite:`andersons91` to spruce it up.

Computational-Level Analysis of Human Cognition
===============================================

A computational-level analysis is one in which we model a system in terms of its functional role(s) and how they would be optimally solved. This is distinguished from algorithmic-level analysis by not caring how this goal achievement state is implemented in terms of the formal structure of the underlying system and from mechanistic-level analysis by not caring about the physical structure of how these systems are implemented (which may vary widely while still meeting the structure of the algorithmic-level which itself accomplishes the goals of the computational level).

A classic example of the three-levels of analysis are different ways of studying flying with the example of bird-flight. The mechanistic-level analysis would be to study feathers, cells and so on to understand the component subparts of individual birds. The algorithmic-level analysis would look at how these subparts fit together to form an active whole that is capable of flying often by flapping its wings in a particular way. The computational-level analysis would be a theory of aerodynamics with specific accounts for the way forces interact to produce flight through the particular motions of flying observed in the birds.

Causal theories: ontology, plausible relations, functional form
===============================================================

Griffiths and Tenenbaum :cite:`griffithst09` identify that their framework generalizes specifying Bayesian network in the same way first-order logic generalizes specifying propositions in propositional logic. It requires the elements necessary to populate a graph with nodes, those nodes with properties, and relations between the nodes, stating which of those relations are plausible(and how plausible), and a specific, precise formulation for how those relations manifest in terms of the semantics. In the terms of :cite:`griffithst09`'s theory-based causal induction, this requires specifying an ontology, plausible relations over those ontologies, and functional forms for parameterizing those relations.

Ontology
^^^^^^^^

This specifies the full space of potential kinds of entities, properties and relations that exist. This is the basis around which everything else will be defined. 

Note that it is easy enough to populate nodes with features using the data field in NetworkX.

Plausible Relations
^^^^^^^^^^^^^^^^^^^

This specifies which of the total set of relations allowed by the ontology are plausible. For example, we know that in most situations a fan is more likely than a tuning fork to blow out a candle. 

As mentioned above, once you have a well-populated world if you do not dramatically restrict the sets of relations you consider, there will be an explosion of possibilities. People, even young children :cite:`griffithst09`, have many expectations about what sorts of things can can feasibly be causally related to one another. This sometimes has been interpreted as the plausible existence of a 

Functional form
^^^^^^^^^^^^^^^

> Even in the most basic cases of causal induction we draw on expectations as to whether the effects of one variable on another are positive or negative, whether multiple causes interact or are independent, and what type of events (binary, continuous, or rates) are relevant to evaluating causal relationships.
:cite:`griffithst09`


Generalizations to other kinds of logical/graphical conditions
==============================================================

The Griffiths and Tenenbaum framework is richer than the set of examples developed in :cite:`griffithst09`. It can also express conditions of graphical connectivity, context-sensitive functional forms, substructures of constrained plausible relations, among many others.

In :cite:`griffithst09`, plausible relations are described in terms of sufficient conditions, implicitly suggesting that most relations are not plausible. However, we can also make necessary statements about the kinds of relations that *must* be there. And one can see this as selecting a subset of all the possible graphs implementable by the set of nodes defined by the ontology. It is for this purpose that I first arrived at the node enumeration.

Part of the aim of developing |cbnx|_ is to provide a programming framework in which the richness of causal theories are able to be expressed. Because of the utilities in :code:`networkX`, with the enumerating, filtering and conditioning functions described above, it becomes much easier to implement higher-order graphical conditions (e.g., a directed path necessarily existing between two nodes) than in the original notation described in the framework. These ideas were entirely expressible in the original mathematical framework, but would have required a good deal more notational infrastructure to represent. Here, we not only provide a notation, but a computational infrastructure for applying these kinds of conditions.


Uses in modeling human cognition
================================

Using this framework, Griffiths and Tenenbaum were able to provide comprehensive coverage for a number of human psychology experiments. This allows them to model people's inferences in causal induction and learning regarding different functional forms, at different points in development, with different amounts of data, with and without interventions, and in continuous time and space (to name only a few of the different conditions covered).

They successfully modeled human behavior using this framework by treating people as optimal solvers of this computational problem [#]_ (at least as defined by their framework). Furthermore, by examining different but related experiments, they were able to demonstrate the different ways in which specific kinds of prior knowledge are called upon differentially to inform human causal induction resulting in quite different inferences on a rational statistical basis.

Cognition as Benchmark, Compass, and Map
========================================

People have always been able to make judgments that are beyond machine learning's state-of-the-art. In domains like object recognition, we are generally confident in people's judgments as veridical, and – as such – they have been used as a benchmark against which to test and train machine learning systems. The eventual goal is that the system reaches a Turing point — the point at which machine performance and human performance are indistinguishable.

But that is not the only way human behavior can guide machine learning. In domains like causal induction, people's judgments cannot form a benchmark in the traditional sense because we cannot trust people to be "correct". Nonetheless, people *do* make these judgments and, more importantly, these judgments exhibit systematic patterns. This systematicity allows the judgments output by cognition to be modeled using formal, computational frameworks. Further, if we formally characterize both the inputs to *and* outputs from cognition, we can define judgments as optimal according to some model. Formal models of individual cognitive processes can then act as a compass for machine learning, providing a direction for how problems and some solutions can be computed.

Formal frameworks for generating models (e.g., causal theories) can be even more powerful. Data can often be interpreted in multiple ways, with each way requiring a model to generate solutions. Holding the data constant, different goals can merit different kinds of solutions. Frameworks that generate models, optimality criteria and solutions not only provide a direction for machine learning, but lay out *sets* of possible directions. Generalized methods using a single system for solving many different kinds of problems provide the ability to relate these different directions to each other. As such, formalizing the inputs, processes and outputs of human cognition produces a map of where machine learning could go, even if it never goes to any particular destination. From this, better navigators with more details about the particular terrain can find better routes. 


.. [#] Optimality in these cases is taken to mean on average approximating the posterior distribution of some inference problem defined by the authors in each case.
