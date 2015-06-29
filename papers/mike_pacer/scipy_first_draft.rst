:author: Mike Pacer
:email: mpacer@berkeley.edu
:institution: University of California at Berkeley
:bibliography: myBibliography

:video: youtube_link_to_come

..  latex::
    :usepackage: booktabs

..  latex::

    %:usepackage: mathpalette, graphicx
    \newcommand{\bigCI}{\mathrel{\text{\scalebox{1.07}{$\perp\mkern-10mu\perp$}}}}
    \newcommand\independent{\protect\mathpalette{\protect\independenT}{\perp}}
    \def\independenT#1#2{\mathrel{\rlap{$#1#2$}\mkern2mu{#1#2}}}


.. raw::latex

    \newcommand{\DUrolesc}{\textsc}
..  \providecommand*\DUrolecitep[1]{\citep{#1}}
..  \newcommand\DUrolecitep[1]{\citeA{#1}}
    \newcommand{\DUroleindep}{\mathrel{\text{\scalebox{1}{$\perp\mkern-9mu\perp$}}}}


.. role:: indep

.. role:: sc



------------------------
Causal Bayesian NetworkX
------------------------

..  class:: abstract

    Humans are existence proofs for the solubility of computational causal inference.

    Computational problems are sometimes thought to be the exclusive domain of computer science, though the solutions found prove vital for many other sciences. But computational cognitive science can also contribute to the solution of computational problems, particularly inductive problems. Causal inference (and inductive problems more generally) have proven resilient to traditional analysis, and the recent progress on these problems observed in machine learning (e.g., neural networks and their extensions) originate in models formulated by cognitive scientists.

    As a computational cognitive scientist, I use a technique called rational analysis, which roughly consists of developing formal models of how *any* cognitive agent might optimally solve a computational problem and comparing that to how people actually solve analogous problems. In the course of doing so we find that people turn out to make much more sense than popular psychology might lead you to believe. At the same time, we create formal frameworks that represent entities and relations in the world as well as reasoning processes over those representations. 

    One of the frameworks successfully used in this way are causal Bayesian networks. Bayesian networks fall within the more general class of probabilistic graphical models, specifically, directed acyclic graphs with associated conditional probability distributions. Directed arrows encode direct dependency relationships going from parents to their children, whose conditional probability distribution is defined in terms of the parents' values. *Causal* Bayesian networks are endowed with an intervention operation that allows "graph surgery" in which one cuts variables off from their parents (usually setting it to a particular value). 

    I have developed tools on top of the :code:`NetworkX` package that allow me to implement some aspects of these models. By treating graph definition as one of enumeration and filtering rather than investigating invidual graphs, we are able to more conveniently state constraints on the graph structures under question. Indeed this gives an alternative view of intervention not as the modification of an single graph, but as a constraint on the total set of graphs. This allows us to treat the graphical aspects of the problem separately from the probabilistic semantics that define particular models on those graphs. I call this set of tools `Causal Bayesian NetworkX`.


..  class:: keywords

    probabilistic graphical models, causality, intervention

Introduction and Aims
---------------------

My first goal in this paper is to provide enough of an introduction to some formal/mathematical tools such that those familiar with :code:`python` and programming more generally will be able to appreciate both why and how one might implement causal Bayesian networks. Especially to exhibit *how*, I have developed parts of a toolkit that allows the creation of these models on top of the :code:`NetworkX` python package. Given the coincidence of the names, it seemed most apt to refer to this toolkit as :code:`Causal Bayesian NetworkX` (the current implementation of which can be found at `Causal Bayesian NetworkX`_).

In order to understand the toolset requires the basics of probabilsitic graphical models, which requires understanding some graph theory and some probability theory. The first few pages are devoted to providing necessary background and illustrative cases for conveying that understanding. 

Notably, contrary to how Bayesian networks are commonly introduced, I say relatively little about inference from observed data. This is intentional, as is this discussion of it. Many of the most trenchant problems with Bayesian networks are found in critiques of their use to infer these networks from observed data. But, many of the aspects of Bayesian networks (especially causal Bayesian networks) that are most useful for thinking about problems of structure and probabilistic relations do not rely on inference from observed data. In fact, I think the immediate focus on inference has greatly hampered widespread understanding of the power and representative capacity of this class of models. Equally – if not more – importantly, I aim to discuss generalizations of Bayesian networks such as those that appear in :cite:`griffithst09,winn2012causality`, and inference in these cases requires a much longer treatement (if a comprehensive treatment can be provided at all). If you are dissatisfied with this approach and wish to read a more conventional introduction to (causal) Bayesian networks I suggest consulting :cite:`pearl2000`.

The Causal Bayesian NetworkX toolkit can be seen as consisting of two main parts: graph enumeration/filtering and the storage and use of probabilistic graphical models in a NetworkX compatible format. Because this topic will more be the focus of my talk which can be viewed at the youtube link above and the source code of the most basic implementation is available at `Causal Bayesian NetworkX`_, in this paper I focus more on the other aspects of the problem. Nonetheless, appreciating these other aspects is made easier by also appreciating the problems of implementation/representation and the early solutions that I propose.

I focus first on establishing a means of building iterators over sets of directed graphs. I then apply operations to those sets. Beginning with the complete directed graph, we enumarte over the subgraphs of that complete graph and enforce graph theoretic conditions such as acyclicity over the entire graph, guarantees on paths between nodes that are known to be able to communicate with one another, or orphan-hood for individual nodes known to have no parents. We accomplish this by using closures that take graphs as their input along with any explicitly defined arguments needed to define the exact desired conditions. 

I then shift focus to a case where there is a specific known directed acyclic graph that is imbued with a simple probabilistic semantics over its nodes and edges, also known as a Bayesian network. I demonstrate how to sample independent trials from these variables in a way consistent with these semantics. I discuss briefly some of the challenges of encoding these semantics in dictionaries as afforded by NetworkX without resorting to :code:`eval` statements and discuss compatibility issues I have found with JSON storage formats. 

.. Then, I will briefly discuss **gates**:cite:`winn2012causality`, an extension to graphical modeling frameworks that allow one to define context-specific dependence relations (which includes context-specific *independence* relations). This extension is of particular interest as it allows us to subsume the classical :code:`do`-calculus :cite:`pearl2000` into the more general semantics of the probabilistic network(though not a Bayesian network since that only expresses context-free independence relations). This work was a key influence in the development of thinking about interventions not as operations on individual nodes, or even individual graphs, but as a particular constraint placed on sets of graphs by some generative process. This interpretation of intervention, however, more difficult to relate to the semantics of probabilistic networks. I expect that **gates** will aid in bridging between this 

I conclude with a discussion of some of the problems that have been addressed in Cognitive Science through the use of graphical models like those described. In particular, I will discuss a framework called **causal theories** :cite:`griffithst09` which allows for defining problems of causal induction. It is out of this framework the perspective expressed in this paper, the associated talk, and the the Causal Bayesian NetworkX toolkit developed. 

.. _Causal Bayesian NetworkX: https://github.com/michaelpacer/Causal-Bayesian-NetworkX

Graphical Models
----------------

Graphs are formal models defined by a set of nodes (:math:`X, |X| = N`) and edges between those nodes (:math:`e \in E \equiv e \in (X \times X)`, where *E* is the set of edges). 

Notes on notation
=================

Nodes
^^^^^

In the examples in `Causal Bayesian NetworkX`_, nodes are given explicit labels individuating them such as :math:`\{A,B,C,\ldots\}` or {'rain','sprinkler','grass_wet'}. Oftentimes, for the purposes of mathematical notation, it will be helpful to index nodes by the integers over a common variable label, e.g., using  :math:`\{X_1,X_2,X_3,\ldots\}`. [#]_ 

.. [#] Despite pythonic counting beginning with 0, I chose not to begin this series with 0 because when dealing with variables that might be used in statistical regressions, the 0 subscript will have a specific meaning that separates it from the rest of the notation. For example when expressing multivariate regession as :math:`Y = \beta X + \epsilon, \epsilon \sim \mathcal{N}(0,\Sigma)`, :math:`\beta_0` refers to the parameter associated with a constant variable :math:`x_0 = 1` and :math:`X` is normally defined as :math:`x_1, x_2, x_3, \ldots`. This allows a simple additive constant to be estimated, which usually(but not always) is not of interest to statistical tests, acting as a scaling constant more than anything else. This also makes for simpler notation than saying :math:`Y = \beta_0 + \beta X + \epsilon`, since that is equivalent to the previous notation (:math:`Y = \beta X + \epsilon`) if :math:`x_0 = 1`. In other cases :cite:`griffithst05,pacerg12`, the 0 index will be used to indicate background sources for events in a system.

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

.. Finding paths using adjacency matrices
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. It is straightforward to interpret questions of the existence of paths between :math:`X_i` and :math:`X_j` using the adjacency matrix perspective and matrix multiplication. The key step is to recognize that you can think of multiplying the adjacency matrix from the right by a binary vector as taking a step in the graph from the nodes whose values in the vector were 1 to the set of children of those nodes. To continue to have a binary vector then requires resetting values in the vector 0 and 1 by taking (for every element of the resulting vector) the minimum of the value of the vector and 1 (which addresses the case where more than one edge leads into the same node). 

.. To use this technique to test whether a matrix has an edge between, if you have a value of 1 at index *i*, and 0's elsewhere, if you multiply this vector from the left by the adjacency matrix, then if there is a path between 

Undirected Graphs
=================

We can still have a coherent view of *undirected* graphs, despite the fact that our primitive notion of an edge is that of a *directed* edge. If a graph is undirected, then if it has an edge from :math:`X_i \rightarrow X_j` then it has an edge from :math:`X_j \rightarrow X_i`. Equivalently, this means that the adjacency matrix of the graph is symmetric, or :math:`A(G)=A(G)^\top`.


Directed Graphs
===============

From the adjacency matrix perspective we've been considering, all graphs are technically directed, and undirected graphs are a special case where both edges are symmetric.

The number of directed graphs that can be obtained from a set of nodes of size :math:`n` can be defined explicitly using the fact that they can be encoded as a unique :math:`n \times n` matrix:

.. math::

    R_n = 2^{n^2}


Directed Acyclic Graphs
^^^^^^^^^^^^^^^^^^^^^^^

A cycle in a directed graph can be understood as the existence of a path from a node to itself. This can be as simple as a self-loop (i.e., if there is an edge :math:`(X_i,X_i)` for any node :math:`X_i`). 

Directed acyclic graphs(:sc:`dag`\s) are directed graphs that contain no cycles.

The number of :sc:`dag`\s that can be obtained from a set of nodes of size :math:`n` can be defined recursively as follows :cite:`mckay2003acyclic` :

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

.. [#] If one is dealing with continuous quantities rather than discrete quantities one will have to use a probability density function (pdf) which does not have as straightforward an interpretation as a probability mass function. This difficult stems from the fact that (under most cases) the probability of any particular event occuring is "measure zero", or "almost surely" impossible. Without getting into measure theory and the foundation of calculus and continuity we can simply note that it is not that any individual event has non-zero probability, but that sets of events have non-zero probability.As a result, continuous random variables are more easily understood in terms a cummulative density function (cdf), which states not how likely any individual event is, but how likely it is that the event in question is less than a value :math:`x`. The notation usually given for a cdf of this sort is :math:`F(X\leq x) = \int_{-\infty}^{x}f(u)du`, where :math:`f(u)` is the associated probability density function.

The conditional probability of a variable :math:`X` taking on value :math:`x` once it is known that another variable :math:`Y` takes on value :math:`y` is :math:`P(X=x|Y=y)`. Much like above, if we want to consider the probability of each possible event without specifying one, sometimes this will be written as :math:`P(X|Y=y)`. If we are considering conditioning on any of the possible values of the known variable, we might use the notation :math:`P(X|Y)`, but that is a slight abuse of the notation. 

You *can* view :math:`P(X|Y)` as a function over the space defined by :math:`X\times Y`. However, if you do so, do not interpret this as a probability function (of any kind). Rather, this defines a probability function for :math:`X` relative to each value of :math:`Y`. Without conditioning on :math:`Y` we have many potential functions of X. Thus, you can think of that as denoting a *family* of probability functions indexed by the various values :math:`Y=y`.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following example:

..  math::
    :type: eqnarray

    X &\sim& \textrm{Bernoulli}_{\{0,1\}}(.5), Y \sim \textrm{Bernoulli}_{\{0,1\}}(.5)\\
    Z &=& X \oplus Y, \oplus \equiv \textsc{xor}\\

Note that, :math:`X \independent Y` but :math:`X \not\independent Y|Z`.

Sampling from Conditional Probability distributions
---------------------------------------------------

Example - Coins and dice
========================

Imagine the following game: 

You have a coin [#]_ (*C*, :sc:`Heads, Tails`), a 6-sided die (:math:`D_6, \{1,2,\ldots,6\}`), and a 20-sided die (:math:`D_{20}, \{1,2,\ldots,20\}`). If for simplicity, you prefer to think of these as fair dice and a fair coin, you are welcome to do so, but my notation will not require that.

.. [#] A coin is effectively 2-sided die, but for clarity of exposition I chose to treat the conditioned-on variable as a different kind of object than the variables relying on that conditioning.

The rules of the game are as follows: flip the coin, and if it lands on :sc:`Heads`, then you roll the 6-sided die to find your score for the round. If instead your coin lands on :sc:`Tails` your score comes from a roll of the 20-sided die. Your score for one round of the game is the value of the die that you roll, and you will only roll one die in each round. 

Suppose we wanted to know your expected score on a single round, but we do not know whether the coin will land on :sc:`Heads` or :sc:`Tails`. We cannot directly compute the probabilities for each die without first considering the probability that the coin will land on :sc:`Heads` or :sc:`Tails`. This is the 

But this discussion hides an important complexity by having the event set of the :math:`D_6` embedded within the event set of the :math:`D_{20}`. Moreover, we assumed that we could treat each event in these sets as belonging to the integers and as a result, that with little interpretation, they can be easily summed.

Coins and dice with hierarchically labeled entities, Example
============================================================

Imagine the following game: 

You have a coin (*C*, :sc:`Heads, Tails`), a *new* 6-sided die (:math:`D_6, \{X_1,X_2,\ldots,X_6\}`), and a 20-sided die (:math:`D_{20}, \{X_1,X_2,\ldots,X_{20}\}`). 

The rules are the same as before: your score for one round of the game is the value of the die that you roll, and you will only roll one die in each round. You flip the coin, and if it lands on :sc:`Heads`, then you roll the 6-sided die to find your score for the round. If instead your coin lands on :sc:`Tails` your score comes from a roll of the 20-sided die.

But note that now we cannot sum over these in the same way that we did before. Without additional information about how to map these different labels onto values, there's no way to describe the "score". Rather, the best we can do is to determine the probability with which each individual case occurs, so that once we know more about the utility curve we can efficiently use the probability distribution regardless of the particular value that is assigned.

Thus we can establish the following statements

..  latex::

    \begin{center}
    \begin{tabular}{lll}
        \toprule
        & \multicolumn{2}{c}{Parent values} \\
        \cmidrule(r){2-3}
        Probs & $P(\cdot|D_6,\textsc{h})$ & $P(\cdot|D_{20})$\\
        \midrule
        $P(X_1|\cdot)$ &$P(X_1|D_6)*P(\textsc{h})$ & $P(X_1|D_{20})*P(\textsc{t})$ \\
        \vdots     &    \vdots     & \vdots       \\
        $P(X_6|\cdot)$       &  $P(X_6|D_6)*P(\textsc{h})$     & $P(X_6|D_{20})*P(\textsc{t})$      \\
        \vdots       & \vdots     & \vdots      \\
        $P(X_{20}|\cdot)$ & 0      & $P(X_{20}|D_{20})*P(\textsc{t})$   \\
        \bottomrule
    \end{tabular}
    \end{center}

Coins and dice with disjoint sets of labeled entities, Example
==============================================================

Imagine the following game: 

You have a coin (*C*, :sc:`Heads, Tails`), a *new* 6-sided die (:math:`D_6, \{\clubsuit,\diamondsuit,\heartsuit,\spadesuit,\odot,\dagger\}`), and a 20-sided die (:math:`D_{20}, \{X_1,X_2,\ldots,X_{20}\}`). 

The rules are the same as before: your score for one round of the game is the value of the die that you roll, and you will only roll one die in each round. You flip the coin, and if it lands on :sc:`Heads`, then you roll the 6-sided die to find your score for the round. If instead your coin lands on :sc:`Tails` your score comes from a roll of the 20-sided die.

But note that now we cannot sum over these in the same way that we did before. Indeed, our event sets for the two dice are mutually disjoint, making the event set for the scores that one can receive on a single round :math:`\{\clubsuit,\diamondsuit,\heartsuit,\spadesuit,\odot,\dagger,X_1,X_2,\ldots,X_{20}\}`. Without additional information about how to map these different labels onto values, there's no way to describe the "score". Rather, the best we can do is to determine the probability with which each individual case occurs.

Bayesian Networks
-----------------

Bayesian networks are a class of graphical models that have particular probabilistic semantics attached to their nodes and edges. This makes them probabilsitic graphical models. 

The most important property of Bayesian networks is that a variable when conditioned on the total set of its parents and children, is conditionally independent of any other variables in the graph. This is known as the "Markov blanket" of that node. [#]_

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

Within a trial, all events are presumed to occur simultaneously. This means two things. First, there is no notion of temporal asynchrony, where one node/variable is takes on value before its children take on a value (even if in reality – i.e., outside the model – that variable is known to occur before its child). Secondly, the probabilistic semantics will be defined over the entirety of the graph meaning that one cannot sample a proper subset of the nodes of a graph unless they have no effects or are marginalized out with their effects being incorporated into their children.

This property also explains why Bayesian networks need to be acyclic. Most of the time when we consider causal cycles in the world the cycle relies on a temporal delay between the causes and their effects to take place. If the cause and its effect is simultaneous, it becomes difficult (if not nonsensical) to determine which is the cause and which is the effect — they seem instead to be mutually definitional. But, as noted above, when sampling in Bayesian networks simultenaity is presumed for *all* of the nodes.

Independence in Bayes Nets
==========================

One of the standard ways of describing the relation between the semantics (probability values) and syntax (graphical structure) of Bayesian networks is in terms of the graph encoding particular conditional independence assumptions between the nodes of the graph. Indeed, in some cases Bayesian networks are *defined as* a convenient representation for the conditional and marginal independence relationships between different variables. 

It is the perspective of the graphs as *merely* representing the independence relationships and the focus on inference that leads to the focus on equivalence classes of Bayes nets. The set of graphs :math:`\{A \rightarrow B \rightarrow C,~ A \leftarrow B \rightarrow C, \textrm{ and } A \leftarrow B \leftarrow C\}` represent the same conditional independence relationships, and thus cannot be distinguished on the basis of observational evidence alone. This also leads to the emphasis on finding *v-structures* or common-cause structures where (at least) two arrows are directed into the same child with no direct link between those parents(e.g.,:math:`\{A \rightarrow B \leftarrow C`). V-structures are observationally distinguishable because any reversing the direction of any of the arrows will alter the conditional independence relations that are guaranteed by the graphical structure. [#]_

.. [#] A more thorough analysis of this relation between graph structures and implied conditional independence relations invokes the discussion of *d-separation*. However, d-separation (despite claims that "[t]he intuition behind [it] is simple") is a more subtle concept than it at first appears as it involves both which nodes are obeserved and the underlying structure.

While this is accurate, it eschews some important aspects of the semantics that distinguish arrows with different directions when you consider the particular kinds of values that the variables take on.

.. Issues surrounding independence in Bayesian networks
.. ====================================================

.. Misplaced Emphasis on Independence in :sc:`dag`\s
.. =================================================

.. I do not agree with the interpretation of Bayes nets as merely representing independence properties, though, not because it is incorrect. Rather, I think it has two unfortunate results. First, it encourages poor statistical practices when it comes to inferring independence from observed data using null hypothesis testing. Second, it deëmphasizes an important assymetry that appears in the semantics of how nodes in Bayes nets relate to one another when they are not exclusively discrete nodes.

.. Null hypothesis testing and inference
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. The assumptions embedded in Bayesian networks are assumptions about the independence of different nodes. But most of the measn 

Directional semantics between different types of nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conditional distributions of child nodes are usually defined with parameter functions that take as arguments their parents' realizations for that trial. Bayes nets often are used to exclusively represent discrete (usually, binary) nodes the distribution is usually defined as an arbitrary probability distribution associated with the label of it's parent's realization. 

If we allow (for example) positive continuous valued nodes to exist in relation to discrete nodes the kind of distributions available to describe relations between these nodes changes depending upon the direction of the arrow. A continuous node taking on positive real values mapping to an arbitrarily labeled binary node taking on values :math:`\{a,b\}` will require a function that maps from :math:`\mathbb{R} \rightarrow [0,1]`, where it maps to the probability that the child node takes on (for instance) the value :math:`a` . [#]_However, if the relationship goes the other direction, one would need to have a function that maps from :math:`\{a,b\} \rightarrow \mathbb{R}`. For example, this might be a gaussian distributions for *a* and *b* (:math:`(\mu_a,\sigma_a),(\mu_b,\sigma_b)`). Regardless of the particular distributions, the key is that the functional form of the distributions are radically different 

.. [#] If the function maps directly to one of the labeled binary values this can be represented as having probabilty 1 of mapping to either :math:`a` or :math:`b`.


Generating samples from Bayes Nets
==================================

This procedure for sampling a trial from Bayesian networks relies heavily on using what I call the *active sample set*. This is the set of nodes for which we have well-defined distributions at the time of sampling.

There will always be at least one node in a Bayesian network that has no parents (for a given trial). We will call these nodes **orphans**. To sample a trial from the Bayesian network we begin with the orphans. 

Because orphans have no parents – in order for the Bayes net to be well-defined – each orphan will have a well-defined marginal probability distribution that we can directly sample from. Thus we start with the set of orphans as the *active sample set*. 

After sampling from all of the orphans, we will take the union of the sets of children of the orphans, and at least one of these nodes will have values sampled for all of its parents. We take the set of orphans whose entire parent-set has sampled values, and sample from the conditional distributions defined relative to their parents' sampled values and make this the *active sample set*.

After each set of samples from the *active sample set* we will either have new variables whose distributions are well-defined or will have sampled all of the variables in the graph for that trial. [#]_ If we have multiple trials, we repeat this procedure for each trial. 

.. [#] One potential worry is the case of disconnected graphs (i.e., graphs that can be divided into at least 2 disjoint sets of nodes where there will be no edges between nodes of different sets). However, because disconnected subgraphs of a :sc:`dag` will also be :sc:`dag`\s, we can count on at least one orphan existing for each of those graphs, and thus we will be able to sample from all disconnected subgraph by following the same algorithm above (they will just be sampled in parallel).

Causal Bayesian Networks
------------------------

Causal Bayesian networks are Bayesian networks that are given an interventional operation that allows for "graph surgery" by cutting nodes off from their parents. [#]_ The central idea is that interventions are cases where some external causal force is able to "reach in" and set the values of individual nodes, rendering intervened on independent of their parent nodes. 

.. [#] This is technically a more general definition than that given in :cite:`pearl2000` as in that case there is a specific semantic flavor given to interventions as they affect the probabilistic semantics of the variables within the network. Because here we are considering a version of intervention that affects the *structure* of a set of graphs rather than an intervention's results on a specific parameterized graph, this greater specificity is unnecessary.

NetworkX
--------

This is a framework for graphs that stores graphs as "a dict of dicts of dicts".

Basic NetworkX operations
=========================

NetworkX is usually imported using the :code:`nx` abbreviation

..  code-block:: python
    
    import networkx as nx  

    G = nx.DiGraph() # initialize a directed graph

    edge_list = G.edges() # returns a list of edges
    edge_data_list = G.edges(data=True) 
    # returns list of edges as tuples with data dictionary 

    node_list = G.nodes() # returns a list of nodes
    node_data_list = G.nodes(data=True) 
    # returns list of nodes as tuples with data dictionary



Causal Bayesian NetworkX: Graphs
--------------------------------

Other packages
==============

In addition to networkX, we need to import numpy and itertools.

..  code-block::python

    import numpy as np
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

.. [#] Note that powerset will need to be built (see `Causal Bayesian NetworkX`_ for details).

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

    def create_is_dag_condition(node_list):
        """ Returns a function that returns true 
        if graph is a dag."""
        def is_dag_condition(G):
            return nx.is_directed_acyclic_graph(G)
        return is_dag_condition

Non-destructive conditional subgraph generators
===============================================

Because the :code:`conditionalSubgraph` generator produces an iterable, if we want to apply a conditional after that intiial set is generated, we need to split it into two copies of the iterable. This involves the :code:`tee` function from the :code:`itertools` core package.

.. code-block:: python

    def new_conditional_graph_set(graph_set,condition_list):
        """
        This returns the old graph_set and a new iterator
        which has with conditions in condition_list applied to it.
        
        Warning: This function will devour the iterator 
        you include as the `graph_set` input, 
        you need to redeclare the variable as 
        one of the return values of the function.
        
        Thus a correct use would be:    
        a,b = new_conditional_graph_set(a,c)
        
        The following would not be a correct use:
        x,y = new_conditional_graph_set(a,c)
        
        Variables: 
        graph_set is a graph-set generator
        condition_list is a list of first order functions returning boolean values when passed a graph.
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

Naming conventions for filters and conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convention I have been following for distinguishing filter and condition functions is that the higher-order function in the case of filters beginning with the word :code:`extract_`, and then both the returned function and the higher-order function ending with the word :code:`filter`. Similarly, conditions have begun with :code:`create_` and finished with :code:`condition`.

..  code-block:: python

    def extract_name_filter(node_list):
        """
        """
        def name_filter(G):
            graph = G.copy()
            # operations removing edges
            return graph
        return name_filter


..  code-block:: python

    def create_name_condition(node_list):
        """
        """
        def name_condition(G):
            # operations checking for whether conditions hold
            return # truth value
        return name_condition

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

Causal Bayesian NetworkX: Sampling
----------------------------------


Causal Theories and Computational Cognitive Science
---------------------------------------------------

**Causal theories** is a formal framework that arose out of the tradition in computational cognitive science to approach problems with rational, comptuational-level analyses :cite:`griffithst09`. In particular, causal theories form generative models for defining classes of parameterized probabilistic graphical models. They rely on defining a set of classes of entities (ontology), potential relationships between those classes of entities and particular entities (plausible relations), and particular parameterizations of how those relations manifest in observable data (or in how other relations eventually ground out into observable data). This allows Griffiths and Tenenbaum to subsume the prediction of a wide array of human causal inductive, learning and reasoning behavior using this framework for generating graphical models and doing inference over the structures they generate.

Rational analysis
=================

A technique used that allows us to model not cognition per se, but the situation into which cognitive capacities are to be placed. If we assume that we know the inputs, the outputs and the goal state of the arbtirary cognitive agent, we can iteratively predict the agent's behavior[#]_.

This is often coupled with comptuational-level analysis inspired by Marr's :cite:`marr82` level's of analysis.  

.. [#] This is not a well-sourced definition. I need to go back to :cite:`andersons91` to spruce it up.

Computational-Level Analysis of Human Cognition
===============================================

A computational-level analysis is one in which we model a system in terms of its functional role(s) and how they would be optimally solved. This is distinguished from algorithmic-level analysis by not caring how this goal achievement state is implemented in terms of the formal structure of the underlying system and from mechanistic-level analysis by not caring about the physical structure of how these systems are implemented (which may vary widely while still meeting the structure of the algorithmic-level which itself accomplishes the goals of the computational level).

A classic example of the three-levels of analysis are different ways of studying flying with the example of bird-flight. The mechanistic-level analysis would be to study feathers, cells and so on to understand the component subparts of individual birds. The algorithmic-level analysis would look at how these subparts fit together to form an active whole that is capable of flying often by flapping its wings in a particular way. The computational-level analysis would be a theory of aerodynamics with specific accounts for the way forces interact to produce flight through the particular motions of flying observed in the birds.

Causal theories: ontology, plausible relations, functional form
===============================================================

Griffiths and Tenenbaum :cite:`griffithst09` point out their framework generalizes the notion of specifying a Bayesian network in the same way first order logic generalizes propositional logic. It does so by requiring the elements necessary to populate a graph with nodes, those nodes with properties, and relations between the nodes, stating which of those relations are plausible(and how plausible), and a specific, precise formulation for how those relations manifest in terms of the semantics. In the terms of :cite:`griffithst09`'s theory-based causal induction, this requires specifying an ontology, plausible relations over those ontologies, and functional forms for parameterizing those relations.

Ontology
^^^^^^^^

This specifies the full space of potential kinds of entities, properties and relations that exist. This is the basis around which everything else will be defined. 

Note that it is easy enough to populate nodes with features using the data field in NetworkX.

Plausible Relations
^^^^^^^^^^^^^^^^^^^

This specifies which of the total set of relations allowed by the ontology are plausible. For example, we know that in most situations a fan is more likely than a tuning fork to blow out a candle. 

As mentioned above, once you have a well-populated world if you do not dramatically restrict the sets of relations you consider, there will be an explosion of possibilities. People, even young children:cite:`griffithst09`, have many expectations about what sorts of things can can feasibly be causally related to one another. This sometimes has been interpreted as the plausible existence of a 

Functional form
^^^^^^^^^^^^^^^

> Even in the most basic cases of causal induction we draw on expectations as to whether the effects of one variable on another are positive or negative, whether multiple causes interact or are independent, and what type of events (binary, continuous, or rates) are relevant to evaluating causal relationships.
:cite:`griffithst09`


Generalizations to other kinds of logical/graphical conditions
==============================================================

The Griffiths and Tenenbaum framework is richer than the examples they develop in :cite:`griffithst09`. We can express conditions of graphical connectivity, alternative functional forms, substructures of constrained plausible relations, among many others.

Because the plausible relations are in general described as sufficiency statements, the idea is that most relations are not plausible. However, we can also make necessary statements about the kinds of relations that must be there. And in general one can see this as selecting a subset of all the possible graphs implementable by the set of nodes defined by the ontology.

Part of the aim of developing `Causal Bayesian NetworkX`_ is to provide a programming framework in which the richness of causal theories are able to be expressed. Because of the utilities in :code:`networkX`, with the enumerating, filtring and conditioning functions described above, it becomes much easier to implement higher-order graphical conditions (e.g., a directed path necessarily existing between two nodes) than in the original notation described in the framework. These ideas were entirely expressable in the original mathematical framework, but would have required a good deal more notational infrastructure to represent. Here, we not only provide a notation, but a computational infrastructure for applying these kinds of conditions.

Uses in modeling human cognition
================================

Using this framework, Griffiths and Tenenbaum were able to provide comprehensive coverage for a number of human psychology experiments. To avoid further overpopulation of the references section, I direct the interested reader to the `original paper`_ (which is well worth reading in its own right).

What is important is that they successfully modeled humans using this framework by treating people as optimal performers[#]_ within the problem defined by their framework. Furthermore, by examining different but related experiments, they were able to demonstrate the different ways in which specific kinds of prior knowledge are called upon differentially to inform human causal induction resulting in quite different inferences on a rational statistical basis.

.. [#] Optimality in these cases is taken to mean on average approximating the posterior distribution of some inference problem defined by the authors in each case.

.. _original paper: https://cocosci.berkeley.edu/tom/papers/tbci.pdf


.. Of course, no paper would be complete without some source code.  Without
.. highlighting, it would look like this::

..     def sum(a, b):
..         """Sum two numbers."""

..         return a + b

.. With code-highlighting:

.. .. code-block:: python

..     def sum(a, b):
..         """Sum two numbers."""

..         return a + b

.. Maybe also in another language, and with line numbers:

.. .. code-block:: c
..     :linenos:

..     int main() {
..         for (int i = 0; i < 10; i++) {
..             /* do something */
..         }
..         return 0;
..     }

.. Or a snippet from the above code, starting at the correct line number:

.. .. code-block:: c
..     :linenos:
..     :linenostart: 2

..     for (int i = 0; i < 10; i++) {
..         /* do something */
..     }


.. Important Part
.. --------------

.. .. It is well known [Atr03]_ that Spice grows on the planet Dune.  Test

.. some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
.. equation on a separate line:

.. .. math::

..     g(x) = \int_0^\infty f(x) dx

.. or on multiple, aligned lines:

.. .. math::
..     :type: eqnarray

..     g(x) &=& \int_0^\infty f(x) dx \\
..          &=& \ldots

.. The area of a circle and volume of a sphere are given as

.. .. math::
..     :label: circarea

..     A(r) = \pi r^2.

.. .. math::
..     :label: spherevol

..     V(r) = \frac{4}{3} \pi r^3

.. We can then refer back to Equation (:ref:`circarea`) or
.. (:ref:`spherevol`) later.


.. .. figure:: figure0.png

..     This is the caption. :label:`egfig`

.. .. figure:: figure0.png
..     :align: center
..     :figclass: w

..     This is a wide figure, specified by adding "w" to the figclass.  It is also
..     center aligned, by setting the align keyword (can be left, right or center).

.. .. figure:: figure0.png
..     :scale: 20%
..     :figclass: bht

..     This is the caption on a smaller figure that will be placed by default at the
..     bottom of the page, and failing that it will be placed inline or at the top.
..     Note that for now, scale is relative to a completely arbitrary original
..     reference size which might be the original size of your image - you probably
..     have to play with it. :label:`egfig2`

.. As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered figures.

.. .. table:: This is the caption for the materials table. :label:`mtable`

..     +------------+----------------+
..     | Material   | Units          |
..     +============+================+
..     | Stone      | 3              |
..     +------------+----------------+
..     | Water      | 12             |
..     +------------+----------------+
..     | Cement     | :math:`\alpha` |
..     +------------+----------------+


.. We show the different quantities of materials required in Table
.. :ref:`mtable`.


.. .. The statement below shows how to adjust the width of a table.

.. .. raw:: latex

..     \setlength{\tablewidth}{0.8\linewidth}


.. .. table:: This is the caption for the wide table.
..     :class: w

..     +--------+----+------+------+------+------+--------+
..     | This   | is |  a   | very | very | wide | table  |
..     +--------+----+------+------+------+------+--------+

.. Unfortunately, restructuredtext can be picky about tables, so if it simply
.. won't work try raw LaTeX:


.. .. raw:: latex

..     \begin{table*}

..       \begin{longtable*}{|l|r|r|r|}
..       \hline
..       \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
..       \cline{2-4}
..        & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
..       \hline
..       Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
..       \hline
..       Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
..       \hline
..       Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
..       \hline
..       Percent Difference & 44\% & 41\% & 43\%\tabularnewline
..       \hline
..       \end{longtable*}

..       \caption{Area Comparisons \DUrole{label}{quanitities-table}}

..     \end{table*}

.. Perhaps we want to end off with a quote by Lao Tse [#]_:

..     *Muddy water, let stand, becomes clear.*

.. .. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.



Outlines
========

Test :cite:`mckay2003acyclic,winn2012causality`


Outline v. 1.1
==============

1. Introduction

   2. Why?
   3. What?
   4. Background recommended

      5. Basic probability

         6. Sum of prob of exclusive events = 1

      6. Basic graph theory ✓

         7.  Nodes (N) and Edges (V = (N × N))✓
             8.  notation notes ✓
             9.  Parents and children

         10.  Adjacency Matrix view of graphs✓
         11.  Directed and Undirected graphs✓
         12.  Directed Acyclic Graphs✓

2. Assumptions

   2. Fixed set of nodes ✓
   3. Discrete trials 
   4. Synchronous activation 
   5. cross trial independence 

3. Graphs: Structure

   1. Complexity of graph enumeration

      2. General directed graphs, ✓

         .. math:: 2^{n^2} 

   2. Reducing complexity:

      3. Enumeration filters
      4. Directed Acyclic Graphs

         4. No trace (no self-loops) ✓
         5. number of graphs

   3. Parents and children

4. Random Variables: Semantics, sampling and graphs

   4. Conditional probability distributions
   5. Conditional independence properties

5. Bayesian Networks.
    
    1. Graphical interpretation of conditional independence ✓


5. Causal Graphs: Interventions

   1. Graph Surgery
   2. Causal graphs as extensions of directed graphs ---

      1. incorporating intervention into the node set

   3. Interventions as constraints on the graph set

      4. Node has no parents = node is intervened on with prior
         distribution equal to the

1. NetworkX
    
    2. graph/network package in python
    

6. Causal Bayesian NetworkX: Graphs

   5. Iterator over graphs
   6. Closures for constraints
      
       8. over graphs
       9. tuples of nodes
       10. individual nodes?
   
   11. Zipping iterators and avoiding early consumption

6. Causal Bayesian NetworkX: Probabilistic Sampling
    
.. 7. Gates and causal networks

8. Causal theories
    
    9. Rational analysis and computational level explanations of human cognition✓
    10. First order logic for probabilistic graphical models ✓
    11. ontology, plausible relations, functional form✓
    12. generalizations to other kinds of logical/graphical conditions✓
    13. uses in understanding human cognition


.. .. raw:: latex

..     \bibliographystyle{IEEEtran}
..     \begingroup
..     \renewcommand{\section}[2]{}%
..     %\renewcommand{\chapter}[2]{}% for other classes
..     \bibliography{uber}
..     \endgroup


.. .. raw:: latex

..     \bibliographystyle{IEEEtran}
..     \providecommand*\DUrolebibliography[1]{\bibliography{#1}}

.. .. role:: bibliography

.. .. [Atr03] P. Atreides. *How to catch a sandworm*,           Transactions on Terraforming, 21(3):261-300, August 2003.
