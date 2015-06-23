:author: Mike Pacer
:email: mpacer@berkeley.edu
:institution: University of California at Berkeley
:bibliography: myBibliography

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690


.. raw:: latex

    \newcommand{\DUrolesc}{\textsc}


.. role:: sc

------------------------
Causal Bayesian NetworkX
------------------------

.. class:: abstract

    Humans are existence proofs for the solubility of computational causal inference.

    Computational problems are sometimes thought to be the exclusive domain of computer science, though the solutions found prove vital for many other sciences. But computational cognitive science can also contribute to the solution of computational problems, particularly inductive problems. Causal inference (and inductive problems more generally) have proven resilient to traditional analysis, and the recent progress on these problems observed in machine learning (e.g., neural networks and their extensions) originate in models formulated by cognitive scientists.

    As a computational cognitive scientist, I use a technique called rational analysis, which roughly consists of developing formal models of how *any* cognitive agent might optimally solve a computational problem and comparing that to how people actually solve analogous problems. In the course of doing so we find that people turn out to make much more sense than popular psychology might lead you to believe. At the same time, we create formal frameworks that represent entities and relations in the world as well as reasoning processes over those representations. 

    One of the frameworks successfully used in this way are causal Bayesian networks. Bayesian networks fall within the more general class of probabilistic graphical models, specifically, directed acyclic graphs with associated conditional probability distributions. Directed arrows encode direct dependency relationships going from parents to their children, whose conditional probability distribution is defined in terms of the parents' values. *Causal* Bayesian networks are endowed with an intervention operation that allows "graph surgery" in which one cuts variables off from their parents (usually setting it to a particular value). 

    I have developed tools on top of the :code:`NetworkX` package that allow me to implement some aspects of these models. By treating graph definition as one of enumeration and filtering rather than investigating invidual graphs, we are able to more conveniently state constraints on the graph structures under question. Indeed this gives an alternative view of intervention not as the modification of an single graph, but as a constraint on the total set of graphs. This allows us to treat the graphical aspects of the problem separately from the probabilistic semantics that define particular models on those graphs. I call this set of tools `Causal Bayesian NetworkX`.


.. class:: keywords

   probabilistic graphical models, causality, intervention

Introduction and Aims
---------------------

My first goal in this paper is to provide enough of an introduction to some formal/mathematical tools such that those familiar with :code:`python` and programming more generally will be able to appreciate both why and how one might implement causal Bayesian networks. In particular, I have developed parts of a toolkit that allows the creation of these models on top of the :code:`NetworkX` python package. Given the coincidence of the names, it seemed most apt to refer to this toolkit as :code:`Causal Bayesian NetworkX` (the current implementation of which can be found at `Causal Bayesian NetworkX`_). 

In these tools, I focus first on establishing a means of building iterators over sets of directed graphs. I then apply operations to those sets. Beginning with the complete directed graph, we enumarte over the subgraphs of that complete graph and enforce graph theoretic conditions such as acyclicity over the entire graph, guarantees on paths between nodes that are known to be able to communicate with one another, or orphan-hood for individual nodes known to have no parents. We accomplish this by using closures that take graphs as their input along with any explicitly defined arguments needed to define the exact desired conditions. 

I then shift focus to a case where there is a known graph over a set of nodes that are imbued with a simple probabilistic semantics, also known as a Bayesian network. I demonstrate how to sample independent trials from these variables in a way consistent with these semantics.

Then, I will briefly discuss **gates**:cite:`winn2012causality`, an extension to graphical modeling frameworks that allow one to define context-specific dependence relations (which includes context-specific *independence* relations). This extension is of particular interest as it allows us to subsume the classical :code:`do`-calculus :cite:`pearl2000` into the more general semantics of the probabilistic network. This work was a key influence in the development of thinking about interventions not as operations on individual nodes, or even individual graphs, but as a particular constraint placed on sets of graphs by some generative process.

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

Defined in this way, edges are all *directed* in the sense that an edge from :math:`X_1 \textrm{to} X_2`is not the same as the edge from :math:`X_2 \textrm{to} X_1`, or :math:`(X_1,X_2) \neq (X_2,X_1)`. An edge :math:`(X_1,X_2)` will sometimes be written as :math:`X_1 \rightarrow X_2`, and the relation may be described using language like ":math:`X_1` is the parent of :math:`X_2`" or ":math:`X_2` is the child of :math:`X_1`".

Directed paths
^^^^^^^^^^^^^^

Paths are a useful way to understand sequences of edges and the structure of a graph. Informally, to say there is a path between :math:`X_i` and :math:`X_j` is to say that one can start at :math:`X_i` and by traveling from parent to child along the edges leading out from the node that you are currently at, you can eventually reach :math:`X_j`.

To define it recursively and more precisely, if the edge :math:`(X_i,X_j)` is in the edge set or if the edges :math:`(X_i,X_k)` and :math:`(X_k,X_j)` are in the edge set there is a path from :math:`X_i` to :math:`X_j`. If a graph has a path from node :math:`X_i` to :math:`X_j`, that means that there is a subset of its set of edges such that the set contains edges :math:`(X_i,X_k)` and :math:`(X_l,X_j)` and if :math:`\big(X_k\neq X_j` and :math: `X_l\neq X_i` and :math:`X_k \neq X_l\big)`, there is a path from :math:`X_k` to :math:`X_l`. 


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

A cycle in a directed graph can be understood as the existance of a set of edges 

The number of directed acyclic graphs (:sc:`dag`\s) that can be obtained from a set of nodes of size :math:`n` can be defined recursively as follows :cite:`mckay2003acyclic` :

.. math::

    R_n = \sum_{k=1}^{n} (-1)^{k+1} {\binom{n}{k}} 2^{k(n-k)} R_{n-k}

Note, because :sc:`dag`\s do not allow any cycles, this means that there can be no self loops. As a result, every value on the diagonal of a  :sc:`dag`\'s adjacency matrix will be 0. 

.. Topological ordering in :sc:`dag`\s
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. It is possible to reorder 


Conditional Probability Distributions
-------------------------------------

A random variable defined by a conditional probability distribution has a distribution indexed by the realization of some other variable (which itself is often a random variable, especially in the context of Bayesian networks).

Notation
========

The probability of a discrete random variable(:math:`X`) will be designated with :math:`P(X)` 


Example - Coins and dice
========================

Imagine the following game: 

You have a coin [#]_ (*C*, :sc:`Heads, Tails`), a 6-sided die(:math:`D_6, \{1,2,\ldots,6\}`), and a 20-sided die(:math:`D_20, \{1,2,\ldots,20\}`). If for simplicity, you prefer to think of these as fair dice and a fair coin, you are welcome to do so, but my notation will not require that.

.. [#] A coin is effectively 2-sided die, but for clarity of exposition I chose to use treat the conditioned-on variable as a different kind of object than the variables relying on that conditioning.

The rules of the game are as follows: flip the coin, and if it lands on :sc:`Heads`, then you roll the 6-sided die to find your score for the round. If instead your coin lands on :sc:`Tails` your score comes from a roll of the 20-sided die. Your score for one round of the game is the value of the die that you roll, and you will only roll one die in each round. 

Suppose we wanted to know your expected score on a single round, but we do not know whether the coin will land on :sc:`Heads` or :sc:`Tails`. We cannot directly compute the probabilities of each 

But this discussion hides an important complexity by having the event set of the :math:`D_6` embedded within the event set of the :math:`D_20`. Moreover, we assumed that we could treat each event in these sets as belonging to the integers and as a result, that with little interpretation, they can be easily summed and divided into.

Example - Coins and dice with labeled ids
=========================================

Imagine the following game: 

You have a coin [#]_ (*C*, :sc:`Heads, Tails`), a *new* 6-sided die(:math:`D_6, \{\clubsuit,\diamondsuit,\heartsuit,\spadesuit,\odot,\dagger\}`), and a 20-sided die(:math:`D_20, \{X_1,X_2,\ldots,X_20\}`). 

The rules are the same as before: your score for one round of the game is the value of the die that you roll, and you will only roll one die in each round. You flip the coin, and if it lands on :sc:`Heads`, then you roll the 6-sided die to find your score for the round. If instead your coin lands on :sc:`Tails` your score comes from a roll of the 20-sided die.

But note that now we cannot sum over these in the same way. Indeed, our event sets for the two dice are actually only partially 


Bayesian Networks
-----------------

Bayesian networks are a class of graphical models that have particular probabilistic semantics attached to the 

Assumptions for Bayesian networks
========================================

There is a fixed set of known nodes with finite cardinality :math:`N`. All events are presumed to occur simultaneously within a single discrete trial. Graph forms a :sc:`dag`\. 


Sampling from Conditional Probability distributions in Bayes Nets
=================================================================


Causal Bayesian Networks
------------------------


Twelve hundred years ago  — – -- in a galaxy just across the hill...

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sapien
tortor, bibendum et pretium molestie, dapibus ac ante. Nam odio orci, interdum
sit amet placerat non, molestie sed dui. Pellentesque eu quam ac mauris
tristique sodales. Fusce sodales laoreet nulla, id pellentesque risus convallis
eget. Nam id ante gravida justo eleifend semper vel ut nisi. Phasellus
adipiscing risus quis dui facilisis fermentum. Duis quis sodales neque. Aliquam
ut tellus dolor. Etiam ac elit nec risus lobortis tempus id nec erat. Morbi eu
purus enim. Integer et velit vitae arcu interdum aliquet at eget purus. Integer
quis nisi neque. Morbi ac odio et leo dignissim sodales. Pellentesque nec nibh
nulla. Donec faucibus purus leo. Nullam vel lorem eget enim blandit ultrices.
Ut urna lacus, scelerisque nec pellentesque quis, laoreet eu magna. Quisque ac
justo vitae odio tincidunt tempus at vitae tortor.

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

    def sum(a, b):
        """Sum two numbers."""

        return a + b

With code-highlighting:

.. code-block:: python

    def sum(a, b):
        """Sum two numbers."""

        return a + b

Maybe also in another language, and with line numbers:

.. code-block:: c
    :linenos:

    int main() {
        for (int i = 0; i < 10; i++) {
            /* do something */
        }
        return 0;
    }

Or a snippet from the above code, starting at the correct line number:

.. code-block:: c
    :linenos:
    :linenostart: 2

    for (int i = 0; i < 10; i++) {
        /* do something */
    }


Important Part
--------------

.. It is well known [Atr03]_ that Spice grows on the planet Dune.  Test

some maths, for example :math:`e^{\pi i} + 3 \delta`.  Or maybe an
equation on a separate line:

.. math::

    g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
    :type: eqnarray

    g(x) &=& \int_0^\infty f(x) dx \\
         &=& \ldots

The area of a circle and volume of a sphere are given as

.. math::
    :label: circarea

    A(r) = \pi r^2.

.. math::
    :label: spherevol

    V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.

Mauris purus enim, volutpat non dapibus et, gravida sit amet sapien. In at
consectetur lacus. Praesent orci nulla, blandit eu egestas nec, facilisis vel
lacus. Fusce non ante vitae justo faucibus facilisis. Nam venenatis lacinia
turpis. Donec eu ultrices mauris. Ut pulvinar viverra rhoncus. Vivamus
adipiscing faucibus ligula, in porta orci vehicula in. Suspendisse quis augue
arcu, sit amet accumsan diam. Vestibulum lacinia luctus dui. Aliquam odio arcu,
faucibus non laoreet ac, condimentum eu quam. Quisque et nunc non diam
consequat iaculis ut quis leo. Integer suscipit accumsan ligula. Sed nec eros a
orci aliquam dictum sed ac felis. Suspendisse sit amet dui ut ligula iaculis
sollicitudin vel id velit. Pellentesque hendrerit sapien ac ante facilisis
lacinia. Nunc sit amet sem sem. In tellus metus, elementum vitae tincidunt ac,
volutpat sit amet mauris. Maecenas [#]_ diam turpis, placerat [#]_ at adipiscing ac,
pulvinar id metus.

.. [#] On the one hand, a footnote.
.. [#] On the other hand, another footnote.

.. figure:: figure0.png

    This is the caption. :label:`egfig`

.. figure:: figure0.png
    :align: center
    :figclass: w

    This is a wide figure, specified by adding "w" to the figclass.  It is also
    center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure0.png
    :scale: 20%
    :figclass: bht

    This is the caption on a smaller figure that will be placed by default at the
    bottom of the page, and failing that it will be placed inline or at the top.
    Note that for now, scale is relative to a completely arbitrary original
    reference size which might be the original size of your image - you probably
    have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

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
..     .. raw:: latex

..      :usepackage: somepackage

..      Some custom LaTeX source here.

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
   6. Graphical interpretation of conditional independence

5. Causal Graphs: Interventions

   1. Graph Surgery
   2. Causal graphs as extensions of directed graphs ---

      1. incorporating intervention into the node set

   3. Interventions as constraints on the graph set

      4. Node has no parents = node is intervened on with prior
         distribution equal to the

1. NetworkX
    
    2. graph package in python
    

6. Causal Bayesian NetworkX: Graphs

   5. Iterator over graphs
   6. Closures for constraints
      
       8. over graphs
       9. tuples of nodes
       10. individual nodes
   
   11. Zipping iterators and avoiding early consumption

6. Causal Bayesian NetworkX: Probabilistic Sampling
    
7. Gates and causal networks
8. Causal theories
    
    9. Rational analysis and computational level explanations of human cognition
    10. First order logic for probabilistic graphical models 
    11. ontology, plausible relations, functional form
    12. generalizations to other kinds of logical/graphical conditions
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
