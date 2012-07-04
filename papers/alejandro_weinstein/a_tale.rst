:author: Alejandro Weinstein
:email: aweinste@mines.edu
:institution: the Colorado School of Mines


:author: Michael Wakin
:email: mwakin@mines.edu
:institution: the Colorado School of Mines



------------------------------------------------
A Tale of Four Libraries
------------------------------------------------

.. class:: abstract 

   A short version of the long version that is way too long to be written as a
   short version anyway.  Still, when considering the facts from first
   principles, we find that the outcomes of this introspective approach is
   compatible with the guidelines previously established.

   In such an experiment, it is then clearl that the potential for further
   development not only depends on previous relationships found but also on
   connections made during exploitation of this novel new experimental
   protocol.

.. class:: keywords

   reinforcement learning, latent semantic analysis, machine learning

Introduction
------------

In addition to bringing efficient array computing and standard mathematical
tools to Python, the NumPy/SciPy libraries provide an ecosystem where multiple
libraries can coexist and interact. This work describes a success story where
we integrate several libraries, developed by different groups, to solve some of
our research problems.

Our research focuses on using Reinforcement Learning (RL) to gather information
in domains described by an underlying linked dataset. For instance, we are
interested in problems such as the following: given a Wikipedia article as a
seed, finding other articles that are interesting relative to the starting
point. Of particular interest is to find articles that are more than one-click
away from the seed, since these articles are in general harder to find by a
human.

In addition to the staples of scientific Python computing NumPy, SciPy,
Matplotlib, and IPython, we use the libraries RL-Glue [Tan09]_, NetworkX
[Hag08]_, Gensim [Reh10]_, and scikit-learn [Ped11]_.

Reinforcement Learning considers the interaction between a given environment
and an agent. The objective is to design an agent able to learn a policy that
allows it to maximize its total expected reward. We use the RL-Glue/RL-Library
libraries for our RL experiments. These libraries provide the infrastructure to
connect an environment and an agent, each one described by an independent
Python program.

We represent the linked datasets we work with as graphs. For this we use
NetworkX, which provides data structures to efficiently represent graphs
together with implementations of many classic graph algorithms. We use NetworkX
graphs to describe the environments implemented in RL-Glue/RL-Library. We also
use these graphs to create, analyze and visualize graphs built from
unstructured data.

One of the contributions of our research is the idea of representing the items
in the datasets as vectors belonging to a linear space. To this end, we build a
Latent Semantic Analysis (LSA) model to project documents onto a vector
space. This allows us, in addition to being able to compute similarities
between documents, to leverage a variety of RL techniques that require a vector
representation. We use the Gensim library to build the LSA model. This library
provides all the machinery to build, among other options, the LSA model. One
place where Gensim shines is in its capability to handle big data sets, like
the entire Wikipedia, that do not fit in memory. We also combine the vector
representation of the items as a property of the NetworkX nodes.

Finally, we also use the manifold learning capabilities of sckit-learn, like
the ISOMAP algorithm, to perform some exploratory data analysis. By reducing
the dimensionality of the LSA vectors obtained using Gensim from 400 to 3, we
are able to visualize the relative position of the vectors together with their
connections.

Reinforcement Learning and the Information Gathering Problem
------------------------------------------------------------


The RL paradigm [Sut98]_ considers an agent that interacts with an environment
described by a Markov Decision Process (MDP). Formally, an MDP is defined by a
state space :math:`\mathcal{X}`, an action space :math:`\mathcal{A}`, a
transition probability function :math:`P`, and a reward function :math:`r`. At
a given sample time, the agent is at state :math:`x \in \mathcal{X}`, and it
chooses action :math:`a \in \mathcal{A}`. Given the current state and selected
action, the probability that the next state is :math:`x'` is determined by
:math:`P(x,a,x')`. After reaching the next state :math:`x'`, the agent observe
an immediate reward :math:`r(x')`. Figure :ref:`figRL` depicts the
agent-environment interaction. In a RL problem, the objective is to find a
function :math:`\pi:\mathcal{X} \mapsto \mathcal{A}`, called the *policy*, that
maximizes the total expected reward

.. math::

   R = \mathbf{E}\left[\sum_{k=0}^\infty \gamma^k r(x_k) \right],

where :math:`\gamma \in (0,1)` is the discount factor. Note that typically the
agent does not known the functions :math:`P` and :math:`r`, an it must find the
optimal policy by interacting with the environment. See [Sze10]_ for a detailed
review of the different algorithms used in RL.

.. figure:: RL_scheme.pdf

   The agent-environment interaction. The agent observes the current state
   :math:`x` and reward :math:`r`; then it executes action
   :math:`\pi(x)=a`. :label:`figRL`

We implement the RL algorithms using the RL-Glue library. The library consists
on the *RL-Glue Core* program and a set of codecs for different laguages [#]_
to communicate with the library. To run an instance of a RL problem one needs
to write three different programs: the *environment*, the *agent*, and the
*experiment*. The environment and the agent programs match exactly the
corresponding functionalities of the RL framework, while the experiment
orchestrates the interaction between these two. The following code snippets
show the main methods that these three programs must implement:

.. code-block:: python

   ################# environment.py #################
   class env(Environment):
       def env_start(self):
           # Set the current state

           return current_state

       def env_step(self, action):
           # Change the current state according to 
           # the current state and given action.

           return reward 

    #################### agent.py ####################
    class agent(Agent):
        def agent_start(self, state):
            # First step during an experiment
            
            return action
            
        def agent_step(self, reward, obs):
            # Execute a step of the RL algorithm
            
            return action

    ################# experiment.py ##################
    RLGlue.init()
    RLGlue.RL_start() 
    RLGlue.RL_episode(100) # Run an episode

    

Note that RL-Glue is a only a thin layer among these programs, allowing us to
use any construction inside them. In particular, as described in the next
section, we use a NetworkX graph to model the environment.


.. [#] Currently there are codecs for Python, C/C++, Java, Lisp, MATLAB, and
       Go.



.. Although there are other alternatives for writting RL programs, in our
   opinion RL-Glue is the best alternative because it is very "thin", it match
   the RL paradigm and allows to mix agents and environments written in diffent
   languages.


Computing the Similarity between Documents
------------------------------------------

Although in principle it is simple to compute the LSA model of a given corpus,
the size of the datasets we are interested on make doing this a significant
challenge. The two main difficulties are that in general (i) we cannot hold the
vector representation of the corpus in RAM memory, and (ii) we need to compute
the SVD of a matrix whose size is beyond the limits of what standard solvers
can handle.


Representing the State Space as Graph
-------------------------------------

We are interested in the problem of gathering information in domains descibed
by linked datasets. It is natural to describe such domains by graphs. We use
the NetworkX library to build the graphs we work with. NetworkX provides data
structures to reperesents different kinds of graphs (undirected, weighted,
directed, etc), together with implementations of many graph
algorithms. NetworkX allows to use any hashable Python object as a node
identifier. Also, any Python object can be used as a node, egde, or graph
attribute. We exploit this capability by using the LSA vector representation of
a Wikipedia article, which is a NumPy array, as a node attribute.

The following code snippet shows a function [#]_ used to build a directed graph
where nodes represent Wikipedia articles, and the edges represent links between
articles. Note that we compute the LSA representation of the article (line 11),
and that this is vector is used as a node attribute (line 13). The function get
up to ``n_max`` articles by breath-first crawling the Wikipedia, starting from
the article defined by ``page``.

.. code-block:: python
   :linenos:

    def crawl(page, n_max):
        G = nx.DiGraph()
        n = 0
        links = [(page, -1, None)]
        while n < n_max:
            link = links.pop()
            page = link[0]
            dist = link[1] + 1
            page_text = page.edit().encode('utf-8')
            # LSI representation of page_text
            v_lsi = get_lsi(page_text)
            # Add node to the graph
            G.add_node(page.name, v=v_lsi)
            if link[2]:
                source = link[2]
                dest = page.name
                if G.has_edge(source, dest):
                    # Link already exist
                    continue
                else:
                    sim = get_similarity(page_text)
                    self.G.add_edge(source,
                                    dest,
                                    weight=sim,
                                    d=dist)
            new_links = [(l, dist, page.name) 
                         for l in page.links()]
            links = new_links + links
            n += 1

        return G

.. [#] The parameter ``page`` is a mwclient page object. See
       http://sourceforge.net/apps/mediawiki/mwclient/.

We now show the result of running the code above for two different setups. In
the first instance we crawl the *Simple English Wikipedia* [#]_ using "Army" as the
seed article. We set the limit on the number of articles to visit to 100. The
result is depicted in Fig. :ref:`figArmy`, where the node corresponding to the
seed article is in light blue and the remaining nodes have a size proportional
to the similarity with respect to the seed. Red nodes are the ones with
similarity bigger than 0.5. We observe two nodes, "Defense" and "Weapon", with
similarities 0.7 and 0.53 respectively, that are three links ahead of the seed.

In the second instance we crawl Wikipedia using the article "James Gleick" [#]_
as seed. We set the limit on the number of articles to visit to 2000. We show
the result in Fig. :ref:`figGleick`, where, as in the previous example, the
node corresponding to the seed is in light blue and the remaining nodes have a
size proportional to the similarity with respect to the seed. The eleven red
nodes are the ones with similarity bigger than 0.7. Of these, 9 are more than
one link ahead of the seed. We see that the article with the biggest
similarity, with a value of 0.8, is about "Robert Wright (journalist)", and it
is two links ahead from the seed (passing through the "Slate magazine"
article). Robert Wright writes books about sciences, history and religion. It
is very reasonable to consider him an author similar to James Gleick. It is
very reasonable to consider him an author similar to James Gleick.

..  Table \ref{tbl:gleick} shows the ten most similar articles and theirs link
    distances from the seed. We see that all of them are related to the
    seed. We claim that these results validate the thesis that there are
    similar articles separated by more than one link.

.. [#] The Simple English Wikipedia (http://simple.wikipedia.org) has articles
       written in *simple English* and has a much smaller number of articles
       than the standard Wikipedia. We use it because of it simplicity.

.. [#] James Gleick is "an American author, journalist, and biographer, whose
    books explore the cultural ramifications of science and technology".

.. figure:: army.pdf 

   Graph for the "Army" article in the simple Wikipedia with 97 nodes and 99
   edges. The seed article is in light blue. The size of the nodes (except for
   the seed node) is proportional to the similarity. In red are all the nodes
   with similarity bigger than 0.5. We found two articles ("Defense" and
   "Weapon") similar to the seed three links ahead. :label:`figArmy`

.. figure:: gleick.pdf
   
   Graph for the "James Gleick" Wikipedia article with 1975 nodes and 1999
   edges. The seed article is in light blue. The size of the nodes (except for
   the seed node) is proportional to the similarity. In red are all the nodes
   with similarity bigger than 0.7. There are several articles with high
   similarity more than one link ahead. :label:`figGleick`
            

Another place where graphs can play an important role is in the RL problem when
we want to find basis functions to appoximate the value-function. The
value-function is the function :math:`V: \mathcal{X} \mapsto \mathbb{R}` defined
as

.. math::

   V^\pi (x) = \mathbf{E}\left[\sum_{k=0}^\infty \gamma^k r(x_k) \bigm\vert 
   x_0 = x, a_k = \pi(x_k) \right],

and plays a key role in many RL algorithms [Sze10]_. When the dimension of
:math:`\mathcal{X}` is significant,




Visualizing the LSA Space
-------------------------

Figure :ref:`figISOMAP` bla bla.

.. figure:: isomap_lsa.pdf

   ISOMAP projection of the LSA space. Each point represents the LSA vecotr of
   a Simple English Wikipedia article projected onto :math:`\mathbb{R}^3` using
   ISOMAP. A line is added if there is a link between the corresponding
   articles. The figure shows a close-up around the "Water" article. We can
   observe that this point is close to points associated to articles with a
   similar semantic. :label:`figISOMAP`
 


Conclusions
-----------

This is much better than using domain specific languages like MATLAB. 
.. This is a comment


References
----------

.. [Tan09] B. Tanner and A. White. *RL-Glue: Language-Independent Software for
           Reinforcement-Learning Experiments*, Journal of Machine Learning
           Research, 10(Sep):2133--2136, 2009

.. [Hag08] A. Hagberg, D. Schult and P. Swart, *Exploring Network Structure,
           Dynamics, and Function using NetworkX*, in Proceedings of the 7th
           Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis
           Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11--15,
           Aug 2008

.. [Ped11] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, 
           O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg,
           J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot
           and E. Duchesnay. *Scikit-learn: Machine Learning in Python*,
           Journal of Machine Learning Research, 12:2825--2830, 2011


.. [Reh10] Řehůřek, R. and Sojka, P, *Software Framework for
           Topic Modelling with Large Corpora*, in Proceedings of the LREC 2010
           Workshop on New Challenges for NLP Frameworks, pp. 45--50 May 2010

.. [Sze10] ﻿Szepesvári, C. *Algorithms for Reinforcement Learning*.  San Rafael,
           CA, Morgan and Claypool Publishers, 2010.

.. [Sut98] ﻿Sutton, R. S. and Barto, A. G. *Reinforcement Learning*. Cambridge,
           Massachusetts, The MIT press, 1998.
