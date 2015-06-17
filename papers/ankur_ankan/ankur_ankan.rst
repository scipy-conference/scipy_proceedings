:author: Ankur Ankan
:email: ankurankan@gmail.com

:author: Abinash Panda
:email: mailme.abinashpanda@gmail.com

--------------------------------------------------
pgmpy: Probabilistic Graphical Models using Python
--------------------------------------------------

.. class:: abstract

   Probabilistic Graphical Models (PGM) is a technique of compactly representing   
   a joint distribution by exploiting dependencies between the random variables.     
   It also allows us to do inference on joint distributions in a computationally 
   cheaper way than the traditional methods. PGMs are widely used in the field 
   of speech recognition, information extraction, image segmentation, modelling 
   gene regulatory networks. 
   
   pgmpy is a python library for working with graphical models. It allows the 
   user to create their own models and answer inference or map queries over 
   them. pgmpy has implementation of many inference algorithms like 
   VariableElimination, Belief Propagation etc.

   This paper first gives a short introduction to PGMs and various other python
   packages available for working with PGMs. Then we discuss about creating and
   doing inference over Bayesian Networks and Markov Networks using pgmpy.

.. class:: keywords

   Graphical Models, Bayesian Networks, Markov Networks, Variable Elimination

Introduction
------------

Probabilistic Graphical Models is a technique of representing Joint
Distributions over random variables in a compact way by exploiting the 
dependencies between the random variables. PGMs use a network structure and some 
parameters to represent the joint distribution. The network structure is used to
encode the relationships between the random variables. There are mainly two 
types of Graphical Models: Bayesian Networks and Markov Networks.


.. figure:: figure1.png
   :align: center
   :figclass: w
   
   A simple Bayesian Network. :label:'bayesian'

Bayesian Network: A Bayesian Network consists of a directed graph and a 
conditional distribution associated with each of the random variables. A 
Bayesian network is used mostly when there is a causal relationship between the
random variables. An example of a Bayesian Network representing a student taking 
some course is shown in Fig :ref:'bayesian'.

.. figure:: figure2.png
   :align: center
   :figclass: w

   A simple Markov Model :label:'markov'

Markov Network: A Markov Network consists of an undirected graph and a few 
factors are associated with it. Unlike in the case of Bayesian network, a factor
does not represent the probabilities of variables in the network. Rather it represents 
how much is a state of a random variable likely to agree to the state 
of the other random variable. An example of 4 friends A, B, C, D agreeing to
some concept is shown in Fig :ref:'markov'.

There are numerous packages available in Python for working with graphical 
models but most of them are focused towards some special case and doesn't 
give complete freedom to the user. Give some examples of some packages.
pgmpy tries to be a complete package for working with graphical models and gives 
the user full control on designing the model. Also pgmpy provides easy extensibility and 
the user can write his own inference algorithms or elimination orders without actually 
looking at the source code.
 
Important Part
--------------

It is well known [Atr03]_ that Spice grows on the planet Dune.  Test
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

.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +============+================+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+

Unfortunately, restructuredtext can be picky about tables, so if it simply
won't work try raw LaTeX:


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|r|r|r|}
     \hline
     \multirow{2}{*}{Projection} & \multicolumn{3}{c|}{Area in square miles}\tabularnewline
     \cline{2-4}
      & Large Horizontal Area & Large Vertical Area & Smaller Square Area\tabularnewline
     \hline
     Albers Equal Area  & 7,498.7 & 10,847.3 & 35.8\tabularnewline
     \hline
     Web Mercator & 13,410.0 & 18,271.4 & 63.0\tabularnewline
     \hline
     Difference & 5,911.3 & 7,424.1 & 27.2\tabularnewline
     \hline
     Percent Difference & 44\% & 41\% & 43\%\tabularnewline
     \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

Perhaps we want to end off with a quote by Lao Tse [#]_:

  *Muddy water, let stand, becomes clear.*

.. [#] :math:`\mathrm{e^{-i\pi}}`

.. Customised LaTeX packages
.. -------------------------

.. Please avoid using this feature, unless agreed upon with the
.. proceedings editors.

.. ::

..   .. latex::
..      :usepackage: somepackage

..      Some custom LaTeX source here.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


