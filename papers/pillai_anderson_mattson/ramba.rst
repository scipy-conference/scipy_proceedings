:author: Padmanabhan Pillai
:email: padmanabhan.s.pillai@intel.com
:institution: Intel Corp.
:equal-contributor:

:author: Todd Anderson
:email: todd.a.anderson@intel.com
:institution: Intel Corp.
:equal-contributor:

:author: Tim Mattson
:email: timothy.g.mattson@intel.com
:institution: Intel Corp.
:corresponding:

:bibliography: ramb

------------------------------------------------
Ramba: Parallel distributed execution of NumPy code
------------------------------------------------

.. class:: abstract

Our quest is to deliver C-like performance for Python applications across a cluster of machines.  
We do this by mapping NumPy arrays onto a distributed system while leveraging the Numba 
JIT compiler. We call this package Ramba.  In this talk we describe how Ramba takes a 
programmer's NumPy code and generates a distributed execution that would make a 
C programmer proud.  Today, Ramba is a stable "proof of concept" system.  
It still needs some work, but it's ready for people to use and, more importantly, to
 join with us to help it grow into a production-ready tool.  

.. class:: keywords

   parallel distributed numpy  

Introduction
------------

Hardware is parallel. Most programmers, however, write serial code. We can train programmers to 
write parallel code.  It would be easier, though, if systems extracted parallelism from code on behalf 
of programmers.   Automatic parallelism in general is unlikely to work any time soon.  However, 
when the elements of an algorithm are "naturally concurrent", that concurrency can be exploited 
to generate parallel executions. The classic example of natural concurrency is array-based 
operations given that they are fundamentally data-parallel.

Programmers comfortable with explicit parallelism are well supported by existing tools such 
as mpi4py, Ray, PyOMP or Dask.  The overwhelming majority of programmers, however, 
are focused on their problem domains and do not have the time (or interest) to master 
parallel programming.  These are the people for whom Ramba was created.  In this paper, 
we discuss our system to extract parallelism from programs expressed in terms of NumPy arrays.  
In contrast to other distributed array systems with similar goals (e.g., Dask Array, Nums), we 
deliver significantly higher performance by leveraging the Numba JIT compiler. We often 
achieve results approaching that from well-crafted C/MPI code.  

Our distributed array package is called Ramba.  It is available at http://github.com/Python-for-HPC/ramba.  
Ramba automatically partitions arrays across the nodes of a distributed system.  It takes a sequence of 
array operations, aggregates them, and then applies operator fusion, reordering and other transformations 
to construct a JIT-compiled parallel function. We then execute across a distributed dataset at speeds on par 
with compiled C code.  Ramba exploits data locality to minimize communication and maximize utilization of parallel resources.  

Ramba is a drop-in replacement for NumPy.  It supports most array creation and initialization routines, arithmetic 
operations, most simple indexing/slicing operations, and the ufunc API.  Fancy indexing, many matrix manipulation 
functions, and library subpackages (linalg, FFT, etc.) have not yet been implemented.  See 
https://github.com/Python-for-HPC/ramba#numpy-compatibility for more details.  

We evaluated Ramba?s performance using benchmarks from the Parallel Research Kernels (PRK)  
(https://github.com/IntelLabs/Kernels/tree/RayPython) and DPbench (https://github.com/IntelPython/dpbench/tree/feature/dist/distributed). 
Our preliminary experiments use the Stencil benchmark from PRK for a 30000x30000 array and a stencil of radius 2.  
The Numpy and Ramba implementations execute the following code in a timed loop:

.. code-block:: python
   :linenos:

    # sz = 30000
    B[2:sz-2, 2:sz-2] += w0*A[0:sz-4, 2:sz-2] + w1*A[1:sz-3, 2:sz-2] + 
           w2*A[2:sz-2, 0:sz-4] + w3*A[2:sz-2, 1:sz-3] + w4*A[2:sz-2, 2:sz-2] + 
           w5*A[2:sz-2, 3:sz-1] + w6*A[2:sz-2, 4:sz] + w7*A[3:sz-1, 2:sz-2] + 
           w8*A[4:sz, 2:sz-2]
    A += 1.0


Ramba achieves 110 times the throughput of baseline NumPy code on a single node, 
and 380 times Numpy throughput on 4 nodes.  It is tens of times faster than Dask and 
Nums (alternative Python-based distributed array frameworks).  Ramba achieves 
85% of the throughput of a C/MPI implementation of the benchmarks with code 
that is essentially unmodified from the NumPy baseline.  
See https://github.com/Python-for-HPC/ramba#performance-comparisons for more details.   

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b


Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension. 

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work. 

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references. 

So what would previously have registered as a in text reference ``[Atr03]_`` for 

:: 

     [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

what you actually see will be an empty reference rendered as **[?]**.

E.g., [Atr03]_.


If you wish to have a block quote, you can just indent the text, as in 

    When it is asked, What is the nature of all our reasonings concerning matter of fact? the proper answer seems to be, that they are founded on the relation of cause and effect. When again it is asked, What is the foundation of all our reasonings and conclusions concerning that relation? it may be replied in one word, experience. But if we still carry on our sifting humor, and ask, What is the foundation of all conclusions from experience? this implies a new question, which may be of more difficult solution and explication. :cite:`hume48`

Dois in bibliographies
++++++++++++++++++++++

In order to include a doi in your bibliography, add the doi to your bibliography
entry as a string. For example:

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = "10.1017/CBO9780511808432",
   }


If there are errors when adding it due to non-alphanumeric characters, see if
wrapping the doi in ``\detokenize`` works to solve the issue.

.. code-block:: bibtex

   @Book{hume48,
     author =  "David Hume",
     year =    "1748",
     title =   "An enquiry concerning human understanding",
     address =     "Indianapolis, IN",
     publisher =   "Hackett",
     doi = \detokenize{10.1017/CBO9780511808432},
   }

Source code examples
--------------------

Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

    # sz = 30000
    B[2:sz-2, 2:sz-2] += w0*A[0:sz-4, 2:sz-2] + w1*A[1:sz-3, 2:sz-2] + 
           w2*A[2:sz-2, 0:sz-4] + w3*A[2:sz-2, 1:sz-3] + w4*A[2:sz-2, 2:sz-2] + 
           w5*A[2:sz-2, 3:sz-1] + w6*A[2:sz-2, 4:sz] + w7*A[3:sz-1, 2:sz-2] + 
           w8*A[4:sz, 2:sz-2]
    A += 1.0

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


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


