:author: Project Blosc
:email:
:institution: Project Blosc
:equal-contributor:
:bibliography: mybib

:author: Francesc Alted
:email: francesc@blosc.org
:institution: Project Blosc
:corresponding:

:author: Marta Iborra
:email: martaiborra24@gmail.com
:institution: Project Blosc

:author: Oscar Guiñón
:email: soscargm98@gmail.com
:institution: Project Blosc


---------------------------------------------------------------------------------
Using Blosc2 NDim As A Fast Explorer Of The Milky Way (Or Any Other NDim Dataset)
---------------------------------------------------------------------------------

.. class:: abstract

    N-dimensional datasets are common in many scientific fields, and quickly accessing subsets of these datasets is critical for an efficient exploration experience. Blosc2 is a compression and format library that recently added support for multidimensional datasets. Compression is crucial in effectively dealing with sparse datasets as the zeroed parts can be almost entirely suppressed, while the non-zero parts can still be stored in smaller sizes than their uncompressed counterparts. Moreover, the new double data partition in Blosc2 reduces the need for decompressing unnecessary data, which allows for top-class slicing speed.

    The Blosc2 NDim layer allows to create and read n-dimensional datasets in an extremely efficient way thanks to a completely general n-dim 2-level partitioning, allowing to slice and dice arbitrary large (and compressed!) data in a more fine-grained way. Having a second partition means that we have better flexibility to fit the different partitions at the different CPU cache levels, making compression more efficient.

    As an example, we will demonstrate how Blosc2 NDim enables a fast exploration of the Milky Way. We will use the Gaia DR3 dataset, which is a catalog of 1.7 billion stars in our galaxy. The dataset is 100 GB in size and it is stored in a 3D array of 1.7 billion rows, 6 columns and 3D coordinates (x, y, z).

.. class:: keywords

    explore datasets, n-dimensional datasets, Gaia DR3, Milky Way, Blosc2, compression

Introduction
------------

Blosc is a high-performance compressor that has been optimized for binary data. Its design allows for faster transmission of data to the processor cache than the traditional, non-compressed, direct memory fetch approach through an memcpy() OS call. This can be useful not only in reducing the size of large datasets on-disk or in-memory, but also in accelerating memory-bound computations, which is typical in vector-vector operations.

Blosc2 is the new iteration of the Blosc 1.x series, which adds more features and better documentation. In particular, the NDim feature of Blosc2 excels at reading multi-dimensional slices, thanks to its innovative pineapple-style partitioning (:cite:`BDT23-blosc2-ndim-intro`). This allows for a very fast exploration of general n-dim datasets (and in particular, a 3-d Milky Way dataset).

Blosc uses the blocking technique (as described here) to reduce activity on the memory bus as much as possible. The blocking technique divides datasets into blocks small enough to fit in the caches of modern processors and performs compression/decompression there. It also leverages SIMD (SSE2) and multi-threading capabilities present in modern multi-core processors to accelerate the compression/decompression process to the maximum.

.. figure:: sum_openmp-rainfall.png
   :scale: 40%

   Speed for summing up a vector of real float32 data using a variety of codecs that come with Blosc2. Observe the maximum speed is achieved when using the maximum number of (logical) threads in that computer (28). :label:`sum-precip`

Using Blosc compression can accelerate real computations when enough cores are dedicated to the task. In Figure :ref:`sum-precip` you can see a real example of this.


Blosc2 NDim
-----------

Blosc2 NDim is a new feature of Blosc2 that allows to create and read n-dimensional datasets in an extremely efficient way thanks to a completely general n-dim 2-level partitioning, allowing to slice and dice arbitrary large (and compressed!) data in a more fine-grained way. Having a second partition means that we have better flexibility to fit the different partitions at the different CPU cache levels, making compression more efficient.

.. figure:: b2nd-2level-parts.png
   :scale: 12%

   Blosc2 NDim 2-level partitioning. :label:`b2nd-2level-parts`

.. figure:: b2nd-3d-dset.png
   :scale: 40%

   Blosc2 NDim 2-level partitioning is flexible: you can specify the dimensions of both partitions in any arbitrary way that fitxs you read access patterns. :label:`b2nd-3d-dset`

With these more fine-grained cubes (aka partitions), it is possible to retrieve arbitrary n-dim slices more rapidly because you don't have to decompress all the data that is necessary for the more coarse-grained partitions typical in other libraries.  See Figures :ref:`b2nd-2level-parts` and :ref:`b2nd-3d-dset` on how this works and can be setup.


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

E.g., :cite:`Atr03`.


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

It is well known :cite:`Atr03` that Spice grows on the planet Dune.  Test
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

Customised LaTeX packages
-------------------------

Please avoid using this feature, unless agreed upon with the
proceedings editors.

::

  .. latex::
     :usepackage: somepackage

     Some custom LaTeX source here.

