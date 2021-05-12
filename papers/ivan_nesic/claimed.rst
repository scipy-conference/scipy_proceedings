:author: Romeo Kienzler
:email: romeo.kienzler@ch.ibm.com
:institution: IBM, Center for Open Source Data and AI Technologies (CODAIT)

:author: Ivan Nesic
:email: ivan.nesic@usb.ch
:institution: University Hospital of Basel


:video: http://www.youtube.com/fake-link-to-video-1sf4a5d22

------------------------------------------------
CLAIMED, a visual and scalable component library for Trusted AI
------------------------------------------------

.. class:: abstract

   Deep Learning models are getting more and more popular but constraints on explainability, adversarial robustness and fairness are often major concerns for production deployment. Although the open source ecosystem is abundant on addressing those concerns, fully integrated, end to end systems are lacking in open source.
Therefore we provide an entirely open source, reusable component framework, visual editor and execution engine for production grade machine learning on top of Kubernetes, a joint effort between IBM and the University Hospital Basel. It uses Kubeflow Pipelines, the AI Explainability360 toolkit, the AI Fairness360 toolkit and the Adversarial Robustness Toolkit on top of ElyraAI, Kubeflow, Kubernetes and JupyterLab. Using the Elyra pipeline editor, AI pipelines can be developed visually with a set of jupyter notebooks. We explain how we've created a COVID-19 deep learning classification pipeline based on CT scans. We use the toolkit to highlight parts of the images which have been crucial for the model’s decisions. We detect bias against age and gender and finally, show how to deploy the model to KFServing to share it across different hospital data centers of the Swiss Personalized Health Network.

.. class:: keywords
   Kubernetes, Kubeflow, JupyterLab, ElyraAI, KFServing, TrustedAI, AI Explainability, AI Fairness, AI Adversarial Robustness

Introduction
------------

Open source software for performing individual AI pipeline tasks are
abundant, but the community lacks a fully integrated, trusted and
scalable visual tool. Therefore we have built CLAIMED, the visual
**C**\ omponent **L**\ ibrary for **AI**, **M**\ achine Learning,
**E**\ TL and **D**\ ata Science which runs on top of ElyraAI capable of
pushing AI pipelines of any kind to Kubernetes. Any containerized
application can become a component of the library. CLAIMED has been
released [3]_ under the Apache v2 open source license. In the following
we introduce the open source components we are integrating in our
current release, followed by an overview of different component
categories paired with a description of exemplary components used in
health care. This pipeline is also available in open source  [4]_.


Containerization and Kubernetes
-------------------------------

Virtualization opened up a lot of potential for managing the
infrastructure, mainly the ability to run different operating systems on
the same hardware at the same time. Next step of isolation can be
performed for each of the microservices running on the server, but
instead of managing access rights and resources on the host operating
system, we can containerize these in separate packages with their own
environments. Practical effect of this is that we are running each of
the microservices as if they have their own dedicated virtual machine,
but without the overhead of such endeavour. This is accomplished by
running containers on top of the host operating system. An example of
the containerization platform is Docker.

Containerization made it possible to run a large number of containers,
which introduced the need of their orchestration. This means something,
and hopefully not someone, needs to constantly take care that the system
is in the desired state, it needs to scale up or down, manage
communication between containers, schedule them, manage authentications,
balance the load etc. Although there are other options like Docker
Swarm, Kubernetes is the market leader in this domain. It was donated to
CNCF by Google, which means a lot of Google’s know-how and years of
experience went into it. The system can run on public, on-prem or on
hybrid clouds. On-prem installation is very important for institutions
dealing with sensitive data. For IBM, Kubernetes is also strategic,
joining CNCF, having moved all Watson Services to Kubernetes and aquired
RedHat, IBM is now 3rd largest committer to Kubernetes.

Deep Learning with TensorFlow
----------------------------

It is the second incarnation of the Google Brain project’s scalable
distributed training and inference system named DistBelief
[tf]_. It supports myriad of hardware platforms, from
mobile phones to GPU/TPU clusters, for both training and inference. It
can even run in browser on the client’s side, without the data ever
leaving the machine. Apart from being a valuable tool in research, it is
also being used in demanding production environments. On a development
side, representing machine learning algorithms in a tree-like structures
makes it a very good expression interface. Lastly, on the performance vs
usability side, both eager and graph modes are supported. Meaning
debugging is much simpler in the first case, and if there is the need
for speed, one can use the latter.

Kubeflow
--------

Kubeflow is a machine learning pipeline management and execution system
running as first class citizen on top of Kubernetes. Besides making use
of Kubernetes scalability it allows for reproducible work as machine
learning pipelines and the results and intermediate artifacts of their
executions are stored in a meta data repository.

ElyraAI
-------

ElyraAI started as a set of extensions for the JupyterLab ecosystem.
Here we concentrate on the pipeline editor of ElyraAI which allows for
expression of machine learning workflows using a drag’n’drop editor and
send them for execution on top of different engines like Kubeflow or
Airflow. This allows for non-programmers to read and understand but also
create machine learning workflows. ElyraAI also supports visualizing
such pipelines in the browser (e.g. from a github repository).

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


