:author: Wanlin Li
:email: Wanlin.Li@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada

:author: Nadia Tahiri
:email: Nadia.Tahiri@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada

:bibliography: mybib


-----------------------------------------------------------------------------------------------------------------------------
aPhyloGeo-Covid: A Web Interface for Reproducible Phylogeographic Analysis of SARS-CoV-2 Variation using Neo4j and Snakemake 
-----------------------------------------------------------------------------------------------------------------------------

.. class:: abstract

   The gene sequencing data, along with the associated lineage tracing and research data generated 
   throughout the Coronavirus disease 2019 (COVID-19) pandemic, constitute invaluable resources that profoundly 
   empower phylogeography research. To optimize the utilization of these resources, we have developed a web-based 
   analysis platform called aPhyloGeo-Covid, leveraging the capabilities of Neo4j, Snakemake, and Python. 
   
   This platform enables users to explore and visualize a wide range of diverse data sources specifically relevant to 
   SARS-CoV-2 for phylogeographic analysis. The integrated Neo4j database acts as a comprehensive repository, 
   consolidating COVID-19 pandemic-related sequences information, climate data, and demographic data obtained from 
   public databases, facilitating efficient filtering and organization of input data for phylogeographical studies. 
   
   Presently, the database encompasses over 113,774 nodes and 194,381 relationships. Once the input dataset is determined, 
   aPhyloGeo-Covid provides a scalable and reproducible workflow for investigating the intricate relationship between geographic 
   features and the patterns of variation in different SARS-CoV-2 variants. 
   
   The platform's codebase is publicly accessible on GitHub (https://github.com/tahiri-lab/iPhyloGeo/tree/iPhylooGeo-neo4j), 
   providing researchers with a valuable tool to analyze and explore the intricate dynamics of SARS-CoV-2 within a phylogeographic context.
   

.. class:: keywords

   Phylogeography, Neo4j, Snakemake, Dash, SARS-CoV-2

Introduction
------------

Phylogeography is a field of study that investigates the geographic distribution of genetic lineages within a particular species, 
including viruses. It combines principles from evolutionary biology and biogeography to understand how genetic variation is distributed 
across different spatial scales :cite:`dellicour2019using`. In the context of viruses, phylogeography seeks to uncover the evolutionary 
history and spread of viral lineages by analyzing their genetic sequences and geographical locations. By examining the genetic diversity 
of viruses collected from various geographic locations, researchers can reconstruct the patterns of viral dispersal and track the movement 
and transmission dynamics of viral populations over time :cite:`vogels2023phylogeographic` :cite:`franzo2022phylodynamic` :cite:`munsey2021phylogeographic`. 
For phylogeographic studies in viruses, researchers typically require integrating genetic sequences, geographic information and 
temporal information. By combining the genetic sequences with geographic information, researchers can analyze the phylogenetic relationships 
among the viral strains and infer the patterns of viral migration and transmission across different regions. By integrating genetic and 
temporal information, researchers can infer the timescale of viral evolution, and trace the origins and dispersal patterns of different viral 
lineages :cite:`holmes2004phylogeography`. Throughout the COVID-19 pandemic, researchers worldwide sequenced the genomes of thousands of SARS-CoV-2 viruses. 
These efforts have helped researchers study the virus's evolution and spread over time and across different geographic regions, which is critical 
to informing public health strategies for controlling future outbreaks. However, the abundance of genetic sequences and the accompanying geographic 
and temporal data are scattered across multiple databases, making it challenging to extract, validate, and integrate the information. For instance, 
to conduct a phylogeographic study in SARS-CoV-2, a researcher would first need access to data on the geographic distribution of specific lineages, 
including the most common countries where they are found, as well as the earliest and latest detected dates. This data is provided by the Cov-Lineages.org 
Lineage Report :cite:`o2021tracking`. Subsequently, based on the most common country and lineage detection dates, the researcher would need to search 
for sequencing data in databases such as NCBI Virus resource :cite:`brister2015ncbi` or GISAID :cite:`khare2021gisaid`. Climate data can be obtained 
from references to datasets like NASA/POWER and DailyGridded weather :cite:`marzouk2021assessment`. Additional data, including epidemiological information 
like COVID-19 testing and vaccination rates, can be retrieved from projects like Our World in Data :cite:`mathieu2021global`. In summary, conducting 
phylogeographic research in viruses involves not only screening and selecting sequencing data but also managing the associated geographic information and 
integrating vast amounts of environmental data. This process can be time-consuming and prone to errors. The challenges associated with data collection, 
extraction, and integration have hindered the advancement of phylogeographic research in the field. To address these challenges, a highly scalable and 
flexible graph database management system Neo4j :cite:`guia2017graph` was applied to store, manage, and query large-scale SARS-CoV-2 variants-related data. 
Unlike traditional relational databases that use tables and rows, Neo4j represents data as a network of interconnected nodes and relationships. 
It leverages graph theory and provides a powerful framework for modelling, storing, and analyzing complex relationships between 
entities :cite:`angles2012comparison`.

On the other hand, while recent phylogeographic studies have extensively analyzed the genetic data of species distributed under different 
geographical locations, many of them have only focused on the distribution of species or provided visual representations without exploring 
the correlation between specific genes (or gene segments) and environmental factors :cite:`uphyrkina2001phylogenetics` :cite:`luo2004phylogeography` 
:cite:`taylor2020intercontinental` :cite:`aziz2022phylogeography`. To fill this gap, a novel algorithm applying sliding windows to scan the genetic 
sequence information related to their climatic conditions was developed by our team :cite:`koshkarov2022phylogeography`. This algorithm utilizes sliding 
windows to scan genetic sequence information in relation to climatic conditions. Multiple sequences are aligned and segmented into numerous alignment windows 
based on predefined window size and step size. To assess the relationship between variation patterns within species and geographic features, the Robinson and 
Foulds metric :cite:`robinson1981comparison` was employed to quantify the dissimilarity between the phylogenetic tree of each window and the topological tree 
of geographic features. However, this process was computationally intensive as each window needed to be processed independently. Additionally, determining 
the optimal sliding window size and step size often required multiple parameter settings to optimize the analysis. Thus, reproducibility played a 
critical role in this process. To address these challenges, we designed a phylogeographic pipeline that leverages Snakemake, a modern computational 
workflow management system :cite:`koster2012snakemake`. Unlike other workflow management systems such as Galaxy :cite:`jalili2020galaxy` and Nextflow 
:cite:`spivsakova2023nextflow`, Snakemake stands out for being written in Python, making it highly portable and requiring only a Python installation to 
run Snakefiles :cite:`wratten2021reproducible`. The Snakemake workflow can harnesses various Python packages, including Biopython :cite:`cock2009biopython` 
and Pandas :cite:`lemenkova2019processing`, enabling efficient handling of sequencing data reading and writing as well as phylogenetic analysis. 
This makes Python-based Snakemake the ideal choice for aPhyloGeo-Covid. Furthermore, the Snakemake pipeline seamlessly integrates with other tools 
through Conda, ensuring efficient dependency and environment management. With a single command, all necessary dependencies can be downloaded and installed. 
Another significant advantage of Snakemake is its scalability, capable of handling large workflows with numerous rules and dependencies. 
It can be executed on various computing environments, including workstations, clusters, and cloud computing platforms like Kubernetes, Google 
Cloud Platform, and Amazon Web Services. Moreover, Snakemake supports parallel execution of jobs, greatly enhancing the pipeline's overall performance and speed.

With these considerations in mind, the main aim of this study is to create an open-source, web-based phylogeographic analysis platform that overcomes 
the aforementioned limitations. This platform comprises two essential components: data pre-processing and phylogeographical analysis. 
In the data pre-processing phase, we utilize searchable graph databases to facilitate rapid exploration and provide a visual overview of 
the SARS-CoV-2 variants and their associated environmental factors. This enables researchers to efficiently navigate through the vast amount of 
data and extract relevant information for their analyses. In the phylogeographical analysis phase, we employ our modularized Snakemake workflow to 
investigate how patterns of genetic variation within different SARS-CoV-2 variants align with geographic features. By utilizing this workflow, 
researchers can analyze the relationship between viral genetic diversity and specific geographic factors in a structured and reproducible manner. 
This comprehensive approach allows for a deeper understanding of the complex interplay between viral evolution, transmission dynamics, 
and environmental influences.


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

Citing software and websites
++++++++++++++++++++++++++++

Any paper relying on open-source software would surely want to include citations.
Often you can find a citation in BibTeX format via a web search.
Authors of software packages may even publish guidelines on how to cite their work.

For convenience, citations to common packages such as
Jupyter :cite:`jupyter`,
Matplotlib :cite:`matplotlib`,
NumPy :cite:`numpy`,
pandas :cite:`pandas1` :cite:`pandas2`,
scikit-learn :cite:`sklearn1` :cite:`sklearn2`, and
SciPy :cite:`scipy`
are included in this paper's ``.bib`` file.

In this paper we not only terraform a desert using the package terradesert :cite:`terradesert`, we also catch a sandworm with it.
To cite a website, the following BibTeX format plus any additional tags necessary for specifying the referenced content is recommended.

.. code-block:: bibtex

   @Misc{terradesert,
     author = {TerraDesert Team},
     title = {Code for terraforming a desert},
     year = {2000},
     url = {https://terradesert.com/code/},
     note = {Accessed 1 Jan. 2000}
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

