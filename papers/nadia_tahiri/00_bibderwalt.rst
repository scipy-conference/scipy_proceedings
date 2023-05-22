:author: Wanlin Li
:email: Wanlin.Li@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada

:author: Nadia Tahiri
:email: Nadia.Tahiri@USherbrooke.ca
:institution: Department of Computer Science, University of Sherbrooke, Sherbrooke, Canada


:bibliography: mybib

----------------------------------------------------------------------------------------------------------------
Reproducible and Interactive Phylogeographic Analysis of SARS-CoV-2 Variation: Leveraging Neo4j Database and Snakemake for an Efficient Interface
----------------------------------------------------------------------------------------------------------------

.. class:: abstract

   The gene sequencing data and related lineage tracing and research data generated during the COVID-19 pandemic are valuable resources for researchers. To leverage these resources effectively, we have developed a web-based analysis platform called aPhyloGeo-Covid, utilizing Neo4j, Snakemake, and Python. This platform enables users to collect and visualize phylogeographic data related to SARS-CoV-2. Additionally, aPhyloGeo-Covid offers a scalable and reproducible workflow for investigating the relationship between geographic features and the patterns of variation in different SARS-CoV-2 variants. The integrated Neo4j database consolidates a diverse range of COVID-19 pandemic-related sequences, climate data, and metadata from public databases, allowing users to filter and organize input data for phylogeographical studies efficiently. Currently, the database contains over 113774 nodes and 194381 relationships. The platform's codebase is available on GitHub, offering researchers a valuable tool for analyzing and exploring the intricate dynamics of SARS-CoV-2 in a phylogeographic context.

.. class:: keywords

   Phylogeography, Neo4j, Snakemake, Dash, SARS-CoV-2

Introduction
------------

Phylogeography is a field of study that investigates the geographic distribution of genetic lineages within a particular species, including viruses. It combines principles from evolutionary biology and biogeography to understand how genetic variation is distributed across different spatial scales :cite:`dellicour2019using`. In the context of viruses, phylogeography seeks to uncover the evolutionary history and spread of viral lineages by analyzing their genetic sequences and geographical locations. By examining the genetic diversity of viruses collected from various geographic locations, researchers can reconstruct the patterns of viral dispersal and track the movement and transmission dynamics of viral populations over time :cite:`vogels2023phylogeographic` :cite:`franzo2022phylodynamic` :cite:`munsey2021phylogeographic`. For phylogeographic studies in viruses, researchers typically require integrating genetic sequences, geographic information and temporal information. By combining the genetic sequences with geographic information, researchers can analyze the phylogenetic relationships among the viral strains and infer the patterns of viral migration and transmission across different regions. By integrating genetic and temporal information, researchers can infer the timescale of viral evolution, and trace the origins and dispersal patterns of different viral lineages :cite:`holmes2004phylogeography`. Throughout the COVID-19 pandemic, researchers worldwide sequenced the genomes of thousands of SARS-CoV-2 viruses. These efforts have helped researchers study the virus's evolution and spread over time and across different geographic regions, which is critical to informing public health strategies for controlling future outbreaks. However, the abundance of genetic sequences and the accompanying geographic and temporal data are scattered across multiple databases, making it challenging to extract, validate, and integrate the information. For instance, to conduct a phylogeographic study in SARS-CoV-2, a researcher would first need access to data on the geographic distribution of specific lineages, including the most common countries where they are found, as well as the earliest and latest detected dates. This data is provided by the Cov-Lineages.org Lineage Report :cite:`o2021tracking`. Subsequently, based on the most common country and lineage detection dates, the researcher would need to search for sequencing data in databases such as NCBI Virus resource :cite:`brister2015ncbi` or GISAID :cite:`khare2021gisaid`. Climate data can be obtained from references to datasets like NASA/POWER and DailyGridded weather :cite:`marzouk2021assessment`. Additional data, including epidemiological information like COVID-19 testing and vaccination rates, can be retrieved from projects like Our World in Data :cite:`mathieu2021global`. In summary, conducting phylogeographic research in viruses involves not only screening and selecting sequencing data but also managing the associated geographic information and integrating vast amounts of environmental data. This process can be time-consuming and prone to errors. The challenges associated with data collection, extraction, and integration have hindered the advancement of phylogeographic research in the field. To address these challenges, a highly scalable and flexible graph database management system Neo4j :cite:`guia2017graph` was applied to store, manage, and query large-scale SARS-CoV-2 variants-related data. Unlike traditional relational databases that use tables and rows, Neo4j represents data as a network of interconnected nodes and relationships. It leverages graph theory and provides a powerful framework for modelling, storing, and analyzing complex relationships between entities :cite:`angles2012comparison`.

On the other hand, while recent phylogeographic studies have extensively analyzed the genetic data of species distributed under different geographical locations, many of them have only focused on the distribution of species or provided visual representations without exploring the correlation between specific genes (or gene segments) and environmental factors :cite:`uphyrkina2001phylogenetics` :cite:`luo2004phylogeography` :cite:`taylor2020intercontinental` :cite:`aziz2022phylogeography`. To fill this gap, a novel algorithm applying sliding windows to scan the genetic sequence information related to their climatic conditions was developed by our team :cite:`koshkarov2022phylogeography`. This algorithm utilizes sliding windows to scan genetic sequence information in relation to climatic conditions. Multiple sequences are aligned and segmented into numerous alignment windows based on predefined window size and step size. To assess the relationship between variation patterns within species and geographic features, the Robinson and Foulds metric :cite:`robinson1981comparison` was employed to quantify the dissimilarity between the phylogenetic tree of each window and the topological tree of geographic features. However, this process was computationally intensive as each window needed to be processed independently. Additionally, determining the optimal sliding window size and step size often required multiple parameter settings to optimize the analysis. Thus, reproducibility played a critical role in this process. To address these challenges, we designed a phylogeographic pipeline that leverages Snakemake, a modern computational workflow management system :cite:`koster2012snakemake`. Unlike other workflow management systems such as Galaxy :cite:`jalili2020galaxy` and Nextflow :cite:`spivsakova2023nextflow`, Snakemake stands out for being written in Python, making it highly portable and requiring only a Python installation to run Snakefiles :cite:`wratten2021reproducible`. The Snakemake workflow can harnesses various Python packages, including Biopython :cite:`cock2009biopython` and Pandas :cite:`lemenkova2019processing`, enabling efficient handling of sequencing data reading and writing as well as phylogenetic analysis. This makes Python-based Snakemake the ideal choice for aPhyloGeo-Covid. Furthermore, the Snakemake pipeline seamlessly integrates with other tools through Conda, ensuring efficient dependency and environment management. With a single command, all necessary dependencies can be downloaded and installed. Another significant advantage of Snakemake is its scalability, capable of handling large workflows with numerous rules and dependencies. It can be executed on various computing environments, including workstations, clusters, and cloud computing platforms like Kubernetes, Google Cloud Platform, and Amazon Web Services. Moreover, Snakemake supports parallel execution of jobs, greatly enhancing the pipeline's overall performance and speed.

With these considerations in mind, the main aim of this study is to create an open-source, web-based phylogeographic analysis platform that overcomes the aforementioned limitations. This platform comprises two essential components: data pre-processing and phylogeographical analysis. In the data pre-processing phase, we utilize searchable graph databases to facilitate rapid exploration and provide a visual overview of the SARS-CoV-2 variants and their associated environmental factors. This enables researchers to efficiently navigate through the vast amount of data and extract relevant information for their analyses. In the phylogeographical analysis phase, we employ our modularized Snakemake workflow to investigate how patterns of genetic variation within different SARS-CoV-2 variants align with geographic features. By utilizing this workflow, researchers can analyze the relationship between viral genetic diversity and specific geographic factors in a structured and reproducible manner. This comprehensive approach allows for a deeper understanding of the complex interplay between viral evolution, transmission dynamics, and environmental influences.

Methodology
------------

Neo4j graph database and Dash platform
++++++++++++++++++++++

A graph database is a type of database management system (DBMS) that uses graph structures for data representation and query processing :cite:`timon2021overview`. Unlike traditional relational databases that store data in tables with rows and columns, graph databases organize data as nodes, edges, and properties. In a graph database, nodes represent entities or objects, edges represent the relationships between nodes, and properties provide additional information about nodes and edges. One of the critical advantages of graph databases is their ability to traverse and query interconnected data efficiently. Graph databases excel at handling queries involving relationship patterns, graph algorithms, and path traversals. They enable efficient navigation through complex networks, enabling robust graph-based analyses and insights :cite:`vicknair2010comparison`.

1.	Data Integration 
++++++++++++++++++++++

Various data sources related to SARS-CoV-2 were integrated into a Neo4j database, covering the period from January 1, 2020, to December 31, 2022. The data sources include SARS-CoV-2 sequences from the SARS-CoV-2 Data Hub :cite:`brister2015ncbi`, lineage development information from Cov-Lineages :cite:`o2021tracking`, population density by country, positivity rates, vaccination rates, diabetes rates, aging data from Our World in Data :cite:`mathieu2021global`, and climate data from NASA/POWER :cite:`marzouk2021assessment`. Within the Neo4j database, we defined several labels to organize the data. These labels include Lineage, Protein, Nucleotide, Location, and LocationDay (See Fig.1). The Protein and Nucleotide labels store sequencing data information such as Accession, length, collection date, and collected country. The Lineage label stores lineage development information, including the most common country, latest date, and earliest date associated with each lineage. The LocationDay label stores climate information such as temperature, precipitation, wind speed, humidity and sky shortwave irradiance for each location and specific day. The Location label contains basic information about hospitals, health, and the economy of each country, including GDP, median age, life expectancy, population, the proportion of people aged 65 and older, proportion of smokers, proportion of extreme poverty, diabetes prevalence, human development index, and more. Lineage nodes are connected to Nucleotide and Protein nodes, representing the relationships between lineages and their associated genetic sequence data. Lineage nodes also have relationships with Location nodes, using the most common occurrence rate as a property. This design allows users to determine the most common countries based on lineage names or search for lineages that were most common in specific countries during a certain time period.

2.	Input exploration
++++++++++++++++++++++

To provide users with an interactive interface, we developed a web-based platform using Dash-Plotly :cite:`liermann2021dynamic`. Connecting the Dash Web platform to the Neo4j graph database enables quick retrieval of relevant data information from related nodes when users provide keywords about lineages or locations. This functionality allows users to quickly identify and filter the appropriate datasets for further phylogeographic analysis. By combining the power of the Neo4j database and the user-friendly web-based platform, our design facilitates efficient data exploration and selection, supporting researchers in their phylogeographic analysis of SARS-CoV-2 variation.

The aPhyloGeo-Covid provids two approaches to select input datasets.

(1).	Determine the most common country for the lineages based on the name of the lineage, and then retrieve the corresponding sequences.

The multi-step process is facilitated by the "Neo4j GraphDatabase" Python package :cite:`jordan2014neo4j` and the interactive Dash web page. As depicted in Fig. 2, firstly, users select specific lineages of interest from a checklist on the Dash web page. Next, utilizing the capabilities of the "Neo4j GraphDatabase" package, the selected lineages are used to query the graph database, retrieving relevant location information such as associated locations, earliest and latest detected dates of the lineages in the most common location, and their most common rates. Once these results are obtained from the database, they are presented on the web page as an interactive Dash Table. This table provides a user-friendly interface, allowing users to apply columns and rows filters. This feature enables the removal of study areas or lineages deemed irrelevant, as well as excluding lineages with a most common rate below a predetermined threshold. Finally, based on the filtered table and the selected sequence type, the "Neo4j GraphDatabase" package extracts all the related sequences by accession number. These filtered sequences were then collected as part of the input data for subsequent phylogeographic analysis.

(2).	Search for lineages that were most common in a specific country during a certain time period, and then retrieve the corresponding sequences.

This approach involved users defining specific locations and a date period through the Dash web page (See Fig.3). Utilizing the capabilities of the GraphDatabase package, the Neo4j database is queried to identify lineages prevalent in the specified locations during the defined time period. The retrieved information includes the earliest and latest detected dates of the lineages in each country and their most common rates. These results were presented to users through an interactive Dash Table, which facilitated the application of filters to eliminate outside study areas or lineages below a predetermined threshold. Then, the GraphDatabase package is utilized again to filter and extract the accession number of the corresponding sequences, which are then collected for subsequent phylogeographic analysis.

In summary, these approaches leveraged the "Neo4j GraphDatabase" package and the interactive Dash web page to enable user-driven sequencing searching. Once input sequencing has been defined, an Input node is generated and labelled accordingly in our graph database. 
This Input node is connected to each sequencing (Nucleotide or Protein) node used in the analysis, establishing relationships between the input data and the corresponding sequences. Each Input node is assigned a unique ID, which is provided to the client for reference.

3.	Parameters setting and tuning
++++++++++++++++++++++

Once the input data has been defined, including sequence data and associated location information, the platform guides users to select the parameters for analysis (See Fig.4). At this step, a Label named Analysis is created, and the values of the parameters are saved in the node as properties. These parameters include step size, window size, RF distance threshold, bootstrap threshold, and the list of the environmental factors involved in the analysis. Then a connection between the Input Node and the Analysis Node is created, which offers several advantages. Firstly, it enables users to compare the differences in results obtained from the same input samples but with different parameter settings. Secondly, it facilitates the comparison of analysis results obtained using the same parameter settings but different input samples. The networks of Input, Analysis, and Output nodes (Fig. 1) ensure repeatability and comparability of the analysis results.

Subsequently, when the user confirms the start of the analysis with the SUBMIT button, the corresponding sequences are downloaded from NCBI :cite:`brister2015ncbi` using the Biopython package :cite:`cock2009biopython`, and multiple sequence alignments (MSA) :cite:`edgar2006multiple` are performed using the MAFFT method :cite:`katoh2013mafft`. With alignment results and related environmental data as input, the Snakemake workflow will be triggered in the backend. Once the analysis is completed, the user is assigned a unique output ID, which they can use to query and visualize the results in the web platform.

4.	Output exploration
++++++++++++++++++++++

At the end of each analysis, an output node with a unique id is created in the Neo4j graph database. The associated nodes containing input and parameter information are connected to it by edges. Therefore, users can retrieve and visualize the analysis results through Output ID. The platform allows users to query individual results but also provides the capability to compare the results of multiple analyses. 

Input, Analysis, and Output nodes created by different users form a network that encompasses numerous combinations of parameter settings and input configurations. As the utilization of the platform expands, this network grows, resulting in an open academic platform that fosters communication and collaboration. This feature enhances the user's ability to gain insights from the data and enables comprehensive analysis of the phylogeographic patterns of SARS-CoV-2 variation.

Snakemake workflow for phylogenetic analysis
++++++++++++++++++++++

In this study, a combination of sliding window strategy and phylogenetic analyses was used to explore the potential correlation between the diversity of specific genes or gene fragments and their geographic distribution. The approach involved partitioning a multiple sequence alignment into windows based on sliding window size and step size. Phylogenetic trees were constructed for each window, and cluster analyses were performed for various geographic factors using distance matrices and the Neighbor-Joining clustering method :cite:`mihaescu2009neighbor`. The correlation between phylogenetic and reference trees was evaluated using Robinson and Foulds (RF) distance calculation. Bootstrap and RF thresholds were applied to identify gene fragments with variation patterns within species that coincided with specific geographic features, providing informative reference points for future studies. The workflow encompassed steps such as reference tree construction, sliding windows, phylogenetic tree construction, preliminary filtering based on bootstrap threshold and RF distance, advanced phylogenetic tree construction, and further filtering based on bootstrap threshold and RF distance. The workflow utilized tools and software like Biopython :cite:`cock2009biopython`, raxml-ng :cite:`kozlov2019raxml`, fasttree :cite:`price2009fasttree`, and Python libraries such as robinson-foulds, NumPy, and pandas for data parsing, phylogenetic inference, RF distance calculation, mutation testing, and filter creation. A manuscript for aPhyloGeo-pipeline is available on Github Wiki (https://github.com/tahiri-lab/aPhyloGeo-pipeline/wiki).


Conclusions
------------

This project showcases the development of an open-source, web-based platform designed to enhance phylogeographic research. By combining graph databases and a modularized Snakemake workflow, the platform overcomes the limitations of manual tools, enabling efficient extraction, validation, and integration of genetic and environmental data. The primary focus of the platform is to advance the analysis of geographic and environmental data associated with SARS-CoV-2. 

The utilization of the platform results in the accumulation of diverse findings from various analyses in the database. This network of data sources and analysis outputs expands as more researchers contribute to the platform. The centralized database serves as a repository for researchers to access and explore a wide range of results, fostering knowledge sharing and exchange within the scientific community. While the platform is currently in its early stage of development and testing before deployment, it is anticipated that the network of analyses will progressively become more interconnected as its popularity grows and attracts more researchers. Researchers can use this network to compare their results and identify patterns. The platform facilitates the dissemination of research findings, encourages researchers to build on each other's work, and promotes a sense of community and scientific progress.

In summary, as a database-driven network of analyses platform, aPhyloGeo-Covid forms an open academic platform that facilitates communication, collaboration, and knowledge sharing. By providing access to diverse results and encouraging interaction among researchers, the platform strengthens the scientific community and contributes to the advancement of research in the field of phylogeography.

Future directions
------------

To further improve aPhyloGeo-Covid, several potential directions can be considered:

1.	Enhancing Data Resources: Expand the available data resources, particularly geographic and environmental data. This could involve incorporating additional datasets, such as epidemiological information. By increasing the richness and diversity of data, aPhyloGeo-Covid can provide more comprehensive insights into the spatial and environmental factors influencing the spread and evolution of SARS-CoV-2.

2.	Expanding Phylogeographic Analysis Workflows: Besides the existing pipeline that explores the correlation between specific genes or gene fragments and their geographic distribution, consider adding more phylogeographic analysis workflows. By incorporating a broader range of analysis approaches, aPhyloGeo-Covid can offer a more comprehensive toolkit for investigating the evolutionary dynamics and spatial spread of the virus.

3.	Scalability and Efficiency: Focus on making the platform more scalable and efficient. This could involve leveraging parallel computing capabilities to handle larger datasets and accommodate growing user demands. Enhancing the platform's scalability and efficiency will ensure that aPhyloGeo-Covid can handle increasing data volumes and provide rapid and reliable analyses for researchers and public health practitioners.

Acknowledgements
------------

The authors thank SciPy conference and reviewers for their valuable comments on this paper. This work was supported by the Natural Sciences and Engineering Research Council of Canada, the Université de Sherbrooke grant, and the Centre de recherche en écologie de l’Université de Sherbrooke (CREUS).






Bibliographies, citations and block quotes
------------------------------------------

If you want to include a ``.bib`` file, do so above by placing  :code:`:bibliography: yourFilenameWithoutExtension` as above (replacing ``mybib``) for a file named :code:`yourFilenameWithoutExtension.bib` after removing the ``.bib`` extension.

**Do not include any special characters that need to be escaped or any spaces in the bib-file's name**. Doing so makes bibTeX cranky, & the rst to LaTeX+bibTeX transform won't work.

To reference citations contained in that bibliography use the :code:`:cite:`citation-key`` role, as in :cite:`hume48` (which literally is :code:`:cite:`hume48`` in accordance with the ``hume48`` cite-key in the associated ``mybib.bib`` file).

However, if you use a bibtex file, this will overwrite any manually written references.

So what would previously have registered as a in text reference ``
Atr03]_`` for

::

     
     Atr03] P. Atreides. *How to catch a sandworm*,
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

