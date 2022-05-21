:author: Wanlin Li
:email: jj@rome.it
:institution: Senate House, S.P.Q.R.
:institution: Egyptian Embassy, S.P.Q.R.

:author: Aleksandr Koshkarov
:email: mark37@rome.it
:institution: Egyptian Embassy, S.P.Q.R.

:author: My-Linh Luu
:email: millman@rome.it
:institution: Egyptian Embassy, S.P.Q.R.
:institution: Yet another place, S.P.Q.R.

:author: Nadia Tahiri
:email: Nadia.Tahiri@USherbrooke.ca
:institution: University of Sherbrooke
:bibliography: mybib


:video: http://www.youtube.com/watch?v=dhRUe-gz690

-------------------------------------------------------------
Phylogeo: Analysis of genetic and climatic data of SARS-CoV-2
-------------------------------------------------------------

.. class:: abstract

   Due to the fact that the SARS-CoV-2 pandemic reaches its peak, researchers around the globe are combining efforts to investigate the genetics of different variants to better deal with its distribution. This paper discusses how patterns of divergence within SARS-CoV-2 coincide with geographic features, such as climatic features, with help of phylogeography. 
   
   Phylogeo is a python-based bioinformatics pipeline dedicated to phylogeographic analysis. It is designed to allow researchers to better understand the virus spread in specific regions via a configuration file, and then run all analyses in a single execution. More specifically, the Phylogeo tool detects which parts of the genetic sequence undergo a high mutation rate based on geographic conditions using a sliding window that moves along the genetic sequence alignments in user defined steps.


.. class:: keywords

   Phylogeography, SARS-CoV-2, Bioinformatics, Genetic, Climatic change

Introduction
------------

The global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is at its peak and more and more variants of SARS-CoV-2 have been described over time. Of these variants, some are considered variants of concern (VOC) by the World Health Organization (WHO) due to their impact on global public health, such as Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), Delta (B.1.617.2), and Omicron (B.1.1.529) (Cascella et al., 2022). Although significant progress has been made in vaccine development and mass vaccination is being implemented in many countries, the continued emergence of new variants of SARS-CoV-2 threatens to reverse the progress made to date. Researchers around the world are collaborating to better understand the genetics of the different variants, as well as the factors that influence the epidemiology of this infectious disease. Genetic studies of the different variants have facilitated the development of vaccines to better combat the spread of the virus. Studying the factors (e.g., environment, host, agent of transmission) that influence the epidemiology helps us to limit the continued spread of infection and prepare for the future re-emergence of diseases caused by subtypes of a coronavirus (Lin et al., 2006). However, few studies have reported associations between environmental factors and the genetics of individual variants. Different variants of SARS-CoV-2 are expected to spread differently depending on geographical conditions, such as the meteorological environment. The main objective of this study is to find clear correlations between the genetics and geographic distribution of different SARS-CoV-2 variants.

Several studies have shown that SARS-CoV-2 cases and associated climatic factors are significantly correlated with each other (Oliveiros et al., 2020; Sobral et al., 2020; Sabarathinam et al., 2022). Oliveiros et al. (2020) reported a decrease in the rate of SARS-CoV-2 progression with the onset of spring and summer in the northern hemisphere. Sobral et al. (2020) suggested a negative correlation between mean temperature by country and the number of SARS-CoV-2 infections, as well as a positive correlation between rainfall and SARS-CoV-2 transmission. This contrasts with the results of the study by Sabarathinam et al. (2022), which showed that an increase in temperature led to an increase in the spread of SARS-CoV-2. The results of Chen et al. (2021) imply that a country located 1000 km closer to the equator can expect 33% fewer cases of SARS-CoV-2 per million population. Some virus variants may be more stable in environments with specific climatic factors. Sabarathinam et al. (2022) compared mutation patterns of SARS-CoV-2 with time series of changes in precipitation, humidity, and temperature. They suggested that temperature between 43°F and 54°F, humidity of 67-75%, and precipitation of 2-4 mm may be the optimal environment for the transition of the mutant form from D614 to G614.

In this study, we examine the geospatial lineage of SARS-CoV-2 by combining genetic data and metadata from associated sampling locations. Thus, an association between genetics and geographic distribution of SARS-CoV-2 variants can be found. We focus on developing a new algorithm to find relationships between a reference tree (i.e., a tree of geographic species distributions, temperature trees, habitat precipitation trees, or others) with their genetic compositions. This new algorithm can help find which genes or which subparts of a gene are sensitive or favorable to a given environment.

Problem statement and proposal
------------------------------

Phylogeography is the study of the principles and processes that govern the distribution of genealogical lineages, particularly at the intraspecific level. The geographic distribution of species is often correlated with the patterns associated with the species' genes (Avise, 2000). This term was introduced to describe the correlation between geographic data and genetic structures within a group of species that links the genetic appearance of species to different habitat environments (Knowles and Maddison, 2002).

In a phylogeographic study, three major processes must be considered (Nagylaki, 1992) which are:

1.	Genetic drift is the result of allele sampling errors. These errors are due to generational transmission of alleles and geographical barriers. Genetic drift is a function of the size of the population. Indeed, the larger the population, the lower the genetic drift. This is due to the ability to maintain genetic diversity in the original population. By convention, we say that an allele is fixed if it reaches the frequency of 100% and we say that it is lost if it reaches the frequency of 0%.

2.	Gene flow or migration is an important process for conducting a phylogeographic study. It is the transfer of alleles from one population to another, increasing intrapopulation diversity and decreasing interpopulation diversity.

3.	There are many selections in all species, let us indicate the two most important ones when they are important for a phylogeographic study.
   
   a.	Sexual selection is a phenomenon resulting from an attractive characteristic between two species. This selection is therefore a function of the size of the population.
   
   b.	Natural selection is a function of both fertility, mortality and adaptation of a species to a habitat. adaptation of a species to a habitat.

Populations living in different environments with varying climatic conditions are subject to pressures that can lead to evolutionary divergence and reproductive isolation [1,2- Orr, M. R., Smith, T. B. (1998);  Schluter, D. (2001)]. Phylogeny and geography are then correlated. This study therefore aims to present an algorithm to show the possible correlation between certain genes or gene fragments and the geographical distribution of species.

In this study, we focused on SARS-CoV-2 to understand the correlation between the occurrence of different variants and the climate environment. Identifying ways in which patterns of divergence within SARS-CoV-2 variants coincide with geographic features can be difficult for several reasons. 

Most studies in phylogeography consider only genetic data without directly considering climatic data. They indirectly take this information as a basis for locating the habitat of the species. We have developed the first version of a phylogeography that integrates climate data. The sliding window strategy will provide more robust results, as it will particularly highlight the area sensitive to climate adaptation. 

Methods and Python scripts
--------------------------

To achieve our goal, we designed a workflow and then developed a script in Python version 3.9. It interacts with multiple bioinformatic programs, taking nucleotide data as input, and performs multiple phylogenetic analyses using sliding window approach. The process is divided into three main steps (see Figure 1).

The first step involves collecting data to search for quality viral sequences that are essential for the condition of our results. All sequences were retrieved from the NCBI Virus website. In total, 20 regions were selected to represent 38 gene sequences of SARS-CoV-2. After collecting genetic data, we extracted 5 climatic factors of the 20 regions, i.e., temperature, humidity, precipitation, wind speed, and sky surface shortwave downward irradiance. This data was obtained from the NASA website (https://power.larc.nasa.gov/).

In the second step, trees are created with climatic data and genetic data, respectively. For climatic data, we calculated the dissimilarity between each pair of variants (i.e., from different climatic conditions), resulting in a symmetric square matrix. From this matrix, the neighbor joining algorithm was used to construct the climate tree. The same approach was implemented for genetic data. Using nucleotide sequences from the 38 SARS-CoV-2 lineages, phylogenetic reconstruction is repeated to construct genetic trees, considering only the data within a window that moves along the alignment in user-defined steps and windows size.

In the third step, the phylogenetic trees constructed in each sliding window are compared with the climatic trees using the Robinson and Foulds topological distance (Robinson and Foulds, 1981). The distance was normalized by 2n-6, where n is the number of leaves (i.e., taxa). The proposed approach considers bootstrapping. The implementation of sliding window technology provides a more accurate identification of regions with high gene mutation rates. 

As a result, we highlighted a correlation between parts of genes with a high rate of mutations depending on the geographic distribution of viruses, which emphasizes the emergence of new variants (i.e., Delta, Alpha, Gamma, Beta, and Omicron).

The creation of phylogenetic trees, as mentioned above, is an important part of the solution and includes the main steps of the developed pipeline. The main parameters of this part are as follows:


.. code-block:: python

   def create_phylo_tree(g...):
    ...
    for file in files:
        try:
            ...
            create_bootstrap()
            run_dnadist()
            run_neighbor()
            run_consense() 
            filter_results(...)
            ...
        except Exception as error:
            raise 


This function takes gene data, windows size, step size, bootstrap threshold, threshold for the Robinson and Foulds distance, and data names as input parameters. Then the function sequentially connects the main steps of the pipeline: align_sequence(gene), sliding_window(window_size, step_size), create_bootstrap(), run_dnadist(), run_neighbor(), run_consense(), and filter_results with parameters. As a result, we get a phylogenetic tree (or several trees), which is written to a file.

The sliding window strategy can detect genetic fragments depending on environmental parameters, but this work depends on time-consuming data preprocessing and the use of several bioinformatics programs. For example, we need to verify that each sequence identifier in the sequencing data always matches the corresponding metadata. If samples are added or removed, we need to check whether the sequencing data set matches the metadata set and make changes accordingly. In the next stage we need to align the sequences and integrate everything step by step into specific software such as MUSCLE, Consense, Seqboot, rf, Dnadist, Neighbor, and raxmlHPC. The use of each software requires expertise in bioinformatics. In addition, the intermediate analysis steps inevitably generate many intermediate files, the management of which not only consumes the biologist's time, but is also subject to errors, which reduces the reproducibility of the study. At present, there are only a few systems designed to automate the analysis of phylogeography. In this context, the development of a computer program for a better understanding of the nature and evolution of coronavirus is essential for the advancement of clinical research.


.. raw:: latex

   \begin{table*}

     \begin{longtable*}{|l|l|l|l||l|l|l|l|}
        \hline
         Lineage & Most Common Country                & Earliest Date & Sequence Accession & Lineage & Most Common Country & Earliest Date & Sequence Accession \\ \hline
         A.2.3 & United Kingdom   100.0\% & 2020-03-12 & OW470304.1 & P.1.7.1   & Peru 94.0\%              & 2021-02-07 & OK594577 \\ \hline
         C.1   & South Africa 93.0\%      & 2020-04-16 & OM739053.1 & P.1.13    & USA 100.0\%              & 2021-02-24 & OL522465 \\ \hline
         C.7   & Denmark 100.0\%          & 2020-05-11 & OU282540   & P.2       & Brazil 58.0\%            & 2020-04-13 & ON148325 \\ \hline
         C.17  & Egypt 69.0\%             & 2020-04-04 & MZ380247   & P.3       & Philippines 83.0\%       & 2021-01-08 & OL989074 \\ \hline
         C.20  & Switzerland 85.0\%       & 2020-10-26 & OU007060   & P.7       & Brazil 71.0\%            & 2020-07-01 & ON148327 \\ \hline
         C.23  & USA 90.0\%               & 2020-05-11 & ON134852   & N.1       & USA 91.0\%               & 2020-03-25 & MT520277 \\ \hline
         C.31  & USA 87.0\%               & 2020-08-11 & OM052492   & N.3       & Argentina 96.0\%         & 2020-04-17 & MW633892 \\ \hline
         C.36  & Egypt 34.0\%             & 2020-03-13 & MW828621   & N.4       & Chile 92.0\%             & 2020-03-25 & MW365278 \\ \hline
         C.37  & Peru 43.0\%              & 2021-02-02 & OL622102   & N.6       & Chile 98.0\%             & 2020-02-16 & MW365092 \\ \hline
         Q.2   & Italy 99.0\%             & 2020-12-15 & OU471040   & N.7       & Uruguay 100.0\%          & 2020-06-18 & MW298637 \\ \hline
         Q.3   & USA 99.0\%               & 2020-07-08 & ON129429   & N.8       & Kenya 94.0\%             & 2020-06-23 & OK510491 \\ \hline
         Q.6   & France 92.0\%            & 2021-03-02 & ON300460   & N.9       & Brazil 96.0\%            & 2020-09-25 & MZ191508 \\ \hline
         Q.7   & France 86.0\%            & 2021-01-29 & ON442016   & B.1.1.107 & United Kingdom   100.0\% & 2020-06-06 & OA976647 \\ \hline
         L.2   & Netherlands 73.0\%       & 2020-03-23 & LR883305   & B.1.1.172 & USA 100.0\%              & 2020-04-06 & MW035925 \\ \hline
         L.4     & United States of   America 100.0\% & 2020-06-29    & OK546730           & AK.2    & Germany 100.0\%     & 2020-09-19    & OU077014           \\ \hline
         D.2   & Australia 100.0\%        & 2020-03-19 & MW320730   & AH.1      & Switzerland 100.0\%      & 2021-01-05 & OD999779 \\ \hline
         D.3   & Australia 100.0\%        & 2020-06-14 & MW320869   & M.2       & Switzerland 90.0\%       & 2020-10-26 & OU009929 \\ \hline
         D.4   & United Kingdom   80.0\%  & 2020-08-13 & OA967683   & AE.2      & Bahrain 100.0\%          & 2020-06-23 & MW341474 \\ \hline
         D.5   & Sweden 65.0\%            & 2020-10-12 & OU370897   & BA.2.24   & Japan 99.0\%             & 2022-01-27 & BS004276 \\ \hline
     \end{longtable*}

     \caption{Area Comparisons \DUrole{label}{quanitities-table}}

   \end{table*}

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


