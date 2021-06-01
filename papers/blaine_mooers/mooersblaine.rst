:author: Blaine H. M. Mooers
:email: blaine-mooers@ouhsc.edu
:institution: Dept of Biochemistry and Molecular Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Stephenson Cancer Center, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Laboratory of Biomolecular Structure and Function, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Biomolecular Structure Core, Oklahoma COBRE in Structural Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:orcid: 0000-0001-8181-8987
:corresponding:



---------------------------------------------------------------------
Modernizing computing by structural biologists with Jupyter and Colab
---------------------------------------------------------------------

.. class:: abstract

Protein crystallography produces most of the molecular structures used in structure-based drug design. 
The process of structure determination is computationally intensive and error-prone because many software packages are involved.
Here, we attempt to support the reproducibility of this computational work by using Jupyter Notebooks in JupyterLab or Google Colab. 
We have made libraries of code templates to ease running the crystallography packages in Jupyter Notebooks.
Our combined use of GitHub, snippet libraries, Jupyter Notebooks, and Colab will help modernize the computing done by structural biologists.

.. class:: keywords

   literate programming,reproducible research, scientific rigor, electronic notebooks, JupyterLab, Jupyter Notebooks, computational structural biology, computational crystallography, biomolecular crystallography, protein crystallography, biomolecular structure, biomedical research, protein*drug interactions, RNA*drug interactions, molecular graphics, scientific communication, molecular artwork, computational molecular biophysics


Introduction
------------

Structural biologists study the molecular structures of proteins and nucleic acids to understand how they function in biology and medicine. 
The underlying premise of the field is that molecular function follows molecular form.
More precise aliases for these scientists include molecular structural biologists, structural biochemists, and molecular biophysicists. 
Some of the methods used to determine the near-atomic resolution molecular structures include molecular modeling, X-ray crystallography, NMR, and cryo-EM.
These scientists often use the molecular structures of these large biomolecules to design small-molecule drugs for improved therapies. 
As a result, structural biology plays a vital role in drug discovery and development, and many structural biologists work in the pharmaceutical industry.
Those in academia in the United States have their work funded by the National Institutes of Health, the National Science Foundation, the Department of Defense, or the Department of Energy.

Structural biology is at the intersection of biochemistry, molecular biology, molecular biophysics, and computer science. 
Structural biologists have diverse backgrounds and varying levels of experience with computer programming ranging from essentially none to very advanced. 
Several decades ago, the barriers to entry into the field included expertise with running command-line-driven programs and the ability to write programs to meet unmet needs not met by existing software packages. 
However, these barriers have been lowered over the past two decades by the widespread availability of GUI-driven software that is often free for academics (e.g., CCP4 [Winn11]_, Phenix [Lieb11]_, CNS [Brun98]_, ATSAS [Mana21]_, BioXTAS [Hopk17]_, CCPEM [Burn17]_ ). 
As a result, biologists have become the largest component of the field.

Computing is involved in the six or more steps from structural data acquisition to publication.
Several alternate software packages are often available for each step. 
Different combinations of these alternatives lead to a combinatorial explosion of possible workflows. 
Although in some special situations, workers have set up software pipelines for some but not all of the steps.
However, these pipelines are difficult to transfer or have trouble with the challenging samples that cannot yet be handled without human intervention.
The current situation makes the computational work vulnerable to errors in tracking input and output files. 
Storing the code and outputs in Jupyter Notebooks would reduce this vulnerability . 

To ease crystal structure determination in Jupyter, we made libraries of code templates for crucial programs [Kluy16]_. 
We formatted the libraries for two code snippet extensions for JupyterLab. 
One extension (jupyterlab-snippets) provides access to snippets organized into nested pull-down menus. 
The other extension (elyra-code-snippet-extension) uses a search box to locate the desired snippet. 
The user can easily add new code snippets to both systems. 

We also ported the libraries to Google Colab.
Colab is an integrated development environment (IDE) for running Jupyter Notebooks on the Google Cloud Platform.
Colab spares the user of hardware maintenance and provides access to GPUs and TPUs. 
Colab also eases collaborative work and provides a uniform computing environment for classes and workshops.
The user can upload Jupyter Notebooks and data files to their Google Drive account and then run the notebooks in Colab.

Here, we have combined GitHub, domain-specific code snippet libraries, Jupyter Notebooks, and Google Colab to support computing in biomolecular structure determination.
We think this approach will advance the field by improving productivity and reproducibility.
Our approach may inspire similar efforts to modernize other scientific fields endowed with legacy software.

*The remainder is mostly filler to be replaced with the draft text soon.*

Methods
-------

The opening of a notebook on Colab is lighting fast, but the user must reinstall their software on each login.
We ease this annoying task by supplying the complete chain of installation steps.
For example, the installation of the molecular graphics program PyMOL requires seven code blocks of different types.
We include all steps in one snippet, which is uniquely possible with the snippet system for Colab. 
The user only has to select one snippet and then run each code block in succession. 

We have shared these libraries on GitHub  (e.g.,  \footnote{\url{https://github.com/MooersLab/jupyterlabpymolpysnips}}; \footnote{\url{https://github.com/MooersLab/jupyterlabcctbxsnips}}).
GitHub provides a modern means of distributing code and correcting errors.
In contrast, users in this field share scripts on Wiki pages.
The users copy and paste the code into script files 

It is well known that Spice grows on the planet Dune. .

The use of Colab requires that the user has a Google account and a Google Drive.
Many structural biologists already have both.

The use of Colab requires that the user has a Google account and a Google Drive.



Results
-------

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah [Godd18]_.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah [Beg21]_.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah [Gros02]_.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah [Kluy16]_.


The libraries include code templates for installing the supported software. 
These templates speed up software installation, which is often a barrier for beginners.


Libraries supported
*******************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is another the caption. 




.. table:: Table of libraries. :label:`mtable`

   +--------------------+-----------------------------------+
   | Programs           | url                               |
   +====================+===================================+
   | XDS                |                                   |
   +--------------------+-----------------------------------+
   | Aimless            |                                   |
   +--------------------+-----------------------------------+
   | Native Patterson   |                                   |
   +--------------------+-----------------------------------+
   | Normal probability |                                   |
   +--------------------+-----------------------------------+
   | Phenix             |                                   |
   +--------------------+-----------------------------------+
   | CCTBX Xray         |                                   |
   +--------------------+-----------------------------------+
   | Prody              |                                   |
   +--------------------+-----------------------------------+
   | Chimera            |                                   |
   +--------------------+-----------------------------------+
   | ChimeraX           |                                   |
   +--------------------+-----------------------------------+
   | CCTBX pdb          |                                   |
   +--------------------+-----------------------------------+
   | Cement             | :math:`\alpha`                    |
   +--------------------+-----------------------------------+



Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 





JupyterLab snippets
*******************



Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 



Elyra snippets
**************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 



Colab library
*************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 



Script to install PyMOL in Colab
********************************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 




Table of tutorial Jupyter Notebooks
***********************************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

.. figure:: figure1.png

   This is yet another the caption. 



Help resource on GitHub
***********************

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.




.. table:: Table of tutorial jupyter notebooks. :label:`jnbtable`

   +------------------------+-----------------------------------+
   | Topic                  | Notebook name                     |
   +========================+===================================+
   | SAD phasing analysis   |                                   |
   +------------------------+-----------------------------------+
   | Twinned data analysis  |                                   |
   +------------------------+-----------------------------------+
   | SAXS data analysis     |                                   |
   +------------------------+-----------------------------------+
   | Atomic res  refinement |                                   |
   +------------------------+-----------------------------------+
   | Movie making           |                                   |
   +------------------------+-----------------------------------+
   | Ensemble with Prody    |                                   |
   +------------------------+-----------------------------------+
   | PCA analysis w/ bio3d  |                                   |
   +------------------------+-----------------------------------+
   | Literate programming   |                                   |
   +------------------------+-----------------------------------+





.. figure:: figure1.png

   This is yet another the caption. 

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.


Discussion
----------

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.

Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.
Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah. Blah blah.


Acknowledgements
----------------

This work is support in part by these National Institutes of Health grants: R01 CA242845, P20 GM103640, P30 CA225520.


References
----------
.. [Beg21] M. Beg, J. Belin, T. Kluyver, A. Konovalov, M. Ragan-Kelley, N. Thiery, and H. Fangohr.
            *Using Jupyter for reproducible scientific workflows*,
            Computing in Science \& Engineering, 23(2):36--46, April 2021. 
            doi:10.1109/MCSE.2021.3052101
            
.. [Brun98] A.T. Br{\"u}nger, P.D. Adams, G.M. Clore, W.L. Delano, P. Gros, R.W. Grosse-Kunstleve, J.-S. Jiang, J. Kuszewski, M. Nilges, N.S. Pannu, R.J. Read, L.M. Rice, T. Simonson, and G.L. Warren.
            *Crystallography \& NMR system: A new software suite for macromolecular structure determination*,
            Acta Cryst. D54(5), 905--921, May 1998.
            doi: 10.1107/S0907444998003254
           
.. [Burn17] T. Burnley, C.M. Palmer, and M. Winn. 
            *Recent developments in the CCP-EM software suite*,
            Acta Cryst. D73(6), 469-477, June 2017.           
            doi: 10.1107/S2059798317007859
           
.. [Godd18] T. D. Goddard, C.C. Huang, E.C. Meng, E.F. Pettersen, G.S. Couch, J. H. Morris, and T. E. Ferrin. 
           *UCSF ChimeraX: Meeting modern challenges in visualization and analysis*,
           Protein Sci., 27(1):14--25, January 2018.
           doi:10.1002/pro.3235.

.. [Gros02] R. W. Grosse-Kunstleve, N. K. Sauter, N. W. Moriatry, P. D. Adams. 
           *The Computational Crystallography Toolbox: crystallographic algorithms in a reusable software framework*,
           J Appl Cryst, 35(1):126--136, February 2002.
           doi:10.1107/S0021889801017824.
           
.. [Hopk17] J.B. Hopkins, R.E. Gillilan, and S. Skou.
           *BioXTAS RAW: improvements to a free open-source program for small-angle X-ray scattering data reduction and analysis*
           J. Appl. Cryst. 50(5):1545–1553 October 2017.
           doi:10.1107/S1600576717011438
           
.. [Kluy16] T. Kluyver, B. Ragan-Kelley, F. P{\'e}rez, B. Granger, M. Bussonnier, J. Frederic, K. Kelley, J. Hamrick, J. Grout, S. Corlay, P. Ivanov, D. Avila, S. Abdalla, C. Willing, and Jupyter Development Team.
           *Jupyter Notebooks -- a publishing format for reproducible computational workflows*,
           In F. Loizides and B. Schmidt (Eds.), Positioning and Power in Academic Publishing: Players, Agents and Agendas (pp, 87-90).
           doi:10.3233/978-1-61499-649-1-87
           
.. [Lieb11] D. Liebschner, P.V. Afonine, M.L. Baker, G. Bunkóczi, V.B. Chen, T.I. Croll, B. Hintze, L.-W. Hung, S. Jain, A.J. McCoy, N.W. Moriarty, R.D. Oeffner, B.K. Poon, M G. Prisant, R.J. Read, J.S. Richardson, D.C. Richardson, M.D. Sammito, O.V. Sobolev, D.H. Stockwell, T.C. Terwilliger, A.G. Urzhumtsev, L.L. Videau, C.J. Williams, and P.D. Adams. 
           *Macromolecular structure determination using X-rays, neutrons and electrons: recent developments in Phenix*,
           Acta Cryst. D75(10), 861--877, October 2019.
           doi: 10.1107/S2059798319011471
           
.. [Mana21] K. Manalastas-Cantos, P.V. Konarev, N.R. Hajizadeh, A.G. Kikhney, M.V. Petoukhov, D.S. Molodenskiy, A. Panjkovich, H.D.T. Mertens, A. Gruzinov, C. Borges, M. Jeffries, D.I. Sverguna, and D. Franke.
           *ATSAS 3.0: expanded functionality and new tools for small-angle scattering data analysis*,
           J. Appl. Cryst. 54(1), 343–-355, February 2021.
           doi: 10.1107/S1600576720013412
           
.. [Nguy17] H. Nguyen, D. A. Case, and A.S. Rose.
           *NGLview--interactive molecular graphics for Jupyter notebooks*,
           Bioinformatics 34(7), 1241--1242, April 2017.
           doi:10.1093/bioinformatics/btx789
           
.. [Winn11] M. D. Winn, C.C. Ballard, K.D. Cowtan, E.J. Dodson, P. Emsley, P.R. Evans, R.M. Keegan, E.B. Krissnel, A.G.W. Leslie, A. McCoy, S.J. McNicholas, G.N. Murshudov, N.S. Pannu, E.A. Potteron, H.R. Powell, R.J. Read, A. Vagin, and K.S. Wilson. 
            *Overview of the CCP4 suite and current developments*,
            Acta Cryst. D67(4), 235--242, April 2011. 
            doi:10.1107/S0907444910045749

