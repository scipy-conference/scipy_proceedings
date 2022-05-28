:author: Blaine H. M. Mooers 
:email: blaine-mooers@ouhsc.edu
:institution: Dept of Biochemistry and Molecular Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Stephenson Cancer Center, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Laboratory of Biomolecular Structure and Function, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Biomolecular Structure Core, Oklahoma COBRE in Structural Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:orcid: 0000-0001-8181-8987
:corresponding: Blaine H. M. Mooers

:author: Francis A. Acquah
:institution: Dept of Biochemistry and Molecular Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:orcid: 0000-0002-4534-9156




------------------------------------------------------------------------
Amalgamated molecular visualization in Colab (Warning: still assembling)
------------------------------------------------------------------------

.. class:: abstract

Protein crystallography produces most of the protein structures used in structure-based drug design.
The process of protein structure determination is computationally intensive and error-prone because many software packages are involved.
Here, we attempt to support the reproducibility of this computational work by using Jupyter notebooks to document the decisions made, the code, and selected output.
We have made libraries of code templates to ease running the crystallography packages in Jupyter notebooks when editing them with JupyterLab or Colab.
Our combined use of GitHub, snippet libraries, Jupyter notebooks, JupyterLab, and Colab will help modernize the computing done by structural biologists.

.. class:: keywords

   literate programming, reproducible research, scientific rigor, electronic notebooks, Colab, computational structural biology,  biomolecular structure, biomedical research, protein-drug interactions, RNA-drug interactions, molecular graphics, molecular visualization, scientific communication, molecular artwork, computational molecular biophysics


Introduction
------------

Chemists, biochemists, and pharmacologists frequently need to visualize the structures of both small and large molecules with molecular graphics software. 
Historically, this software has been expensive and proprietary. 
However, open-source alternatives are now widely available. 
In addition, several Python packages support molecular visualization in Jupyter notebooks (e.g., rdkit, nglview, py3dmol, and PyMOL). 
The first package supports 2-D views of small molecules. 
The following two packages provide interactive 3-D views of small molecules and proteins. 
The fourth package provides static images of 3-D molecular scenes that are of publication quality. 
The first and last packages also support numerous kinds of structural analysis in addition to molecular visualization. 
They were also designed to be run on local computers via desktop applications.

However, users can now run these packages in Google Colaboratory, Google Colab, or just Colab [Carn18]_ [Cola21]_. 
Colab provides a universally accessible computing platform. 
Universal access to this cloud platform eliminates the problem of installing software on heterogeneous personal computers. 
This feature of Colab can save time and frustration for both instructors and students in laboratories, classrooms, and workshops [Nelson20]_. 
The user interacts with the cloud computing resources by interacting with an electronic computational notebook known as the Colab Notebook, which is related to the Jupyter Notebook. 
The latter has become the de facto computing platform in scientific computing.

The Colab notebook provides access to generic scientific computing packages preinstalled on Colab. 
This software includes widely used numerical computing packages like NumPy and pandas. 
If the required software is already available on Colab, the user can import the packages or modules and start executing code blocks in the notebook. 
Unfortunately, molecular graphics packages are not part of this software collection.

One solution to this issue is to install this software via Anaconda. 
However, Anaconda must be retrieved and installed in the correct location in Colab. 
These software installation steps are complex and take time if executed individually. 
This complexity can discourage new users.

To address this and other problems with adding software to Colab, we created code fragments that reduce these tasks to a single click of an icon by the user. 
We added these code fragments to snippet libraries for each molecular graphics package. 
The new libraries are called colabrdkit, colabpy3dmolsnips, and colabnglviewsnips. 
In the case of PyMOL, we reformatted the published pymolpysnips library for Colab and named it colabpymolpysnips [Moo21a]_.

The useer-installed software is lost when the user logs out of Colab. 
This loss is inconvenient because the installation step takes 6-8 minutes. 
A second code fragment archives the installed software as a tar file and stores the tar file on the user's Google Drive. 
A third fragment retrieves the tar file and unpacks it in the correct location after the user logs into Colab again. 
This third step takes less than a minute and reduces the inconvenience of reinstalling software at the beginning of each Colab work session.

We combine the use of GitHub repositories, domain-specific code snippet libraries, and Google Colab to support polyglot molecular visualization in the cloud, where the user can combine the powerful features of several software packages in a single computational notebook. 
The code generating the images resides in the notebook, thereby supporting reproducible research. 
We think this approach will advance the field by improving productivity and reproducibility.


Methods
-------

We created snippet libraries for each package to support molecular visualization in Colab.
We created a GitHub site for each library to ease the downloading of only those libraries that interest users.
This modularization of the project should ease the correction and augmentation of individual libraries as the extensions and molecular graphics software packages evolve.

Code snippet for installing PyMOL on Colab 
******************************************

The <> icon opens a menu on the left side that lists all of the snippets. 
The search term 'pymol' was used to reduce the list of candidate snippets. The highlighted snippets name 'Install PyMOL is new Colab notebook'. 
Selecting this snippets opens the snippet below. The snippet description is displayed followed by the seven blocks of code. 
The description includes the seven steps for installing the molecular graphics programs. 
Clicking with the mouse cursor on 'INSERT' in blue inserts the code into in the cells on the notebook on the fight.

A search box at the top of the list of snippets is used to recover a snippet.
The user enters a snippet name in the search box to display the snippet and its documentation.
The user hits the 'Install' button to install the snippet's code at the current position in the  notebook.

The list snippet for a library will print in a table below the current cell a list of the snippets in the library and a brief description. 
This table is stored in a pandas DataFrame that can be searched with the pandas search function.
This table can also be searched for key terms with the search function in the notebook.
The code block and output can be hidden by clicking on the three blue dots on the left margin of the cell. 

Notebooks on Colab open very quickly, but the user must reinstall their software on each login.
We ease this task by supplying the complete chain of installation steps.
For example, the installation of the molecular graphics program PyMOL requires seven code blocks of different types.
Some involve the use of curl, and others use the conda package management system.
We include all steps in one snippet, which is uniquely possible with the snippet system for Colab. 
The user only has to select one snippet and then run each code block in succession.

The use of Colab requires that the user has a Google account and a Google Drive.
Many structural biologists already have both.


Notebooks with sample workflows
*******************************

We created a library of Colab Notebooks with sample workflows.
This library of notebooks is only representative and not exhaustive because the combinatorial explosion of possible workflows makes covering all workflows impractical.
These notebooks can serve as templates for the creation of new notebooks and are available on our GitHub site [MLGH]_.

Availability of the snippet libraries
*************************************

We have shared these libraries on GitHub [MLGH]_.
Each library is also archived in zenodo.


Results
-------

We describe here a set of libraries of code templates to support 


Discussion
----------


Acknowledgements
----------------

This work is support in part by these National Institutes of Health grants: R01 CA242845, P20 GM103640, P30 CA225520, P30 AG05091.


References
----------

.. [Bias13] M. Biasini, T. Schmidt, S. Bienert, V. Mariani, G. Studer, J. Haas, N. Johner, A. D. Schenk, A. Philippsen, and T. Schwede. 
            *OpenStructure: an integrated software framework for computational structural biology*,
            Acta Cryst. D69(5):701–709, May 2013.
            doi: 10.1107/S0907444913007051
            
.. [Carn18] T. Carneiro, R. V. M. Da N{\'o}brega, T. Nepomuceno, G.-B. Bian, V. H. C. De Albuquerque and P. P. Reboucas Filho.
            *Performance analysis of google colaboratory as a tool for accelerating deep learning applications*,
            IEEE Access 6:61677-61685, November 2018.
            doi: 10.1109/ACCESS.2018.2874767
            
.. [Cola21] https://colab.research.google.com
            
           
.. [Godd18] T. D. Goddard, C.C. Huang, E.C. Meng, E.F. Pettersen, G.S. Couch, J. H. Morris, and T. E. Ferrin. 
           *UCSF ChimeraX: Meeting modern challenges in visualization and analysis*,
           Protein Sci., 27(1):14-25, January 2018.
           doi: 10.1002/pro.3235.
           
.. [Gran21] B. E. Granger and F. Pérez.
           *Jupyter: Thinking and Storytelling With Code and Data*,
           Computing in Science & Engineering, 23(2):7-14, March-April 2021.
           doi: 10.1109/MCSE.2021.3059263
           
.. [Kluy16] T. Kluyver, B. Ragan-Kelley, F. P{\'e}rez, B. Granger, M. Bussonnier, J. Frederic, K. Kelley, J. Hamrick, J. Grout, S. Corlay, P. Ivanov, D. Avila, S. Abdalla, C. Willing, and Jupyter Development Team.
           *Jupyter Notebooks -- a publishing format for reproducible computational workflows*,
           In F. Loizides and B. Schmidt (Eds.), Positioning and Power in Academic Publishing: Players, Agents and Agendas (pp, 87-90).
           doi: 10.3233/978-1-61499-649-1-87
           
.. [jLsnip] https://github.com/QuantStack/jupyterlab-snippets
           
.. [Mott10] S. E. Mottarella, M. Rosa, A. Bangura, H. J. Bernstein, and P. A. Craig.
           *Conscript: RasMol to PyMOL script converter*,
           Biochem. Mol. Biol. Educ., 38(6):419-422, November 2010.
           doi: 10.1002/bmb.20450
           
.. [MLGH]   https://github.com/MooersLab
           
.. [Moo21a] B. H. M. Mooers and M. E. Brown.
           *Templates for writing PyMOL scripts*,
           Pro. Sci., 30(1):262-269, January 2021.
           doi: 10.1002/pro.3997

.. [Moo21b] B. H. M. Mooers.
           *A PyMOL snippet library for Jupyter to boost researcher productivity*,
           Computing Sci. \& Eng., 23(2):47-53, April 2021.
           doi: 10.1109/mcse.2021.3059536
           
.. [Nelson20] M. J. Nelson and Amy K. Hoover
           *Notes on using Google Colaboratory in AI education*,
           ITiCSE '20: Proceedings of the 2020 ACM conference on innovation and Technology in Computer Science Education, 533-534, June 2020. 
           doi: 10.1145/3341525.3393997

.. [Nguy17] H. Nguyen, D. A. Case, and A.S. Rose.
           *NGLview--interactive molecular graphics for Jupyter notebooks*,
           Bioinformatics, 34(7):1241-1242, April 2017.
           doi: 10.1093/bioinformatics/btx789
           
.. [PyMO21] https://pymol.org/2/
           
.. [Rese20] https://blog.jupyter.org/reusable-code-snippets-in-jupyterlab-8d75a0f9d207
            
.. [SciP20] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. {van der Walt}, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J.Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, I. Polat, Y. Feng, E. W. Moore, J. {VanderPlas}, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. {van Mulbregt}, Paul and {SciPy 1.0 Contributors}.
           *{{{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in Python}}*,
           Nature Methods, 17(3):261-272, February 2020.
           doi: 10.1038/s41592-019-0686-2

