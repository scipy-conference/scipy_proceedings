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

Here, we have combined GitHub, domain-specific code snippet libraries, Jupyter notebooks, and Google Colab to support computing in biomolecular structure determination.
We think this approach will advance the field by improving productivity and reproducibility.
Our approach may inspire similar efforts to modernize other scientific fields endowed with legacy software.

*The remainder is mostly filler to be replaced with the draft text soon.*

Methods
-------

We created snippet libraries for each structural biology package to support structural biology computations in Jupyter and Colab.
Any particular workflow is unlikely to require all of the libraries.
For example, a beginner’s workflow is unlikely to use CCTBX, a library of Python wrapped C++ routines for building molecular structure determination software.
Likewise, a cryo-electron microscopy workflow will not need XDS, a package for processing X-ray diffraction images.
We created a GitHub site for each library to ease the downloading of only those libraries that interest users (Table :ref:`libraries`).
This modularization of the project should ease the correction and augmentation of individual libraries as the extensions, and structural biology software packages evolve.
We only provided libraries for JupyterLab because the Jupyter Project plans to phase out support for the Jupyter Notebook software.
Among the several alternative extensions for code snippets in JupyterLab, we choose jupyterlab-snippets [jLsnip]_ and Elyra [Elyra]_ because these two extensions are actively maintained and have different features.
We also support a snippet library for Juptyer notebooks on Google Colab as described below because Colab provides access to GPUs, which can accelerate some of the larger computational tasks.


jupyterlab-snippets externsion
******************************

The jupyterlab-snippets extension adds a snippet menu to the JupyterLab menu bar.
The user accesses the snippets through a cascading pulldown menu.
Each snippet resides in a separate plain text file without any formatting.
This feature dramatically eases adding new snippets by users and eases keeping the snippets under version control.
The snippets are stored in the Jupyter data directory (which is found by entering :code:`jupyter --path`; it is in ~/Library/Jupyter/snippets).
Each snippet library is stored in a separate subfolder, which appears on the menu bar as a part of a cascading pulldown menu (Figure :ref:`snippetCascade`). 


.. figure:: snippetCascade.png 

   Cacading pull-down menu for the Juptyer categories of the jupyterlabpymolsnips library. :label:`snippetCascade`

We clustered snippets into categories. 
Each category has a cascading submenu.
Clicking on a snippet name in the submenu triggers its insertion into the current cell in the notebook.
The nested menu hierarchy serves well the user who is familiar with the content of the snippet libraries.

Like most other snippet extnesions for Jupyter Notebook and JupyterLab, the jupyterlab-snippets externsion does not support tab stops nor tab triggers.
These are common features of snippet libraries for most text editors and IDEs that accelerate the editing of parameter values in snippets.
The tab stops are particularly valuable because they direct the user to sites that may need changes in their parameter values and they guide the user to all of the site to ensure that none are overlooked.
The overlooking of parameter values that require changing can be a major source of bugs.
The tab triggers are also often mirrored, so a change at one instance of the same parameter will be propagate automatically to other identical instances of the parameter.
To compensate for the lack of tab triggers, we inlcude a second copy of the code in the same snippet but in a comment and with the tab triggers marked with curly braces and numbers (Figure :ref:`compareSnips`).
The user uses the code in the comment to direct their editing of the active code.
The user can delete the commented out comment when they have finished editing.
Separate versions of the libraries were made with commented out code.
These versions are distinguished by having "plus" appended to their names.

.. figure:: compareSnips.png 

   Comparison of active snippet at the bottom and commented out snippet at the top. The snippet at the top serves as a guide for editing because it has curly braces marking sites to be edited. :label:`compareSnips`


elyra-code-snippet extension
****************************

A menu icon labeled with `</>` provides access to snippets in the elyra-code-snippet-extension system.
After the icon is clicked, the snippets appear in the left margin of the JupyterLab GUI.
Snippets from all libraries appear in alphabetical order. 
The user can scroll through the list of snippets.
Hovering the mouse cursor over the snippet's name triggers the display of a description of the snippet.


.. figure:: hbondsElyra.png 

   This is yet another the caption. :label:`hbondElyra`

Alternatively, the user can enter a search term in the search box at the top of the menu to reduce the list of snippets.
The search terms can be part of a snippet name or a tag stored with each snippet.

A tag icon displays all of the available tags in the snippets as separate icons.
The user can select tags to be used to choose snippets by clicking on the icons.

Each snippet is displayed with several icons (Figure).
A triangular toggle can trigger the display of the snippet in a textbox.
A pencil icon enables the editing of the code.
Other icons enable copying the code to the clipboard, inserting code into the current cell in the notebook, and deleting the snippet. 

A plus sign in the upper-righthand corner opens a GUI for the creation of a new snippet.
The GUI occupies a new tab in the window that houses the Jupyter notebooks.
The GUI has a text box for each kind of metadata: name, description, tags, language, and the snippet code.
There is a save button at the bottom to add the new snippet to the current library.


.. figure:: newElyraSnip.png

   The GUI for the creation of new snippets. The Learn more link take the user to the documentation in Read-the-docs. :label:`newElyraSnip`


Each snippet is stored in a separate JSON file.
Each JSON file has the snippet code plus several rows of metadata, including a list of tags and the programming language of the snippet.
The latter provides a sanity check.
For example, an attempt to insert a C++ snippet into a notebook with an active Python kernel will trigger the opening of a window with a warning.

The directory ~/Library/Jupyter/metadata/code-snippets stores the snippet files on the Mac.
There are no subfolders for individual snippet libraries, unlike the jupyterlab-snippets externsion.
The snippets from multiple libraries are stored together in the code-snippets folder.
The tag system can be used to select all snippets from one library.
The tag system serves well the user who is not familiar with the content of the installed libraries.


Colab snippet library
***********************

The Colab snippet system resembles the Elyra snippet system in that search terms in a search box retrieve snippets.
However, the Colab system differs from the Elyra system ins that a single notebook stores all of the snippets. The user's Google Cloud account stores the notebook of snippets.
The use of Colab requires that the user has a Google account and a Google Drive.
Many structural biologists already have both.

Notebooks on Colab open lighting fast, but the user must reinstall their software on each login.
We ease this annoying task by supplying the complete chain of installation steps.
For example, the installation of the molecular graphics program PyMOL requires seven code blocks of different types.
Some involve the use of curl, and others use conda.
We include all steps in one snippet, which is uniquely possible with the snippet system for Colab. 
The user only has to select one snippet and then run each code block in succession.


Availability of the snippet libraries
*************************************

We have shared these libraries on GitHub  (e.g., Table (:ref:`libraries`)).
Each library is also archived in zenodo.


.. table:: Table of the snipppet libraries. :label:`libraries`

   +--------------------+-----------------------------------------------------------+
   | library            | url on GitHub                                             |
   +====================+===========================================================+
   | xds                | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+
   | cctbx              | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+
   | phenix             | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+
   | PyMOL              | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+
   | chimerax           | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+
   | prody              | https://github.com/MooersLab/JL-snippets-cctbxsnips       |
   |                    | https://github.com/MooersLab/elyra-cctbxsnips             |
   |                    | https://github.com/MooersLab/colab-cctbxsnips             |
   +--------------------+-----------------------------------------------------------+



Results
-------

The 





Structure determination and refinement workflows with Phenix
************************************************************

A team of professional software developers based at the Berkeley-Lawrence National Laboratory (BLNL) develops the Phenix software to refine protein crystal structures determined from X-ray diffraction data. 
The project includes several collaborators located around the world who develop auxiliary components of the package.
Paul Adams leads the team.
He had spent the 1990s developing the two very successful protein crystallography software packages: XPLOR and CNS.
Shortly after arriving in the Bay Area around 2000, Paul Adams was influenced by Warren Delano to use Python to wrap the Computational Crystallography Tool Box (CCTBX), which is written in C++.
Phenix uses CCTBX modules for intensive computations.
(Warren Delano was the developer of the PyMOL, a molecular graphics program that was written in C and wrapped with Python.)
While Python eases the use of CCTBX, mastery of CCTBX requires at least an intermediate level of Python programming skills.
On the other hand, Phenix is easy to use via the command line or a GUI.

The Phenix project greatly eased the incorporation of simulated annealing into crystal structure refinement by hiding the tedious preparation of the required parameter files from the user.
The PDB file does not have sufficient information about chemical bonding for MD simulations.
The molecular dynamics software that carries out the simulated annealing requires two parameter files and the coordinate file.
The preparation and debugging of the parameter files manually take many hours, but Phenix automates this takes.

Simulated annealing involves molecular dynamics simulation at high temperatures to move parts of a molecular model out of local energy minima and into conformations that fit the experimental data better.
Twenty minutes of applying simulated annealing to an early model that still has numerous errors can significantly improve the model while saving the user a day or more of the tedious manual rebuilding of the molecular model. 

More recently, Phenix has been extended to refine crystal structures with neutron diffraction data and for structure determination and refinement with cryo-electron microscopy (cryo-EM) data.
The addition of support for cryo-EM help address the recent need for the ability to fit atomic models to cryo-EM maps that have recently become available at near atomic resolution because of the dramatic imprvoements in detector technology []_.
Users can interact with Phenix via a GUI interface or the command line, as mentioned before, but users can also use PHIL, domain-specific language scripting language for more precise parameter settings for Phenix. 
In addition, users can use the :code:`phenix.python` interpreter. Unfortunately, the phenix.python interpreter is still limited to Python2, whereas CCTBX has been available for Python3 for over a year.

Jupyter Lab and its extensions are also best run with Python3.
The most practical approach to using Phenix in Jupyter Lab is to invoke Phenix by utilizing the shell rather than using Python.
For example, the command shown below invokes statistical analysis of the B-factors in a Protein Data Bank (PDB) file by using one line of code in the shell (Figure :figure:XXXXXX).
The PDB file uses a legacy, fixed-format file for storing the atomic coordinates and B-factors of crystal structures.
The B-factors are a measure of the atomic motion in individual atoms in a protein structure. 
The PDB file format was defined and popularized by the Protein Data Bank, a repository for atomic coordinates and structural data that has over 170,000 entries from around the world. 
The PDB was started in 1972 and unified with the branches in Japan and Europe in 2003 as the wwPDB [ ]. 
The wwPDB continues to play a central role in promoting the principles of open science and reproducible research in structural biology.

Since 2019, the wwPDB requires the PDBx/mmCIF format for new depositions [Adam19]_.
Many structural biology software packages now have the ability to read files in the PDBx/mmCIF format.

.. code-block:: bash

    !phenix.b_factor_statistics 1lw9.pdb 


The output form this command is printed below the cell that invokes the command. 
Some of the output is shown below.

.. code-block:: bash
    
    Starting phenix.b_factor_statistics
    on Wed Jun  2 04:49:01 2021 by blaine
    
    Processing files:
    
      Found model, /Users/blaine/pdbFiles/1lw9.pdb
    
    Processing PHIL parameters:
    
      No PHIL parameters found
    
    Final processed PHIL parameters:
    
      data_manager {
        model {
          file = "/Users/blaine/pdbFiles/1lw9.pdb"
        }
        default_model = "/Users/blaine/pdbFiles/1lw9.pdb"
      }
    
    
    Starting job
    Validating inputs
                    min    max   mean <Bi,j>   iso aniso
       Overall:    6.04 100.00  24.07    N/A  1542     0
       Protein:    6.04 100.00  23.12    N/A  1328     0
       Water:      9.98  55.93  30.47    N/A   203     0
       Other:     14.11  35.47  21.10    N/A    11     0
       Chain  A:   6.04 100.00  24.07    N/A  1542     0
       Histogram:
           Values      Number of atoms
         6.04 - 15.44       309
        15.44 - 24.83       858
        24.83 - 34.23       187
        34.23 - 43.62        78
        43.62 - 53.02        32
        53.02 - 62.42        16
        62.42 - 71.81         8
        71.81 - 81.21         6
        81.21 - 90.60         2
        90.60 - 100.00       46
    
    Job complete
    usr+sys time: 1.92 seconds
    wall clock time: 2.93 seconds


There are several dozen commands that can be run via the shell and return useful output that can be captured in one Jupyter Notebook rather than in dozens of log files.
The output can be copied and pasted into a new cell and then reformatted in markdown as a table or the copied output be used as input data to make a plot with matplotlib.
While these are basic data science tasks, they are intimidating to new users of Jupyter and some of the details are easy for more experienced users to forget.
To overcome this problem, we supply snippets that demonstrate how to transform the output and that can be used as templates by the users.  

These commands are becoming harder to find as the on-line documentation has been migrating to serving only the GUI interface.
The bash script files that run the phenix commands can be found by running 

.. code-block:: bash

    !ls /Applications/phenix-*/build/bin/phenix.\*

These shell scripts invoke Python scripts that capture the command line arguments and pass them to the phenix Python interpreter.


.. code-block:: bash

    ls /Applications/phenix-1.19.2-4158/modules/phenix/phenix/command_line/*.py.










Libraries supported
*******************

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

.. figure:: figure1.png

   This is yet another the caption. 


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

.. [Adam19] P.D. Adams, P.V. Afonine, K. Baskaran, H.M. Berman, J. Berrisford, G. Bricogne, D.G. Brown, S.K. Burley, M. Chen, Z. Feng, C. Flensburg, A. Gutmanas, J.C. Hoch, Y. Ikegawa, Y. Kengaku, E. Krissinel, G. Kurisu, Y. Liang, D. Liebschner, L. Mak, J.L Markley, N.W. Moriarty, G.N. Murshudov, M. Noble, E. Peisach, I. Persikova, B.K. Poon, O.V. Sobolev, E.L. Ulrich, S. Velankar, C. Vonrhein, J. Westbrook, M. Wojdyr, M. Yokochi, and J.Y. Young.
            *Announcing mandatory submission of PDBx/mmCIF format files for crystallographic depositions to the Protein Data Bank (PDB)*,
            Acta Crystallographica Section D: Structural Biology, 75(4):451--454, April 2019.
            doi:10.1107/S2059798319004522

.. [Adam21]

            
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
            
.. [Elyra]  https://github.com/elyra-ai/elyra/blob/master/docs/source/getting_started/overview.md 
           
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
           
.. [jLsnip] https://github.com/QuantStack/jupyterlab-snippets
           
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

