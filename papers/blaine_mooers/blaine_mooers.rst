:author: Blaine H. M. Mooers
:email: blaine-mooers@ouhsc.edu
:institution: Dept of Biochemistry and Molecular Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Stephenson Cancer Center, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Laboratory of Biomolecular Structure and Function, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:institution: Biomolecular Structure Core, Oklahoma COBRE in Structural Biology, University of Oklahoma Health Sciences Center, Oklahoma City, OK 97104
:orcid: 0000-0001-8181-8987
:corresponding: Blaine H. M. Mooers


=======================================================================
 Biomolecular Crystallographic Computing with Jupyter
=======================================================================

.. class:: abstract

   The ease of use of Jupyter notebooks has helped biologists enter scientific computing,
   especially in protein crystallography, where a collaborative community develops extensive
   libraries, user-friendly GUIs, and Python APIs. The APIs allow users to use the libraries in Jupyter.
   To further advance this use of Jupyter, we developed a collection of code fragments that use
   the vast Computational Crystallography Toolbox library for novel analyses. We made versions
   of this library for use in JupyterLab and Colab. We also made versions of the snippet library
   for the text editors VS Code, Vim, and Emacs that support editing live code cells in Jupyter
   notebooks via GhostText. Readers of this paper may be inspired to adapt this latter capability
   to their domains of science.

.. class:: keywords

   literate programming, reproducible research, scientific rigor, electronic notebooks, JupyterLab, Jupyter notebooks, computational structural biology, computational crystallography, biomolecular crystallography, protein crystallography, biomolecular structure, biomedical research, protein*drug interactions, RNA*drug interactions, molecular graphics, molecular visualization, scientific communication, molecular artwork, computational molecular biophysics

Introduction
================

Biomolecular crystallography involves the determination of the molecular structure of proteins and nucleic acids and their complexes by using X-rays, neutrons, or electrons.
The molecular structures determines the protein's biological function, so the experimentally determined structures provide valuable insights vital for understanding biology and to developing new therapies in medicine.
The recent *resolution revolution* in cryo-EM and the breakthrough in protein prediction with deep learning models now provide complimentary sources of insights to biomolecular structure, but the crystallographic approach continues to remain vital because it still supplies the most precise structures [Kulhbrandt14]_ [Jumper2021]_.

Biological crystallographers are familiar with Jupyter's browser-based environment and interactive cells, especially after ColabFold enabled running AlphaFold2 from Colab notebooks [Mirdita21]_.
Nonetheless, most protein crystallographers continue to use well-developed, user friendly GUIs to run crystallographic software in routine analyses.
However, these users sometimes need non-routine analyses that require new code.

The Computational Crystallography Toolbox (CCTBX) provides a vast library of computational crystallography software written in C++ and wrapped with Python [Gros02]_.
This library is used to build new analysis tools.
CCTBX was hard to install three years ago due to its complex dependencies, but the addition of the CCTBX package to Anaconda dramatically eased the installation of CCTBX.
The lowering of this barrier to the installation of CCTBX has raised interest in the use of CCTBX for novel structure analyses.
Nonetheless, many users still find CCTBX to be impenetrable.
Several online tutorials and workshops have addressed this problem, but the adoption of CCTBX remains low.

To ease the use of CCTBX by my and other labs to develop custom crystallographic Analyses0, we assembled a collection of CCTBX code snippets for use in Jupyter notebooks.
Jupyter provides an excellent platform for exploring the CCTBX library and developing new analysis tools.
The Python API of CCTBX simplifies running CCTBX in Jupyter via a kernel specific for its conda environment.
We formatted the snippet library for several snippet extensions for the Classic Notebook and for Jupyter Lab.
To overcome the absence of tab triggers in the Jupyter ecosystem to invoke the insertion of snippets, we also made the snippets available for leading text editors.
The user can use the GhostText browser plugin to edit the contents of a Jupyter cell in a full-powered external editor.
GhostText enables the user to experience the joy interactive computing in Jupyter while working from the comfort of their favorite text editor.


Results
=========


jupyterlabcctbxsnips
************************

To ease the running of cctbx in Jupyter notebooks, we developed the jupyterlabcctbxsnips of code templates. Access to the code templates or snippets requires the editing of the Jupyter notebook from inside of JupyterLab , a browser based IDE for Jupyter notebooks. This JupyterLab enables the writing or editing of a document in a pane next to the Jupyter notebook. This is useful for writing up documentation, protocols, tutorials, blog posts, and manuscripts next to the notebook that is being described. The document can be plain text, html, markdown, or LaTeX.

The figure below shows part of the cascading menus for the cctbx library after it has been installed successfully. The submenus correspond to the names of subfolders in the cctbx folder in the multimenus_snippets folder, which you create inside of the Jupyter folder in your local library folder (i.e., ~/Library on the Mac). Each ultimate menu item is a Python snippet file. The selection of a snippet file by clicking on it with the left-mouse button inserts its content into a new cell below the current cell. The *millerArrayFromMtz.py* snippet at the bottom of the pulldown menu was selected and inserted in the figure below. Commented lines have text that describes what this snippet does. The code in this cell would be executed by entering Shift-Enter.

.. figure:: ./figs/Fig1Pulldown.png
   :align: center
   :scale: 40%
   :figclass: bht

   The cascading menus for the cctbx library.

The mtzObjectSummary.py snippet prints a summary of an mtz file.
The data in this mtz has columns of I(+) and I(-).
We use these data to make a I(+) vs I(-) scatter plot below.
The mtz file contains data for SirA-like protein (DSY4693) from Desultobacterium hafniense, Northeast Structural Genomics Consortium Target DhR2A.

.. figure:: ./figs/Fig5mtzSummary.png
   :align: center
   :scale: 50%
   :figclass: bht

   The output from millerArrayFromMtz.py snippet. 

The I(+) vs I(-) plot below was made after reading the X-ray data into a cctbx Miller array, a data structure designed for X-ray data. The I(+) and I(-) were eventually read into separate lists. We plot the two lists against each other in a scatter plot. This plot was adapted from an example in the ReciprocalSpaceship project from the Hekstra Lab. This new project takes a more Pythonic approach. For example, it uses the Pandas package to manage diffraction data whereas cctbx uses a special C++ data structure for diffraction data.

.. figure:: ./figs/Fig2IpImPlot.png
   :align: center
   :scale: 50%
   :figclass: bht

   The Ip Im plot.

CCTBX is most easily installed into its own environment by using Anaconda with the command conda create -n my_env -c conda-forge cctbx-base python=3.8.


jupyterlabcctbxsnipsplus
******************************

This is the variant of the jupyterlabcctbxsnips library with comments to guide editing of the snippets.

taggedcctbxsnips
*********************



colabcctbxsnips
*******************


On Colab, the snippets are stored in a Google Colab notebook. See this website for an excellent introduction to CCTBX (Computational Crystallography Toolbox). The colabcctbxsnips library is a collection of the code fragment to aid doing routine and not so routine computational tasks in protein crystallography. The URL for the snippets notebook is unused to access the snippets from a new notebook.

Click on the blue button below to open the notebook on Colab and follow the instructions at the top of the notebook on how to copy the notebook to Google Drive and then make it available to new Colab notebooks. This step has to be done only once. The snippets will be available on your next log-in; however, files and software installed on Colab with not be available on your next login to Colab.



cctbxsnips for text editors
******************************


Using GhostText to edit Jupyter cells from a favorite text editor
***************************************************************************

The snippet extensions for the Classic Jupyter Notebook and JuptyerLab lack support for tab triggers to insert snippets was you type and tab stops inside the snippet to advance to sites in the snippet that may need to be edited.
These two features are standard in the software that supports the use of snippet libraries in most text editors.
By adding the GhostText extension to the web browser and a server to one of several leading text editors, it is possible to send the text from the browser through a WebSocket to a server in the text editor.
Thus, it is possible to edit the contents of a computational notebook cell from inside a text editor.
Changes made in the text editor instantly appear in the notebook and vice versa.
By applying the power of a text editor to computational notebooks, experienced developers can continue to use familiar editing commands and tools.

As a quick reminder, tab triggers in text editors insert chunks of computer code after the user enters the tab trigger name and hits the TAB key.
The tab trigger name can be as short as several letters.
Many text editors and IDEs have pop-up menus that aid the selection of the correct tab trigger.
Tab stops are sites within the code snippet where the cursor advances to after entering TAB again.
These sites often have placeholder values that can be either accepted by entering TAB again or edited.
Sites with identical placeholder values can be mirrored so that a change in value at one site is propagated to the other tab stops with the same placeholder value.
The absence of tab stops can increase the number of bugs introduced by the developer by overlooking parameter values in the code snippet that need to be changed to adapt the snippet to the current program.
The lack of support for tab triggers and tab stops in computational notebooks can dampen the enthusiasm of experienced developers for using computational notebooks.
Of course, one solution is to write an extension for Jyptyer Notebook or JuptyerLab in JavaScript that supports tab triggers and tab stops.

Another approach is to send the text in the active code cell to a powerful text editor on your local computer via the browser extension known as GhostText https://ghosttext.fregante.com/.
GhostText is a Javascript program developed by Federico Brigante, a prolific JavaScript developer. Versions of the extension are available for the Google Chrome, Firefox, Edge, Opera, and Safari. The extension for the Google Chrome browser works in the Brave browser, and the extension for Firefox works in the Waterfox and Icecat browsers.

The text editor also needs to be extended with a server that enables two-way communication with the web page via a WebSocket.
Edits made on the browser side of the WebSocket are immediately sent to an open page in the Text Editor and vice versa; however, the text editor's snippets and other editing tools only work in the text editor.
The connection can be closed from either side of the WebSocket.
It is closed on the web browser side via an option in GhostTest’s pulldown menu, and it closed on the text editor side by closing the active buffer.

.. Figure:: ./figs/tabtrigger.png
   :align: center
   :scale: 53%
   :figclass: bht

   Example of a tab trigger being entered in Sublime Text 3 editor and appearing in a Jupyter Notebook cell. A pop-up menu lists the available snippets.  The list was narrowed to one snippet by the entry of three letters.

A Jupyter notebook with two code snippets from the cctbxsnips library for the Sublime Text 3 editor are shown in Figure 2.
The two code cells have been run, and the output from the second cell is printed in the notebook.
The first cell is being edited to change the name of the mtz file that is to be read.
A pop-up menu in Jupyter has appeared with a list of candidate file names.

.. Figure:: ./figs/twoCodeCells.png
   :align: center
   :scale: 70%
   :figclass: bht

   Two code cells with executed Jupyter code cells.

The servers for the editors are editor specific.
The following text editors are supported: Visual Studio Code, Vim, NeoVim, Emacs, Atom, and Sublime Text 3. GhostText was developed initially for Sublime Text 3, so Sublime Text 3 can serve as a positive control even if another editor in the list is your favorite editor. (Sublime Text 3 is available for most platforms for a free trial period of infinite length.)
For example, the server for Emacs is provided by the atomic-chrome package that is available in the Milkypostman’s Emacs Lisp Package Archive (MELPA) and on GitHub (https://github.com/alpha22jp/atomic-chrome).
The configuration for atomic-chrome in my Emacs initialization file (e.g., init.el) is listed in code listing 1 below.

The third line in Code listing 1 sets the default Emacs mode (equivalent to a programming language scope): I set it to Python for Jupyter code cells.
Atomic-chrome uses text-mode by default. You can change the default mode to other programming languages that you may use inJupyter, like Julia or R.
The last three lines specify the Emacs mode to be used when text is imported from the text areas of webpages on github.com, Overleaf.com, and 750words.com.
Similar configuration options are available in the other text editors, or you manually change the language scope for the window with the text imported from Jupyter.

.. Figure:: ./figs/atomicChrome.png
   :align: center
   :scale: 92%
   :figclass: bht

   Emacs lisp to configure atomic-chrome.

Ghost Text provides a keyboard shortcut for the browser to open or close the connection to the text editor.
These shortcut keep the developer's hands on the keyboard and avoid breaks in context by moving the hand to the mouse.
The shortcut by operating system is as follows: macOS, command-shift-K; Linux, control-shift-H; and Windows, control-shift-K.

I have been using GhostText daily since mid-May 2022 with Emacs and either Google Chrome or Firefox; I have infrequently encountered three difficulties.
First, other servers inside Emacs can occupy the port for GhostText and block the atomic-chrome server.
I have had to kill the offending server or restart Emacs. Second, saving the text in the Emacs buffer to a file can cause the text to become out of sync with the text in the editor and on the web page.
The lack of updating can lead to a loss of work; making intermediate saves of the text on the web server is safer than saving to a local file.
Of course, the final text can be saved locally.
Third, multiple open editors with GhostText servers installed can compete for the same WebSocket.
This problem is solved by closing the editor that is not in current use or configuring its server to use an alternate WebSocket.

To support the use of GhostText to edit electronic notebooks containing code from the CCTBX library, we have made variants of a collection of CCTBX snippets (Mooers 2021) for Visual Studo Code, Atom, Sublime Text 3, Vim, NeoVim, and Emacs. For Vim and NeoVim, the snippets are available for the UltiSnips, Snipmate, and neosnippets plugins. The snippets are available for download on GitHub (https://github.com/MooersLab/MooersLab/blob/main/README.md#cctbxsnips-for-editors).
I found that Sublime Text 3 had the most effortless setup while Emacs provided the highest degree of customization.
This snippet library was previously only available for use Juptyer notebooks via extensions for the Classic Juptyer Notebook application or Juptyer Lab.

Note that the snippet library cannot be used with the program nteract (https://nteract.io/).
The nteract is an easy-to-install and use desktop application for editing and running Jupyter notebooks offline.
The ease of installation makes nteract popular with new users of Jupyter notebooks. Obviously interact is not browser-based, so it cannot work with GhostText.
Interact has yet to be extended to support the use of code snippet libraries.

While the focus of this report was on the use of the GhostText browser plugin to edit Juptyer notebooks hosted in a web browser, the cctbxsnips snippet library can be used to aid the development of Python scripts in plain text files, which have the advantage of saner version control.
The snippets can also be used in other kinds of literate programming documents that operate off-line like org-mode files in Emacs and the Quarto (http://quarto.org) markdown representation of Jupyter notebooks.
Quarto is available for several leading text editors.
In the later case, you may have to extend the scope of the editing session in the editor to include Python source code.


Discussion
=============


What is new
**************

We report a set of code template libraries for doing biomolecular crystallographic computing in Jupyter.
These template libraries only need to be installed once because they persist between logins.
These templates include the code for installing the software required for crystallographic computing.
These installation templates save time because the installation process involves as many as seven operations that would be difficult to remember.
Once the user adds the installation code to the top of a given notebook, the user only needs to rerun these blocks of code upon logging into Colab to be able to reinstall the software.
The user can modify the installation templates to install the software on their local machines.
Examples of such adaptations are provided on a dedicated GitHub web page.
The template libraries presented here lower an important barrier to the use of Colab by those interested in crystallographic computing on the cloud.


Relation to other work with snippet libraries
***************************************************


To the best of our knowledge, we are the first to provide snippet libraries for crystallographic computing.
This snippet library is among the first that is domain specific.
Most snippet libraries are for programming languages or for hypertext languages like HTML, markdown and LaTeX.
The average snippet also tends to be quite short and the size of the libraries tends to be quite small.
The audience for these libraries are millions of professional programmers and web page developers.
We reasoned that this great tool should be brought to the aid of the thousands of workers in crystallography.

The other area where domain specific snippets have been provided is in molecular graphics.
The pioneering work on a scripting wizard provided templates for use in the molecular graphics program RasMol [Hort99]_.
The conscript program provided a converter from RasMol to PyMOL [Mott10]_.
We also provided snippets for PyMOL, which has 100,000 users, for use in text editors [Moo21a]_ and Jupyter notebooks [Moo21b]_.
The former support tab triggers and tab stops; the latter does not.

We have also worked out how to deploy this snippet libraries in OnDemand notebooks at High-Performance Computing centers.
These notebooks resemble Colab notebooks in that JupyterLab extensions cannot be installed.
However, they do not have any alternate support for accessing snippets from menus in the GUI.
Instead, we had to create IPython magics for each snippet that load the snippet's code into the code cell.
This system would also work on Colab and may be preferred by expert users because the snippet names used to invoke magic are under autocompletion.
That is, the user enters the start of a name and IPython suggests the remainder of the name in a pop-up menu.
We offer a variant library that inserts a commented out copy of the code that has been annotated with the sites that are to be edited by the user.



Opportunities for interoperability
**************************************

The set of template libraries can encourage synergistic interoperability between software packages supported by the snippet libraries.
That is the development of notebooks that use two or more software packages and even programming languages.
More general and well-known examples of interoperability include the Cython packages in Python that enable the running of C++ code inside Python, the reticulate package that enables the running of Python code in R , and the PyCall package in Julia that enables the running of the Python packages in Julia.
The latter package is widely used to run matplotlib in Julia.
Interoperability already occurs between the CCP4 [Agirre2023]_, clipper [McNicholas2018]_, gemmi [Wojdyr2022]_ and CCTBX projects and to a limited extent between CCTBX and PyMOL, but interoperability could be more widespread if the walls around the software silos were lowered.
The snippet libraries provided here can prompt interoperability on Colab by their proximity on Colab.



Acknowledgments
======================

This work was supported by the Oklahoma Center for the Advancement of Science and Technology: HR20-002, the  National Institutes of Health grants: R01 CA242845, P30 CA225520, and P30 AG050911-07S1. In particular, we thank the Biomolecular Structure Core of the COBRE in Structural Biology (PI: Ann West, P20 GM103640, P30 GM145423).


..


References
==============

.. [Kulhbrandt14] W. Kuhlbrandt.
            "he resolution revolution",
            *Science* 343:1443-1445, March 2014.
            doi:10.1126/science.1

.. [Jumper2021] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Zidek, A. Potapenko, A. Bridgland, C. Meyer, S. A. A. Kohl, A. J. Ballard, A. Cowie, B. Romera-Paredes, S. Nikolov, R. Jain, J. Adler, T. Back, S. Petersen, D. Reiman, E. Clancy, M. Zielinski, M. Steinegger, M. Pacholska, T. Berghammer, S. Bodenstein, D. Silver, O. Vinyals, A. W. Senior, K. Kavukcuoglu, P. Kohli and D. Hassabis.
            ""Highly accurate protein structure prediction with AlphaFold,
            *Nature* 596:583–589, July 2021.
            doi:10.1038/s41586-021-03819-2

.. [Mirdita21] M. Mirdita, K. Schutze, Y. Moriwaki, L. Heo, S. Ovchinnikov, and M. Steinegger.
            "ColabFold: making protein folding accessible to all",
            *Nature Methods*, 19:679-682, May 2022.
            doi:10.1038/s41592-022-01488-1

.. [Beg21] M. Beg, J. Belin, T. Kluyver, A. Konovalov, M. Ragan-Kelley, N. Thiery, and H. Fangohr.
            "Using Jupyter for reproducible scientific workflows",
            *Computing in Science and Enginering*, 23(2):36-46, April 2021.
            doi:10.1109/MCSE.2021.3052101

.. [Berm03] H. Berman, K. Hendrick, and H. Nakamura.
            "Announcing the worldwide Protein Data Bank",
            *Nature Structural and Molecular Biology*, 10(12):980, December 2003.
.. no doi available

.. [Bias13] M. Biasini, T. Schmidt, S. Bienert, V. Mariani, G. Studer, J. Haas, N. Johner, A. D. Schenk, A. Philippsen, and T. Schwede.
            "OpenStructure: an integrated software framework for computational structural biology",
            *Acta Cryst. D*69(5):701–709, May 2013.
            doi:10.1107/S0907444913007051

.. [Brun98] A.T. Brünger, P.D. Adams, G.M. Clore, W.L. Delano, P. Gros, R.W. Grosse-Kunstleve, J.-S. Jiang, J. Kuszewski, M. Nilges, N. S. Pannu, R. J. Read, L. M. Rice, T. Simonson, and G. L. Warren.
            "Crystallography \& NMR system: A new software suite for macromolecular structure determination",
            *Acta Cryst. D*54(5):905-921, May 1998.
            doi:10.1107/S0907444998003254

.. [Burn17] T. Burnley, C.M. Palmer, and M. Winn.
            "Recent developments in the CCP-EM software suite"
            *Acta Cryst. D* 73(6):469-477, June 2017.
            doi:10.1107/S2059798317007859

.. [Carn18] T. Carneiro, R. V. M. Da Nóbrega, T. Nepomuceno, G.-B. Bian, V. H. C. De Albuquerque and P. P. Reboucas Filho.
            "Performance analysis of google colaboratory as a tool for accelerating deep learning applications",
            *IEEE Access* 6:61677-61685, November 2018.
            doi:10.1109/ACCESS.2018.2874767

.. [Cola21] https://colab.research.google.com

.. [ELSN]   https://elyra.readthedocs.io/en/latest/user_guide/code-snippets.html

.. [Elyra]  https://github.com/elyra-ai/elyra/blob/master/docs/source/getting_started/overview.md

.. [Godd18] T. D. Goddard, C.C. Huang, E.C. Meng, E.F. Pettersen, G.S. Couch, J. H. Morris, and T. E. Ferrin.
           "UCSF ChimeraX: Meeting modern challenges in visualization and analysis",
           *Protein Science*, 27(1):14-25, January 2018.
           doi:10.1002/pro.3235.

.. [Gran21] B. E. Granger and F. Perez.
           "Jupyter: Thinking and Storytelling With Code and Data",
           *Computing in Science & Engineering*, 23(2):7-14, March-April 2021.
           doi:10.1109/MCSE.2021.3059263

.. [Gros02] R. W. Grosse-Kunstleve, N. K. Sauter, N. W. Moriatry, P. D. Adams.
           "The Computational Crystallography Toolbox: crystallographic algorithms in a reusable software framework",
           *Journal of Applied Crystallography*, 35(1):126-136, February 2002.
           doi:10.1107/S0021889801017824.

.. [Hopk17] J. B. Hopkins, R. E. Gillilan, and S. Skou.
           "BioXTAS RAW: improvements to a free open-source program for small-angle X-ray scattering data reduction and analysis",
           *Journal of Applied Crystallography*, 50(5):1545–1553, October 217.
           doi:10.1107/S1600576717011438

.. [Hort99] R. M. Horton.
           "Scripting Wizards for Chime and RasMol",
           *Biotechniques*, 26(5):874-876, May 1999.
           doi:10.2144/99265ir01

.. [Kluy16] T. Kluyver, B. Ragan-Kelley, F. Perez, B. Granger, M. Bussonnier, J. Frederic, K. Kelley, J. Hamrick, J. Grout, S. Corlay, P. Ivanov, D. Avila, S. Abdalla, C. Willing, and Jupyter Development Team.
           "Jupyter Notebooks -- a publishing format for reproducible computational workflows",
           In F. Loizides and B. Schmidt (Eds.), *Positioning and Power in Academic Publishing: Players, Agents and Agendas* (pp, 87-90).
           doi:10.3233/978-1-61499-649-1-87

.. [jLsnip] https://github.com/QuantStack/jupyterlab-snippets

.. [Mana21] K. Manalastas-Cantos, P. V. Konarev, N. R. Hajizadeh, A. G. Kikhney, M. V. Petoukhov, D. S. Molodenskiy, A. Panjkovich, H. D. T. Mertens, A. Gruzinov, C. Borges, M. Jeffries, D. I. Sverguna, and D. Franke.
           "ATSAS 3.0: expanded functionality and new tools for small-angle scattering data analysis",
           *Journal of Applied Crystallography*, 54(1):343–355, February 2021.
           doi:10.1107/S1600576720013412

.. [Mott10] S. E. Mottarella, M. Rosa, A. Bangura, H. J. Bernstein, and P. A. Craig.
           "Conscript: RasMol to PyMOL script converter",
           *Biochemical and  Molecular Biology Education*, 38(6):419-422, November 2010.
           doi: 10.1002/bmb.20450

.. [MLGH]   https://github.com/MooersLab

.. [Moo21a] B. H. M. Mooers and M .E. Brown.
           "Templates for writing PyMOL scripts",
           *Protein Science*, 30(1):262-269, January 2021.
           doi:10.1002/pro.3997

.. [Moo21b] B. H. M. Mooers.
           "A PyMOL snippet library for Jupyter to boost researcher productivity",
           *Computing in Science and Engineering*, 23(2):47-53, April 2021.
           doi: 10.1109/mcse.2021.3059536

.. [Nguy17] H. Nguyen, D. A. Case, and A. S. Rose.
           "NGLview--interactive molecular graphics for Jupyter notebooks",
           *Bioinformatics*, 34(7):1241-1242, April 2017.
           doi:10.1093/bioinformatics/btx78

.. [PyMO21] https://pymol.org/2/

.. [Rese20] https://blog.jupyter.org/reusable-code-snippets-in-jupyterlab-8d75a0f9d207

.. [SciP20] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. {van der Walt}, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J.Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, I. Polat, Y. Feng, E. W. Moore, J. {VanderPlas}, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. {van Mulbregt}, Paul and {SciPy 1.0 Contributors}.
           "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python",
           *Nature Methods*, 17(3):261-272, February 2020.
           doi:10.1038/s41592-019-0686-2

.. [Agirre2023] Agirre, J., Atanasova, M., Bagdonas, H., Ballard, C., Baslé, A., Beilsten-Edmands, J., Borges, R., Brown, D., Burgos-Mármol, J., Berrisford, J. and Others
          The CCP4 suite: integrative software for macromolecular crystallography.
          *Acta Crystallographica Section D: Structural Biology*. 79, 449-461, 2023.
         doi:10.1107/S2059798323003595

.. [McNicholas2018] S. McNicholas, T. Croll, T. Burnley, C. Palmer, S. Hoh, H. Jenkins, E.  Dodson, K. Cowtan and J. Agirre
         "Automating tasks in protein structure determination with the clipper python module",
         *Protein Science*, 27, 207-216,  XXXX 2018.
         doi:10.1002/pro.3299

.. [Wojdyr2022] Wojdyr, M.
         "GEMMI: A library for structural biology",
         *Journal Of Open Source Software*, 7, 4200 (2022)
         doi:10.21105/joss.04200
