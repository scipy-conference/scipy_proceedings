:author: Bradley D. Dice
:email: bdice@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor

:author: Brandon L. Butler
:email: butlerbr@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Vyas Ramasubramani
:email: vramasub@umich.edu
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor

:author: Alyssa Travitz
:email: atravitz@umich.edu
:institution: TODO Department, University of Michigan, Ann Arbor

:author: Michael M. Henry
:email: mike.henry@choderalab.org
:institution: TODO

:author: Hardik Ojha
:email: hojha@ee.iitr.ac.in
:institution: TODO Department, Indian Institute of Technology Roorkee

:author: Carl S. Adorf
:email: TODO
:institution: TODO

:author: Sharon C. Glotzer
:email: sglotzer@umich.edu
:institution: Department of Physics, University of Michigan, Ann Arbor
:institution: Department of Chemical Engineering, University of Michigan, Ann Arbor
:institution: Department of Materials Science and Engineering, University of Michigan, Ann Arbor
:institution: Biointerfaces Institute, University of Michigan, Ann Arbor

:bibliography: paper

-------------------------------------------------------------------
signac: Data Management and Workflows for Computational Researchers
-------------------------------------------------------------------

.. class:: abstract

Abstract goes here.

.. class:: keywords

   data management, TODO


Introduction
------------

Past signac papers: :cite:`signac_commat, signac_scipy_2018`

The full source code of all examples in this paper can be found online [#]_.

.. [#] https://github.com/glotzerlab/signac-examples


Applications of signac
----------------------

Lorem ipsum.

Executing complex workflows via groups and aggregation
------------------------------------------------------

Lorem ipsum.

.. code-block:: python

   import signac
   import flow
   print(signac.__version__)
   print(flow.__version__)

Synced Collections: Backend-agnostic, persistent, mutable data structures
-------------------------------------------------------------------------

Lorem ipsum.

Project Evolution
-----------------

Lorem ipsum.

Conclusions
-----------

Lorem ipsum.

Getting signac
--------------

The signac framework is tested for Python 3.6+ and is compatible with Linux, macOS, and Windows.
To install, execute

.. code-block:: bash

    conda install -c conda-forge signac signac-flow signac-dashboard

or

.. code-block:: bash

    pip install signac signac-flow signac-dashboard

Source code is available on GitHub [#]_ [#]_ and documentation is hosted online by ReadTheDocs [#]_.

.. [#] https://github.com/glotzerlab/signac
.. [#] https://github.com/glotzerlab/signac-flow
.. [#] https://docs.signac.io/


Acknowledgments
---------------

B.D. is supported by a National Science Foundation Graduate Research Fellowship Grant DGE 1256260.
TODO...
