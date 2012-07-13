SciPy Proceedings
=================

Paper Format
------------

Papers are formatted using reStructuredText and the compiled version should be
no longer than 7 pages, including figures.  Here are the steps to produce a
paper:

- Fork the 2012 branch of the `scipy_proceedings
  <https://github.com/scipy/scipy_proceedings>`__ repository on GitHub.  An
  example paper is provided in ``papers/00_vanderwalt``.  Create a new
  directory ``papers/firstname_surname``, copy the example paper into it, and
  modify to your liking.

- Run ``./make_paper.sh papers/firstname_surname`` to compile your paper to PDF
  (requires LaTeX, docutils, Python).  The output appears in
  ``output/firstname_surname/paper.pdf``.

- Once you are ready to submit your paper, file a pull request on GitHub.

Pull requests are to be submitted by **July 15th**, but modifications may be
pushed until August 12th.

General Guidelines
``````````````````
- All figures and tables should have captions.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Code snippets should be formatted to fit inside a single column without
  overflow.
- Avoid custom LaTeX markup where possible.

Other markup
------------
Please refer to the example paper in ``papers/00_vanderwalt`` for
examples of how to:

 - Label figures, equations and tables
 - Use math markup
 - Include code snippets

Requirements
------------
 - IEEETran and AMSmath LaTeX classes
 - **Latest** docutils (development version, they haven't released in years)
 - Pygments for code highlighting
