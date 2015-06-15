# SciPy Proceedings

## Paper Format

Papers are formatted using reStructuredText and the compiled version should be
no longer than 8 pages, including figures.  Here are the steps to produce a
paper:

- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.

- Check out the 2015 branch (``git checkout 2015``).

- An example paper is provided in ``papers/00_vanderwalt``.  Create a new
  directory ``papers/firstname_surname``, copy the example paper into it, and
  modify to your liking.

- Run ``./make_paper.sh papers/firstname_surname`` to compile your paper to
  PDF (requires LaTeX, docutils, Python--see below).  The output appears in
  ``output/firstname_surname/paper.pdf``.

- Once you are ready to submit your paper, file a pull request on GitHub.
  **Please ensure that you file against the correct branch**--your branch
  should be named 2015, and the pull-request should be against our 2015
  branch.

- Please do not modify any files outside of your paper directory.

Pull requests are to be submitted by **June 15th**, but modifications are
allowed during the open review period which ends July 5th.


## General Guidelines

- All figures and tables should have captions.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Code snippets should be formatted to fit inside a single column without
  overflow.
- Avoid custom LaTeX markup where possible.

## Review Criteria

A small subcommittee of the SciPy 2015 organizing committee has created [this
set of suggested review
criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md)
to help guide authors and reviewers alike. Suggestions and amendments to these
review criteria are enthusiastically welcomed via discussion or pull request.

## Other markup

Please refer to the example paper in ``papers/00_vanderwalt`` for
examples of how to:

 - Label figures, equations and tables
 - Use math markup
 - Include code snippets

## Requirements

 - IEEETran (often packaged as ``texlive-publishers``, or download from
   [CTAN](http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/) LaTeX
   class
 - AMSmath LaTeX classes (included in most LaTeX distributions)
 - `docutils` 0.8 or later (``easy_install docutils``)
 - `pygments` for code highlighting (``easy_install pygments``)

On Debian-like distributions:

```
sudo apt-get install python-docutils texlive-latex-base texlive-publishers \
                     texlive-latex-extra texlive-fonts-recommended
```

## Build Server

Thanks to the great and wonderful Stefan van der Walt, there is a server online 
building the open pull requests [here](http://nipy.bic.berkeley.edu:5000/). You may be 
able to pull a built PDF for review from there.

## For organizers

To build the whole proceedings, see the Makefile in the publisher directory.
