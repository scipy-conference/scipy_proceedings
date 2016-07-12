# SciPy Proceedings

## Instructions for Reviewers

- Click on the Pull Requests Tab and browse to find the papers assigned to you
- After reading the paper, you can start the review conversation by simply commenting
  on the paper, taking into consideration
  [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md).
- Authors will then respond to the comments and/or modify the paper to address the comments. 
- This will begin an iterative review process where authors and reviewers can discuss the
  evolving submission.
- Reviewers may also apply one of the labels 'needs-more-review', 'pending-comment', or 
  'unready' to flag the current state of the review process.
- Only once a reviewer is satisfied that the review process is complete and the submission should
  be accepted to the proceedings, should they affix the 'ready' label. 
- Reviewers should come to a final 'ready', 'unready' decision before **July 10th** at 18:00 PST.

## Instructions for Authors

Submissions must be received by **May 30th** at 23:59 PST, but modifications are
allowed during the open review period which ends July 5th at 18:00 PST.  Submissions are
considered received once a Pull Request has been opened following the procedure
outlines below.

Papers are formatted using reStructuredText and the compiled version should be
no longer than 8 pages, including figures.  Here are the steps to produce a
paper:

- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.

- Check out the 2016 branch (``git checkout 2016``).

- An example paper is provided in ``papers/00_vanderwalt``.  Create a new
  directory ``papers/firstname_surname``, copy the example paper into it, and
  modify to your liking.

- Run ``./make_paper.sh papers/firstname_surname`` to compile your paper to
  PDF (requires LaTeX, docutils, Python--see below).  The output appears in
  ``output/firstname_surname/paper.pdf``.

- Once you are ready to submit your paper, file a pull request on GitHub.
  **Please ensure that you file against the correct branch**--your branch
  should be named 2016, and the pull-request should be against our 2016
  branch.

- Please do not modify any files outside of your paper directory.

## Schedule Summary

Authors may make changes to their submisions throughout the review process.

There are many different styles of review (some do paragrpah comments, others
do 'code review' style line edits) and the process is open.

We encourage authors and reviewers to work together iteratively to make each 
others papers the best they can be.
Combine the best principles of open source development and academic publication.

These dates are the 

- May 30th - Initial submissions
- June 7th - Reviewers assigned
- June 30th - Reviews due
- June 30th- July 10th: Authors revised papers based on reviews
- July 10th - Acceptance/rejection of papers.
- July 11-18th - Conference
- July 30th - Final submissions due
- August 8th - Publication.

## General Guidelines

- All figures and tables should have captions.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Code snippets should be formatted to fit inside a single column without
  overflow.
- Avoid custom LaTeX markup where possible.

## Review Criteria

A small subcommittee of the SciPy 2016 organizing committee has created [this
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
   [CTAN](http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/)) LaTeX
   class
 - AMSmath LaTeX classes (included in most LaTeX distributions)
 - alphaurl (often packaged as ``texlive-bibtex-extra``, or download from
   [CTAN](https://www.ctan.org/pkg/urlbst)) urlbst BibTeX style
 - `docutils` 0.8 or later (``easy_install docutils``)
 - `pygments` for code highlighting (``easy_install pygments``)
 - Due to a bug in the Debian packaging of ``pdfannotextractor``, you may have
   to execute ``pdfannotextractor --install`` to fetch the PDFBox library.

On Debian-like distributions:

```
sudo apt-get install python-docutils texlive-latex-base texlive-publishers \
                     texlive-latex-extra texlive-fonts-recommended \
                     texlive-bibtex-extra
```

Note you will still need to install `docutils` with `easy-install` or `pip` even on a Debian system.

On Fedora, the package names are slightly different

```
su -c `dnf install python-docutils texlive-collection-basic texlive-collection-fontsrecommended texlive-collection-latex texlive-collection-latexrecommended texlive-collection-latexextra texlive-collection-publishers texlive-collection-bibtexextra`
```

## Build Server

Thanks to the great and wonderful Katy Huff, there is a server online 
building the open pull requests [here](http://zibi.bids.berkeley.edu:5000). You may be 
able to pull a built PDF for review from there.

## For organizers

To build the whole proceedings, see the Makefile in the publisher directory.
