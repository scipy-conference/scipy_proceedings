# SciPy Proceedings

This is the repository for submitting to and managing the Proceedings for the Annual Conference.

If you are an *Author*, please see [Instructions-for-Authors](#Instructions-for-Authors).

If you are a *Reviewer*, please see [Instructions-for-Reviewers](#Instructions-for-Reviewers).


Papers are formatted using reStructuredText and the compiled version should be
no longer than 8 pages, including figures.  Here are the steps to produce a
paper:

- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.

- Check out the 2017 branch (``git checkout 2017``).

- Create a new environment (using your choice of environment manager, e.g., ``pyenv`` or ``conda``).

- Install/update the required python libraries (``pip install -U -r requirements.txt``).

- An example paper is provided in ``papers/00_vanderwalt``.  Create a new
  directory ``papers/firstname_surname``, copy the example paper into it, and
  modify to your liking.

- Run ``./make_paper.sh papers/firstname_surname`` to compile your paper to
  PDF (requires LaTeX, docutils, Python--see below).  The output appears in
  ``output/firstname_surname/paper.pdf``.

- Once you are ready to submit your paper, file a pull request on GitHub.
  **Please ensure that you file against the correct branch**--your branch
  should be named 2017, and the pull-request should be against our 2017
  branch.

- Please do not modify any files outside of your paper directory.

## Schedule Summary

Authors may make changes to their submisions throughout the review process.

There are many different styles of review (some do paragraph comments, others
do 'code review' style line edits) and the process is open.
## Instructions for Authors

We encourage authors and reviewers to work together iteratively to make each 
others papers the best they can be.
Combine the best principles of open source development and academic publication.

These dates are the tentative timeline for 2017:

- May 4th - Authors invited to submit full papers
- June 6th - Initial submissions due
- June 7th - Reviewers assigned
- June 21st - Reviews due
- June 21st -July 7th: Authors revise papers based on reviews
- July 7th - Acceptance/rejection of papers
- July 8th - Papers must be camera-ready
- July 10-16th - Conference
- July 20th - Publication

## General Guidelines

### General Information and Guidelines for Authors:

- Papers are formatted using reStructuredText.
- Example papers are provided in `papers/00_bibderwalt` and `papers/00_vanderwalt`.
    - These papers provide examples of how to:
        - Label figures, equations and tables
        - Use math markup
        - Include code snippets
    - `00_bibderwalt` also shows how to use a bib file for citations
- All figures and tables should have captions.
- Figures and tables should be positioned inline, close to their explanatory text.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Code snippets should be formatted to fit inside a single column without
  overflow.
- Avoid custom LaTeX markup where possible.
- Do not modify any files outside of your paper directory. 
- The compiled version of the paper (PDF) should be at most 8 pages,
  including figures and references.

### Reviewer Workflow

- Read [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md)
- Click on the Pull Requests Tab and find the papers assigned to you
- After reading the paper, you can start the review conversation however you prefer
    - You can use line comments (on the paper itself) or high-level comments.
- Authors will respond to your comments, possibly via their own comments or by 
  modifying their paper.  
- This begins an iterative review process where authors and reviewers can discuss the
  evolving submission.
- As you review the paper, it will help to apply **labels** to the PR to flag the 
  current state of the review process. 
     - The **labels** in question are:
        - **needs-more-review** if the paper needs further review, 
        - **pending-comment** if the paper is waiting on an authors' response, or 
        - **unready** if the paper is not ready for the proceedings.
- By the *Final Recommendation Deadline*, we ask that you give two things
    1. A comprehensive review of the paper as it stands. This will act as the final 
       review.  
    2. A final recommendation to include the paper in the proceedings or not.
    - When you make the Final Recommendation, please (@)mention the editor(s) assigned 
      to the paper. For 2018, this will be some of:
        - M Pacer (@mpacer)
        - David Lippa (@dalippa)
        - Dillon Niederhut (@deniederhut)
        - Fatih Akici (@FatihAkici)
- Editors should come to a final 'ready', 'unready' decision before the **Final Editorial Decisions for Proceedings Contents** deadline.

## Review Criteria

A small subcommittee of the SciPy 2017 organizing committee has created [this
set of suggested review
criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md)
to help guide authors and reviewers alike. Suggestions and amendments to these
review criteria are enthusiastically welcomed via discussion or pull request.


## Requirements

 - Install the requirements in the requirements.txt file: `pip install -r requirements.txt`
 - IEEETran (often packaged as ``texlive-publishers``, or download from
   [CTAN](http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/)) LaTeX
   class
 - AMSmath LaTeX classes (included in most LaTeX distributions)
 - alphaurl (often packaged as ``texlive-bibtex-extra``, or download from
   [CTAN](https://www.ctan.org/pkg/urlbst)) urlbst BibTeX style

### Debian-like distributions:

```
sudo apt-get install python-docutils texlive-latex-base texlive-publishers \
                     texlive-latex-extra texlive-fonts-recommended \
                     texlive-bibtex-extra
```

Note you will still need to install `docutils` with `pip` even on a Debian system.

### Fedora

On Fedora, the package names are slightly different:

```
su -c `dnf install python-docutils texlive-collection-basic texlive-collection-fontsrecommended texlive-collection-latex texlive-collection-latexrecommended texlive-collection-latexextra texlive-collection-publishers texlive-collection-bibtexextra`
```

## Build Server

There will be a server online building the open pull requests 
[here](http://zibi.bids.berkeley.edu:7001). You should be able to pull a built PDF 
for review from there.

TODO: update server link

## For organizers

To build the whole proceedings, see the Makefile in the publisher directory.
