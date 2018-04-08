# SciPy Proceedings

This is the repository for submitting to and managing the Proceedings for the Annual Conference.

If you are an *Author*, please see [Instructions-for-Authors](#Instructions-for-Authors).

If you are a *Reviewer*, please see [Instructions-for-Reviewers](#Instructions-for-Reviewers).











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

### Author Workflow

Below we outline the steps to submit a paper.

Before you begin, you should have a GitHub account. If we refer to `<username>`
in code examples, you should replace that with your GitHub username. 

More generally, angle brackets with a value inside are meant to be replaced with
the value that applies to you. 

For example, if your GitHub username was `mpacer`, you would transform 

```
git clone https://github.com/<username>/scipy_proceedings
```

into: 

```
git clone https://github.com/mpacer/scipy_proceedings
```

1. Get a local copy of the `scipy_proceedings` repo.
2. Update your local copy of the `scipy_proceedings` repo.
3. Create a new branch for your paper based off the latest `2018` branch.
    - If you submit multiple papers, you will need a new branch for each.
4. Set up your environment.
5. Write your paper, commit changes, and build your paper.
6. Repeat step 6, while also responding to reviewer feedback.

#### Getting a local copy of the scipy_proceedings repo 

- If you do not have a GitHub account, create one. 
- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.
- Clone the repo locally 
    - `git clone https://github.com/<username>/scipy_proceedings`
- Add the `scipy-conference` repository as your `upstream` remote
    - `git remote add upstream https://github.com/scipy-conference/scipy_proceedings`

If you run `git remote -v  ` you should see something like the following: 
```
origin	https://github.com/<username>/scipy_proceedings.git (fetch)
origin	https://github.com/<username>/scipy_proceedings.git (push)
upstream	https://github.com/scipy-conference/scipy_proceedings.git (fetch)
upstream	https://github.com/scipy-conference/scipy_proceedings.git (push)
```

#### Getting the latest `2018` branch

- Fetch the latest version of the `scipy_proceedings` repo
    - `git fetch upstream`
- Check out the upstream `2018` branch 
    - `git checkout -b 2018 --track upstream/2018`

#### Creating a new branch based off of `2018`

If you are submitting only one paper, you can use the `2018` directly.

Otherwise, you will need to create a new branch based on `2018` and set its
upstream to origin.

```
git checkout 2018
git checkout -b <your_branch_name> 
git push --set-upstream origin <your_branch_name>
```

#### Setting up your environment

- Create a new environment (using your choice of environment manager, e.g., `pyenv` or `conda`).
- Install/update the required python libraries (`pip install -U -r requirements.txt`).
- Install LaTeX and any other non-python dependencies 
- Create a new directory `papers/<your_directory_name>`
    - if you are submitting one paper, we recommend you use `<firstname_surname>`
    - if you are submitting more than one paper, you will need to use a different 
      directory name for each paper

#### Write your paper

- Copy an example paper into your directory.
    - You must have only one reST file in the top level of `<your_directory_name>`.
- As you make changes to your paper, commit those changes in discrete chunks. 
- Do not modify any files outside of your paper directory.

#### Build your paper

- Run `./make_paper.sh papers/firstname_surname` to make a PDF of your paper 
- Check the output in `output/<your_directory_name>/paper.pdf`.

#### Create a paper PR

- Once you are ready to submit your paper, file a pull request on GitHub.
  **Please ensure that you file against the correct branch**
- Create a pull-request against our `2018` branch.


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
