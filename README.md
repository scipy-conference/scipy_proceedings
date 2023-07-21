# SciPy Proceedings

This is the repository for submitting to and managing the Proceedings for the
Annual Scientific Computing with Python Conference.

This repository is a home for authors, reviewers and editors to collaboratively
create the proceedings for the conference.

You can find more information about the [proceedings' organising principles](#organising-principles-openness) below.

**All** communication between authors and reviewers should be civil and
respectful. There are no exceptions to this rule. Please see the
[SciPy Code of Conduct](https://www.scipy2022.scipy.org/code-of-conduct)
for more info.

You can find the [schedule for 2023](#timeline-for-2023) below.

Please use @-mentions in issues and pull requests(PRs) to [contact the proceedings Co-Chairs](#contacting-the-proceedings-co-chairs).

If you are an *Author*, please see [Instructions for Authors](#instructions-for-authors).

If you are a *Reviewer*, please see [Instructions for Reviewers](#instructions-for-reviewers).

If you are an *Editor*, please see [Instructions for Editors](#instructions-for-editors).

If you are a *Publisher*, please see [Instructions for Publishers](#instructions-for-publishers).

If you are *Submitting Slides*, please see [Instructions for Slides](#instructions-for-slides).

## Organising Principles: Openness

Overall, the SciPy proceedings are organised to be a fully open proceedings.

We aim to combine the best aspects of open source development, open peer review,
and open access publication.

### Built by and for Open Source Communities on Open Source Tech

The technologies used for running the conference are themselves developed in the
open and built on open source tools.

Open Development:
- with many people contributing code over more than a decade
    - many contributors start as authors submitting to the proceedings
    - provides a natural pathway for new members to join the proceedings committee
- technologies are managed via public, open source GitHub repositories:
    - build system: https://github.com/scipy-conference/scipy_proceedings
    - server: https://github.com/scipy-conference/procbuild

The systems for running the conference are built on top of open source tools:
- build system:
    - LaTeX
    - ReStructured Text (reST)
    - Python: docutils, lxml, pygments, pytest
- server:
    - Flask & waitress
    - pyzmq
    - Docker
    - Python: asyncio

### Open Peer Review meets Open Source Code Review

The entire submission and review procedure occurs through public PRs attached to
identifiable individuals.

- Authors and reviewers are encouraged to work collaboratively to improve
  submissions throughout the review process, much like open source code-review.

- Reviews are collaborative, aiming to improve the publication quality. This is
  possible because the content was already vetted by the program committee.

- Conversations occur attached to people's real GitHub usernames and are open to
  the public.
    - This allows for a transparent open review process.
    - This holds authors and reviewers accountable and encourages civil communication practices.

### Open Access for an Open Community

The papers are published as true Open Access (OA) articles with Creative Commons
Attribution (CC By) license.

- There are no article processing charges barring authors from submitting papers.
    - Reviewers and co-chairs volunteer their time.
    - Services with free tiers (like GitHub and Heroku) allow distributing the
      underlying technologies with minimal cost.

- Papers are openly available at http://conference.scipy.org/proceedings/, with
  no pay walls barring consumption or author processing charges.

- From 2010 onward, papers have DOIs (making them easily citable) and are also
  openly available from those DOIs.

The community is involved in the entire process for creating the proceedings,
which ensures relevance to the community that created them.

- Papers are submitted by authors who will be presenting talks and posters at the
  annual SciPy conference. Because we know the content is relevant to the SciPy
  community, review can focus on improving papers, not vetting them.

- Reviewers are invited by the editors, but community members may volunteer to
  review papers that interest them. The only barrier to participation is having
  a GitHub account.


## Contacting the Proceedings Co-Chairs

The most effective way to contact the Proceedings Co-Chairs for issues related to this GitHub repository is to use GitHub's issues and "@"-mentioning the Co-Chairs.

In 2023, the Proceedings Co-Chairs are
- Meghann Agarwal (@mepa)
- Chris Calloway (@cbcunc)
- Rohit Goswami (@HaoZeke)
- Dillon Niederhut (@deniederhut)

## Timeline for 2023

In addition to the following list, we break up the deadlines in the respective documents for authors and reviewers.

- April 14: Authors invited to submit full papers
- June 2: 1st Draft for Submission
- June 2–July 21: Open Review Period
- June 2: Reviewers Assigned
- June 23: Initial Complete Review
- July 14: Final Author Revision Deadline
- July 21: Final Recommendation and Comprehensive Review Deadlines
- August 4: Final Editorial Decisions for Proceedings Contents Deadline
- August 11: Time Window for Publishing Conference Ready Proceedings

## Instructions for Authors

Please submit your papers by 23:59 PST of the *1st Draft for Submission*
Deadline.

Submit your papers as a reStructuredText (rst) or LaTeX file via PR against this repository. Supporting LaTeX submissions is very new this year, so please consider it to be in beta, and please only use this option if you are already familiar with writing papers in LaTeX.

During the Open Review Period authors should work with their reviewers to refine
and improve their submission.

Proceedings Co-Chairs have final say in determining whether a paper is to be
accepted to the proceedings.

Authors should respond to all the reviewers' comments.

Authors should default to modifying their papers in response to reviewers'
comments.

Authors may not agree with the reviewers comments or may not wish to implement
the suggested changes. In those cases, the authors and reviewers should
attempt to discuss this in the PR's comment sections. It is important to
remember in these cases that we expect **all** communication between authors and
reviewers to be civil and respectful.

In the event that authors and reviewers are deadlocked, they should alert the
Proceedings Co-Chairs to this situation. As always, the Proceedings Co-Chairs
have final say in whether to accept or reject a paper.

### Author Deadlines

- April 14: Authors invited to submit full papers
- June 2: 1st Draft for Submission
- June 2–July 21: Open Review Period
- July 14: Final Author Revision Deadline
- July 21: Final Editorial Decisions for Proceedings Contents Deadline

### General Information and Guidelines for Authors:

- Papers are formatted using reStructuredText.
- Example papers are provided in `papers/00_bibderwalt` and `papers/00_vanderwalt`.
    - These papers provide examples of how to:
        - Label figures, equations and tables
        - Use math markup
        - Include code snippets
    - `00_bibderwalt` shows how to use a bib file for citations.
- For your paper to be found by the build system at http://procbuild.scipy.org
  your PR needs to have a title that begins with "Paper:". If you do not do
  this, the co-chairs will change your title on your behalf.
- Authors may include a project or consortium (e.g. [The Jupyter Project](https://raw.githubusercontent.com/scipy-conference/scipy_proceedings/2018/papers/project_jupyter/paper.rst))
- There must be at least one corresponding author, and this must be a specific person with a valid email address
- Authors of papers from previous SciPys may change their name on their published work by contacting the Proceedings Co-chairs
- All citations that have DOIs should include those DOIs in the paper's
  references section, see [`mybib.bib`](./papers/00_bibderwalt/mybib.bib).
- All figures and tables should have captions.
- Figures and tables should be positioned inline, close to their explanatory text.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Images and figures should be reasonably sized and formatted for viewing online; typically a few hundred kilobytes and less than 1 MB.
- Code snippets should be formatted to fit inside a single column without
  overflow.
- Avoid custom LaTeX markup where possible.
- Do not modify any files outside of your paper directory.
- The compiled version of the paper (PDF) should be at most 8 pages,
  including figures but not including references.

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

#### Author workflow steps

1. Get a local copy of the `scipy_proceedings` repo.
2. Update your local copy of the `scipy_proceedings` repo.
3. [Create a new branch](#creating-a-new-branch) for your paper based off the latest `2023` branch.
    - If you submit multiple papers, you will need a new branch for each.
    - Your branch should only contain your paper and the example papers.
    - Do not delete the example papers from your branch.
4. [Set up your environment](#setting-up-your-environment).
5. [Write your paper](#write-your-paper), [commit changes](#commit-your-changes), and [build your paper](#build-your-paper)
6. [Create a PR](#create-a-paper-pr) or [push changes to your PR's branch](#push-your-changes) and [check your paper](#check-your-paper) on http://procbuild.scipy.org.
    - If you want to alter the build system, do not include it in your
      submission's PR, create a separate PR against `dev`
      ([see below](#creating-build-system-prs) for more details).
7. Repeat steps 5 and 6, while also responding to reviewer feedback.
8. Unless instructed otherwise by the proceedings team, **DO NOT** fetch or pull the original `2023` branch after you start your paper.
    - Likewise, **DO NOT** update (merge/rebase) your branch with the original `2023` branch.
    - Otherwise, your branch may contain more than one paper, someone else's paper.
    - If you do, you may need to create your branch again and start over.
    - All changes to your branch should be confined to only your [paper directory](#setting-up-your-environment).

#### Getting a local copy of the scipy_proceedings repo

- If you do not have a GitHub account, create one.
- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.
- Clone the repo locally
    - `git clone https://github.com/<username>/scipy_proceedings`
    - `cd scipy_proceedings/`
- Add the `scipy-conference` repository as your `upstream` remote
    - `git remote add upstream https://github.com/scipy-conference/scipy_proceedings`

If you run `git remote -v  ` you should see something like the following:
```
origin	https://github.com/<username>/scipy_proceedings.git (fetch)
origin	https://github.com/<username>/scipy_proceedings.git (push)
upstream	https://github.com/scipy-conference/scipy_proceedings.git (fetch)
upstream	https://github.com/scipy-conference/scipy_proceedings.git (push)
```

#### Getting the latest branch

- Fetch the latest version of the `scipy_proceedings` repo
    - `git fetch upstream`
- Check out the upstream `2023` branch
    - `git checkout -b 2023 --track upstream/2023`
.
#### Creating a new branch

If you are submitting only one paper, you can use the `2023` branch directly.

Otherwise, you will need to create a new branch based on `2023` and set its
upstream to origin.

```
git checkout 2023
git checkout -b <your_branch_name>
git push --set-upstream origin <your_branch_name>
```

#### Setting up your environment

- Create a new environment (using your choice of environment manager, e.g., `pyenv` or `conda`).
- Install/update the required python libraries (`pip install -U -r requirements.txt`).
- Install LaTeX and any other non-python dependencies.
- Create a new directory `papers/<your_directory_name>`.
    - If you are submitting one paper, we recommend you use `<firstname_surname>`.
    - If you are submitting more than one paper, you will need to use a different
      branch and directory name for each paper.

#### Write your paper

- Copy an example paper into your directory.
    - You must have only one reST file in the top level of `<your_directory_name>`.
- As you make changes to your paper, commit those changes in discrete chunks.

#### Commit your changes

- Commit any changes inside the `paper/<your_directory_name>`
- When you push your commits to your PR's branch, the paper will be autobuilt
- Do not commit any changes to files outside of your paper directory.

If you want to change the way the build system works, we use a separate
submission procedure ([see below](#creating-build-system-prs)).

#### Build your paper

- Run `./make_paper.sh papers/firstname_surname` to make a PDF of your paper
- Check the output in `output/<your_directory_name>/paper.pdf`.
- Check that this output matches what you see on the
  [build server](http://procbuild.scipy.org).

#### Create a paper PR

- Once you are ready to submit your paper, make a pull request on GitHub.
  **Please ensure that you file against the correct branch.**
- Create a pull request against our `2023` branch.
- Do not modify any files outside of your paper directory. Create a separate PR for any changes to the build system.

#### Creating build system PRs

If you want to change the way the build system works, we use a separate
submission procedure.

- Create a new branch against `dev`.
- Make your changes to the build system.
- Do **not** commit any changes from your paper PR to this new branch.
- Make a separate PR against the `dev` branch, it will be reviewed separately.

#### Push to your PR

When you push to your repositories branch it automatically updates the PR. This
triggers a new build on the provided [build server](http://procbuild.scipy.org).

#### Check your paper's build

We encourage reviewers to review the PDFs built on our
[build server](http://procbuild.scipy.org).

You should regularly check to see if the paper(s) that you build locally match the
paper(s) that you see on the server.

If it is not the same, please immediately contact us with a GitHub issue
describing the discrepancy. Please include screenshots and an explanation of the
differences. For best results, please [@-mention the Proceedings Co-Chairs](#contacting-the-proceedings-co-chairs).

## Instructions for Reviewers

You will be reviewing authors' pull requests. While authors should have a proper
draft of their paper ready for you by *1st Draft Submission* deadline.

We ask that you read [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md) before beginning any reviews.

**All** communication between authors and reviewers should be civil and respectful at all times.

The goal of our review process is to improve the paper that the authors are
working on. Our aim is to have you and the author collaborate on making their
better by using an iterative process.

While our basic approach is to have you and the author iterate, we ask you to
complete an initial review and start that conversation by the *Initial Complete Review
Deadline*.

We ask that by the *Final Recommendation Deadline* you have a recommendation to
either **accept** or **reject** the paper at that point and time.

**Note**:
You many recommend changes after the *Final Recommendation Deadline*. If there
are any major changes after the *Final Recommendation Deadline* you should
immediately contact the Proceedings Committee Co-Chairs. As a heuristic, if you
think the paper should not be in the proceedings unless the authors make the
change in question, then that change should be requested and made before the
*Final Recommendation Deadline*.

### Reviewer Deadlines

- June 2: Reviewers Assigned
- June 23: Initial Complete Review
- July 21: Final Recommendation and Comprehensive Review Deadlines

### Reviewer Workflow

- Read [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md)
- Click on the Pull Requests Tab and find the papers assigned to you
- After reading the paper, you can start the review conversation however you prefer
    - You can use line comments (on the paper itself) or high-level comments.
- Authors will respond to your comments, possibly via their own comments or by
  modifying their paper.
- This begins an iterative review process where authors and reviewers can discuss the
  evolving submission.
- By the *Final Recommendation Deadline*, we ask that you give two things
    1. A comprehensive review of the paper as it stands. This will act as the final
       review.
    2. A final recommendation to include the paper in the proceedings or not.
        - When you make the Final Recommendation, please [contact the proceedings Co-Chairs](#contacting-the-proceedings-co-chairs) in the PR in question.

## Review Criteria

A small subcommittee of the SciPy 2017 organizing committee has created
[this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/master/review_criteria.md)
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

There will be a server online building open pull requests at http://procbuild.scipy.org.

Authors: you should check to ensure that your local builds match the papers
built on this site. Please create an issue if they do not match.

Reviewers: You should be able to pull a built PDF for review from there.

## For organisers

### Instructions for Publishers

To information about how to manage the whole proceedings, please see
`publisher/README.md` and `publisher/Makefile`.

#### Publisher Deadlines

- April 14: Authors invited to submit full papers
- June 2 – July 21: Open Review Period
    - The [build server](#build-server) should be maintained throughout the Open Review Period.
- August 11: Time Window for Publishing Conference Ready Proceedings

### Instructions for Editors

As reviewers review papers, editors should apply **labels** to the PR to flag the
current state of the review process.
  - The **labels** in question are:
    - **needs-more-review** if the paper needs further review,
    - **pending-comment** if the paper is waiting on an authors' response, or
    - **unready** if the paper is not ready for the proceedings.

Editors should come to a final 'ready', 'unready' decision before the **Final Editorial Decisions for Proceedings Contents** deadline.

#### Editor Deadlines

- April 14: Authors invited to submit full papers
- June 2 – July 21: Open Review Period
- June 2: Reviewers Assigned
- June 23: Initial Complete Review
    - Editors should verify that reviews have been completed
- August 4: Final Editorial Decisions for Proceedings Contents Deadline

## Instructions for Slides

#### Slide/Poster submission steps

1. Get a local copy of the `scipy_proceedings` repo.
2. Update your local copy of the `scipy_proceedings` repo.
3. [Create a new branch](#creating-a-new-branch) for your paper based off the latest `2023` branch.
4. Inside the `presentations` folder, there are directories for:
    1. 3-minute lightning talk slide decks (lightning)
    2. Posters presented at the poster session (posters)
    3. 30-minute talk slide decks (slides)
    4. SciPy tools plenary slide decks (tools)
5. Choose the appropriate folder, and make a new directory inside it (it needs a unique name)
6. Copy your slide deck or poster into the directory, and add a file called `info.json` with the following fields needed for uploading to Zenodo (using an empty string for author orcid or
affiliation if these cannot be provided):
```json
{
    "title": "The title of your presentation",
    "authors": [
        {
            "name": "The first author or presenter",
            "affiliation": "first author's affiliation",
            "orcid": "0000-0000-0000-0000"
        },
        {
            "name": "The second author or presenter",
            "affiliation": "second author's affiliation",
            "orcid": "0000-0000-0000-0001"
        }
    ],
    "description": "1-4 sentences explaining what your presentation is about"
}
```
7. [Create a PR](#create-a-paper-pr)

You can see examples of submissions in the `example` folder in each presentation directory.
