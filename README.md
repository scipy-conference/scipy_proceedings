# SciPy Proceedings

This is the repository for submitting to and managing the Proceedings for the
Annual Scientific Computing with Python Conference.

This repository is a home for authors, reviewers and editors to collaboratively
create the proceedings for the conference.

You can find more information about the [proceedings' organising principles](#organising-principles-openness) below.

**All** communication between authors and reviewers should be civil and
respectful. There are no exceptions to this rule. Please see the [NumFOCUS Code of Conduct](https://numfocus.org/code-of-conduct)
for more info. Attendees at SciPy 2024 are subject to the NumFOCUS Code of Conduct.

You can find the [schedule for 2024](#timeline-for-2024) below.

Please use @-mentions in issues and pull requests(PRs) to [contact the proceedings Co-Chairs](#contacting-the-proceedings-co-chairs).

If you are an _Author_, please see [Instructions for Authors](#instructions-for-authors).

If you are a _Reviewer_, please see [Instructions for Reviewers](#instructions-for-reviewers).

If you are an _Editor_, please see [Instructions for Editors](#instructions-for-editors).

If you are a _Publisher_, please see [Instructions for Publishers](#instructions-for-publishers).

If you are _Submitting Slides_, please see [Instructions for Slides](#instructions-for-slides).

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
- technologies are managed via public, open source GitHub repositories

The systems for running the conference are built on top of open source tools, including:

- MyST Markdown ([mystmd.org](https://mystmd.org))
- Typst - for fast PDF generation (e.g. [SciPy template](https://github.com/myst-templates/scipy))

### Open Peer Review meets Open Source Code Review

The entire submission and review procedure occurs through public PRs attached to
identifiable individuals.

- Authors and reviewers are encouraged to work collaboratively to improve submissions throughout the review process, much like open source code-review.

- Reviews are collaborative, aiming to improve the publication quality. This is possible because the content was already vetted by the program committee.

- Conversations occur attached to people's real GitHub usernames and are open to the public.
  - This allows for a transparent open review process.
  - This holds authors and reviewers accountable and encourages civil communication practices.

### Open Access for an Open Community

The papers are published as true Open Access (OA) articles with Creative Commons Attribution ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)) license.

- There are no article processing charges barring authors from submitting papers.

  - Reviewers and co-chairs volunteer their time.
  - Services with free tiers (like GitHub) allow distributing the underlying technologies with minimal cost.

- Papers are openly available at http://proceedings.scipy.org, with no pay walls barring consumption or author processing charges.

- From 2010 onward, papers have DOIs (making them easily citable) and are also openly available from those DOIs.

- From 2023 onwards, full HTML is the _preferred_ format in addition to the PDF being available.

The community is involved in the entire process for creating the proceedings, which ensures relevance to the community that created them.

- Papers are submitted by authors who will be presenting talks and posters at the annual SciPy conference. Because we know the content is relevant to the SciPy community, review can focus on improving papers, not vetting them.

- Reviewers are invited by the editors, but community members may volunteer to review papers that interest them. The only barrier to participation is having a GitHub account.

## Contacting the Proceedings Co-Chairs

The most effective way to contact the Proceedings Co-Chairs for issues related to this GitHub repository is to use GitHub's issues and "@"-mentioning the Co-Chairs.

In 2024, the Proceedings Co-Chairs are:

- Meghann Agarwal (@mepa)
- Amey Ambade (@ameyxd)
- Chris Calloway (@cbcunc)
- Rowan Cockett (@rowanc1)
- Sanhita Joshi (@sanhitamj)
- Charles Lindsey (@cdlindsey)
- Hongsup Shin (@hongsupshin)

## Timeline for 2024

In addition to the following list, we break up the deadlines in the respective documents for authors and reviewers.

- Apr 9:  Reviewer invitations sent
- Apr 23: Deadline to respond to offer to be a reviewer
- Apr 26: Authors invited to submit full papers
- May 3:  Webinar offered to authors
- May 31: Deadline to submit first draft by authors
- May 31: Assignment of reviewers to papers
- May 31: Open Review Period begins
  - Reviewers comment on papers to authors during this period.
  - Authors also respond to review comments with improvements to papers during this period.
- Jun 21: Initial complete review
  - Reviewers continue to comment on paper improvements during this period.
  - Authors also respond to review comments with further improvements to papers during this period.
- Jul 26: Final review deadline
  - Authors continue to make revisions in response to final review comments during this period.
- Jul 31: Final author revision deadline
- Jul 31: Open Review Period ends
  - Authors put down their pens.
  - Reviewers make an up or down decision on publication readiness of papers during this period.
- Aug 9:  Final reviewer decision deadline
- Aug 16: Proceedings final sign-off by editors
  - The publication process begins after final sign-off.

## Instructions for Authors

Please submit your papers by 23:59 PST of the _Deadline to submit first draft_.

Submit your papers as a MyST Markdown ([mystmd.org](https://mystmd.org)) or
LaTeX file via PR against this repository.
Please only use LaTeX if you are already familiar with writing papers in LaTeX.
The build process are using the `mystmd` CLI in 2024, which allows us to support
a web-first reading experience.
In future years this will allow us to accept notebooks and computational
environments, however, this is not available in 2024.

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

### Getting Help

If you have a challenge with any technical aspect of authoring your paper in MyST or LaTeX,
please do not hesitate to reach out via your GitHub pull request or issue on this repository.
A member of the Proceedings Co-chairs will help you directly or identify a work-around.

### Author Deadlines

- Apr 26: Authors invited to submit full papers
- May 3:  Webinar offered to authors
- May 31: Deadline to submit first draft by authors
  - Reviewers comment on papers to authors during this period.
  - Authors also respond to review comments with improvements to papers during this period.
- Jul 31: Final author revision deadline
  - Authors put down their pens.

### General Information and Guidelines for Authors

- Papers are formatted using MyST ([mystmd.org](https://mystmd.org)) or LaTeX (which also uses [MyST](https://mystmd.org), please see notes on LaTeX below)
- The paper is written and reviewed using the interactive HTML view (i.e. `myst start`), the PDF is built upon acceptance only
- Example papers are provided in `papers/00_myst_template` and `papers/00_tex_template`
  - These papers provide examples of how to:
    - Label figures, equations and tables
    - Use math markup
    - Include code snippets
    - Use a BibTeX files and/or DOIs for citations
- When creating your pull-request, add a pull-request label of `paper` to trigger the build process. If you do not add this, a proceedings chair member will add it for you.
- Authors may include a project or consortium (e.g. [The Jupyter Project](https://raw.githubusercontent.com/scipy-conference/scipy_proceedings/2018/papers/project_jupyter/paper.rst))
- There must be at least one corresponding author, and this must be a specific person with a valid email address
- Authors of papers from previous SciPys may change their name on their published work by contacting the Proceedings Co-chairs
- All citations that have DOIs should include those DOIs in the paper's references section, see [`mybib.bib`](./papers/00_myst_template/mybib.bib).
- All figures and tables should have captions.
- Figures and tables should be positioned close to their explanatory text.
- All abbreviations should be identified in your `myst.yml` ([docs](https://mystmd.org/guide/glossaries-and-terms#abbreviations))
- License conditions on images and figures must be respected (Creative Commons,
  etc.)
- Images and figures should be reasonably sized and formatted for viewing online; typically less than 1 MB
- Do not modify any files outside of your paper directory
- When using the LaTeX option, please consider:
  - SciPy is supporting _HTML_. LaTeX is not involved in reading or rendering (as of 2024 we use [Typst for building PDFs](https://github.com/myst-templates/scipy))
  - Custom LaTeX macros are **not** supported and some packages may not be supported
- The compiled version of the paper should be at most 6000 words

including figures but not including references; this is about 8 pages for the published PDF that will be created upon acceptance.

### Author Workflow

Below we outline the steps to submit a paper.

Before you begin, you should have a GitHub account. If we refer to `<username>`
in code examples, you should replace that with your GitHub username.

More generally, angle brackets with a value inside are meant to be replaced with
the value that applies to you.

For example, if you typically clone using the web URL, and your GitHub username was `mpacer`, you would transform

```
git clone <scheme>github.com/<username>/scipy_proceedings
```

into:

```
git clone https://github.com/mpacer/scipy_proceedings
```

#### Author workflow steps

1. [Get a local copy](#getting-a-local-copy-of-the-scipy_proceedings-repo) of the `scipy_proceedings` repo.
2. [Update your local copy](#getting-the-latest-branch) of the `scipy_proceedings` repo.
3. [Create a new branch](#creating-a-new-branch) for your paper based off the latest `2024` branch.
   - If you submit multiple papers, you will need a new branch for each.
4. [Install MyST Markdown and Node](#setting-up-your-environment) and [copy a template](#setting-up-your-environment).
5. [Write your paper](#write-your-paper), [commit changes](#commit-your-changes), and [build your paper](#preview-your-paper)
6. [Create a PR](#create-a-paper-pr) or [push changes to your PR's branch](#commit-your-changes) and [check your paper](#check-your-papers-build).
   - If you want to alter the build system, do not include it in your
     submission's PR, create a separate PR against `dev`
     ([see below](#creating-build-system-prs) for more details).
7. Repeat steps 5 and 6, while also responding to reviewer feedback.

#### Getting a local copy of the scipy_proceedings repo

- If you do not have a GitHub account, create one.
- Fork the
  [scipy_proceedings](https://github.com/scipy-conference/scipy_proceedings)
  repository on GitHub.
- Clone the repo locally
  - replace `<scheme>` with `git@` or `https://`, for example
  - replace `<username>` with your GitHub username
  - `git clone <scheme>github.com/<username>/scipy_proceedings`
  - `cd scipy_proceedings/`
- Add the `scipy-conference` repository as your `upstream` remote
  - `git remote add upstream <scheme>github.com/scipy-conference/scipy_proceedings`

If you run `git remote -v ` you should see something like the following:

```
origin	<scheme>github.com/<username>/scipy_proceedings.git (fetch)
origin	<scheme>github.com/<username>/scipy_proceedings.git (push)
upstream	<scheme>github.com/scipy-conference/scipy_proceedings.git (fetch)
upstream	<scheme>github.com/scipy-conference/scipy_proceedings.git (push)
```

#### Getting the latest branch

- Fetch the latest version of the `scipy_proceedings` repo
  - `git fetch upstream`
- Check out the upstream `2024` branch
  - `git checkout -b 2024 --track upstream/2024`

#### Creating a new branch

If you are submitting only one paper, you can use the `2024` branch directly.

Otherwise, you will need to create a new branch based on `2024` and set its
upstream to origin.

```
git checkout 2024
git checkout -b <your_branch_name>
git push --set-upstream origin <your_branch_name>
```

#### Setting up your environment

- _Optional_: Create a new environment (using your choice of environment manager, e.g., `pyenv` or `conda`).
- Install MyST Markdown from [mystmd.org](https://mystmd.org/guide/quickstart)
  - `pip install mystmd`
  - Install `nodejs` (see [options](https://mystmd.org/guide/installing-prerequisites))
- Create a new directory `papers/<your_directory_name>`
  - if you are submitting one paper, we recommend you use `<firstname_surname>`
  - if you are submitting more than one paper, you will need to use a different
    directory name for each paper
- Copy an example paper into your directory: either `papers/00_myst_template` or `papers/00_tex_template`
  - Update the `id` in the `myst.yml` to by `scipy-2024-<your_directory_name>`

#### Write your paper

- To have a live preview of your changes:
  - Change directories `cd papers/<your_directory_name>`
  - Run `myst start` and open the web-server provided
- Refer to the syntax in the template papers or online at [mystmd.org](https://mystmd.org/guide)
- Update the author information and affiliations in `myst.yml`
- As you make changes to your paper, commit those changes in discrete chunks
- If you come across any challenges, ask the Proceedings Co-chairs for help via a GitHub issue or comment on your PR

Note: The templates are setup for a _single_ MyST/LaTeX file in the top level of `<your_directory_name>`. If you have more than one file run `myst init --write-toc` ([docs](https://mystmd.org/guide/table-of-contents)), ensuring that the `root` is the main file of your manuscript.

#### Commit your changes

- Commit any changes inside the `papers/<your_directory_name>`
- When you push your commits to your PR's branch, the paper will be auto-built in GitHub actions
- Do not commit any changes to files outside of your paper directory

If you want to change the way the build system works, we use a separate
submission procedure ([see below](#creating-build-system-prs)).

#### Preview your paper

Your paper will be **edited and reviewed in HTML**, the PDF will only be built on acceptance.

To preview your paper:

- Ensure `mystmd` is installed ([guide](https://mystmd.org/guide/quickstart))
- In `papers/<your_directory_name>` run `myst start`
- Open the web-server from your console
- Check that this output matches what is built on your PR

#### Create a paper PR

Once you are ready to submit your paper, make a pull request on GitHub. **Please ensure that you file against the correct branch.**

- Create a pull request against the `2024` branch
- Do not modify any files outside of your paper directory. Create a separate PR for any changes to the build system.
- Ensure that your PR title begins with `Paper:`.

#### Creating build system PRs

If you want to change the way the build system works, the documentation, etc., we use a separate submission procedure.

- Create a new branch against `dev`
- Make your changes to the build system
- Do **not** commit any changes from your paper PR to this new branch
- Make a separate PR against the `dev` branch, it will be reviewed separately

#### Push to your PR

When you push to your repositories branch it automatically run GitHub actions on the PR.
Note that this will require authorization for your first commit only.
The build process takes about a minute, and then posts or updates a comment on the PR with a link to the build result on Curvenote. The build page has a link to your preview.

#### Check your paper's build

The review process will be completed on the HTML, and you can check to see if the paper(s) that you preview locally match the paper(s) that you see online. These will be available in a GitHub comment or through the logs in the GitHub action.

If it is not the same, please immediately contact us with a GitHub issue
describing the discrepancy. Please include screenshots and an explanation of the
differences. For best results, please [@-mention the Proceedings Co-Chairs](#contacting-the-proceedings-co-chairs).

#### A note on notebooks for 2024
We are interested in working towards full support for publishing computational notebooks as part of the proceedings, and are trialing this part of the submission process for interested authors - please get in touch with the Proceedings Co-Chairs with your interest.

## Instructions for Reviewers

You will be reviewing authors' pull requests. While authors should have a proper
draft of their paper ready for you by the _Deadline to submit first draft_.

We ask that you read [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/2024/review_criteria.md) before beginning any reviews.

**All** communication between authors and reviewers should be civil and respectful at all times.

The goal of our review process is to improve the paper that the authors are
working on. Our aim is to have you and the author collaborate on making their
better by using an iterative process.

While our basic approach is to have you and the author iterate, we ask you to
complete an initial review and start that conversation by the _Initial Complete Review_
deadline.

We ask that by the _Final Reviewer Decision_ deadline you have a recommendation to
either **accept** or **reject** the paper at that point and time.

**Note**:
You many recommend changes after the _Final Reviewer Decision_ deadline. If there
are any major changes after the _Final Reviewer Decision_ deadline you should
immediately contact the Proceedings Committee Co-Chairs. As a heuristic, if you
think the paper should not be in the proceedings unless the authors make the
change in question, then that change should be requested and made before the
_Final Reviewer Decision_ deadline.

### Reviewer Deadlines

- Apr 9:  Reviewer invitations sent
- Apr 23: Deadline to respond to offer to be a reviewer
- May 31: Assignment of reviewers to papers
  - Reviewers comment on papers to authors during this period.
  - Authors also respond to review comments with improvements to papers during this period.
- Jun 21: Initial complete review
  - Reviewers continue to comment on paper improvements during this period.
  - Authors also respond to review comments with further improvements to papers during this period.
- Jul 26: Final review deadline
  - Authors continue to make revisions in response to final review comments during this period.
- Jul 31: Final author revision deadline
  - Authors put down their pens.
  - Reviewers make an up or down decision on publication readiness of papers during this period.
- Aug 9:  Final reviewer decision deadline

### Reviewer Workflow

- Read [this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/2024/review_criteria.md)
- Click on the Pull Requests Tab and find the papers assigned to you
- A comment at the top of the PR will have a link to the paper to review online
- After reading the paper online, you can start the review conversation however you prefer
  - You can use in-line comments (on the paper itself) or high-level comments.
- Authors will respond to your comments, possibly via their own comments or by
  modifying their paper.
- This begins an iterative review process where authors and reviewers can discuss the
  evolving submission.
- By the _Final Reviewer Decision_ deadline, we ask that you give two things
  1. A comprehensive review of the paper as it stands. This will act as the final
     review.
  2. A final recommendation to include the paper in the proceedings or not.
     - When you make the Final Recommendation, please [contact the proceedings Co-Chairs](#contacting-the-proceedings-co-chairs) in the PR in question.

## Review Criteria

A small subcommittee of the SciPy 2017 organizing committee has created
[this set of suggested review criteria](https://github.com/scipy-conference/scipy_proceedings/blob/2024/review_criteria.md)
to help guide authors and reviewers alike. Suggestions and amendments to these
review criteria are enthusiastically welcomed via discussion or pull request.

## Requirements

- MyST Markdown (`mystmd`) and NodeJS (>18)
- GitHub actions for the build process

## Build Process

The build process is completed through GitHub actions on every commit.
A comment is posted after the build process completes with a list of checks
and a link to the built output on Curvenote.

**Authors**: you should check to ensure that your local builds match the papers
built online. Please create an issue if they do not match.

**Reviewers**: You should be able to see the built article from the GitHub comment, and review from the preview link.

## For organisers

### Instructions for Publishers

To information about how to manage the whole proceedings, please see
`publisher/README.md` and `publisher/Makefile`.

#### Publisher Deadlines

- Apr 26: Authors invited to submit full papers
  - The [build process](#build-process) is supported by Curvenote (a SciPy sponsor) and it is maintained throughout this period.
- Aug 16: Proceedings final sign-off by editors
  - The publication process begins after final sign-off.

### Instructions for Editors

As reviewers review papers, editors should apply **labels** to the PR to flag the
current state of the review process. All paper PRs must have the `paper` label before the GitHub action will be triggered. Additionally, as editors and reviewers are assigned, the editors should add the reviewers GitHub handles to the PR summary comment.

Other **labels** that should be used are:

- **needs-more-review** if the paper needs further review,
- **pending-comment** if the paper is waiting on an authors' response, or
- **unready** if the paper is not ready for the proceedings.

Editors should come to a final 'ready', 'unready' decision before the **Final Editorial Decisions for Proceedings Contents** deadline.

#### Editor Deadlines

- Apr 9:  Reviewer invitations sent
- Apr 23: Deadline to respond to offer to be a reviewer
- Apr 26: Authors invited to submit full papers
- May 31: Assignment of reviewers to papers
  - Reviewers comment on papers to authors during this period.
  - Authors also respond to review comments with improvements to papers during this period.
- Jun 21: Initial complete review
  - Reviewers continue to comment on paper improvements during this period.
  - Authors also respond to review comments with further improvements to papers during this period.
  - Editors should verify that reviews have been completed
- Aug 16: Proceedings final sign-off by editors
  - The publication process begins after final sign-off.

## Instructions for Slides

#### Slide/Poster submission steps

1. Get a local copy of the `scipy_proceedings` repo.
2. Update your local copy of the `scipy_proceedings` repo.
3. [Create a new branch](#creating-a-new-branch) for your paper based off the latest `2024` branch.
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
