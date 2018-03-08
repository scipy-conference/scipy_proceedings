# For editors publishing the proceedings.

## Structure of the proceedings

This section will give you an introduction to the various documents and resources that you will need to create in the course of managing the proceedings. 

The proceedings themselves made up of 2 primary parts, the front-matter and the papers. Building the proceedings is only one of the tasks you will need to handle in the course of managing the proceedings, but that is what we'll focus on for right now.

### Front-matter:

The front-matter is composed of:

- title section
- the copyright section
- the organinization section: shows the committee chairs and reviewers
- the students section: shows the students who have been given scholarships

### Papers

The papers are a continuous sequence of the individual papers submitted by authors via PRs.

Building an individual paper is done by running build_paper.py on the paper directory.

In order to ensure that the papers will appear in order with the correct page numbers, you need to build all of them at once. This is the distinction between running build_papers.py

## Structure of the website

In addition to the proceedings pdf, you will need to construct the html files needed to share the proceedings with the world.

The html pages are derived from a few different templates:

- the main landing page (index.html.tmpl)
- a header shared across the html pages (header.html.tmpl)
- pages for the individual articles (article.html.tmpl)
- a page for the information about the conference organisers (organization.html.tmpl) 
- a page for the information about students receiving scholarships (students.html.tmpl)

The resulting html files will include:

- `index.html`
- a page for each article, named ``<article_author>.html`
- a `bib/`` directory, containing:
    - a bibfile for each article, named `<article_author>.bib`
    - a bibfile for the complete proceedings, named `proc-scipy-<year of proceedings>.bib`
- a `pdf/` directory, containing:
    - a pdf for each article, named `<article_author>.pdf`
    - a pdf for the complete proceedings, named `proceedings.pdf`

## Building the proceedings: Makefile

There are many make targets available in the Makefile, some of them are built of others.

That means you tend to use the different targets when you want to see a complete published version of the proceedings

### When publishing

To see the complete version of the proceedings you need to build everything to do that run:

- `make proceedings-html` 

If you need to share this version of the proceedings you can automatically zip it into a single file by running:

- `make proceedings-html-zip`

### When working on the build system

When you are working with the build system, you will often want to build only some parts of the proceedings. If you want to do that you can use the following commands

1. `make papers`: builds a pdf composed of all of the papers, as well as pdfs for the individual papers
2. `make front-pdf`: builds the pdfs for the front-matter elements
3. `make html`: builds the html pages for displaying the proceedings and papers
4. `make zip`: builds a zip file for the html and proceedings
5. `make proceedings`: builds the pdf of the proceedings (front-matter + papers)
6. `make proceedings-html`: builds the proceedings and then builds the html (proceedings + html)
7. `make html-zip`: builds the html, and then zips the proceedings as they are (html + zip)

NB: You will tend to use `html-zip` when you want to iterate on the website without needing to rebuild the entire proceedings.

## Build styles

There are three different modes for publishing the proceedings, you will need to
set the mode inside the `publisher/conf.py` file under the `status_file_base`
value. The main use of this feature is to make it easier to switch from "draft"
(the default) to a "conference ready" or "ready" version of the proceedings that
can be served on the official SciPy organisation website. 

- "draft": (default)
  The draft mode of the proceedings.
  This should be kept as the value in the living repository.
- "conference":
  This is the mode of the proceedings that can be published in time for the 
  conference itself. The watermark is lighter and it indicates that only small
  changes will be applied (e.g., adding video links once they are posted)
- "ready":
  This is the version of the proceedings with no watermark. This should only
  be used to publish the final version of the proceedings.
