# For editors publishing the proceedings.

**Note**: This is changing in 2024 and aspects of this README are out of date.

## Structure of the proceedings

This section will give you an introduction to the various documents and resources that you will need to create in the course of managing the proceedings.

The proceedings themselves are made up of 2 primary parts, the front-matter and the papers.

This section focuses on building the pdfs for the proceedings, other sections will focus on the html and other metadata information.

### Front-matter:

The front-matter is composed of:

- title section
- the copyright section
- the organinization section: shows the committee chairs and reviewers
- the students section: shows the students who have been given scholarships
- the listing of posters, plenaries, and slides

### Papers

The papers are a continuous sequence of the individual papers submitted by authors via PRs.

Building an individual paper is done by running build_paper.py on the paper directory.

In order to ensure that the papers will appear in order with the correct page numbers, you need to build all of them at once. This is the distinction between running build_papers.py and running build_paper.py on each of the individual papers.

## Structure of the website

In addition to the proceedings pdf, you will need to construct the html files needed to share the proceedings with the world.

The html pages are derived from a few different templates:

- the main landing page (index.html.tmpl)
- a header shared across the html pages (header.html.tmpl)
- pages for the individual articles (article.html.tmpl)
- a page for the information about the conference organisers (organization.html.tmpl)
- a page for the information about students receiving scholarships (students.html.tmpl)
- a page for the slides and posters (slides.html.tmpl)

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

There are many more make targets available in the Makefile than we will discuss here.

You will need to use different targets when you want to see a complete published version of the proceedings vs. individual parts of the proceedings.

### Publishing complete proceedings

To build the complete version of the proceedings (html & pdfs) run:

- `make proceedings-html`

Then, you can view the website by running `open _build/html/index.html`.

If you need to share the proceedings, it is easier to zip the files automatically by running:

- `make proceedings-html-zip`

### Working on the build system

When you are working with the build system, you will often want to iterate
between making changes the build system and building parts of the
proceedings.

The following commands are some of the most useful for partial builds:

1. `make papers`: builds pdfs for the individual papers
2. `make front-pdf`: builds the pdfs for the front-matter elements
3. `make html`: builds the html pages for displaying the proceedings and papers
4. `make proceedings`: builds the pdf of the proceedings (front-matter + papers)
5. `make html-zip`: builds the html, and then zips the proceedings as they are (html + zip)

NB: You will tend to use `html-zip` if you need to iterate on the portable copy
of the website without needing to rebuild the entire proceedings. This is most
useful after authors' PRs are no longer being updated.

## Requirements

These requirements are for installing LaTeX, which should not be necessary for MyST based submissions.

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
  be used to publish the final version of the proceedings. Using "ready" is
  responsible for triggering the creation of DOIs and XML submission files.

## DOI metadata

As of 2015, each SciPy conference proceedings and the individual papers in
one proceedings are assigned DOIs. As of 2018, the conference itself has a
DOI, such that the linking structure now looks the following:

- ISSN + Series DOI (10.25080/issn.2575-9752)
  - Conference DOI for SciPy 2018
  - Conference DOI for SciPy 2017
    - Paper DOI for Poore 2017
    - Paper DOI for Niederhut 2017
    - etc.

The organization that administers ISSN requests that [the series-level DOI
be constructed following these guidelines]
(http://www.issn.org/understanding-the-issn/issn-uses/use-of-issn-in-doi/).
The series DOI is static and is stored in `scipy_proc.json`. The logic for
dynamically generated proceedings and paper DOIs is contained in `doitools.py`.
DOIs are built from:

1. Nodename
2. Latest commit
3. Ordinal of paper

This ends up looking like `10.25080/shinma-7f4c6e7-002`. When displaying DOIs,
crossref.org asks that we use this format -
[https://doi.org/10.25080/shinma-7f4c6e7-002]
(https://doi.org/10.25080/shinma-7f4c6e7-002)

- which resolves the proceedings page where the paper is hosted.

Submitting DOIs and associated metadata about each paper to crossref.org
requires an xml file with the [following
schema](http://data.crossref.org/reports/help/schema_doc/4.4.1/index.html).
The logic for doing this is in `xreftools.py`.
