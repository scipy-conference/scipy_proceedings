# For editors publishing the proceedings.

## Building the proceedings: Makefile

There are a few commands of use when publishing the proceedings. 

The primary four tasks are:

1. Building the individual papers: `papers`
2. Building the front-matter: `front-pdf`
3. Building complete proceedings: `proceedings`
4. Creating the website to display the proceedings and papers: `html`
5. Zipping the website, proceedings and papers into a sharable file: `zip`

Each of these tasks can be performed individually by running `make <command>`
inside this directory. 

Additionally, these commands can be combined:

- `proceedings`: includes building `papers` and `front-pdf`
- `proceedings-html` builds the proceedings and then builds the html
- `html-zip` builds the html, and then zips the proceedings as they are
- `proceedings-html-zip` builds everything and then zips it up


## Build styles
There are three different modes for publishing the proceedings, you will need to
set the mode inside the `publisher/conf.py` file under the `status_file_base`
value. The main use of this feature is to make it easier to switch from "draft"
the default to a "conference ready" or "ready" version of the proceedings that
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
