BUILDDIR = _build
TEXDIR = _build/tex
HTMLDIR = _build/html
PDFDIR = _build/pdfs
PAPERDIR = ../output
TEMPLATES = _templates
STATIC = _static
YEARDIR = scipy`date "+%Y"`
ZIPNAME = draft_proceedings.zip
DOIXMLNAME = doi_batch.xml

BUILDTMPL = ./build_template.py
TEX2PDF := cd $(TEXDIR) && pdflatex -interaction=batchmode

.PHONY: front-pdf proceedings papers toc clean

all: clean proceedings

clean:
	rm -rf $(PAPERDIR)/* $(BUILDDIR)/*

$(TEXDIR):
	mkdir -p $@

$(TEXDIR)/%.tex: $(TEXDIR)
	$(BUILDTMPL) $(@F)

$(TEXDIR)/%.pdf: $(TEXDIR)/%.tex
	($(TEX2PDF) $(<F) 1>/dev/null)

title-pdf: $(TEXDIR)/title.pdf
copyright-pdf: $(TEXDIR)/copyright.pdf
organization-pdf: $(TEXDIR)/organization.pdf
students-pdf: $(TEXDIR)/students.pdf

front-pdf: title-pdf copyright-pdf organization-pdf students-pdf

papers:
	./build_papers.py

proceedings: papers $(TEXDIR)/proceedings.tex front-pdf
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)
	cp $(TEXDIR)/proceedings.pdf $(PDFDIR)/proceedings.pdf

html:
	python build_html.py
	-convert $(STATIC)/logo.png -resize x100 $(HTMLDIR)/logo.png

zip:
	cp -r $(HTMLDIR) $(YEARDIR)
	cp $(BUILDDIR)/$(DOIXMLNAME) $(YEARDIR)/$(DOIXMLNAME)
	zip -r $(ZIPNAME) $(YEARDIR)
	rm -rf $(YEARDIR)

html-zip: html zip

proceedings-html: proceedings html

proceedings-html-zip: proceedings html-zip
