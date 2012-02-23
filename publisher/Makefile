BUILDDIR = _build
TEXDIR = _build/tex
HTMLDIR = _build/html
PDFDIR = _build/pdfs
PAPERDIR = ../output
TEMPLATES = _templates
STATIC = _static

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

proceedings: front-pdf papers $(TEXDIR)/proceedings.tex
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)
	cp $(TEXDIR)/proceedings.pdf $(PDFDIR)/proceedings.pdf 

proceedings-html: proceedings
	python build_html.py
	-convert $(STATIC)/logo.png -resize x120 $(HTMLDIR)/logo.png
