BUILDDIR = _build
TEXDIR = _build/latex
HTMLDIR = _build/html
PDFDIR = _build/pdfs
PAPERDIR = ../output
TEMPLATES = _templates

TEX2PDF := cd $(TEXDIR) && pdflatex -interaction=batchmode

.PHONY: front-pdf proceedings papers toc clean

all: clean proceedings

clean:
	rm -rf $(PAPERDIR)/* $(BUILDDIR)/*

papers:
	./make_all.sh

front-pdf:
	$(MAKE) -C $(TEMPLATES) $@

proceedings: front-pdf papers
	build_template.py $(TEMPLATES)/proceedings.tex.tmpl $(BUILDDIR)/toc.json > $(TEXDIR)/proceedings.tex
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)
	cp $(TEXDIR)/proceedings.pdf $(PDFDIR)/proceedings.pdf 

front-html:
	$(MAKE) -C $(TEMPLATES) $@

proceedings-html: front-html proceedings
	ln -s ../pdfs $(HTMLDIR)/pdfs
	cp $(TEMPLATES)/toc.css $(HTMLDIR)/.
	build_template.py $(TEMPLATES)/toc.html.tmpl $(BUILDDIR)/toc.json > $(HTMLDIR)/index.html
