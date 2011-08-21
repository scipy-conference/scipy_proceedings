BUILDDIR = ../_build
TEXDIR = ../_build/latex
HTMLDIR = ../_build/html
BUILDTMPL = ../build_template.py
PROC_CFG = ../../scipy_proc.json
TOC_CFG = $(BUILDDIR)/toc.json

TEX2PDF := cd $(TEXDIR) && pdflatex -interaction=batchmode

.PHONY: cover-pdf copyright-pdf organization-pdf organization-html 

$(TEXDIR): 
	mkdir -p $@

$(HTMLDIR):
	mkdir -p $@

$(TEXDIR)/cover.pdf: $(TEXDIR)
	inkscape --export-dpi=600 --export-pdf=$@ cover.svg

$(HTMLDIR)/%.html: $(HTMLDIR)
	$(BUILDTMPL) $(@F).tmpl $(PROC_CFG) > $@

$(TEXDIR)/%.tex: $(TEXDIR)
	$(BUILDTMPL) $(@F).tmpl $(PROC_CFG) > $@

$(TEXDIR)/%.pdf: $(TEXDIR)/%.tex
	($(TEX2PDF) $(<F) 1>/dev/null)

cover-pdf: $(TEXDIR)/cover.pdf
copyright-pdf: $(TEXDIR)/copyright.pdf
organization-pdf: $(TEXDIR)/organization.pdf
organization-html: $(HTMLDIR)/organization.html

front-pdf: cover-pdf copyright-pdf organization-pdf
front-html: organization-html
