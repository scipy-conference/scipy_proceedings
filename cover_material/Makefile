BUILDDIR = ../output
BUILDTMPL = ../publisher/build_template.py
PROC_CFG = ../scipy_proc.json

TEX2PDF := cd $(BUILDDIR) && pdflatex -interaction=batchmode

.PHONY: cover-pdf copyright-pdf organization-pdf organization-html 

$(BUILDDIR)/cover.pdf:
	inkscape --export-dpi=600 --export-pdf=$@ cover.svg

$(BUILDDIR)/%.html:
	$(BUILDTMPL) $(@F).tmpl $(PROC_CFG) > $@

$(BUILDDIR)/%.tex:
	$(BUILDTMPL) $(@F).tmpl $(PROC_CFG) > $@

$(BUILDDIR)/%.pdf: $(BUILDDIR)/%.tex
	($(TEX2PDF) $(<F) 1>/dev/null)

cover-pdf: $(BUILDDIR)/cover.pdf
copyright-pdf: $(BUILDDIR)/copyright.pdf
organization-pdf: $(BUILDDIR)/organization.pdf
organization-html: $(BUILDDIR)/organization.html

front-pdf: cover-pdf copyright-pdf organization-pdf
