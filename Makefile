TEX2PDF := cd output && pdflatex -interaction=batchmode
MKTEMPLATE := publisher/build_template.py cover_material/TARGET.tmpl JSON > output/TARGET
PROCTEMPLATE := $(subst JSON,scipy_proc.json,$(MKTEMPLATE))
TOCTEMPLATE := $(subst JSON,output/toc.json,$(MKTEMPLATE))

.PHONY: proceedings organization papers toc cover clean copyright

all: clean proceedings

clean:
	rm -rf output/*

cover:
	inkscape --export-dpi=600 --export-pdf=output/cover.pdf cover_material/cover.svg

organization:
	$(subst TARGET,organization.tex,$(PROCTEMPLATE))
	$(subst TARGET,organization.html,$(PROCTEMPLATE))
	($(TEX2PDF) organization 1>/dev/null)

copyright:
	$(subst TARGET,copyright.tex,$(PROCTEMPLATE))
	($(TEX2PDF) copyright 1>/dev/null)

papers:
	./make_all.sh

toc: papers 
	$(subst TARGET,toc.tex,$(TOCTEMPLATE))
	$(subst TARGET,toc.html,$(TOCTEMPLATE))
	cp cover_material/toc.css output/
	($(TEX2PDF) toc 1>/dev/null)

proceedings: cover copyright organization papers
	$(subst TARGET,proceedings.tex,$(TOCTEMPLATE))
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)

