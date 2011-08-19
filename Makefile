TEX2PDF := cd output && pdflatex -interaction=batchmode
MKTEMPLATE := publisher/build_template.py cover_material/TARGET.tmpl JSON > output/TARGET
PROCTEMPLATE := $(subst JSON,scipy_proc.json,$(MKTEMPLATE))
TOCTEMPLATE := $(subst JSON,output/toc.json,$(MKTEMPLATE))

.PHONY: front-pdf proceedings papers toc clean

all: clean proceedings

clean:
	rm -rf output/*

front-pdf:
	$(MAKE) -C cover_material $@

papers:
	./make_all.sh

toc: papers 
	$(subst TARGET,toc.tex,$(TOCTEMPLATE))
	$(subst TARGET,toc.html,$(TOCTEMPLATE))
	cp cover_material/toc.css output/
	($(TEX2PDF) toc 1>/dev/null)

proceedings: front-pdf papers
	$(subst TARGET,proceedings.tex,$(TOCTEMPLATE))
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)
