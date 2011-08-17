TEX2PDF := cd output && pdflatex -interaction=batchmode

.PHONY: proceedings organization papers toc cover clean copyright

all: clean proceedings

clean:
	rm -rf output/*

cover:
	inkscape --export-dpi=600 --export-pdf=output/cover.pdf cover_material/cover.svg

organization:
	publisher/build_template.py cover_material/organization.tex.tmpl scipy_proc.json > output/organization.tex
	publisher/build_template.py cover_material/organization.html.tmpl scipy_proc.json > output/organization.html
	($(TEX2PDF) organization 1>/dev/null)

copyright:
	publisher/build_template.py cover_material/copyright.tex.tmpl scipy_proc.json > output/copyright.tex
	($(TEX2PDF) copyright 1>/dev/null)

papers:
	./make_all.sh

toc: papers 
	publisher/build_template.py cover_material/toc.tex.tmpl output/toc.json > output/toc.tex
	publisher/build_template.py cover_material/toc.html.tmpl output/toc.json > output/toc.html
	cp cover_material/toc.css output/
	($(TEX2PDF) toc 1>/dev/null)

proceedings: cover copyright organization papers
	publisher/build_template.py cover_material/proceedings.tex.tmpl output/toc.json > output/proceedings.tex
	($(TEX2PDF) proceedings 1>/dev/null)
	($(TEX2PDF) proceedings 1>/dev/null)

