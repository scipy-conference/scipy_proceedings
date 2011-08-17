.PHONY: proceedings organization papers toc cover clean copyright

all: clean proceedings

clean:
	rm -rf output/*

cover:
	inkscape --export-dpi=600 --export-pdf=output/cover.pdf cover_material/cover.svg

organization:
	publisher/build_template.py cover_material/organization.tex.tmpl scipy_proc.json > output/organization.tex
	publisher/build_template.py cover_material/organization.html.tmpl scipy_proc.json > output/organization.html
	(cd output && pdflatex organization.tex)

copyright:
	publisher/build_template.py cover_material/copyright.tex.tmpl scipy_proc.json > output/copyright.tex
	(cd output && pdflatex copyright.tex)

papers:
	./make_all.sh

toc: papers 
	publisher/build_template.py cover_material/toc.tex.tmpl output/toc.json > output/toc.tex
	publisher/build_template.py cover_material/toc.html.tmpl output/toc.json > output/toc.html
	cp cover_material/toc.css output/
	(cd output && pdflatex toc.tex)

proceedings: cover copyright organization papers
	publisher/build_template.py cover_material/proceedings.tex.tmpl output/toc.json > output/proceedings.tex
	(cd output && pdflatex proceedings.tex)
