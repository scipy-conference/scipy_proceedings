.PHONY: proceedings

proceedings:
	# Build all papers
	./make_all.sh
	# Count page nrs and build toc
	./publisher/build_index.py
	# Build again with new page numbers
	./make_all.sh
	# Build front material
	publisher/build_template.py cover_material/organization.tex.tmpl cover_material/committees.json > output/organization.tex
	publisher/build_template.py cover_material/toc.tex.tmpl output/toc.json > output/toc.tex
	publisher/build_template.py cover_material/toc.html.tmpl output/toc.json > output/toc.html
	(cd output && pdflatex toc.tex)
	(cd output && pdflatex organization.tex)
	# Concatenate front material and paper PDFs
	./publisher/concat_proceedings_pdf.sh

