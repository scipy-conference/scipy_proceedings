.PHONY: proceedings

proceedings:
	# Build all papers
	./make_all.sh
	# Count page nrs and build toc
	./publisher/build_index.py
	# Build again with new page numbers
	./make_all.sh
	# Build front material
	(cd output && pdflatex toc.tex)
	cp cover_material/organization.tex output
	(cd output && pdflatex organization.tex)
	#cp cover_material/*.pdf output
	# Concatenate front material and paper PDFs
	./publisher/concat_proceedings_pdf.sh


