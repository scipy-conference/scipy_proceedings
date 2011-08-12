.PHONY: proceedings

proceedings:
	# Build all papers
	./make_all.sh
	# Count page nrs and build toc
	./publisher/build_index.py
	# Build twice to get correct references
	./make_all.sh
	./make_all.sh
	# Concatenate front material and paper PDFs
	./publisher/concat_proceedings_pdf.sh

