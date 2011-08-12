.PHONY: proceedings

proceedings:
	./make_all.sh
	./publisher/build_index.py
	./make_all.sh
	./publisher/concat_proceedings_pdf.sh

