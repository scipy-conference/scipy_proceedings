manuscript = main
references = $(wildcard *.bib)
latexopt   = -halt-on-error -file-line-error

all-pygments: all-via-pdf-pygments

all: all-via-pdf

all-via-pdf-pygments: $(manuscript).tex $(references)
	pdflatex $(latexopt) --shell-escape $<
	bibtex $(manuscript).aux
	pdflatex $(latexopt) --shell-escape $<
	pdflatex $(latexopt) --shell-escape $<

all-via-pdf: $(manuscript).tex $(references)
	pdflatex $(latexopt) --shell-escape $<
	bibtex $(manuscript).aux
	pdflatex $(latexopt) $<
	pdflatex $(latexopt) $<

all-via-dvi: 
	latex $(latexopt) $(manuscript)
	bibtex $(manuscript).aux
	latex $(latexopt) $(manuscript)
	latex $(latexopt) $(manuscript)
	dvipdf $(manuscript)

epub: 
	latex $(latexopt) $(manuscript)
	bibtex $(manuscript).aux
	mk4ht htlatex $(manuscript).tex 'xhtml,charset=utf-8,pmathml' ' -cunihtf -utf8 -cvalidate'
	ebook-convert $(manuscript).html $(manuscript).epub

clean:
	rm -f *.pdf *.dvi *.toc *.aux *.out *.log *.bbl *.blg *.log *.spl *~ *.spl *.zip *.acn *.glo *.ist *.epub *.glsdefs *.lof *.lot

realclean: clean
	rm -rf $(manuscript).dvi
	rm -f $(manuscript).pdf

%.ps :%.eps
	convert $< $@

%.png :%.eps
	convert $< $@

zip:
	zip paper.zip *.tex *.eps *.bib

.PHONY: all clean
