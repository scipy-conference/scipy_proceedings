all: paper figs

paper:
	cd ../../; ./make_paper.sh papers/bergstra/; cd papers/bergstra;

figs:
	enscript -B --line-numbers=1 --color -E -p - logreg1.py | ps2pdf - logreg1.pdf
	pdfcrop --margins '0 0 0 10' logreg1.pdf logreg1.pdf
	enscript -B --line-numbers=12 --color -E -p - logreg2.py | ps2pdf - logreg2.pdf
	pdfcrop --margins '0 0 0 10' logreg2.pdf logreg2.pdf
	enscript -B --line-numbers=18 --color -E -p - logreg3.py | ps2pdf - logreg3.pdf
	pdfcrop --margins '0 0 0 10' logreg3.pdf logreg3.pdf
	enscript -B --line-numbers=24 --color -E -p - logreg4.py | ps2pdf - logreg4.pdf
	pdfcrop --margins '0 0 0 10' logreg4.pdf logreg4.pdf
	enscript -B -C --color -E -p - fig_simple.py | ps2pdf - fig_simple.pdf
	pdfcrop --margins '0 0 0 10' fig_simple.pdf fig_simple.pdf
