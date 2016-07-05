poster.pdf:

eos_nom_true.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure1eos.pdf $@
real.pdf:
	ln -s ../papers/andrew_fraser/fit_v.pdf $@
stick_CJ.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure1.pdf $@
stick_fisher.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure2.pdf $@
stick_xt.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure3.pdf $@
gun_tv.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure4.pdf $@
gun_fisher.pdf:
	ln -s ../papers/andrew_fraser/scipy2016_figure5.pdf $@

FIGS =  eos_nom_true stick_xt stick_CJ stick_fisher gun_tv gun_fisher\
real
FIGURES = $(patsubst %, %.pdf, $(FIGS)) 

poster.pdf: poster.tex ${FIGURES}
	pdflatex poster
###---------------
### Local Variables:
### mode: makefile
### End:
