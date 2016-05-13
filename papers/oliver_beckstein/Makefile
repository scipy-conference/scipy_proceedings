# quick and dirty paper generation (remove before final PR)

NAME := oliver_beckstein
PAPERDIR := papers/$(NAME)
OUTPUTDIR := output/$(NAME)

RST  := $(NAME).rst
BIB  := mdanalysis.bib
PDF  := paper.pdf

TOPDIR  := $(shell git rev-parse --show-toplevel)

.phony: all pdf

all: pdf

pdf: $(PDF)

$(PDF): $(RST) $(BIB)
	cd $(TOPDIR) && \
	./make_paper.sh $(PAPERDIR) && \
	cp -v $(OUTPUTDIR)/$(PDF) $(PAPERDIR)/
