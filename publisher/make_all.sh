#!/bin/bash

PDFDIR=_build/pdfs

# Build all papers
for d in ../papers/*; do
    ARTICLE=`basename $d`
    (cd .. && ./make_paper.sh papers/$ARTICLE)
done

# Count page nrs and build toc
./build_index.py

# Build again with new page numbers
mkdir -p $PDFDIR
for d in ../papers/*; do
    ARTICLE=`basename $d`
    (cd .. && ./make_paper.sh papers/$ARTICLE)
    PAPERDIR="../output/$ARTICLE"
    cp $PAPERDIR/paper.pdf $PDFDIR/$ARTICLE.pdf
    (cd $PDFDIR && pdfannotextractor $ARTICLE.pdf)
done
