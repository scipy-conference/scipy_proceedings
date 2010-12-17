#!/bin/bash

DIR=$1
AUTHOR=`basename $DIR`
OUTDIR="output/$AUTHOR"

mkdir -p $OUTDIR
cp $DIR/* $OUTDIR
python publisher/build_paper.py $DIR $OUTDIR
cd $OUTDIR
pdflatex paper.tex && pdflatex paper.tex

