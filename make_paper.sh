#!/bin/bash

DIR=$1

if [[ ! -d $DIR ]]; then
  echo "Usage: make_paper.sh source_dir"
  exit -1
fi

AUTHOR=`basename $DIR`
OUTDIR="output/$AUTHOR"

mkdir -p $OUTDIR
cp $DIR/* $OUTDIR
python publisher/build_paper.py $DIR $OUTDIR
cd $OUTDIR
pdflatex paper.tex && pdflatex paper.tex

