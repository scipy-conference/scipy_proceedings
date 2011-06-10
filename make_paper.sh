#!/bin/bash

DIR=$1
WD=`pwd`

if [[ ! -d $DIR ]]; then
  echo "Usage: make_paper.sh source_dir"
  exit -1
fi

AUTHOR=`basename $DIR`
OUTDIR="output/$AUTHOR"
if [ x${USER} == x"koning" ] ; then
PYTHON=~/Python/$SYS_TYPE/py2.6/bin/python
else
PYTHON=python
fi

mkdir -p $OUTDIR
cp $DIR/* $OUTDIR
${PYTHON} publisher/build_paper.py $DIR $OUTDIR
if [ "$?" -ne "0" ]; then
    echo "Error building paper $DIR. Aborting."
    exit 1
fi
cd $OUTDIR
pdflatex paper.tex && pdflatex paper.tex | (python $WD/publisher/paper_cut.py)

