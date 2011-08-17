#!/bin/bash

# Build all papers
for d in papers/*; do
    ./make_paper.sh $d
done

# Count page nrs and build toc
publisher/build_index.py

# Build again with new page numbers
for d in papers/*; do
    ./make_paper.sh $d 1>/dev/null
    AUTHOR=`basename $d`
    OUTDIR="output/$AUTHOR"
    (cd $OUTDIR && pdfannotextractor paper.pdf) 1>/dev/null
done
