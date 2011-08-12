#!/usr/bin/env bash
# Concatenate all proceedings PDFs

if [[ ! -d output ]]; then
    echo -e "\nRun this script from the proceedings root instead."
    exit -1
fi

PDFS=`find ./output -name "paper.pdf" | sort`
pdftk output/toc.pdf $PDFS cat output output/proceedings.pdf

