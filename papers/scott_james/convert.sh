#!/bin/bash
pandoc -f markdown -t rst ~/Dropbox/notes/relate-scipy-2015-paper.txt -o relate-scipy.rst
cat header.rst relate-scipy.rst > relate-scipy2015.rst
rm relate-scipy.rst