#!/bin/bash

DIR=$1


python publisher/build_paper.py $DIR
if [ "$?" -ne "0" ]; then
    echo "Error building paper $DIR. Aborting."
    exit 1
fi

#cd $OUTDIR
#$TEX2PDF > /dev/null && $TEX2PDF | (python $WD/publisher/page_count.py)
