#!/bin/bash

if [ $# -eq 0 ]; then
   echo "Usage: $0 file.log [file.log ...] > output.csv"
   exit 1
fi

for lf in "$@"; do
   base=`echo $lf | cut -d. -f1 | cut -d _ --output-delimiter , -f1-`
   rates=`grep Rate $lf | awk "{ print \\$3 }"`
   echo -n $base
   for r in $rates; do
      echo -n ,$r
   done
   echo
done

