#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#SBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

srun hostname
for i in `seq 10`; do
  python -u ~/ramba/tests/stencil-numpy.py 5 $SIZE
done

