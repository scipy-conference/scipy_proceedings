#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#XSBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

echo Dask workers on:
srun hostname

master=`hostname`

echo
echo Dask scheduler on $master
echo

(cd /tmp; dask-scheduler)&
sleep 5
(cd /tmp; OMP_NUM_THREADS=112 mpirun -np $SLURM_JOB_NUM_NODES -ppn 1 dask-worker --nworkers 1 tcp://$master:8786 )&

nt=0
while [ $nt -lt $(( $SLURM_JOB_NUM_NODES * 112 )) ]; do
    sleep 5
    nt=`DASK_MASTER=$master python test-dask.py`
    echo $nt threads up
done
echo Workers ready
echo `date`  START

for i in `seq 10`; do
  python -u ~/Kernels2/DASK/stencil-numpy.py 20 $SIZE
done

echo `date`  STOP
echo Terminating Dask scheduler and workers


