#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#SBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

srun hostname
for i in `seq 10`; do
  OMP_NUM_THREADS=28 I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=compact KMP_AFFINITY=granularity=fine,compact,1,0 mpirun -np $(( $SLURM_JOB_NUM_NODES * 2 )) -ppn 2 ~/Kernels/MPIOPENMP/Stencil/stencil 28 20 $SIZE
done

