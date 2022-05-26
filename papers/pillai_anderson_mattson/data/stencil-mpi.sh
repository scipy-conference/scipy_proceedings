#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#XSBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

srun hostname
for i in `seq 10`; do
  OMP_NUM_THREADS=1 I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=compact mpirun -np $(( $SLURM_JOB_NUM_NODES * 56 )) -ppn 56 ~/Kernels/MPI1/Stencil/stencil 20 $SIZE
done

