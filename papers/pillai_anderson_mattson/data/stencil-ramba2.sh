#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#SBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

srun hostname
for i in `seq 10`; do
  NUMBA_CACHE_DIR=/tmp OMP_NUM_THREADS=2  RAMBA_USE_ZMQ=1 RAMBA_NUM_THREADS=1 I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=compact mpirun -np $(( $SLURM_JOB_NUM_NODES * 56 +1 )) -ppn 56 python -u ~/ramba/tests/stencil-ramba3.py 20 $SIZE
  #OMP_NUM_THREADS=2  RAMBA_USE_ZMQ=1 RAMBA_NUM_THREADS=1 mpirun -np $(( $SLURM_JOB_NUM_NODES * 56 +1 )) -ppn 56 python -u ~/ramba/tests/stencil-ramba3.py 20 $SIZE
done

