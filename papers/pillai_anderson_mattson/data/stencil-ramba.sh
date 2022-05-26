#!/bin/bash
#SBATCH --partition=clxtrb
#SBATCH --time=3:00:00
#XXSBATCH --exclude=pcl-skx16,pcl-skx24

# Must specify --nodes n and SIZE=xxx;  should specify -o to get right log name

srun hostname
for i in `seq 10`; do
  OMP_NUM_THREADS=28  RAMBA_USE_ZMQ=1 RAMBA_NUM_THREADS=28 I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=compact mpirun -np $(( $SLURM_JOB_NUM_NODES * 2 +1 )) -ppn 2 python -u ~/ramba/tests/stencil-ramba3.py 20 $SIZE
  #OMP_NUM_THREADS=56  RAMBA_USE_ZMQ=1 RAMBA_NUM_THREADS=28 I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=scatter KMP_AFFINITY=granulariuty=fine,compact,1,0 mpirun -np $(( $SLURM_JOB_NUM_NODES * 2 +1 )) -ppn 2 python -u ~/ramba/tests/stencil-ramba3.py 20 $SIZE
done

