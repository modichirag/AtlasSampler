#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH -J uturn
#SBATCH -o logs/%x.o%j
##SBATCH -C ib-rome

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

OMP_NUM_THREADS=1

exp=$1
n=$2
echo $exp $n 

nchains=32
nsamples=50001

srun -n $nchains python -u compare_uturn.py --exp $exp -n $n --n_samples $nsamples 

echo "done"
