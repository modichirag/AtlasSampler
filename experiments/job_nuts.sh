#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH -J nuts
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load cuda cudnn gcc slurm
source activate jaxenv

export OMP_NUM_THREADS=1

exp=$1
n=$2
target=$3

nchains=32
nsamples=2001

srun python -u nuts.py --exp $exp -n $n --target_accept $target --n_chains $nchains --n_samples $nsamples --max_treedepth 11 --metric 'diag_e'

echo "done"
    

