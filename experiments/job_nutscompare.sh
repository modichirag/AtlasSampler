#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH -J compnuts
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
nsamples=50001

for stepfac in 1.1 1.2 ;
do
    srun python -u compare_nuts.py --exp $exp -n $n --target_accept $target --n_chains $nchains --n_samples $nsamples --max_treedepth 10 --n_burnin 0 --step_factor $stepfac
done

echo "done"
    

