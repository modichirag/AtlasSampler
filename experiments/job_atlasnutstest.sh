#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH -J atlastest
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

export OMP_NUM_THREADS=1

exp='rosenbrockhy3'
n=1
exp='multifunnel'
n=10
stepdist='lognormal'
ctraj=2
nchains=32
nsamples=50001

hessian_mode='bfgs'
suffix='reg1'
stepsig=1.2
stepdist='lognormal'
nleapdist='uniform'
combinechains=0
time srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist --max_nleapfrog 2048 --n_hessian_attempts 10 --hessian_mode $hessian_mode --suffix 2048

