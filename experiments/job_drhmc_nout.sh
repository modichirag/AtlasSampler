#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH -J drhmcnout
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

export OMP_NUM_THREADS=1

exp=$1
n=$2
echo $exp $n

nchains=30
nsamples=2001

#prob=1
#nleapadapt=100
#ctraj=2
#hessian_mode='bam-bfgsinit'
#suffix='reg1'
# stepsig=1.2
# stepdist='lognormal'
# nleapdist='constant'
# nleap=100
# combinechains=0
#time srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist  --step_factor $stepfac --probabilistic $prob  --n_leapfrog $nleap



for stepfac in  2 5 10   ; do 
    for combinechains in  0 1    ; do
        srun -n $nchains python -u compare_drhmc_nout.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --max_nleapfrog 2048 --step_factor $stepfac 
    done
done
