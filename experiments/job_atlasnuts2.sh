#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH -J atlasnuts2
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

export OMP_NUM_THREADS=1

exp=$1
n=$2
ctraj=$3
stepdist=$4
echo $exp $n

nchains=32
nsamples=10001

#prob=1
#hessian_mode='bam-bfgsinit'
#suffix='reg1'
# stepsig=1.2
# stepdist='lognormal'
# nleapdist='constant'
# nleap=100
# combinechains=0
#time srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist  --step_factor $stepfac --probabilistic $prob  --n_leapfrog $nleap


if [ "$stepdist" == "beta" ] ; then 
    for nleapdist in    "uniform"  "empirical";
    do for combinechains in  0 1  ;
        do
            srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples   --combine_chains $combinechains   --constant_trajectory $ctraj  --nleap_distribution $nleapdist
            #--step_factor $stepfac --probabilistic $prob   --hessian_mode $hessian_mode --stepsize_distribution $stepdist  # --suffix $suffix
        done
    done
fi



if [ "$stepdist" == "lognormal" ] ; then 
    for stepfac in  1. 1.1 1.2 1.25 ; do 
        for combinechains in  1   ; do
            for stepsig in  1.2   ; do
                for nleapdist in  "uniform"   ;  do
                    srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist --max_nleapfrog 2048 --step_factor $stepfac
                    # --step_factor $stepfac --probabilistic $prob  --hessian_mode $hessian_mode  # --suffix $suffix
                done
            done
        done
    done
fi
echo "done"
