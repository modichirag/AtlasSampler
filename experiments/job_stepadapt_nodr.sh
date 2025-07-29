#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
##SBATCH -C ib-rome
#SBATCH --time=2:00:00
#SBATCH -J sadaptnodr
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

export OMP_NUM_THREADS=1

exp=$1
n=$2
combinechains=$3
stepdist=$4
echo $exp $n 

nchains=32
nsamples=50001

if [ "$stepdist" == "beta" ] ; then 
   for nleapdist in   "uniform" "empirical"   ;
      do srun -n $nchains python -u compare_stepadapt_nodr.py --exp $exp -n $n --n_samples $nsamples  --combine_chains $combinechains   --nleap_distribution $nleapdist --stepsize_distribution $stepdist
   done
fi 

if [ "$stepdist" == "lognormal" ] ; then 
    for stepsig in  1.2    ; 
    do for nleapdist in "uniform"  "empirical"  ; 
         do srun -n $nchains python -u compare_stepadapt_nodr.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist --max_nleapfrog 1024
      done
   done
fi
