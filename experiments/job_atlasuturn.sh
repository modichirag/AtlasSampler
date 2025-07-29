#!/bin/bash
#SBATCH -p ccm
#SBATCH -C ib-icelake
#SBATCH --time=6:00:00
#SBATCH -J adaptive
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

exp=$1
n=$2
offsetd=$3
echo $exp $n

nchains=16
nsamples=5001


stepdist="lognormal"
for offset in 1.0 ; #0.33 0.5 0.66 ;
do for stepsig in 2 1.5 1.2 1.1 ;        
   do srun -n $nchains python -u compare_atlasuturn.py --exp $exp -n $n --n_samples $nsamples --offset $offset --stepsize_distribution $stepdist --offset_delayed $offsetd --stepsize_sigma $stepsig
   done
done

  
stepdist="beta"
for offset in 1.0 ; #0.33 0.5 0.66 ;
   do srun -n $nchains python -u compare_atlasuturn.py --exp $exp -n $n --n_samples $nsamples --offset $offset --stepsize_distribution $stepdist --offset_delayed $offsetd
done

echo "done"
 
