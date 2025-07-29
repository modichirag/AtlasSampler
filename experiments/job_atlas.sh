#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
##SBATCH -C ib-rome
#SBATCH --time=2:00:00
#SBATCH -J atlas
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

export OMP_NUM_THREADS=1

exp=$1
n=$2
target=$3
ctraj=$4
echo $exp $n

nchains=32
nsamples=50001

stepdist='lognormal'
stepsig=1.2
combinechains=0
nleapdist='uniform'
hessian='bfgs'
nleapadapt=200

srun -n $nchains python -u atlas.py --exp $exp -n $n --n_samples $nsamples --stepsize_distribution $stepdist --combine_chains $combinechains  --stepsize_sigma $stepsig --target_accept $target --constant_trajectory $ctraj --max_nleapfrog 2048 --nleap_distribution $nleapdist --hessian_mode $hessian --n_leapfrog_adapt $nleapadapt --suffix "nladapt200"


# stepdist='beta'
# srun -n $nchains python -u atlas.py --exp $exp -n $n --n_samples $nsamples --stepsize_distribution $stepdist --combine_chains $combinechains  --stepsize_sigma $stepsig --target_accept $target --constant_trajectory $ctraj --max_nleapfrog 2048 --nleap_distribution $nleapdist --hessian_mode $hessian 


# for stepdist in "beta" "lognormal" ;
# do
#        echo $stepdist;
#        srun -n $nchains python -u atlas.py --exp $exp -n $n --n_samples $nsamples  --stepsize_distribution $stepdist --combine_chains $combinechains --target_accept $target
# done

# stepdist="lognormal"
# for stepsig in 1.2 1.1 1.5  ; 
# do
#     srun -n $nchains python -u atlas.py --exp $exp -n $n --n_samples $nsamples --stepsize_distribution $stepdist --combine_chains $combinechains  --stepsize_sigma $stepsig --target_accept $target
# done

echo "done"
    

