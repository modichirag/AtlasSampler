#!/bin/bash
#SBATCH -p ccm
#SBATCH -C ib-rome
#SBATCH --time=4:00:00
#SBATCH -J atlasnuts
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

exp=$1
n=$2
combinechains=$3
stepdist=$4
echo $exp $n

nchains=32
nsamples=50001
offset=1.0
ctraj=1


if [ "$stepdist" == "beta" ] ; then 
    for nleapdist in "empirical" "uniform"   ;
    do srun -n $nchains python -u compare_atlasnuts3.py --exp $exp -n $n --n_samples $nsamples   --combine_chains $combinechains   --constant_trajectory $ctraj  --nleap_distribution $nleapdist  
    done
fi

if [ "$stepdist" == "lognormal" ] ; then 
    for nleapdist in "empirical" "uniform"   ;
    do for stepsig in 1.2  2 ; 
    do srun -n $nchains python -u compare_atlasnuts3.py --exp $exp -n $n --n_samples $nsamples  --stepsize_distribution $stepdist --combine_chains $combinechains  --stepsize_sigma $stepsig --constant_trajectory $ctraj  --nleap_distribution $nleapdist    # --suffix $suffix
        done
    done
    echo "done"
fi

# for offset in 1.0 ;
# do for nleapdist in  "empirical" "uniform"  ;
#    do
#        srun -n $nchains python -u compare_atlasnuts3.py --exp $exp -n $n --n_samples $nsamples   --combine_chains $combinechains   --constant_trajectory 1  --nleap_distribution $nleapdist  # --suffix $suffix
#    done
# done

# for offset in 1.0 ;
# do for stepdist in "beta"  ;
#    do
#        echo $stepdist;
#        srun -n $nchains python -u compare_atlasnuts3.py --exp $exp -n $n --n_samples $nsamples --n_leapfrog_adapt $nleapadapt --offset $offset --stepsize_distribution $stepdist --combine_chains $combinechains  --step_factor $stepfac --probabilistic $probabilistic --delayed_proposals $dr --n_hessian_samples $nhess  --nleap_distribution $nleapdist   --max_stepsize_reduction $red --constant_trajectory 1  # --suffix $suffix
#    done
# done