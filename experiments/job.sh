#!/bin/bash
#SBATCH -p ccm
#SBATCH -C ib-rome
#SBATCH --time=0:20:00
#SBATCH -J test
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

OMP_NUM_THREADS=1

exp=$1

if [ "$exp" == "funnel" ] ; then
    n=10
elif [ "$exp" == "rosenbrock" ] ; then
    n=1
else
    n=0
fi    

ctraj=2
nchains=32
nsamples=10001
prob=1
target=0.8
stepdist="lognormal"
combinechains=0
stepsig=1.2


hessian_mode='bam-bfgscorrect'
stepfac=1.0
suffix='zeromean-test'

for nleapdist in "uniform" "empirical" ; do
    srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains  --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist  --step_factor $stepfac --probabilistic $prob --suffix $suffix --hessian_mode $hessian_mode
done


# nleapdist="empirical"
# nleap=100
# low_nleap_percentile=25
# high_nleap_percentile=75
# nhess=10
# nladapt=200
# suffix='nladapt200-lownl25-highnl75-test'
# for stepfac in 1.5 2.0 ;
# do
#     srun -n $nchains python -u compare_atlasnuts2.py --exp $exp -n $n --n_samples $nsamples --combine_chains $combinechains   --constant_trajectory $ctraj --nleap_distribution $nleapdist  --stepsize_sigma $stepsig  --stepsize_distribution $stepdist  --step_factor $stepfac --probabilistic $prob --n_leapfrog $nleap --suffix $suffix  --low_nleap_percentile $low_nleap_percentile --high_nleap_percentile $high_nleap_percentile --n_leapfrog_adapt $nladapt 
# done
