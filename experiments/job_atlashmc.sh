#!/bin/bash
#SBATCH -p ccm
##SBATCH -C ib-rome
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH -J ahmc
#SBATCH -o logs/%x.o%j

module purge
module load modules/2.2-20230808
module load slurm cuda cudnn gcc 
source activate jaxenv

OMP_NUM_THREADS=1


exp=$1
n=$2
echo $exp $n

nchains=32
nsamples=50001
offset=1.0
#hessian_mode='bam'

stepdist="lognormal"
for stepsig in  1.2 ;
   do srun -n $nchains python -u compare_atlashmc.py --exp $exp -n $n --n_samples $nsamples --offset $offset --stepsize_distribution $stepdist  --stepsize_sigma $stepsig  --max_nleapfrog 2048
done

stepdist="beta"
srun -n $nchains python -u compare_atlashmc.py --exp $exp -n $n --n_samples $nsamples --offset $offset --stepsize_distribution $stepdist  --stepsize_sigma $stepsig  --max_nleapfrog 2048


echo "done"
    

