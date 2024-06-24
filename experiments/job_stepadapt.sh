#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J adaptive
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

exp=$1
n=$2
echo $exp $n 

nchains=32
nleapadapt=100
nsamples=50001
burnin=100
stepadapt=0
targetaccept=0.8
constant_traj=1

srun -n $nchains python -u compare_stepadapt.py --exp $exp -n $n --n_samples $nsamples  --n_leapfrog_adapt $nleapadapt  --target_accept $targetaccept  --constant_traj $constant_traj
echo "done"
    

