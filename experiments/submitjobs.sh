#!/bin/bash



for chains in 0 1 ;
do for ctraj in 1 2 ;
   do sbatch job_atlasnuts3.sh funnel 10   $chains $ctraj   ; done
done

#for c in 0 1 ; do sbatch job_atlasnuts2.sh funnel 10   $c   ; done
#for c in 0 1 ; do sbatch job_atlasnuts.sh rosenbrock 1   $c   ; done

# for c in 0 1 ; do sbatch job_atlasnuts.sh hmm 0   $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh lotka_volterra 0  $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh arK 0  $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh corr_normal95 100  $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh ill_normal 100  $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh stochastic_volatility  0 $c   ; done
# for c in 0 1 ; do sbatch job_atlasnuts.sh irt_2pl  0 $c   ; done
