import numpy as np
import os, sys
import matplotlib.pyplot as plt

from atlassampler import Atlas, DRHMC_AdaptiveStepsize
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import models

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)

# Set some paths
SAVEFOLDER = '/mnt/ceph/users/cmodi/atlassampler/'
#SAVEFOLDER = './tmp/'
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
MODELDIR = '../'


#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
parser.add_argument('--seed', type=int, default=999, help='seed')
parser.add_argument('--n_leapfrog', type=int, default=20, help='number of leapfrog steps')
parser.add_argument('--n_samples', type=int, default=1001, help='number of samples')
parser.add_argument('--n_burnin', type=int, default=0, help='number of iterations for burn-in')
parser.add_argument('--n_stepsize_adapt', type=int, default=0, help='number of iterations for step size adaptation')
parser.add_argument('--n_leapfrog_adapt', type=int, default=0, help='number of iterations for trajectory length adaptation')
parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--step_size', type=float, default=1.0, help='initial step size')
parser.add_argument('--offset', type=float, default=1.0, help='offset for uturn sampler')
parser.add_argument('--constant_trajectory', type=int, default=1, help='trajectory length of delayed stage, deault=1') 
parser.add_argument('--probabilistic', type=int, default=1, help='probabilistic atlas, default=1')
parser.add_argument('--suffix', type=str, default="origin", help='suffix, default=""')
parser.add_argument('--min_leapfrog', type=int, default=3, help='minimum number of leapfrog steps')
parser.add_argument('--max_leapfrog', type=int, default=1024, help='maximum number of leapfrog steps')

args = parser.parse_args()
print("Model name : ", args.exp)
model, D, lp, lp_g, ref_samples, files = models.stan_model(args.exp, args.n, 
                                                           bridgestan_path=BRIDGESTAN, 
                                                           model_directory=MODELDIR, 
                                                           reference_samples_path=None,
                                                           run_nuts=False)
if args.n == 0 : savepath = f'{SAVEFOLDER}/{args.exp}'
else: savepath = f'{SAVEFOLDER}/{args.exp}-{D}/'

if ref_samples is not None:
    assert len(ref_samples.shape) == 3
    idx = np.random.choice(ref_samples.shape[1])
    q0 = np.array(ref_samples[0, idx])
    q0 = model.param_unconstrain(q0)
else:
    q0 = np.random.uniform(-2, 2, size=D)


q0 *= 0
print(q0)

# Save folder
folder = 'stepadapt'
savefolder = f"{savepath}/{folder}/target{args.target_accept:0.2f}-offset{args.offset:0.2f}/"
if args.suffix != "":
    savefolder = f"{savefolder}"[:-1] + f"-{args.suffix}/"
    
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving runs in folder : {savefolder}")

# Start run
np.random.seed(0)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, 
                mass_matrix = np.eye(D), 
                constant_trajectory = args.constant_trajectory,
                offset = args.offset,
                min_nleapfrog = args.min_leapfrog,
                max_nleapfrog = args.max_leapfrog)

sampler = kernel.sample(q0, 
                        n_leapfrog = args.n_leapfrog, 
                        step_size = args.step_size, 
                        n_samples = args.n_samples, 
                        n_burnin = args.n_burnin,
                        n_stepsize_adapt = args.n_stepsize_adapt,
                        n_leapfrog_adapt = args.n_leapfrog_adapt,
                        target_accept = args.target_accept)

print(kernel.step_size)
print(f"Acceptance for Atlas Sampler in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
samples_constrained = []
for s in sampler.samples:
    samples_constrained.append(model.param_constrain(s))
np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)
comm.Barrier()

samples = comm.gather(samples_constrained, root=0)
accepts = comm.gather(sampler.accepts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)
comm.Barrier()

#####################
# Diagnostics
if wrank == 0:
    if wrank == 0 :
        print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts), return_counts=True))

    # plot
    plt.figure()
    plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    samples = np.stack(samples, axis=0)
    plt.hist(samples[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='Atlas');
    
    print()
    print("\nPlotting")
    plt.title(f"{args.exp}:D={D}; " + savefolder.split(f'{folder}/')[1][:-1])
    plt.legend()
    plt.savefig('tmp.png')
    plt.savefig(f"{savefolder}/hist")
    plt.close()

comm.Barrier()

sys.exit()
