import numpy as np
import os, sys
import matplotlib.pyplot as plt

from atlassampler import DRHMC_AdaptiveStepsize
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import models
import plotting
import nuts
import default_args

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)

# Set some paths
SAVEFOLDER = '/mnt/ceph/users/cmodi/atlassampler/'
REFERENCE_FOLDER = "/mnt/ceph/users/cmodi/PosteriorDB/"
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
MODELDIR = '../'


#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser = default_args.add_default_args(parser)
parser.add_argument('--combine_chains', type=int, default=0, help='combine nleap from chains')
parser.add_argument('--step_factor', type=float, default=1, help='stepsize factor over nuts')

args = parser.parse_args()
args.n_chains = wsize
print("Model name : ", args.exp)

reference_path =  f'{REFERENCE_FOLDER}/'
run_nuts = bool(not wrank)
model, D, lp, lp_g, ref_samples, files = models.stan_model(args.exp, args.n, 
                                                           bridgestan_path=BRIDGESTAN, 
                                                           model_directory=MODELDIR, 
                                                           reference_samples_path=reference_path,
                                                           run_nuts=run_nuts)
if args.n == 0 : savepath = f'{SAVEFOLDER}/{args.exp}'
else: savepath = f'{SAVEFOLDER}/{args.exp}-{D}/'
comm.Barrier()
###################################
# NUTS
# Load samples if present. Run NUTS otherwise
np.random.seed(args.seed)
if wrank == 0:
    savefolder_nuts = f"{savepath}/nuts/target{args.target_accept:0.2f}/"
    print(f"\nTrying to load NUTS results on rank 0 from folder :  {savefolder_nuts}")
    try:
        samples_nuts, step_size, n_leapfrogs_nuts = nuts.load_results(savefolder_nuts, wsize, args.n_samples)
    except Exception as e:
        print("Exception in loading NUTS results : ", e)
        args.nuts = 1
        print(args.nuts)
    if args.nuts == 1:
        print("\nNow run NUTS on rank 0")
        samples_nuts, sampler, step_size, n_leapfrogs_nuts = nuts.run_nuts(stanfile = files[0], 
                                                                    datafile = files[1], 
                                                                    args = args, 
                                                                    savefolder=savefolder_nuts, 
                                                                    return_all=True)
else:
    step_size = 0.
    samples_nuts = np.zeros([args.n_chains, D])
comm.Barrier()

# Scatter step size and initial point to different ranks
if wrank == 0 : print()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, 0], root=0)
q0 = model.param_unconstrain(q0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()

############################
if ref_samples is not None:
    assert len(ref_samples.shape) == 3
    idx = np.random.choice(ref_samples.shape[1])
    q0 = np.array(ref_samples[0, idx])
    q0 = model.param_unconstrain(q0)
else:
    q0 = np.random.uniform(-2, 2, size=D)
q0 = np.random.uniform(-2, 2, size=D)


# Save folder
if args.combine_chains: folder = 'dr_stepadapt'
else: folder = 'dr_stepadapt-indep'
savefolder = f"{savepath}/{folder}/target{args.target_accept:0.2f}/"
if args.n_leapfrog_adapt == 0:
    savefolder = f"{savefolder}"[:-1] + f"nleap{args.n_leapfrog}/"
else:
    if args.nleap_distribution == 'uniform':
        savefolder = f"{savefolder}"[:-1] + f"-uninleap/"
if args.stepsize_distribution == 'lognormal':
    savefolder = f"{savefolder}"[:-1] + f"-{args.stepsize_distribution}/"
    if args.stepsize_sigma != 2.:
        savefolder = f"{savefolder}"[:-1] + f"-stepsig{args.stepsize_sigma}/"
if args.step_factor != 1.:
    savefolder = f"{savefolder}"[:-1] + f"-stepfac{args.step_factor}/"
if args.hessian_mode != 'bfgs':
    savefolder = f"{savefolder}"[:-1] + f"-{args.hessian_mode}/"
if args.suffix != "":
    savefolder = f"{savefolder}"[:-1] + f"-{args.suffix}/"
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving runs in folder : {savefolder}")


##########
np.random.seed(0)
if args.combine_chains : communicator = comm
else: communicator = None

kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, 
                                mass_matrix = np.eye(D), 
                                nleap_distribution = args.nleap_distribution,
                                constant_trajectory = args.constant_trajectory,
                                stepsize_distribution = args.stepsize_distribution,
                                stepsize_sigma = args.stepsize_sigma,
                                max_stepsize_reduction = args.max_stepsize_reduction,
                                hessian_mode = args.hessian_mode)

sampler = kernel.sample(q0,
                        seed = wrank,
                        n_leapfrog = args.n_leapfrog, 
                        step_size = step_size * args.step_factor, 
                        n_samples = args.n_samples, 
                        n_burnin = 0,
                        n_stepsize_adapt = 0, #args.n_stepsize_adapt,
                        n_leapfrog_adapt = args.n_leapfrog_adapt,
                        target_accept = args.target_accept,
                        comm = communicator)

print(f"Acceptance for {args.stepsize_distribution} distribution in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

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

# #####################
# # Diagnostics
# if wrank == 0:
#     if wrank == 0 :
#         print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts), return_counts=True))

#     # plot
#     plt.figure()
#     plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
#     samples = np.stack(samples, axis=0)
#     plt.hist(samples[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='Atlas');

#     print()
#     print("\nPlotting")
#     plt.title(f"{args.exp}:D={D}; " + savefolder.split(f'{folder}/')[1][:-1])
#     plt.legend()
#     plt.savefig('tmp.png')
#     plt.savefig(f"{savefolder}/hist")
#     plt.close()

comm.Barrier()

sys.exit()
