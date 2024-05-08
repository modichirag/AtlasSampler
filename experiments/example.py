import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

sys.path.append('../adsampler/')
sys.path.append('../adsampler/algorithms/')
from hmc import HMC
from uturn_samplers import HMC_Uturn_Sampler, HMC_Uturn_Jitter_Sampler 
from stepadapt_samplers import DRHMC_AdaptiveStepsize
from adsampler import ADSampler
import util
from cmdstanpy_wrapper import cmdstanpy_wrapper
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
savepath = '/mnt/ceph/users/cmodi/adaptive_hmc/'


#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
#arguments for GSM
parser.add_argument('--seed', type=int, default=999, help='seed')
parser.add_argument('--n_leapfrog', type=int, default=40, help='number of leapfrog steps')
parser.add_argument('--n_samples', type=int, default=1001, help='number of samples')
parser.add_argument('--n_burnin', type=int, default=0, help='number of iterations for burn-in')
parser.add_argument('--n_stepsize_adapt', type=int, default=0, help='step size adaptation')
parser.add_argument('--n_leapfrog_adapt', type=int, default=0, help='step size adaptation')
parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--step_size', type=float, default=0.1, help='initial step size')
parser.add_argument('--offset', type=float, default=1.0, help='offset for uturn sampler')
parser.add_argument('--constant_trajectory', type=int, default=0, help='run hmc')
parser.add_argument('--probabilistic', type=int, default=0, help='run hmc')
parser.add_argument('--hmc', type=int, default=0, help='run hmc')
parser.add_argument('--nuts', type=int, default=0, help='run nuts')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

args = parser.parse_args()
experiment = args.exp
n = args.n

print("Model name : ", experiment)
model, D, lp, lp_g, ref_samples, files = models.setup_model(experiment, n)
if n!= 0 : savepath = f'/mnt/ceph/users/cmodi/adaptive_hmc/{experiment}'
else: savepath = f'/mnt/ceph/users/cmodi/adaptive_hmc/{experiment}-{D}/'


###################################
##### Setup the algorithm parameters

n_leapfrog = args.n_leapfrog
step_size = args.step_size
n_samples = args.n_samples
n_burnin = args.n_burnin
n_stepsize_adapt = args.n_stepsize_adapt
n_chains = wsize
target_accept = args.target_accept
print(f"Saving runs in parent folder : {savepath}")


###################################
# NUTS
np.random.seed(args.seed)
if wrank == 0:
    savefolder = f"{savepath}/nuts/"
    if args.nuts == 0 :
        print("\nTrying to load NUTS results on rank 0")
        try:
            samples_nuts = np.load(f"{savefolder}/samples.npy")
            step_size = np.load(f"{savefolder}/stepsize.npy")
            assert samples_nuts.shape[0] >= wsize
            samples_nuts = samples_nuts[:wsize, :args.n_samples]
            step_size = step_size[:wsize]
            print(f"Loaded nuts samples and stepsize from {savefolder}")
        except Exception as e:
            print("Exception in loading NUTS results : ", e)
            args.nuts = 1
    
    if args.nuts == 1:
        print("\nNow run NUTS on rank 0")
        stanfile, datafile = files
        cmd_model = csp.CmdStanModel(stan_file = stanfile)
        sample = cmd_model.sample(data=datafile, chains=wsize, iter_sampling=n_samples-1,
                                  seed = args.seed,
                                  metric="unit_e",
                                  adapt_delta=target_accept,
                                  adapt_metric_window=0,
                                  adapt_init_phase=1000,
                                  adapt_step_size=1000,
                                  show_console=False, show_progress=True, save_warmup=False)
        draws_pd = sample.draws_pd()
        samples_nuts, leapfrogs_nuts = cmdstanpy_wrapper(draws_pd, savepath=f'{savefolder}/')
        np.save(f'{savefolder}/stepsize', sample.step_size)

        difference = np.diff(samples_nuts[..., 0])
        print("accept/reject for NUTS: ", difference.size - (difference == 0 ).sum(),  (difference == 0 ).sum())
        step_size = sample.step_size
        
else:
    step_size = 0.
    samples_nuts = np.zeros([n_chains, D])
comm.Barrier()

# Scatter step size and initial point to different ranks
comm.Barrier()
if wrank == 0 : print()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, 0], root=0)
#q0 = comm.scatter(inits, root=0)
q0 = model.param_unconstrain(q0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()


#####################
# UTurn
folder = 'test'
savefolder = f"{savepath}/{folder}/offset{args.offset:0.2f}/"
if args.suffix != "":
    savefolder = f"{savefolder}"[:-1] + f"-{args.suffix}/"
    
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving runs in folder : {savefolder}")

# Start run
np.random.seed(0)
#kernel = HMC(D, lp, lp_g, mass_matrix=np.eye(D))
#kernel = HMC_Uturn_Sampler(D, lp, lp_g, mass_matrix=np.eye(D), 
#                           min_nleapfrog=1, max_nleapfrog=128, offset=args.offset)
# kernel = HMC_Uturn_Jitter_Sampler(D, lp, lp_g, mass_matrix=np.eye(D), 
#                                   min_nleapfrog=1, max_nleapfrog=128, offset=args.offset)
# kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D),
#                                 constant_trajectory=args.constant_trajectory,
#                                 high_nleap_percentile=50,
#                                 min_nleapfrog=1, max_nleapfrog=128, offset=args.offset)
kernel = ADSampler(D, lp, lp_g, mass_matrix=np.eye(D), 
                   constant_trajectory=args.constant_trajectory,
                   probabilistic=args.probabilistic,
                   min_nleapfrog=3, max_nleapfrog=128, offset=args.offset)

sampler = kernel.sample(q0, n_leapfrog=args.n_leapfrog, step_size=step_size, n_samples=n_samples, n_burnin=n_burnin,
                        n_stepsize_adapt=args.n_stepsize_adapt,
                        n_leapfrog_adapt=args.n_leapfrog_adapt,
                        target_accept=target_accept)

print(f"Acceptance for Uturn HMC in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
samples_constrained = []
for s in sampler.samples:
    samples_constrained.append(model.param_constrain(s))
np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)
comm.Barrier()

samples_uturn = comm.gather(samples_constrained, root=0)
accepts_uturn = comm.gather(sampler.accepts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)
if wrank == 0 :
    samples_uturn = np.stack(samples_uturn, axis=0)
    print("Means")
    print(samples_nuts.mean(axis=(0,1)))
    print(samples_uturn.mean(axis=(0,1)))
    print("Standard deviation")
    print(samples_nuts.std(axis=(0,1)))
    print(samples_uturn.std(axis=(0,1)))
    np.save(f'{savefolder}/stepsize', stepsizes)
comm.Barrier()

#####################
if wrank == 0:
    # plot
    plt.figure()
    #plt.hist(np.random.normal(0, 3, 100000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(samples_nuts[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='NUTS');
    samples_uturn = np.stack(samples_uturn, axis=0)
    plt.hist(samples_uturn[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='1step ADR-HMC');
    
    print("\nPlotting")
    plt.title(f"{D-1} dimension {args.exp}")
    plt.legend()
    plt.savefig('tmp.png')
    plt.savefig(f"{savefolder}/hist")
    plt.close()

    print()
    #print("Total accpetances for HMC : ", np.unique(np.stack(accepts_hmc), return_counts=True))
    print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts_uturn), return_counts=True))

comm.Barrier()


    

sys.exit()
