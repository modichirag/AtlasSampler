import numpy as np
import os, sys
import matplotlib.pyplot as plt

import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

from atlassampler.wrappers import cmdstanpy_wrapper
import models
import nuts

# Set some paths
SAVEFOLDER = '/mnt/ceph/users/cmodi/atlassampler/'
REFERENCE_FOLDER = "/mnt/ceph/users/cmodi/PosteriorDB/"
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
MODELDIR = '../'


#######
import argparse
parser = argparse.ArgumentParser(description='Arguments for NUTS sampler')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--n_chains', type=int, default=16, help='number of chains')
parser.add_argument('--n_samples', type=int, default=100, help='number of samples')
parser.add_argument('--n_burnin', type=int, default=1000, help='number of burnin/warmup iterations')
parser.add_argument('--n_stepsize_adapt', type=int, default=100, help='number of iterations for step size adaptation')
parser.add_argument('--n_metric_adapt', type=int, default=0, help='number of iterations for metric adaptation')
parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance rate')
parser.add_argument('--step_size', type=float, default=0.1, help='initial step size')
parser.add_argument('--metric', type=str, default='unit_e', help='metric for NUTS')
parser.add_argument('--suffix', type=str, default='', help='suffix for folder name')
parser.add_argument('--step_factor', type=float, default=1.0, help='target acceptance rate')
parser.add_argument('--max_treedepth', type=int, default=10, help='target acceptance rate')


args = parser.parse_args()
print("Model name : ", args.exp)

experiment = args.exp
n = args.n
parentfolder = '/mnt/ceph/users/cmodi/atlassampler/'
print("Model name : ", experiment)   

# Load model and reference samples in rank 0. Generate if necessary.
reference_path =  f'{REFERENCE_FOLDER}/'
model, D, lp, lp_g, ref_samples, files = models.stan_model(args.exp, args.n, 
                                                           bridgestan_path=BRIDGESTAN, 
                                                           model_directory=MODELDIR, 
                                                           reference_samples_path=reference_path,
                                                           run_nuts=False)
stanfile, datafile = files
if n == 0 : savefolder = f'{parentfolder}/{experiment}/nuts/target{args.target_accept:0.2f}/'
else: savefolder = f'{parentfolder}/{experiment}-{D}/nuts/target{args.target_accept:0.2f}/'
if args.step_factor != 1.:
    savefolder = f"{savefolder}"[:-1] + f"-stepfac{args.step_factor}/"
if args.suffix != '': savefolder = savefolder[:-1] + f'-{args.suffix}/'
#if args.n == 0 : savepath = f'{SAVEFOLDER}/{args.exp}'
#else: savepath = f'{SAVEFOLDER}/{args.exp}-{D}/'
###################################
# NUTS
# Load samples if present. Run NUTS otherwise
if n == 0 : savefolder_nuts = f"{parentfolder}/{experiment}/nuts/target{args.target_accept:0.2f}/"
else: savefolder_nuts = f"{parentfolder}/{experiment}-{D}/nuts/target{args.target_accept:0.2f}/"
print(f"\nLoading NUTS results on rank 0 from folder :  {savefolder_nuts}")
samples_nuts, step_size, n_leapfrogs_nuts = nuts.load_results(savefolder_nuts, args.n_chains, args.n_samples)
   
np.random.seed(args.seed)
if savefolder is not None:
    os.makedirs(savefolder, exist_ok=True)

print(f"Run nuts with step factor : ", args.step_factor)
print(f"Save results in {savefolder}")
cmd_model = csp.CmdStanModel(stan_file = stanfile)
sampler = cmd_model.sample(data = datafile, 
                           chains = args.n_chains,
                           iter_sampling = args.n_samples,
                           iter_warmup = args.n_burnin,
                           seed = args.seed,
                           metric = args.metric,
                           max_treedepth = args.max_treedepth,
                           step_size = list(step_size * args.step_factor),
                           #    adapt_delta = args.target_accept,
                           ## cmdstanpy bug? does not adapt if specified
                           adapt_engaged = False,
                           #    adapt_metric_window = 0, 
                           #    adapt_init_phase = 0,
                           #    adapt_step_size = 0,
                           show_console = False,
                           show_progress = True,
                           save_warmup = False)

draws_pd = sampler.draws_pd()
step_size = sampler.step_size
metric = sampler.metric
samples, n_leapfrogs = cmdstanpy_wrapper(draws_pd)
difference = np.diff(samples[..., 0])
print("accept/reject for NUTS: ", difference.size - (difference == 0 ).sum(),  (difference == 0 ).sum())

if savefolder is not None:
    np.save(f'{savefolder}/samples', samples)
    np.save(f'{savefolder}/leapfrogs', n_leapfrogs)
    np.save(f'{savefolder}/stepsize', step_size)
    if metric is not None: np.save(f'{savefolder}/metric', metric)

