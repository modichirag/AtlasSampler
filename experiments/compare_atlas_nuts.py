"""
Script to run nuts and atlas on the same problem with the stepsize tuned by nuts, and compare results. 
The script also generates reference samples by running nuts with target acceptance of 0.95 if needed.
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
    
from atlassampler import Atlas
from atlassampler.wrappers import cmdstanpy_wrapper
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import models
import plotting
import nuts

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)

# Set some paths
#SAVEFOLDER = '/mnt/ceph/users/cmodi/atlassampler/'
#REFERENCE_FOLDER = "/mnt/ceph/users/cmodi/PosteriorDB/"
SAVEFOLDER = './tmp/'
REFERENCE_FOLDER = "./tmp/reference/"
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
parser.add_argument('--n_burnin', type=int, default=200, help='number of iterations for burn-in')
parser.add_argument('--n_stepsize_adapt', type=int, default=100, help='number of iterations for step size adaptation')
parser.add_argument('--n_leapfrog_adapt', type=int, default=100, help='number of iterations for trajectory length adaptation')
parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--step_size', type=float, default=0.1, help='initial step size')
parser.add_argument('--offset', type=float, default=1.0, help='offset for uturn sampler')
parser.add_argument('--constant_trajectory', type=int, default=1, help='trajectory length of delayed stage, deault=1') 
parser.add_argument('--probabilistic', type=int, default=1, help='probabilistic atlas, default=1')
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--min_leapfrog', type=int, default=3, help='minimum number of leapfrog steps')
parser.add_argument('--max_leapfrog', type=int, default=1024, help='maximum number of leapfrog steps')
parser.add_argument('--low_nleap_percentile', type=int, default=10, help='lower percentile of trajectory distribution')
parser.add_argument('--high_nleap_percentile', type=int, default=50, help='higher percentile of trajectory distribution')
parser.add_argument('--metric', type=str, default="unit_e", help='metric for NUTS')
parser.add_argument('--n_metric_adapt', type=int, default=0, help='number of iterations for NUTS metric adaptation')
parser.add_argument('--nuts', type=int, default=0, help='run nuts')


args = parser.parse_args()
args.n_chains = wsize
print("Model name : ", args.exp)

# Load model and reference samples in rank 0. Generate if necessary.
reference_path =  f'{REFERENCE_FOLDER}/{args.exp}/'
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
    print("\nTrying to load NUTS results on rank 0")
    try:
        samples_nuts = np.load(f"{savefolder_nuts}/samples.npy")
        step_size = np.load(f"{savefolder_nuts}/stepsize.npy")
        n_leapfrogs_nuts = np.load(f"{savefolder_nuts}/leapfrogs.npy")
        assert samples_nuts.shape[0] >= wsize
        assert samples_nuts.shape[1] >= args.n_samples
        samples_nuts = samples_nuts[:wsize, :args.n_samples]
        n_leapfrogs_nuts = n_leapfrogs_nuts[:wsize, :args.n_samples]
        step_size = step_size[:wsize]
        print(f"Loaded nuts samples and stepsize from {savefolder_nuts}")
    except Exception as e:
        print("Exception in loading NUTS results : ", e)
        args.nuts = 1
    
    if args.nuts == 1:
        print("\nNow run NUTS on rank 0")
        savefolder_nuts = f"{savepath}/nuts/target{args.target_accept:0.2f}/"
        print(f"NUTS results will be saved in {savefolder_nuts}")
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
comm.Barrier()
if wrank == 0 : print()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, 0], root=0)
q0 = model.param_unconstrain(q0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()


#####################
# Atlas
folder = 'atlas-nuts'
savefolder = f"{savepath}/{folder}/offset{args.offset:0.2f}/"
if args.probabilistic !=0:
    savefolder = f"{savefolder}"[:-1] + f"-prob{args.probabilistic}/"
if args.constant_trajectory !=0:
    savefolder = f"{savefolder}"[:-1] + f"-ctraj{args.constant_trajectory}/"
if args.suffix != "":
    savefolder = f"{savefolder}"[:-1] + f"-{args.suffix}/"
    
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving Atlas runs in folder : {savefolder}")

# Start run
np.random.seed(args.seed)
kernel = Atlas(D, lp, lp_g, 
                mass_matrix = np.eye(D), 
                constant_trajectory = args.constant_trajectory,
                probabilistic = args.probabilistic,
                offset = args.offset,
                min_nleapfrog = args.min_leapfrog,
                max_nleapfrog = args.max_leapfrog,
                low_nleap_percentile = args.low_nleap_percentile,
                high_nleap_percentile = args.high_nleap_percentile)

sampler = kernel.sample(q0, 
                        n_leapfrog = args.n_leapfrog, 
                        step_size = step_size, 
                        n_samples = args.n_samples, 
                        n_burnin = 0,
                        n_stepsize_adapt = 0,
                        n_leapfrog_adapt = args.n_leapfrog_adapt,
                        target_accept = args.target_accept)


print(f"Acceptance for Atlas Sampler in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

# Save samples and gather on rank 0 for diagnostics
sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
samples_constrained = []
for s in sampler.samples:
    samples_constrained.append(model.param_constrain(s))
np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)
comm.Barrier()

samples = comm.gather(samples_constrained, root=0)
accepts = comm.gather(sampler.accepts, root=0)
gradcounts = comm.gather(sampler.gradcounts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)

#####################
# Diagnostics
if wrank == 0:
    
    samples = np.stack(samples, axis=0)
    print("\nTotal accpetances for Atlas Sampler : ", np.unique(np.stack(accepts), return_counts=True))
    
    print("\nPlotting")
    suptitle = f"{args.exp}:D={D}; " + savefolder.split(f'atlas-nuts/')[1][:-1]
    samples_list = [samples_nuts, samples]
    labels = ["NUTS", "Atlas"]

    # plot histograms
    plotting.plot_histograms(samples_list, 
                            nplot = min(5, D), 
                            labels = labels, 
                            savefolder = savefolder, 
                            suptitle = suptitle,
                            reference_samples = ref_samples)
    plt.savefig('tmp.png')
    plt.close()

    # plot cost
    plt.figure()
    normalize = n_leapfrogs_nuts.sum(axis=1).mean()
    toplot = [n_leapfrogs_nuts.sum(axis=1)/normalize, np.array(gradcounts).sum(axis=1)/normalize]
    plt.boxplot(toplot, patch_artist=True,
            boxprops=dict(facecolor='C0', color='C0', alpha=0.5), labels=labels)
    plt.grid(which='both', lw=0.3)
    plt.ylabel('# Leapfrogs', fontsize=12)
    plt.axhline(1., color='k', ls=":")
    plt.suptitle(suptitle)
    plt.savefig(f"{savefolder}/gradcounts")
    plt.close()

    # plot rmse
    if ref_samples is not None:
        counts_list = [n_leapfrogs_nuts, np.array(gradcounts)]    
        plotting.boxplot_rmse(ref_samples, samples_list, counts_list, labels, 
                savefolder=savefolder, suptitle=suptitle)
        plt.close()

    if args.exp == 'funnel':
        samples_list0 = [i[..., 0:1] for i in samples_list]
        plotting.boxplot_rmse(ref_samples[..., 0:1], samples_list0, counts_list, labels, 
                savefolder=savefolder, suptitle=suptitle, savename='rmse_logscale')
        plt.close()
        #
        samples_list1 = [i[..., 1:] for i in samples_list]
        plotting.boxplot_rmse(ref_samples[..., 1:], samples_list1, counts_list, labels, 
                savefolder=savefolder, suptitle=suptitle, savename='rmse_latents')
        plt.close()

comm.Barrier()    

sys.exit()
