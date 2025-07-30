# ATLAS Sampler
Adapting Trajectory Length and Stepsize (ATLAS) for Hamiltonian Monte Carlo in a delayed rejection framework. 

This repository contains research code for the paper https://arxiv.org/pdf/2410.21587.
To use the package, clone the repo, build and install the code locally.
The following will install it in the dev version.
```
cd AtlasSampler
python3 -m build
pip install --editable .
```

## Dependencies
The code is written in pure python and currently only installs the package.
It has additional dependencies required to run experiments and replicate results that are not automatically installed.
You need to install jax to use Hessian approximation from BAM (Batch and Match, https://arxiv.org/abs/2402.14758).
Some Stan models are specified in the `stan` folder. To run these example files, you will need `bridgestan` to access log-probability and gradients.. 
In the `experiments/example.py`, you will also need to change the `BRIDGESTAN` path to point to your local install.

## Example
The code runs only 1 chain for every Sampler object. For multiple chains, we recommend using MPI or Pool.
An example of how to use the code is given in the script `experiments/example.py` and  `notebooks/example.ipynb`
For instance, the following command will run Atlas sampler with 4 chains for a 10-dimensional normal model specified in `stan` folder. 
```
mpirun -n 4 python -u example.py --exp normal -n 10 
```
Note: When running multiple chains, we recommend combining trajectory lengths to U-turn explored during warmup. 

The algorithmic kernel allows for multiple different arguments to customize and experiment with the sampler.
For most problems, the default values of these parameters specified in the file experiments/default_args.py work well.
These can be overridden from command line. For example, a call for the Neal's funnel model can look like:
```
mpirun -n 8 python -u example.py --exp funnel -n 10  --n_samples 5000 --n_stepsize_adapt 200 --n_leapfrog_adapt 200 --constant_trajectory 2  --probabilistic 1 --target_accept 0.80
```



## Comparison with NUTS
Script `experiments/compare_atlas_nuts.py` runs NUTS and Atlas on a particular problem with the same stepsize as tuned by NUTS. <br>
The generated samples are saved in the path specified by `SAVEFOLDER` at the top of the script. <br>
The script also generates reference samples by running NUTS with target accepatance = 0.95 for non-analytic model.
These are saved in the path specified by the path `REFERENCE_FOLDER` at the top of the script. <br>
These reference samples are used to generate diagnostic plots comparing-
i) the cost (number of gradient evaluations) and ii) RMSE on z-scaled parameters for NUTS and Atlas.
These are also saved in the `SAVEFOLDER`.

Example calls looks like this
```
# For a 2D Rosenbrock
mpirun -n 8 python -u compare_atlas_nuts.py --exp rosenbrock -n 1  --n_samples 5000 
# For a non-analytic model like hmm
mpirun -n 8 python -u compare_atlas_nuts.py --exp hmm  --n_samples 5000 
```


## Variants
In folder atlassampler/algorithms, we provide other variants and components on which ATLAS is built. These can also be used for sampling themselves.
Specifically we provide an implementation of DRHMC (https://arxiv.org/abs/2110.00610), No U-Turn sampler from GIST (https://arxiv.org/html/2404.15253v2),
and ATLAS without DR framework. See different `compare_xyz.py` files in experiments folder on how to use these. 
