# Atlas Sampler
Adapting Trajectory Lenght and Stepsize for Hamiltonian Monte Carlo in a delayed rejection framework. 

This repository contains research code and is currently under active development. 
To use the package, clone the repo, build and install the code locally.
The following will install it in the dev version.
```
cd AtlasSampler
python3 -m build
pip install --editable .
```
The code is written in pure python and currently only installs the package.
It has additional dependencies required to run experiments and replicate results that are not automatically installed.
To run the example files, you will need `bridgestan` to access log-probability and gradients of stan models defined in `stan` folder. 
In the `experiments/example.py`, you will also need to change the `BRIDGESTAN` path to point to your local install.

## Example
An example of how to use the code is given in the script `experiments/example.py`.
For example, the following command will run Atlas sampler with 4 chains for a 10-dimensional normal model specified in `stan` folder. 
```
mpirun -n 4 python -u example.py --exp normal -n 10 
```
Note: We recommend running 4 to 8 chains as the warmup stage combines information from all the chains to construct an empirical distribution of trajectory lengths to U-turn. 

The script takes different arguments to customize the sampler. For example, a typical call for the Neal's funnel model will look as follows.
```
mpirun -n 8 python -u example.py --exp funnel -n 10  --n_samples 5000 --n_stepsize_adapt 200 --n_leapfrog_adapt 200 --constant_trajectory 1  --probabilistic 1 --target_accept 0.70
```
For most problems, including funnel and rosenbrock, the default values of these parameters should work well.


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
