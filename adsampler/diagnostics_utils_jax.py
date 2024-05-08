import jax.numpy as jnp
import numpy as np
import sys
from  jax import jit
import jax
from functools import partial


@jit
def cumulative_mean(samples, mode=1):
    if len(samples.shape) == 1:
        return jnp.cumsum(samples**mode) / (1+jnp.arange(samples.size))
     
    if len(samples.shape) == 2: #assume shape (nchains, nsamples)
        return jnp.cumsum((samples.T)**mode) / (1+jnp.arange(samples.size))

    else:
        print("Not implemented for sample of shape : ", samples.shape)
        print("Expected 1-d or 2-d samples of structure (nchains, nsamples)")
            


@jit
def cumulative_error(samples, counts, true_val=None, true_scatter=None, ref_samples=None, mode=1, relative=False, verbose=False):

    if (true_val is None) & (ref_samples is None):
        print("baseline is not given")
        raise
    if (true_scatter is None) & (ref_samples is None) & (relative == 'scatter'):
        print("Need reference samples or true scatter to normalize relative to scatter")
        raise

    if ref_samples is not None:
        assert len(ref_samples.shape) == 1
        true_val = np.mean(ref_samples**mode, axis=0)
        true_scatter = np.mean(ref_samples**mode, axis=0)
        if verbose: print(true_val)

    err = cumulative_mean(samples, mode) - true_val
        
    if relative=='val':
        err /= true_val
    elif relative=='scatter':
        err /= true_scatter
        
    count = np.cumsum(counts.T)
    return count, err



@jit
def cumulative_rmse(samples, counts, ref_samples, mode=1, nevals=None, relative=0, verbose=False):

    print("jit")
    D = samples.shape[-1]
    assert len(samples.shape) == 3

    assert ref_samples.shape[-1] == D
    ref_samples = ref_samples.reshape(-1, D)
    true_val = np.mean(ref_samples**mode, axis=0)
    true_scatter = np.std(ref_samples**mode, axis=0)

    cost = np.cumsum(counts.T)
    if nevals is not None:
        if (len(counts.shape) > 1) & (counts.shape[0] != 1):
            print("cost threshold is only implemented for single chains")
            raise
        else:
            if nevals > cost[-1]: idx = -1
            else: idx = np.where(cost > nevals)[0][0]
            cost = cost[:idx]
            samples = samples[:, :idx]
            
    errors = jax.vmap(partial(cumulative_mean, mode=mode), in_axes=2)(samples).T -  true_val
    errors1 = errors / true_val
    errors2 = errors /true_scatter

    rmse = ((errors**2).mean(axis=-1))**0.5
    rmse1 = ((errors1**2).mean(axis=-1))**0.5
    rmse2 = ((errors2**2).mean(axis=-1))**0.5
    return cost, rmse, rmse1, rmse2



def cumulative_error_bootstrap(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 2
    assert len(counts.shape) == 2
    nchains, nsamples = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        idx = list(np.arange(nchains))
        idx.pop(i)
        count, err = cumulative_error(samples[idx], counts[idx], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)


def cumulative_error_per_chain(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 2
    assert len(counts.shape) == 2
    nchains, nsamples = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        count, err = cumulative_error(samples[i:i+1], counts[i:i+1], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)


def cumulative_rmse_bootstrap(samples, counts, true_val=None, ref_samples=None, mode=1, relative=False, verbose=False):
    
    assert len(samples.shape) == 3
    assert len(counts.shape) == 2
    nchains, nsamples, D = samples.shape

    count_list, err_list = [], []
    for i in range(nchains):
        idx = list(np.arange(nchains))
        idx.pop(i)
        count, err = cumulative_rmse(samples[idx], counts[idx], true_val=true_val, ref_samples=ref_samples, mode=mode, relative=relative, verbose=verbose)
        count_list.append(count)
        err_list.append(err)

    return np.array(count_list), np.array(err_list)

                    

def cumulative_rmse_per_chain(samples, counts, true_val=None, ref_samples=None, mode=1, nevals=None, relative=False, verbose=False):
    
    assert len(samples.shape) == 3
    assert len(counts.shape) == 2
    nchains, nsamples, D = samples.shape
    if not relative: relative = 0
    if relative == 'val': relative = 1
    if relative == 'scatter': relative = 2

    f = jax.vmap(partial(cumulative_rmse, ref_samples=ref_samples, mode=mode, nevals=nevals, relative=relative, verbose=verbose), [1, 1])
    errors = f(np.expand_dims(samples, axis=0), np.expand_dims(counts, axis=0))
    counts = errors[0]
    errors = errors[1:][relative]
    return counts, errors

