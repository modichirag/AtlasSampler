import numpy as np
from scipy.stats import beta


def setup_stepsize_distribution(epsmean, epsmax, epsmin, distribution='beta'):
    if distribution == 'beta':
        return beta_dist(epsmean, epsmax, epsmin)
    else:
        raise NotImplementedError

    
def beta_dist(epsmean, epsmax, epsmin):
    """Return a beta distribution with mode=eps_mean/2."""
    scale = epsmax-epsmin
    eps_scaled = epsmean/epsmax
    b = 2 * (1-eps_scaled)**2/eps_scaled
    a = 2 * (1-eps_scaled)
    dist = beta(a=a, b=b, loc=epsmin, scale=scale)
    return dist

