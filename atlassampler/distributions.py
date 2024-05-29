import numpy as np
from scipy.stats import beta
from scipy.stats import norm


def setup_stepsize_distribution(epsmean, epsmax, epsmin, distribution='beta'):
    if distribution == 'beta':
        return beta_dist(epsmean, epsmax, epsmin)
    if distribution == 'lognormal':
        return Lognormal_distribution(epsmean)
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



class Lognormal_distribution():
    
    def __init__(self, epsmean, sigma=np.log(2)):
        self.epsmean = epsmean
        self.offset = sigma**2/2
        self.mu = np.log(self.epsmean) - self.offset
        self.sigma = sigma
        self.normal_dist = norm(loc=self.mu,  scale=self.sigma)
        
    def rvs(self, size):
        y = self.normal_dist.rvs(size)
        x = np.exp(y)
        return x
    
    def logpdf(self, x):
        y = np.log(x)
        jac = 1/x #dy/dx
        lp_y = self.normal_dist.logpdf(y) 
        lp_x = lp_y + np.log(jac)
        return lp_x
