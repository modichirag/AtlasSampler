import numpy as np


class ReferenceSamples():


    def __init__(self, ):
        self.analytic_models = ['normal', 'funnel', 'rosenbrock', 'multifunnel']


    def gen_analytic_samples(self):

        if self.exp == 'normal':
            ref_samples = np.random.normal(0, 1, self.n_samples*self.D).reshape(self.n_samples, self.D)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'funnel':
            log_scale = np.random.normal(0, 3, self.n_samples)
            latents = np.array([np.random.normal(0, np.exp(log_scale/2)) for _ in range(self.D-1)]).T
            log_scale = log_scale.reshape(-1, 1)
            ref_samples = np.concatenate([log_scale, latents], axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'rosenbrock':
            x = np.random.normal(1, 1, self.n_samples)
            y = np.array([np.random.normal(x**2, 0.1) for _ in range(self.D-1)]).T
            x = np.expand_dims(x, axis=1)
            ref_samples = np.concatenate([x, y], axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        if self.exp == 'multifunnel':
            assert self.D % 10 == 0 # D must be a multiple of 10
            copies = self.D//10
            ref_samples = []
            for i in range(copies):
                log_scale = np.random.normal(0, 3, self.n_samples).reshape(-1, 1)
                ref_samples.append(log_scale)
            for i in range(copies):                
                latents = np.array([np.random.normal(0, np.exp(ref_samples[i][:, 0]/2)) for _ in range(9)]).T
                ref_samples.append(latents)
            ref_samples = np.concatenate(ref_samples, axis=1)
            if self.chain_index: ref_samples = np.expand_dims(ref_samples, axis=0)

        return ref_samples


    def nuts_samples(self):
        raise NotImplementedError


    def generate_samples(self, exp, D, n_samples=100000, chain_index=True, run_nuts=True):
        self.D = D
        self.exp = exp
        self.n_samples = n_samples
        self.chain_index = chain_index
        
        if exp in self.analytic_models:
            return self.gen_analytic_samples()
        else:
            if run_nuts:
                return self.nuts_samples()
            else:
                return None
