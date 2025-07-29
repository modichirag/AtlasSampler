import numpy as np
from scipy.linalg import sqrtm
from . import util
import jax  # for BAM Hessian
import jax.numpy as jnp

def Hessian_approx(positions, gradients, H=None, mode='bfgs', rank=-1):
    if mode == 'bfgs':
        return BFGS_hessian_approx(positions, gradients, H, rank=rank)
    elif mode == 'fisher':
        raise Fisher_hessian_approx(positions, gradients, H, rank=rank)
    elif mode == 'bam':
        return BAM_hessian_approx(positions, gradients, H, rank=rank)
    elif mode == 'bam-bfgsinit':
        return BAM_hessian_approx_bfgsinit(positions, gradients, H, rank=rank)
    elif mode == 'bam-bfgscorrect':
        return BAM_hessian_approx_bfgscorrect(positions, gradients, H, rank=rank)
    elif mode == 'hybrid':
        H1, n1 = BAM_hessian_approx(positions, gradients, H, rank)
        H2, n2 = BFGS_hessian_approx(positions, gradients, H, rank)
        e1 = util.power_iteration(H1)[0]
        e2 = util.power_iteration(H2)[0]
        if e1 > e2: return H1, n1
        else: return H2, n2


def BFGS_hessian_approx(positions, gradients, H=None, rank=-1):
    '''Returns Hessian approximation (B matrix) & number of points used to estimate it'''

    try:
        d = positions.shape[1]
        npos = positions.shape[0]
        if (rank != -1) & (npos > rank + 1):
                idx = np.linspace(0, npos-1, rank).astype(int)
                positions = positions[-rank-1:]
                gradients = gradients[-rank-1:]
        #if H is None: 
        #    H = np.eye(d) # initial hessian. Moved to initilization below which is better

        nabla = gradients[0]
        x = positions[0]

        it = 0 
        points_used = 0 
        for i in range(npos-1):
            it += 1
            x_new = positions[i+1]
            nabla_new = gradients[i+1]
            s = x_new - x
            y = nabla_new - nabla
            r = 1/(np.dot(y,s))
            if r < 0:               #Curvature condition to ensure positive definite
                continue
            if (H is None) : #initialize based on, but before first update. Taken from Nocedal
                #H = np.eye(x.size) * np.dot(y,s)/np.dot(y, y) #gets confusing if we multiply or divide
                H = np.eye(x.size) / np.dot(y,s) *np.dot(y, y)
            points_used +=1
            z = np.dot(H, s)
            update = np.outer(y, y) / np.dot(s, y) - np.outer(z, z) / np.dot(s, z)
            H += update
            nabla = nabla_new[:].flatten()
            x = x_new[:].flatten()
    except:
        H, points_used = None, 0
    return H, points_used 

    

def BFGS_inverse_Hessian_approx(positions, gradients, H=None): # needs to be cleaned based on Hessian_approx code
    '''Returns Inverse Hessian approximation (H matrix) & number of points used to estimate it'''
    d = positions.shape[1]
    npos = positions.shape[0]
    if H is None:
        H = np.eye(d) # initial hessian
    
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    points_used = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1)) 
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        if r < 0:               # Curvature condition to ensure positive definite
            continue
        points_used += 1
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update

        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    
    return H, points_used


# def compute_Q_host(U_B):
#     U, B = U_B
#     UU, DD, VV = spys.linalg.svds(U, k=B)
#     return UU * np.sqrt(DD)

# def compute_Q(U_B):
#     result_shape = jax.ShapeDtypeStruct((U_B[0].shape[0], U_B[1]), U_B[0].dtype)
#     return jax.pure_callback(compute_Q_host, result_shape, U_B)


# def get_sqrt(M):
#     M_root = sqrtm(M)
#     # if xla_bridge.get_backend().platform == 'gpu':
#     #     result_shape = jax.ShapeDtypeStruct(M.shape, M.dtype)
#     #     M_root = jax.pure_callback(lambda x:sqrtm_sp(x).astype(M.dtype), result_shape, M) # sqrt can be complex sometimes, we only want real part                                                                                                                                                                             
#     # elif xla_bridge.get_backend().platform == 'cpu':
#     # else:
#     #     print("Backend not recongnized in get_sqrt function. Should be either gpu or cpu")
#     #     raise
#     return M_root.real




def BAM_hessian_approx(samples, vs, mu0=None, S0=None, reg=1e4, rank=-1):
    """
    Returns updated mean and covariance matrix with GSM updates.                                                                                                                                       For a batch, this is simply the mean of updates for individual samples.
    """

    try:
        assert len(samples.shape) == 2
        assert len(vs.shape) == 2
        B = samples.shape[0]
        D = samples.shape[1]

        if mu0 is None: mu0 = samples.mean(axis=0)
        if S0 is None:
            for i in range(1, B):
                s = samples[i] - samples[0]
                y = vs[i] - vs[0] 
                r = 1/(np.dot(y,s))
                if r > 0 :
                    S0 = np.eye(D) / np.dot(y,s) *np.dot(y, y) 
                    break
            if  r <= 0:
                S0 = np.eye(D) * 1e-3

        xbar = np.mean(samples, axis=0)
        outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
        xdiff = samples - xbar
        C = np.mean(outer_map(jnp.array(xdiff), jnp.array(xdiff)), axis=0)

        gbar = np.mean(vs, axis=0)
        gdiff = vs - gbar
        G = np.mean(outer_map(gdiff, gdiff), axis=0)
        I = np.identity(D)

        U = reg * G + (reg)/(1+reg) * np.outer(gbar, gbar)
        V = S0 + reg * C + (reg)/(1+reg) * np.outer(mu0 - xbar, mu0 - xbar)
        # U = G 
        # V = C 

        mat = I + 4 * np.matmul(U, V)
        # S = 2 * np.matmul(V, np.linalg.inv(I + sqrtm(mat).real))
        # S = 2 * np.linalg.solve(I + get_sqrt(mat).T, V.T)
        mat_root = sqrtm(mat).real
        S = 2 * np.linalg.solve(I + mat_root.T, V.T)

        n = B
        if np.isnan(S).any() or (S is None):
            raise
    except Exception as e:
        print('exception in bam hessian : ', e)
        S, n = BFGS_hessian_approx(samples, vs)
    return S, n



def BAM_hessian_approx_bfgsinit(samples, vs, mu0=None, S0=None, reg=1e6):
    """
    Returns updated mean and covariance matrix with GSM updates.                                                                                                                                       For a batch, this is simply the mean of updates for individual samples.
    """
    try:
        S0 = BFGS_hessian_approx(samples, vs)[0]
    except:
        S0 = None
    return BAM_hessian_approx(samples, vs, S0=S0, reg=reg)


def BAM_hessian_approx_bfgscorrect(samples, vs, mu0=None, S0=None, reg=1e6):
    """
    Returns updated mean and covariance matrix with GSM updates.                                                                                                                                       For a batch, this is simply the mean of updates for individual samples.
    """
    S, n = BAM_hessian_approx(samples, vs)
    if S is not None :
        S, n = BFGS_hessian_approx(samples, vs, S)
    return S, n


def Fisher_hessian_approx(samples, vs, H=None):
    """
    """
    assert len(samples.shape) == 2
    assert len(vs.shape) == 2
    B = samples.shape[0]
    D = samples.shape[1]

    if H is None: 
        for i in range(1, B):
            s = samples[i] - samples[0]
            y = vs[i] - vs[0] 
            r = 1/(np.dot(y,s))
            if r > 0 :
                H = np.eye(D) / np.dot(y,s) *np.dot(y, y) 
                break

    outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
    H = H + np.mean(outer_map(vs, vs), axis=0)

    return H, B
