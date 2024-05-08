import numpy as np

def Hessian_approx(positions, gradients, H=None, approx_type='bfgs'):
    if approx_type == 'bfgs':
        return BFGS_hessian_approx(positions, gradients, H)
    elif approx_type == 'fisher':
        raise NotImplementedError
    elif approx_type == 'gsm':
        raise NotImplementedError


def BFGS_hessian_approx(positions, gradients, H=None):
    '''Hessian approximation with BFGS Quasi-Newton Method (B matrix)'''
    d = positions.shape[1]
    npos = positions.shape[0]
    #if H is None: 
    #    H = np.eye(d) # initial hessian. Moved to initilization below which is better
   
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    not_pos = 0
    point_used = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla
        r = 1/(np.dot(y,s))
        if r < 0:               #Curvature condition to ensure positive definite
            not_pos +=1 
            continue
        if (H is None) : #initialize based on, but before first update. Taken from Nocedal
            #H = np.eye(x.size) * np.dot(y,s)/np.dot(y, y) #gets confusing if we multiply or divide
            H = np.eye(x.size) / np.dot(y,s) *np.dot(y, y) 
        point_used +=1
        z = np.dot(H, s)
        update = np.outer(y, y) / np.dot(s, y) - np.outer(z, z) / np.dot(s, z)
        H += update
        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    return H, point_used 

    
