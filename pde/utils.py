import numpy as np

def one_d_poisson(n, h, include_boundary=True):
    """
    Returns an nxn matrix that solves the difference 
    equations with fixed boundaries
    """
    a = np.zeros((n,n))
    if include_boundary:
        np.fill_diagonal(a[1:-1,1:-1], 2.)
        np.fill_diagonal(a[1:-1,:-1], -1.)
        np.fill_diagonal(a[1:-1,2:], -1.)
        a = a/h**2
        a[0,0]=a[-1,-1]=1.
        return  a
    else:
        np.fill_diagonal(a, 2.)
        np.fill_diagonal(a[:-1,1:], -1.)
        np.fill_diagonal(a[1:,:-1], -1.)
        a = a/h**2
        return  a

def one_d_heat_btcs(n, dx, dt):
    """Au^{n+1} = u^n for heatequation

    Backward time, centered difference space
    """
    a = np.zeros((n,n), dtype=np.double)
    beta = dt/(dx**2)
    np.fill_diagonal(a[1:-1,1:-1], 1+2.*beta)
    np.fill_diagonal(a[1:-1,:-1], -beta)
    np.fill_diagonal(a[1:-1,2:],  -beta)
    a[0,0]=a[-1,-1]=1.
    return  a

def interpolater(n):
    """n must be odd
    """
    N = (n-1)/2
    R = np.zeros((N,n))
    col = 0
    for i in range(N):
        R[i,col:col+3] = [1./4., 1./2., 1./4.]
        col += 2
        
    return R

