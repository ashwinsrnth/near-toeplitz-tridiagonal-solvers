import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import kernels
from near_toeplitz import *
from scipy.linalg import *
from numpy.testing import *
np.random.seed(1352031225)
def scipy_solve_banded(a, b, c, rhs):
    '''
    Solve the tridiagonal system described
    by a, b, c, and rhs.
    a: lower off-diagonal array (first element ignored)
    b: diagonal array
    c: upper off-diagonal array (last element ignored)
    rhs: right hand side of the system
    '''
    l_and_u = (1, 1)
    ab = np.vstack([np.append(0, c[:-1]),
                    b,
                    np.append(a[1:], 0)])
    x = solve_banded(l_and_u, ab, rhs)
    return x

nz = 512
ny = 512
nx = 512
d = np.random.rand(nz, ny, nx)
d_d = gpuarray.to_gpu(d)
cfd = NearToeplitzSolver(d.shape, [1., 2., 1./4, 1., 1./4, 2., 1.])
start = cuda.Event()
end = cuda.Event()

for i in range(10):
    start.record()
    cfd.solve(d_d, (1, 1))
    end.record()
    end.synchronize()
    print start.time_till(end)*1e-3

'''
a = np.ones(nx, dtype=np.float64)*1./4
b = np.ones(nx, dtype=np.float64)
c = np.ones(nx, dtype=np.float64)*1./4
a[0] = 0
c[-1] = 0
c[0] = 2
a[-1] = 2

x = d_d.get()

for i in range(nz):
    for j in range(ny):
        print i, j
        x_true = scipy_solve_banded(a, b, c, d[i, j, :])
        assert_allclose(x_true, x[i, j, :])
'''
