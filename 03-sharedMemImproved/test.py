from pycuda import autoinit
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from kernels import get_funcs
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose

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

nz = 256
ny = 256
nx = 256

a = np.random.rand(nx)
b = np.random.rand(nx)
c = np.random.rand(nx)
d = np.random.rand(nz, ny, nx)

a_d = gpuarray.to_gpu(a)
b_d = gpuarray.to_gpu(b)
c_d = gpuarray.to_gpu(c)

smemcyclicreduction, = get_funcs('kernels.cu', 'sharedMemCyclicReduction')
smemcyclicreduction.prepare('PPPPiiiii')

by = 1
bz = 1

for i in range(5):
    d_d = gpuarray.to_gpu(d)
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    smemcyclicreduction.prepared_call((1, ny/by, nz/bz), (nx/2, by, bz),
            a_d.gpudata,
            b_d.gpudata,
            c_d.gpudata,
            d_d.gpudata,
            nx,
            ny,
            nz,
            nx,
            by)
    end.record()
    end.synchronize()
    print start.time_till(end)*1e-3

#x = d_d.get()
#for i in range(nz):
#    for j in range(ny):
#        x_true = scipy_solve_banded(a, b, c, d[i, j, :])
#        assert_allclose(x_true, x[i, j, :])
