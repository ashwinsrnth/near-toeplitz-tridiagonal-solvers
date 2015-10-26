from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from kernels import *
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

nz = 16
ny = 16
nx = 16

a = np.random.rand(nx)
b = np.random.rand(nx)
c = np.random.rand(nx)
d = np.random.rand(nz, ny, nx)

a_d = gpuarray.to_gpu(a)
b_d = gpuarray.to_gpu(b)
c_d = gpuarray.to_gpu(c)
d_d = gpuarray.to_gpu(d)

forward_reduction, back_substitution = get_funcs('kernels.cu',
    'forwardReduction', 'backwardSubstitution')
forward_reduction.prepare('PPPPii')
back_substitution.prepare('PPPPii')

# CR algorithm
# ============================================
start = cuda.Event()
end = cuda.Event()

start.record()
for i in range(1):
    stride = 1
    for i in np.arange(np.log2(nx)):
        stride *= 2
        forward_reduction.prepared_call((nz*ny, 1, 1), (nx/stride, 1, 1),
                a_d.gpudata, b_d.gpudata, c_d.gpudata,
                    d_d.gpudata, nx, stride)

    # `stride` is now equal to `system_size`
    for i in np.arange(np.log2(nx)-1):
        stride /= 2
        back_substitution.prepared_call((nz*ny, 1, 1), (nx/stride, 1, 1),
                a_d.gpudata, b_d.gpudata, c_d.gpudata,
                    d_d.gpudata, nx, stride)
end.record()
end.synchronize()
print start.time_till(end)*1e-3

x = d_d.get()
for i in range(nz):
    for j in range(ny):
        x_true = scipy_solve_banded(a, b, c, d[i, j, :])
        assert_allclose(x_true, x[i, j, :])
