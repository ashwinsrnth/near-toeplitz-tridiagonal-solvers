import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import kernels
from near_toeplitz import *

nz = 128
ny = 128
nx = 128
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
