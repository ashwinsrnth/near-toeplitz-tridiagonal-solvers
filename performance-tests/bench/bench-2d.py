import sys
sys.path.append('../../sharedmem')
import neato
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np

from neato import NearToeplitzSolver
from prettytable import PrettyTable

sizes = (32, 64, 128, 256, 512, 1024, 2048)
num_systems = (32**2, 64**2, 128**2, 256**2, 512**2, 1024**2, 2048**2)

start = cuda.Event()
end = cuda.Event()
nsteps = 100

table = PrettyTable()
for N, nrhs in zip(sizes, num_systems):
    in_bytes = N*nrhs*8
    out_bytes = N*nrhs*8
    d = np.random.rand(nrhs, N)
    d_d = gpuarray.to_gpu(d)
    solver = NearToeplitzSolver(N, nrhs, (1., 2., 1./4, 1., 1./4, 2., 1.))
    for i in range(nsteps):
        if i == 1: start.record()
        solver.solve(d_d)
    end.record()
    end.synchronize()
    time_per_solve = (start.time_till(end)/(nsteps-1))*1e-3
    throughput = (in_bytes+out_bytes)/(time_per_solve * 1e9)
    table.add_row((N, nrhs, time_per_solve, throughput))
table.field_names = ('System size', 'Number of systems', 'Time to solve (ms)', 'Bandwidth (GB/s)')
table.align = 'l'

print table
