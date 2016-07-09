#!/usr/bin/python

import sys
sys.path.append('../')
import neato
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from neato import NearToeplitzSolver
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('N',
        help='Size of tridiagonal system to solve',
        type=int)
parser.add_argument('nrhs',
        help='Number of right hand sides to solve system for',
        type=int)
parser.add_argument('--use_shmem',
        dest='use_shmem', action='store_true')
args = parser.parse_args()

N, nrhs = args.N, args.nrhs

start = cuda.Event()
end = cuda.Event()
nsteps = 100

d = np.random.rand(nrhs, N)
d_d = gpuarray.to_gpu(d)
solver = NearToeplitzSolver(N, nrhs,
        (1., 2., 1./4, 1., 1./4, 2., 1.), use_shmem=args.use_shmem)
solver.solve(d_d) # warm up

start.record()
for i in range(nsteps):
    solver.solve(d_d)
end.record()
end.synchronize()

print start.time_till(end)/nsteps
