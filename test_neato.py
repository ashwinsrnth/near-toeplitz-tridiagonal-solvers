from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from neato import NearToeplitzSolver, ToeplitzSolver, NearToeplitzBoundaryCorrectedSolver
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose
import pytest

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

def solve_near_toeplitz(coeffs, rhs):
    n = rhs.size
    a = np.ones(n, dtype=np.float64)*coeffs[2]
    b = np.ones(n, dtype=np.float64)*coeffs[3]
    c = np.ones(n, dtype=np.float64)*coeffs[4]
    a[0] = 0
    b[0] = coeffs[0]
    c[0] = coeffs[1]
    a[-1] = coeffs[5]
    b[-1] = coeffs[6]
    c[-1] = 0
    x = scipy_solve_banded(a, b, c, rhs)
    return x

def solve_toeplitz(coeffs, rhs):
    n = rhs.size
    a = np.ones(n, dtype=np.float64)*coeffs[0]
    b = np.ones(n, dtype=np.float64)*coeffs[1]
    c = np.ones(n, dtype=np.float64)*coeffs[2]
    a[0] = 0.
    c[-1] = 0.
    x = scipy_solve_banded(a, b, c, rhs)
    return x

@pytest.mark.parametrize("shmem", [True, False])
def test_single_near_toeplitz_system(shmem):
    n = 32
    d = np.random.rand(n)
    d_d = gpuarray.to_gpu(d)
    coeffs = np.random.rand(7)
    solver = NearToeplitzSolver(n, 1, coeffs, use_shmem=shmem)
    solver.solve(d_d)
    x = d_d.get()
    x_true = solve_near_toeplitz(coeffs, d)
    assert_allclose(x_true, x)

@pytest.mark.parametrize("shmem", [True, False])
def test_many_near_toeplitz_sytems(shmem):
    n = 32
    nrhs = 16
    d = np.random.rand(nrhs, n)
    d_d = gpuarray.to_gpu(d)
    coeffs = np.random.rand(7)
    solver = NearToeplitzSolver(n, nrhs, coeffs, use_shmem=shmem)
    solver.solve(d_d)
    x = d_d.get()

    for i in range(nrhs):
        x_true = solve_near_toeplitz(coeffs, d[i, :])
        assert_allclose(x_true, x[i, :])

def test_single_toeplitz_system(shmem=True):
    n = 32
    d = np.random.rand(n)
    d_d = gpuarray.to_gpu(d)
    coeffs = np.random.rand(3)
    solver = ToeplitzSolver(n, 1, coeffs, use_shmem=shmem)
    solver.solve(d_d)
    x = d_d.get()
    x_true = solve_toeplitz(coeffs, d)
    assert_allclose(x_true, x)

def test_many_toeplitz_systems(shmem=True):
    n = 32
    nrhs = 16
    d = np.random.rand(nrhs, n)
    d_d = gpuarray.to_gpu(d)
    coeffs = np.random.rand(3)
    solver = ToeplitzSolver(n, nrhs, coeffs, use_shmem=shmem)
    solver.solve(d_d)
    x = d_d.get()

    for i in range(nrhs):
        x_true = solve_toeplitz(coeffs, d[i, :])
        assert_allclose(x_true, x[i, :])

def test_solve_inner_systems():
    n = 34
    d = np.random.rand(n)
    d_d = gpuarray.to_gpu(d)
    coeffs = np.random.rand(7)
    solver = NearToeplitzBoundaryCorrectedSolver(n, 1, coeffs)
    solver._solve_inner_toeplitz_systems(d_d)
    x = d_d.get()
    x_true = np.zeros_like(x)*0.0
    x_true[1:-1] = solve_toeplitz(coeffs[2:5], d[1:-1])
    x_true[0] = -d[0]/coeffs[0]
    x_true[-1] = -d[-1]/coeffs[-1]
    assert_allclose(x_true, x)
