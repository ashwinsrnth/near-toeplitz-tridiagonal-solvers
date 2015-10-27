import numpy as np
from scipy.linalg import solve_banded
from near_toeplitz import NearToeplitzSolver, _precompute_coefficients
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

N = 16
coeffs = (1., 2., 1./4, 1., 2./4, 2., 1.) 
solver = NearToeplitzSolver(N, 1, coeffs)
d = np.random.rand(N)
a = np.ones(N, dtype=np.float64)*coeffs[2]
b = np.ones(N, dtype=np.float64)*coeffs[3]
c = np.ones(N, dtype=np.float64)*coeffs[4]
b[0] = coeffs[0]
c[0] = coeffs[1]
a[-1] = coeffs[-2]
b[-1] = coeffs[-1]

x_true = scipy_solve_banded(a, b, c, d)
solver.solve(d)

assert_allclose(x_true, d)

a, b, c, k1, k2, b_first, k1_first, k1_last = _precompute_coefficients(N, coeffs) 
print 'a: ', a
print 'b: ', b
print 'c: ', c
print 'k1:', k1
print 'k2:', k2
print 'b_first: ', b_first
print 'k1_first: ', k1_first
print 'k1_last: ', k1_last
