import numpy as np
from scipy.linalg import solve_banded 
from near_toeplitz import *

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
solver = NearToeplitzSolver(N, 1, (1., 2., 1./4, 1., 1./4, 2., 1.))
d = np.random.rand(N)
a = np.ones(N, dtype=np.float64)*(1./4)
b = np.ones(N, dtype=np.float64)
c = np.ones(N, dtype=np.float64)*(1./4)
a[-1] = 2
c[0] = 2

x_true = scipy_solve_banded(a, b, c, d)
solver.solve(d)
print x_true
print d
