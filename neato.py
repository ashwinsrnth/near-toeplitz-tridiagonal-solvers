import os
import jinja2
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_la

class NearToeplitzSolver:

    def __init__(self, n, nrhs, coeffs, use_shmem=False):
        '''
        Parameters
        ----------
        n: The size of the tridiagonal system.
        nrhs: The number of right hand sides
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            [b1, c1, ai, bi, ci, an, bn]
        use_shmem: Use shared memory
        '''
        self.n = n
        self.nrhs = nrhs
        self.coeffs = coeffs
        self.use_shmem = use_shmem

        # check that system_size is a power of 2:
        assert np.int(np.log2(self.n)) == np.log2(self.n)

        # compute coefficients a, b, etc.,
        a, b, c, k1, k2, b_first, k1_first, k1_last = _precompute_coefficients(self.n, self.coeffs)

        # copy coefficients to buffers:
        self.a_d = gpuarray.to_gpu(a)
        self.b_d = gpuarray.to_gpu(b)
        self.c_d = gpuarray.to_gpu(c)
        self.k1_d = gpuarray.to_gpu(k1)
        self.k2_d = gpuarray.to_gpu(k2)
        self.b_first_d = gpuarray.to_gpu(b_first)
        self.k1_first_d = gpuarray.to_gpu(k1_first)
        self.k1_last_d = gpuarray.to_gpu(k1_last)
        
        # get kernels:
        self.forward_reduction, self.backward_substitution = self._get_globalmem_kernels()
        self.shmem_cyclic_reduction = self._get_sharedmem_kernels()

    def solve(self, x_d):
        '''
        Solve the tridiagonal system
        for rhs d, given storage for the solution
        vector in x.
        '''
        [b1, c1,
                ai, bi, ci,
                an, bn] = self.coeffs

        # CR algorithm
        # ============================================
        
        if self.use_shmem:
            self.shmem_cyclic_reduction.prepared_call(
                     (self.nrhs, 1, 1),
                     (self.n//2, 1, 1),
                     self.a_d.gpudata,
                     self.b_d.gpudata,
                     self.c_d.gpudata,
                     x_d.gpudata,
                     self.k1_d.gpudata,
                     self.k2_d.gpudata,
                     self.b_first_d.gpudata,
                     self.k1_first_d.gpudata,
                     self.k1_last_d.gpudata,
                     b1, c1, ai, bi, ci)

        else:
            stride = 1
            for i in np.arange(int(np.log2(self.n))):
                stride *= 2
                self.forward_reduction.prepared_call(
                    (self.nrhs, 1, 1), (self.n//stride, 1, 1),
                    self.a_d.gpudata,
                    self.b_d.gpudata,
                    self.c_d.gpudata,
                    x_d.gpudata,
                    self.k1_d.gpudata, self.k2_d.gpudata,
                    self.b_first_d.gpudata,
                    self.k1_first_d.gpudata, self.k1_last_d.gpudata,
                    self.n, stride)

            # `stride` is now equal to `n`
            for i in np.arange(int(np.log2(self.n))-1):
                stride //= 2
                self.backward_substitution.prepared_call(
                    (self.nrhs, 1, 1), (self.n//stride, 1, 1),
                    self.a_d.gpudata,
                    self.b_d.gpudata,
                    self.c_d.gpudata,
                    x_d.gpudata,
                    self.b_first_d.gpudata,
                    b1, c1, ai, bi, ci,
                    self.n, stride)
    
    def _get_globalmem_kernels(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        with open(dir+'/_impls/globalmem.cu') as f:
            kernel_template = f.read()
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=self.n, shared_size=self.n/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])
        forward_reduction = module.get_function(
                'forwardReductionKernel')
        backward_substitution = module.get_function(
                'backwardSubstitutionKernel')
        forward_reduction.prepare('PPPPPPPPPii')
        backward_substitution.prepare('PPPPPdddddii')
        return forward_reduction, backward_substitution

    def _get_sharedmem_kernels(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        with open(dir+'/_impls/sharedmem.jinja2') as f:
            kernel_template = f.read()
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=self.n, shared_size=self.n/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])
        shmem_cyclic_reduction = module.get_function(
            'nearToeplitzCyclicReductionKernel')
        shmem_cyclic_reduction.prepare('PPPPPPPPPddddd')
        return shmem_cyclic_reduction

class NearToeplitzBoundaryCorrectedSolver:
    def __init__(self, n, nrhs, coeffs, use_shmem=True):
        '''
        Parameters
        ----------
        n: The size of the tridiagonal system.
        nrhs: The number of right hand sides
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            [b1, c1, ai, bi, ci, an, bn]
        use_shmem: Use shared memory
        '''
        self.n = n
        self.nrhs = nrhs
        self.coeffs = coeffs
        self.use_shmem = use_shmem

        # check that inner system size is a power of 2:
        assert np.int(np.log2(self.n - 2)) == np.log2(self.n - 2)

        # compute coefficients a, b, etc.,
        b1, c1, ai, bi, ci, an, bn = self.coeffs
        a, b, c, k1, k2, _, _, _ = _precompute_coefficients(self.n-2, [bi, ci, ai, bi, ci, ai, bi])

        # copy coefficients to buffers:
        self.a_d = gpuarray.to_gpu(a)
        self.b_d = gpuarray.to_gpu(b)
        self.c_d = gpuarray.to_gpu(c)
        self.k1_d = gpuarray.to_gpu(k1)
        self.k2_d = gpuarray.to_gpu(k2)

        # get kernels:
        self.shmem_cyclic_reduction_inner, self.boundary_correction = self._get_sharedmem_kernels()

        # prepare boundary correction:
        self._prepare_boundary_correction()
        
    def solve(self, x_d):
        self._solve_inner_toeplitz_systems(x_d)
        self._apply_boundary_correction(x_d)

    def _solve_inner_toeplitz_systems(self, x_d):
        '''
        Solve the inner toeplitz systems system
        for rhs d, given storage for the solution
        vector in x.
        '''
        b1, c1, ai, bi, ci, an, bn = self.coeffs

        # CR algorithm
        # ============================================
        self.shmem_cyclic_reduction_inner.prepared_call(
                 (self.nrhs, 1, 1),
                 ((self.n-2)//2, 1, 1),
                 self.a_d.gpudata,
                 self.b_d.gpudata,
                 self.c_d.gpudata,
                 x_d.gpudata,
                 self.k1_d.gpudata,
                 self.k2_d.gpudata,
                 b1, ai, bi, ci, bn
                 )


    def _apply_boundary_correction(self, x_d):
        self.boundary_correction.prepared_call(
                (self.nrhs, 1, 1),
                (1, 1, 1),
                x_d.gpudata,
                self.x_UH_i_d.gpudata,
                self.x_LH_i_d.gpudata,
                self.a_reduced_d.gpudata,
                self.b_reduced_d.gpudata,
                self.c_reduced_d.gpudata,
                self.c2_reduced_d.gpudata,
                self.x_LH_1,
                self.x_UH_N)

    def _get_sharedmem_kernels(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        with open(dir+'/_impls/sharedmem.jinja2') as f:
            kernel_template = f.read()
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=(self.n-2), shared_size=(self.n-1)/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])

        shmem_cyclic_reduction_inner = module.get_function(
            'innerToeplitzCyclicReductionKernel')
        shmem_cyclic_reduction_inner.prepare('PPPPPPddddd')

        boundary_correction = module.get_function(
            'boundaryCorrectionKernel')
        boundary_correction.prepare('PPPPPPPdd')

        return shmem_cyclic_reduction_inner, boundary_correction

    def _prepare_boundary_correction(self):
        N = self.n - 2
        
        # upper homogeneous solution for middle
        rhs_UH_i = np.zeros(N)*0.0
        rhs_UH_i[0] = -self.coeffs[2]
        print(rhs_UH_i)
        print([self.coeffs[2], self.coeffs[3], self.coeffs[4]])
        x_UH_i = _solve_toeplitz_system([self.coeffs[2], self.coeffs[3], self.coeffs[4]], rhs_UH_i)
        print(x_UH_i)

        # lower homogeneous solution for middle
        rhs_LH_i = np.zeros(N)*0.0
        rhs_LH_i[-1] = -self.coeffs[4]
        x_LH_i = _solve_toeplitz_system([self.coeffs[2], self.coeffs[3], self.coeffs[4]], rhs_LH_i)

        # lower and upper homogeneous solution for top and bottom
        self.x_LH_1 = -self.coeffs[1]/self.coeffs[0]
        self.x_UH_N = -self.coeffs[-2]/self.coeffs[-1]

        # form the reduced system LHS: 
        a_reduced = np.array([0, -1, x_UH_i[-1], -1.])
        b_reduced = np.array([self.x_LH_1, x_UH_i[0], x_LH_i[-1], self.x_UH_N])
        c_reduced = np.array([-1, x_LH_i[0], -1, 0.])

        # copy everything to GPU:
        self.x_UH_i_d = gpuarray.to_gpu(x_UH_i)
        self.x_LH_i_d = gpuarray.to_gpu(x_LH_i)
        self.a_reduced_d = gpuarray.to_gpu(a_reduced)
        self.b_reduced_d = gpuarray.to_gpu(b_reduced)
        self.c_reduced_d = gpuarray.to_gpu(c_reduced)
        self.c2_reduced_d = gpuarray.to_gpu(c_reduced)


class ToeplitzSolver:

    def __init__(self, n, nrhs, coeffs, use_shmem=False):
        '''
        Parameters
        ----------
        n: The size of the tridiagonal system.
        nrhs: The number of right hand sides
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            [b1, c1, ai, bi, ci, an, bn]
        use_shmem: Use shared memory
        '''
        self.n = n
        self.nrhs = nrhs
        self.coeffs = coeffs
        self.use_shmem = use_shmem

        # check that system_size is a power of 2:
        assert np.int(np.log2(self.n)) == np.log2(self.n)

        # compute coefficients a, b, etc.,
        ai, bi, ci = self.coeffs
        a, b, c, k1, k2, _, _, _ = _precompute_coefficients(self.n, [bi, ci, ai, bi, ci, ai, bi])

        # copy coefficients to buffers:
        self.a_d = gpuarray.to_gpu(a)
        self.b_d = gpuarray.to_gpu(b)
        self.c_d = gpuarray.to_gpu(c)
        self.k1_d = gpuarray.to_gpu(k1)
        self.k2_d = gpuarray.to_gpu(k2)
        
        # get kernels:
        self.shmem_cyclic_reduction = self._get_sharedmem_kernels()

    def solve(self, x_d):
        '''
        Solve the tridiagonal system
        for rhs d, given storage for the solution
        vector in x.
        '''
        ai, bi, ci = self.coeffs

        # CR algorithm
        # ============================================
        self.shmem_cyclic_reduction.prepared_call(
                 (self.nrhs, 1, 1),
                 (self.n//2, 1, 1),
                 self.a_d.gpudata,
                 self.b_d.gpudata,
                 self.c_d.gpudata,
                 x_d.gpudata,
                 self.k1_d.gpudata,
                 self.k2_d.gpudata,
                 ai, bi, ci
                 )

    def _get_sharedmem_kernels(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        with open(dir+'/_impls/sharedmem.jinja2') as f:
            kernel_template = f.read()
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=self.n, shared_size=self.n/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])
        shmem_cyclic_reduction = module.get_function(
            'toeplitzCyclicReductionKernel')
        shmem_cyclic_reduction.prepare('PPPPPPddd')
        return shmem_cyclic_reduction

def _precompute_coefficients(system_size, coeffs):
    '''
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of constant coefficients,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(system_size)-1,
    as opposed to arrays sized system_size.

    Values at the first and last point at each step
    need to be stored seperately.

    The last values for a and b are required only at
    the final stage of forward reduction (the 2-by-2 solve),
    so for convenience, these two scalar values are stored
    at the end of arrays a and b.

    -- See the paper
    "Fast Tridiagonal Solvers on the GPU"
    '''
    # these arrays technically have length 1 more than required:
    log2_system_size = int(np.log2(system_size))

    a = np.zeros(log2_system_size, np.float64)
    b = np.zeros(log2_system_size, np.float64)
    c = np.zeros(log2_system_size, np.float64)
    k1 = np.zeros(log2_system_size, np.float64)
    k2 = np.zeros(log2_system_size, np.float64)

    b_first = np.zeros(log2_system_size, np.float64)
    k1_first = np.zeros(log2_system_size, np.float64)
    k1_last = np.zeros(log2_system_size, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_system_size - 1
    for i in range(num_reductions):
        if i == 0:
            k1[i] = ai/bi
            k2[i] = ci/bi
            a[i] = -ai*k1[i]
            b[i] = bi - ci*k1[i] - ai*k2[i]
            c[i] = -ci*k2[i]

            k1_first[i] = ai/b1
            b_first[i] = bi - c1*k1_first[i] - ai*k2[i]

            k1_last[i] = an/bi
            a_last = -(ai)*k1_last[i]
            b_last = bn - (ci)*k1_last[i]
        else:
            k1[i] = a[i-1]/b[i-1]
            k2[i] = c[i-1]/b[i-1]
            a[i] = -a[i-1]*k1[i]
            b[i] = b[i-1] - c[i-1]*k1[i] - a[i-1]*k2[i]
            c[i] = -c[i-1]*k2[i]

            k1_first[i] = a[i-1]/b_first[i-1]
            b_first[i] = b[i-1] - c[i-1]*k1_first[i] - a[i-1]*k2[i]

            k1_last[i] = a_last/b[i-1]
            a_last = -a[i-1]*k1_last[i]
            b_last = b_last - c[i-1]*k1_last[i]

    # put the last values for a and b at the end of the arrays:
    a[-1] = a_last
    b[-1] = b_last

    return a, b, c, k1, k2, b_first, k1_first, k1_last


def _solve_toeplitz_system(coeffs, rhs):
    N = rhs.size
    M = sparse.diags((np.ones(N-1)*coeffs[0], np.ones(N)*coeffs[1], np.ones(N-1)*coeffs[2]),
                offsets=[-1, 0, +1])
    return sparse_la.spsolve(M, rhs)
