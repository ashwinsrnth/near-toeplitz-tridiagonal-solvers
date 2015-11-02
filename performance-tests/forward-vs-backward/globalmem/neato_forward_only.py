import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import jinja2
import pycuda.compiler as compiler

kernel_template = '''
__global__ void globalForwardReduction(const double *a_d,
                                const double *b_d,
                                const double *c_d,
                                double *d_d,
                                const double *k1_d,
                                const double *k2_d,
                                const double *b_first_d,
                                const double *k1_first_d,
                                const double *k1_last_d,
                                const int n,
                                int stride)
{
    int tix = threadIdx.x;
    int offset = blockIdx.x*n;
    int i;
    int j, k;
    int idx;
    double x_j, x_k;

    // forward reduction
    if (stride == n)
    {
        stride /= 2;
        j = log2((float)stride) - 1;
        k = log2((float)stride); // the last element
        x_j = (d_d[offset+stride-1]*b_d[k] - c_d[j]*d_d[offset+2*stride-1])/ \
                        (b_first_d[j]*b_d[k] - c_d[j]*a_d[k]);

        x_k = (b_first_d[j]*d_d[offset+2*stride-1] - d_d[offset+stride-1]*a_d[k])/ \
                        (b_first_d[j]*b_d[k] - c_d[j]*a_d[k]);
        d_d[offset+stride-1] = x_j;
        d_d[offset+2*stride-1] = x_k;
    }
    else
    {
        i = (stride-1) + tix*stride;
        idx = log2((float)stride) - 1;
        if (tix == 0)
        {
            d_d[offset+i] = d_d[offset+i] - d_d[offset+i-stride/2]*k1_first_d[idx] - d_d[offset+i+stride/2]*k2_d[idx];
        }
        else if (i == (n-1))
        {
            d_d[offset+i] = d_d[offset+i] - d_d[offset+i-stride/2]*k1_last_d[idx];
        }
        else
        {
            d_d[offset+i] = d_d[offset+i] - d_d[offset+i-stride/2]*k1_d[idx] - d_d[offset+i+stride/2]*k2_d[idx];
        }
    }
}

__global__ void globalBackSubstitution(const double *a_d,
                                    const double *b_d,
                                    const double *c_d,
                                    double *d_d,
                                    const double *b_first_d,
                                    const double b1,
                                    const double c1,
                                    const double ai,
                                    const double bi,
                                    const double ci,
                                    const int n,
                                    const int stride)

{
    int tix = threadIdx.x;
    int offset = blockIdx.x*n;
    int i;
    int idx;

    i = (stride/2-1) + tix*stride;

    if (stride == 2)
    {
        if (i == 0)
        {
            d_d[offset+i] = (d_d[offset+i] - c1*d_d[offset+i+1])/b1;
        }
        else
        {
            d_d[offset+i] = (d_d[offset+i] - (ai)*d_d[offset+i-1] - (ci)*d_d[offset+i+1])/bi;
        }
    }
    else
    {
        // rint rounds to the nearest integer
        idx = rint(log2((double)stride)) - 2;
        if (tix == 0) 
        {   
            d_d[offset+i] = (d_d[offset+i] - c_d[idx]*d_d[offset+i+stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[offset+i] = (d_d[offset+i] - a_d[idx]*d_d[offset+i-stride/2] - c_d[idx]*d_d[offset+i+stride/2])/b_d[idx];
        }
    }
}

'''

class NearToeplitzSolver:

    def __init__(self, n, nrhs, coeffs):
        '''
        Parameters
        ----------
        n: The size of the tridiagonal system.
        nrhs: The number of right hand sides
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            [b1, c1, ai, bi, ci, an, bn]
        '''
        self.n = n
        self.nrhs = nrhs
        self.coeffs = coeffs

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
        
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=self.n, shared_size=self.n/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])
        self.forward_reduction = module.get_function(
                'globalForwardReduction')
        self.back_substitution = module.get_function(
                'globalBackSubstitution')
        self.forward_reduction.prepare('PPPPPPPPPii')
        self.back_substitution.prepare('PPPPPdddddii')

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
                
        stride = 1
        for i in np.arange(int(np.log2(self.n))):
            stride *= 2
            self.forward_reduction.prepared_call((self.nrhs, 1, 1), (self.n/stride, 1, 1),
                self.a_d.gpudata, self.b_d.gpudata, self.c_d.gpudata, x_d.gpudata, self.k1_d.gpudata, self.k2_d.gpudata,
                    self.b_first_d.gpudata, self.k1_first_d.gpudata, self.k1_last_d.gpudata,
                            self.n, stride)


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

