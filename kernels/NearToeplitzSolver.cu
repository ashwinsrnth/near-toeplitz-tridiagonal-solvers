__global__ void forwardReductionKernel(const double *a_d,
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

__global__ void backwardSubstitutionKernel(const double *a_d,
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
