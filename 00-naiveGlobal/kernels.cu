#include <stdio.h>
__global__ void forwardReduction( double *a_d,
                                double *b_d,
                                double *c_d,
                                double *d_d,
                                int n,
                                int stride)
{
    int tix = threadIdx.x;
    int offset = blockIdx.x*n;
    int i = (stride-1) + tix*stride;
    double k1, k2;
    double x_j, x_k;
    int j, k;

    // forward reduction
    if (stride == n)
    {
        // now solve the two-by-two system:
        stride /= 2;
        j = stride-1;
        k = 2*stride-1;
        x_j = (d_d[offset+j]*b_d[k] - c_d[j]*d_d[offset+k])/ \
                 (b_d[j]*b_d[k] - c_d[j]*a_d[k]);
        x_k = (b_d[j]*d_d[offset+k] - d_d[offset+j]*a_d[k])/ \
                 (b_d[j]*b_d[k] - c_d[j]*a_d[k]);
        d_d[offset+j] = x_j;
        d_d[offset+k] = x_k;
    }
    else
    {
        if (i == (n-1))
        {
            k1 = a_d[i]/b_d[i-stride/2];
            a_d[i] = -a_d[i-stride/2]*k1;
            b_d[i] = b_d[i] - c_d[i-stride/2]*k1;
            d_d[offset+i] = d_d[offset+i] - d_d[offset+i-stride/2]*k1;
        }
        else
        {
            k1 = a_d[i]/b_d[i-stride/2];
            k2 = c_d[i]/b_d[i+stride/2];
            a_d[i] = -a_d[i-stride/2]*k1;
            b_d[i] = b_d[i] - c_d[i-stride/2]*k1 - a_d[i+stride/2]*k2;
            c_d[i] = -c_d[i+stride/2]*k2;
            d_d[offset+i] = d_d[offset+i] - d_d[offset+i-stride/2]*k1 - d_d[offset+i+stride/2]*k2;
        }
    }
}

__global__ void backwardSubstitution( double *a_d,
                                    double *b_d,
                                    double *c_d,
                                    double *d_d,
                                    int n, 
                                    int stride)
{
    int tix = threadIdx.x;
    int offset = blockIdx.x*n;    
    int i = (stride/2-1) + tix*stride;
    if (tix == 0)
    {
        d_d[offset+i] = (d_d[offset+i] - c_d[i]*d_d[offset+i+stride/2])/b_d[i];
    }
    else
    {
        d_d[offset+i] = (d_d[offset+i] - a_d[i]*d_d[offset+i-stride/2] - c_d[i]*d_d[offset+i+stride/2])/b_d[i];
    }
}
