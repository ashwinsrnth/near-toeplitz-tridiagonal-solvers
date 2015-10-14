#include <stdio.h>
__global__ void forwardReduction( double *a_d,
                                double *b_d,
                                double *c_d,
                                double *d_d,
                               int nx,
                               int ny,
                               int nz,
                               int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.y;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
    int gi3d0 = giz*(nx*ny) + giy*nx + 0; 
    int i = (stride-1) + gix*stride;
    int m, n;
    double k1, k2;
    double x_m, x_n;

    // forward reduction
    if (stride > nx/2)
    {
        // now solve the two-by-two system:
        stride /= 2;
        m = stride-1;
        n = 2*stride-1;
        x_m = (d_d[gi3d0+m]*b_d[n] - c_d[m]*d_d[gi3d0+n])/ \
                 (b_d[m]*b_d[n] - c_d[m]*a_d[n]);
        x_n = (b_d[m]*d_d[gi3d0+n] - d_d[gi3d0+m]*a_d[n])/ \
                 (b_d[m]*b_d[n] - c_d[m]*a_d[n]);
        d_d[gi3d0+m] = x_m;
        d_d[gi3d0+n] = x_n;
    }
    else
    {
        if (i == (nx-1))
        {
            k1 = a_d[i]/b_d[i-stride/2];
            a_d[i] = -a_d[i-stride/2]*k1;
            b_d[i] = b_d[i] - c_d[i-stride/2]*k1;
            d_d[gi3d0+i] = d_d[gi3d0+i] - d_d[gi3d0+i-stride/2]*k1;
        }
        else
        {
            k1 = a_d[i]/b_d[i-stride/2];
            k2 = c_d[i]/b_d[i+stride/2];
            a_d[i] = -a_d[i-stride/2]*k1;
            b_d[i] = b_d[i] - c_d[i-stride/2]*k1 - a_d[i+stride/2]*k2;
            c_d[i] = -c_d[i+stride/2]*k2;
            d_d[gi3d0+i] = d_d[gi3d0+i] - d_d[gi3d0+i-stride/2]*k1 - d_d[gi3d0+i+stride/2]*k2;
        }
    }
}

__global__ void backwardSubstitution( double *a_d,
                                    double *b_d,
                                    double *c_d,
                                    double *d_d,
                                    int nx, 
                                    int ny,
                                    int nz,
                                    int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
    int gi3d0 = giz*(nx*ny) + giy*nx + 0; 
    int i = (stride/2-1) + gix*stride;
    if (i < stride)
    {
        d_d[gi3d0+i] = (d_d[gi3d0+i] - c_d[i]*d_d[gi3d0+i+stride/2])/b_d[i];
    }
    else
    {
        d_d[gi3d0+i] = (d_d[gi3d0+i] - a_d[i]*d_d[gi3d0+i-stride/2] - c_d[i]*d_d[gi3d0+i+stride/2])/b_d[i];
    }
}
