#include <stdio.h>
__global__ void sharedMemCyclicReduction( double *a_d,
                                double *b_d,
                                double *c_d,
                                double *d_d,
                                double *k1_d,
                                double *k2_d,
                                double *b_first_d,
                                double *k1_first_d,
                                double *k1_last_d,
                                const double b1,
                                const double c1,
                                const double ai,
                                const double bi,
                                const double ci,
                                int nx,
                                int ny,
                                int nz,
                                int bx,
                                int by) {
    /*

    */
    __shared__ double d_l[32];

    int ix = blockIdx.x*blockDim.x + threadIdx.x; 
    int iy = blockIdx.y*blockDim.y + threadIdx.y; 
    int iz = blockIdx.z*blockDim.z + threadIdx.z; 
    int lix = threadIdx.x; 
    int liy = threadIdx.y; 
    int liz = threadIdx.z; 
    int i, m, n;
    int idx, stride;

    int i3d = iz*(nx*ny) + iy*nx + ix;
    int i3d0 = iz*(nx*ny) + iy*nx + 0;
    int li3d = liz*(bx*by) + liy*bx + lix;
    int li3d0 = liz*(bx*by) + liy*bx + 0;

    double k1, k2;
    double d_m, d_n;

    /* When loading to shared memory, perform the first
       reduction step */
    idx = 0;
    if (lix == 0) {
        d_l[li3d0+lix] = d_d[i3d0+2*lix+1] - \
                    d_d[i3d0+2*lix]*k1_first_d[idx] - \
                    d_d[i3d0+2*lix+2]*k2_d[idx];
    }
    else if (lix == (nx/2-1)) {
        d_l[li3d0+lix] = d_d[i3d0+2*lix+1] - \
                     d_d[i3d0+2*lix]*k1_last_d[idx];
    }
    else {
        d_l[li3d0+lix] = d_d[i3d0+2*lix+1] - d_d[i3d0+2*lix]*k1_d[idx] - \
                    d_d[i3d0+2*lix+2]*k2_d[idx];
    }
    __syncthreads();
    
    /* First step of reduction is complete and 
       the coefficients are in shared memory */
    
    /* Do the remaining forward reduction steps: */
    stride = 1;
    for (int step=0; step<rint(log2((float) nx/2)); step++) {
        stride = stride*2;
        idx = idx + 1;
        if (lix < nx/(2*stride)) {
            i = (stride-1) + lix*stride;

            if (stride == nx/2) {
                if (lix == 0) {
                    m = rint(log2((float) nx/2)) - 1;
                    n = rint(log2((float) nx/2));

                    d_m = (d_l[li3d0+nx/4-1]*b_d[n] - \
                           c_d[m]*d_l[li3d0+nx/2-1])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

                    d_n = (b_first_d[m]*d_l[li3d0+nx/2-1] - \
                           d_l[li3d0+nx/4-1]*a_d[n])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

                    d_l[li3d0+nx/4-1] = d_m;
                    d_l[li3d0+nx/2-1] = d_n;
                }
            }

            else {
                ix = li3d0 + i;
                if (lix == 0) {
                    d_l[ix] = d_l[ix] - \
                                d_l[ix - stride/2]*k1_first_d[idx] - \
                                d_l[ix + stride/2]*k2_d[idx];
                }
                else if (i == (nx/2-1)) {
                    d_l[ix] = d_l[ix] - \
                                 d_l[ix - stride/2]*k1_last_d[idx];
                }
                else {
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1_d[idx] - \
                                d_l[ix+stride/2]*k2_d[idx];
                }
            }
        }
        __syncthreads();
    }
     
    idx = rint(log2((float) nx))-2;
    for (int step=0; step<rint(log2((float) nx))-2; step++) {
        stride = stride/2;
        idx = idx - 1;
        if (lix < nx/(2*stride)){
            i = (stride/2-1) + lix*stride;
            ix = li3d0 + i;
            if (lix == 0) {
                d_l[ix] = (d_l[ix] - c_d[idx]*d_l[ix+stride/2])/\
                            b_first_d[idx];
            }
            else {
                d_l[ix] = (d_l[ix] - a_d[idx]*d_l[ix-stride/2] -\
                            c_d[idx]*d_l[ix+stride/2])/b_d[idx];
            }
        }
        __syncthreads();
    }
    
    //When writing from shared memory, perform the last
    //substitution step
    if (lix == 0) {
        d_d[i3d0+2*lix] = (d_d[i3d0+2*lix] - c1*d_l[li3d0+lix])/b1;
        d_d[i3d0+2*lix+1] = d_l[li3d0+lix];
    }
    else {
        d_d[i3d0+2*lix] = (d_d[i3d0+2*lix] - ai*d_l[li3d0+lix-1] - ci*d_l[li3d0+lix])/bi;
        d_d[i3d0+2*lix+1] = d_l[li3d0+lix];
    } 
    
    __syncthreads();
}

