__global__ void sharedMemCyclicReduction( double *a_g,
                                double *b_g,
                                double *c_g,
                                double *d_g,
                               int nx,
                               int ny,
                               int nz,
                               int bx,
                               int by) {
    /*
        Solve several systems by cyclic reduction,
        each of size block_size.
        
        bx and by are the block size.
        Specifically, they do not refer to the actual
        number of threads per block.
        bx = nx
        but nx/2 threads per block are launched.
    */
    __shared__ double a_l[1024];
    __shared__ double b_l[1024];
    __shared__ double c_l[1024];
    __shared__ double d_l[1024];
    int ix = blockIdx.x*blockDim.x + threadIdx.x; 
    int iy = blockIdx.y*blockDim.y + threadIdx.y; 
    int iz = blockIdx.z*blockDim.z + threadIdx.z; 
    int lix = threadIdx.x; 
    int liy = threadIdx.y; 
    int liz = threadIdx.z; 
    int i, m, n;
    int stride;

    int i3d = iz*(nx*ny) + iy*nx + ix;
    int i3d0 = iz*(nx*ny) + iy*nx + 0;
    int li3d = liz*(bx*by) + liy*bx + lix;
    int li3d0 = liz*(bx*by) + liy*bx + 0;

    double k1, k2;
    double d_m, d_n;

    /* each block reads two elements to shared memory */
    a_l[li3d0+2*lix]  = a_g[2*lix];
    a_l[li3d0+2*lix+1] = a_g[2*lix+1];
    b_l[li3d0+2*lix]  = b_g[2*lix];
    b_l[li3d0+2*lix+1] = b_g[2*lix+1];
    c_l[li3d0+2*lix]  = c_g[2*lix];
    c_l[li3d0+2*lix+1] = c_g[2*lix+1];
    d_l[li3d0+2*lix] = d_g[i3d0+2*lix];
    d_l[li3d0+2*lix+1] = d_g[i3d0+2*lix+1];
    __syncthreads();
    
    /* solve the block in shared memory */
    stride = 1;
    for (int step=0; step<rint(log2((float) nx)); step++) {
        stride = stride*2;

        if (lix < nx/stride) {
            i = (stride-1) + lix*stride;
            ix = li3d0 + i;

            if (stride == nx) {
                m = li3d0 + nx/2 - 1;
                n = li3d0 + nx - 1;

                d_m = (d_l[m]*b_l[n] - c_l[m]*d_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_n = (b_l[m]*d_l[n] - d_l[m]*a_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_l[m] = d_m;
                d_l[n] = d_n;
            }

            else {
                if (i == (nx-1)) {
                    ix = li3d0 + i;
                    k1 = a_l[ix]/b_l[ix-stride/2];
                    a_l[ix] = -a_l[ix-stride/2]*k1;
                    b_l[ix] = b_l[ix] - c_l[ix-stride/2]*k1;
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1;
                }
                else {
                    k1 = a_l[ix]/b_l[ix-stride/2];
                    k2 = c_l[ix]/b_l[ix+stride/2];
                    a_l[ix] = -a_l[ix-stride/2]*k1;
                    b_l[ix] = b_l[ix] - c_l[ix-stride/2]*k1 - a_l[ix+stride/2]*k2;
                    c_l[ix] = -c_l[ix+stride/2]*k2;
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1 - d_l[ix+stride/2]*k2;
                }
            }
        }
        __syncthreads();
    }


    
    for (int step=0; step<rint(log2((float) nx))-1; step++) {
        stride = stride/2;

        if (lix < nx/stride){
            i = (stride/2-1) + lix*stride;
            ix = li3d0 + i;

            if (i < stride) {
                d_l[ix] = (d_l[ix] - c_l[ix]*d_l[ix+stride/2])/b_l[ix];
            }

            else {
                d_l[ix] = (d_l[ix] - a_l[ix]*d_l[ix-stride/2] - c_l[ix]*d_l[ix+stride/2])/b_l[ix];
            }
        }

        __syncthreads();
    }
    
    /* write from shared memory to x_d */
    
    d_g[i3d0+2*lix] = d_l[li3d0+2*lix];
    d_g[i3d0+2*lix+1] = d_l[li3d0+2*lix+1];
    __syncthreads();
}

