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
    */
    __shared__ double a_l[256];
    __shared__ double b_l[256];
    __shared__ double c_l[256];
    __shared__ double d_l[256];
    int ix = blockIdx.x*blockDim.x + threadIdx.x; 
    int iy = blockIdx.y*blockDim.y + threadIdx.y; 
    int iz = blockIdx.z*blockDim.z + threadIdx.z; 
    int lix = threadIdx.x; 
    int liy = threadIdx.y; 
    int liz = threadIdx.z; 
    int i, m, n;
    int stride;

    int i3d = iz*(nx*ny) + iy*nx + ix;
    int li3d = liz*(bx*by) + liy*bx + lix;
    int lix0 = liz*(bx*by) + liy*bx + 0;

    double k1, k2;
    double d_m, d_n;

    /* each block reads its portion to shared memory */
    a_l[li3d] = a_g[ix];
    b_l[li3d] = b_g[ix];
    c_l[li3d] = c_g[ix];
    d_l[li3d] = d_g[i3d];
    __syncthreads();

    /* solve the block in shared memory */
    stride = 1;
    for (int step=0; step<rint(log2((float) nx)); step++) {
        stride = stride*2;

        if (lix < nx/stride) {
            
            i = (stride-1) + lix*stride;
            ix = lix0 + i;

            if (stride == nx) {
                m = lix0 + nx/2 - 1;
                n = lix0 + nx - 1;

                d_m = (d_l[m]*b_l[n] - c_l[m]*d_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_n = (b_l[m]*d_l[n] - d_l[m]*a_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_l[m] = d_m;
                d_l[n] = d_n;
            }

            else {
                if (i == (nx-1)) {
                    ix = lix0 + i;
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
        __syncthreads();}

    
    for (int step=0; step<rint(log2((float) nx))-1; step++) {
        stride = stride/2;

        if (lix < nx/stride){
            i = (stride/2-1) + lix*stride;
            ix = lix0 + i;

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
    d_g[i3d] = d_l[li3d];
    __syncthreads();
}

