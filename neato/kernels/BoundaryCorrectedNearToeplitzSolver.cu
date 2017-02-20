
__global__ void innerToeplitzCyclicReductionKernel( double *a_d,
                                double *b_d,
                                double *c_d,
                                double *d_d,
                                double *k1_d,
                                double *k2_d,
                                const double b1,                           
                                const double ai,
                                const double bi,
                                const double ci,
                                const double bn)
                                {
    /*

    */
    __shared__ double d_l[{{shared_size | int}}];

    int tix = threadIdx.x; 
    int offset = blockIdx.x*({{n}}+2)+1;
    int i, j, k;
    int idx;
    double d_j, d_k;

    /* When loading to shared memory, perform the first
       reduction step */
    idx = 0;
    if (tix == ({{(n/2) | int}}-1)) {
        d_l[tix] = d_d[offset+2*tix+1] - \
                    d_d[offset+2*tix]*k1_d[idx];
        d_d[offset+{{n}}] = d_d[offset+{{n}}]/bn;
    }
    else {
        d_l[tix] = d_d[offset+2*tix+1] - \
                    d_d[offset+2*tix]*k1_d[idx] - \
                    d_d[offset+2*tix+2]*k2_d[idx];
    }
    __syncthreads();
    
    /* First step of reduction is complete and 
       the coefficients are in shared memory */
    
    /* Do the remaining forward reduction steps: */
    for (int stride=2; stride<{{(n/2) | int}}; stride=stride*2) {
        idx = idx + 1;
        i = (stride-1) + tix*stride;
        if (tix < {{n}}/(2*stride)) {
            if (i == ({{n}}/2-1)) {
                d_l[i] = d_l[i] - \
                             d_l[i-stride/2]*k1_d[idx];
            }
            else {
                d_l[i] = d_l[i] - d_l[i-stride/2]*k1_d[idx] - \
                            d_l[i+stride/2]*k2_d[idx];
            }
        }
        __syncthreads();
    }

    if (tix == 0) {
        j = rint(log2((float) {{(n/2) | int}})) - 1;
        k = rint(log2((float) {{(n/2) | int}}));

        d_j = (d_l[{{n}}/4-1]*b_d[k] - \
               c_d[j]*d_l[{{n}}/2-1])/ \
            (b_d[j]*b_d[k] - c_d[j]*a_d[k]);

        d_k = (b_d[j]*d_l[{{n}}/2-1] - \
               d_l[{{n}}/4-1]*a_d[k])/ \
            (b_d[j]*b_d[k] - c_d[j]*a_d[k]);

        d_l[{{n}}/4-1] = d_j;
        d_l[{{n}}/2-1] = d_k;
    }
    __syncthreads();
    
    idx = rint(log2((float) {{n}}))-2;
    for (int stride={{n}}/4; stride>1; stride=stride/2) {
        idx = idx - 1;
        i = (stride/2-1) + tix*stride;
        if (tix < {{n}}/(2*stride)){
            if (tix == 0) {
                d_l[i] = (d_l[i] - c_d[idx]*d_l[i+stride/2])/\
                            b_d[idx];
            }
            else {
                d_l[i] = (d_l[i] - a_d[idx]*d_l[i-stride/2] -\
                            c_d[idx]*d_l[i+stride/2])/b_d[idx];
            }
        }
        __syncthreads();
    }

    //When writing from shared memory, perform the last
    //substitution step
    if (tix == 0) {
        d_d[offset+2*tix] = (d_d[offset+2*tix] - ci*d_l[tix])/bi;
        d_d[offset+2*tix+1] = d_l[tix];
        d_d[offset-1] = d_d[offset-1]/b1;
    }
    else {
        d_d[offset+2*tix] = (d_d[offset+2*tix] - \
                                ai*d_l[tix-1] - ci*d_l[tix])/bi;
        d_d[offset+2*tix+1] = d_l[tix];
    } 
    
    __syncthreads();
}


__global__ void boundaryCorrectionKernel(
                                double *d_d,
                                double *x_UH_i_d,
                                double *x_LH_i_d,
                                double *a_reduced_d,
                                double *b_reduced_d,
                                double *c_reduced_d,
                                double *c2_reduced_d,
                                const double x_LH_1,
                                const double x_UH_N) {

    int tix = blockIdx.x*blockDim.x + threadIdx.x;
    double bmac;

    /* first thread of the block makes and solves the block's reduced system */
    __shared__ double d_reduced_d[4];

    if (threadIdx.x == 0) {
        d_reduced_d[0] = -d_d[tix];
        d_reduced_d[1] = -d_d[tix+1];
        d_reduced_d[2] = -d_d[tix+({{n}})];
        d_reduced_d[3] = -d_d[tix+({{n}}+1)];

        /* each thread solves its reduced system */
        c2_reduced_d[0] = c_reduced_d[0]/b_reduced_d[0];
        d_reduced_d[0] = d_reduced_d[0]/b_reduced_d[0];

        for (int i=1; i<4; i++)
        {
            bmac = b_reduced_d[i] - a_reduced_d[i]*c2_reduced_d[i-1];
            c2_reduced_d[i] = c_reduced_d[i]/bmac;
            d_reduced_d[i] = (d_reduced_d[i] - a_reduced_d[i]*d_reduced_d[i-1])/bmac;
        }

        for (int i=2; i >= 0; i--)
        {
            d_reduced_d[i] = d_reduced_d[i] - c2_reduced_d[i]*d_reduced_d[i+1];
        }
    }

    __syncthreads();

    /* with the reduced solution, each thread computes the true solution */
    if (threadIdx.x == 0) {
        d_d[tix] = d_d[tix] + x_LH_1*d_reduced_d[0];
    }
    else if (threadIdx.x == ({{n}}+1)) {
        d_d[tix] = d_d[tix] + x_UH_N*d_reduced_d[3];
    }
    else {
        d_d[tix] = d_d[tix] + x_UH_i_d[threadIdx.x-1]*d_reduced_d[1] + x_LH_i_d[threadIdx.x-1]*d_reduced_d[2];
    }
}
