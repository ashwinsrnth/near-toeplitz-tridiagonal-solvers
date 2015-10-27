__global__ void globalForwardReduction(const double *a_d,
                                    const double *b_d,
                                    const double *c_d,
                                    double *d_d,
                                    const double *k1_d,
                                    const double *k2_d,
                                    const double *b_first_d,
                                    const double *k1_first_d,
                                    const double *k1_last_d,
                                    const int nx,
                                    const int ny,
                                    const int nz,
                                    int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
    int i, idx, i0;
    int m, n;
    double x_m, x_n;

    i0 = giz*(nx*ny) + giy*nx + 0;

    // forward reduction
    if (stride == nx)
    {
        stride /= 2;

        m = log2((float)stride) - 1;
        n = log2((float)stride); // the last element

        x_m = (d_d[i0 + stride-1]*b_d[n] - c_d[m]*d_d[i0 + 2*stride-1])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        x_n = (b_first_d[m]*d_d[i0 + 2*stride-1] - d_d[i0 + stride-1]*a_d[n])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        d_d[i0 + stride-1] = x_m;
        d_d[i0 + 2*stride-1] = x_n;
    }
    else
    {
        i = i0 + (stride-1) + gix*stride;;

        idx = log2((float)stride) - 1;
        if (gix == 0)
        {
            d_d[i] = d_d[i] - d_d[i - stride/2]*k1_first_d[idx] - d_d[i + stride/2]*k2_d[idx];
        }
        else if (i == (nx-1))
        {
            d_d[i] = d_d[i] - d_d[i - stride/2]*k1_last_d[idx];
        }
        else
        {
            d_d[i] = d_d[i] - d_d[i - stride/2]*k1_d[idx] - d_d[i + stride/2]*k2_d[idx];
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
                                    const int nx,
                                    const int ny,
                                    const int nz,
                                    const int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
    int i, idx, i0;

    i0 = giz*(nx*ny) + giy*nx + 0;
    i = i0 + (stride/2-1) + gix*stride;

    if (stride == 2)
    {
        if (i == 0)
        {
            d_d[i] = (d_d[i] - c1*d_d[i + 1])/b1;
        }
        else
        {
            d_d[i] = (d_d[i] - (ai)*d_d[i - 1] - (ci)*d_d[i + 1])/bi;
        }
    }
    else
    {
        // rint rounds to the nearest integer
        idx = rint(log2((double)stride)) - 2;
        if (gix == 0) 
        {   
            d_d[i] = (d_d[i] - c_d[idx]*d_d[i + stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[i] = (d_d[i] - a_d[idx]*d_d[i - stride/2] - c_d[idx]*d_d[i + stride/2])/b_d[idx];
        }
    }
}
