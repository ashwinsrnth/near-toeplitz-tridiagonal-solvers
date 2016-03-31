#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mkl.h>
#include <assert.h>


double getRandomDouble()
{
    return ((double)rand()/(double)RAND_MAX);
}

void initRand(double* ary, int size)
{
    /* Initialize ary with random doubles */
    for (int i=0; i<size; i++) {
        ary[i] = getRandomDouble();
    }
}

void initConst(double *ary, double a, int size)
{
    /* Initialize ary with the constant 'a' */
    for (int i=0; i<size; i++) {
        ary[i] = a;
    }
}


void printArray(double* ary, int size) 
{
    for (int i=0; i<size; i++) {
        printf("%f\n", ary[i]);
    }
    printf("\n");
}

void solve_tridiagonal_in_place_reusable(double x[], const size_t N, const double a[], const double b[], const double c[]) {
    size_t in;

    /* Allocate scratch space. */
    double* cprime = (double*)malloc(sizeof(double) * N);

    if (!cprime) {
        /* do something to handle error */
    }

    cprime[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    /* loop from 1 to N - 1 inclusive */
    for (in = 1; in < N; in++) {
        double m = 1.0 / (b[in] - a[in] * cprime[in - 1]);
        cprime[in] = c[in] * m;
        x[in] = (x[in] - a[in] * x[in - 1]) * m;
    }

    /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
    for (in = N - 1; in-- > 0; )
        x[in] = x[in] - cprime[in] * x[in + 1];

    /* free scratch space */
    free(cprime);
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    int N, nrhs, info; 
    double *a, *b, *c, *d, *d2, *x;
    float total_time;
    struct timeval start, end;
    int step, i, j, k;

    /* Get the size (command line argument) */
    if (argc < 3) {
        printf("Provide N, nrhs as command-line arguments!\n");
        return 1;
    }
    N = atoi(argv[1]);
    nrhs = atoi(argv[2]);

    /* Generate the three coefficient arrays and a RHS */ 
    a = (double*)malloc(N*sizeof(double));
    b = (double*)malloc(N*sizeof(double));
    c = (double*)malloc(N*sizeof(double));
    d = (double*)malloc(N*nrhs*sizeof(double));
    d2 = (double*)malloc(N*nrhs*sizeof(double));
    x = (double*)malloc(N*sizeof(double));

    initConst(a, 1./4, N);
    initConst(b, 1., N); 
    initConst(c, 1./4, N);
    initRand(d, N*nrhs);

    a[0] = 0;
    c[N-1] = 0;
    a[N-1] = 2;
    c[0] = 2;

    initRand(a, N);
    initRand(b, N);
    initRand(c, N);
    initRand(d, N*nrhs);
    
    for (step=0; step<100; step++) {    
        // d will be destroyed, so let's keep a copy of it 
        memcpy(d2, d, N*nrhs*sizeof(double)); 
        gettimeofday(&start, NULL);
        LAPACKE_dgtsv(LAPACK_COL_MAJOR, N, nrhs, a+1, b, c, d, N);
        //ddtsvb(&N, &nrhs, a+1, b, c, d, &N, &info);
        gettimeofday(&end, NULL);
        total_time += (float)(((end.tv_sec * 1000000 + end.tv_usec)
                          - (start.tv_sec * 1000000 + start.tv_usec)))/1000000.0;
    }

    printf("%f\n", total_time*1000/100);

    /* Check that the solution is correct 
    for (i=0; i<nz; i++) { 
        for (j=0; j<ny; j++) {
            memcpy(x, d2+(i*N*ny+j*N), N*nrhs*sizeof(double));
            solve_tridiagonal_in_place_reusable(x, N,
                    a, b, c);
            for (k=0; k<N; k++) {
                assert(x[k] == d[i*N*ny+j*N+k]);
            }
        }
    }
    */

    free(a);
    free(b);
    free(c);
    free(d);
    free(d2);
}
