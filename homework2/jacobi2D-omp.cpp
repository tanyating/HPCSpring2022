#include <stdio.h>
// #include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

const long ITER=500;

/* compute global residual, assuming ghost values are updated */
double compute_residual(long N, double *u, double* f, double invhsqr) {
    // param 1: size N
    // param 2: u array
    // param 3: f array (data)
    // param 5: square of interval length h

    double tmp, res = 0.0;

    for (long i = 1; i <= N; i++){
        for (long j = 1; j <= N; j++) {
            tmp = (4*u[i*(N+2)+j] - u[(i-1)*(N+2)+j] - u[i*(N+2)+(j-1)] - u[(i+1)*(N+2)+j] - u[i*(N+2)+(j+1)]) * invhsqr - f[i*(N+2)+j];
            res += tmp * tmp;
        }
    }
    return sqrt(res);
}

int main() {

    long N = 10;
    double h = 1.0/(N+1);
    double hsqr = h*h;
    double invhsqr = 1/hsqr;
    double res, res0, tol = 1e-10;

    double* u = (double *) calloc(sizeof(double), (N+2)*(N+2));
    double* unew = (double *) calloc(sizeof(double), (N+2)*(N+2));
    double* f = (double *) calloc(sizeof(double), (N+2)*(N+2));

    // initialize boundary conditions
    for (long i=0; i<N; i++) { 
        u[i*(N+2)+0] = u[i*(N+2)+N+1] = u[0*(N+2)+i] = u[(N+1)*(N+2)+i] = 0.0;
        for (long j=1; j<=N; j++) {
            f[(i+1)*(N+2)+j] = 1.0;
        }
    }

    /* initial residual */
    res0 = compute_residual(N, u, f, invhsqr);
    res = res0;
    printf("Initial Residual: %g\n", res0);

    for (long k=0; k<ITER && res/res0 > tol; k++) { // stop when reached max steps or residual decays enough

        // Jacobi iteration to update nodes row-wise
        for (long i=1; i <= N; i++) {
            for (long j=1; j <= N; j++) {
                unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)])/4;
            }
        }

        // flip pointer to update
        double* utmp = u;
        u = unew;
        unew = utmp;

        // print out residuals every 100 iterations
        if ((k % 100) == 0) {
            res = compute_residual(N, u, f, invhsqr);
            printf("Iter %ld: Residual: %g\n", k, res);
        }
    
    }

    // for (long i=1; i <= N; i++) {
    //         for (long j=1; j<=N; j++) {
    //             printf("%.6f\t", u[i*(N+2)+j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");

    free(u);
    free(unew);
    free(f);
    
    
    
}