#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <omp.h>
#ifdef _OPENMP_
#include<omp.h> 
#endif

const long ITER=1000;

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

    long N = 999;
    double h = 1.0/(N+1);
    double hsqr = h*h;
    double invhsqr = 1/hsqr;
    double res, res0, tol = 1e-10;
    int ntr = 16;

    double* u = (double *) calloc(sizeof(double), (N+2)*(N+2));
    double* unew = (double *) calloc(sizeof(double), (N+2)*(N+2));
    double* f = (double *) calloc(sizeof(double), (N+2)*(N+2));

    // initialize boundary condition and right hand side (f)
    for (long i=0; i<N; i++) {
        u[i*(N+2)+0] = u[i*(N+2)+N+1] = u[0*(N+2)+i] = u[(N+1)*(N+2)+i] = 0.0;
        unew[i*(N+2)+0] = unew[i*(N+2)+N+1] = unew[0*(N+2)+i] = unew[(N+1)*(N+2)+i] = 0.0;
        for (long j=1; j<=N; j++) {
            f[(i+1)*(N+2)+j] = 1.0;
        }
    }

    printf("Gauss Seidel Method for N=%ld (%d threads):\n", N, ntr);
    
    /* initial residual */
    res0 = compute_residual(N, u, f, invhsqr);
    res = res0;
    printf("Initial Residual: %g\n", res0);

    /* timing */
    //double t = omp_get_wtime();
    Timer t;
    t.tic();

    for (long k=0; k<ITER && res/res0 > tol; k++) { // stop when reached max steps or residual decays enough

        #pragma omp parallel num_threads(ntr)
{ 
	// GS iteration to update red points
	#pragma omp for collapse(2) nowait
	for (long i=1; i <= N; i+=2) { // iterate through every odd row of red points (starting at 1)
            for (long j=1; j <= N; j+=2) { // iterate through every other column of red points at each row
                unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)])/4;
            }
            
        }

        #pragma omp for collapse(2)
	for (long i=2; i <= N; i+=2) { // iterate through every even row of red points (starting at 2)
            for (long j=2; j <= N; j+=2) {
                unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)])/4;
            }
            
        }
	
	// GS iteration to update black points
 	#pragma omp for collapse(2) nowait
	for (long i=1; i <= N; i+=2) { // iterate through every odd row of black points (starting at 2)
	    for (long j=2; j <= N; j+=2) { // iterate through every other column of black points at each row
                unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + unew[(i-1)*(N+2)+j] + unew[i*(N+2)+(j-1)] + unew[(i+1)*(N+2)+j] + unew[i*(N+2)+(j+1)])/4;
            }
        }

        #pragma omp for collapse(2)
	for (long i=2; i <= N; i+=2) { // iterate through every even row of black points (starting at 1)
            for (long j=1; j <= N; j+=2) {
                unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + unew[(i-1)*(N+2)+j] + unew[i*(N+2)+(j-1)] + unew[(i+1)*(N+2)+j] + unew[i*(N+2)+(j+1)])/4;
            }
        }
}
        double* utmp = u;
        u = unew;
        unew = utmp;

        // print out residuals every 100 iterations
        if ((k % 100) == 0) {
            res = compute_residual(N, u, f, invhsqr);
            printf("Iter %ld: Residual: %g\n", k, res);
        }
    
    }

    /* timing */
    //t = omp_get_wtime() - t;
    printf("Time elapsed is %f.\n", t.toc());

    free(u);
    free(unew);
    free(f);
    
    
    
}
