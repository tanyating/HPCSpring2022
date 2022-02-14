#include <stdio.h>
#include "utils.h"

const long ITER=100;

void GaussSeidel(long N, double* u, double* f, double eps) {
    // double* u;
    // param 2: u array
    // param 3: f array (data)
    // param 4: episilon (the factor of residual norms) -> when to stop iterations

    double h = 1.0/(N+1); // length of interval
    double err, err0;
    long k = 0; // iteration count
    
    while (k<ITER) { // terminates at certain number of iterations
    // while (k==0 || err/err0 >= eps) { // terminates when residual small
        // compute residuals
        err = 0.0;
        for (int i=0; i<N; i++) {
            double cur = f[i];
            if (i>0) cur -= -u[i-1]/pow(h,2);
            if (i<N-1) cur -= -u[i+1]/pow(h,2);
            cur -= 2.0*u[i]/pow(h,2);
            err += pow(cur,2); 
        }
        err = sqrt(err);
        // printf("At iteration %ld, the residual norm is %.10f\n", k,err);
        if (k==0) err0 = err; // initial residual norm

        // update u_{k+1}
        for (int i=0; i<N; i++) {
           double cur = f[i];
           if (i>0) cur += u[i-1]/pow(h,2);
           if (i<N-1) cur += u[i+1]/pow(h,2);
           u[i] = cur/2.0*pow(h,2);
        }

        k++;
    }

    printf("The number of iterations for Gauss Seidel is %ld.\n", k);


}

void Jacobi(long N, double* u, double* f, double eps) {
    // param 1: size N
    // param 2: u array
    // param 3: f array (data)
    // param 4: episilon (the factor of residual norms) -> when to stop iterations

    double h = 1.0/(N+1); // length of interval
    double* res; // keep track of residuals
    res = (double*) malloc(N * sizeof(double));
    double err0, err;
    long k = 0; // iteration count
    
    while (k<ITER) { // terminates at certain number of iterations
    // while (k==0 || err/err0 >= eps) { // terminates when residual small
        err = 0.0;
        // update u_{k+1}
        for (int i=0; i<N; i++) {
           double cur = f[i]; 
           if (i>0) cur -= -u[i-1]/pow(h,2);
           if (i<N-1) cur -= -u[i+1]/pow(h,2);
           res[i] = cur;
           cur -= 2.0*u[i]/pow(h,2); // complete the actual residual component
           err += pow(cur,2); 
        }
        for (int j=0; j<N; j++) u[j] = res[j]/2.0*pow(h,2); // update u^k
        err = sqrt(err);
        // printf("At iteration %ld, the residual norm is %.10f\n", k,err);
        if (k==0) err0 = err; // initial residual norm
        k++;
    }

    printf("The number of iterations for Jacobi is %ld.\n", k);

    free(res);

}

int main() {

    double* u1;
    double* u2;
    double* f;
    Timer t;
    double time;
    long N = 10000; //10000
    u1 = (double*) malloc(N * sizeof(double));
    u2 = (double*) malloc(N * sizeof(double));
    f = (double*) malloc(N * sizeof(double));
    for (int i=0; i<N; i++) {
        u1[i]=0.0; // u_0 = 0
        u2[i]=0.0; // u_0 = 0
        f[i]=1.0; // f(i) = 1
    }

    double eps = 1e-6;
    
    // printf("Use a decay factor of %.10f to decide when to terminate the iterations.\n\n",eps);

    printf("Jacobi Method for N=%ld:\n", N);
    t.tic();
    Jacobi(N,u1,f,eps);
    // Change optimization flags and print the runtimes
    printf("The run time of Jacobi for 100 iterations using ompiler optimization flags -O0 is %.10f s.\n", t.toc());


    printf("\n\nGauss-Sidel Method for N=%ld:\n", N);
    t.tic();
    GaussSeidel(N,u2,f,eps);
    // Change optimization flags and print the runtimes
    printf("The run time of Gauss Seidel for 100 iterations using ompiler optimization flags -O0 is %.10f s.\n", t.toc());

    free(u1);
    free(u2);
    free(f);
    return 0;
}