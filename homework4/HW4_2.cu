#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

// referenced jacobi method computed on CPU (with OMP)
void jacobi2d(long N, double hsqr, double *u, double *unew, double *f) {

    #pragma omp parallel
    #pragma omp for collapse(2)
    // Jacobi iteration to update all nodes row-wise
    for (long i=1; i <= N; i++) {
        for (long j=1; j <= N; j++) {
            unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)])/4;
        }
    }

}

//kernel function for Jacobi method on GPU
__global__
void jacobi_kernel(long N, double hsqr, double *u, double *unew, double *f, long offset) {

    long i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    long j = offset + blockIdx.y * blockDim.y + threadIdx.y;
    //printf("\nkernel hsqr:%e",hsqr);
    if (i>=1 && i<=N && j >=1 && j>=N){
        unew[i*(N+2)+j] = (hsqr*f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)])/4;
    }

}



int main() {

    const int blockSizeX = 32, blockSizeY = 32;
    long N = 30 * blockSizeX - 2;
    dim3 GridDim((N+2)/blockSizeX, (N+2)/blockSizeY, 1);
    dim3 BlockDim(blockSizeX, blockSizeY, 1);

    double h = 1.0/(N+1);
    double hsqr = h*h;

    double *u, *unew, *f;
    cudaMallocHost((void**)&u, (N+2)*(N+2) * sizeof(double));
    cudaMallocHost((void**)&unew, (N+2)*(N+2) * sizeof(double));
    cudaMallocHost((void**)&f, (N+2)*(N+2) * sizeof(double));
    double* u_ref = (double *) malloc((N+2)*(N+2) * sizeof(double));

    // initialize boundary conditions and right hand side (f)
    for (long i=0; i<N; i++) { 
        u[i*(N+2)+0] = u[i*(N+2)+N+1] = u[0*(N+2)+i] = u[(N+1)*(N+2)+i] = 0.0;
        u_ref[i*(N+2)+0] = u_ref[i*(N+2)+N+1] = u_ref[0*(N+2)+i] = u_ref[(N+1)*(N+2)+i] = 0.0;
	    unew[i*(N+2)+0] = unew[i*(N+2)+N+1] = unew[0*(N+2)+i] = unew[(N+1)*(N+2)+i] = 0.0;
        for (long j=1; j<=N; j++) {
            f[(i+1)*(N+2)+j] = 1.0;
        }
    }

    printf("\nJacobi Method for (dimension) N=%ld (first %ld iterations):\n", N, ITER);

    //Jacobi on CPU (reference)
    double tt = omp_get_wtime();
    for (long k=0; k<ITER; k++) { // stop when reached max steps

        jacobi2d(N, hsqr, u_ref, unew, f);

        // flip pointer to update
        double* utmp = u_ref;
        u_ref = unew;
        unew = utmp;

    }
    printf("CPU time elapsed is %f.\n", omp_get_wtime() - tt);

    double *u_d, *unew_d, *f_d;
    cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&unew_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(double));
    
    //Jacobi on GPU
    tt = omp_get_wtime();
    for (long k=0; k<ITER; k++) { // stop when reached max steps

        cudaMemcpyAsync(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
        jacobi_kernel<<<GridDim,BlockDim>>>(N, hsqr, u_d, unew_d, f_d, 0);
        cudaMemcpyAsync(unew, unew_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // flip pointer to update
        double* utmp = u;
        u = unew;
        unew = utmp;

    }
    printf("GPU time elapsed is %f.\n", omp_get_wtime() - tt);

    double err = 0;
    for (long i = 1; i <= N; i++){
        for (long j = 1; j <= N; j++) {
            err = std::max(err, std::abs(u_ref[i*(N+2)+j] - u[i*(N+2)+j]));
        }
    }
    printf("Max Difference (compared with CPU) = %10e\n", err);

    cudaFree(u_d);
    cudaFree(unew_d);
    cudaFree(f_d);

    cudaFreeHost(u);
    cudaFreeHost(unew);
    cudaFreeHost(f);
    free(u_ref);
    
    
    
}
