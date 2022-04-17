#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

const long BLOCK_SIZE = 32;
// Check errors
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

// reference Matrix-vector multiplication on CPU (with OMP)
void VMult0(long m, long n, double *a, double *b, double *c) {
    // A: input matrix of size m*n (row-first order)
    // b: input vector of size n*1
    // c: output vector of size m*1 (c = A*b)
    #pragma omp parallel for
    for (long i = 0; i < m; i++) { // parallell each row (inner-prod) with OMP
      for (long j=0; j < n; j++) {
	      c[i] = c[i] + a[i*n+j]*b[j];
      }
    }
}

// kernel function for scalar product on GPU
__global__ 
void smult(long m, long n, double *a, double *b, double *c, long offset){
  // A: input matrix of size m*n (row-first order)
  // b: input vector of size n*1
  // c: output vector of size m*n (c[i,j] = A[i,j]*b[j])
  
  long i = offset + blockIdx.x * blockDim.x + threadIdx.x;
  long j = offset + blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    c[i*n+j] = a[i*n+j]*b[j];
  }

}

__global__ void 
reduction_sum(double * sum, double * a , long N){
    //const long BLOCK_SIZE = N; 
    __shared__ double smem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    // each thread reads data from global into shared memory
    if (idx < N) smem[threadIdx.x] = a[idx];
    else smem[threadIdx.x] = 0;
    __syncthreads();

    // x > >= 1 means " set x to itself shifted by one bit to the right "
    // means divide x by 2 in each iteration
    for (unsigned int s = blockDim.x/2; s > 0; s>>=1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    // write to global memory
    if (threadIdx.x == 0){ 
	sum[blockIdx.x] = smem[threadIdx.x];
	//printf("\nidx: %d, sum: %e\n",blockIdx.x,sum[blockIdx.x]);
}
}


int main(int argc, char** argv) {


  const long PFIRST = 10;
  const long PLAST = PFIRST+1;
  const long PINC = 16;

  for (long p = PFIRST; p < PLAST; p += PINC) {
    
    const int blockSizeX = 32, blockSizeY = 32;
    long m = p * blockSizeX, n = p * blockSizeY;
    dim3 GridDim(p, p, 1);
    dim3 BlockDim(blockSizeX, blockSizeY, 1);

    printf("\nDimension %ld:\n", n);

    long NREPEATS = 1;//large dimension, only one repeat
    double *a, *b, *c;//, *tmp;
    cudaMallocHost((void**)&a, m*n * sizeof(double));
    // cudaMallocHost((void**)&tmp, m*n * sizeof(double));
    cudaMallocHost((void**)&b, n * sizeof(double));
    cudaMallocHost((void**)&c, m*n * sizeof(double));
    double* c_ref = (double*) malloc(m * sizeof(double));

    // Initialize matrix and vectors
    #pragma omp parallel for
    for (long i = 0; i < m*n; i++) {
      a[i] = 1;
      c[i] = 0.;
    }
    
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
      b[i] = 2;
    }

    #pragma omp parallel for
    for (long i=0; i < m; i++) {
      c_ref[i] = 0.;
    //   c[i] = 0.;
    }


    double tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      VMult0(m, n, a, b, c_ref);
    }
    printf("CPU Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *a_d, *b_d, *c_d;//, *tmp_d;
    cudaMalloc(&a_d, m*n*sizeof(double));
    // cudaMalloc(&tmp_d, m*n*sizeof(double));
    cudaMalloc(&b_d, n*sizeof(double));
    cudaMalloc(&c_d, m*n*sizeof(double));

    tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute on GPU (1 stream)
      cudaMemcpyAsync(a_d, a, m*n*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
      smult<<<GridDim,BlockDim>>>(m, n, a_d, b_d, c_d, 0);
    //   cudaMemcpyAsync(tmp, tmp_d, m*n*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      for (long i=0; i<m; i++){
          for (long j=n; j>0; j /= BLOCK_SIZE)
          reduction_sum<<<j/BLOCK_SIZE,BLOCK_SIZE>>>((c_d+i), (c_d+i*n), j);
      }
    //  reduction_sum<<<m/BLOCK_SIZE,BLOCK_SIZE>>>(c_d, a_d, n);
      //reduction_sum<<<1,n>>>(c_d, b_d, n);
      //printf("test sum: %e\n",c_d[0]);
      cudaMemcpyAsync(c, c_d, m*sizeof(double), cudaMemcpyDeviceToHost);
      //printf("test sum: %e\n",c[0]);
      cudaDeviceSynchronize();
      //printf("test sum: %e\n",c[0]);
    }
    printf("GPU (1 stream) Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double err = 0;
    for (long i = 0; i < m; i++) {
      err = std::max(err, std::abs(c_ref[i] - c[i]));
      //c[i] = 0; // reinitialize c for stream computation
    }
    printf("Max Error = %10e\n", err);
/*
    printf("\nIntermidate matrix (tmp):\n");
    for (long i=0; i < m; i++) {
        for (long j=0; j < n; j++) {
            printf("%e\t", tmp[i*n+j]);
        }
        printf("\n");
    }

    printf("\nresulting c:\n");
    for (long j=0; j < n; j++) {
        printf("%e\t", c[j]);
    }

    printf("\nresulting cref:\n");
    for (long j=0; j < n; j++) {
        printf("%e\t", c_ref[j]);
    }
  */  


    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(tmp_d);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(tmp);
    free(c_ref);
  }

  return 0;
}

