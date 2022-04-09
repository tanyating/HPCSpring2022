#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>


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

// kernel function for inner product on GPU
__global__ 
void inn_prod(long m, long n, double *a, double *b, double *c, long offset){
  // A: input matrix of size m*n (row-first order)
  // b: input vector of size n*1
  // c: output vector of size m*1 (c = A*b)
  
  long idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    c[idx] = 0;
    for (long j=0; j < n; j++) {
      c[idx] = c[idx] + a[idx*n+j]*b[j];
    }
  }

}


int main(int argc, char** argv) {

  const int blockSize = 1024, nStreams = 4;

  const long PFIRST = 10;
  const long PLAST = 21;
  const long PINC = 2;

  for (long p = PFIRST; p < PLAST; p += PINC) {
    long streamSize = p * blockSize;
    long m = streamSize * nStreams, n = streamSize * nStreams;
    long streamBytes = streamSize * sizeof(double);

    printf("\nDimension %ld:\n", n);

    long NREPEATS = 1;//large dimension, only one repeat
    double *a, *b, *c;
    cudaMallocHost((void**)&a, m*n * sizeof(double));
    cudaMallocHost((void**)&b, n * sizeof(double));
    cudaMallocHost((void**)&c, m * sizeof(double));
    double* c_ref = (double*) malloc(m * sizeof(double));

    // Initialize matrix and vectors
    #pragma omp parallel for
    for (long i = 0; i < m*n; i++) {
      a[i] = 1e-5;
    }
    
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
      b[i] = 1e-5;
    }

    #pragma omp parallel for
    for (long i=0; i < m; i++) {
      c_ref[i] = 0.;
      c[i] = 0.;
    }


    double tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      VMult0(m, n, a, b, c_ref);
    }
    printf("CPU Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, m*n*sizeof(double));
    cudaMalloc(&b_d, n*sizeof(double));
    cudaMalloc(&c_d, m*sizeof(double));

    tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute on GPU (1 stream)
      cudaMemcpyAsync(a_d, a, m*n*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
      inn_prod<<<m/blockSize,blockSize>>>(m, n, a_d, b_d, c_d, 0);
      cudaMemcpyAsync(c, c_d, m*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }
    printf("GPU (1 stream) Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double err = 0;
    for (long i = 0; i < m; i++) {
      err = std::max(err, std::abs(c_ref[i] - c[i]));
      c[i] = 0; // reinitialize c for stream computation
    }
    printf("Max Error = %10e\n", err);

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
      cudaStreamCreate(&stream[i]);

    
    tt = omp_get_wtime();
    for (int i = 0; i < nStreams; ++i) { // Compute on GPU (multiple streams)
      int offset = i * streamSize;
      cudaMemcpyAsync(&a_d[offset*n], &a[offset*n],
                                streamBytes*n, cudaMemcpyHostToDevice,
                                stream[i]);
      cudaMemcpyAsync(&b_d[offset], &b[offset],
                                streamBytes, cudaMemcpyHostToDevice,
                                stream[i]);
      inn_prod<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(m, n, a_d, b_d, c_d, offset);
      cudaMemcpyAsync(&c[offset], &c_d[offset],
                                streamBytes, cudaMemcpyDeviceToHost,
                                stream[i]);
    }
    cudaDeviceSynchronize();
    printf("GPU (%d streams) Bandwidth = %f GB/s\n", nStreams, (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    err = 0;
    for (long i = 0; i < m; i++) err = std::max(err, std::abs(c_ref[i] - c[i]));
    printf("Max Error = %10e\n", err);


    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    free(c_ref);
  }

  return 0;
}

