#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

const long BLOCK_SIZE = 1024; // block size for reduction sum

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

// kernel function for vectorized scalar product on GPU
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

// kernel function for computing sum in reduction
__global__ void 
reduction_sum(double * sum, double * a , long N, long offset){

    __shared__ double smem[BLOCK_SIZE];
    int idx = offset + (blockIdx.x) * blockDim.x + threadIdx.x;

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
	    sum[blockIdx.x+offset] = smem[threadIdx.x];
    }
}


int main(int argc, char** argv) {

  const int nStreams = 4;
  const long PFIRST = 4;
  const long PLAST = PFIRST+1;
  const long PINC = 4;

  for (long p = PFIRST; p < PLAST; p += PINC) {

    long streamSize = p * BLOCK_SIZE;
    long m = streamSize * nStreams, n = streamSize * nStreams;
    long streamBytes = streamSize * sizeof(double);

    printf("\nDimension %ld:\n", n);

    long NREPEATS = 1;//large dimension, only one repeat
    double *a, *b, *c;
    cudaMallocHost((void**)&a, m*n * sizeof(double));
    cudaMallocHost((void**)&b, n * sizeof(double));
    cudaMallocHost((void**)&c, m * sizeof(double));
    double* c_ref = (double*) malloc(m * sizeof(double));
    

    #pragma omp parallel for
    for (long i = 0; i < m*n; i++) {
      a[i] = 1;
    }
    
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
      b[i] = 1;
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

    // Compute on GPU (1 stream) with reduction sum
    dim3 GridDim(m/32, n/32, 1);
    dim3 BlockDim(32, 32, 1);
    tt = omp_get_wtime();

    for (long rep = 0; rep < NREPEATS; rep++) { 
      cudaMemcpyAsync(a_d, a, m*n*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
      // first compute vectorized product between each row of matrix A and vector b
      smult<<<GridDim,BlockDim>>>(m, n, a_d, b_d, a_d, 0);
      // compute reduction sum for each row of new matrix A
      for (long i=0; i<m; i++){
          long j = n;
          while (j>BLOCK_SIZE){ // recursively call the reduction sum (starting with the BLOCK_SIZE)
              reduction_sum<<<j/BLOCK_SIZE,BLOCK_SIZE>>>((a_d+i*n), (a_d+i*n), j, 0);
	          //cudaDeviceSynchronize();
	          j /= BLOCK_SIZE;
          }
          reduction_sum<<<1,j>>>((a_d+i*n), (a_d+i*n), j, 0);
          //cudaDeviceSynchronize();
          reduction_sum<<<1,1>>>((c_d+i), (a_d+i*n), 1, 0);
      }
      cudaMemcpyAsync(c, c_d, m*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }

    printf("GPU with reduction sum (1 stream) Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double err = 0;
    for (long i = 0; i < m; i++) {
      err = std::max(err, std::abs(c_ref[i] - c[i]));
    }
    printf("Max Error = %10e\n", err);    


    // Compute on GPU (multiple streams) with reduction sum
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
      cudaStreamCreate(&stream[i]);
    
    dim3 GridDim1(streamSize/32, streamSize/32, 1);
    dim3 BlockDim1(32, 32, 1);

    tt = omp_get_wtime();
    for (int i = 0; i < nStreams; ++i) { 
      int offset = i * streamSize;
      cudaMemcpyAsync(&a_d[offset*n], &a[offset*n],
                                streamBytes*n, cudaMemcpyHostToDevice,
                                stream[i]);
      cudaMemcpyAsync(&b_d[offset], &b[offset],
                                streamBytes, cudaMemcpyHostToDevice,
                                stream[i]);
      // first compute vectorized product between each row of matrix A and vector b
      smult<<<GridDim1,BlockDim1, 0, stream[i]>>>(m, n, a_d, b_d, a_d, offset);
      // compute reduction sum for each row of new matrix A
      for (long k=offset; k<offset+streamSize; k++){
          long j = n;
          while (j>BLOCK_SIZE){ // recursively call the reduction sum (starting with the BLOCK_SIZE)
              reduction_sum<<<j/BLOCK_SIZE,BLOCK_SIZE, 0, stream[i]>>>((a_d+k*n+offset*n), (a_d+k*n+offset*n), j, offset);
	          //cudaDeviceSynchronize();
	          j /= BLOCK_SIZE;
          }
          reduction_sum<<<1,j, 0, stream[i]>>>((a_d+k*n+offset*n), (a_d+k*n+offset*n), j, offset);
          //cudaDeviceSynchronize();
          reduction_sum<<<1,1, 0, stream[i]>>>((c_d+k+offset), (a_d+k*n+offset*n), 1, offset);
      }
      cudaMemcpyAsync(&c[offset], &c_d[offset],
                                streamBytes, cudaMemcpyDeviceToHost,
                                stream[i]);
    }
    cudaDeviceSynchronize();
    printf("GPU with reduction sum (%d streams) Bandwidth = %f GB/s\n", nStreams, (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

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

