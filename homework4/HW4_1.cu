// g++ -std=c++11 -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>


// Check errors
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

void VMult0(long m, long n, double *a, double *b, double *c) {
    // A: input matrix of size m*n (row-first order)
    // b: input vector of size n*1
    // c: output vector of size m*1 (c = A*b)
    #pragma omp parallel
    #pragma omp for
    for (long i = 0; i < m; i++) { // parallell each row (inner-prod) with OMP
      for (long j=0; j < n; j++) {
        double A_ij = a[i*n+j];
        double b_j = b[j];
        double c_j = c[j];
        c_j = c_j + A_ij * b_j;
        c[j] = c_j;
      }
    }
}

// vector-vector inner product on GPU
__global__ 
void inn_prod(long m, long n, double *a, double *b, double *c, long offset){
  // A: input matrix of size m*n (row-first order)
  // b: input vector of size n*1
  // c: output vector of size m*1 (c = A*b)
  
  long idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    for (long j=0; j < n; j++) {
      c[idx] = c[idx] + a[idx*n+j]*b[j];
    }
  }

}


int main(int argc, char** argv) {

  const int blockSize = 1024, nStreams = 4;

  const long PFIRST = 1000;
  const long PLAST = 100000;
  const long PINC = 4*blockSize; // multiple of BLOCK_SIZE

  for (long p = PFIRST; p < PLAST; p += PINC) {
    long streamSize = p * blockSize;
    long m = streamSize * nStreams, n = streamSize * nStreams;
    long streamBytes = streamSize * sizeof(double);

    printf(" Dimension %ld:\n", n);

    long NREPEATS = 1e9/(m*n)+1;
    double *a, *b, *c;
    cudaMallocHost((void**)&a, m*n * sizeof(double));
    cudaMallocHost((void**)&b, n * sizeof(double));
    cudaMallocHost((void**)&c, n * sizeof(double));
    double* c_ref = (double*) malloc(n * sizeof(double));

    // Initialize matrices
    for (long i = 0; i < m*n; i++) {
      a[i] = drand48();
    }
    
    for (long i = 0; i < n; i++) {
      b[i] = drand48();
      c_ref[i] = 0;
      c[i] = 0;
    }

    double tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      VMult0(m, n, a, b, c_ref);
    }
    printf("CPU Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, m*n*sizeof(double));
    cudaMalloc(&b_d, n*sizeof(double));
    cudaMalloc(&c_d, n*sizeof(double));

    tt = omp_get_wtime();
    for (long rep = 0; rep < NREPEATS; rep++) {
      cudaMemcpyAsync(a_d, a, m*n*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
      inn_prod<<<m/blockSize,blockSize>>>(m, n, a_d, b_d, c_d, 0);
      cudaMemcpyAsync(c, c_d, n*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }
    printf("GPU (no stream) Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double max_err = 0;
    for (long i = 0; i < n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf("Error = %10e\n", max_err);

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
      cudaStreamCreate(&stream[i]);

    tt = omp_get_wtime();
    for (int i = 0; i < nStreams; ++i) {
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
    printf("GPU (4 streams) Bandwidth = %f GB/s\n", (2*m+2*m*n)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double max_err = 0;
    for (long i = 0; i < n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf("Error = %10e\n", max_err);

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

