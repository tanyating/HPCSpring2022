// g++ -std=c++11 -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 24

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  #pragma omp parallel // parallelized version using OpenMP (also in column-first order) 
  #pragma omp for collapse(2)
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i++) {
      for (long p = 0; p < k; p++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
  
  // number of blocks in each dimension
  long N1 = (long)(m/BLOCK_SIZE);
  long N2 = (long)(n/BLOCK_SIZE);
  long N3 = (long)(k/BLOCK_SIZE);
  
  // iterate over blocks and parallelize using OpenMP
  #pragma omp parallel 
  #pragma omp for collapse(2)
  for (long J = 0; J < N2; J++) {
    for (long I = 0; I < N1; I++) {
	for (long P = 0; P < N3; P++) {

	// block-block multiplication
	for (long j = 0; j < BLOCK_SIZE; j++) {
          for (long p = 0; p < BLOCK_SIZE; p++) {
            for (long i = 0; i < BLOCK_SIZE; i++) {
              double A_ip = a[I*BLOCK_SIZE+P*BLOCK_SIZE*m+i+p*m];
              double B_pj = b[P*BLOCK_SIZE+J*BLOCK_SIZE*k+p+j*k];
              double C_ij = c[I*BLOCK_SIZE+J*BLOCK_SIZE*m+i+j*m];
              C_ij = C_ij + A_ip * B_pj;
              c[I*BLOCK_SIZE+J*BLOCK_SIZE*m+i+j*m] = C_ij;
            }
          }
        }

      }
    }
  }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = 1;
    for (long i = 0; i < k*n; i++) b[i] = 1;
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
	MMult1(m, n, k, a, b, c);
    }

    double time = t.toc();
    double flops = (2*m*n*k)*NREPEATS/1e9/time; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = (2*m*n+2*m*n*k)*sizeof(double)*NREPEATS/1e9/time; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
