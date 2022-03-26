#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan

  if (n == 0) return;
  int p = 8; // number of threads
  long m = ceil(n/p); // chunk size (static per thread)

  #pragma parallel omp num_threads(p)
  //#pragma omp for schedule(static)
  for (int j=0; j<p; j++) { // parallelize each of p chunks
    #pragma omp task 
{
    prefix_sum[j*m] = 0;
    for (long k=j*m+1; k<(j+1)*m && k<n; k++) {
      prefix_sum[k] = prefix_sum[k-1] + A[k-1];
    }
}
  }

  // serial correction
  for (int j = 1; j < p; j++) {
    double s = prefix_sum[j*m-1] + A[j*m-1];
    for (long k=j*m; k<(j+1)*m && k<n; k++) {
      prefix_sum[k] = prefix_sum[k] + s;
    }
  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
