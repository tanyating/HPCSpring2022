#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
long p = 8; // number of threads

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
  long m = (long)(ceil(double(n)/p)); // chunk size (static per thread)

  #pragma omp parallel num_threads(p)
{
  
  #pragma omp for schedule(static)
  for (long j=0; j<p; j++) { // parallelize each of p chunks
//    #pragma omp task 
{
    prefix_sum[j*m] = 0;
    for (long k=j*m+1; k<(j+1)*m && k<n; k++) {
      prefix_sum[k] = prefix_sum[k-1] + A[k-1];
    }
}
  }
  #pragma omp single // add up partial sums in serial (one thread)
{
  long s1 = 0;
  for (long j = 1; j < p; j++) {
    s1 += prefix_sum[j*m-1] + A[j*m-1];
    prefix_sum[j*m] += s1; // sum of partial sums
  }
}
  #pragma omp barrier // wait till the partial sums (correction terms) are updated in serial
  
  #pragma omp for schedule(static)
  for (long j=1; j<p; j++) { // parallelize each of p chunks (except for the 1st chunk)
//    #pragma omp task 
{
    long s2 = prefix_sum[j*m];
    for (long k=j*m+1; k<(j+1)*m && k<n; k++) {
      prefix_sum[k] = prefix_sum[k] + s2; // add back the correction term
    }
}
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
  long est = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
