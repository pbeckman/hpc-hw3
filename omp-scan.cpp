#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  int b = 0;
  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    b = n/nt;
    int t = omp_get_thread_num();
    prefix_sum[b*t] = A[b*t];
    for (long i = b*t+1; i < b*(t+1); i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i];
    }
  }
  
  long s = 0;
  for (int j = 1; j < n/b; j++) {
    s = prefix_sum[b*j-1];
    for (long i = 0; i < b; i++) {
      prefix_sum[b*j+i] += s;
    }
  }
}

int main() {
  long N = 1000000000;
  long* A  = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  int NREPEATS = 1;

  double tt = omp_get_wtime();
  for (int i = 0; i < NREPEATS; i++) scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", (omp_get_wtime() - tt) / NREPEATS);

  tt = omp_get_wtime();
  for (int i = 0; i < NREPEATS; i++)  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", (omp_get_wtime() - tt) / NREPEATS);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
