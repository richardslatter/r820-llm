#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
// read-only bandwidth: each thread sums its own slice with 256-bit loads
int main(int argc,char**argv){
  size_t gb = argc>1?atol(argv[1]):8; size_t n = gb<<30; size_t nf=n/4;
  float *a = aligned_alloc(1<<21,n); if (getenv("NOHUGE")) madvise(a, n, MADV_NOHUGEPAGE);
  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<nf;i++) a[i]=1.0f;
  double best=0;
  for(int r=0;r<5;r++){
    double t=omp_get_wtime(); float tot=0;
    #pragma omp parallel reduction(+:tot)
    { __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;
      #pragma omp for schedule(static)
      for(size_t i=0;i<nf;i+=32){ s0=_mm256_add_ps(s0,_mm256_load_ps(a+i)); s1=_mm256_add_ps(s1,_mm256_load_ps(a+i+8)); s2=_mm256_add_ps(s2,_mm256_load_ps(a+i+16)); s3=_mm256_add_ps(s3,_mm256_load_ps(a+i+24)); }
      float o[8]; _mm256_storeu_ps(o,_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3))); for(int k=0;k<8;k++) tot+=o[k]; }
    t=omp_get_wtime()-t; double bw=n/t/1e9; if(bw>best)best=bw; if(tot<0)puts("x");
  }
  printf("read %.1f GB/s\n",best); return 0; }
