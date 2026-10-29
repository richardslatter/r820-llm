// Read bandwidth over a tmpfs file mapping, after moving it to one node and optionally splitting
// its huge pages:   rdfile <file> <GB> <node> <split_order|-1>
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
int main(int argc, char **argv) {
    size_t n = (size_t)atol(argv[2]) << 30; int node = atoi(argv[3]); int order = atoi(argv[4]);
    int fd = open(argv[1], O_RDWR | O_CREAT, 0644); ftruncate(fd, n);
    float *a = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    unsigned long mask[2] = { 1ul << node, 0 };
    syscall(SYS_mbind, a, n, 2, mask, 65, 2);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n / 4; i += 1024) a[i] = 1.0f;          // allocate pages on the bound node
    if (order >= 0) {
        char cmd[512]; snprintf(cmd, sizeof cmd, "echo %s,0x0,0x%zx,%d | sudo tee /sys/kernel/debug/split_huge_pages >/dev/null", argv[1], n / 4096, order);
        if (system(cmd)) fprintf(stderr, "split failed\n");
    }
    size_t nf = n / 4; double best = 0;
    for (int r = 0; r < 5; r++) {
        double t = omp_get_wtime(); float tot = 0;
        #pragma omp parallel reduction(+:tot)
        { __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
          #pragma omp for schedule(static)
          for (size_t i = 0; i < nf; i += 32) { s0 = _mm256_add_ps(s0, _mm256_load_ps(a+i)); s1 = _mm256_add_ps(s1, _mm256_load_ps(a+i+8)); s2 = _mm256_add_ps(s2, _mm256_load_ps(a+i+16)); s3 = _mm256_add_ps(s3, _mm256_load_ps(a+i+24)); }
          float o[8]; _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3))); for (int k = 0; k < 8; k++) tot += o[k]; }
        t = omp_get_wtime() - t; double bw = n / t / 1e9; if (bw > best) best = bw; if (tot < 0) puts("x");
    }
    printf("%.1f\n", best); munmap(a, n); close(fd); unlink(argv[1]); return 0;
}
