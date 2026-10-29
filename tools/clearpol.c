// Drop the shared NUMA policy of tmpfs files (mbind MPOL_DEFAULT over the whole mapping).
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        int fd = open(argv[i], O_RDONLY); struct stat st; fstat(fd, &st);
        void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        long r = syscall(SYS_mbind, p, (st.st_size + 4095) & ~4095L, 0 /*MPOL_DEFAULT*/, NULL, 0, 0);
        printf("%s: %s\n", argv[i], r == 0 ? "policy cleared" : "mbind failed");
        munmap(p, st.st_size); close(fd);
    }
    return 0;
}
