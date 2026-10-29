#!/usr/bin/env python3
"""Split the huge pages of tmpfs files to 4 KiB, in place, 1 GiB at a time.

The debugfs file-path form refuses these files (EINVAL), the pid/address form works: map the
file, populate a window, ask the kernel to split that range of our own mapping, unmap.
    sudo is needed for /sys/kernel/debug/split_huge_pages;  usage: splitpages.py FILE...
"""
import ctypes, mmap, os, subprocess, sys, time
libc = ctypes.CDLL(None, use_errno=True)
W = 1 << 30
def huge():
    return int([l for l in open("/proc/meminfo") if l.startswith("ShmemHugePages")][0].split()[1]) // 2048
for path in sys.argv[1:]:
    t0, h0 = time.time(), huge()
    fd = os.open(path, os.O_RDONLY); size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    for off in range(0, size, W):
        n = min(W, size - off)
        libc.madvise(ctypes.c_void_p(base + off), ctypes.c_size_t(n), 22)          # MADV_POPULATE_READ
        cmd = f"{os.getpid()},0x{base + off:x},0x{base + off + n:x},0"
        subprocess.run(["sudo", "tee", "/sys/kernel/debug/split_huge_pages"], input=cmd.encode(), stdout=subprocess.DEVNULL, check=True)
        libc.madvise(ctypes.c_void_p(base + off), ctypes.c_size_t(n), 4)           # MADV_DONTNEED: drop our mapping
    del mm; os.close(fd)
    print(f"{os.path.basename(path)}: {h0 - huge()} huge pages split in {time.time() - t0:.0f} s", flush=True)
