#!/usr/bin/env python3
"""Check how well GGML_NUMA_PLACE put each weight's row quarters on their nodes.

    placecheck.py <shard.gguf> [tensor-substring] [max-tensors]

For a sample of 2D slices of each matching tensor, asks the kernel (move_pages with no target)
which node every 4 KiB page sits on, and reports the share of each row block that is on the
node meant to read it (block k of 4 belongs to node k).
"""
import ctypes
import mmap
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_tensors  # noqa: E402

SYS_move_pages = 279  # x86_64
PAGE = 4096
N_NODES = 4

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def page_nodes(addr, n_pages):
    # move_pages only reports pages mapped into this process: fault them in (read, no copy)
    for i in range(n_pages):
        ctypes.c_char.from_address(addr + i * PAGE).value
    pages = (ctypes.c_void_p * n_pages)(*[addr + i * PAGE for i in range(n_pages)])
    status = (ctypes.c_int * n_pages)()
    if libc.syscall(SYS_move_pages, 0, ctypes.c_ulong(n_pages), pages, None, status, 0) != 0:
        raise OSError(ctypes.get_errno(), "move_pages")
    return list(status)


def main():
    path = sys.argv[1]
    pat = sys.argv[2] if len(sys.argv) > 2 else "_exps"
    max_t = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    _meta, infos = gguf_tensors.tensors(path)
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        # private + writable only so ctypes can take its address; nothing is written
        mm = mmap.mmap(f.fileno(), size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    addr0 = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    data_start = size - sum(i[4] for i in infos)  # the tensors fill the file after the header
    random.seed(0)
    done = 0
    for name, _ttype, dims, off, nbytes in infos:
        if pat not in name or len(dims) < 2:
            continue
        ne1 = dims[1]
        n_slices = dims[2] if len(dims) > 2 else 1
        slice_bytes = nbytes // n_slices
        row_bytes = slice_bytes // ne1
        ok = [0] * N_NODES
        tot = [0] * N_NODES
        for e in random.sample(range(n_slices), min(8, n_slices)):
            s0 = addr0 + data_start + off + e * slice_bytes
            for k in range(N_NODES):
                lo = s0 + (k * ne1 // N_NODES) * row_bytes
                hi = s0 + ((k + 1) * ne1 // N_NODES) * row_bytes
                p_lo = lo // PAGE * PAGE
                n_p = (hi - p_lo + PAGE - 1) // PAGE
                for i, node in enumerate(page_nodes(p_lo, n_p)):
                    a = max(lo, p_lo + i * PAGE)
                    b = min(hi, p_lo + (i + 1) * PAGE)
                    tot[k] += b - a
                    if node == k:
                        ok[k] += b - a
        share = " ".join(f"n{k}={100.0 * ok[k] / tot[k]:5.1f}%" for k in range(N_NODES))
        print(f"{name:32s} on its node: {share}")
        done += 1
        if done >= max_t:
            break


if __name__ == "__main__":
    main()
