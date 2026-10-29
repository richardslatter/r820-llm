#!/usr/bin/env python3
"""Follow-up to numa_patch.py (run from the ik_llama.cpp root, after numa_patch.py).

GGML_NUMA_BLOCKED=1 now pins every thread to ONE hardware thread instead of its whole node:
node k's threads take node k's cpus in /sys order (on the R820: the 8 physical cores first, then
their hyperthread siblings). Node-wide masks let the scheduler stack two spinning threads on one
cpu, which made identical runs vary between ~0.7 and ~2.2 tok/s.
GGML_NUMA_SKIP=n leaves the first n cpus of every node to the OS (CPU 0 takes the ACPI work).

GGML_NUMA_PLACE=1 now faults each range in (MADV_POPULATE_READ) before mbind(MPOL_MF_MOVE):
the kernel only migrates pages this process has mapped, so on a freshly mmapped model the move
silently did nothing.
"""
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


C = "ggml/src/ggml.c"

patch(C,
      "static bool g_numa_blocked = false;\n",
      "static bool g_numa_blocked = false;\n"
      "// GGML_NUMA_SKIP=n: blocked threads start at cpu n of their node, leaving the first n to the OS\n"
      "static int  g_numa_skip    = 0;\n")
patch(C,
      "        g_numa_blocked = e && atoi(e) != 0;\n",
      "        g_numa_blocked = e && atoi(e) != 0;\n"
      "        const char * k = getenv(\"GGML_NUMA_SKIP\");\n"
      "        g_numa_skip = k ? atoi(k) : 0;\n")

patch(C,
      "    CPU_ZERO_S(setsize, cpus);\n"
      "    for (size_t i = 0; i < node->n_cpus; ++i) {\n"
      "        CPU_SET_S(node->cpus[i], setsize, cpus);\n"
      "    }\n"
      "\n"
      "    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);\n",
      "    CPU_ZERO_S(setsize, cpus);\n"
      "    if (g_numa_blocked && g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_DISTRIBUTE) {\n"
      "        // one hardware thread each, so the scheduler can never stack two spinning threads\n"
      "        const int n_nodes = (int)g_state.numa.n_nodes;\n"
      "        const int local   = thread_n - (node_num * n_threads + n_nodes - 1) / n_nodes;\n"
      "        CPU_SET_S(node->cpus[(local + g_numa_skip) % node->n_cpus], setsize, cpus);\n"
      "    } else {\n"
      "        for (size_t i = 0; i < node->n_cpus; ++i) {\n"
      "            CPU_SET_S(node->cpus[i], setsize, cpus);\n"
      "        }\n"
      "    }\n"
      "\n"
      "    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);\n")

patch(C,
      "                unsigned long mask[2] = { 0, 0 };\n"
      "                mask[0] = 1ul << k;\n",
      "                unsigned long mask[2] = { 0, 0 };\n"
      "                mask[0] = 1ul << k;\n"
      "                // MPOL_MF_MOVE only migrates pages mapped into this process: map them first\n"
      "                madvise((void *)lo, hi - lo, 22 /*MADV_POPULATE_READ*/);\n")

patch(C,
      "#include <sys/syscall.h>\n#include <unistd.h>\n",
      "#include <sys/mman.h>\n#include <sys/syscall.h>\n#include <unistd.h>\n")

print("numa pin patch applied")
