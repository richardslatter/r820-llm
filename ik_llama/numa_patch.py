#!/usr/bin/env python3
"""Apply the NUMA-local placement changes to an ik_llama.cpp checkout (run from its root).

GGML_NUMA_BLOCKED=1 (with --numa distribute):
    thread ith is pinned to node ith*n_nodes/nth, so each node's threads cover one contiguous
    quarter of every weight matrix's rows, and MoE expert chunks are handed out statically by node.
GGML_NUMA_PLACE=1:
    after loading, row quarter k of every weight matrix (per expert slice) is mbind-moved to node k.
"""
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


GGML_C = "ggml/src/ggml.c"
GGML_H = "ggml/include/ggml.h"
LLAMA_CPP = "src/llama.cpp"

# --- public API -----------------------------------------------------------------------------
patch(GGML_H,
      "    GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node\n",
      "    GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node\n"
      "    // Move row block k (of n_nodes) of every 2D slice of t onto NUMA node k; returns bytes placed.\n"
      "    // Pairs with GGML_NUMA_BLOCKED=1, which makes node k's threads compute exactly those rows.\n"
      "    GGML_API size_t  ggml_numa_place_tensor(const struct ggml_tensor * t);\n")

# --- blocked mode flag, set at numa init ----------------------------------------------------
patch(GGML_C,
      "bool ggml_is_numa(void) {\n",
      "// GGML_NUMA_BLOCKED=1: distribute threads to nodes in contiguous blocks instead of round robin\n"
      "static bool g_numa_blocked = false;\n"
      "\n"
      "bool ggml_is_numa(void) {\n")

patch(GGML_C,
      "    return g_state.numa.n_nodes > 1;\n",
      "    return g_state.numa.n_nodes > 1;\n")  # sanity: exactly one ggml_is_numa body

s = open(GGML_C).read()
i = s.index("void ggml_numa_init(enum ggml_numa_strategy numa_flag) {")
j = s.index("\n{", i) if False else s.index("{", i) + 1
s = s[:j] + ("\n    {\n        const char * e = getenv(\"GGML_NUMA_BLOCKED\");\n"
             "        g_numa_blocked = e && atoi(e) != 0;\n    }\n") + s[j:]
open(GGML_C, "w").write(s)

# --- thread pinning ---------------------------------------------------------------------------
patch(GGML_C,
      "static void set_numa_thread_affinity(int thread_n) {\n    if (!ggml_is_numa()) {",
      "static void set_numa_thread_affinity(int thread_n, int n_threads) {\n    if (!ggml_is_numa()) {")
patch(GGML_C,
      "            node_num = thread_n % g_state.numa.n_nodes;\n",
      "            node_num = g_numa_blocked ? thread_n * (int)g_state.numa.n_nodes / n_threads\n"
      "                                      : thread_n % (int)g_state.numa.n_nodes;\n")
patch(GGML_C,
      "static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }",
      "static void set_numa_thread_affinity(int thread_n, int n_threads) { UNUSED(thread_n); UNUSED(n_threads); }")
patch(GGML_C,
      "    set_numa_thread_affinity(state->ith);\n",
      "    set_numa_thread_affinity(state->ith, state->shared->n_threads);\n")

# --- placement -----------------------------------------------------------------------------
patch(GGML_C,
      "static void clear_numa_thread_affinity(void) {\n",
      r'''size_t ggml_numa_place_tensor(const struct ggml_tensor * t) {
    const int n_nodes = (int)g_state.numa.n_nodes;
    if (n_nodes < 2 || !t || !t->data || t->ne[1] < n_nodes) {
        return 0;
    }
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const int64_t nrows = t->ne[1];
    size_t placed = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            const uintptr_t base = (uintptr_t)t->data + i2*t->nb[2] + i3*t->nb[3];
            for (int k = 0; k < n_nodes; ++k) {
                // same split as the kernels: rows [k*nrows/n, (k+1)*nrows/n); pages straddling a
                // boundary go to the lower block
                uintptr_t lo = (base + (k*nrows/n_nodes)*t->nb[1]) & ~(page - 1);
                uintptr_t hi = (base + ((k + 1)*nrows/n_nodes)*t->nb[1]) & ~(page - 1);
                if (k == n_nodes - 1) hi = (base + nrows*t->nb[1] + page - 1) & ~(page - 1);
                if (hi <= lo) continue;
                unsigned long mask[2] = { 0, 0 };
                mask[0] = 1ul << k;
                // MPOL_BIND = 2, MPOL_MF_MOVE = 2: also migrates pages already present (page cache / tmpfs)
                if (syscall(SYS_mbind, (void *)lo, hi - lo, 2, mask, 8*sizeof(mask[0]) + 1, 2) == 0) {
                    placed += hi - lo;
                }
            }
        }
    }
    return placed;
}

static void clear_numa_thread_affinity(void) {
''', count=1)

patch(GGML_C,
      "#include <stdarg.h>\n",
      "#include <stdarg.h>\n#if defined(__gnu_linux__)\n#include <sys/syscall.h>\n#include <unistd.h>\n#endif\n", count=1)

# non-linux builds need the symbol too
patch(GGML_C,
      "static void set_numa_thread_affinity(int thread_n, int n_threads) { UNUSED(thread_n); UNUSED(n_threads); }",
      "static void set_numa_thread_affinity(int thread_n, int n_threads) { UNUSED(thread_n); UNUSED(n_threads); }\n"
      "size_t ggml_numa_place_tensor(const struct ggml_tensor * t) { UNUSED(t); return 0; }")

# --- MoE: static, node-aware expert chunks ------------------------------------------------------
NODE_SPLIT = '''        // GGML_NUMA_BLOCKED: node k holds rows quarter k of every expert, so its threads take only
        // the chunks of that quarter, for every active expert (static, no shared counter)
        const int n_nodes = (int)g_state.numa.n_nodes;
        if (g_numa_blocked && n_nodes > 1 && nth % n_nodes == 0 && {CPE} % n_nodes == 0) {
            const int node = ith * n_nodes / nth;
            const int tpn  = nth / n_nodes;
            const int c0   = node * {CPE} / n_nodes;
            const int c1   = (node + 1) * {CPE} / n_nodes;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] == 0) continue;
                for (int local_chunk = c0 + ith - node * tpn; local_chunk < c1; local_chunk += tpn) {
{BODY}
                }
            }
            return;
        }
'''

# down projection (mul_mat_id)
patch(GGML_C,
      "        int chunk_id = ith;\n        while (chunk_id < total_chunks) {\n",
      NODE_SPLIT.replace("{CPE}", "chunks_per_expert").replace("{BODY}",
      '''                    const char * src0_cur = (const char *) src0->data + cur_a*nb02;
                    if (!iqk_mul_mat_moe(ne01, matrix_row_counts[cur_a], ne00, ne11,
                                src0->type, src0_cur, nb01,
                                vec_dot_type, (const char *)wdata_mm, row_size_mm,
                                (float *)dst->data, nb1, nb2,
                                matrix_rows + cur_a*ne12, local_chunk, chunks_per_expert)) goto IQK_MulMat_Not_Available0;''')
      + "\n        int chunk_id = ith;\n        while (chunk_id < total_chunks) {\n")

def dedent4(block):
    return "\n".join(l[4:] if l.startswith("    ") else l for l in block.split("\n"))


# fused up/gate (mul_mat_id_up_gate); this loop sits one level shallower than the down projection's
patch(GGML_C,
      "    int last_a = 0;\n    int last_acc = 0;\n    int chunk_id_ug = ith;\n",
      dedent4(NODE_SPLIT.replace("{CPE}", "chunks_per_expert_ug").replace("{BODY}",
      '''                    const char *src0_1_cur, *src0_2_cur, *up_b_cur = NULL, *gate_b_cur = NULL;
                    if (src0_2) {
                        src0_1_cur = (const char *) src0_1->data + cur_a*nb02;
                        src0_2_cur = (const char *) src0_2->data + cur_a*nb02;
                        up_b_cur   = up_b   ? (const char *)up_b->data + cur_a*nb41 : NULL;
                        gate_b_cur = gate_b ? (const char *)gate_b->data + cur_a*nb51 : NULL;
                    } else {
                        src0_2_cur = (const char *) src0_1->data + cur_a*nb02;
                        src0_1_cur = src0_2_cur + nb02/2;
                        if (up_b) {
                            GGML_ASSERT(!gate_b);
                            gate_b_cur = (const char *)up_b->data + cur_a*nb41;
                            up_b_cur   = gate_b_cur + nb41/2;
                        }
                    }
                    if (!iqk_moe_fused_up_gate(nr0_base, matrix_row_counts[cur_a], ne00, ne11, dst->op_params[0],
                                type, src0_1_cur, src0_2_cur, nb01,
                                vec_dot_type, (const char *)wdata_ug, row_size_ug,
                                up_b_cur, gate_b_cur,
                                (float *)dst->data, nb1, nb2,
                                matrix_rows + cur_a*ne12, limit, situ_beta, situ_beta_linear, local_chunk, chunks_per_expert_ug)) GGML_ABORT("fatal error");'''))
      + "    int last_a = 0;\n    int last_acc = 0;\n    int chunk_id_ug = ith;\n")

# --- loader hook ------------------------------------------------------------------------------
patch(LLAMA_CPP,
      "        if (defer_ple_mmap) {\n            ml.apply_ple_mmap_policy();\n        }\n",
      '''        if (defer_ple_mmap) {
            ml.apply_ple_mmap_policy();
        }

        // GGML_NUMA_PLACE=1: put row quarter k of every weight matrix on node k (pairs with
        // GGML_NUMA_BLOCKED=1). Lookup tables (token / Engram embeddings) are left where they are.
        if (const char * e = getenv("GGML_NUMA_PLACE"); e && atoi(e) && ggml_is_numa()) {
            const int64_t t0 = ggml_time_us();
            size_t placed = 0;
            int n_placed = 0;
            for (auto & it : model.tensors_by_name) {
                const ggml_tensor * t = it.second;
                if (!t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) continue;
                if (strstr(t->name, "engram") || strstr(t->name, "token_embd")) continue;
                if (const size_t b = ggml_numa_place_tensor(t)) {
                    placed += b;
                    ++n_placed;
                }
            }
            LLAMA_LOG_INFO("%s: NUMA-placed %d tensors, %.1f GB, in %.1f s\\n", __func__, n_placed,
                           placed/1e9, (ggml_time_us() - t0)/1e6);
        }
''')

print("numa patch applied")
