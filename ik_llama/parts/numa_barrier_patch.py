#!/usr/bin/env python3
"""Two-level NUMA barrier and a per-op profiler for ggml.c (run from the ik_llama.cpp root).

With GGML_NUMA_BLOCKED=1 (threads pinned to nodes in contiguous blocks) the graph barrier
becomes hierarchical: threads of a node meet on a counter in their own cache line, the last
one of each node meets the other nodes on a global counter, and the release is written to one
flag per node so every waiting thread spins on a line held by its own socket. Decode has
~1500 barriers per token and the flat libgomp barrier bounces one line between 4 sockets.
GGML_HBARRIER=0 turns it off.

GGML_OP_PROF=1 (=2: only graphs of <= 8 tokens) makes thread 0 time every node: wall time from node start to barrier exit and
the slowest thread's compute time, accumulated per op (and weight type for matmuls), printed
at exit. wall - slowest compute = barrier cost; slowest - average = load imbalance.
"""
import sys

p = "ggml/src/ggml.c"
s = open(p).read()
if "ggml_hbarrier" in s:
    sys.exit(f"{p}: already patched")


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"{p}: expected {count} of {old[:60]!r}, found {n}")
    s = s.replace(old, new)


# shared state: barrier counters, one cache line each
sub("""    atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads

    enum ggml_status ec;
};
""", """    atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads

    enum ggml_status ec;

    // two-level barrier for GGML_NUMA_BLOCKED (ggml_hbarrier); hb_n_nodes == 0 means off
    int hb_n_nodes;
    int hb_node_n[8];
    struct { _Alignas(64) atomic_int v; } hb_count[8];
    struct { _Alignas(64) atomic_int v; } hb_release[8];
    _Alignas(64) atomic_int hb_global;
};
""")

# the barrier itself, placed before the OpenMP/plain ggml_barrier definitions
sub("""#ifdef GGML_USE_OPENMP
static void ggml_barrier(struct ggml_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }
""", """static _Thread_local int g_hb_node = 0;

static void ggml_hbarrier(struct ggml_compute_state_shared * shared) {
    const int node = g_hb_node;
    const int epoch = atomic_load_explicit(&shared->hb_release[node].v, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&shared->hb_count[node].v, 1, memory_order_acq_rel) == shared->hb_node_n[node] - 1) {
        atomic_store_explicit(&shared->hb_count[node].v, 0, memory_order_relaxed);
        if (atomic_fetch_add_explicit(&shared->hb_global, 1, memory_order_acq_rel) == shared->hb_n_nodes - 1) {
            atomic_store_explicit(&shared->hb_global, 0, memory_order_relaxed);
            for (int k = 0; k < shared->hb_n_nodes; ++k) {
                atomic_store_explicit(&shared->hb_release[k].v, epoch + 1, memory_order_release);
            }
            return;
        }
    }
    while (atomic_load_explicit(&shared->hb_release[node].v, memory_order_acquire) == epoch) {
#if defined(__SSE3__)
        _mm_pause();
#endif
    }
}

#ifdef GGML_USE_OPENMP
static void ggml_barrier(struct ggml_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }
    if (shared->hb_n_nodes > 1) {
        ggml_hbarrier(shared);
        return;
    }
""")

sub("""#else
static void ggml_barrier(struct ggml_compute_state_shared * shared) {
    ggml_barrier_impl(shared);
}
#endif
""", """#else
static void ggml_barrier(struct ggml_compute_state_shared * shared) {
    if (shared->n_threads > 1 && shared->hb_n_nodes > 1) {
        ggml_hbarrier(shared);
        return;
    }
    ggml_barrier_impl(shared);
}
#endif
""")

# setup (needs g_state / g_numa_blocked, so it goes right before the compute thread function)
sub("""static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
""", """static void ggml_hbarrier_setup(struct ggml_compute_state_shared * shared) {
    shared->hb_n_nodes = 0;
#if defined(__gnu_linux__)
    const char * e = getenv("GGML_HBARRIER");
    if (e && atoi(e) == 0) return;
    const int n_nodes = (int)g_state.numa.n_nodes, n = shared->n_threads;
    if (!g_numa_blocked || g_state.numa.numa_strategy != GGML_NUMA_STRATEGY_DISTRIBUTE ||
        n_nodes < 2 || n_nodes > 8 || n < 2*n_nodes) return;
    for (int k = 0; k < n_nodes; ++k) shared->hb_node_n[k] = 0;
    for (int i = 0; i < n; ++i) shared->hb_node_n[i*n_nodes/n]++;   // same split as set_numa_thread_affinity
    for (int k = 0; k < n_nodes; ++k) {
        atomic_store(&shared->hb_count[k].v, 0);
        atomic_store(&shared->hb_release[k].v, 0);
    }
    atomic_store(&shared->hb_global, 0);
    shared->hb_n_nodes = n_nodes;
#endif
}

// GGML_OP_PROF=1: per-op wall / slowest-thread time, printed at exit
#include <x86intrin.h>
#define GGML_PROF_KEYS (GGML_OP_COUNT*64)
static int      g_prof_on = -1;
static int64_t  g_prof_wall[GGML_PROF_KEYS], g_prof_max[GGML_PROF_KEYS], g_prof_sum[GGML_PROF_KEYS], g_prof_cnt[GGML_PROF_KEYS];
static int64_t  g_prof_thr[512];
static int64_t  g_prof_graphs = 0, g_prof_tsc0 = 0, g_prof_us0 = 0;
// matmuls again, per weight name with the "blk.N." prefix dropped
static char     g_prof_names[256][48];
static int64_t  g_prof_nwall[256], g_prof_nmax[256], g_prof_nsum[256], g_prof_ncnt[256], g_prof_nbytes[256];
static int      g_prof_nn = 0;

static int ggml_prof_name_idx(const char * name) {
    if (strncmp(name, "blk.", 4) == 0) {
        const char * d = strchr(name + 4, '.');
        if (d) name = d + 1;
    }
    for (int i = 0; i < g_prof_nn; ++i) if (strcmp(g_prof_names[i], name) == 0) return i;
    if (g_prof_nn == 256) return -1;
    snprintf(g_prof_names[g_prof_nn], sizeof(g_prof_names[0]), "%s", name);
    return g_prof_nn++;
}

static void ggml_prof_print(void) {
    const double tsc_per_us = (double)(__rdtsc() - g_prof_tsc0)/(double)(ggml_time_us() - g_prof_us0);
    int64_t tot = 0;
    for (int k = 0; k < GGML_PROF_KEYS; ++k) tot += g_prof_wall[k];
    fprintf(stderr, "\\nop profile: %lld graphs, %.2f ms/graph total wall\\n", (long long)g_prof_graphs, tot/tsc_per_us/1e3/g_prof_graphs);
    fprintf(stderr, "%-28s %8s %10s %10s %10s %10s\\n", "op", "n/graph", "wall ms", "slowest", "average", "barrier");
    for (;;) {
        int best = -1;
        for (int k = 0; k < GGML_PROF_KEYS; ++k) if (g_prof_cnt[k] && (best < 0 || g_prof_wall[k] > g_prof_wall[best])) best = k;
        if (best < 0) break;
        const int op = best/64, t = best%64;
        char name[64];
        if (op == GGML_OP_MUL_MAT || op == GGML_OP_MUL_MAT_ID || op == GGML_OP_MOE_FUSED_UP_GATE) {
            snprintf(name, sizeof(name), "%s.%s", ggml_op_name((enum ggml_op)op), ggml_type_name((enum ggml_type)t));
        } else {
            snprintf(name, sizeof(name), "%s", ggml_op_name((enum ggml_op)op));
        }
        const double g = (double)g_prof_graphs*tsc_per_us*1e3;
        fprintf(stderr, "%-28s %8.1f %10.2f %10.2f %10.2f %10.2f\\n", name, (double)g_prof_cnt[best]/g_prof_graphs,
                g_prof_wall[best]/g, g_prof_max[best]/g, g_prof_sum[best]/g, (g_prof_wall[best] - g_prof_max[best])/g);
        g_prof_cnt[best] = 0;
    }
    fprintf(stderr, "\\n%-28s %8s %10s %10s %10s %10s\\n", "matmul weight", "n/graph", "wall ms", "slowest", "average", "GB/s");
    for (;;) {
        int best = -1;
        for (int k = 0; k < g_prof_nn; ++k) if (g_prof_ncnt[k] && (best < 0 || g_prof_nwall[k] > g_prof_nwall[best])) best = k;
        if (best < 0) break;
        const double g = (double)g_prof_graphs*tsc_per_us*1e3;
        fprintf(stderr, "%-28s %8.1f %10.2f %10.2f %10.2f %10.1f\\n", g_prof_names[best], (double)g_prof_ncnt[best]/g_prof_graphs,
                g_prof_nwall[best]/g, g_prof_nmax[best]/g, g_prof_nsum[best]/g,
                g_prof_nbytes[best]/(g_prof_nwall[best]/tsc_per_us*1e3));
        g_prof_ncnt[best] = 0;
    }
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
""")

# per-thread node id, profiler around each node
sub("""    set_numa_thread_affinity(state->ith, state->shared->n_threads);
    set_cpu_thread_affinity(cplan, state->ith); // explicit affinity takes precedence over NUMA
""", """    set_numa_thread_affinity(state->ith, state->shared->n_threads);
    set_cpu_thread_affinity(cplan, state->ith); // explicit affinity takes precedence over NUMA
    if (state->shared->hb_n_nodes > 1) {
        g_hb_node = state->ith*state->shared->hb_n_nodes/state->shared->n_threads;
    }
    // GGML_OP_PROF=2: decode-sized graphs (<= 8 tokens) only
    const bool prof = g_prof_on > 0 && state->shared->n_threads <= 512 && (g_prof_on == 1 || state->shared->n_batch <= 8);
    if (prof && state->ith == 0) g_prof_graphs++;
""")

sub("""        node_n = ggml_compute_forward(&params, node, cgraph, node_n);
#if IK_PRINT_TIMING
        int64_t tim2 = ggml_time_us();
        t_eval += tim2 - tim1;
#endif

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);
""", """        const int64_t pt0 = prof ? (int64_t)__rdtsc() : 0;
        node_n = ggml_compute_forward(&params, node, cgraph, node_n);
        if (prof) g_prof_thr[state->ith] = (int64_t)__rdtsc() - pt0;
#if IK_PRINT_TIMING
        int64_t tim2 = ggml_time_us();
        t_eval += tim2 - tim1;
#endif

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);

        if (prof && state->ith == 0) {
            const int64_t wall = (int64_t)__rdtsc() - pt0;
            int64_t mx = 0, sm = 0;
            for (int j = 0; j < state->shared->n_threads; ++j) { mx = MAX(mx, g_prof_thr[j]); sm += g_prof_thr[j]; }
            int key = node->op*64;
            if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MOE_FUSED_UP_GATE) && node->src[0]) {
                key += node->src[0]->type % 64;
            }
            g_prof_wall[key] += wall; g_prof_max[key] += mx; g_prof_sum[key] += sm/state->shared->n_threads; g_prof_cnt[key]++;
            if (key%64 || node->op == GGML_OP_MUL_MAT) {
                const int ni = node->src[0] ? ggml_prof_name_idx(node->src[0]->name) : -1;
                if (ni >= 0) {
                    // bytes actually read: all of a dense weight, the selected experts' share of an expert tensor
                    int64_t bytes = ggml_nbytes(node->src[0]);
                    if (node->op != GGML_OP_MUL_MAT && node->src[0]->ne[2] > 1) {
                        const struct ggml_tensor * ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : node->src[3];
                        if (ids) bytes = bytes/node->src[0]->ne[2]*ggml_nelements(ids)*(node->op == GGML_OP_MOE_FUSED_UP_GATE ? 2 : 1);
                    }
                    g_prof_nwall[ni] += wall; g_prof_nmax[ni] += mx; g_prof_nsum[ni] += sm/state->shared->n_threads;
                    g_prof_ncnt[ni]++; g_prof_nbytes[ni] += bytes;
                }
            }
        }
""")

# set up the barrier once the real thread count is known
sub("""                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
            }
""", """                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
                ggml_hbarrier_setup(&state_shared);
            }
""")

sub("""    struct ggml_compute_state * workers = alloca(sizeof(struct ggml_compute_state)*n_threads);
""", """    struct ggml_compute_state * workers = alloca(sizeof(struct ggml_compute_state)*n_threads);
    ggml_hbarrier_setup(&state_shared);
""")

# profiler switch, read once at graph compute entry
sub("""    int n_threads = cplan->n_threads;

#if defined(__gnu_linux__)
    // the calling thread is worker 0 and gets pinned below; remember its affinity
""", """    int n_threads = cplan->n_threads;

    if (g_prof_on < 0) {
        const char * e = getenv("GGML_OP_PROF");
        g_prof_on = e ? atoi(e) : 0;
        if (g_prof_on) {
            g_prof_tsc0 = (int64_t)__rdtsc();
            g_prof_us0  = ggml_time_us();
            atexit(ggml_prof_print);
        }
    }

#if defined(__gnu_linux__)
    // the calling thread is worker 0 and gets pinned below; remember its affinity
""")

open(p, "w").write(s)
print("two-level NUMA barrier and GGML_OP_PROF profiler applied")
