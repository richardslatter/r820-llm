#!/usr/bin/env python3
# LLAMA_DECODE_TRACE=1: one stderr line per llama_decode call (start time, tokens, layers, total and graph-compute us)
import sys
root = sys.argv[1] if len(sys.argv) > 1 else '.'

def patch(path, old, new, count=1):
    p = f'{root}/{path}'
    s = open(p).read()
    if new in s:
        return
    n = s.count(old)
    assert n == count, f'{path}: expected {count} match(es), found {n}: {old[:60]!r}'
    open(p, 'w').write(s.replace(old, new))

patch('src/llama.cpp',
'''static void llama_graph_compute(
''',
'''static int64_t g_decode_trace_compute_us = 0;
static bool llama_decode_trace() {
    static const bool on = [] { const char * e = getenv("LLAMA_DECODE_TRACE"); return e && atoi(e) != 0; }();
    return on;
}

static void llama_graph_compute_impl(
        llama_context & lctx,
          ggml_cgraph * gf,
                  int   n_threads);

static void llama_graph_compute(
        llama_context & lctx,
          ggml_cgraph * gf,
                  int   n_threads) {
    const int64_t t0 = llama_decode_trace() ? ggml_time_us() : 0;
    llama_graph_compute_impl(lctx, gf, n_threads);
    if (llama_decode_trace()) {
        ggml_backend_sched_synchronize(lctx.sched);
        g_decode_trace_compute_us += ggml_time_us() - t0;
    }
}

static void llama_graph_compute_impl(
''')

patch('src/llama.cpp',
'''int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_decode_internal(*ctx, batch);
''',
'''int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int64_t t0 = ggml_time_us(), c0 = g_decode_trace_compute_us;
    const int ret = llama_decode_internal(*ctx, batch);
    if (llama_decode_trace()) {
        fprintf(stderr, "DTRACE t=%lld n=%d L=%d total=%lld compute=%lld\\n", (long long) t0, batch.n_tokens,
                (int) ctx->model.hparams.n_layer, (long long)(ggml_time_us() - t0), (long long)(g_decode_trace_compute_us - c0));
    }
''')
print('ok')
