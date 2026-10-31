#!/usr/bin/env python3
# GGML_CUDA_PIN_MAX_GB=<n>: once NUMA placement is done (pinned pages cannot move), pin the mmapped
# host buffers of at most n GB with cudaHostRegister. With n = 10 that is the dense shard, so the
# weight uploads of large batches run at PCIe speed instead of through pageable staging.
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

patch('ggml/include/ggml-cuda.h',
'''GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);
''',
'''GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);
// pins without the GGML_CUDA_REGISTER_HOST check
GGML_API GGML_CALL bool ggml_backend_cuda_pin_host_buffer(void * buffer, size_t size);
''')

patch('ggml/src/ggml-cuda.cu',
'''GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {''',
'''GGML_CALL bool ggml_backend_cuda_pin_host_buffer(void * buffer, size_t size) {
    // read-only mappings need the ReadOnly flag where the device supports it
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        cudaGetLastError();
        err = cudaHostRegister(buffer, size, cudaHostRegisterPortable);
    }
    if (err != cudaSuccess) {
        cudaGetLastError();
        GGML_CUDA_LOG_WARN("%s: failed to pin %.2f MiB: %s\\n", __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {''')

patch('src/llama.cpp',
'''            LLAMA_LOG_INFO("%s: NUMA-placed %d tensors, %.1f GB, in %.1f s\\n", __func__, n_placed,
                           placed/1e9, (ggml_time_us() - t0)/1e6);
        }
''',
'''            LLAMA_LOG_INFO("%s: NUMA-placed %d tensors, %.1f GB, in %.1f s\\n", __func__, n_placed,
                           placed/1e9, (ggml_time_us() - t0)/1e6);
        }
#ifdef GGML_USE_CUDA
        // GGML_CUDA_PIN_MAX_GB=<n>: pin the mmapped host buffers of at most n GB (after placement)
        if (const char * e = getenv("GGML_CUDA_PIN_MAX_GB"); e && n_layer >= n_gpu_layers) {
            const size_t cap = size_t(atof(e)*1e9);
            for (ggml_backend_buffer_t buf : model.bufs) {
                if (ggml_backend_buffer_get_type(buf) != ggml_backend_cpu_buffer_type()) continue;
                const size_t size = ggml_backend_buffer_get_size(buf);
                if (size > cap) continue;
                const int64_t t0 = ggml_time_us();
                const bool ok = ggml_backend_cuda_pin_host_buffer(ggml_backend_buffer_get_base(buf), size);
                LLAMA_LOG_INFO("%s: pinned %.2f GB of host weights for the GPU: %s, %.1f s\\n", __func__, size/1e9,
                               ok ? "ok" : "failed", (ggml_time_us() - t0)/1e6);
            }
        }
#endif
''')
print('ok')
