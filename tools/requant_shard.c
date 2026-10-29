// Requantize the BF16/F32 weights of ONE GGUF shard, copying everything else byte for byte.
//
// DeepSeek-V4.1-Flash UD-Q8_K_XL keeps every BF16 weight in its last shard, so rewriting that
// shard alone (next to hard links of the untouched ones) changes the dense weights without a
// second 500 GB copy of the model.
//
//   requant_shard in.gguf out.gguf <type> [name-substring=<type>|keep ...]
//
// <type> applies to every BF16 tensor with 2+ dims; overrides are matched in order, first hit
// wins. F32 tensors are only converted when an override names them.
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define KEEP GGML_TYPE_COUNT

static enum ggml_type parse_type(const char * s) {
    if (!strcmp(s, "keep")) return KEEP;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((enum ggml_type) t);
        if (n && !strcmp(n, s) && ggml_blck_size((enum ggml_type) t) > 0) return (enum ggml_type) t;
    }
    fprintf(stderr, "unknown type '%s'\n", s);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s in.gguf out.gguf <type> [name-substring=<type>|keep ...]\n", argv[0]);
        return 1;
    }
    const enum ggml_type def = parse_type(argv[3]);
    const int n_ov = argc - 4;
    char ** ov_pat = calloc(n_ov + 1, sizeof(char *));
    enum ggml_type * ov_type = calloc(n_ov + 1, sizeof(enum ggml_type));
    for (int i = 0; i < n_ov; ++i) {
        char * eq = strchr(argv[4 + i], '=');
        if (!eq) { fprintf(stderr, "bad override '%s'\n", argv[4 + i]); return 1; }
        *eq = 0;
        ov_pat[i] = argv[4 + i];
        ov_type[i] = parse_type(eq + 1);
    }

    struct ggml_context * meta = NULL;
    struct gguf_init_params ip = { .no_alloc = true, .ctx = &meta };
    struct gguf_context * in = gguf_init_from_file(argv[1], ip);
    if (!in) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    const int fd = open(argv[1], O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    const char * map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    const char * data = map + gguf_get_data_offset(in);

    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);

    size_t bytes_in = 0, bytes_out = 0;
    const int64_t n = gguf_get_n_tensors(in);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        struct ggml_tensor * t = ggml_get_tensor(meta, name);
        const char * src = data + gguf_get_tensor_offset(in, i);
        gguf_add_tensor(out, t);

        enum ggml_type to = (t->type == GGML_TYPE_BF16 && ggml_n_dims(t) >= 2) ? def : KEEP;
        for (int k = 0; k < n_ov; ++k) {
            if (strstr(name, ov_pat[k])) { to = ov_type[k]; break; }
        }
        if (to != KEEP && t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_F32) to = KEEP;
        if (to != KEEP && t->ne[0] % ggml_blck_size(to) != 0) {
            fprintf(stderr, "%s: row %lld not a multiple of %s blocks, kept\n", name, (long long) t->ne[0], ggml_type_name(to));
            to = KEEP;
        }
        bytes_in += ggml_nbytes(t);
        if (to == KEEP || to == t->type) {
            gguf_set_tensor_data(out, name, src);
            bytes_out += ggml_nbytes(t);
            continue;
        }

        const int64_t ne0 = t->ne[0], nrows = ggml_nrows(t);
        const size_t row_out = ggml_row_size(to, ne0), row_in = ggml_row_size(t->type, ne0);
        char * dst = malloc(row_out * nrows);
        ggml_quantize_init(to);
        #pragma omp parallel
        {
            float * f = malloc(sizeof(float) * ne0 * 64);
            #pragma omp for schedule(dynamic, 1)
            for (int64_t r0 = 0; r0 < nrows; r0 += 64) {
                const int64_t nr = r0 + 64 <= nrows ? 64 : nrows - r0;
                for (int64_t r = 0; r < nr; ++r) {
                    const char * s = src + (r0 + r) * row_in;
                    if (t->type == GGML_TYPE_BF16) {
                        ggml_bf16_to_fp32_row((const ggml_bf16_t *) s, f + r * ne0, ne0);
                    } else {
                        memcpy(f + r * ne0, s, sizeof(float) * ne0);
                    }
                }
                ggml_quantize_chunk(to, f, dst + r0 * row_out, 0, nr, ne0, NULL);
            }
            free(f);
        }
        gguf_set_tensor_type(out, name, to);
        gguf_set_tensor_data(out, name, dst);
        bytes_out += row_out * nrows;
        printf("%-40s %-5s -> %-5s %8.1f MB -> %8.1f MB\n", name, ggml_type_name(t->type), ggml_type_name(to),
               ggml_nbytes(t) / 1e6, row_out * nrows / 1e6);
    }
    printf("total %.2f GB -> %.2f GB, writing %s\n", bytes_in / 1e9, bytes_out / 1e9, argv[2]);
    fflush(stdout);
    if (!gguf_write_to_file(out, argv[2], false)) { fprintf(stderr, "write failed\n"); return 1; }
    return 0;
}
