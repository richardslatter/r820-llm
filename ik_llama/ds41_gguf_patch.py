#!/usr/bin/env python3
"""Let ik_llama.cpp load DeepSeek-V4.1 GGUFs in the mainline llama.cpp / Unsloth layout
(run from the ik_llama.cpp root).

Both layouts describe the same Engram module and the same rolling-xor n-gram hash; they differ
in names and packing:

    ik                                   mainline / Unsloth
    engram.layer_ids                     ple.layers
    engram.head_count                    ple.heads_per_ngram
    engram.key_length                    embedding_length_per_layer_input
    engram.max_ngram_size                ple.ngram_size
    engram.multipliers/primes/offsets    ple.layer_multipliers/head_vocab_sizes/head_offsets
    engram.pad_id (compressed id)        engram.pad_token_id (token id, mapped through token_map)
    blk.N.engram_embd (one per layer)    per_layer_token_embd (one table, offsets are global)
    blk.N.engram_wkv                     blk.N.engram_kv
    blk.N.engram_k, blk.N.engram_q       blk.N.engram_gate (= engram_k * engram_q, folded)

The gate is sum(rms(key)*wk * rms(x)*wq), so the folded weight goes in engram_k and the query
side runs without a weight.
"""
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


HP = "src/llama-hparams.cpp"
I = "                        "  # indent of the DeepSeek-V4.1 engram block

# --- hparams -------------------------------------------------------------------------------
patch(HP,
      I + "ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_n_layer);\n",
      I + "// mainline / Unsloth GGUFs name the same hash parameters ple.* (see ds41_gguf_patch.py)\n" +
      I + "const bool ple_names = !ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_n_layer, false);\n" +
      I + "const std::string ids_key = ple_names ? std::string(\"deepseek41.ple.layers\") : ml.llm_kv(LLM_KV_ENGRAM_LAYER_IDS);\n" +
      I + "if (ple_names) {\n" +
      I + "    ml.get_arr_n(ids_key, hparams.engram_n_layer);\n" +
      I + "}\n")
patch(HP,
      I + "    ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_LAYER_IDS), ids);\n",
      I + "    ml.get_arr(ids_key, ids);\n")
patch(HP,
      I + "ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);\n" +
      I + "ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);\n" +
      I + "ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);\n" +
      I + "ml.get_key(LLM_KV_ENGRAM_PAD_ID,         model.engram_pad_id);\n",
      I + "if (ple_names) {\n" +
      I + "    ml.get_key(\"deepseek41.ple.heads_per_ngram\",              hparams.engram_n_head);\n" +
      I + "    ml.get_key(\"deepseek41.embedding_length_per_layer_input\", hparams.engram_key_length);\n" +
      I + "    ml.get_key(\"deepseek41.ple.ngram_size\",                   hparams.engram_max_ngram_size);\n" +
      I + "} else {\n" +
      I + "    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);\n" +
      I + "    ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);\n" +
      I + "    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);\n" +
      I + "    ml.get_key(LLM_KV_ENGRAM_PAD_ID,         model.engram_pad_id);\n" +
      I + "}\n")
patch(HP,
      I + "ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_MULTIPLIERS), model.engram_multipliers);\n" +
      I + "ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_PRIMES),      model.engram_primes);\n" +
      I + "ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_OFFSETS),     model.engram_offsets);\n" +
      I + "ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_TOKEN_MAP),   model.engram_token_map);\n",
      I + "if (ple_names) {\n" +
      I + "    ml.get_arr(\"deepseek41.ple.layer_multipliers\", model.engram_multipliers);\n" +
      I + "    ml.get_arr(\"deepseek41.ple.head_vocab_sizes\",  model.engram_primes);\n" +
      I + "    ml.get_arr(\"deepseek41.ple.head_offsets\",      model.engram_offsets);\n" +
      I + "} else {\n" +
      I + "    ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_MULTIPLIERS), model.engram_multipliers);\n" +
      I + "    ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_PRIMES),      model.engram_primes);\n" +
      I + "    ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_OFFSETS),     model.engram_offsets);\n" +
      I + "}\n" +
      I + "ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_TOKEN_MAP),   model.engram_token_map);\n" +
      I + "if (ple_names) {\n" +
      I + "    // ple pads with a token id; the hash works on compressed ids\n" +
      I + "    uint32_t pad_token = 0;\n" +
      I + "    ml.get_key(\"deepseek41.engram.pad_token_id\", pad_token);\n" +
      I + "    if (pad_token >= model.engram_token_map.size()) {\n" +
      I + "        throw std::runtime_error(\"DeepSeek-V4.1 engram pad token is outside the token map\");\n" +
      I + "    }\n" +
      I + "    model.engram_pad_id = model.engram_token_map[pad_token];\n" +
      I + "}\n")

# --- tensors -------------------------------------------------------------------------------
patch("src/llama-load-tensors.cpp",
      "            layer.engram_q   = create_tensor_from_meta(ctx_split, format(\"blk.%d.engram_q.weight\", i));\n"
      "        }\n",
      "            layer.engram_q   = create_tensor_from_meta(ctx_split, format(\"blk.%d.engram_q.weight\", i));\n"
      "        } else if (model.arch == LLM_ARCH_DEEPSEEK41 && hparams.engram_index(i) >= 0 &&\n"
      "                   ml.get_tensor_meta(\"per_layer_token_embd.weight\") != nullptr) {\n"
      "            // mainline / Unsloth layout: one table shared by all Engram layers (its offsets are\n"
      "            // global), and the key and query norm weights folded into one gate\n"
      "            ggml_tensor * table = ggml_get_tensor(ctx_input, \"per_layer_token_embd.weight\");\n"
      "            layer.engram_embd = table ? table : create_tensor_from_meta(ctx_input, \"per_layer_token_embd.weight\");\n"
      "            layer.engram_wkv  = create_tensor_from_meta(ctx_split, format(\"blk.%d.engram_kv.weight\", i));\n"
      "            layer.engram_k    = create_tensor_from_meta(ctx_split, format(\"blk.%d.engram_gate.weight\", i));\n"
      "            layer.engram_q    = nullptr;\n"
      "        }\n")

# the table is a sparse lookup: keep it out of the bulk prefetch like engram_embd
patch("src/llama-model-loader.cpp",
      "        if (name.find(\"engram_embd.weight\") == std::string::npos) {\n",
      "        if (name.find(\"engram_embd.weight\") == std::string::npos && name != \"per_layer_token_embd.weight\") {\n")

# --- graph: the query side may have no weight ------------------------------------------------
G = "src/graphs/build_deepseek4.cpp"
patch(G,
      "        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);\n"
      "        t = ggml_mul(ctx0, t, w);\n",
      "        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);\n"
      "        if (w) t = ggml_mul(ctx0, t, w);\n")
patch(G,
      "    ggml_tensor * q = grouped_norm(ggml_cont(ctx0, x), as_f32(layer.engram_q, \"engram_q\"));\n",
      "    ggml_tensor * q = grouped_norm(ggml_cont(ctx0, x), layer.engram_q ? as_f32(layer.engram_q, \"engram_q\") : nullptr);\n")

print("deepseek41 mainline-GGUF patch applied")
