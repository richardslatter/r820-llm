#!/bin/bash
# CUDA builds for the Quadro P2200 (Pascal, sm_61); nvcc 12.4 needs gcc <= 13 as host compiler
set -x
C="-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DLLAMA_CURL=OFF"
cd ~/src/ik_llama.cpp && cmake -B build-cuda $C -DGGML_IQK_AVX1_SIMDE=ON > ../ik-cuda-cmake.log 2>&1 && cmake --build build-cuda -j 32 > ../ik-cuda-build.log 2>&1; echo "ik rc=$?"
cd ~/src/llama.cpp && cmake -B build-cuda $C > ../ml-cuda-cmake.log 2>&1 && cmake --build build-cuda -j 32 > ../ml-cuda-build.log 2>&1; echo "ml rc=$?"
echo BUILD_DONE
