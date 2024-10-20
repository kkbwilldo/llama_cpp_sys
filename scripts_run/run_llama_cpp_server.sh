#!/bin/bash

MODEL_DIR=./outputs/base_fp16_fa_build.gguf
# MODEL_DIR=./outputs/ggml-model-Q4_K_S_fa_build.gguf
CONTEXT_SIZE=26880 # (2560+768+32)*8
N_GPU_LAYERS=30
N_PARALLEL=8

./llama.cpp/llama-server -fa -ngl $N_GPU_LAYERS -np $N_PARALLEL -m $MODEL_DIR --ctx-size $CONTEXT_SIZE
# sudo nsys profile --output llama_profile_3 ./llama.cpp/llama-server -fa -ngl $N_GPU_LAYERS -np $N_PARALLEL -m $MODEL_DIR --ctx-size $CONTEXT_SIZE
