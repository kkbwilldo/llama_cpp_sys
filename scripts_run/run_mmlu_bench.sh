#!/bin/bash

# MODEL_PATH="./outputs/checkpoints_42dot_LLM-PLM-1.3B-42dot-PLM-F16.gguf"
MODEL_PATH="./outputs/ggml-model-Q4_0.gguf"
DATA_PATH="./mmlu.bin"
CONTEXT_SIZE=2048

./llama.cpp/llama-perplexity \
  --multiple-choice \
  -m $MODEL_PATH \
  -bf $DATA_PATH \
  --ctx-size $CONTEXT_SIZE \
  |& tee ./outputs/mmlu_f16_pre_tok_gpt2.log
