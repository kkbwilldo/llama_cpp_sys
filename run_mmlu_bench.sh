#!/bin/bash

MODEL_PATH="./outputs/checkpoints_42dot_LLM-PLM-1.3B-42dot-PLM-F16.gguf"
DATA_PATH="./llama.cpp/test.bin"
CONTEXT_SIZE=1024

./llama.cpp/llama-perplexity \
  --multiple-choice \
  -m $MODEL_PATH \
  -bf $DATA_PATH \
  --ctx-size $CONTEXT_SIZE
