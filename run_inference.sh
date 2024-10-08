#!/bin/bash

MODEL_PATH="./outputs/checkpoints_42dot_LLM-PLM-1.3B-42dot-PLM-F16.gguf"
PROMPT="I believe the meaning of life is"
NUM_NEW_TOKENS=128

./llama.cpp/llama-cli \
  -m $MODEL_PATH \
  -p $PROMPT \
  -n $NUM_NEW_TOKENS
