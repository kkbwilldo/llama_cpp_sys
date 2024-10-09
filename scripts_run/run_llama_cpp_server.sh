#!/bin/bash

MODEL_DIR=./outputs/checkpoints_42dot_LLM-PLM-1.3B-42dot-PLM-F16.gguf
CONTEXT_SIZE=2560

./llama.cpp/llama-server -m $MODEL_DIR --ctx-size $CONTEXT_SIZE
