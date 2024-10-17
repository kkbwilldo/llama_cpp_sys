#!/bin/bash

MODEL_DIR="./outputs/base_fp16.gguf"
QUANT_TYPE="Q4_K_S"

./llama.cpp/llama-quantize $MODEL_DIR $QUANT_TYPE
