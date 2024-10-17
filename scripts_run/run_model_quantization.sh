#!/bin/bash

MODEL_DIR="./outputs/base_fp16.gguf"
QUANT_TYPE="Q4_K_S"
IMATRIX_DIR="./outputs/imatrix_groups_merged_enhanced_v3.dat"

./llama.cpp/llama-quantize --imatrix $IMATRIX  $MODEL_DIR $QUANT_TYPE
