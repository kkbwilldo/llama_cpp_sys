#!/bin/bash

MODEL_DIR="./outputs/base_fp16.gguf"
IMATRIX_DIR="./outputs/imatrix_groups_merged_enhanced_v3.dat"
NO_IMAT=(
  "Q4_0"
  "Q4_1"
)

# 양자화 옵션을 인자로 받지 않는 경우 기본값 설정
if [ -z "$1" ]; then
  QUANT_TYPE="Q4_0"
else
  QUANT_TYPE=$1
fi

need_imatrix() {
  for q in "${NO_IMAT[@]}"; do
    if [ "$q" == "$QUANT_TYPE" ]; then
      return 1
    fi
  done
  return 0
}

if need_imatrix ; then
  ./llama.cpp/llama-quantize --imatrix "$IMATRIX_DIR"  "$MODEL_DIR" "$QUANT_TYPE"
else
  ./llama.cpp/llama-quantize "$MODEL_DIR" "$QUANT_TYPE"
fi

