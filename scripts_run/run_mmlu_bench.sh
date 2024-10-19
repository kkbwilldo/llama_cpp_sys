#!/bin/bash

if [ -z "$1" ]; then
  MODEL_PATH="./outputs/base_fp16.gguf"
else
  MODEL_PATH="./outputs/$1"
fi


if [ -z "$2" ]; then
  BUILD_NUM="1"
else
  BUILD_NUM="$2"
fi


DATA_PATH="./mmlu.bin"
CONTEXT_SIZE=2048

# MODEL_PATH에서 파일명만 추출
MODEL_FILENAME=$(basename "$MODEL_PATH")

./llama.cpp/llama-perplexity \
  --multiple-choice \
  -m $MODEL_PATH \
  -bf $DATA_PATH \
  --ctx-size $CONTEXT_SIZE \
  |& tee "./outputs/build_test/build_${BUILD_NUM}_inference_test_${MODEL_FILENAME}.log"
