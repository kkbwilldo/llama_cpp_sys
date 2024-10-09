#!/bin/bash

MODEL_DIR="../checkpoints_42dot_LLM-PLM-1.3B"
OUTPUT_DIR="../outputs"
CONVERSION_TYPE="f16"

cd llama.cpp
python convert_hf_to_gguf_update.py $HUGGINGFACE_TOKEN
python convert_hf_to_gguf.py $MODEL_DIR --outfile $OUTPUT_DIR --outtype $CONVERSION_TYPE
