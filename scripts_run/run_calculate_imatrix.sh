#!/bin/bash

./llama.cpp/llama-imatrix -f ./outputs/groups_merged-enhancedV3.txt -o ./outputs/imatrix_groups_merged_enhanced_v3.dat --process-output -m ./outputs/base_fp16.gguf
