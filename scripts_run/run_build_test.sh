#!/bin/bash                                                                                                                                                                                                                 echo "DELETE ALL GGUF FILES AND IMAT DAT"
cd /home/kbkim/outputs
rm *.gguf
rm *.dat                                                                                                                                                                                                                    BUILD="GGML_CUDA=1"                                                                                                                                                                                                         echo "BUILD OPTIONS: " "$BUILD"
cd /home/kbkim/llama.cpp
make clean
make -j 32 "$BUILD"                                                                                                                                                                                                         echo "HF -> GGUF"
cd /home/kbkim
./run_checkpoint_to_gguf_conversion.sh                                                                                                                                                                                      echo "CALCULATE IMATRIX"
./run_calculate_imatrix.sh
                                                                                                              echo "QUANTIZATION"
./run_model_quantization.sh                                                                                                                                                                                                 echo "MMLU BENCHMARK"
./run_mmlu_bench.sh
