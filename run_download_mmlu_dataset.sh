#!/bin/bash

# llama.cpp의 perplexity.cpp에서 사용하는 데이터셋을 다운로드 받는 스크립트입니다.
# https://github.com/ggerganov/llama.cpp/blob/dca1d4b58a7f1acf1bd253be84e50d6367f492fd/examples/perplexity/perplexity.cpp#L1398

wget https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/mmlu-test.bin -O test.bin
wget https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/arc-challenge-validation.bin -O arc_challenge.bin

