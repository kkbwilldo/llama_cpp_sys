#! /bin/bash

# 현재 스크립트의 경로를 기준으로 프로젝트 루트 디렉토리 설정
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

docker run \
  --gpus '"device=0"' \
  -it \
  --rm \
  --network=host \
  --name llama_cpp \
  --cap-add=SYS_ADMIN \
  -v $PROJECT_ROOT/checkpoints_42dot_LLM-PLM-1.3B:/home/kbkim/checkpoints_42dot_LLM-PLM-1.3B:ro \
  -v $PROJECT_ROOT/scripts_run/run_download_mmlu_dataset.sh:/home/kbkim/run_download_mmlu_dataset.sh \
  -v $PROJECT_ROOT/scripts_run/run_checkpoint_to_gguf_conversion.sh:/home/kbkim/run_checkpoint_to_gguf_conversion.sh \
  -v $PROJECT_ROOT/scripts_run/run_model_quantization.sh:/home/kbkim/run_model_quantization.sh \
  -v $PROJECT_ROOT/scripts_run/run_calculate_imatrix.sh:/home/kbkim/run_calculate_imatrix.sh \
  -v $PROJECT_ROOT/scripts_run/run_llama_cli.sh:/home/kbkim/run_llama_cli.sh \
  -v $PROJECT_ROOT/scripts_run/run_perplexity.sh:/home/kbkim/run_perplexity.sh \
  -v $PROJECT_ROOT/scripts_run/run_kl_divergence.sh:/home/kbkim/run_kl_divergence.sh \
  -v $PROJECT_ROOT/scripts_run/run_mmlu_bench.sh:/home/kbkim/run_mmlu_bench.sh \
  -v $PROJECT_ROOT/scripts_run/run_llama_cpp_server.sh:/home/kbkim/run_llama_cpp_server.sh \
  -v $PROJECT_ROOT/scripts_run/run_locust.sh:/home/kbkim/run_locust.sh \
  -v $PROJECT_ROOT/scripts_download/to_bin.cpp:/home/kbkim/to_bin.cpp \
  -v $PROJECT_ROOT/scripts_download/json.hpp:/home/kbkim/json.hpp \
  -v $PROJECT_ROOT/scripts_download/download_dataset.py:/home/kbkim/download_dataset.py \
  -v $PROJECT_ROOT/scripts_download/run_download_mmlu_dataset_and_convert_to_bin.sh:/home/kbkim/run_download_mmlu_dataset_and_convert_to_bin.sh \
  -v $PROJECT_ROOT/locustfile.py:/home/kbkim/locustfile.py \
  -v $PROJECT_ROOT/outputs:/home/kbkim/outputs \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  llama_cpp:pytorch-24.02

# --cap-add=SYS_ADMIN: for ncu
# --gpus all: use all gpus
# --gpus '"device=0"': use specific gpu
# --network=host: use host network (for locust)
