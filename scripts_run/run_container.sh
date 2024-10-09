#! /bin/bash

# 현재 스크립트의 경로를 기준으로 프로젝트 루트 디렉토리 설정
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

docker run \
  --gpus all \
  -it \
  --rm \
  --network=host \
  --name llama_cpp \
  --cap-add=SYS_ADMIN \
  -v $PROJECT_ROOT/checkpoints_42dot_LLM-PLM-1.3B:/home/kbkim/checkpoints_42dot_LLM-PLM-1.3B:ro \
  -v $PROJECT_ROOT/scripts_run/run_download_mmlu_dataset.sh:/home/kbkim/run_download_mmlu_dataset.sh \
  -v $PROJECT_ROOT/scripts_run/run_mmlu_bench.sh:/home/kbkim/run_mmlu_bench.sh \
  -v $PROJECT_ROOT/scripts_run/run_llama_cpp_server.sh:/home/kbkim/run_llama_cpp_server.sh \
  -v $PROJECT_ROOT/scripts_run/run_locust.sh:/home/kbkim/run_locust.sh \
  -v $PROJECT_ROOT/scripts_run/run_checkpoint_to_gguf_conversion.sh:/home/kbkim/run_checkpoint_to_gguf_conversion.sh \
  -v $PROJECT_ROOT/locustfile.py:/home/kbkim/locustfile.py \
  -v $PROJECT_ROOT/outputs:/home/kbkim/outputs \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  llama_cpp:pytorch-24.02

# --cap-add=SYS_ADMIN: for ncu
# --gpus all: use all gpus
# --gpus '"device=0"': use specific gpu
# --network=host: use host network (for locust)
