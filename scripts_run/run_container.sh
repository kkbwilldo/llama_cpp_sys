#! /bin/bash

docker run \
  --gpus all \
  -it \
  --rm \
  --network=host \
  --name llama_cpp \
  --cap-add=SYS_ADMIN \
  -v ../checkpoints_42dot_LLM-PLM-1.3B:/home/kbkim/checkpoints_42dot_LLM-PLM-1.3B:ro \
  -v ./run_download_mmlu_dataset.sh:/home/kbkim/run_download_mmlu_dataset.sh \
  -v ./run_mmlu_bench.sh:/home/kbkim/run_mmlu_bench.sh \
  -v ./run_llama_cpp_server.sh:/home/kbkim/run_llama_cpp_server.sh \
  -v ./run_locust.sh:/home/kbkim/run_locust.sh \
  -v ./run_checkpoint_to_gguf_conversion.sh:/home/kbkim/run_checkpoint_to_gguf_conversion.sh \
  -v ../locustfile.py:/home/kbkim/locustfile.py \
  -v ../outputs:/home/kbkim/outputs \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  llama_cpp:pytorch-24.02

# --cap-add=SYS_ADMIN: for ncu
# --gpus all: use all gpus
# --gpus '"device=0"': use specific gpu
# --network=host: use host network (for locust)
