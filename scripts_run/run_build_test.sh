#!/bin/bash

# 빌드 옵션 조합
BUILD_OPTIONS=(
  "GGML_CUDA=1 GGML_CUDA_FA_ALL_QUANTS=1 GGML_CUDA_F16=1"
  "GGML_CUDA=1 GGML_CUDA_FA_ALL_QUANTS=1 GGML_CUDA_FORCE_MMQ=1"
  "GGML_CUDA=1 GGML_CUDA_FA_ALL_QUANTS=1 GGML_CUDA_F16=1 GGML_CUDA_FORCE_MMQ=1"
)

# 양자화 옵션
QUANT_TYPES=(
  "Q4_0"
  "Q4_1"
  "Q4_K_S"
  "Q4_K"
  "IQ4_XS"
  "IQ4_NL"
)

# gguf 파일 리스트
GGUF_FILES=(
  "base_fp16.gguf"
  "ggml-model-Q4_0.gguf"
  "ggml-model-Q4_1.gguf"
  "ggml-model-Q4_K_S.gguf"
  "ggml-model-Q4_K.gguf"
  "ggml-model-IQ4_XS.gguf"
  "ggml-model-IQ4_NL.gguf"
)
length=${#BUILD_OPTIONS[@]}


# 로그 디렉토리 생성
LOG_DIR="./outputs/build_test"

# 로그 디렉토리가 없는 경우 생성
if [ ! -d "$LOG_DIR" ]; then
    echo "로그 디렉토리가 존재하지 않습니다. 로그 디렉토리를 생성합니다: $LOG_DIR"
    mkdir -p "$LOG_DIR"
else
    echo "로그 디렉토리가 이미 존재합니다: $LOG_DIR"
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# 테스트 데이터 다운로드
# wget https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/mmlu-test.bin -O mmlu.bin
./run_download_mmlu_dataset_and_convert_to_bin.sh

# 빌드 옵션 조합에 따라 빌드 및 테스트 실행

for ((i=0; i<length; i++)); do
  BUILD="${BUILD_OPTIONS[i]}"

  # 기존 체크포인트 및 데이터 삭제
  echo "기존 GGUF 파일 및 DAT 파일 삭제"
  cd /home/kbkim/outputs
  rm *.gguf
  rm *.dat

  echo "빌드 조합: " "$BUILD"
  cd /home/kbkim/llama.cpp

  # 빌드 초기화 후 빌드
  make clean
  make -j 32 $(echo $BUILD)

  # HF 체크포인트를 GGUF로 변환
  echo "HF -> GGUF 변환"
  cd /home/kbkim
  ./run_checkpoint_to_gguf_conversion.sh

  # base_fp16을 기준으로 imatrix를 계산
  echo "IMATRIX 계산"
  ./run_calculate_imatrix.sh

  for QUANT in "${QUANT_TYPES[@]}"; do
    echo "양쟈화 타입: " "$QUANT"
    ./run_model_quantization.sh "$QUANT"
  done
  for GGUF in "${GGUF_FILES[@]}"; do
    echo "GGUF 파일: " "$GGUF"
    ./run_mmlu_bench.sh "$GGUF" "$i"
  done
done
