# NVIDIA PyTorch container를 베이스 이미지로 사용합니다.
FROM nvcr.io/nvidia/pytorch:24.02-py3

# root 및 kbkim 패스워드 설정
RUN useradd -m kbkim && echo 'kbkim:kbkim' | chpasswd && adduser kbkim sudo
RUN echo 'root:kbkim' | chpasswd

# 워킹 디렉토리를 설정합니다.
WORKDIR /home/kbkim

# vimrc를 복사하여 세팅합니다.
COPY vimrc ./.vimrc
COPY vimrc /root/.vimrc

# 패키지 목록을 업데이트하고 업그레이드합니다.
RUN apt update && apt upgrade -y

# sudo, git, tmux, gdb를 설치합니다.
RUN apt-get update && apt-get install -y sudo git tmux gdb

# llama.cpp를 클론합니다. 빌드 이슈로 인해 빌드는 컨테이너에서 진행합니다.
RUN git clone https://github.com/kkbwilldo/llama.cpp.git
WORKDIR /home/kbkim/llama.cpp
RUN git checkout kbkim_b3661
WORKDIR /home/kbkim/

# 필요한 라이브러리를 설치합니다.
RUN pip install sentencepiece transformers safetensors locust datasets
