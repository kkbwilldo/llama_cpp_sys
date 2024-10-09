#! /bin/bash
cd ../
docker build --no-cache -t llama_cpp:pytorch-24.02 -f Dockerfile .

