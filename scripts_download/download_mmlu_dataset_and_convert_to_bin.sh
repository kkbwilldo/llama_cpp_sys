#! /bin/bash

g++ -o convert_to_bin to_bin.cpp
python download_dataset.py
./convert_to_bin mmlu.json mmlu.bin

