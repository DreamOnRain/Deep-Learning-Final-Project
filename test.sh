#!/bin/bash

model_list=("state-spaces/mamba-130m" "openai-community/gpt2")
prompt="This is CS5242 Project, what should I do?"

#python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[0]}" --prompt "$prompt" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

python finetune/finetune.py
