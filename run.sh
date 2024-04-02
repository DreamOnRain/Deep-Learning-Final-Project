#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --job-name=cs5340_project
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiyuyu@comp.nus.edu.sg
#SBATCH --gpus=a100:1
#SBATCH --partition=medium

#python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 64

python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-2.8b" --prompt "This is my cs5242 project" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

#python benchmarks/benchmark_generation_mamba_simple.py --model-name "EleutherAI/pythia-2.8b" --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
