#!/bin/bash

#SBATCH --time=20:30:00
#SBATCH --job-name=cs5242_project
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiyuyu@comp.nus.edu.sg
#SBATCH --gpus=a100-80:1
#SBATCH --partition=long

model_list=("DreamOnRain/mamba-1.4b-msmath" "state-spaces/mamba-1.4b-hf" "EleutherAI/pythia-1.4b")
prompt="This is my CS5242 Project, I want to do some math problems. For the past n days, the average daily production at a company was a certain number of units. If today's production of 90 units raises the average to 58 units per day, and the value of n is 4, what was the initial average daily production?"
tasks="wikitext,lambada_standard,lambada_openai,arc_easy,arc_challenge,hellaswag,cola,openbookqa,mathqa,winogrande"

python evals/lm_harness_eval.py --model hf --model_args pretrained=DreamOnRain/mamba-1.4b-msmath --tasks "${tasks}" --device cuda --batch_size 64

python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-1.4b-hf --tasks "${tasks}" --device cuda --batch_size 64

python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-1.4b --tasks "${tasks}" --device cuda --batch_size 64

python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[0]}" --prompt "${prompt}"

python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[1]}" --prompt "${prompt}"

python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[2]}" --prompt "${prompt}"

