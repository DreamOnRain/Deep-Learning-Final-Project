#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --job-name=cs5242_project
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiyuyu@comp.nus.edu.sg
#SBATCH --gpus=a100:1
#SBATCH --partition=standard

model_list=("DreamOnRain/mamba-1.4b-hf-quotes" "state-spaces/mamba-130m-hf" "openai-community/gpt2")
prompt="This is CS5242 Project, what should I do?"
model="EleutherAI/gpt-neo-2.7B"

#tasks="wikitext,lambada_standard,lambada_openai,arc_easy,arc_challenge,hellaswag,cola,openbookqa,mathqa,winogrande"
tasks="wikitext,lambada_standard,lambada_openai,arc_easy,arc_challenge"
#tasks="hellaswag,cola,openbookqa,mathqa,winogrande"

#python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-2.8b-hf --tasks "${tasks}" --device cuda --batch_size 64 --output_path mamba-2.8b-${tasks}.json

python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/gpt-neo-2.7B --tasks "${tasks}" --device cuda --batch_size 64 --output_path ${model}-${tasks}.json

#python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[0]}" --prompt "${prompt}" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

#python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[1]}" --prompt "${prompt}" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

#python finetune/finetune.py

