# CS5242 Project


## About

Our study focuses on Mamba, an emerging text generation model that integrates the strengths of the long-range perception capabilities and transformers to address the efficiency and effectiveness of sequence modeling. Mamba employs structured state space models (SSMs) to optimize prompt responses and enhance mathematical reasoning capabilities. Our experiments demonstrate that Mamba outperforms traditional transformer models in computational efficiency and reasoning tasks by effectively balancing correlation and causality within sequence data. We further enhance the Mamba's capabilities in mathematical reasoning by fine-tuning it. This successfully validates that fine-tuning methods based on Parameter Efficient Fine-Tuning (PEFT) can strengthen the model's performance in specific domains. The findings suggest that Mamba's selective model architecture, focusing on context-aware sequence modeling, can significantly improve performance in natural language processing tasks, particularly in mathematical reasoning and prompt optimization. This research advances our understanding of sequence model architectures and sets a foundation for future explorations into efficient and effective language modeling.

## Experiments

### lm-evaluation-harness
```bash
python evals/lm_harness_eval.py --model hf --model_args pretrained=DreamOnRain/mamba-1.4b-msmath --tasks "${tasks}" --device cuda --batch_size 64
```

### fine-tune
```bash
python finetune/finetune.py
python finetune/merge.py
```

### mathematical reasoning
```bash
python benchmarks/benchmark_generation_mamba_simple.py --model-name "${model_list[0]}" --prompt "${prompt}"
```

## Usage
1. Accessing the Compute Cluster
https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access
2. Download SoC VPN
3. Download a SSH client, VSCode or Git Bash
4. Log in ssh client: ssh jiyuyu@xlog1.comp.nus.edu.sg (soc account name and password)
5. git config --global user.name "github_name"
6. git config --global user.email "github_email"
7. git clone https://github.com/DreamOnRain/Deep-Learning-Final-Project.git
8. if github password is required, use token:
GitHub/Settings/Developer Settings/Personal access tokens/Fine-grained tokens/Generate new token
9. Installation
10. modify #SBATCH info in run.sh
11. sbatch run.sh


## Installation

Requirements:
- mamba-ssm
- PyTorch 1.12+
- CUDA 11.6+

## Reference

> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752


@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
