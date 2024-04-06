# CS5242 Project


## About

Mamba is a new state space model architecture, we try to use it in prompt generation.
Our work is:
1. generate completions of a user-specified prompt,
2. benchmark the inference speed of this generation,
3. compare mamba with former architectures like transformer in generation quantity, number of parameters
4. etc.


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

![Mamba](assets/selection.png "Selective State Space")
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752


Original github: https://github.com/Raul-A-P/mamba_Fin_ChapGPT


@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}