import torch

y = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
print(y)