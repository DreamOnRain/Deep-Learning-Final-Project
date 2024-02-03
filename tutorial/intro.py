import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.__version__

data = ([1,2],[3,4])
x_data = torch.tensor(data)
print(x_data)

x= torch.Tensor([5.3, 2.1, -3.1])
print(x)

A = torch.rand(10, 2)
print(A)

A = torch.zeros(10, 2)
print(A)

A = torch.rand(3, 2, 4, 8)
print(A)

print(A.dim())

print(A.size())

print(A.size(2))

print(A.shape)

print(torch.FloatTensor(2,3))

print(torch.DoubleTensor(2,3))

print(torch.HalfTensor(2,3))

tensor = torch.randn(2,2)
long_tensor = tensor.long()
half_tensor = tensor.half()
int_tensor = tensor.int()
print(int_tensor)

# x1 = torch.tensor(1.0)
# x2 = torch.tensor(1.0, device='cuda:0')
# x3 = torch.tensor(1.0, requires_grad=True).cuda()
# print('x1:',x1)
# print('x2:',x2)
# print('x3:',x3)

x = torch.arange(10)
print(x)

print(x.view(2,5))

print(x.view(5,2))

print(x)

y = x.view(5,2)

a = torch.arange(4)
print(torch.reshape(a, (2,2)))

b = torch.tensor([[0,1],[2,3]])
print(torch.reshape(b, (-1,4)))

a= torch.randn(1,2,3,4)
print(a)

b = a.transpose(1,2)
print(b)

c = a.view(1,3,2,4)
print(c)

print(torch.equal(b,c))

class two_layer_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(two_layer_net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        p = F.softmax(x, dim=0)

        return p

net = two_layer_net(2,5,3)
print(net)

x = torch.Tensor([1,1])
print(x)

p = net.forward(x)
print(p)

p = net(x)
print(p)

print(net.layer1)
print(net.layer1.weight)
print(net.layer1.bias)
list_of_param = list(net.parameters())
print(list_of_param)