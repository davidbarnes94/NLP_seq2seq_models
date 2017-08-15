import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

x = torch.Tensor(4, 2)
#print(x)

y = torch.rand(4, 2)
#print(y)

z = torch.add(x, y)
#print(z)

#print(z.size())

#print(z[:, 0])
#print(z[3, :])

I = torch.eye(5, 5)
#print(I)

I_np = I.numpy()
#print(I_np)

ones_np = np.ones(5)
#print(ones_np)

ones = torch.from_numpy(ones_np)
#print(ones)

##Create a Variable
a = Variable(torch.ones(2, 2), requires_grad=True)
#print(a)

b = a + 2
#print(b)
#b has a creator since it was created as a result of an operation

c = b * b * 3
#print(c)

d = c.mean()
#print(d)

#print(d.backward())

#print(a.grad)

##Create a neural network

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        #nn.Conv2d(num_input_channel, num_output_channel, kernel_size)

        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #linear apply a linear transformation to data: y = Ax + b
        #nn.Linear(size_of_each_input_sample,size_of_each_output sample)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
params = list(net.parameters())
#print(params[0])
print(len(params))

##IMAGE CLASSIFICATION


