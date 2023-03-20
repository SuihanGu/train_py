import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F;
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

image_size=28
num_classse=10
num_epochs=20
batch_size=64#一批次的大小，64张图片
train_dataset=dsets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
#dataset数据集         dataloader加载器           sampler采样器
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
indices=range(len(test_dataset))
indices_val=indices[:5000]
indices_test=indices[5000:]

sampler_val=torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test=torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=sampler_val)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=sampler_test)

idx = 10
muetimg=train_dataset[idx][0].numpy()
plt.imshow(muetimg[0,...])
plt.show()
print("标签是：",train_dataset[idx][0])