import torch
import numpy as np
from torch.autograd import Variable#自动微分变量包
import matplotlib.pyplot as plt
x = Variable(torch.linspace(0,100).type(torch.FloatTensor))
rand = Variable(torch.rand(100))*10
y = x+rand
#画图
x_train = x[:-10]
x_test = x[-10:]
y_train = y[:-10]
y_test = y[-10:]
plt.figure(figsize=(10,8))
plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
a = Variable(torch.rand(1),requires_grad = True)
b = Variable(torch.rand(1),requires_grad = True)
learning_rate = 0.0001
for i in range(1000):
    predictions = a.expand_as(x_train)*x_train+b.expand_as(x_train)
    loss = torch.mean((predictions-y_train)**2)
    print('loss:',loss)
    loss.backward()
    a.data.add_(- learning_rate*a.grad.data)
    b.data.add_(- learning_rate*b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()
x_data = x_train.data.numpy()
plt.figure(figsize=(10,8))
xplot, = plt.plot(x_data,y_train.data.numpy(),"o")
yplot, =plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy())
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
predictions = a.expand_as(x_test)*x_test+b.expand_as(x_test)
print(predictions)
x_data = x_train.data.numpy()
x_pred = y_test.data.numpy()
plt.figure(figsize=(10,8))
plt.plot(x_data,y_train.data.numpy(),"o")
plt.plot(x_pred,y_test.data.numpy(),"s")
x_data = np.r_[x_data,x_test.data.numpy()]
plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy())
plt.plot(x_pred,a.data.numpy()*x_pred+b.data.numpy(),"o")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
