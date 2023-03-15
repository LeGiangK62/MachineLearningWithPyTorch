import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from numpy import pi

# Model param

num_epoch = 100
learning_rate = 1e-2

x_train = torch.rand(100,1) * pi * 4
x_train = x_train.numpy()
y_train = np.array(np.sin(x_train))

print(x_train.shape)
print(y_train.shape)

# Data Pre-processing

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.legend()

# Logistic Regression


class LogisticRegressionBase(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegressionBase, self).__init__()
        self.logistic = nn.Linear(in_dim, out_dim)

    def forward(self, input_data):
        input_data = self.logistic(input_data)
        output_data = input_data
        return output_data


model = LogisticRegressionBase(1, 1)
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    print('*' * 10)
    print(f'epoch {epoch + 1}')

    print(f'Finish {epoch + 1} epoch, Loss: {  / i:.6f}, Acc: {running_acc / i:.6f}')




