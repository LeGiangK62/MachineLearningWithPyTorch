import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from numpy import pi

# Model param

num_epoch = 10000
learning_rate = 5e-3

x_train = torch.rand(100,1) * pi * 2
x_train = x_train.numpy()
y_train = np.array(np.sin(x_train))

# Data Pre-processing
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# fig = plt.figure(figsize=(10, 5))
# plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
# plt.legend()
# plt.show()

# Logistic Regression


class LogisticRegressionBase(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegressionBase, self).__init__()
        self.layer1 = nn.Linear(in_dim, 100)
        self.activFunc1 = nn.Sigmoid()
        self.activFunc2 = nn.Tanh()
        self.layer2 = nn.Linear(100, out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.activFunc1(out)
        out = self.layer2(out)
        out = self.activFunc2(out)
        return out


model = LogisticRegressionBase(1, 1)
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    in_data = x_train
    target_data = y_train
    if use_gpu:
        in_data = in_data.cuda()
        out_data = out_data.cuda()

    # Train model
    model.train()
    out_data = model(in_data)
    loss = criterion(out_data, target_data)
    print(out_data)
    if (epoch+1) % 20 == 0:
        print(f'Epoch[{epoch+1}/{num_epoch}], loss: {loss.item():.6f}')

    # BackPropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
# x_test = torch.rand(100,1) * pi * 4
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, 'b*', label='Fit data')
plt.legend()
plt.show()

