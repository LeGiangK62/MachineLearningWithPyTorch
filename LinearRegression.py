import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable

x_train = np.array([[147], [150], [153], [158], [163],
                    [165], [168], [170], [173], [175],
                    [178], [180], [183]], dtype=np.float32)

y_train = np.array([[ 49], [50], [51],  [54], [58],
                    [59], [60], [62], [63], [64],
                    [66], [67], [68]], dtype=np.float32)
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
print(x_train)

# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313]], dtype=np.float32)
#
# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65]], dtype=np.float32)

# x_train = np.array([[1.1], [2.2], [3.1], [4.1], [5.1],
#                     [6], [7], [8], [9], [10],
#                     [11], [12], [13], [14], [15]], dtype=np.float32)
#
#
# y_train = np.array([[ 2], [5], [6],  [8], [10.5],
#                     [11.9], [14], [15], [19], [19],
#                     [22], [24], [26], [28], [30]], dtype=np.float32)


x_norm = x_train.max()
y_norm = y_train.max()

x_train = x_train/x_train.max()
y_train = y_train/y_train.max()
# x_train = torch.rand(100,1) * np.pi * 2
# x_train = x_train.numpy()
# y_train = np.array(np.sin(x_train))

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


x_train = Variable(x_train)
y_train = Variable(y_train)


################################################################
# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 200)  # input and output is 1 dimension
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 300)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(300, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


model = LinearRegression()

print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

num_epochs = 1000

for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)

    # backward - we DID need this part to LEARN???
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print report after 20 epoch
    if (epoch+1) % 20 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy() * x_norm, y_train.numpy() * y_norm, 'ro', label='Original data')
plt.plot(x_train.numpy() * x_norm, predict * y_norm, '*', label='Fitting Line')

plt.legend()
plt.show()
