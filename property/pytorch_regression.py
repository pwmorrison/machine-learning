import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""
Linear regression, for modelling a very simple time series.
"""

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # create dummy data for training
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train += np.random.uniform(-1., 1, y_train.shape)
    y_train = y_train.reshape(-1, 1)

    model = RegressionModel(1, 1)
    model = model.to(device)

    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_iter = 500
    for iteration in range(n_iter):
        optimizer.zero_grad()

        input = Variable(torch.from_numpy(x_train).to(device))
        target = Variable(torch.from_numpy(y_train).to(device))

        predict = model(input)
        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()

        print(f'Iteration {iteration} loss {loss}')

    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    predict = model(input).cpu().detach().numpy()
    ax.scatter(x_train, y_train, label='train')
    ax.scatter(x_train, predict, label='predict')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
