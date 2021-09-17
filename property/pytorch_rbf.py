import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

from rbf import evaluate_rbf, plot_surface_points

"""
Learns the weights of a TPS using a Pytorch model and backpropagation.

The (x, y) locations of the parameters are fixed when creating the model. The weights associated those locations
are the learned parameters.

The below code currently fits the weights using points at the same locations. This doesn't have to be the case - 
we can use different points during the optimisation.

The idea is that this model is fit to the properties in a suburb, where there is a weight over the location of each
property. Then, the weights are optimised iteratively using subsets of house sales. We can learn a similar 1D model for 
time, and sum the time prediction and the location prediction, to arrive at the final prediction for the property.
"""

class Rbf2dModel(nn.Module):
    def __init__(self, xy_param, basis_fn='gaussian', epsilon=None):
        super(Rbf2dModel, self).__init__()
        # Points used to fit the RBF. These locations are fixed.
        self.xy_param = xy_param
        self.epsilon = epsilon
        # A weight for each parameter point.
        self.weights = torch.nn.Parameter(torch.zeros(xy_param.shape[0]))

        if basis_fn == 'gaussian':
            self.basis_fn = self.gaussian_fn
        elif basis_fn == 'thin_plate':
            self.basis_fn = self.thin_plate_spline_fn

    def thin_plate_spline_fn(self, r):
        return torch.xlogy(r ** 2, r)  # xlogy => x * log(y). torch.xlogy

    def gaussian_fn(self, r):
        return torch.exp(-(r/self.epsilon)**2)

    def forward(self, xy):
        # Evaluate the TPS at the given coordinates.
        # Distance between each test point and each train point.  torch.cdist
        r = torch.cdist(xy, self.xy_param)
        # Weighted sum of the result of applying the TPS function to the distances.
        results = torch.matmul(self.basis_fn(r), self.weights)
        return results


class Rbf1dModel(nn.Module):
    def __init__(self, x_param, basis_fn='gaussian', epsilon=None):
        super(Rbf1dModel, self).__init__()
        # Points used to fit the RBF. These locations are fixed.
        self.x_param = x_param
        self.epsilon = epsilon
        # A weight for each parameter point.
        self.weights = torch.nn.Parameter(torch.zeros(x_param.shape[0]))

        if basis_fn == 'gaussian':
            self.basis_fn = self.gaussian_fn
        elif basis_fn == 'thin_plate':
            self.basis_fn = self.thin_plate_spline_fn

    def thin_plate_spline_fn(self, r):
        return torch.xlogy(r ** 2, r)  # xlogy => x * log(y). torch.xlogy

    def gaussian_fn(self, r):
        return torch.exp(-(r/self.epsilon)**2)

    def forward(self, x):
        # Evaluate the TPS at the given coordinates.
        # Distance between each test point and each train point.  torch.cdist
        r = torch.cdist(x, self.x_param)
        # Weighted sum of the result of applying the TPS function to the distances.
        results = torch.matmul(self.basis_fn(r), self.weights)
        return results


def main_properties():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    basis_fn = 'gaussian'  # gaussian, thin_plate
    epsilon = 0.1

    df = pd.read_csv(f'D:\data\property\ERSKINE PARK_properties_geocodes.csv')
    x = df['lat'].to_numpy()
    y = df['long'].to_numpy()
    d = df['purchase_price'].to_numpy()

    # Standardise inputs and output.
    x_mean = x.mean()
    x_sd = x.std()
    x = (x - x_mean) / x_sd
    y_mean = y.mean()
    y_sd = y.std()
    y = (y - y_mean) / y_sd
    d_mean = d.mean()
    d_sd = d.std()
    d = (d - d_mean) / d_sd

    # Create parameter tensor at the fixed locations.
    # This produces a Nx2 tensor.
    xy_param = np.asarray([x, y], dtype=np.float_)
    xy_param = torch.Tensor(xy_param).to(device).T

    # Target values at each parameter location.
    target = torch.Tensor(d).to(device)

    model = Rbf2dModel(xy_param, basis_fn, epsilon)
    model = model.to(device)
    predict = model(xy_param)

    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_iter = 1000
    for iteration in range(n_iter):
        optimizer.zero_grad()

        predict = model(xy_param)
        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()

        print(f'Iteration {iteration} loss {loss}')

    weights = model.weights.detach().to('cpu').numpy()

    # Evalute the learned TPS using numpy code.
    xi = np.arange(-3, 3, 0.05)
    yi = np.arange(-3, 3, 0.05)
    xi, yi = np.meshgrid(xi, yi)
    results = evaluate_rbf(xi, yi, weights, x, y, 'euclidean', basis_fn=basis_fn, epsilon=epsilon)
    plot_surface_points(xi, yi, results, x, y, d)
    plt.show()


def main_2d():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    basis_fn = 'gaussian'
    epsilon = 0.5

    x = [0.25, 0.75, 0.75, 0.25]
    y = [0.25, 0.25, 0.75, 0.75]
    d = [0.5, 0.2, 0.8, 0.1]

    # Create parameter tensor at the fixed locations.
    # This produces a Nx2 tensor.
    xy_param = np.asarray([x, y], dtype=np.float_)
    xy_param = torch.Tensor(xy_param).to(device).T

    # Target values at each parameter location.
    target = torch.Tensor(np.array([0.5, 0.2, 0.8, 0.1])).to(device)

    model = Rbf2dModel(xy_param, basis_fn, epsilon)
    model = model.to(device)
    predict = model(xy_param)

    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_iter = 1000
    for iteration in range(n_iter):
        optimizer.zero_grad()

        predict = model(xy_param)
        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()

        print(f'Iteration {iteration} loss {loss}')

    weights = model.weights.detach().to('cpu').numpy()

    # Evalute the learned TPS using numpy code.
    xi = np.arange(0, 1, 0.05)
    yi = np.arange(0, 1, 0.05)
    xi, yi = np.meshgrid(xi, yi)
    results = evaluate_rbf(xi, yi, weights, x, y, 'euclidean', basis_fn, epsilon)
    plot_surface_points(xi, yi, results, x, y, d)
    plt.show()


def main_1d():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    basis_fn = 'thin_plate'
    epsilon = 0.5

    x = np.linspace(0., 5., 20)
    y = np.sin(x)

    # Create parameter tensor at the fixed locations.
    # This produces a Nx2 tensor.
    x_param = np.asarray([x], dtype=np.float_)
    x_param = torch.Tensor(x_param).to(device).T

    model = Rbf1dModel(x_param, basis_fn, epsilon)
    model = model.to(device)
    y_predict = model(x_param)

    target = torch.Tensor(y).to(device)

    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_iter = 100
    for iteration in range(n_iter):
        optimizer.zero_grad()

        predict = model(x_param)
        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()

        print(f'Iteration {iteration} loss {loss}')

    y_predict = model(x_param)

    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    ax.scatter(x, y)
    ax.scatter(x, y_predict.detach().cpu().numpy())
    plt.show()


if __name__ == '__main__':
    #main_2d()
    main_1d()
    #main_properties()
