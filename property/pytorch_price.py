import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

from rbf import evaluate_rbf, plot_surface_points
from pytorch_rbf import Rbf1dModel, Rbf2dModel

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


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


# class Rbf1dModel(nn.Module):
#     def __init__(self, x_param, epsilon=None):
#         super(Rbf1dModel, self).__init__()
#         # Points used to fit the RBF. These locations are fixed.
#         self.x_param = x_param
#         self.epsilon = epsilon
#         # A weight for each parameter point.
#         self.weights = torch.nn.Parameter(torch.zeros(x_param.shape[0]))
#
#         self.basis_fn = self.gaussian_fn
#
#     def thin_plate_spline_fn(self, r):
#         return torch.xlogy(r ** 2, r)  # xlogy => x * log(y). torch.xlogy
#
#     def gaussian_fn(self, r):
#         return torch.exp(-(r/self.epsilon)**2)
#
#     def forward(self, x):
#         # Evaluate the TPS at the given coordinates.
#         # Distance between each test point and each train point.  torch.cdist
#         r = torch.cdist(x, self.x_param)
#         # Weighted sum of the result of applying the TPS function to the distances.
#         results = torch.matmul(self.basis_fn(r), self.weights)
#         return results


# class Rbf2dModel(nn.Module):
#     def __init__(self, xy_param, epsilon):
#         super(Rbf2dModel, self).__init__()
#         # Points used to fit the RBF. These locations are fixed.
#         self.xy_param = xy_param
#         self.epsilon = epsilon
#         # A weight for each parameter point.
#         self.weights = torch.nn.Parameter(torch.zeros(xy_param.shape[0]))
#         self.basis_fn = self.gaussian_fn
#
#     def thin_plate_spline_fn(self, r):
#         return torch.xlogy(r ** 2, r)  # xlogy => x * log(y). torch.xlogy
#
#     def gaussian_fn(self, r):
#         return torch.exp(-(r/self.epsilon)**2)
#
#     def forward(self, xy):
#         # Evaluate the TPS at the given coordinates.
#         # Distance between each test point and each train point.  torch.cdist
#         r = torch.cdist(xy, self.xy_param)
#         # Weighted sum of the result of applying the TPS function to the distances.
#         results = torch.matmul(self.basis_fn(r), self.weights)
#         return results


class PriceModel(nn.Module):
    def __init__(self, xy_param, xy_basis_fn, xy_epsilon, t_param=None, t_basis_fn=None, t_epsilon=None):
        super(PriceModel, self).__init__()
        if t_param is None:
            # Use a regression time model.
            self.time_model = RegressionModel(1, 1)
        else:
            # Use a RBF time model.
            self.time_model = Rbf1dModel(t_param, t_basis_fn, t_epsilon)
        self.tps_model = Rbf2dModel(xy_param, xy_basis_fn, xy_epsilon)

    def forward(self, t, xy):
        t = self.time_model(t)
        d = self.tps_model(xy)
        out = t + d
        return out


def main_properties():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    xy_basis_fn = 'gaussian'
    xy_epsilon = 0.2
    t_basis_fn = 'gaussian'
    t_epsilon = 0.2

    df = pd.read_csv(f'D:\data\property\ERSKINE PARK_properties_geocodes.csv')

    #df = df.iloc[:100]

    x = df['lat'].to_numpy()
    y = df['long'].to_numpy()
    d = df['purchase_price'].to_numpy()

    # The time of the purchase.
    # Normalised days since the first date in the sequence.
    t = pd.to_datetime(df['contract_date'], format='%Y%m%d')
    t_min = t.min()
    t = t - t_min
    t = t.dt.days
    t = t.astype(float) / t.max()
    t = t.to_numpy()

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

    t_param = np.asarray([t], dtype=np.float_)
    t_param = torch.Tensor(t_param).to(device).T

    t = torch.Tensor(t.reshape(-1, 1)).to(device)

    # Target values at each parameter location.
    target = torch.Tensor(d).to(device)

    model = PriceModel(xy_param, xy_basis_fn, xy_epsilon, t_param, t_basis_fn, t_epsilon)
    model = model.to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_iter = 10000
    for iteration in range(n_iter):
        optimizer.zero_grad()

        predict = model(t, xy_param)
        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()

        print(f'Iteration {iteration} loss {loss}')

    if 1:
        # Evaluate the learned TPS using numpy code.
        weights = model.tps_model.weights.detach().to('cpu').numpy()
        xi = np.arange(-3, 3, 0.05)
        yi = np.arange(-3, 3, 0.05)
        xi, yi = np.meshgrid(xi, yi)
        results = evaluate_rbf(xi, yi, weights, x, y, 'euclidean', xy_basis_fn, xy_epsilon)
        plot_surface_points(xi, yi, results, x, y, d)

    if 1:
        # Visualise the time series model.
        t_test = np.arange(0, 1, 0.01)
        t_test = torch.Tensor(t_test.reshape(-1, 1)).to(device)
        y_pred = model.time_model(t_test)
        y_pred = y_pred.detach().to('cpu').numpy()
        fig, axes = plt.subplots(1, 1, squeeze=False)
        plt.scatter(t_test.to('cpu').numpy(), y_pred)

    plt.show()


if __name__ == '__main__':
    main_properties()
