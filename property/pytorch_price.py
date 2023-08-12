import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from rbf import evaluate_rbf, plot_surface_points
from pytorch_rbf import Rbf1dModel, Rbf2dModel

#pd.set_option("display.max_rows", None, "display.max_columns", None)

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


def filter_date(df: pd.DataFrame, dt_min: datetime.datetime = None, dt_max: datetime.datetime = None):
    df = df.copy()
    df['contract_date_dt'] = pd.to_datetime(df['contract_date'], format='%Y%m%d')
    if dt_min is not None:
        df = df.loc[df['contract_date_dt'] >= dt_min]
    if dt_max is not None:
        df = df.loc[df['contract_date_dt'] <= dt_max]
    return df


def extract_inputs(df):
    # The time of the purchase.
    # Normalised days since the first date in the sequence.
    t = pd.to_datetime(df['contract_date'], format='%Y%m%d')
    t_min = t.min()
    t = t - t_min
    t = t.dt.days
    t = t.to_numpy()

    x = df['lat'].to_numpy()
    y = df['long'].to_numpy()
    d = df['purchase_price'].to_numpy()

    return t, x, y, d


def standardise_vals(v, v_mean, v_sd):
    v = (v - v_mean) / v_sd
    return v


def unstandardise_vals(v, v_mean, v_sd):
    v = v * v_sd + v_mean
    return v


def standardise_inputs(df):

    t, x, y, d = extract_inputs(df)

    # The time of the purchase.
    # Normalised days since the first date in the sequence.
    #t = pd.to_datetime(df['contract_date'], format='%Y%m%d')
    #t_min = t.min()
    #t = t - t_min
    #t = t.dt.days
    t = t.astype(float) / t.max()
    #t = t.to_numpy()

    # Standardise inputs and output.
    #x = df['lat'].to_numpy()
    #y = df['long'].to_numpy()
    #d = df['purchase_price'].to_numpy()

    x_mean = x.mean()
    x_sd = x.std()
    x = standardise_vals(x, x_mean, x_sd)
    #x = (x - x_mean) / x_sd
    y_mean = y.mean()
    y_sd = y.std()
    #y = (y - y_mean) / y_sd
    y = standardise_vals(y, y_mean, y_sd)
    d_mean = d.mean()
    d_sd = d.std()
    #d = (d - d_mean) / d_sd
    d = standardise_vals(d, d_mean, d_sd)

    return t, x, y, d, (x_mean, x_sd), (y_mean, y_sd), (d_mean, d_sd)


def train_model(t, x, y, d, device, xy_basis_fn, xy_epsilon, t_basis_fn, t_epsilon):

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

    return model


def test_model(model, t, x, y, d, device):
    # Create parameter tensor at the fixed locations.
    # This produces a Nx2 tensor.
    xy_param = np.asarray([x, y], dtype=np.float_)
    xy_param = torch.Tensor(xy_param).to(device).T

    t_param = np.asarray([t], dtype=np.float_)
    t_param = torch.Tensor(t_param).to(device).T

    t = torch.Tensor(t.reshape(-1, 1)).to(device)

    # Target values at each parameter location.
    target = torch.Tensor(d).to(device)

    predict = model(t, xy_param)

    loss_fn = nn.MSELoss()
    loss = loss_fn(predict, target)

    # Convert predictions to dollar values.

    print(loss)

    predict = predict.cpu().detach().numpy()

    return predict



def main_evaluate():
    """
    Main function to train the model on the start of the dataset, and evaluate on the end.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    xy_basis_fn = 'gaussian'
    xy_epsilon = 0.1
    t_basis_fn = 'gaussian'
    t_epsilon = 0.3

    train_end_df = datetime.datetime(year=2019, month=12, day=31)

    df = pd.read_csv(f'D:\data\property\ERSKINE PARK_properties_geocodes.csv')

    df_train = filter_date(df, dt_min=None, dt_max=train_end_df)
    df_test = filter_date(df, dt_min=train_end_df + datetime.timedelta(days=1), dt_max=train_end_df + datetime.timedelta(days=60))

    print(df_train.shape)
    print(df_test.shape)

    print(f'Train date range: {df_train["contract_date_dt"].min()} to {df_train["contract_date_dt"].max()}')
    print(f'Test date range: {df_test["contract_date_dt"].min()} to {df_test["contract_date_dt"].max()}')

    t, x, y, d, (x_mean, x_sd), (y_mean, y_sd), (d_mean, d_sd) = standardise_inputs(df_train)
    model = train_model(t, x, y, d, device, xy_basis_fn, xy_epsilon, t_basis_fn, t_epsilon)

    if 0:
        # Evaluate the learned TPS using numpy code.
        weights = model.tps_model.weights.detach().to('cpu').numpy()
        if 1:
            xi = np.arange(-3, 3, 0.05)
            yi = np.arange(-3, 3, 0.05)
        else:
            ti, xi, yi, di = standardise_inputs(df_test)
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

    if 1:
        if 1:
            t_test, x_test, y_test, d_test = extract_inputs(df_test)
            t_test = np.ones_like(t_test)
            x_test = standardise_vals(x_test, x_mean, x_sd)
            y_test = standardise_vals(y_test, y_mean, y_sd)
            d_test = standardise_vals(d_test, d_mean, d_sd)
        else:
            # Test using training data.
            t_test = t
            x_test = x
            y_test = y
            d_test = d

        predict = test_model(model, t_test, x_test, y_test, d_test, device)

        d_test_unstandardised = unstandardise_vals(d_test, d_mean, d_sd)
        predict_unstandardised = unstandardise_vals(predict, d_mean, d_sd)

        error = predict_unstandardised - d_test_unstandardised
        mae = np.mean(np.abs(error))
        #print(f'Predictions: {predict_unstandardised}')
        #print(f'GT: {d_test_unstandardised}')
        #print(f'Error: {error}')
        print(f'MAE: {mae}')

        # Check whether a test property is in the training dataset.
        is_x_in_train = np.isin(x_test, x)
        is_y_in_train = np.isin(y_test, y)
        is_in_train = np.logical_and(is_x_in_train, is_y_in_train)
        #print(f'Is in training set: {is_in_train}')

        df_results = pd.DataFrame(
            {
                "predictions": predict_unstandardised,
                "gt": d_test_unstandardised,
                "error": error,
                "is_in_training_set": is_in_train
            })
        print(df_results)

        #test_model(model, t, x, y, d, device)

    plt.show()


def main_properties():
    """
    Main function to simply run the model over all data and show some plots.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    xy_basis_fn = 'gaussian'
    xy_epsilon = 0.2
    t_basis_fn = 'gaussian'
    t_epsilon = 0.2

    df = pd.read_csv(f'D:\data\property\ERSKINE PARK_properties_geocodes.csv')
    # df = df.iloc[:100]
    t, x, y, d, (x_mean, x_sd), (y_mean, y_sd), (d_mean, d_sd) = standardise_inputs(df)

    model = train_model(t, x, y, d, device, xy_basis_fn, xy_epsilon, t_basis_fn, t_epsilon)

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
    #main_properties()
    main_evaluate()
