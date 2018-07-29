import os
import numpy as np
import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# for CI testing
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

N = 100  # size of toy data


def build_linear_dataset(N, p=1, noise_std=0.01):
    X = np.random.rand(N, p)
    # w = 3
    w = 3 * np.ones(p)
    # b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data


class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)


def main_pytorch():

    regression_model = RegressionModel(1)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
    num_iterations = 1000 if not smoke_test else 2

    data = build_linear_dataset(N)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print("%s: %.3f" % (name, param.data.numpy()))

    print('Finished')


regression_model = RegressionModel(1)


def model(data):
    # Create unit normal priors over the parameters
    loc, scale = torch.zeros(1, 1), 10 * torch.ones(1, 1)
    bias_loc, bias_scale = torch.zeros(1), 10 * torch.ones(1)
    w_prior = Normal(loc, scale).independent(1)
    b_prior = Normal(bias_loc, bias_scale).independent(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.iarange("map", N):
        x_data = data[:, :-1]
        y_data = data[:, -1]

        # run the regressor forward conditioned on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, 0.1 * torch.ones(data.size(0))),
                    obs=y_data)


softplus = torch.nn.Softplus()

def guide(data):
    # define our variational parameters
    w_loc = torch.randn(1, 1)
    # note that we initialize our scales to be pretty narrow
    w_log_sig = torch.tensor(-3.0 * torch.ones(1, 1) + 0.05 * torch.randn(1, 1))
    b_loc = torch.randn(1)
    b_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).independent(1)
    b_dist = Normal(mb_param, sb_param).independent(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overload the parameters in the module with random samples
    # from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor (which also samples w and b)
    return lifted_module()


def main_pyro():
    optim = Adam({"lr": 0.05})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    num_iterations = 1000 if not smoke_test else 2

    pyro.clear_param_store()
    data = build_linear_dataset(N)
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))

    for name in pyro.get_param_store().get_all_param_names():
        print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))

    X = np.linspace(6, 7, num=20)
    y = 3 * X + 1
    X, y = X.reshape((20, 1)), y.reshape((20, 1))
    x_data, y_data = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
    loss = nn.MSELoss()
    y_preds = torch.zeros(20, 1)
    for i in range(20):
        # guide does not require the data
        sampled_reg_model = guide(None)
        # run the regression model and add prediction to total
        y_preds = y_preds + sampled_reg_model(x_data)
    # take the average of the predictions
    y_preds = y_preds / 20
    print("Loss: ", loss(y_preds, y_data).item())

if __name__ == '__main__':
    # main_pytorch()
    main_pyro()
