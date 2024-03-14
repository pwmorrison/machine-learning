import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import Dataset, DataLoader

from dataset import PixelDataset

# TODO: Consider doing the rolling standardisation within the network.
# TODO: Test with some synthetic data where we know the ground truth. Some kind of small random walk, with some probabilities of large jumps.
# TODO: Maybe weight the training data to locations where the series actually changes.

# TODO: Try a fixed standardisation/normalisation, which may avoid any issue with loss being dependent on how each sequence is normalised.


assert pyro.__version__.startswith('1.8.1')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = MNIST(root=root, train=True, transform=trans,
                      download=download)
    test_set = MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        #loc_img = self.sigmoid(self.fc21(hidden))
        loc_img = self.fc21(hidden)
        return loc_img

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.input_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class DecoderConv(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, stride=1)
        self.conv21 = nn.Conv1d(1, 1, kernel_size=3, stride=1)

        self.softplus = nn.Softplus()

    def forward(self, z):
        # Add channels dimension.
        z = z[:, None, :]

        hidden = self.softplus(self.conv1(z))
        loc_img = self.conv21(hidden)

        # Remove channels dimension.
        loc_img = loc_img[:, 0, :]

        return loc_img


class DecoderDeconv(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=1)
        self.conv21 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=1)

        self.softplus = nn.Softplus()

    def forward(self, z):
        # Add channels dimension.
        z = z[:, None, :]

        hidden = self.softplus(self.conv1(z))
        loc_img = self.conv21(hidden)

        # Remove channels dimension.
        loc_img = loc_img[:, 0, :]

        return loc_img



class EncoderConv(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        # setup the three linear transformations used
        self.conv11 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
        self.conv12 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
        self.conv13 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
        self.conv21 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, z_dim)
        # self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, self.input_dim)
        # then compute the hidden units
        hidden1 = self.softplus(self.conv11(x))
        hidden2 = self.softplus(self.conv12(hidden1))
        hidden3 = self.softplus(self.conv13(hidden2))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.conv21(hidden3)
        z_scale = torch.exp(self.conv22(hidden3))

        # Remove channels dimension.
        z_loc = z_loc[:, 0, :]
        z_scale = z_scale[:, 0, :]

        return z_loc, z_scale


class DecoderNull(nn.Module):
    """
    Decoder that simply passes the latent straight through.
    This means that the observed RV is effectively sampled from a normal dist (the latent), which may not be ideal.
    """

    def __init__(self):
        super().__init__()
        # self.conv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=1)
        # self.conv21 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=1)

        # self.softplus = nn.Softplus()

    def forward(self, z):
        # # Add channels dimension.
        # z = z[:, None, :]
        #
        # hidden = self.softplus(self.conv1(z))
        # loc_img = self.conv21(hidden)
        #
        # # Remove channels dimension.
        # loc_img = loc_img[:, 0, :]

        return z#loc_img


# # define the model p(x|z)p(z)
# def model(self, x):
#     # register PyTorch module `decoder` with Pyro
#     pyro.module("decoder", self.decoder)
#     with pyro.plate("data", x.shape[0]):
#         # setup hyperparameters for prior p(z)
#         z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
#         z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
#         # sample from prior (value will be sampled by guide when computing the ELBO)
#         z_scale = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#         # decode the latent code z
#         loc_img = self.decoder(z)
#         # score against actual images
#         pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, input_dim=784, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, y):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            #pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.input_dim))
            pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(1), obs=y.reshape(-1, self.input_dim))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, y):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


class VAEConv(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    # TODO: Try to derive the z-dim from the input dim.
    def __init__(self, input_dim, output_dim, z_dim=96, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = EncoderConv(input_dim, z_dim, hidden_dim)
        #self.decoder = DecoderDeconv(z_dim, hidden_dim, input_dim)
        self.decoder = DecoderConv(z_dim, hidden_dim, input_dim)
        #self.decoder = DecoderNull()

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, y):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            if 1:
                z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
                z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
                # sample from prior (value will be sampled by guide when computing the ELBO)
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            else:
                # Using a uniform latent (with normal in guide) results in inf loss.
                z1_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim))) - 1
                z2_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim))) + 1
                z = pyro.sample("latent", dist.Uniform(z1_loc, z2_loc).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            #pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.input_dim))
            # TODO: The std. dev. is being applied to normalised data. Try to derive it from an external NETD value.
            pyro.sample("obs", dist.Normal(loc_img, 0.7).to_event(1), obs=y.reshape(-1, self.output_dim))  # 0.35

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, y):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


# # define the guide (i.e. variational distribution) q(z|x)
# def guide(self, x):
#     # register PyTorch module `encoder` with Pyro
#     pyro.module("encoder", self.encoder)
#     with pyro.plate("data", x.shape[0]):
#         # use the encoder to get the parameters used to define q(z|x)
#         z_loc, z_scale = self.encoder(x)
#         # sample the latent code z
#         pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    #for x, _ in train_loader:
    for batch in train_loader:
        x = batch['x_standardised']
        y = batch['y_standardised']
        x = x.float()
        y = y.float()
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, y)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    #for x, _ in test_loader:
    for batch in test_loader:
        x = batch['x_standardised']
        y = batch['y_standardised']
        x = x.float()
        y = y.float()
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, y)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def test(vae, loader, use_cuda=False, out_padding=0):
    for batch in loader:
        x = batch['x_standardised']
        x = x.float()
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        y = vae.reconstruct_img(x)
        print(y)

        x = x.cpu().numpy()
        y = y.cpu().detach().numpy()

        x_indices = range(x.shape[-1])
        y_indices = range(out_padding, y.shape[-1] + out_padding)

        n = 1
        fig, axes = plt.subplots(1, n, squeeze=False)
        for i in range(n):#x.shape[0]):
            ax = axes[0, i]
            ax.plot(x_indices, x[i][0], label=f'input_{i}')  # Plot the first and only feature.
            ax.plot(y_indices, y[i], label=f'predict_{i}')
        ax.legend()

        break


def test_sequence(vae, dataset, pix_y, pix_x, num_frames, use_cuda=False, out_padding=0):
    """Tests a trained VAE using data from a single pixel.
    """
    x = dataset.x[:num_frames, pix_y, pix_x]
    x_standardised = dataset.x_standardised[:num_frames, pix_y, pix_x]
    x_mean = dataset.x_mean[:num_frames, pix_y, pix_x]

    # TODO: Move this distinction into the dataset.
    if 1:
        x_std = dataset.x_std[:num_frames, pix_y, pix_x]
        y_std = x_std[out_padding: -out_padding]
    else:
        x_std = dataset.x_std_mode[pix_y, pix_x]
        y_std = x_std

    # Form the data for input to the model.
    x_input = x_standardised[None, None, :]  # Add batch and feature dimensions.
    x_input = torch.from_numpy(x_input)
    x_input = x_input.float()
    if use_cuda:
        x_input = x_input.cuda()

    # Use the model to perform the prediction.
    y_pred = vae.reconstruct_img(x_input)
    y_pred = y_pred.cpu().detach().numpy()

    # Unstandardise the model input and predictions.
    x_unstandardised = x_standardised * x_std + x_mean
    y_unstandardised = y_pred[0] * y_std + x_mean[out_padding: -out_padding]

    # Plot
    x_indices = range(x.shape[-1])
    y_indices = range(out_padding, y_pred.shape[-1] + out_padding)
    n = 1
    fig, axes = plt.subplots(2, n, squeeze=False)
    for i in range(n):  # x.shape[0]):
        ax = axes[0, i]
        ax.plot(x_indices, x_standardised, label=f'input_{i}')  # Plot the first and only feature.
        ax.plot(y_indices, y_pred[i], label=f'predict_{i}')
    ax.set_title('Standardised input and prediction')
    ax.legend()
    for i in range(n):  # x.shape[0]):
        ax = axes[1, i]
        ax.plot(x_indices, x_unstandardised, label=f'input_{i}')  # Plot the first and only feature.
        ax.plot(y_indices, y_unstandardised, label=f'predict_{i}')
    ax.set_title('Unstandardised input and unstandardised prediction')
    ax.legend()



def main():
    # Run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = True

    # Run only for a single iteration for testing
    NUM_EPOCHS = 1000# if smoke_test else 100
    TEST_FREQUENCY = 5

    filename = r'C:\Users\pwmor\data\140d\5229\20220824\data.h5'
    series_length = 100
    standardise = False
    rolling_standardise = True
    rolling_standardise_window = 21
    difference = False
    out_padding = 6

    #train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)
    train_dataset = PixelDataset(
        filename,
        series_length,
        standardise=standardise,
        rolling_standardise=rolling_standardise,
        rolling_standardise_window=rolling_standardise_window,
        difference=difference,
        out_padding=out_padding
    )
    test_dataset = train_dataset
    # test_dataset = PixelDataset(
    #     filename,
    #     series_length,
    #     standardise=standardise,
    #     rolling_standardise=rolling_standardise,
    #     rolling_standardise_window=rolling_standardise_window,
    #     difference=difference,
    #     out_padding=out_padding
    # )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    input_dim = series_length
    output_dim = series_length - 2 * out_padding

    # clear param store
    pyro.clear_param_store()

    # setup the VAE
    vae = VAEConv(use_cuda=USE_CUDA, input_dim=input_dim, z_dim=input_dim - 8, output_dim=output_dim)

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    test(vae, train_loader, use_cuda=USE_CUDA, out_padding=out_padding)

    # Test a complete pixel sequence.
    pix_y = 15
    pix_x = 15
    num_frames = 500
    pixel_data = test_dataset.get_data(standardised=True)
    #x = test_dataset.x[:num_frames, pix_y, pix_x]
    x_standardised = test_dataset.x_standardised[:num_frames, pix_y, pix_x]
    x_mean = test_dataset.x_mean[:num_frames, pix_y, pix_x]
    x_std = test_dataset.x_std[:num_frames, pix_y, pix_x]

    data = pixel_data[:, 15, 15]
    test_sequence(vae, test_dataset, pix_y, pix_x, num_frames, use_cuda=USE_CUDA, out_padding=out_padding)

    plt.show()


if __name__ == '__main__':
    main()
