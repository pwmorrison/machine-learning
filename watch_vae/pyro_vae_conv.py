# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import torch
import torch.nn as nn
import visdom
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

#from .unet_parts import *


"""
TODO:
* (done) Reconstruction function.
* (done) Function to get the embedding of an image.
* (done) Function to find close embeddings of a query embedding.
* Watch dataset.
* Choosing to freeze latents of the image spatially, and get similar watches by varying the other latents.
* Somehow varying other latents.
"""


class EncoderConv(nn.Module):
    def __init__(self, n_z_features, n_input_channels=3, n_downsamples=1):
        super(EncoderConv, self).__init__()

        self.conv1 = nn.Conv2d(n_input_channels, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.fc_loc = nn.Linear(14 * 14 * 8, z_dim)
        # self.fc_scale = nn.Linear(14 * 14 * 8, z_dim)
        self.conv_loc = nn.Conv2d(8, n_z_features, 3, padding=1)
        self.conv_scale = nn.Conv2d(8, n_z_features, 3, padding=1)

    # x = (batch_size, 3, 28, 28)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # (batch_size, 8, 14, 14)
        x = self.pool2(torch.relu(self.conv2(x)))  # (batch_size, 8, 7, 7)

        z_loc = self.conv_loc(x)  # (batch_size, 8, 7, 7)
        z_scale = torch.exp(self.conv_scale(x))  # (batch_size, 8, 7, 7)

        return z_loc, z_scale


class DecoderConv(nn.Module):
    def __init__(self, n_z_features, n_output_channels=3, n_upsamples=1):
        super(DecoderConv, self).__init__()

        self.conv1 = nn.Conv2d(n_z_features, 8, 3, padding=1)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv_out = nn.Conv2d(8, n_output_channels, 3, padding=1)

    # z_loc = (batch_size, 8, 7, 7)
    def forward(self, z):
        x = self.up1(torch.relu(self.conv1(z)))  # (batch_size, 8, 14, 14)
        x = self.up2(torch.relu(self.conv2(x)))  # (batch_size, 8, 28, 28)
        # x = torch.sigmoid(self.conv2(x))

        # x = (batch_size, 3, 28, 28)
        x = self.conv_out(x)

        # loc_img = torch.sigmoid(x)
        loc_img = x
        # loc_img = loc_img.reshape(loc_img.shape[0], -1)

        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, input_dim, n_downsamples, n_z_features=8, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = EncoderConv(n_z_features)
        self.decoder = DecoderConv(n_z_features)

        # Record the dimensions of the z-latent.
        z_spatial_dim = input_dim
        for _ in range(n_downsamples):
            z_spatial_dim /= 2
        self.z_spatial_dim = int(z_spatial_dim)
        self.z_feature_dim = n_z_features

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        #self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_loc = torch.zeros(x.shape[0], self.z_feature_dim, self.z_spatial_dim, self.z_spatial_dim,
                                dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_feature_dim, self.z_spatial_dim, self.z_spatial_dim,
                                 dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            #z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            # pyro.sample(
            #     "obs",
            #     dist.Bernoulli(loc_img, validate_args=False).to_event(1),
            #     obs=x.reshape(-1, 784),
            # )
            pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(3), obs=x)
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            #pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    def encode_x(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()

        return z_loc, z_scale, z


def find_similar(vae, dataloader, cuda, mnist, input_scale):
    # Find the latent space vector for every example in the test set.
    x_all = []
    z_all = []
    x_reconst_all = []
    #filenames_all = []
    n_batches = 5
    batch_num = 0
    for batch in dataloader:
        # x, z, x_reconst = test_minibatch(dmm, test_batch, args, sample_z=True)
        #x = batch['series']
        x = batch[0]

        if mnist:
            # Create image shape.
            x = x.reshape(-1, 1, 28, 28)
            # Add more channels.
            x = x.repeat(1, 3, 1, 1)
            # Upscale.
            x = torch.repeat_interleave(x, input_scale, dim=2)
            x = torch.repeat_interleave(x, input_scale, dim=3)

        if cuda:
            x = x.cuda()
        x = x.float()
        x_reconst = vae.reconstruct_img(x)
        z_loc, z_scale, z = vae.encode_x(x)
        x = x.cpu().numpy()
        x_reconst = x_reconst.cpu().detach().numpy()
        z_loc = z_loc.cpu().detach().numpy()

        x_all.append(x)
        z_all.append(z_loc)
        x_reconst_all.append(x_reconst)
        #filenames_all.extend(batch['filename'])

        batch_num += 1
        if batch_num == n_batches:
            break

    x_all = np.concatenate(x_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    x_reconst_all = np.concatenate(x_reconst_all, axis=0)

    # Flatten z dimensions for the nearest neighbours.
    z_all = z_all.reshape(z_all.shape[0], -1)


    # Get the closest latent to the query.
    n_neighbours = 5
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1 + n_neighbours, algorithm='ball_tree').fit(z_all)
    distances, indices = nbrs.kneighbors(z_all)

    #query_indices = [0, 1]
    #query_indices = np.random.randint(0, len(filenames_all), size=10)
    n_queries = 10
    query_indices = np.random.randint(0, z_all.shape[0], size=n_queries)

    fig, axes = plt.subplots(n_queries, 1 + n_neighbours, squeeze=False)

    for i, query_index in enumerate(query_indices):
        print(f'Query index {query_index}')
        # Skip the first closest index, since it is just the query index.
        closest_indices = indices[query_index][1:]

        # Plot the query.
        ax = axes[i, 0]
        x = x_all[query_index]
        #ax.plot(range(x_series.shape[0]), x_series, c='r')
        ax.imshow(x.transpose(1, 2, 0))
        #ax.set_title(filenames_all[query_index])
        ax.set_title(query_index)

        # ax.grid()
        for j in range(n_neighbours):
            ax = axes[i, j + 1]
            x = x_all[closest_indices[j]]
            # ax.plot(range(x_series.shape[0]), x_series, c='b')
            ax.imshow(x.transpose(1, 2, 0))
            #ax.set_title(filenames_all[closest_indices[i]])
            ax.set_title(closest_indices[j])
            #ax.grid()
    plt.show()

    return


def main(args):
    # clear param store
    pyro.clear_param_store()

    # # setup MNIST data loaders
    # # train_loader, test_loader
    # train_loader, test_loader = setup_data_loaders(
    #     MNIST, use_cuda=args.cuda, batch_size=256
    # )

    n_downsamples = 2
    mnist = True

    if mnist:
        train_loader, test_loader = setup_data_loaders(MNIST, use_cuda=args.cuda, batch_size=256)
        input_scale = 1
        input_dim = 28 * input_scale
    else:
        pass
        # input_dim = 128
        # root_dir = '/Users/paul/data/20171106_subset/results_20171023'
        # dataset = create_dataset(root_dir, lambda x: True if x.endswith('final.jpg') else False, input_dim)
        # train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        # test_loader = train_loader

    # setup the VAE
    vae = VAE(input_dim, n_downsamples, n_z_features=8, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = {}
    test_elbo = {}
    # training loop
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch} of {args.num_epochs}')
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:

            if mnist:
                # Create image shape.
                x = x.reshape(-1, 1, 28, 28)
                # Add more channels.
                x = x.repeat(1, 3, 1, 1)
                # Upscale.
                x = torch.repeat_interleave(x, input_scale, dim=2)
                x = torch.repeat_interleave(x, input_scale, dim=3)

            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.0
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):

                if mnist:
                    # Create image shape.
                    x = x.reshape(-1, 1, 28, 28)
                    # Add more channels.
                    x = x.repeat(1, 3, 1, 1)
                    # Upscale.
                    x = torch.repeat_interleave(x, input_scale, dim=2)
                    x = torch.repeat_interleave(x, input_scale, dim=3)

                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if args.visdom_flag:
                        plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(
                                test_img.reshape(28, 28).detach().cpu().numpy(),
                                opts={"caption": "test image"},
                            )
                            vis.image(
                                reco_img.reshape(28, 28).detach().cpu().numpy(),
                                opts={"caption": "reconstructed image"},
                            )

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
            plot_llk(train_elbo, test_elbo)

        if epoch == args.tsne_iter:
            mnist_test_tsne(input_scale, vae=vae, test_loader=test_loader)

    find_similar(vae, train_loader, args.cuda, mnist, input_scale)

    return vae


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.6")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-tf",
        "--test-frequency",
        default=1,  # 5
        type=int,
        help="how often we evaluate the test set",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--jit", action="store_true", default=False, help="whether to use PyTorch jit"
    )
    parser.add_argument(
        "-visdom",
        "--visdom_flag",
        action="store_true",
        help="Whether plotting in visdom is desired",
    )
    parser.add_argument(
        "-i-tsne",
        "--tsne_iter",
        default=6,
        type=int,
        help="epoch when tsne visualization runs",
    )
    args = parser.parse_args()

    model = main(args)