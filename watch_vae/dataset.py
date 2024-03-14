import torch
import h5py
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity

from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path
import os


# TODO: Normalise input to make training work.


class PixelDataset(Dataset):

    def __init__(self, h5_filename, series_length, standardise, rolling_standardise, rolling_standardise_window, difference, out_padding=0):

        self.h5_filename = h5_filename
        self.series_length = series_length
        self.standardise = standardise
        self.rolling_standardise = rolling_standardise
        self.rolling_standardise_window = rolling_standardise_window
        self.out_padding = out_padding

        # Read the pixel data.
        with h5py.File(h5_filename, 'r') as f:
            x = f['pixel_data'][()]
            temp_factor = f['metadata/temp_factor'][()][0]

        x = x.astype(float) / temp_factor

        if difference:
            # Compute differences from one pixel to the next.
            x = x[1:, ...] - x[:-1, ...]

        # Compute rolling mean and std dev.
        half_window = rolling_standardise_window // 2
        x_pix = []
        x_mean = []
        x_std = []
        for pix_y in range(x.shape[1]):
            row_pix = []
            row_mean = []
            row_std = []
            for pix_x in range(x.shape[2]):
                pix = x[:, pix_y, pix_x]
                pix_series = pd.Series(pix)
                pix_mean = pix_series.rolling(window=rolling_standardise_window, center=True).mean().to_numpy()
                pix_std = pix_series.rolling(window=rolling_standardise_window, center=True).std().to_numpy()

                # Remove nans at the start and end.
                pix = pix[half_window: -half_window]
                pix_mean = pix_mean[half_window: -half_window]
                pix_std = pix_std[half_window: -half_window]

                row_pix.append(pix)
                row_mean.append(pix_mean)
                row_std.append(pix_std)

            x_pix.append(row_pix)
            x_mean.append(row_mean)
            x_std.append(row_std)

        x_pix = np.array(x_pix)
        x_pix = x_pix.transpose((2, 0, 1))  # (n_frames, n_rows, n_cols)
        x_mean = np.array(x_mean)
        x_mean = x_mean.transpose((2, 0, 1))  # (n_frames, n_rows, n_cols)
        x_std = np.array(x_std)
        x_std = x_std.transpose((2, 0, 1))  # (n_frames, n_rows, n_cols)

        #x_std /= 5

        if 1:
            # Calculate the std mode at every pixel.
            self.x_std_mode = np.empty(x_std.shape[1:])
            for pix_y in range(x_std.shape[1]):
                for pix_x in range(x_std.shape[2]):
                    if 0:
                        pix_std = x_std[:1000, pix_y, pix_x]
                        pix_std_mode = calculate_std_mode(pix_std)
                        self.x_std_mode[pix_y, pix_x] = pix_std_mode
                        print(f'Pixel ({pix_x}, {pix_y}) std mode: {pix_std_mode}')
                    else:
                        x_std[:, pix_y, pix_x] = 0.1#pix_std_mode

            # TODO: Remove. Playing with the effect of this on the prediction results.
            #self.x_std_mode *= 4

            #x_std[:] =

        #self.x_standardised = (x_pix - x_mean) / self.x_std_mode
        self.x_standardised = (x_pix - x_mean) / x_std

        # self.pixel_data = self.pixel_data[half_window: -half_window]
        self.x = x_pix
        self.x_mean = x_mean
        self.x_std = x_std

        # rolling_sum = convolve1d(self.pixel_data, np.ones(rolling_standardise_window), axis=0)
        # rolling_mean = rolling_sum / rolling_standardise_window

        # Calcua


    def get_data(self, standardised=False):
        if standardised:
            return self.x_standardised
        else:
            return self.x

    def __len__(self):
        # Length of the dataset is the number of pixels.
        return self.x.shape[1] * self.x.shape[2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Index to pixel location.
        pix_y = idx // self.x.shape[2]
        pix_x = idx % self.x.shape[2]

        # Get a random time series.
        start = np.random.randint(0, self.x.shape[0] - self.series_length)

        if 0:
            plt.figure()
            plt.plot(self.x[start: start + self.series_length, pix_y, pix_x])
            plt.show()

        def form_item(arr, pix_y, pix_x, start, standardise=False):
            series = arr[start: start + self.series_length, pix_y, pix_x]
            if standardise:
                mean = np.mean(series)
                std = np.std(series)
                series = (series - mean) / std
            # Add extra feature dimension, to produce shape (1, L).
            # When batched, this will produce tensors of shape (N, 1, L).
            series = series[np.newaxis, ...]
            # Convert to torch tensor.
            series = torch.from_numpy(series)
            return series

        x = form_item(self.x, pix_y, pix_x, start, standardise=self.standardise)
        x_mean = form_item(self.x_mean, pix_y, pix_x, start, standardise=False)
        x_std = form_item(self.x_std, pix_y, pix_x, start, standardise=False)
        x_standardised = form_item(self.x_standardised, pix_y, pix_x, start, standardise=False)

        # pixel = self.x[:, pix_y, pix_x]
        #
        # series = pixel[start: start + self.series_length]
        #
        # if self.standardise:
        #     mean = np.mean(series)
        #     std = np.std(series)
        #     series = (series - mean) / std
        #
        # # Add extra feature dimension, to produce shape (1, L).
        # # When batched, this will produce tensors of shape (N, 1, L).
        # series = series[np.newaxis, ...]
        # # Convert to torch tensor.
        # series = torch.from_numpy(series)
        #
        # x = series

        if self.out_padding > 0:
            # Remove the padding from the output to account for unpadded convolutions in the model.
            y = x[:, self.out_padding: -self.out_padding]
            y_standardised = x_standardised[:, self.out_padding: -self.out_padding]
        else:
            y = x.clone()
            y_standardised = x_standardised.clone()

        sample = {
            'x': x,
            'x_mean': x_mean,
            'x_std': x_std,
            'x_standardised': x_standardised,
            'y': y,
            'y_standardised': y_standardised,
        }

        return sample


def calculate_std_mode(std):
    # KDE
    x_plot = np.linspace(np.amin(std), np.amax(std), 1000)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.0005).fit(std[:, None])
    log_dens = kde.score_samples(x_plot[:, None])
    max_std_dens = np.argmax(log_dens)
    std_mode = x_plot[max_std_dens]

    return std_mode



def test_pixel_dataset():
    filename = r'C:\Users\pwmor\data\140d\5229\20220824\data.h5'
    series_length = 100
    batch_size = 4

    dataset = PixelDataset(
        filename,
        series_length,
        standardise=False,
        rolling_standardise=True,
        rolling_standardise_window=21,
        difference=False,
        out_padding=2
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if 0:
        # Analyse the std dev.
        std = dataset.x_std[:, 15, 15]
        std_mode = calculate_std_mode(std)
        print(f'Mode of std: {std_mode}')

        fig, axes = plt.subplots(2, 1, squeeze=False)

        print(std.shape)
        ax = axes[0, 0]
        ax.hist(std, bins=1000)
        ax.axvline(x=std_mode, color='r')

        ax = axes[1, 0]
        ax.fill(x_plot, np.exp(log_dens), fc="#AAAAFF")
        ax.axvline(x=std_mode, color='r')

        plt.show()
        return

    for b, batch in enumerate(dataloader):
        x = batch['x']
        x_mean = batch['x_mean']
        x_std = batch['x_std']
        x_standardised = batch['x_standardised']
        y = batch['y']

        # Reconstruct x from standardised.
        x_reconst = x_standardised * x_std + x_mean

        fig, axes = plt.subplots(6, 1, squeeze=False)
        ax = axes[0, 0]
        for i in range(x.shape[0]):
            ax.plot(x[i, 0, :], label=i)
        ax.set_title('x')
        ax.legend()
        ax = axes[1, 0]
        for i in range(x_mean.shape[0]):
            ax.plot(x_mean[i, 0, :], label=i)
        ax.set_title('x_mean')
        ax.legend()
        ax = axes[2, 0]
        for i in range(x_std.shape[0]):
            ax.plot(x_std[i, 0, :], label=i)
        ax.set_title('x_std')
        ax.legend()
        ax = axes[3, 0]
        for i in range(x_standardised.shape[0]):
            ax.plot(x_standardised[i, 0, :], label=i)
        ax.set_title('x_standardised')
        ax.legend()
        ax = axes[4, 0]
        for i in range(y.shape[0]):
            ax.plot(y[i, 0, :], label=i)
        ax.set_title('y')
        ax.legend()
        ax = axes[5, 0]
        for i in range(x_reconst.shape[0]):
            ax.plot(x_reconst[i, 0, :], label=i)
        ax.set_title('x_reconst')
        ax.legend()

        plt.show()


if __name__ == '__main__':
    test_pixel_dataset()
