import torch
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import glob
from pathlib import Path
import os


# TODO: Normalise input to make training work.


class WatchesDataset(Dataset):

    def __init__(self, metadata_filename, images_dir, transform=None):

        self.metadata_filename = metadata_filename
        self.images_dir = images_dir
        self.transform = transform

        # Read the metadata.
        # Read only brand, name, image_name
        self.df = pd.read_csv(metadata_filename, usecols=[1, 2, 4])

    def __len__(self):
        return self.df.shape[0]
        #return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        print(row)

        img_path = os.path.join(self.images_dir, row['image_name'])
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label

        sample = image

        return sample



def test_watches_dataset():
    metadata_filename = r'D:\data\a_dataset_of_watches\watches\watches\metadata.csv'
    images_dir = r'D:\data\a_dataset_of_watches\watches\watches\images'

    batch_size = 4

    # Probably replace these with a pre-processing spatial transformer network, that learns how to select the case
    # of the watch. Can probably do supervised learning for this, where we transform the input.
    # Maybe a UNet predicting the transformed image, and we can augment the input with various transforms.
    transforms = v2.Compose([
        #v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.CenterCrop((224, 224)),
        v2.Resize((224, 224)),
        #v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = WatchesDataset(
        metadata_filename,
        images_dir,
        transform=transforms,#None,#ToTensor(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for b, batch in enumerate(dataloader):
        x = batch

        x = x.cpu().numpy()
        x = x.transpose((0, 2, 3, 1))

        fig, axes = plt.subplots(batch_size, 1, squeeze=False)

        for i in range(batch_size):
            ax = axes[i, 0]
            ax.imshow(x[i])

        # ax = axes[0, 0]
        # for i in range(x.shape[0]):
        #     ax.plot(x[i, 0, :], label=i)
        # ax.set_title('x')
        # ax.legend()
        # ax = axes[1, 0]
        # for i in range(x_mean.shape[0]):
        #     ax.plot(x_mean[i, 0, :], label=i)
        # ax.set_title('x_mean')
        # ax.legend()
        # ax = axes[2, 0]
        # for i in range(x_std.shape[0]):
        #     ax.plot(x_std[i, 0, :], label=i)
        # ax.set_title('x_std')
        # ax.legend()
        # ax = axes[3, 0]
        # for i in range(x_standardised.shape[0]):
        #     ax.plot(x_standardised[i, 0, :], label=i)
        # ax.set_title('x_standardised')
        # ax.legend()
        # ax = axes[4, 0]
        # for i in range(y.shape[0]):
        #     ax.plot(y[i, 0, :], label=i)
        # ax.set_title('y')
        # ax.legend()
        # ax = axes[5, 0]
        # for i in range(x_reconst.shape[0]):
        #     ax.plot(x_reconst[i, 0, :], label=i)
        # ax.set_title('x_reconst')
        # ax.legend()

        plt.show()


if __name__ == '__main__':
    test_watches_dataset()
