import torch
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import torch.optim as optim
from torchvision.transforms import ToTensor
from glob import glob
from pathlib import Path
import os


class STNDataset(Dataset):

    def __init__(self, images_path, targets_path, transform=None, target_transform=None):

        self.images_path = images_path
        self.targets_path = targets_path
        self.transform = transform
        self.target_transform = target_transform

        # The targets are a subset of the images, so get a list of the ids from the targets.
        targets_filenames = glob(str(targets_path / '*.jpg'))
        self.ids = []
        for filename in targets_filenames:
            id = Path(filename).stem
            self.ids.append(id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = self.ids[idx]
        image_path = self.images_path / f'{id}.jpg'
        target_path = self.targets_path / f'{id}.jpg'

        image = read_image(str(image_path))
        target = read_image(str(target_path))
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        # return image, label

        sample = {
            'x': image,
            'y': target
        }

        return sample

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            #nn.Linear(10 * 52 * 52, 32),
            nn.Linear(10 * 24 * 24, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        #xs = xs.view(-1, 10 * 52 * 52)  # With 2 conv layers in the localisation network
        xs = xs.view(-1, 10 * 24 * 24)  # With 3 conv layers in the localisation network
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


def main():
    image_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images')
    target_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images_faces_filtered')

    batch_size = 16
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = v2.Compose([
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(255, 255, 255)),
        v2.Resize(224),  # Resize so shortest side is 224.
        v2.CenterCrop((224, 224)),  # Crop out the centre, expecting it to contain the face.
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = STNDataset(
        image_path,
        target_path,
        transform=transforms,  # None,#ToTensor(),
        target_transform=target_transforms,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = STN()
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            x = batch['x']
            y = batch['y']
            x = x.to(device)
            y = y.to(device)
            #data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    if 1:
        model.eval()
        for batch_idx, batch in enumerate(train_loader):
            x = batch['x']
            y = batch['y']
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            fig, axes = plt.subplots(batch_size, 3, squeeze=False)

            x = x.cpu().numpy()
            x = x.transpose((0, 2, 3, 1))
            y = y.cpu().numpy()
            y = y.transpose((0, 2, 3, 1))
            output = output.cpu().detach().numpy()
            output = output.transpose((0, 2, 3, 1))

            for i in range(batch_size):

                ax = axes[i, 0]
                ax.imshow(x[i])

                ax = axes[i, 1]
                ax.imshow(y[i])

                ax = axes[i, 2]
                ax.imshow(output[i])
            plt.show()



def main_dataset():
    image_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images')
    target_path = Path(r'D:\data\a_dataset_of_watches\watches\watches\images_faces_filtered')

    batch_size = 4

    transforms = v2.Compose([
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(255, 255, 255)),
        v2.Resize(224),  # Resize so shortest side is 224.
        v2.CenterCrop((224, 224)),  # Crop out the centre, expecting it to contain the face.
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = STNDataset(
        image_path,
        target_path,
        transform=transforms,  # None,#ToTensor(),
        target_transform=target_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for b, batch in enumerate(dataloader):
        x = batch['x']
        y = batch['y']

        x = x.cpu().numpy()
        x = x.transpose((0, 2, 3, 1))
        y = y.cpu().numpy()
        y = y.transpose((0, 2, 3, 1))

        fig, axes = plt.subplots(batch_size, 2, squeeze=False)

        for i in range(batch_size):
            ax = axes[i, 0]
            ax.imshow(x[i])

            ax = axes[i, 1]
            ax.imshow(y[i])


        plt.show()


if __name__ == '__main__':
    main()
    #main_dataset()
