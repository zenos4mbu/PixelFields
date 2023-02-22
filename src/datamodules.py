import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from torchvision.transforms import Compose, ToTensor

import cv2
import torch
from torch.utils.data import Dataset
import random


def batch_generator(image, batch_size, scale=1.0, shuffle=True):
    h, w, channels = image.shape
    h = int(h * scale)
    w = int(w * scale)

    ys = np.linspace(0, h, h, endpoint=False, dtype=np.int64)
    xs = np.linspace(0, w, w, endpoint=False, dtype=np.int64)
    positions = np.stack(np.meshgrid(ys, xs), 0).T.reshape(-1, 2)
    positions = torch.from_numpy(positions)
    batches = torch.split(positions, batch_size)
    # batches = [positions[i::batch_size] for i in range(int((h*w)/batch_size))]
    # random.shuffle(batches)
    return batches#torch.tensor(batches)

def create_grid(h, w, device="cpu"):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


class ImageDataset(Dataset):

    def __init__(self, image_path, img_dim, trainset_size, batch_size):
        self.trainset_size = trainset_size
        self.batch_size = batch_size
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255

        grid = create_grid(*self.img_dim[::-1])

        # batches = batch_generator(image, self.batch_size, shuffle=True)

        return grid, torch.tensor(image, dtype=torch.float32)#, batches

    def __len__(self):
        return self.trainset_size

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, FLAGS):#, data_dir: str = "./input/frattaglia.png"):
        super().__init__()
        self.FLAGS = FLAGS
        self.data_dir = FLAGS.image_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage = None):
        self.dataset = ImageDataset(self.data_dir, self.FLAGS.image_size, self.FLAGS.trainset_size, self.FLAGS.batch_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.FLAGS.batch_size)