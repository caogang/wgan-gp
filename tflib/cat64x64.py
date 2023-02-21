import numpy as np

import os
import glob
import urllib
import gzip
import kaggle
import zipfile
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


def download_dataset(DATA_DIR='cat64x64'):
    kaggle.api.dataset_download_files(
        "spandan2/cats-faces-64x64-for-generative-models", path="./")

    dir_to_zip_file = 'cats-faces-64x64-for-generative-models.zip'
    dir_to_extract_to = DATA_DIR

    with zipfile.ZipFile(dir_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dir_to_extract_to)


class CatDataset(Dataset):
    def __init__(self, DATA_DIR="cat64x64", transforms=T.ToTensor()):
        images_paths = glob.glob(f"{DATA_DIR}/*")
        length = len(images_paths)

        self.x = images_paths
        self.length = length
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_dir = self.x[idx]
        img = Image.open(img_dir)
        transformed_img = self.transforms(img)

        return transformed_img
