# imports
import torch
import os
import pandas as pd
import torch.utils.data as dataset
import skimage as io


class CatsAndDogsDataset(dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir,
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotation.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label

