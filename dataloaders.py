import os
import numpy as np
import wfdb
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2


class EcgDataset1D(Dataset):
    def __init__(self, ann_path, mapping_path):
        super().__init__()
        self.data = json.load(open(ann_path))
        self.mapper = json.load(open(mapping_path))

    def __getitem__(self, index):
        img = np.load(self.data[index]["path"]).astype("float32")
        img = img.reshape(1, img.shape[0])

        return {"image": img, "class": self.mapper[self.data[index]["label"]]}

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        return data_loader

    def __len__(self):
        return len(self.data)
    

def callback_get_label(dataset, idx):
    return dataset[idx]["class"]

augment = Compose([Normalize(), ToTensorV2()])


class EcgDataset2D(Dataset):
    def __init__(self, ann_path, mapping_path):
        super().__init__()
        self.data = json.load(open(ann_path))
        self.mapper = json.load(open(mapping_path))

    def __getitem__(self, index):
        img = cv2.imread(self.data[index]["path"])
        img = augment(**{"image": img})["image"]

        return {"image": img, "class": self.mapper[self.data[index]["label"]]}

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        return data_loader

    def __len__(self):
        return len(self.data)


def callback_get_label(dataset, idx):
    return dataset[idx]["class"]
