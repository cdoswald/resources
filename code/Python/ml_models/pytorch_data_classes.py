"""PyTorch-based dataset classes."""

from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(
        self,
        paths_dict: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        # Create integer encoding for string labels
        self.label_encoding = {k:idx for idx, k in enumerate(paths_dict.keys())}
        # Create dataframe of image path and label columns
        image_label_dict = {v:k for k, values in paths_dict.items() for v in values}
        self.data = pd.DataFrame(
            list(image_label_dict.items()), columns=["image_path", "label"]
        )
        # Map string labels to corresponding integer encoding
        self.data["label"] = self.data["label"].map(self.label_encoding)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path = self.data.iloc[idx, 0]
        img_label = self.data.iloc[idx, 1]
        if self.transform:
            img = self.transform(torch_io.read_image(img_path, mode=RGBMODE)).float().to(DEVICE)
        else:
            img = torch_io.read_image(img_path, mode=RGBMODE).float().to(DEVICE)
        img_label = torch.tensor(img_label).float().to(DEVICE)
        return (img, img_label)


class TripletImageDataset(Dataset):

    def __init__(
        self,
        paths_dict: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        # Create integer encoding for string labels
        self.label_encoding = {k:idx for idx, k in enumerate(paths_dict.keys())}
        # Create dataframe of image path and label columns
        self.image_label_dict = {v:k for k, values in paths_dict.items() for v in values}
        self.data = pd.DataFrame(
            list(self.image_label_dict.items()), columns=["image_path", "label"]
        )
        # Map string labels to corresponding integer encoding
        self.data["label"] = self.data["label"].map(self.label_encoding)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Get anchor, positive, and negative sample paths and labels
        anchor_path = self.data.iloc[idx, 0]
        anchor_label = self.data.iloc[idx, 1]
        pos_path, pos_label = self._get_positive_sample(anchor_label)
        neg_path, neg_label = self._get_negative_sample(anchor_label)
        # Import images
        if self.transform:
            anchor_img = self.transform(torch_io.read_image(anchor_path, mode=RGBMODE)).float().to(DEVICE)
            pos_img = self.transform(torch_io.read_image(pos_path, mode=RGBMODE)).float().to(DEVICE)
            neg_img = self.transform(torch_io.read_image(neg_path, mode=RGBMODE)).float().to(DEVICE)
        else:
            anchor_img = torch_io.read_image(anchor_path, mode=RGBMODE).float().to(DEVICE)
            pos_img = torch_io.read_image(pos_path, mode=RGBMODE).float().to(DEVICE)
            neg_img = torch_io.read_image(neg_path, mode=RGBMODE).float().to(DEVICE)
        # Convert labels to tensors
        anchor_label = torch.tensor(anchor_label).float().to(DEVICE)
        pos_label = torch.tensor(pos_label).float().to(DEVICE)
        neg_label = torch.tensor(neg_label).float().to(DEVICE)
        return {
            "imgs": (anchor_img, pos_img, neg_img),
            "labels": (anchor_label.item(), pos_label.item(), neg_label.item())
        }

    def _get_positive_sample(self, anchor_label):
        """Randomly select a positive sample from same class as anchor sample."""
        pos_data = self.data.loc[self.data["label"] == anchor_label].reset_index(drop=True)
        idx = np.random.randint(0, len(pos_data))
        return pos_data.iloc[idx, 0], pos_data.iloc[idx, 1]

    def _get_negative_sample(self, anchor_label):
        """Randomly select a negative sample from different class as anchor sample."""
        neg_label = None
        while neg_label is None or neg_label == anchor_label:
            idx = np.random.randint(0, len(self.data))
            neg_label = self.data.iloc[idx, 1]
        return self.data.iloc[idx, 0], neg_label
