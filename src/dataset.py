import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256)):
        self.img_dir = os.path.join(root_dir, split)
        self.mask_dir = os.path.join(root_dir, f"{split}annot")
        self.filenames = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img  = Image.open(os.path.join(self.img_dir,  fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname))

        img  = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        img  = TF.to_tensor(img)
        img  = (img - self.mean) / self.std
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
