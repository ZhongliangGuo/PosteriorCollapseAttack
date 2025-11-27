import os.path
from os.path import join
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from constant import X_ADV_FOLDER_NAME


class SubValSet(Dataset):
    def __init__(self, label_path, folder_path, img_size=512):
        super().__init__()
        self.label = pd.read_csv(label_path)
        self.folder_path = folder_path
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img = self.img_transforms(Image.open(join(self.folder_path, self.label.iloc[index]['filename'])).convert('RGB'))
        label = torch.tensor(self.label.iloc[index]['label'] - 1)
        caption = self.label.iloc[index]['caption']
        return img, label, caption


class PipeSet(Dataset):
    def __init__(self, label_path, folder_path, log_dir, img_size=512):
        super().__init__()
        self.label = pd.read_csv(label_path)
        self.folder_path = folder_path
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        self.adv_path = str(join(log_dir, X_ADV_FOLDER_NAME))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img = self.img_transforms(Image.open(join(self.folder_path, self.label.iloc[index]['filename'])).convert('RGB'))
        jpg = join(self.adv_path, f"{index}.jpg")
        png = join(self.adv_path, f"{index}.png")
        if os.path.exists(jpg):
            x_adv = self.img_transforms(Image.open(jpg).convert('RGB'))
        elif os.path.exists(png):
            x_adv = self.img_transforms(Image.open(png).convert('RGB'))
        else:
            raise NotImplemented
        label = torch.tensor(self.label.iloc[index]['label'] - 1)
        caption = self.label.iloc[index]['caption']
        return {"x": img,
                "x_adv": x_adv,
                "label": label,
                "caption": caption}
