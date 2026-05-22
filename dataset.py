import torch
import torchvision.datasets as d
from torch.utils.data import DataLoader
import torchvision.transforms as T
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def get_oxford_pet_dataloder(batch_size: int = 32, num_workers: int = 2):
    train_transforms = T.Compose([
        T.Resize(256, 256),
        T.RandomCrop(224, 224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = T.Compose([
        T.Resize(256, 256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = d.OxfordIIITPet(root="./data/oxford-pets", download=True, split="trainval", transform=train_transforms)
    val_dataset = d.OxfordIIITPet(root="./data/oxford-pets", download=True, split="test", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader