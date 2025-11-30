import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data_dir():
    """
    Return the absolute path to the image folder,
    using a path relative to this file's parent directories.
    """
    # __file__ is the path of this script, data/data.py
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # assume project root is two levels up from data/ (i.e. Final Project Files)
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    # relative path from project root to the image directory
    rel_path = os.path.join('..', 'input',
                            'toxic-plant-classification', 'tpc-imgs')
    data_dir = os.path.normpath(os.path.join(project_root, rel_path))
    return data_dir

def load_metadata(data_dir):
    full_meta = pd.read_csv(os.path.join(data_dir, 'full_metadata.csv'))
    full_meta = full_meta.sort_values(['toxicity','class_id'])
    return full_meta

def mixup_collate_fn(batch):
    """Apply MixUp after batching"""
    images, labels = torch.utils.data.default_collate(batch)
    mixup = v2.MixUp(num_classes=2, alpha=1.0)
    return mixup(images, labels)

class PlantDatasetWithSpecies(torch.utils.data.Dataset):
    """
    Takes in a subset and metadata dataframe. Retains plants species metadata in    addition to toxicity label and image to be used in the evaluation of model performance by plant species.
    """
    def __init__(self, subset, metadata):
        self.subset = subset
        self.metadata = metadata
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        actual_idx = self.subset.indices[idx]
        species = self.metadata.iloc[actual_idx]['slang']
        return image, label, species
    
    def __len__(self):
        return len(self.subset)

def get_dataloaders(meta_df, data_dir,
                    img_height=224, img_width=224,
                    batch_size=32, random_state=42):
    """
    Splits metadata into train/val/test, builds ImageFolder dataset, 
    creates DataLoaders for each split.
    """

    # Transforms
    train_transform = transforms.v2.Compose([
        transforms.v2.Resize((img_height, img_width)),
        transforms.v2.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_transform_strong = transforms.v2.Compose([
    transforms.v2.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),
    transforms.v2.RandomHorizontalFlip(),
    transforms.v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.v2.Compose([
        transforms.v2.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Build full dataset
    full_dataset = datasets.ImageFolder(data_dir)

    # Split indices
    train_idx, temp_idx = train_test_split(
        meta_df.index,
        test_size=0.3,
        random_state=random_state,
        stratify=meta_df[['toxicity', 'scientific_name']]
    )
    temp_meta = meta_df.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_meta[['toxicity', 'scientific_name']]
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_idx)
    train_strong_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Assign transforms
    train_dataset.dataset.transform = train_transform
    train_strong_dataset.dataset.transform = train_transform_strong
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Test dataset with species
    test_dataset_with_species = PlantDatasetWithSpecies(test_dataset, load_metadata(data_dir))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    train_strong_loader = DataLoader(train_strong_dataset,
                                     batch_size=batch_size,
                                     shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset_with_species, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, train_strong_loader, val_loader, test_loader
