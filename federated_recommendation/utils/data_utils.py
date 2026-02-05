"""
Dataset classes for multi-modal social media data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
from transformers import BertTokenizer
import torchvision.transforms as transforms


class SyntheticSocialMediaDataset(Dataset):
    """Synthetic dataset for testing"""
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        tokenizer: BertTokenizer = None,
        max_text_length: int = 128,
        mode: str = 'train'
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.mode = mode
        
        self.sample_texts = {
            0: ["Beautiful sunset at the beach", "Ocean waves", "Sandy shores"],
            1: ["Delicious homemade pizza", "Chocolate cake", "Fresh pasta"],
            2: ["Working on AI project", "Machine learning", "Coding"],
            3: ["Hiking in mountains", "Nature walk", "Trail running"],
            4: ["New movie release", "Cinema night", "Films"],
            5: ["Fitness workout", "Gym session", "Running"],
            6: ["Travel adventures", "Exploring places", "Vacation"],
            7: ["Art exhibition", "Painting", "Gallery"],
            8: ["Reading books", "Library", "Novel"],
            9: ["Music concert", "Performance", "Festival"]
        }
        
        print(f"Generated {num_samples} synthetic samples for {mode}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = idx % self.num_classes
        text = np.random.choice(self.sample_texts[label])
        
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            input_ids = torch.randint(0, 1000, (self.max_text_length,))
            attention_mask = torch.ones(self.max_text_length)
        
        image = torch.randn(3, 224, 224) + label * 0.1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader