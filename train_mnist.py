import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from utilities import patchify, get_positional_embeddings
from models import MSA, ViTBlock, ViT

run_dir = 'runs'
os.makedirs(run_dir, exist_ok=True)

transform = ToTensor()

MNIST(root='data/', download=True)

train_full = MNIST(root='data/', train=True, transform=transform)
train_subset = Subset(train_full, list(range(8000)))
train_size = int(0.8 * len(train_subset))
val_size = len(train_subset) - train_size
train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])

test_full = MNIST(root='data/', train=False, transform=transform)
test_subset = Subset(test_full, list(range(2000)))

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_subset, shuffle=False, batch_size=128, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT((1, 28, 28), n_patches=7, n_blocks=4, hidden_d=32, n_heads=4, out_d=10).to(device)
epochs = 60
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()
scaler = GradScaler()  

train_losses, val_losses, test_losses, test_accuracies, train_step_loss = [], [], [], [], []

if __name__ == '__main__':
    pass # to be implemented
