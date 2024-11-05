import numpy as np
import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from model import *

run_dir = 'runs'
os.makedirs(run_dir, exist_ok=True)

transform=ToTensor()

train=CIFAR10(root='data',train=True,download=True,transform=transform)
test=CIFAR10(root='data',train=False,download=True,transform=transform)

train_loader = DataLoader(train, shuffle=True, batch_size=128)
test_loader = DataLoader(test, shuffle=False, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT((3, 32, 32), n_patches=8, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)

epochs=10
lr=0.05

optimizer = Adam(model.parameters(), lr=lr)

criterion = CrossEntropyLoss()

if __name__=='__main__':
    train_losses, test_losses, test_accuracies = [], [], []
    
    #Training
    for epoch in range(epochs):
        model.train()
        train_loss=0.0
        for batch in train_loader:
            x,y=batch
            x,y=x.to(device),y.to(device)
            
            preds= model(x)
            
            loss=criterion(preds,y)
            
            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
        
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.3f}")
            
    #Testing
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)
            
            preds = torch.argmax(y_hat, dim=1)
            correct += torch.sum(preds == y).detach().cpu().item()
            total += len(x)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_accuracy = correct / total * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}%")

    model_path = os.path.join(run_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    np.savetxt(os.path.join(run_dir, 'confusion_matrix.csv'), cm, delimiter=',')
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.savefig(os.path.join(run_dir, 'accuracy_curve.png'))
    plt.close()
    
