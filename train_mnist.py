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
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast():
                preds = model(x)
                loss = criterion(preds, y) / 4

            scaler.scale(loss).backward()

            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_loss = loss.detach().cpu().item() * 4
            train_loss += batch_loss / len(train_loader)
            train_step_loss.append(batch_loss)
            del x, y, preds, loss

        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.3f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                with autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    val_loss += loss.detach().cpu().item() / len(val_loader)

                del x, y, y_hat, loss

        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.3f}")

        correct, total = 0, 0
        test_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                with autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    test_loss += loss.detach().cpu().item() / len(test_loader)

                preds = torch.argmax(y_hat, dim=1)
                correct += torch.sum(preds == y).detach().cpu().item()
                total += len(x)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                del x, y, y_hat, loss

        test_accuracy = correct / total * 100
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%")

    model_path = os.path.join(run_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    np.savetxt(os.path.join(run_dir, 'confusion_matrix.csv'), cm, delimiter=',')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_step_loss) + 1), train_step_loss, label="Train Loss per Step")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Steps")
    plt.savefig(os.path.join(run_dir, 'train_loss_steps.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Test Accuracy over Epochs")
    plt.savefig(os.path.join(run_dir, 'accuracy_curve.png'))
    plt.close()

    model.eval()
    num_samples = 10
    test_samples, test_labels = next(iter(DataLoader(test_subset, shuffle=True, batch_size=num_samples)))

    test_samples = test_samples.to(device)
    with torch.no_grad():
        predictions = model(test_samples)
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

    test_samples = test_samples.cpu()
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_samples[i].squeeze(), cmap="gray")
        plt.title(f"True: {test_labels[i]}, Pred: {predicted_labels[i]}")
        plt.axis("off")

    plt.suptitle("Predictions on Test Samples")
    plt.savefig(os.path.join(run_dir, 'sample_predictions.png'))
    plt.show()
