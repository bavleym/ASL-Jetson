# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:15:09 2024

@author: Gaiaf
"""

import os
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Dataset class for loading the ASL dataset
class ASLDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.x = []
        self.y = []

        # Automatically generate label map based on folder names
        folders = os.listdir(root)
        self.label_map = {folder: idx for idx, folder in enumerate(sorted(folders))}

        # Load image paths and labels
        for folder in folders:
            for dirpath, _, filenames in os.walk(os.path.join(root, folder)):
                for filename in filenames:
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        self.x.append(os.path.join(dirpath, filename))
                        self.y.append(self.label_map[folder])
        self.len = len(self.x)
        print("Loaded dataset with label map:", self.label_map)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('RGB')
        label = self.y[index]
        if self.transform:
            img = self.transform(img)
        return img, label

# Set the root path for the new dataset
root = "C:/Users/Gaiaf/OneDrive - CSULB/Documents/CECS 490 B/Train"

# Transformations for training and testing
transform_train = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the datasets and DataLoaders
train_dataset = ASLDataset(root, transform=transform_train)
test_dataset = ASLDataset(root, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model Setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 28)  # 28 output classes
)

# Set device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

# Early Stopping setup
best_acc = 0.0
patience = 5
trigger_times = 0

# Arrays to store losses for visualization
train_losses = []
test_losses = []

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{50}", leave=False)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
    return avg_loss

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_losses.append(avg_loss)
    print(f"Test set - Avg. loss: {avg_loss:.4f}, Accuracy: {test_acc:.2f}%")
    return avg_loss, test_acc

# Training loop with early stopping
for epoch in range(1, 51):
    train_loss = train(epoch)
    scheduler.step()
    test_loss, test_acc = test()

    # Plot loss after each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.title(f"Loss Plot After Epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Early Stopping based on test accuracy
    if test_acc > best_acc:
        best_acc = test_acc
        trigger_times = 0
        torch.save(model.state_dict(), "best_resnet18_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

save_path = "C:/Users/Gaiaf/OneDrive - CSULB/Documents/CECS 490 B/Code/fivek_resnet18_finetuned_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Final model saved to {save_path}")
